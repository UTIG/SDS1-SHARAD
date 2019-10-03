# -*- coding: utf-8 -*-
authors__ = ['Christopher Gerekos, christopher.gerekos@unitn.it',
               'Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'August 23, 2019',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}

import subprocess
from scipy.constants import c, pi
import numpy as np
import matplotlib.pyplot as plt
import gdal
import cmp.pds3lbl as pds3
import misc.coord as crd
import misc.prog as prg


def incoherent_sim(state, rxwot, pri, dtm_path, ROIstart, ROIstop,
                   r_sphere=3396000,B=10E+6, trx=135E-6, fs=26.67E+6,
                   r_circle=27E+3,sigmas=0.02, of=8,
                   save_path=None, plot=False):

    """
    Incoherent Clutter Simulator for ranging based on code by R. Orosei [1],
    with modifications from C. Gerekos [2] and G. Steinbruegge [3].

    [1] Istituto Nazionale di Astrofisica (Italy).
    [2] Università degli Studi di Trento (Italy)
    [3] University of Texas at Austin (USA).

    Input:
    -----------------
    state: State vector(s) of s/c in body fixed (in km)
    rxwot: Receive window opening times in samples
    dtm_path: Path to geotiff DTM file
    ROIstart: Range line start
    ROIend: Range line end
    r_sphere (optional): Mars mean radius in m
    f0 (optional): Radar central frequency in Hz
    B (optional): Radar bandwidth in Hz
    trx (optional): Radar receive window duration in s
    fs (optional): Radar A/D converter sampling frequency in Hz
    r_circle (optional): Radius of the area contributing to scattering in m
    r_a (optional): Resolution along track after synthetic aperture processing
    sigmas (optional): rms slope of the facets, a free parameter
    of (optional): Oversampling factor of radar echo

    Output:
    """

    # Open DTM
    geotiff = gdal.Open(dtm_path)
    band = geotiff.GetRasterBand(1)
    dem = band.ReadAsArray()
    CornerLats, CornerLons = GetCornerCoordinates(dtm_path)

    # Number of Rangelines
    Necho = ROIstop-ROIstart
    # Derive pulse repetition frequency
    prf = np.zeros(Necho)
    prf[pri == 1] = 1/1428e-6
    prf[pri == 2] = 1/1492e-6
    prf[pri == 3] = 1/1290e-6
    prf[pri == 4] = 1/2856e-6
    prf[pri == 5] = 1/2984e-6
    prf[pri == 6] = 1/2580e-6

    # Convert spacecraft state vectors in MKS units
    state = state*1000
    # Find planetocentric coordinates of the ground track
    sph = crd.cart2sph(np.transpose(state[0:3]))
    latsc = sph[:, 0]
    lonsc = sph[:, 1]-360

    # Compute distance between points in the ground track as the arc between
    # two consecutive points on a sphere of radius equal to mean Mars radius
    gt = crd.sph2cart(np.transpose(np.vstack((lonsc, latsc, np.ones(Necho)))))
    gtx = gt[:, 0]
    gty = gt[:, 1]
    gtz = gt[:, 2]

    dist = r_sphere * \
           np.arccos(gtx[0:-1]*gtx[1:]+gty[0:-1]*gty[1:]+gtz[0:-1]*gtz[1:])
    dist = np.hstack((0, dist))
    dist = np.cumsum(dist)

    # Relsease memory
    del sph, gt, gtx, gty, gtz

    # Compute quantities used in echo simulation computations
    # Compute number of echo samples and the frequency spectrum of the
    # simulated signal
    Nsample = int(np.ceil(trx*fs))
    Nosample = int(np.ceil(of*trx*fs))
    [t, f] = fftvars(fs, Nsample)
    [to, fo] = fftvars(of*fs, Nosample)

    # Create a waveform for incoherent scattering simulation, that is a squared
    # cardinal sine, as the signal is expressed as power instead of voltage
    pulsesp = np.ones(len(fo))
    pulsesp[abs(fo) > B/2] = 0
    pulse = np.fft.ifft(pulsesp)
    pulse = pulse**2
    pulsesp = np.fft.fft(pulse)

    # Extract topography and simulate scattering
    # Create array to store result. echosp will contain the fourrier transform
    # of each rangeline.
    echosp = np.zeros((Nsample, Necho), dtype=complex)

    p = prg.Prog(Necho)
    for pos in range(0, Necho):
        # Extract DTM topography
        [lon_w, lon_e, lat_s, lat_n] = lonlatbox(lonsc[pos], latsc[pos],
                                                 r_circle, r_sphere)
        [hrsc, lon, lat] = dtmgrid(dem, lon_w, lon_e, lat_s, lat_n,
                                   CornerLats, CornerLons)

        hrsc = hrsc+r_sphere
        DEMshp = hrsc.shape

        # Compute position and orientation of the observed surface
        [theta, phi] = np.meshgrid(pi/180*lon, pi/180*lat)
        sphDEM = np.transpose(np.vstack((phi.flatten(), theta.flatten(),
                                         hrsc.flatten())))

        cartDEM = crd.sph2cart(sphDEM, indeg=False)
        X = cartDEM[:, 0].reshape(DEMshp)
        Y = cartDEM[:, 1].reshape(DEMshp)
        Z = cartDEM[:, 2].reshape(DEMshp)
        [la, lb, Ux, Uy, Uz, R] = surfaspect(X, Y, Z,
                                             state[0, pos],
                                             state[1, pos],
                                              state[2, pos])

        # Compute reflected power and distance from radar for every surface
        # element.
        E = facetgeopt(la, lb, Uz, R, sigmas)
        P = E**2
        delay = 2*R/c

        # Compute echo as the incoherent sum of power reflected by surface
        # elements and delayed in time according to their distance from the
        # spacecraft, but only for the most significant scatterers
        delay = delay.flatten()
        P = P.flatten()
        iP = np.argsort(-P)   # sort in descending order of power
        delay = delay[iP]
        P = P[iP]
        delay = np.mod(delay, trx)
        idelay = np.around(delay*of*fs)-1
        idelay[idelay < 0] = 0
        idelay[idelay >= Nosample] = Nosample
        idelay = idelay.astype(int)
        reflections = np.zeros(Nosample)

        # Remove 1/5 if you want all the echo, not just the top 20% brightest
        thresh = np.rint(len(P)/5)
        thresh = int(thresh)
        for j in range(0, thresh):
            reflections[idelay[j]] = reflections[idelay[j]] + P[j]

        spectrum = np.conj(pulsesp)*np.fft.fft(reflections)
        echosp[:, pos] = spectrum[np.abs(fo) <= fs/2]

        p.print_Prog(pos)


    # Align to a common reference, convert power to voltage, apply a window
    # Align echoes to a common reference point in time
    deltat = -(rxwot/fs+1428e-6)+11.96e-6
    for pos in range(0, Necho):
        phase = np.exp(-2j*pi*deltat[pos]*np.conj(f))
        echosp[:, pos] = echosp[:, pos]*phase

    # Convert echoes from power to voltage
    echo = np.fft.ifft(echosp, axis=0)
    echo = np.sqrt(echo)
    echosp = np.fft.fft(echo, axis=0)

    # Create a Hann window and apply it to data
    # TODO: Change this to a Hamming Window to be consistent with cmp!
    w = 0.5*(1+np.cos(2*pi*f/B))
    w[np.abs(f) > B/2] = 0
    for pos in range(0, Necho):
        echo[:, pos] = np.fft.ifft(np.conj(w) * echosp[:, pos])

    rdrgr = 20*np.log10(np.abs(echo))

    # Save results in an output file
    if save_path is not None:
        np.save(save_path, rdrgr)

    # Plot radargram
    if plot:
        fig = plt.imshow(rdrgr)
        plt.show(fig)

    return rdrgr

def fftvars(fs, Ns):
    """
    Input:
    -----------------
    fs: Sampling frequency
    Ns: Number of samples

    Output:
    -----------------
    t: Vector of time values
    f: Vector of frequencies
    """
    # Pathologies
    #if fs <= 0 :
    #    raise ValueError("The sampling frequency for the FFT vector must be a positive number.")
    #if np.round(Ns) != Ns or Ns <= 0 :
    #    raise ValueError("The number of samples for the FFT is not a positive integer.")
    # Some numerical parameters are computed:
    #    dt          Sampling interval in the time domain.
    #    df          Sampling interval in the frequency domain.
    dt = 1/fs
    df = fs/Ns
    # The vector of time values is created
    t = dt * np.arange(0, Ns, 1)
    # Preallocation of memory space results in faster execution
    f = np.zeros(Ns)
    # Order of frequencies is determined according to even or odd
    demiNs = int(Ns/2)
    demiNs1 = int((Ns+1)/2)
    if Ns % 2 == 0:
        f[0:demiNs] = df*np.arange(0, demiNs, 1)
        f[demiNs:Ns] = df*np.arange(-demiNs, 0, 1)
    else:
        f[0:demiNs1] = df*np.arange(0, demiNs1, 1)
        f[demiNs1:Ns] = df*np.arange(demiNs1-Ns, 0, 1)
    # outputs
    return (t, f)

def lonlatbox(lon_0, lat_0, r_circle, r_sphere):
    """
    Gives edges of box on a sphere within a given radius centered around
    a given point

    Input:
    -----------------
    lon_0: Longitude of center
    lat_0: Latitude of center
    r_circle: radius of area if interest
    r_sphere: radius of sphere

    Output:
    -----------------
    lon_w: Longitude of western edge
    lon_e: Longitude of eastern edge
    lat_n: Latitude of northern edge
    lat_s:Latitude of southern edge
    """

    # Pathologies
    if r_circle / r_sphere >= pi/2:
        lon_w = 0
        lon_e = 360
        lat_s = -90
        lat_n = 90

    delta_lat = 180/pi * r_circle / r_sphere

    lat_s = lat_0 - delta_lat
    lat_n = lat_0 + delta_lat

    if lat_n >= 90:
        lon_w = 0
        lon_e = 360
        lat_n = 90

    if lat_s <= -90:
        lon_w = 0
        lon_e = 360
        lat_s = -90

    # use the sine theorem: sin A / sin a = sin B / sin b
    a = r_circle/r_sphere
    b = pi/180*(90-np.abs(lat_0))

    delta_lon = 180/pi*np.arcsin(np.sin(a)/np.sin(b))
    lon_w = lon_0-delta_lon
    lon_e = lon_0+delta_lon

    # gather and return
    return (lon_w, lon_e, lat_s, lat_n)


def dtmgrid(DEM, lon_w, lon_e, lat_s, lat_n, CornerLats, CornerLons):
    """
    Picks pixels inside box from DTM

    Input:
    --------------
    DEM: Digital terrain model
    lon_w: Western longitude of box
    lon_e: Eastern longitude of box
    lat_s: Southern latitude of box
    lat_n: Northern latitude of box

    Output:
    --------------
    hrsc: Terrein heights
    lon: Longitude of pixels
    lat: Latitude of pixels

    """

    [DEMv, DEMh] = np.shape(DEM)

    # coordinates of the DEM corners.
    maxlat = CornerLats[0]
    minlat = CornerLats[1]
    minlon = CornerLons[0]

    # pixels per degree (remqrk: ppd = DEMh/np.abs(maxlon-minlon) gives the same result ==> GOOD)
    ppd = DEMv/np.abs(maxlat-minlat)

    # latitude
    lat = maxlat - 1/ppd * np.arange(1, DEMv+1) + 1/(2*ppd)
    I_g = np.where((lat >= lat_s) & (lat <= lat_n))
    lat = lat[I_g]

    # longitude
    lon = minlon+1/ppd*np.arange(1, DEMh+1)-1/(2*ppd)
    J_g = np.where((lon >= lon_w) & (lon <= lon_e))
    lon = lon[J_g]

    # select DEM portion
    start1 = np.array(I_g)[0, 0]; end1 = np.array(I_g)[0, -1]
    start2 = np.array(J_g)[0, 0]; end2 = np.array(J_g)[0, -1]
    hrsc = DEM[start1:end1+1, start2:end2+1]
    hrsc = hrsc.astype(float)

    # gather and return
    return (hrsc, lon, lat)


def facetgeopt(la, lb, Uz, R, sigma_s):
    # Calculate field strength
    tant = np.sqrt(1-Uz**2)/Uz
    E = la*lb*np.exp(-tant**2/(4*sigma_s**2))/(np.sqrt(2)*sigma_s*Uz**2)/R**2
    return E

def surfaspect(X, Y, Z, x0, y0, z0):
    # given a surface described by a set of points expressed in cartesian
    # coordinates and a point external to the surface, computes the size,
    # orientation and distance of each portion of the discretized surface
    # w.r.t. the external point

    Xu = np.vstack((X[0, :], X[0:-1, :]))
    Xd = np.vstack((X[1:, :], X[-1, :]))
    Xcol1 = X[:, 0]
    Xcole = X[:, -1]
    Xcol1 = Xcol1[:, np.newaxis]
    Xcole = Xcole[:, np.newaxis]
    Xl = np.hstack((Xcol1, X[:, 0:-1]))
    Xr = np.hstack((X[:, 1:], Xcole))

    Yu = np.vstack((Y[0, :], Y[0:-1, :]))
    Yd = np.vstack((Y[1:, :], Y[-1, :]))
    Ycol1 = Y[:, 0]
    Ycole = Y[:, -1]
    Ycol1 = Ycol1[:, np.newaxis]
    Ycole = Ycole[:, np.newaxis]
    Yl = np.hstack((Ycol1, Y[:, 0:-1]))
    Yr = np.hstack((Y[:, 1:], Ycole))

    Zu = np.vstack((Z[0, :], Z[0:-1, :]))
    Zd = np.vstack((Z[1:, :], Z[-1, :]))
    Zcol1 = Z[:, 0]
    Zcole = Z[:, -1]
    Zcol1 = Zcol1[:, np.newaxis]
    Zcole = Zcole[:, np.newaxis]
    Zl = np.hstack((Zcol1, Z[:, 0:-1]))
    Zr = np.hstack((Z[:, 1:], Zcole))

    Xa = Xr-Xl
    Ya = Yr-Yl
    Za = Zr-Zl

    Xb = Xu-Xd
    Yb = Yu-Yd
    Zb = Zu-Zd

    del Xu; del Yu; del Zu
    del Xd; del Yd; del Zd
    del Xl; del Yl; del Zl
    del Xr; del Yr; del Zr

    Xn = Ya*Zb-Za*Yb
    Yn = Za*Xb-Xa*Zb
    Zn = Xa*Yb-Ya*Xb

    XR = x0-X
    YR = y0-Y
    ZR = z0-Z

    # horizontal facet sizes
    la = np.sqrt(Xa**2+Ya**2+Za**2)
    Xa = Xa/la
    Ya = Ya/la
    Za = Za/la
    la = la/2

    # vertical facet sizes
    lb = np.sqrt(Xb**2+Yb**2+Zb**2)
    Xb = Xb/lb
    Yb = Yb/lb
    Zb = Zb/lb
    lb = lb/2

    n = np.sqrt(Xn**2+Yn**2+Zn**2)
    Xn = Xn/n
    Yn = Yn/n
    Zn = Zn/n

    # distances
    R = np.sqrt(XR**2+YR**2+ZR**2)
    XR = XR/R
    YR = YR/R
    ZR = ZR/R

    # angles
    Ux = XR*Xa + YR*Ya + ZR*Za
    Uy = XR*Xb + YR*Yb + ZR*Zb
    Uz = XR*Xn + YR*Yn + ZR*Zn

    return (la, lb, Ux, Uy, Uz, R)

def GetCornerCoordinates(FileName):
    GdalInfo = subprocess.check_output('gdalinfo {}'.format(FileName),
                                       shell=True)
    GdalInfo = GdalInfo.splitlines() # Creates a line by line list.
    CornerLats, CornerLons = np.zeros(5), np.zeros(5)
    GotUL, GotUR, GotLL, GotLR, GotC = False, False, False, False, False
    for line in GdalInfo:
        if line[:10] == b'Upper Left':
            CornerLats[0], CornerLons[0] = GetLatLon(line)
            GotUL = True
        if line[:10] == b'Lower Left':
            CornerLats[1], CornerLons[1] = GetLatLon(line)
            GotLL = True
        if line[:11] == b'Upper Right':
            CornerLats[2], CornerLons[2] = GetLatLon(line)
            GotUR = True
        if line[:11] == b'Lower Right':
            CornerLats[3], CornerLons[3] = GetLatLon(line)
            GotLR = True
        if line[:6] == b'Center':
            CornerLats[4], CornerLons[4] = GetLatLon(line)
            GotC = True
        if GotUL and GotUR and GotLL and GotLR and GotC:
            break
    return CornerLats, CornerLons

def GetLatLon(line):
    coords = str(line).split(') (')[1]
    coords = coords[:-1]
    LonStr, LatStr = coords.split(',')
    # Longitude
    LonStr = LonStr.replace('\\', '').split('d')   # Get the degrees, and the rest
    LonD = int(LonStr[0])
    LonStr = LonStr[1].split('\'')# Get the arc-m, and the rest
    LonM = int(LonStr[0])
    LonStr = LonStr[1].split('"') # Get the arc-s, and the rest
    LonS = float(LonStr[0])
    Lon = LonD + LonM/60. + LonS/3600.
    if LonStr[1] in ['W', 'w']:
        Lon = -1*Lon
    # Same for Latitude
    LatStr = LatStr.replace('\\', '').split('d')
    LatD = int(LatStr[0])
    LatStr = LatStr[1].split('\'')
    LatM = int(LatStr[0])
    LatStr = LatStr[1].split('"')
    LatS = float(LatStr[0])
    Lat = LatD + LatM/60. + LatS/3600.
    if LatStr[1] in ['S', 's']:
        Lat = -1*Lat
    return Lat, Lon

