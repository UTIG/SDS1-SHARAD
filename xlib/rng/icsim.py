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
import logging
import sys
import gdal
import json
import re

from scipy.constants import c, pi
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory of icsim.py  so we can import the below
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

#import cmp.pds3lbl as pds3
import misc.coord as crd
import misc.prog as prg


def incoherent_sim(state, rxwot, pri, dtm_path, ROIstart, ROIstop,
                   r_sphere=3396000,B=10E+6, trx=135E-6, fs=26.67E+6,
                   r_circle=27E+3,sigmas=0.02, of=8,
                   save_path=None, plot=False, do_progress=True, maxechoes=None):

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

    # TODO: can we just load sections of the ROI?

    # Number of Rangelines
    
    Necho = ROIstop-ROIstart
    if maxechoes is not None:
        Necho1 = min(Necho, maxechoes)
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

    # Release memory
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
    pulsesp_c = np.conj(pulsesp)

    # Extract topography and simulate scattering
    # Create array to store result. echosp will contain the fourrier transform
    # of each rangeline.
    echosp = np.zeros((Nsample, Necho), dtype=complex)

    p = prg.Prog(Necho1) if do_progress else None

    for pos in range(Necho1):
        # Extract DTM topography
        lon_w, lon_e, lat_s, lat_n = lonlatbox(lonsc[pos], latsc[pos],
                                               r_circle, r_sphere)
        hrsc, lon, lat = dtmgrid(dem, lon_w, lon_e, lat_s, lat_n,
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
        thresh = int(np.rint(len(P)/5))
        for j in range(thresh):
            #reflections[idelay[j]] = reflections[idelay[j]] + P[j]
            reflections[idelay[j]] += P[j]

        #spectrum = np.conj(pulsesp)*np.fft.fft(reflections)
        spectrum = pulsesp_c*np.fft.fft(reflections)
        echosp[:, pos] = spectrum[np.abs(fo) <= fs/2]
        if p:
            p.print_Prog(pos)
    if p:
        p.close_Prog()


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
    w_c = np.conj(w)
    for pos in range(Necho):
        echo[:, pos] = np.fft.ifft(w_c * echosp[:, pos])

    rdrgr = 20*np.log10(np.abs(echo))

    # Save results in an output file
    if save_path is not None:
        logging.debug("incoherent_sim: Saving radargram to " + save_path)
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
    lat_s: Latitude of southern edge
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
    hrsc: Terrain heights
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
    hrsc = DEM[start1:end1+1, start2:end2+1].astype(float)

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
    # w.r.t. the external point (x0, y0, z0)

    # GN TODO: summarize computations in some comments
    # GN TODO: implement this using more streamlined matrix math?
    logging.debug("surfaspect using {:d} points".format(len(X)))

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
    # right minus left
    Xa = Xr-Xl
    Ya = Yr-Yl
    Za = Zr-Zl

    # up minus down
    Xb = Xu-Xd
    Yb = Yu-Yd
    Zb = Zu-Zd

    del Xu, Yu, Zu, \
        Xd, Yd, Zd, \
        Xl, Yl, Zl, \
        Xr, Yr, Zr

    Xn = Ya*Zb-Za*Yb
    Yn = Za*Xb-Xa*Zb
    Zn = Xa*Yb-Ya*Xb

    XR = x0-X
    YR = y0-Y
    ZR = z0-Z

    # horizontal facet sizes
    la = np.sqrt(Xa**2+Ya**2+Za**2)
    Xa /= la
    Ya /= la
    Za /= la
    la /= 2

    # vertical facet sizes
    lb = np.sqrt(Xb**2+Yb**2+Zb**2)
    Xb /= lb
    Yb /= lb
    Zb /= lb
    lb /= 2

    n = np.sqrt(Xn**2+Yn**2+Zn**2)
    Xn /= n
    Yn /= n
    Zn /= n

    # distances
    R = np.sqrt(XR**2+YR**2+ZR**2)
    XR /= R
    YR /= R
    ZR /= R

    # angles
    Ux = XR*Xa + YR*Ya + ZR*Za
    Uy = XR*Xb + YR*Yb + ZR*Zb
    Uz = XR*Xn + YR*Yn + ZR*Zn

    return (la, lb, Ux, Uy, Uz, R)

def surfaspect1(X, Y, Z, x0, y0, z0):
    # given a surface described by a set of points expressed in cartesian
    # coordinates and a point external to the surface, computes the size,
    # orientation and distance of each portion of the discretized surface
    # w.r.t. the external point (x0, y0, z0)

    # GN TODO: summarize computations in some comments
    # GN TODO: implement this using more streamlined matrix math?
    logging.debug("surfaspect using {:d} points".format(len(X)))

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
    # right minus left
    Pa = np.empty((len(Xr), 3))
    Pa[:, 0] = Xr - Xl #Xa = Xr-Xl
    Pa[:, 1] = Yr - Yl #Ya = Yr-Yl
    Pa[:, 2] = Zr - Zl #Za = Zr-Zl

    # up minus down
    Pb = np.empty((len(Xr), 3))
    Pb[:, 0] = Xu - Xu #Xb = Xu-Xd
    Pb[:, 1] = Yu - Yu #Yb = Yu-Yd
    Pb[:, 2] = Zu - Zu #Zb = Zu-Zd

    del Xu, Yu, Zu, \
        Xd, Yd, Zd, \
        Xl, Yl, Zl, \
        Xr, Yr, Zr

    Xn = Ya*Zb-Za*Yb
    Yn = Za*Xb-Xa*Zb
    Zn = Xa*Yb-Ya*Xb

    XR = x0-X
    YR = y0-Y
    ZR = z0-Z

    # horizontal facet sizes
    la = np.sqrt(Xa**2+Ya**2+Za**2)
    Xa /= la
    Ya /= la
    Za /= la
    la /= 2

    # vertical facet sizes
    lb = np.sqrt(Xb**2+Yb**2+Zb**2)
    Xb /= lb
    Yb /= lb
    Zb /= lb
    lb /= 2

    n = np.sqrt(Xn**2+Yn**2+Zn**2)
    Xn /= n
    Yn /= n
    Zn /= n

    # distances
    R = np.sqrt(XR**2+YR**2+ZR**2)
    XR /= R
    YR /= R
    ZR /= R

    # angles
    Ux = XR*Xa + YR*Ya + ZR*Za
    Uy = XR*Xb + YR*Yb + ZR*Zb
    Uz = XR*Xn + YR*Yn + ZR*Zn

    return (la, lb, Ux, Uy, Uz, R)

def GetCornerCoordinates2(FileName):
    """ Use gdalinfo to get corner coordinates, but use json format """
    cmd = ['gdalinfo', '-json', FileName]
    logging.debug("GetCornerCoordinates: CMD: " + " ".join(cmd))
    resp = subprocess.check_output(cmd)
    try:
        gdal_info = json.loads(resp.decode())
    except json.decoder.JSONDecodeError:
        logging.error("Could not decode json: " + resp.decode())
        raise
    # How do we get center?
    # print(resp.decode())

    print(gdal_info['extent']['coordinates'][0])

    list_lon, list_lat = zip(*gdal_info['extent']['coordinates'][0])
    return np.array([list_lat[0], list_lat[1], list_lat[3], list_lat[2], list_lat[4]]), \
           np.array([list_lon[0], list_lon[1], list_lon[3], list_lon[2], list_lon[4]])




def GetCornerCoordinates(FileName):
    # TODO: convert this to not use gdalinfo, but to do things natively?
    cmd = 'gdalinfo {}'.format(FileName)
    logging.debug("GetCornerCoordinates: CMD:  " + cmd)
    GdalInfo = subprocess.check_output(cmd,
                                       shell=True)
    GdalInfo = GdalInfo.splitlines() # Creates a line by line list.
    CornerLats, CornerLons = np.zeros(5), np.zeros(5)
    GotUL, GotUR, GotLL, GotLR, GotC = False, False, False, False, False
    for line in GdalInfo:
        #print(line)
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

def test_GetCornerCoordinates1():
    import os
    dtm = os.path.join(os.getenv('SDS'), 'orig/supl/hrsc/MC11E11_HRDTMSP.dt5.tiff') # example DTM file
    x1 = GetCornerCoordinates(dtm)
    print("x1=" + str(x1))
    x2 = GetCornerCoordinates2(dtm)
    print("x2=" + str(x2))
    for idx in range(4):
        assert abs(x1[0][idx] - x2[0][idx]) < 1e-1 # lat
        assert abs(x1[1][idx] - x2[1][idx]) < 1e-1 # lon


latlonpat = re.compile(r"""
    (?P<name>.+?)
    \((?P<x>[\-\d\.\s]+),(?P<y>[\-\d\.\s]+)\)\s+
    \(\s*(?P<lond>\d+)d
    \s*(?P<lonm>\d+)'
    \s*(?P<lons>[\d\.]+)"(?P<lonew>\w)
    ,\s+
    \s*(?P<latd>\d+)d
    \s*(?P<latm>\d+)'
    \s*(?P<lats>[\d\.]+)"(?P<latns>\w)

    """
    , re.X) 
def GetLatLon(line):
    """ Parse latitude and longitude from gdalinfo line 
Corner Coordinates:
Upper Left  (-1333625.000, 1778175.000) ( 22d30' 1.15"W, 30d 0' 2.04"N)
Lower Left  (-1333625.000,     -25.000) ( 22d30' 1.15"W,  0d 0' 1.52"S)
Upper Right (      25.000, 1778175.000) (  0d 0' 1.52"E, 30d 0' 2.04"N)
Lower Right (  25.0000000, -25.0000000) (  0d 0' 1.52"E,  0d 0' 1.52"S)
Center      ( -666800.000,  889075.000) ( 11d14'59.82"W, 15d 0' 0.26"N)
    
    """

    m = latlonpat.match(line.decode())
    if not m:
        return None, None

    lon = int(m.group('lond')) + int(m.group('lonm')) / 60.0 + float(m.group('lons')) / 3600.0
    lat = int(m.group('latd')) + int(m.group('latm')) / 60.0 + float(m.group('lats')) / 3600.0

    if m.group('lonew').lower() == 'w':
        lon *= -1
    if m.group('latns').lower() == 's':
        lat *= -1

    return lat, lon


def test_latlon():
    
    data = b"""Upper Left  (-1333625.000, 1778175.000) ( 22d30' 1.15"W, 30d 0' 2.04"N)
Lower Left  (-1333625.000,     -25.000) ( 22d30' 1.15"W,  0d 0' 1.52"S)
Upper Right (      25.000, 1778175.000) (  0d 0' 1.52"E, 30d 0' 2.04"N)
Lower Right (  25.0000000, -25.0000000) (  0d 0' 1.52"E,  0d 0' 1.52"S)
Center      ( -666800.000,  889075.000) ( 11d14'59.82"W, 15d 0' 0.26"N)""".split(b"\n")
    for line in data:
        lat1, lon1 = GetLatLon(line)
        print(lat1,lon1)


def test_icsim1(do_progress=True):
    """ Run icsim using parameters from run_ranging.py and icd.py """
    logging.debug("test_icsim1()")

    #------------------------
    # input parameters 
    inpath='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/data/rm271/edr4992001/e_4992001_001_ss19_700_a_a.dat'
    idx_start = 53243
    idx_end = 63243
    #------------------------

    # create cmp path
    path_root_rng = '/disk/kea/SDS/targ/xtra/SHARAD/rng/'
    path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
    path_root_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    dtm_path = '/disk/daedalus/sds/orig/supl/hrsc/MC11E11_HRDTMSP.dt5.tiff'
    # Relative path to this file
    fname = os.path.basename(inpath)

    # Relative directory of this file
    reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
    logging.debug("inpath: " + inpath)
    logging.debug("reldir: " + reldir)
    logging.debug("path_root_edr: " + path_root_edr)
    cmp_path = os.path.join(path_root_cmp, reldir, 'ion', fname.replace('_a.dat','_s.h5') )
    label_science = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
    label_aux  = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'


    science_path = inpath.replace('_a.dat','_s.dat')
    
    if not os.path.exists(cmp_path):
        logging.warning(cmp_path + " does not exist")
        return 0

    aux_path = inpath

    # Number of range lines
    Necho = idx_end - idx_start

    # Data for RXWOTs
    data = pds3.read_science(science_path, label_science, science=True,
                              bc=False)
    # Range window starts
    rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]

    aux = pds3.read_science(aux_path, label_aux, science=False, bc=False)
    pri_code = np.ones(Necho)
    p_scx = aux['X_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
    p_scy = aux['Y_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
    p_scz = aux['Z_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
    v_scx = aux['X_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
    v_scy = aux['Y_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
    v_scz = aux['Z_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
    state = np.vstack((p_scx, p_scy, p_scz, v_scx, v_scy, v_scz))
    # GNG 2020-01-27 transpose seems to give matching dimensions to pulse compressed radargrams
    sim = incoherent_sim(state, rxwot, pri_code, dtm_path, idx_start, idx_end, maxechoes=100, do_progress=do_progress).transpose()
    #try:
    #    assert sim.shape == data.shape
    #except AssertionError:
    #    print("sim.shape = {:s}".format(str(sim.shape)))
    #    print("data.shape = {:s}".format(str(data.shape)))
    #    raise



def main():
    debug = True
    loglevel = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=loglevel, stream=sys.stdout)
    test_latlon()
    test_GetCornerCoordinates1()
    test_icsim1(do_progress = not debug)


if __name__ == "__main__":
    # execute only if run as a script
    import os
    import cmp.pds3lbl as pds3
    main()



