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
import json
import re
import argparse
import time

from scipy.constants import c, pi
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# Add the parent directory of icsim.py  so we can import the below
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

#import cmp.pds3lbl as pds3
import misc.coord as crd
import misc.prog as prg


def incoherent_sim(state, rxwot, pri, dtm_path, ROIstart, ROIstop,
                   r_sphere=3396000,B=10E+6, trx=135E-6, fs=26.67E+6,
                   r_circle=27E+3,sigmas=0.02, of=8,
                   save_path=None, plot=False, do_progress=True, maxechoes=None, fast=False):

    """
    Incoherent Clutter Simulator for ranging based on code by R. Orosei [1],
    with modifications from C. Gerekos [2] and G. Steinbruegge [3], G. Ng [3].

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
    
    t0 = time.time()
    # Open DTM
    geotiff = gdal.Open(dtm_path)
    band = geotiff.GetRasterBand(1)
    dem = band.ReadAsArray()
    logging.debug("dem.shape = " + str(dem.shape))
    CornerLats, CornerLons = GetCornerCoordinates(dtm_path)

    # TODO: can we just load sections of the ROI?

    # Number of Rangelines
    
    Necho = ROIstop-ROIstart
    Necho1 = min(Necho, maxechoes) if maxechoes else Necho
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

    # Compute distance between points din the ground track as the arc between
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
    #pulsesp = np.fft.fft(pulse)
    pulsesp_c = np.conj(np.fft.fft(pulse))
    # Portion of the output spectrum that is of interest
    #spec_idx = np.abs(fo) <= fs/2

    # Create a Hann window to apply it to data
    # TODO: Change this to a Hamming Window to be consistent with cmp!
    w = 0.5*(1+np.cos(2*pi*f/B))
    w[np.abs(f) > B/2] = 0
    w_c = np.conj(w)



    # Extract topography and simulate scattering
    logging.debug("incoherent_sim: setup elapsed time at {:0.3f} sec".format(time.time() - t0))
    
    
    # Calculate cartesian coordinates for all coordinates on the map being used.
    logging.debug("Precomputing cartesian coordinates of interest")
    dem_mask = np.zeros_like(dem, dtype=np.bool)
    for pos in range(Necho1):
        # Extract DTM topography
        lon_w, lon_e, lat_s, lat_n = lonlatbox(lonsc[pos], latsc[pos],
                                               r_circle, r_sphere)
        _, _, aidx = argwhere_dtmgrid(dem.shape, lon_w, lon_e, lat_s, lat_n,
                                 CornerLats, CornerLons)
        dem_mask[aidx[0]:aidx[1], aidx[2]:aidx[3]] = True
    
    # Calculate cartesian and generate a full spherical.
    dem_cart = calc_dem_cart(dem, dem_mask, CornerLats, CornerLons, r_sphere)
    del dem, dem_mask
    logging.debug("incoherent_sim: Done precomputing cartesian coordinates at {:0.3f} sec".format(time.time() - t0))

    # Create array to store result. echosp will contain the Fourier transform
    # of each rangeline.
    echosp = np.empty((Nsample, Necho1), dtype=complex)
    p = prg.Prog(Necho1) if do_progress else None

    for pos in range(Necho1):
        # Extract DTM topography
        lon_w, lon_e, lat_s, lat_n = lonlatbox(lonsc[pos], latsc[pos],
                                               r_circle, r_sphere)
        #hrsc, lon, lat, aidx = dtmgrid(dem, lon_w, lon_e, lat_s, lat_n,
        #                         CornerLats, CornerLons)
        # dem_cart has almost the same shape as dem 
        _, _, aidx = argwhere_dtmgrid(dem_cart.shape[0:2], lon_w, lon_e, lat_s, lat_n,
                                 CornerLats, CornerLons)


        if True:
            la, lb, _, _, Uz, R = surfaspect(
                dem_cart[aidx[0]:aidx[1], aidx[2]:aidx[3], 0],  # X
                dem_cart[aidx[0]:aidx[1], aidx[2]:aidx[3], 1],  # Y
                dem_cart[aidx[0]:aidx[1], aidx[2]:aidx[3], 2],  # Z
                state[0, pos], state[1, pos], state[2, pos])
        else:
            la, lb, _, _, Uz, R = surfaspect1(dem_cart[aidx[0]:aidx[1], aidx[2]:aidx[3], :], state[0:3, pos])

        # Compute reflected power and distance from radar for every surface element.
        #E = s(la, lb, Uz, R, sigmas)
        #P = E**2
        P = facetgeopt(la, lb, Uz, R, sigmas).flatten() ** 2 # power from electric field
        delay = (2/c)*R.flatten()

        # Compute echo as the incoherent sum of power reflected by surface
        # elements and delayed in time according to their distance from the
        # spacecraft, but only for the most significant scatterers

        # TODO: a more sophisticated way would be to go to say 98% of cumulative power
        # thresh is the count of reflectors to retain. Increase this to 100% to retain all
        # thresh = int(np.rint(len(P) * 0.20))
        # Remove 1/5 if you want all the echo, not just the top 20% brightest
        thresh = int(np.rint(len(P)/5))

        # Partition powers into top x%, then sort that x%
        idxtop = np.argpartition(-P, thresh)[0:thresh]
        # Sort this top x% from smallest to largest, so they're added in this order,
        # to maximize numerical stability/accuracy.
        # This step could be optional.
        iP = idxtop[np.argsort(P[idxtop])]
        
        if pos == 0:
            tot_power = np.sum(P)
            used_power = np.sum(P[iP])
            logging.info("incoherent_sim: Using 20% of reflectors ({:d} total), got {:0.2f}% of power".format(len(P), used_power / tot_power * 100))
        
        delay = delay[iP]     # sort top x% delays in ascending order of power
        P = P[iP]             # sort top x% powers in ascending order of power
        delay = np.mod(delay, trx) # wrap delay by radar receive window
        idelay = np.clip(np.around(delay*of*fs)-1, 0, Nosample).astype(int)
        #-------------------------------------
        # Accumulate these reflections into the fast-time record
        reflections = np.zeros(Nosample)
        for i, pwr in zip(idelay, P):
            reflections[i] += pwr

        # Convert to frequency domain, then extract in-band portion of spectrum,
        # and apply pulse shaping filter
        #spectrum = np.conj(pulsesp)*np.fft.fft(reflections)
        spectrum = pulsesp_c*np.fft.fft(reflections)
        echosp[:, pos] = spectrum[np.abs(fo) <= fs/2]

	
        if p:
            p.print_Prog(pos)
    if p:
        p.close_Prog()

    logging.debug("incoherent_sim: reflection sim: {:0.3f} sec".format(time.time() - t0))

    # Align to a common reference, convert power to voltage, apply a window
    # Align echoes to a common reference point in time
    deltat = -(rxwot/fs+1428e-6)+11.96e-6

    conjf = np.conj(f)
    for pos in range(Necho1):
        phase = np.exp(-2j*pi*deltat[pos]*conjf)
        echosp[:, pos] *= phase #echosp[:, pos]*phase

    # Convert echoes from power to voltage
    logging.debug("incoherent_sim: delay: {:0.3f} sec".format(time.time() - t0))
    echo = np.fft.ifft(echosp, axis=0)
    echo = np.sqrt(echo)
    logging.debug("incoherent_sim: power to voltage: {:0.3f} sec".format(time.time() - t0))
    echosp = np.fft.fft(echo, axis=0)
    

    for pos in range(Necho1):
        echo[:, pos] = np.fft.ifft(w_c * echosp[:, pos])

    rdrgr = 20*np.log10(np.abs(echo))
    logging.debug("incoherent_sim: hanning filter and freq to time domain: {:0.3f} sec".format(time.time() - t0))

    # Save results in an output file
    if save_path is not None:
        logging.debug("incoherent_sim: Saving radargram to " + save_path)
        np.save(save_path, rdrgr)



    # Plot radargram
    if plot:
        fig, axs = plt.subplots(2, 1)
        im = axs[0].imshow(rdrgr, aspect='auto')
        axs[0].set_title('Radargram')
        fig.colorbar(im, ax=axs[0])
        #im = axs[1].imshow(rdrgr - rdrgr2, aspect='auto')
        #axs[1].set_title('Diff')
        #fig.colorbar(im, ax=axs[1])
        plt.show()

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


def calc_dem_cart(dem, dem_mask, corner_lats, corner_lons, r_sphere):
    """ Calculate the cartesian coordinates of a DEM 
    dem - an array of digital elevation model heights in meters
    dem_mask - a boolean array with the same dimensions as dem, indicating for which array values
    to compute.  Entries with True values will be computed.
    
    corner_lats: a 2-element array describing the minimum and maximum latitude of the array
    corner_lons: a 2-element array describing the minimum and maximum longitude of the array
    r_sphere: Radius (meters) of the sphere of the body to which the DEM elevations are referenced.
        
    """
    
    
    DEMv, DEMh = np.shape(dem)

    # coordinates of the DEM corners.
    maxlat = corner_lats[0]
    minlat = corner_lats[1]
    minlon = corner_lons[0]

    # pixels per degree (remark: ppd = DEMh/np.abs(maxlon-minlon) gives the same result ==> GOOD)
    ppd = DEMv/np.abs(maxlat-minlat)
    #ppdh = DEMh / np.abs(corner_lons[1] - corner_lons[0])
    #assert np.abs(ppd - ppdh) < 1e-3

    # latitude (radians)
    lat_r = pi / 180 * (maxlat - 1/ppd * np.arange(1, DEMv+1) + 1/(2*ppd))

    # longitude (radians)
    lon_r = pi / 180 * (minlon + 1/ppd * np.arange(1, DEMh+1) - 1/(2*ppd))
        
    ## Get indices of locations where mask is nonzero
    maskidx = np.nonzero(dem_mask)
    sph_dem = np.empty((len(maskidx[0]), 3))
    sph_dem[:, 0] = lat_r[maskidx[0]]
    sph_dem[:, 1] = lon_r[maskidx[1]]
    sph_dem[:, 2] = dem[maskidx[0], maskidx[1]] + r_sphere

    cart_dem = np.empty(dem.shape +  (3,))
    cart_dem[maskidx[0], maskidx[1], :] = crd.sph2cart(sph_dem, indeg=False)
    
    return cart_dem.reshape(dem.shape + (3,))


def argwhere_dtmgrid(dem_shape, lon_w, lon_e, lat_s, lat_n, CornerLats, CornerLons):
    """
    Picks indexes inside box from DTM

    Input:
    --------------
    dem_shape: Shape of digital terrain model array
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

    DEMv, DEMh = dem_shape

    # coordinates of the DEM corners.
    maxlat = CornerLats[0]
    minlat = CornerLats[1]
    minlon = CornerLons[0]

    # pixels per degree (remark: ppd = DEMh/np.abs(maxlon-minlon) gives the same result ==> GOOD)
    ppd = DEMv/np.abs(maxlat-minlat)

    # latitude
    lat = maxlat - 1/ppd * np.arange(1, DEMv+1) + 1/(2*ppd)
    I_g = np.where((lat >= lat_s) & (lat <= lat_n))
    lat = lat[I_g]

    # longitude
    lon = minlon + 1/ppd * np.arange(1, DEMh+1) - 1/(2*ppd)
    J_g = np.where((lon >= lon_w) & (lon <= lon_e))
    lon = lon[J_g]

    # select DEM portion
    start1 = np.array(I_g)[0, 0]; end1 = np.array(I_g)[0, -1]
    start2 = np.array(J_g)[0, 0]; end2 = np.array(J_g)[0, -1]
    
    # Array indices in the DEM used
    arrayidx = (start1, end1+1, start2, end2+1)

    # gather and return
    return (lat, lon, arrayidx)

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
    lat, lon, arrayidx = argwhere_dtmgrid(DEM.shape, lon_w, lon_e, lat_s, lat_n, CornerLats, CornerLons)

    hrsc = DEM[arrayidx[0]: arrayidx[1], arrayidx[2]:arrayidx[3]].astype(float)
    
    # gather and return
    return (hrsc, lon, lat, arrayidx)


def facetgeopt(la, lb, Uz, R, sigma_s):
    """
    Calculate field strength
    la: length of facet on dimension a
    lb: length of facet on dimension b
    Uz: cosine zf
    R:  distance from obzerver to reflector
    sigma_s: 
    """
    # Calculate field strength
    Uz2 = Uz**2
    #tant = np.sqrt(1-Uz2)/Uz
    #E = la*lb*np.exp(-tant**2/(4*sigma_s**2))/(np.sqrt(2)*sigma_s*Uz2)/R**2
    # we did have $^2 previously.
    tant2 = (1-Uz2)/Uz2
    E = la*lb*np.exp(-tant2/(4*sigma_s**2))/(np.sqrt(2)*sigma_s*Uz2)/R**2
    return E

def surfaspect(X, Y, Z, x0, y0, z0):
    # given a surface described by a set of points expressed in cartesian
    # coordinates and a point external to the surface, computes the size,
    # orientation and distance of each portion of the discretized surface
    # w.r.t. the external point (x0, y0, z0)

    # GN: implement this using more streamlined matrix math? (see surfaspect1)
    #logging.debug("surfaspect using {:d} points".format(len(X)))

    Xu = np.vstack((X[0, :], X[0:-1, :]))
    #np.vstack((np.diff(X, axis=0), 0.0))
    Xd = np.vstack((X[1:, :], X[-1, :]))

    Yu = np.vstack((Y[0, :], Y[0:-1, :]))
    Yd = np.vstack((Y[1:, :], Y[-1, :]))

    Zu = np.vstack((Z[0, :], Z[0:-1, :]))
    Zd = np.vstack((Z[1:, :], Z[-1, :]))


    # up minus down
    Xb = Xu-Xd
    Yb = Yu-Yd
    Zb = Zu-Zd
    del Xu, Yu, Zu, Xd, Yd, Zd

    """
    Xcol1 = X[:, 0]
    Xcole = X[:, -1]    
    Xcol1 = Xcol1[:, np.newaxis]
    Xcole = Xcole[:, np.newaxis]
    Xl = np.hstack((Xcol1, X[:, 0:-1]))
    Xr = np.hstack((X[:, 1:], Xcole))
    """
    #Xcol1, Xcole = None, None
    Xl = np.empty_like(X)
    Xr = np.empty_like(X)
    Xl[:, 0] = X[:, 0]
    Xl[:, 1:] = X[:, 0:-1]
    Xr[:, 0:-1] = X[:, 1:]
    Xr[:, -1] = X[:, -1]

    """
    Ycol1 = Y[:, 0]
    Ycole = Y[:, -1]    
    Ycol1 = Ycol1[:, np.newaxis]
    Ycole = Ycole[:, np.newaxis]
    Yl = np.hstack((Ycol1, Y[:, 0:-1]))
    Yr = np.hstack((Y[:, 1:], Ycole))
    """
    #Ycol1, Ycole = None, None
    Yl = np.empty_like(Y)
    Yr = np.empty_like(Y)
    Yl[:, 0] = Y[:, 0]
    Yl[:, 1:] = Y[:, 0:-1]
    Yr[:, 0:-1] = Y[:, 1:]
    Yr[:, -1] = Y[:, -1]    

    """
    Zcol1 = Z[:, 0]
    Zcole = Z[:, -1]
    
    Zcol1 = Zcol1[:, np.newaxis]
    Zcole = Zcole[:, np.newaxis]
    Zl = np.hstack((Zcol1, Z[:, 0:-1]))
    Zr = np.hstack((Z[:, 1:], Zcole))
    """
    #Zcol1, Zcole = None, None
    Zl = np.empty_like(Z)
    Zr = np.empty_like(Z)
    Zl[:, 0] = Z[:, 0]
    Zl[:, 1:] = Z[:, 0:-1]
    Zr[:, 0:-1] = Z[:, 1:]
    Zr[:, -1] = Z[:, -1]
    
    # right minus left
    Xa = Xr-Xl
    Ya = Yr-Yl
    Za = Zr-Zl
    del Xl, Yl, Zl, Xr, Yr, Zr#, \
        #Xcol1, Xcole, Ycol1, Ycole, Zcol1, Zcole


    # Compute normal vector to each facet by computing cross product
    Xn = Ya*Zb-Za*Yb
    Yn = Za*Xb-Xa*Zb
    Zn = Xa*Yb-Ya*Xb

    XR = x0-X
    YR = y0-Y
    ZR = z0-Z

    # get horizontal facet sizes, obtain unit horizontal vector
    la = np.sqrt(Xa**2+Ya**2+Za**2)
    # We don't use the unit vector later, so don't compute.
    # Xa /= la
    # Ya /= la
    # Za /= la
    la /= 2

    # get vertical facet sizes, obtain unit vertical vector
    lb = np.sqrt(Xb**2+Yb**2+Zb**2)
    # We don't use the unit vector later, so don't compute.
    # Xb /= lb
    # Yb /= lb
    # Zb /= lb
    lb /= 2

    # get normal vector length; obtain unit normal vector
    n = np.sqrt(Xn**2+Yn**2+Zn**2)
    Xn /= n
    Yn /= n
    Zn /= n

    # distances of each point on surface to external point
    R = np.sqrt(XR**2+YR**2+ZR**2)
    XR /= R
    YR /= R
    ZR /= R

    # cos of angles using dot product
    # We don't actually use Ux and Uy in any downstream uses, so
    # don't bother computing for now.
    Ux = None #Ux = XR*Xa + YR*Ya + ZR*Za # Ux = dot(R, va)
    Uy = None #Uy = XR*Xb + YR*Yb + ZR*Zb # Uy = dot(R, vb)
    Uz = XR*Xn + YR*Yn + ZR*Zn # Uz = dot(R, vn)

    return (la, lb, Ux, Uy, Uz, R)
    
    

def surfaspect1(surf, p0):
    """
    This is a simplified (but slower!?) version of surfaspect.  The output is binary-compatible.
    given a surface described by a grid of points surf, expressed in cartesian
    coordinates, and a point external to the surface, computes the size,
    orientation and distance of each portion of the discretized surface
    w.r.t. the external point p0=(x0, y0, z0)
    
    surf is expected to be an N x M x 3 grid of point coordinates, where
    X = surf[:, :, 0] is the x coordinate of each point
    Y = surf[:, :, 1] is the y coordinate of each point
    Z = surf[:, :, 2] is the z coordinate of each point
    """

    # Compute facet's vector along one axis: upper coordinates minus lower coordinates
    vb = np.empty(surf.shape)  # store upper coordinates in final result
 
    # up minus down (position of point below minus point above)
    vb[0, :, :] = surf[0, :, :]      # up
    vb[1:, :, :] = surf[0:-1, :]     # up
    vb[0:-1, :, :] -= surf[1:, :, :] # down
    vb[-1, :, :] -= surf[-1, :, :]   # down

    # Compute facet's vector along perpendicular axis: right coordinate minus left coordinate
    va = np.empty(surf.shape) # left coordinates in final result

    # right minus left (position of  point right minus point left)
    va[:, 0:-1, :] = surf[:, 1:, :]   # right
    va[:, -1, :] = surf[:, -1, :]     # right
    va[:, 0, :] -= surf[:,  0, :]     # left
    va[:, 1:, :] -= surf[:,  0:-1, :] # left
         
    # Compute normal vector to each facet
    vn = np.cross(va, vb)

    # Compute vector from each point on the surface to the external point
    PR = p0 - surf

    # get horizontal facet sizes (la), obtain unit horizontal vector (va)
    la = np.linalg.norm(va, axis=2)
    va /= la[:, :, None]
    la /= 2

    # get vertical facet sizes (lb), obtain unit vertical vectors (vb)
    lb = np.linalg.norm(vb, axis=2)
    vb /= lb[:, :, None]
    lb /= 2

    # Obtain unit normal vector 
    vn /= np.linalg.norm(vn, axis=2)[:, :, None]

    # distance from external point to each point on surface
    R = np.linalg.norm(PR, axis=2)
    # Unit vector to external point
    PR /= R[:, :, None]

    # cos of angles to external point, using dot product
    Ux = np.sum(PR*va, axis=2)
    Uy = np.sum(PR*vb, axis=2)
    Uz = np.sum(PR*vn, axis=2)

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


def test_icsim1(save_path=None, do_plot=False, do_progress=True):
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
    sim = incoherent_sim(state, rxwot, pri_code, dtm_path, idx_start, idx_end,
                         maxechoes=100, plot=do_plot, save_path=save_path, do_progress=do_progress)
    #try:
    #    assert sim.shape == data.shape
    #except AssertionError:
    #    print("sim.shape = {:s}".format(str(sim.shape)))
    #    print("data.shape = {:s}".format(str(data.shape)))
    #    raise

def test_surfaspect():
    """ TODO: test equivalence of surfaspect() and surfaspect1() """

def main():
    debug = True
    parser = argparse.ArgumentParser(description='Run SHARAD ranging processing')
    parser.add_argument('-o','--output', help="Output file for icsim test data")
    parser.add_argument('--tracklist', default="xover_idx.dat",
        help="List of tracks with xover points to process")
    parser.add_argument('--clutterpath', default=None, help="Cluttergram path")
    parser.add_argument('--noprogress', action="store_true", help="don't show progress")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-d','--debug', action="store_true", help="Display debugging plots")
    #parser.add_argument('--SDS', default=os.getenv('SDS'), help="Override SDS environment variable")


    args = parser.parse_args()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)
    test_latlon()
    test_GetCornerCoordinates1()


    test_icsim1(save_path=args.output, do_plot=args.debug, do_progress=not args.noprogress)


if __name__ == "__main__":
    # execute only if run as a script
    import os
    import cmp.pds3lbl as pds3
    main()



