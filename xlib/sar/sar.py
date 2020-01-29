__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '2.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'function library for SAR processing'},
    '2.0':
        {'date': 'April 29 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Version 2 function library for SAR processing'}}

import sys
import os
import glob
import math
import logging
import numpy as np
import pvl
import pandas as pd
import matplotlib.pyplot as plt
import scipy

#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import when running from xlib parent directory
# TODO: fix this import system in parents?
try:
    from sar import smooth
except ImportError:
    import smooth

def sar_posting(dpst, La, idx, tlp, et):
    '''
    Algorithm for defining the centers and extent of output SAR columns along
    groundtrack.

    Inputs:
    -------------
    dpst: distance along the track to post radargram columns [m]
      La: aperture length [s]
     idx: number of range lines within the area to be focused (size of the
          radargram)
     tlp: vector of interpolated spacecraft positions along its ground
              track [km]
      et: vector of ephemeris times [s]

    Outputs:
    -------------
    pst_trc: extent and center of SAR apertures expressed in terms of rangeline
             numbers
     pst_et: extent and center of SAR apertures expressed in terms of ephemeris
             times [s]
    '''

    ## define posting centers for SAR columns
    ### post radargram columns
    num_pst = np.floor((np.max(tlp)-tlp[0])/(dpst/1000))
    # TODO: pst_trc could be a list of tuples 3xn
    pst_trc = np.zeros((int(num_pst), 3), dtype=float)
    pst_trc[:, 0] = np.round(np.arange(0, idx, idx/num_pst))
    ## define trace ranges to include in SAR window from aperture length
    ### aperture length will be defined in SECONDS so need the ephemeris time
    ### for each post trace
    # TODO: pst_et could be a list of tuples
    pst_et = np.zeros((len(pst_trc), 3), dtype=float)
    for ii in range(len(pst_et)):
        pst_et[ii, 0] = et[int(pst_trc[ii, 0])]
    pst_et[:, 1] = pst_et[:, 0] - La/2
    pst_et[:, 2] = pst_et[:, 0] + La/2
    ### extract the trace numbers comprising each aperture bin
    for ii in range(len(pst_et)):
        if pst_et[ii, 1] < pst_et[0, 0]:
            pst_trc[ii, 1] = 0
            pst_trc[ii, 2] = 0
        elif pst_et[ii, 2] > pst_et[len(pst_et)-1, 0]:
            pst_trc[ii, 1] = 0
            pst_trc[ii, 2] = 0
        else:
            pst_trc[ii, 1] = pst_trc[np.argmin(abs(pst_et[:, 0] - pst_et[ii, 1])), 0]
            pst_trc[ii, 2] = pst_trc[np.argmin(abs(pst_et[:, 0] - pst_et[ii, 2])), 0]
    pst_trc = pst_trc.astype(int)

    return pst_trc, pst_et

# vector_interp moved to smooth.py
#def vector_interp(vect):
#    '''
#    Linear interpolation of a vector to smooth out steps. Values are plotted in
#    middle of any discretized zones.
#
#    Inputs:
#    -------------
#      vect: one-dimensional vector to be interpolated
#
#     Outputs:
#    -------------
#     out: interpolated vector
#    '''
#
#    # interpolate a vector
#    t1 = np.unique(vect)
#    unix = np.zeros((len(t1), 1), dtype=float)
#    for ii in range(len(t1)):
#        t2 = abs(vect - t1[ii])
#        t3 = np.argmin(t2)
#        t4 = abs(len(vect) - np.argmin(np.flipud(t2)))
#        unix[ii] = np.round(np.mean([t3, t4]))
#    out = np.interp(np.arange(0, len(vect), 1), unix[:, 0], t1)
#
#    return out

def twoD_filter(data, dt, rf0, rf1, af0, af1, dres, m, n):
    '''
    Create a 2D filter to pre-filter the radar data to desired time and Doppler
    frequencies.

    The resulting filter is Hann shaped in the frequency domain between the
    lower and upper specified time frequencies (both positive and negative) and
    boxcar shaped in the Doppler domain with cosine tapers between the
    specified low-pass and high-cut Doppler frequencies (both positive and
    negative).

    Inputs:
    -------------
     data: array of complex-valued range lines expressed in the
           slow-time/fast-time domain
           - range lines organized by row
           - frequency samples organized by column
       dt: time step [s]
      rf0: low time frequency [Hz]
      rf1: upper time frequency [Hz]
      af0: low-pass azimuth frequency [Hz]
      af1: high-cut azimuth frequency [Hz]
     dres: frequency resoltuion of the azimuth spectrum [Hz]
        m: number of range lines within the aperture
        n: number of time samples along the range lines within the aperture

     Outputs:
    -------------
     out: 2D filter
    '''

    # range
    rf0i = int(math.floor((rf0 * dt * n) + 0.5)) + 1
    rf1i = int(math.floor((rf1 * dt * n) + 0.5)) + 1
    bw = rf1i - rf0i
    rhann = np.zeros((bw, 1))
    rhann[:, 0] = np.hanning(bw)
    rfilt = np.zeros((n, 1))
    rfilt[0:len(rhann)] = rhann
    rfilt[len(rfilt)-len(rhann):len(rfilt)] = np.flipud(rhann)
    # azimuth
    Dfreq = np.fft.fftfreq(m, dres)
    af0i = np.argmin(abs(Dfreq-af0))
    af1i = np.argmin(abs(Dfreq-af1))
    bw = af1i - af0i
    ahann = np.reshape(0.5 + 0.5*np.cos(np.linspace(0.0, 1.0, bw+1) * np.pi), (-1, 1))
    afilt = np.zeros((m, 1))
    afilt[af0i-1:af0i+bw] = ahann
    afilt[m+2-af1i-1:m+2-af0i] = np.flipud(ahann)
    afilt[0:af0i-1] = 1.0
    afilt[m+3-af0i-1:m] = 1.0
    # combined
    filt = np.multiply(rfilt.transpose(), afilt.conj())

    out = np.fft.fft2(data, norm='ortho')
    out = np.multiply(out, filt)
    out = np.fft.ifft2(out, norm='ortho')

    return out

def rx_opening(data, rxwot, dt):
    '''
    frequency domain-applied phase shift to align traces relative to changes in
    receive window opening times

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - frequency samples organized by column
     rxwot: vector of receive window opening times after chirp emission
        dt: fast-time sampling interval

    Output:
    -----------
        out: array of correctly time-positioned range lines
    '''

    # define required temporary parameters
    n = np.size(data, axis=1)
    fs = np.fft.fftfreq(n, dt)

    # define the output
    out = np.zeros((len(data), n), dtype=complex)

    # apply phase shift
    for jj in range(len(data)):
        tempA = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                            np.exp(-1j * 2 * np.pi * rxwot[jj] * fs))
        out[jj] = np.fft.ifft(tempA, norm='ortho')

    return out

def dD_rngmig(data, R0, et, vt, dt):
    '''
    Range migration step of the delay Doppler focusing algorithm.
    Intended to shift traces within the aperture in fast-time according to the
    predicted hyperolic shape for the surface echo at the mid-aperture range
    line.

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - fast-time samples organized by column
        R0: TODO
        et: relative ephemris times for each range line within the aperture [s]
            - defined relative to the ephemeris time for the mid-aperture range
              line
        vt: vector of spacecraft velocities within the aperture [m/s]
        dt: fast-time sampling interval [s]

    Output:
    -----------
        out: array of range migrated range lines
    '''

    # GNG: TODO -- in theory the output lines could be a list of numpy arrays

    # define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    # calculate distance from nadir mid-aperture surface point to spacecraft
    # for each range line
    R = np.zeros((len(data), ))
    for jj in range(len(data)):
        R[jj] = np.sqrt(R0**2 + ((et[jj]) * vt[jj])**2)

    #convert distance to two-way time
    #dt_aperture = 2 * (R - R0) / 299792458
    dt_aperture = (R - R0) * (2 / 299792458)

    # apply phase shift to each range line corresponding to required range
    # migration
    fs = np.fft.fftfreq(np.size(data, axis=1), dt)
    for jj in range(len(data)):
        tempB = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                            np.exp(1j * 2 * np.pi * dt_aperture[jj] * fs))
        out[jj] = np.fft.ifft(tempB, norm='ortho')

    return out

def dD_azmig(data, R0, et, vt):
    '''
    Azimuth migration step in the delay Doppler SAR processing.
    Intended to calculate the differential Doppler phase shift for each range
    line relative to the nadir surface point in the middle of the aperture and
    then correct for it.

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - fast-time samples organized by column
        R0: distance to surface at mid-point of aperture [m]
        et: relative ephemris times for each range line within the aperture [s]
            - defined relative to the ephemeris time for the mid-aperture range
              line
        vt: vector of spacecraft velocities within the aperture [m/s]

    Output:
    -----------
        out: array of azimuth migrated range lines
    '''

    # define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    # calculate distance from nadir mid-aperture surface point to spacecraft
    # for each range line
    R = np.zeros((len(data), ))
    for jj in range(len(data)):
        R[jj] = np.sqrt(R0**2 + ((et[jj]) * vt[jj])**2)

    # apply azimuth migration
    dr = -1 * (R - R0)
    dphi = np.exp(-1j * ((4 * np.pi / (299792458 / 25E6)) * dr))
    dphi = np.transpose(np.broadcast_to(np.transpose(dphi),
                                        (np.size(data, axis=1), len(dphi))))
    out = np.multiply(data, np.conj(dphi))

    return out

def dD_traceconstructor(data):
    '''
    Construct the final SAR focused range line.
    Calculate the frequency spectrum for each range spectrum and apply a Hann
    window to suppress azimuth sidelobes

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - fast-time samples organized by column

    Output:
    -----------
        out: array of correctly time-positioned range lines
    '''

    temp = np.fft.fftshift(np.fft.fft(data, axis=0, norm='ortho'), axes=(0,))
    hann = np.hanning(len(temp))
    hann = np.transpose(np.broadcast_to(np.transpose(hann),
                                        (np.size(data, axis=1), len(hann))))
    return np.abs(np.fft.ifftshift(np.multiply(temp, hann), axes=(0,)))

def delay_doppler_v1(data, dpst, La, dBW, tlp, et, scrad, tpgpy, rxwot, vt,
                  comb_ml=True, debugtag="SAR"):
    '''
    Attempt at delay Doppler SAR processing of SHARAD radar data. Based on
    US methodology as presented in the US PDS data descriptions

    Inputs:
    -------------
        data: complex-valued ionosphere-corrected radar data
        dpst: distance along the track to post radargram columns [m]
          La: aperture length [s]
         dBW: Doppler frequency bandwidth [Hz]
         tlp: vector of interpolated spacecraft positions along its ground
              track [km]
          et: vector of ephemeris times [s]
       scrad: vector of distance from spacecraft to Martian centre of mass [km]
       tpgpy: vector of the estimated radius of Mars including topography [km]
       rxwot: vector of receiving antenna opening times [s]
          vt: vector of spacecraft velocities [m/s]
     comb_ml: optional flag for what type of multilook product we want to
              generate if multilook processing is to be performed
              - True: combine and produce a single multilook-ed image
              - False: produce a three axis array of individual looks

     debugtag: Optional unique tag prepended to debugging messaages

     Outputs:
    -------------
     sar_data: SAR focused radar data
      pst_trc: range line numbers of posted SAR-corrected range lines
    '''

    # interpolate the positional vectors as required
    tlp, _ = smooth.smooth(tlp)
    vt, _ = smooth.smooth(vt)
    #tlp = vector_interp(tlp)
    #vt = vector_interp(vt)

    ## define posting centers for SAR columns
    pst_trc, pst_et = sar_posting(dpst, La, int(len(data)), tlp, et)
    del pst_et

    ## define the number of looks
    looks = int(np.round(2 * La * dBW))
    if looks == 0:
        looks = 1
    elif looks % 2 == 0:
        looks = looks - 1

    logging.debug("{:s}: Number of looks in delay Doppler "
                  "SAR processing is {:d}".format(debugtag, looks))

    # predefine output and start sar processor
    if looks != 1:
        rl = np.zeros((looks, len(pst_trc), len(data[0])), dtype=float)
        rl2 = np.zeros((len(pst_trc), len(data[0])), dtype=float)
    else:
        rl = np.zeros((len(pst_trc), len(data[0])), dtype=float)

    for ii in range(len(pst_trc)):
        if ii % 50 == 0:
            logging.debug("{:s}: Working delay Doppler SAR column "
                          "{:d} of {:d}".format(debugtag, ii, len(pst_trc)) )

        if pst_trc[ii, 1] == 0 or pst_trc[ii, 2] == 0:
            continue

        # TODO: gc old temp variables to keep mem footprint low

        # select data within the aperture
        temp_data = data[pst_trc[ii, 1]:pst_trc[ii, 2]]

        # time shift to align traces and remove rx opening time changes and
        # spacecraft radius changes
        temp_data2 = rx_opening(temp_data,
                                rxwot[pst_trc[ii, 1]:pst_trc[ii, 2]],
                                0.0375E-6)

        # range migration
        R0 = (scrad[pst_trc[ii, 0]] - tpgpy[pst_trc[ii, 0]]) * 1000
        temp_data3 = dD_rngmig(temp_data2, R0,
                               et[pst_trc[ii, 1]:pst_trc[ii, 2]] - et[pst_trc[ii, 0]],
                               vt[pst_trc[ii, 1]:pst_trc[ii, 2]], 0.0375E-6)
        # azimuth migration
        temp_data4 = dD_azmig(temp_data3, R0,
                               et[pst_trc[ii, 1]:pst_trc[ii, 2]] - et[pst_trc[ii, 0]],
                               vt[pst_trc[ii, 1]:pst_trc[ii, 2]])
        logging.debug("{:s}: Shape of azimuth-migrated line {:3d}: {!r}".format(debugtag, ii, temp_data4.shape))
        # construct the radargram range line
        temp_data5 = dD_traceconstructor(temp_data4)
        # multilook and output
        if looks != 1:
            tempA = np.arange(-np.floor(looks/2), np.ceil(looks/2), 1, dtype=int)
            for jj in range(looks):
                rl[jj, ii, :] = temp_data5[tempA[jj], :]
        else:
            rl[ii] = temp_data5[0, :]
    # end for i

    # produce a final combined multilook product if desired
    if looks != 1 and comb_ml:
        logging.debug("{:s}: Producing final combined multilook product".format(debugtag))
        for ii in range(looks):
            for jj in range(len(pst_trc)):
                rl2[jj, :] = rl2[jj, :] + np.square(np.abs(rl[ii, jj, :]))
                # TODO: faster?
                #rl2[jj, :] += np.square(np.abs(rl[ii, jj, :]))
        rl = np.sqrt(rl2)

    return rl, pst_trc

def mf_create(Er, recal_int, Rmin, R0, surft, dx, n):
    '''
    Define the matched filters to be used in the matched filter SAR processor.

    Input:
    -----------
            Er: assumed relative dielectric permittivity of the subsurface
     recal_int: range sample interval to recalculate the matched filter if an
                Er other than 1 is given
          Rmin: distance to the top of the data array considering the full
                track [m]
            R0: spacecraft-surface distance at the mid-aperture position [m]
         surft: time samples between window start and nadir surface echo [s]
                -- assumes surface echo to be the strongest return
            dx: along-track distance offsets relative to the mid-aperture
                position [m]
             n: number of range samples

    Output:
    -----------
        out: array of matched filters
    '''

    # define the output
    mf_int = recal_int + 1
    mf = np.zeros((len(dx), n), dtype=float)

    # calculate matched filters
    for jj in range(n):
        if jj <= int(surft):
            # if time sample is above or at the nadir surface in the
            # middle of the aperture
            R = Rmin + jj * 0.0375E-6 * 299792458 / 2
            mf[:, jj] = (np.sqrt(R**2 + dx**2) - R)
        else:
            if int(Er) == 1:
                R = Rmin + jj * 0.0375E-6 * 299792458 / 2
                mf[:, jj] = (np.sqrt(R**2 + dx**2) - R)
            else:
                if mf_int < recal_int:
                    mf_int += 1
                    mf[:, jj] = mf[:, jj - 1]
                else:
                    mf_int = 0
                    # if the time sample is below the surface
                    d = (jj - surft) * 0.0375E-6 * (299792458 / np.sqrt(Er)) / 2
                    p = np.zeros((len(dx), 5))
                    p[:, 0] = (Er - 1) * np.ones((len(dx), 1))[:, 0]
                    p[:, 1] = -2 * (Er - 1) * dx
                    p[:, 2] = (Er - 1) * dx**2 + (Er * R0**2) - d**2
                    p[:, 3] = 2 * d**2 * dx
                    p[:, 4] = -d**2 * dx**2
                    # solve the quadratic for a flat interface
                    for kk in range(len(dx)):
                        # select the appropriate root
                        R = np.roots(p[kk, :])
                        if dx[kk] <= 0:
                            if dx[kk] - np.real(R[3]) <= 0:
                                S = np.real(R[3])
                            else:
                                S = np.real(R[2])
                        elif dx[kk] >= 0:
                            if dx[kk] - np.real(R[3]) >= 0:
                                S = np.real(R[3])
                            else:
                                S = np.real(R[2])
                        dd = np.sqrt(d**2 + S**2) - d
                        dR = np.sqrt(R0**2 + (dx[kk] - S)**2) - R0
                        mf[kk, jj] = (dR + np.sqrt(Er) * dd)

    return mf

def mf_apply(data, mf):
    '''
    Function to apply the pre-determine matched filters to the unfocused
    radargram.

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - fast-time samples organized by column
        mf: matched filter matrix

    Output:
    -----------
        out: matched filter sar processed radar data
    '''

    mf = np.exp(1j * ((4 * np.pi) / (299792458 / 25E6)) * mf)
    temp_data_fd = np.fft.fft(data, axis=1, norm='ortho')
    out = np.fft.ifft(np.multiply(temp_data_fd, np.conj(mf)), axis=1, norm='ortho')

    return out

def matched_filter(data, dpst, La, Er, af0, recal_int, tlp, et, rxwot, comb_ml=True):
    '''
    Matched filter SAR processing of SHARAD radar data. Based on existing
    airborne radar (HiCARS) focuser.

    This version is not setup to handle sloping surfaces and assumes everything
    below the surface is homogeneous (no change in Er)

    Inputs:
    -------------
           data: complex ionosphere-corrected radar data
           dpst: distance along the track to post radargram columns [m]
             La: aperture length [s]
             Er: relative dielectric permittivity of the subsurface
            af0: low-pass Doppler frequency for 2D filter [Hz]
                 -- effectively the Doppler frequency bandwidth
      recal_int: range index interval to recalculate the matched filter below
                 the surface
            tlp: vector of interpolated spacecraft positions along its ground
                 track [km]
             et: vector of ephemeris times [s]
          rxwin: vector of corrected times for receive window openings [s]
        comb_ml: optional flag for what type of multilook product we want to
                 generate if multilook processing is to be performed
                 - True: combine and produce a single multilook-ed image
                 - False: produce a three axis array of individual looks

     Outputs:
    -------------
     sar_data: SAR focused radar data
      pst_trc: range line numbers of posted SAR-corrected range lines
    '''

    # define posting centers for SAR columns
    pst_trc, pst_et = sar_posting(dpst, La, int(len(data)), tlp, et)
    del pst_et

    # calculate clostest distance to the top of the data array (i.e. sample 0)
    min_rxwin = min(rxwot)
    Rmin = min_rxwin * 299792458 / 2

    # interpolate the positional vectors as required
    tlp, _ = smooth.smooth(tlp)
    #tlp = vector_interp(tlp)

    ## define the number of looks
    looks = int(np.round(2 * La * af0))
    if looks == 0:
        looks = 1
    elif looks % 2 == 0:
        looks = looks - 1
    print('Number of looks in matched filter SAR processing is', looks)

    # predefine output and start sar processor
    if looks != 1:
        rl = np.zeros((looks, len(pst_trc), len(data[0])), dtype=float)
        rl2 = np.zeros((len(pst_trc), len(data[0])), dtype=float)
    else:
        rl = np.zeros((len(pst_trc), len(data[0])), dtype=float)
    for ii in range(len(pst_trc)):
        if ii % 50 == 0:
            print('Working matched filter SAR column', ii, 'of', len(pst_trc))
        if pst_trc[ii, 1] != 0 and pst_trc[ii, 2] != 0:
            # select data within the aperture
            temp_data = data[pst_trc[ii, 1]:pst_trc[ii, 2]]
            # time shift to align traces and remove rx opening time changes and
            # spacecraft radius changes
            temp_data2 = rx_opening(temp_data,
                                    rxwot[pst_trc[ii, 1]:pst_trc[ii, 2]] - min(rxwot),
                                    0.0375E-6)
            # 2D filter the data
            temp_data3 = twoD_filter(temp_data2, 0.0375E-6, 0, 13E6, 10 * af0, 10 * af0 + 0.1,
                                     1/La, len(temp_data), np.size(temp_data, axis=1))
            # create the matched filter matrix
            surft = np.argmax(np.abs(temp_data[pst_trc[ii, 0] - pst_trc[ii, 1]]))
            R0 = Rmin + surft * 0.0375E-6 * 299792458 / 2
            dx = (tlp[pst_trc[ii, 1]:pst_trc[ii, 2]] - tlp[pst_trc[ii, 0]]) * 1000
            mf = mf_create(Er, recal_int, Rmin, R0, surft, dx,
                           np.size(temp_data, axis=1))
            # apply the matched filter matrix
            temp_data4 = np.abs(np.fft.fft(mf_apply(temp_data3, mf),
                                           axis=0, norm='ortho'))
            # multilook and output
            if looks != 1:
                tempA = np.arange(-np.floor(looks/2),
                                  np.ceil(looks/2), 1, dtype=int)
                for jj in range(looks):
                    rl[jj, ii, :] = temp_data4[tempA[jj], :]
            else:
                rl[ii] = temp_data4[0, :]

    # produce a final combined multilook product if desired
    if looks != 1 and comb_ml:
        for ii in range(looks):
            for jj in range(len(pst_trc)):
                rl2[jj, :] = rl2[jj, :] + np.square(np.abs(rl[ii, jj, :]))
        rl = np.sqrt(rl2)

    return rl, pst_trc

def time_to_rxwindow(dt, rxwindow_samp, scradius, pri, instrument):
    '''
    Determine the time after pulse transmission at which the receive window
    opens while accounting for changes in spacecraft radius along the
    groundtrack

    Inputs:
    -------------
                  dt: fast-time sample interval [s]
       rxwindow_samp: number of samples before receive window opening for each
                      range line
            scradius: spacecraft distance from the center of mass of target
                      body for each range line [km]
                 pri: pulse repitition interval code (SHARAD only)

     Outputs:
    -------------
      out: time-of-flight between signal transmission and receive window
           opening
    '''

    if instrument == 'SHARAD':
        out = np.zeros(len(pri), dtype=float)
        for ii in range(len(pri)):
            if pri[ii] == 1: pri_step = 1428E-6
            elif pri[ii] == 2: pri_step = 1429E-6
            elif pri[ii] == 3: pri_step = 1290E-6
            elif pri[ii] == 4: pri_step = 2856E-6
            elif pri[ii] == 5: pri_step = 2984E-6
            elif pri[ii] == 6: pri_step = 2580E-6
            else: pri_step = 0
            out[ii] = rxwindow_samp[ii] * dt + pri_step - 11.98E-6
        out = out - (2 * (scradius - min(scradius)) * 1000 / 299792458)
    elif instrument == 'MARSIS':
        out = rxwindow_samp * dt
        for ii in range(len(rxwindow_samp)):
            if ii == 0:
                scrad = scradius[ii]
            elif rxwindow_samp[ii] != rxwindow_samp[ii - 1]:
                scrad = scradius[ii]
            out[ii] = out[ii] - (2 * (scrad - min(scradius)) * 1000 / 299792458)

    return out

def track_interpolation(indata, intrack, step, datatype='vector'):
    '''
    interpolate a dataset to a new constant trace spacing

    Inputs:
    -------------
           indata: input data to be interpolated
            track: initial track trace spacing [km]
             step: desired constant along track trace spacing
      indata_type: flag for if the data to be interpolated is a vector or a
                   radargram

     Outputs:
    -------------
        outputs an array or vector at the new desired trace spacing
    '''

    # normalize the alongtrack positions such that they start at 0 and set up
    # the output
    track_norm, _ = smooth.smooth(intrack - np.min(intrack))
    #track_norm = vector_interp(intrack - np.min(intrack))
    interpolated_track = np.arange(0, np.max(track_norm), step)
    if datatype == 'radargram':
        interpolated_data = np.zeros((len(interpolated_track), np.size(indata, axis=1)), dtype=complex)
    elif datatype == 'vector':
        interpolated_data = np.zeros((len(interpolated_track), ), dtype=float)

    # perform the interpolation
    if datatype == 'radargram':
        for ii in range(np.size(indata, axis=1)):
            #if ii % 250 == 0:
            #    print('-- interpolating range sample', ii, 'of', np.size(interpolated_data, axis=1))
            interpolated_data[:, ii] = np.interp(interpolated_track, track_norm, indata[:, ii])
    elif datatype == 'vector':
        roll = indata - np.roll(indata, 1)
        tempY = indata[np.argwhere(roll != 0)[:, 0]]
        tempX = track_norm[np.argwhere(roll != 0)[:, 0]]
        interpolated_data = np.interp(interpolated_track, tempX, tempY)

    return interpolated_data, interpolated_track

def apertures(alongtrack, posting_interval, aperture_length, debug=False):
    '''
    algorithm to define aperture limits along the range line of interest.
    apertures are defined by which samples belong in which aperture

    Inputs:
    -------------
             alongtrack: vector of along-track ephemeris times [s]
       posting_interval: slow-time sample interval at which to post sar columns
                         [samples]
        aperture_length: aperture length [km]

    Outputs:
    -------------
      N x 3 array with along track indices corresponding to the start (column
      1), midpoint (column 1), and end (column 2) of each aperture
    '''
    # normalize alongtrack distance
    alongtrack = alongtrack - np.min(alongtrack)

    start_trace = np.min(np.argwhere(alongtrack >= (aperture_length / 2)))
    end_trace = np.max(np.argwhere(alongtrack - alongtrack[-1] <= -1 * (aperture_length / 2)))
    aperture_centers = np.arange(start_trace, end_trace, posting_interval)
    centers = alongtrack[aperture_centers]
    aperture_bounds = np.zeros((len(centers), 3), dtype=int)
#    ii = 0
#    if ii == 0:
    for ii in range(len(aperture_centers)):
        aperture_bounds[ii, 1] = aperture_centers[ii]
        lower = np.abs(alongtrack - (centers[ii] - (aperture_length / 2)))
        upper = np.abs(alongtrack - (centers[ii] + (aperture_length / 2)))
        aperture_bounds[ii, 0] = np.argwhere(lower == np.min(lower))[0]
        aperture_bounds[ii, 2] = np.argwhere(upper == np.min(upper))[0]

    if debug:
        plt.figure()
        plt.plot(aperture_bounds[:, 0], label='lower')
        plt.plot(aperture_bounds[:, 1], label='center')
        plt.plot(aperture_bounds[:, 2], label='upper')
        plt.legend()

    return aperture_bounds

def define_R(aperture, latitude, longitude, scradius, topography):
    '''
    Estimate the distance between the spacecraft and the mid-aperture nadir
    surface for each position within the aperture

    Inputs:
    -------------
        aperture: range line bounds on the aperture under consideration
        latitude: vector of interpolated latitudes across the FULL orbit track
                  [deg]
       longitude: vector of interpolated longitudes across the FULL orbit track
                  [deg]
        scradius: vector of interpolated spacecraft radii across the FULL
                  orbit track [km]
      topography: vector of interpolated surface radii across the FULL
                  orbit track [km]

     Outputs:
    -------------
        output a vector of distances between the spacecraft at every point
        within the aperture and the mid-aperture nadir surface
    '''

    # select data within aperture
    apt_longitude = longitude[aperture[0]:aperture[2], ]
    apt_latitude = latitude[aperture[0]:aperture[2], ]
    apt_scradius = scradius[aperture[0]:aperture[2], ]

    # define the position of the mid-aperture nadir surface point in spherical
    # coordinates
    srf_rad = topography[aperture[1], ]
    srf_inc = np.pi * ((90 - latitude[aperture[1], ]) / 180)
    srf_azi = 2 * np.pi * (longitude[aperture[1], ] / 360)

    # define the position of spacecraft across the aperture in spherical
    # coordinates
    sc_rad = apt_scradius
    sc_inc = np.pi * ((90 - apt_latitude) / 180)
    sc_azi = 2 * np.pi * (apt_longitude / 360)

    # define the distance between spacecraft and nadir surface at the midpoint
    # of the aperture
    R = np.zeros(len(sc_azi), dtype=float)
    for jj in range(len(R)):
        tempA = np.square(sc_rad[jj])
        tempB = np.square(srf_rad)
        tempC = 2 * sc_rad[jj] * srf_rad
        tempD = np.sin(sc_inc[jj]) * np.sin(srf_inc) * np.cos(sc_azi[jj] - srf_azi)
        tempE = np.cos(sc_inc[jj]) * np.cos(srf_inc)
        R[jj] = np.sqrt(tempA + tempB - tempC * (tempD + tempE))

    return R

def range_migration(data, R, dt):
    '''
    Fast-time phase shift of range lines within the aperture according to the
    predicted distance to the mid-aperture nadir surface point. Shifts are
    applied relative to the mid-aperture range line.

    Inputs:
    -------------
      data: input radar data within the aperture
         R: distance between spacecraft and mid-aperture nadir surface across
            the aperture [km]
        dt: fast-time temporal sampling interval [s]

     Outputs:
    -------------
        range-migrated radar data
    '''

    # define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    #convert distance to two-way time
#    dt_aperture = 2000 * (R - np.min(R)) / 299792458
    Rmid = R[int(np.ceil(np.divide(len(R), 2)))]
    dt_aperture = np.divide(np.multiply(2000, R - Rmid), 299792458)
#    plt.figure(); plt.plot(dt_aperture)

    # apply phase shift to each range line corresponding to required range
    # migration
    fs = np.fft.fftfreq(np.size(data, axis=1), dt)
    for jj in range(len(data)):
        tempB = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                            np.exp(1j * 2 * np.pi * dt_aperture[jj] * fs))
        out[jj] = np.fft.ifft(tempB, norm='ortho')

    return out

def azimuth_migration(data, R, f):
    '''
    Doppler shift range lines within the aperture according to the predicted
    distance to the mid-aperture nadir surface point. Shifts are applied
    relative to the mid-aperture range line

    Inputs:
    -------------
      data: input radar data within the aperture
         R: distance between spacecraft and mid-aperture nadir surface across
            the aperture [km]
         f: highest frequency within the radar bandwidth

     Outputs:
    -------------
        azimuth-migrated radar data
    '''

#    dr = -1000 * (R - np.min(R))
    Rmid = R[int(np.ceil(np.divide(len(R), 2)))]
    dr = np.multiply(-1000, (R - Rmid))

    dphi = np.exp(-1j * ((4 * np.pi / (299792458 / f)) * dr))
    dphi = np.transpose(np.broadcast_to(np.transpose(dphi), (np.size(data, axis=1), len(dphi))))
    out = np.multiply(data, np.conj(dphi))

    return out

def hann_window(data):
    '''
    calculate the frequency spectrum for each range spectrum and apply a Hann
    window to suppress azimuth sidelobes

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain

    Output:
    -----------
        out: array of correctly time-positioned range lines
    '''

    temp = np.fft.fftshift(np.fft.fft(data, axis=0, norm='ortho'), axes=(0,))
    hann = np.hanning(len(temp))
    hann = np.transpose(np.broadcast_to(np.transpose(hann), (np.size(data, axis=1), len(hann))))
    out = np.fft.ifftshift(np.multiply(temp, hann), axes=(0,))

    return out

def doppler_centroid(R, PRF, f, window=0, plot=False):
    '''
    Estimate the instantaneous Doppler centroid for the mid-aperture range
    line. This Doppler centroid will used to extract the focused radar data
    from the correct Doppler bin.

    Input:
    -----------
         R: range to the mid-aperture range line from evey position within the
            aperture [km]
       PRF: pulse repetition frequency [Hz]
         f: signal frequency [Hz]
    window: number of samples around the mid-aperture range line used to
            define the tangent


    Output:
    -----------
        out: index of the estimated Doppler centroid
    '''

    # estimate Doppler frequency for the mid-aperture range line
    l = 299792458 / f
    t = np.multiply(np.divide(1, PRF), np.arange(0, len(R)))
    midt = t[int(np.ceil(np.divide(len(R), 2)))]
    i0 = np.argmin(np.abs(t - midt))
    if window == 0:
        x1 = t[i0:i0 + 2]
        y1 = np.multiply(R[i0:i0 + 2], 1000)
        dydx, = np.diff(y1)/np.diff(x1)
    else:
        step_down = int(np.floor(window / 2))
        step_up = int(np.ceil(window / 2))
        x1 = t[i0 - step_down:i0 + step_up]
        y1 = np.multiply(R[i0 - step_down:i0 + step_up], 1000)
        dydx = np.divide(y1[len(y1) - 1] - y1[0], x1[len(x1) - 1] - x1[0])

    if plot:
        tngnt = lambda x: dydx*x + (y1[0]-dydx*x1[0])
        plt.figure()
        plt.plot(t, R * 1000)
        plt.plot(t[i0], R[i0] * 1000, "or")
        plt.plot(t, tngnt(t), label="tangent")
        plt.legend()

    fDop = np.multiply(np.divide(-2, l), dydx)
    # print(fDop)

    # find the closest discrete Doppler frequency
    discrete_fDop = np.fft.fftfreq(len(R), np.divide(1, PRF))
    out = np.argmin(np.abs(discrete_fDop - fDop))

    return int(out)

def extract_mola(data, lat, lon, bounds):
    '''
    extracting mola surface elevation at a defined latitude and longitude

    Inputs:
    -------------
        data: matrix of mola topography
         lat: latitude at which we want to extract surface heights
         lon: longitude at which we want to extract surface heights
      bounds: latitude and longitude bounds on the data matrix
              [min_lat, max_lat, min_lon, max_lon]

    Outputs:
    -------------
        topo: interpolated surface at the specified latitude and longitude
    '''

    minlat = bounds[0]
    maxlat = bounds[1]
    minlon = bounds[2]
    maxlon = bounds[3]

    # define latitude vector
    y = np.flipud(np.linspace(minlat, maxlat, np.size(data, axis=0)))
    x = np.linspace(minlon, maxlon, np.size(data, axis=1))

    # find lat/log positions surrounding the point of interest
    A = np.argmin(np.abs(y - lat))
    if y[A] - lat > 0: B = A - 1
    elif y[A] - lat < 0: B = A + 1
    else: B = A
    C = np.argmin(np.abs(x - lon))
    if x[C] - lon > 0: D = C - 1
    elif x[C] - lon < 0: D = C + 1
    else: D = C

    # weight surrounding points for topography at the point of interest
    if A != B and D != C:
        crnrAC = data[A, C]
        crnrAD = data[A, D]
        crnrBC = data[B, C]
        crnrBD = data[B, D]
        dA = np.abs(y[A] - lat) / y[1]
        dB = np.abs(y[B] - lat) / y[1]
        dC = np.abs(x[C] - lon) / x[1]
        dD = np.abs(x[D] - lon) / x[1]
        dcrnrAC = (1 - (np.sqrt(dA**2 + dC**2) / np.sqrt(2)))
        dcrnrAD = (1 - (np.sqrt(dA**2 + dD**2) / np.sqrt(2)))
        dcrnrBC = (1 - (np.sqrt(dB**2 + dC**2) / np.sqrt(2)))
        dcrnrBD = (1 - (np.sqrt(dB**2 + dD**2) / np.sqrt(2)))
        wAC = dcrnrAC / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wAD = dcrnrAD / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wBC = dcrnrBC / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wBD = dcrnrBD / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        topo = wAC * crnrAC + wAD * crnrAD + wBC * crnrBC + wBD * crnrBD
    elif A != B and D == C:
        crnrAC = data[A, C]
        crnrBC = data[B, C]
        dA = np.abs(y[A] - lat) / y[1]
        dB = np.abs(y[B] - lat) / y[1]
        dcrnrAC = (1 - dA)
        dcrnrBC = (1 - dB)
        wAC = dcrnrAC / (dcrnrAC + dcrnrBC)
        wBC = dcrnrBC / (dcrnrAC + dcrnrBC)
        topo = wAC * crnrAC + wBC * crnrBC
    elif A == B and D != C:
        crnrAC = data[A, C]
        crnrAD = data[A, D]
        dC = np.abs(x[C] - lon) / x[1]
        dD = np.abs(x[D] - lon) / x[1]
        dcrnrAC = (1 - dC)
        dcrnrAD = (1 - dD)
        wAC = dcrnrAC / (dcrnrAC + dcrnrAD)
        wAD = dcrnrAD / (dcrnrAC + dcrnrAD)
        topo = wAC * crnrAC + wAD * crnrAD
    else:
        topo = data[A, C]

    return topo

def load_mola(pth):
    '''
    Python script for plotting MOLA .img data

    Inputs:
    -------------
        pth: path to folder containing all required MOLA .img files

    Outputs:
    -------------
       topo: matrix of MOLA radii
    '''

    label = pvl.load(pth.replace('.img', '.lbl'))
    lon_samp = label['IMAGE']['LINE_SAMPLES']
    lat_samp = label['IMAGE']['LINES']
    dtype = []
    for ii in range(lon_samp):
        dtype.append(('Line_' + str(ii), '>i2'))
    dtype = np.dtype(dtype)
    fil = glob.glob(pth)[0]
    arr = np.fromfile(fil, dtype=dtype)
    out = np.reshape(arr, [1, lat_samp])[0]
    dfr = pd.DataFrame(arr)

    topo = np.zeros((lat_samp, lon_samp))
    for jj in range(lon_samp):
        topo[:, jj] = dfr[0:lat_samp]['Line_' + str(jj)].as_matrix()
    del dtype, arr, dfr, fil, out

    return topo

def inco_stack(data, fresnel):
    '''
    Incoherent stacking data to a new trace posting interval

    Inputs:
    ------------
         data: data to be stacked
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams

    Output:
    ------------
        stacked array
    '''

    # incoherently stack to desired trace spacing
    indices = np.arange(np.floor(fresnel / 2) + 1, np.size(data, axis=1), fresnel) - 1
    if np.size(data, axis=1) - indices[-1] < np.floor(fresnel / 2):
        col = len(indices) - 1
    else:
        col = len(indices)
    output = np.zeros((np.size(data, axis=0), col), dtype=float)
    for ii in range(col):
        start_ind = int(indices[ii] - np.floor(fresnel / 2))
        end_ind = int(indices[ii] + np.floor(fresnel / 2))
        output[:, ii] = np.sum(data[:, start_ind:end_ind], axis=1)

    return output

def snr(data, noise_window=250):
    '''
    converting radargram voltages to SNR powers

    Inputs:
    ------------
         data: radargram

    Output:
    ------------
        radargram in SNR
    '''

    out = np.zeros((len(data), np.size(data, axis=1)), dtype=float)
    for jj in range(len(data)):
        noise = np.sqrt(np.mean(np.abs(data[jj, 0:noise_window])**2))
        out[jj, :] = np.divide(np.abs(data[jj, :]), noise)

    out = 20 * np.log10(out)

    return out

def ellipsoidal_distance(lat1, long1, lat2, long2, a, b, number_iterations=8):
    '''
    Calculating distance between two points on a reference ellipsoid

    Inputs:
    ------------
                   lat1: latitude of position 1 [deg]
                  long1: longitude of position 1 [deg]
                   lat2: latitude of position 2 [deg]
                  long2: longitude of position 2 [deg]
                      a: equatorial radius of the reference ellipsoid [m]
                      b: semi-minor axis of the reference ellipsoid [m]
      number_iterations: number of iterations used to solve for ellipsoidal
                         distance

    Output:
    ------------
        output is the distance between the two points along the ellipsoid
    '''

    f = np.divide((a - b), a)   # elipsoid flattening
    tolerance = 1e-4           # to stop iteration

    phi1, phi2 = lat1, lat2
    # calculate the reduced latitudes
    U1 = scipy.arctan(np.multiply((1 - f), scipy.tan(phi1)))
    U2 = scipy.arctan(np.multiply((1 - f), scipy.tan(phi2)))
    L1, L2 = long1, long2
    L = L2 - L1

    lambda_old = L + 0
    ind = 0

    while True:

        ind += 1
        t = np.square(scipy.cos(U2) * scipy.sin(lambda_old))
        t += np.square(scipy.cos(U1) * scipy.sin(U2) - scipy.sin(U1) * scipy.cos(U2) * scipy.cos(lambda_old))
        sin_sigma = np.sqrt(t)
        cos_sigma = scipy.sin(U1) * scipy.sin(U2) + scipy.cos(U1) * scipy.cos(U2) * scipy.cos(lambda_old)
        sigma = scipy.arctan2(sin_sigma, cos_sigma)

        sin_alpha = scipy.cos(U1) * scipy.cos(U2) * scipy.sin(lambda_old) / sin_sigma
        cos_sq_alpha = 1 - np.square(sin_alpha)
        cos_2sigma_m = cos_sigma - 2 * scipy.sin(U1) * scipy.sin(U2) / cos_sq_alpha
        C = f * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha)) / 16

        t = sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * np.square(cos_2sigma_m)))
        lambda_new = L + (1 - C) * f * sin_alpha * t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        elif ind == number_iterations:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha * ((np.square(a) - np.square(b)) / np.square(b))
    A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    t = cos_2sigma_m + 0.25 * B * (cos_sigma * (-1 + 2 * np.square(cos_2sigma_m)))
    t -= (B / 6) * cos_2sigma_m * (-3 + 4 * np.square(sin_sigma))*(-3 + 4 * np.square(cos_2sigma_m))
    delta_sigma = B * sin_sigma * t
    out = b * A * (sigma - delta_sigma)

    return out

def groundtrack(latitude, longitude, a=3396.19E3, b=3376.2E3):
    '''
    Calculating progressive groundtrack distances for each range line. The
    first range line is positioned at 0 meters.

    Inputs:
    ------------
      lat: nadir latitude coordinates for each range line [deg]
     long: nadir longitude coordinates for each range line [deg]
        a: equatorial radius of the ellipsoid [m] (defaults to values for Mars)
        b: semi-minor axis of the ellipsoid [m] (defaults to values for Mars)

    Output:
    ------------
        output is vector of groundtrack positions [km]
    '''

    output = np.zeros(len(latitude))

    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)
    for ii in range(1, len(latitude)):
        s = ellipsoidal_distance(latitude[ii-1], longitude[ii-1], 
                                 latitude[ii], longitude[ii], a, b)
        output[ii] = s + output[ii - 1]

    return np.divide(output, 1000)

def delay_doppler_v2(indata, interpolate_dx, posting_interval, data_trim, 
                  aperture_length, ephemeris, topography, pri, scradius, 
                  rxwindow_samp, latitude, longitude, band, 
                  instrument='SHARAD', marID='1', debugtag='SAR'):
    '''
    Second generation of a Delay Doppler SAR focuser that tries to move towards
    dealing with some of the peculiarities of REASON. This version of the
    focuser can handle both SHARAD and MARSIS data.

    New features in this version of the processor:
    -- range cell migration is characterized using spherical coordinates
    -- apertures are defined in terms of groundtrack distances
    -- radar data are interpolated to a constant groundtrack range line spacing
    -- Doppler centroid is estimated for each aperture based on the
       instantaneous Doppler frequency at the mid-aperture position.

    It is important to note that there is no multilooking in this SAR focuser.
    Multilooking is expected to be done by incoherent stacking of the
    focused radar product to some final trace spacing interval.

    Inputs:
    -------------
                indata: complex-valued ionosphere-corrected radar data
        interpolate_dx: groundtrack distance [m] between interpolated range
                        lines
                        -- set to 0 to skip interpolation
      posting_interval: interval in interpolated range lines at which to place
                        a SAR column
             data_trim: maximum fast-time sample at which to trim the input
                        data
                        -- set to 0 to skip trim
       aperture_length: groundtrack length [km] of the desired synthetic
                        aperture
             ephemeris: list of ephemeris times [s] associated with each input
                        range line
            topography: list of nadir surafce radius [km] from the center of
                        target mass associated with each input range line
                   pri: list of pri codes associated with each input range line
                        -- empty matrix for 'MARSIS' processing
              scradius: list of spacecraft orbital radii [km] associated with
                        each input range line
         rxwindow_samp: list of enumber of samples between pulse transmission
                        and receive window opening associated with each input
                        range line
              latitude: list of nadir latitude [deg] associated with each
                        input range line
             longitude: list of nadir longitude [deg] associated with each
                        input range line
                  band: list of 'MARSIS' band (center frequency) associated
                        with each input range line
                        -- empty matrix for 'SHARAD' processing
            instrument: 'SHARAD' or 'MARSIS'
                 marID: 'MARSIS' frequency channel selector; '1' or '2'

              debugtag: Optional unique tag prepended to debugging messaages

    Outputs:
    -------------
           focused_data: complex-valued array of focused radar data
       interp_ephemeris: interpolated ephemeris data associated with each SAR
                         range line
               aperture: bounds and center positions of each range line in the
                         INPUT radar data array
    '''

    # setup instrument parameters
    if instrument == 'SHARAD':
        PRF = 175         # alongtrack PRF [Hz]
        fc = 20E6         # signal center frequency [Hz]
        BW = 10E6         # signal bandwidth [Hz]
        dt = 0.0375E-6    # fast-time sample interval [s]
        fs = 1 / dt       # fast time sampling rate [Hz]
    elif instrument == 'MARSIS':
        if marID == '1':
            F = 'F1'
        elif marID == '2':
            F = 'F2'
        PRF = 130         # alongtrack PRF [Hz]
        BW = 1E6          # signal bandwidth [Hz]
        dt = 1 / 2.8E6    # fast-time sample interval [s]
        fs = 1 / dt       # fast time sampling rate [Hz]
        if F == 'F1':
            if int(band[0]) == 0: fc = 4.0E6
            elif int(band[0]) == 1: fc = 1.8E6
            elif int(band[0]) == 2: fc = 3.0E6
            elif int(band[0]) == 3: fc = 5.0E6
        elif F == 'F2':
            if int(band[0]) == 0: fc = 1.8E6
            elif int(band[0]) == 1: fc = 4.0E6
            elif int(band[0]) == 2: fc = 3.0E6
            elif int(band[0]) == 3: fc = 5.0E6

    # define positions in terms of groundtrack distances
    grndtrck = groundtrack(latitude, longitude)

    # trim radar data if desired
    if len(data_trim) != 0:
        indata = indata[:, 0:data_trim[0]]

    # correct time to start of receiver window
    rxwindow_time = time_to_rxwindow(dt, rxwindow_samp, scradius, pri,
                                     instrument)

    # align data relative to a common start time
    # --> common start time set to be the minimum rx opening time. shifting is
    #     done in the fast-time Fourier domain
    dc_rxshift = np.min(rxwindow_time)
    rx_shifts = rxwindow_time - dc_rxshift
    aligned_data = rx_opening(indata, rx_shifts, dt)

    #plt.figure()
    #plt.imshow(np.transpose(np.abs(aligned_data)), aspect='auto')
    #plt.show()

    # apply 2D filter (both in Fourier domain)
    # --> step is currently being skipped for 'MARSIS' data
    if instrument == 'SHARAD':
        filtered_data = twoD_filter(aligned_data, dt, 0, 13E6, 10, 60, (1 / PRF),
                                    len(aligned_data), 
                                    np.size(aligned_data, axis=1))
    elif instrument == 'MARSIS':
    #    filtered_data = sar.twoD_filter(aligned_data, dt, 0, 2E6, 10, 30, (1 / PRF),
    #                            len(aligned_data), np.size(aligned_data, axis=1))
        filtered_data = aligned_data

    # interpolate data to constant alongtrack trace spacing
    if interpolate_dx == 0:
        interp_ephemeris = ephemeris
        interp_latitude = latitude
        interp_longitude = longitude
        interp_scradius = scradius
        interp_topography = topography
        interp_data = filtered_data
        interp_rxwindow_time = rxwindow_time
        interp_groundtrack = grndtrck
    else:
        interp_ephemeris, _ = track_interpolation(ephemeris, grndtrck, interpolate_dx / 1000)
        interp_latitude, _ = track_interpolation(latitude, grndtrck, interpolate_dx / 1000)
        interp_longitude, _ = track_interpolation(longitude, grndtrck, interpolate_dx / 1000)
        interp_scradius, _ = track_interpolation(scradius, grndtrck, interpolate_dx / 1000)
        interp_topography, _ = track_interpolation(topography, grndtrck, interpolate_dx / 1000)
        interp_rxwindow_time, _ = track_interpolation(rxwindow_time, grndtrck, interpolate_dx / 1000)
        interp_data, interp_groundtrack = track_interpolation(filtered_data, grndtrck, interpolate_dx / 1000, datatype='radargram')

    # define the bounds of individual sar apertures
    aperture = apertures(interp_groundtrack, posting_interval, aperture_length)

    # predefine the SAR-focused output
    focused_data = np.zeros((len(aperture), np.size(interp_data, axis=1)), dtype=complex)

    # perform Delay Doppler sar focusing
    #ii = 100
    #if ii == 100:
    #for ii in range(400):
    for ii in range(len(aperture)):

        if ii % 1000 == 0:
            logging.debug("{:s}: Working delay Doppler SAR v2 column {:d} of {:d}".format(debugtag, ii, len(aperture)) )

        # select data within aperture
        apt_data = interp_data[aperture[ii, 0]:aperture[ii, 2], :]
    #    plt.figure()
    #    plt.subplot(211); plt.imshow(np.abs(np.transpose(apt_data)), aspect='auto', cmap='jet'); plt.title('input magnitude')
    #    plt.subplot(212); plt.imshow(np.angle(np.transpose(apt_data)), aspect='auto', cmap='jet'); plt.title('input phase')

        # define the distance between the spacecraft and mid-aperture nadir surface
        # point across the aperture
        R = define_R(aperture[ii, :], interp_latitude, interp_longitude, interp_scradius, interp_topography)
    #    plt.figure(); plt.plot(R); plt.title('Distance to mid-aperture nadir position')

        # range migration
        rm_data = range_migration(apt_data, R, dt)
    #    plt.figure()
    #    plt.subplot(211); plt.imshow(np.abs(np.transpose(rm_data)), aspect='auto', cmap='jet'); plt.title('range migrated magnitude')
    #    plt.subplot(212); plt.imshow(np.angle(np.transpose(rm_data)), aspect='auto', cmap='jet'); plt.title('range migrated phase')

        # azimuth migration
        am_data = azimuth_migration(rm_data, R, fc + BW / 2)
    #    plt.figure()
    #    plt.subplot(211); plt.imshow(np.abs(np.transpose(am_data)), aspect='auto', cmap='jet'); plt.title('azimuth migrated magnitude')
    #    plt.subplot(212); plt.imshow(np.angle(np.transpose(am_data)), aspect='auto', cmap='jet'); plt.title('azimuth migrated phase')

        # hann window
        hann_data = hann_window(am_data)
    #    plt.figure(); plt.imshow(np.abs(np.transpose(hann_data)), aspect='auto', cmap='jet'); plt.title('after hann window')

        # Doppler centroid estimation and extraction
        if instrument == 'SHARAD':
            centroid = -1 * doppler_centroid(R, PRF, fc + BW / 2)
        elif instrument == 'MARSIS':
            centroid = doppler_centroid(R, PRF, fc + BW / 2)
        focused_data[ii, :] = hann_data[ centroid, :]
    #    print(ii, centroid)
    #    plt.figure(); plt.plot(np.abs(focused_data[ii, :])); plt.title('focused range line: Doppler bin index ' + str(centroid))

    return focused_data, interp_ephemeris, aperture



