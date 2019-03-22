__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'function library for SAR processing'}}

import logging
import numpy as np

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

    import numpy as np

    ## define posting centers for SAR columns
    ### post radargram columns
    num_pst = np.floor((np.max(tlp)-tlp[0])/(dpst/1000))
    pst_trc = np.zeros((int(num_pst), 3), dtype=float)
    pst_trc[:, 0] = np.round(np.arange(0, idx, idx/num_pst))
    ## define trace ranges to include in SAR window from aperture length
    ### aperture length will be defined in SECONDS so need the ephemeris time
    ### for each post trace
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

def vector_interp(vect):
    '''
    Linear interpolation of a vector to smooth out steps. Values are plotted in
    middle of any discretized zones.

    Inputs:
    -------------
      vect: one-dimensional vector to be interpolated

     Outputs:
    -------------
     out: interpolated vector
    '''

    import numpy as np

    # interpolate a vector
    t1 = np.unique(vect)
    unix = np.zeros((len(t1), 1), dtype=float)
    for ii in range(len(t1)):
        t2 = abs(vect - t1[ii])
        t3 = np.argmin(t2)
        t4 = abs(len(vect) - np.argmin(np.flipud(t2)))
        unix[ii] = np.round(np.mean([t3, t4]))
    out = np.interp(np.arange(0, len(vect), 1), unix[:, 0], t1)

    return out

def twoD_filter(data, rf0, rf1, af0, af1, dres, m, n):
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
           - fast-time samples organized by column
      rf0: low time frequency [Hz]
      rf1: upper time frequency [Hz]
      af0: low-pass Doppler frequency [Hz]
      af1: high-cut Doppler frequency [Hz]
     dres: frequency resoltuion of the Doppler spectrum [Hz]
        m: number of range lines within the aperture
        n: number of time samples along the range lines within the aperture

     Outputs:
    -------------
     out: 2D filtered radargram
    '''

    import numpy as np
    import math

    # range
    rf0i = int(math.floor((rf0 * 0.0375E-6 * n)+0.5)) + 1
    rf1i = int(math.floor((rf1 * 0.0375E-6 * n)+0.5)) + 1
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
    ahann = np.reshape(0.5 + 0.5*np.cos(np.linspace(0.0, 1.0, bw+1) * np.pi),
                       (-1, 1))
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
    Frequency domain-applied phase shift to align traces relative to the
    minimum receive window opening time

    Input:
    -----------
      data: array of complex-valued range lines expressed in the
            slow-time/fast-time domain
            - range lines organized by row
            - fast-time samples organized by column
     rxwot: vector of corrected receive window opening times after chirp
            emission [s]
        dt: fast-time sampling interval [s]

    Output:
    -----------
        out: array of correctly time-positioned range lines
    '''

    # define required temporary parameters
    n = np.size(data, axis=1)
    f = np.fft.fftfreq(n, dt)

    # define the output
    out = np.zeros((len(data), n), dtype=complex)

    # apply phase shift
    for jj in range(len(data)):
        tempA = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                            np.exp(-1j * 2 * np.pi * rxwot[jj] * f))
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

    # define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    # calculate distance from nadir mid-aperture surface point to spacecraft
    # for each range line
    R = np.zeros((len(data), ))
    for jj in range(len(data)):
        R[jj] = np.sqrt(R0**2 + ((et[jj]) * vt[jj])**2)

    #convert distance to two-way time
    dt_aperture = 2 * (R - R0) / 299792458
    #dt_aperture = (R - R0) * (2 / 299792458)

    # apply phase shift to each range line corresponding to required range
    # migration
    f = np.fft.fftfreq(np.size(data, axis=1), dt)
    for jj in range(len(data)):
        tempB = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                            np.exp(1j * 2 * np.pi * dt_aperture[jj] * f))
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

    import numpy as np

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
    import numpy as np

    temp = np.fft.fftshift(np.fft.fft(data, axis=0, norm='ortho'), axes=(0,))
    hann = np.hanning(len(temp))
    hann = np.transpose(np.broadcast_to(np.transpose(hann),
                                        (np.size(data, axis=1), len(hann))))
    out = np.abs(np.fft.ifftshift(np.multiply(temp, hann), axes=(0,)))

    return out

def delay_doppler(data, dpst, La, dBW, tlp, et, scrad, tpgpy, rxwot, vt, comb_ml=True,
                  debugtag="SAR"):
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

    import numpy as np

    # interpolate the positional vectors as required
    tlp = vector_interp(tlp)
    vt = vector_interp(vt)

    ## define posting centers for SAR columns
    pst_trc, pst_et = sar_posting(dpst, La, int(len(data)), tlp, et)
    del pst_et

    ## define the number of looks
    looks = int(np.round(2 * La * dBW))
    if looks == 0:
        looks = 1
    elif looks % 2 == 0:
        looks = looks - 1

    logging.debug("{:s}: Number of looks in delay Doppler SAR processing is {:d}".format(debugtag,looks))

    # predefine output and start sar processor
    if looks != 1:
        rl = np.zeros((looks, len(pst_trc), len(data[0])), dtype=float)
        rl2 = np.zeros((len(pst_trc), len(data[0])), dtype=float)
    else:
        rl = np.zeros((len(pst_trc), len(data[0])), dtype=float)
    for ii in range(len(pst_trc)):
        if ii % 50 == 0:
            logging.debug("{:s}: Working delay Doppler SAR column {:d} of {:d}".format(debugtag, ii, len(pst_trc)) )
        # TODO: convert this to a continue block
        if pst_trc[ii, 1] != 0 and pst_trc[ii, 2] != 0:
            # TODO: gc old temp variables to keep mem footprint low
            # select data within the aperture
            temp_data = data[pst_trc[ii, 1]:pst_trc[ii, 2]]

            # time shift to align traces and remove rx opening time changes and
            # spacecraft radius changes
            temp_data2 = rx_opening(temp_data,
                                    rxwot[pst_trc[ii, 1]:pst_trc[ii, 2]],
                                    0.0375E-6)

            # Shared quantities in param 3

            # range migration
            R0 = (scrad[pst_trc[ii, 0]] - tpgpy[pst_trc[ii, 0]]) * 1000
            temp_data3 = dD_rngmig(temp_data2, R0,
                                   et[pst_trc[ii, 1]:pst_trc[ii, 2]] - et[pst_trc[ii, 0]],
                                   vt[pst_trc[ii, 1]:pst_trc[ii, 2]], 0.0375E-6)
            # azimuth migration
            temp_data4 = dD_azmig( temp_data3, R0,
                                   et[pst_trc[ii, 1]:pst_trc[ii, 2]] - et[pst_trc[ii, 0]],
                                   vt[pst_trc[ii, 1]:pst_trc[ii, 2]])
            # construct the radargram range line
            temp_data5 = dD_traceconstructor(temp_data4)
            # multilook and output
            if looks != 1:
                tempA = np.arange(-np.floor(looks/2), np.ceil(looks/2), 1, dtype=int)
                for jj in range(looks):
                    rl[jj, ii, :] = temp_data5[tempA[jj], :]
            else:
                rl[ii] = temp_data5[0, :]

    # produce a final combined multilook product if desired
    if looks != 1 and comb_ml:
        for ii in range(looks):
            for jj in range(len(pst_trc)):
                rl2[jj, :] = rl2[jj, :] + np.square(np.abs(rl[ii, jj, :]))
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

    import numpy as np

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

    import numpy as np

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

    import numpy as np

    # define posting centers for SAR columns
    pst_trc, pst_et = sar_posting(dpst, La, int(len(data)), tlp, et)
    del pst_et

    # calculate clostest distance to the top of the data array (i.e. sample 0)
    min_rxwin = min(rxwot)
    Rmin = min_rxwin * 299792458 / 2

    # interpolate the positional vectors as required
    tlp = vector_interp(tlp)

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
            temp_data3 = twoD_filter(temp_data2, 0, 13E6, 10 * af0, 10 * af0 + 0.1,
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
