__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'library of functions required for dual-frequency ionospheric correction'}}

'''
Library of fucntions required to attempt a dual frequency ionospheric
correction. The library will be set up for testing with SHARAD and MARSIS data
but will follow the L2 flowchart proposed for REASON.

The goal is to have a function defined for each of the 'Process' boxes found in
the flowchart.
-> The only Process box that likely won't be able to be produced here is the
   VHF co-registration and coherent summation box as since we're testing with
   SHARAD and MARSIS, we don't have two identical channels that can be combined

Version History
    --- V0.1
        --- Author: Kirk Scanlan
        --- Date: Thursday, September 27 2018
'''


import glob
import numpy as np
import pandas as pd



import sys
import pvl

def load_mola(pth):
    '''
    Python script for plotting MOLA .img data

    Inputs:
    -------------
       pth: path to saved .img file

    Outputs:
    -------------
       topo: matrix of MOLA topographies [km]
    '''


    label = pvl.load(pth.replace('.img', '.lbl'))
    lon_samp = label['IMAGE']['LINE_SAMPLES']
    lat_samp = label['IMAGE']['LINES']
    dtype = []
    for ii in range(lon_samp):
        dtype.append(('Line_'+str(ii), '>i2'))
    dtype = np.dtype(dtype)
    fil = glob.glob(pth)[0]
    arr = np.fromfile(fil, dtype=dtype)
    out = np.reshape(arr, [1, lat_samp])[0]
    dfr = pd.DataFrame(arr)

    topo = np.zeros((lat_samp, lon_samp))
    for jj in range(lon_samp):
        topo[:, jj] = dfr[0:lat_samp]['Line_'+str(jj)].values
    del dtype, arr, dfr, fil, out

    return topo

def extract_mola(data, lat, lon, bounds):
    '''
    Fucntion to extract mola surface elevation at a defined latitude and
    longitude.

    Inputs:
    -------------
        data: matrix of mola topography [km]
         lat: latitude at which we want to extract surface heights [deg]
         lon: longitude at which we want to extract surface heights [deg]
      bounds: latitude and longitude bounds on the data matrix [deg]
              [min_lat, max_lat, min_lon, max_lon]

    Outputs:
    -------------
        topo: interpolated surface at the specified latitude and longitude
              [km]
    '''

    import numpy as np

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

def rngcmp(data, chirp_param, instrument):
    '''
    Range compression algorithm using synthetic reference chirp defined by the
    user inputs in chirp_param.

    Input:
    -----------
            data: array of complex-valued range lines expressed in the
                  slow-time/frequency domain.
                  - range lines organized by row
                  - frequency samples organized by column
     chirp_param: array containing parameters definings the reference chirp
                  - entry 0: centre frequency [Hz]
                  - entry 1: temporal sampling rate [Hz]
                  - entry 2: pulse length [s]
                  - entry 3: chirp bandwdith [Hz]
      instrument: specific instrument used in data collection. required to set
                  specific details of the range compression

    Output:
    -----------
        out: array of range compressed range lines expressed in the
             slow-time/fast-time domain.
    '''
    import numpy as np

    # pre-define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    # compute reference chirp phase function
    chrpt = (1 / chirp_param[1]) * np.arange(0, int(chirp_param[2] / (1 / chirp_param[1])))
    a = chirp_param[3] / chirp_param[2]
    fl = chirp_param[0] - chirp_param[3] / 2
    phi = 2 * np.pi * (chrpt * fl + (a/2) * chrpt**2)

    # compute reference chirp
    if instrument == 'SHARAD':
        # SHARAD chirp must be flipped due to aliasing performed during data
        # acquisition
        chrp = np.pad(np.exp(1j * (phi)),
                      (np.size(data, axis=1) - len(chrpt), 0),
                      'constant', constant_values=0)
        chrp_fd = np.fft.fft(np.flipud(chrp), norm='ortho')
    else:
        chrp = np.pad(np.exp(1j * (phi)),
                      (0, np.size(data, axis=1) - len(chrpt)),
                      'constant', constant_values=0)
        chrp_fd = np.fft.fft(chrp, norm='ortho')

    # perform range compression
    for ii in range(len(data)):
        product_fd = np.multiply(np.conj(chrp_fd), data[ii])
        out[ii] = np.fft.ifft(product_fd, norm='ortho')

    return out

def ioncorr(data, TEC, chirp_param, instrument):
    '''
    Ionospheric correction of the radar data for some total electron content.

    Ionospheric correction will be implemented in the slow-time/frequency
    domain as a phase shift applied to each frequency. Attempting to
    implement eqn. 3 from Campbell et al. [2011] DOI:10.1109/LGRS.2011.2143692

    Input:
    -----------
             data: array of complex-valued range lines expressed in the
                   slow-time/fast-time domain.
                   - range lines organized by row
                   - frequency samples organized by column
              TEC: estimate of total electron content [m^-3]
      chirp_param: array containing parameters definings the reference chirp
                   - entry 0: centre frequency [Hz]
                   - entry 1: temporal sampling rate [Hz]
                   - entry 2: pulse length [s]
                   - entry 3: chirp bandwdith [Hz]
       instrument: specific instrument used in data collection. required to set
                   specific details in the padding of the ionosphere correction
                   function

    Output:
    -----------
        out: array of ionosphere corrected range lines expressed in the
             slow-time/fast-time domain.
    '''
    import numpy as np

    # pre-define the output
    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)

    # calculate TEC correction
    t = (1 / chirp_param[1]) * np.arange(0, int(chirp_param[2] / (1 / chirp_param[1])))
    a = chirp_param[3] / chirp_param[2]
    fh = chirp_param[0] + chirp_param[3] / 2
    freq = fh - a * t
    dphi = -1.69E-6 * TEC / freq
    if instrument == 'SHARAD':
        dphi = np.pad(dphi, (np.size(data, axis=1) - len(t), 0),
                      'constant', constant_values=0)
    else:
        dphi = np.pad(dphi, (0, np.size(data, axis=1) - len(t)),
                      'constant', constant_values=0)

    # apply phase shift to each range line
    for jj in range(len(data)):
        # transform to the frequency domain
        data_fd = np.fft.fft(data[jj, :], norm='ortho')
        # apply phase correction and transform back to fast-time
        out[jj] = np.fft.ifft(np.multiply(data_fd, np.exp(-1j * dphi)),
                                           norm='ortho')

    return out

def surfpick(data, threshold, n1, n2):
    '''
    Automatically pick the surface echo from the radargrams as the first
    reflection exhibitng an SNR above the threshold specified by the user. If
    no fast-time sample exhibits an SNR above the defined threshold, or if the
    maximum is below a certain fast-time sample number a NaN is assigned to
    that range line.
    Also able to restrict the analysis to specific latitude ranges by passing
    minimum and maximum latitude bounds. If no bounds are passed, the algorithm
    will consider all data.

    Input:
    -----------
           data: array of complex-valued range lines expressed in the
                 slow-time/fast-time domain.
                 - range lines organized by row
                 - frequency samples organized by column
      threshold: SNR threshold to employ when picking the surface echo
          n1/n2: lower and upper fast-time sample bounds to be used when
                 determining the noise level along each range line

    Output:
    -----------
        out: fast-time samples corresponding to the picked surface echo
    '''

    import numpy as np

    # pre-define the output
    out = np.zeros((len(data), 1), dtype=float)

    # pick surface echo for each range line
    for ii in range(len(data)):
        n = np.mean(np.abs(data[ii, n1:n2]))
        snr = np.abs(data[ii, :]) / n
        if np.max(snr) >= threshold:
            out[ii] = np.argmax(snr)
        else:
            out[ii] = np.nan

    return out

def marsis_band_mute(data, raw_bands, bands_to_use, mar_channel):
    '''
    Remove undesired MARSIS bands from the analysis by eliminating their
    surface picks

    Input:
    -----------
              data: vector of the surface picks for each range line
         raw_bands: vector identifying the MARSIS band each range line in data
                    was collected with
      bands_to_use: MARSIS bands to be included in the final comparison with
                    SHARAD [MHz]
       mar_channel: string identifying which MARSIS channel is under analysis
                    -- different centre frequencies correspond to different
                       indices in the raw_bands vector

    Output:
    -----------
        out: modified vector of surface echo picks
    '''

    import numpy as np

    # find range lines not corresponding to the MARSIS frequencies of interest
    # and mute their surface picks
    if mar_channel == '1':
        for ii in range(len(data)):
            if int(raw_bands[ii]) == 0 and '4.0' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 1 and '1.8' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 2 and '3.0' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 3 and '5.0' not in bands_to_use:
                data[ii] = np.nan
    elif mar_channel == '2':
        for ii in range(len(data)):
            if int(raw_bands[ii]) == 0 and '1.8' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 1 and '4.0' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 2 and '3.0' not in bands_to_use:
                data[ii] = np.nan
            elif int(raw_bands[ii]) == 3 and '5.0' not in bands_to_use:
                data[ii] = np.nan

    return data

def topography(data, lat, long, mola, bounds):
    '''
    Extract mola topography for each surface pick that is not NaN.

    Input:
    -----------
              data: vector of the surface picks for each range line
               lat: vector of the surface latitude of each range line [deg]
              long: vector of the surface longitude of each range line [deg]
              mola: array of pre-loaded MOLA topography [km]
            bounds: latitude and longitude bounds on the MOLA grid [deg]

    Output:
    -----------
         out: vector of MOLA topogrpahies for each range line where a surface
              pick exists [km]
    '''

    import numpy as np

    # pre-define the output
    out = np.zeros((len(data), 1), dtype=float)

    # extract MOLA topography for each range line where a surface pick has been
    # made
    for ii in range(len(data)):
        if np.isnan(data[ii])[0] == False:
            out[ii] = extract_mola(mola, lat[ii], long[ii], bounds)
        else:
            out[ii] = np.nan

    return out

def tec_calc(TEC, sha_sp, sha_topo, sha_scalt, sha_rx, mar_sp, mar_topo, mar_scalt,
             mar_rx, mar_band, mar_channel):
    '''
    Estimate the TEC of the ionosphere by comparing two-way time delays between
    SHARAD and MARSIS surface echoes. Will have to adjust to compensate for the
    differences in spacecraft height and topography.

    Input:
    -----------
              TEC: estimated TEC from the previous iteration step (or zero for
                   the first iteration)
           sha_sp: vector of SHARAD surface picks for each range line
         sha_topo: vector of MOLA topogrpahy at the lat/long position of each
                   SHARAD range line [km]
          sha_alt: vector of MRO altitudes for each range line [km]
           sha_rx: vector of SHARAD receive window opening times [s]
           mar_sp: vector of MARSIS surface picks for each range line
         mar_topo: vector of MOLA topogrpahy at the lat/long position of each
                   MARSIS range line [km]
          mar_alt: vector of MEX altitudes for each range line [km]
           mar_rx: vector of MARSIS receive window opening times [s]
         mar_band: vector identifying the MARSIS centre frequency for each
                   range line [MHz]
      mar_channel: string identifying which MARSIS channel is under analysis
                   -- modification to receive window opening times and
                      different centre frequencies

    Output:
    -----------
         out: vector of TEC estimates for each combination of SHARAD and MARSIS
              range lines [m^-3]
    '''

    import numpy as np

    # prepare the output
    sha_num = len(sha_sp) - len(np.argwhere(np.isnan(sha_sp)))
    mar_num = len(mar_sp) - len(np.argwhere(np.isnan(mar_sp)))
    out = np.zeros((sha_num * mar_num, 1), dtype=float)

    # estimate a TEC for each combination of MARSIS and SHARAD
    ind = 0
    for ii in range(len(sha_sp)):
        if np.isnan(sha_sp[ii])[0] == False:
            for jj in range(len(mar_sp)):
                if np.isnan(mar_sp[jj])[0] == False:
                    # calculate spacecraft altitude and topography
                    # modifications
                    dtopo = mar_topo[jj] - sha_topo[ii]
                    dscalt = mar_scalt[jj] - sha_scalt[ii]
                    # calculate effective time of the surface echo based on
                    # surface picks
                    if mar_channel == '1':  
                        Te_mar_rx = (mar_rx[jj] * (1 / 2.8E6))
                    elif mar_channel == '2':
                        Te_mar_rx = ((mar_rx[jj] - 450E-6 / (1 / 2.8E6)) * (1 / 2.8E6))
                    Te_mar = Te_mar_rx + mar_sp[jj] * (1 / 1.4E6)
                    Te_sha = sha_rx[ii] + sha_sp[ii] * 0.0375E-6
                    # calculate change in effective time
                    dTe = Te_mar - Te_sha - (2 / 299792458) * (dtopo + dscalt * 1000)
                    # add correction for already estimated ionosphere
                    # TODO: GNG turn this into a dict or list lookup table
                    if mar_channel == '1':
                        if int(mar_band[jj]) == 0:
                            df = ((1 / (4.0E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 1:
                            df = ((1 / (1.8E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 2:
                            df = ((1 / (3.0E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 3:
                            df = ((1 / (5.0E6**2)) - (1 / (20E6**2)))
                    elif mar_channel == '2':
                        if int(mar_band[jj]) == 0:
                            df = ((1 / (1.8E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 1:
                            df = ((1 / (4.0E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 2:
                            df = ((1 / (3.0E6**2)) - (1 / (20E6**2)))
                        elif int(mar_band[jj]) == 3:
                            df = ((1 / (5.0E6**2)) - (1 / (20E6**2)))
                    # calculate group delay due to TEC levels already estimated
                    # during previous iterations and remove from the effective
                    # time delay. ioncorr only corrects de-focusing (phase
                    # velocity differences), it does not shift traces to
                    # account for group delays
                    diono = 1.69E-6 * TEC * df / (2 * np.pi)
                    dTe = dTe - diono
                    # calculate TEC
                    out[ind] = (dTe / df) * (2 * np.pi) / 1.69E-6
                    ind += 1

    return out

def optimal_tecu(data, b=50):
    '''
    Extract an estimate for the optimal TEC from the distribution.

    Input:
    -----------
       data: vector of TEC estimates for each combination of SHARAD and MARSIS
             range lines [m^-3]
          b: optional number of bins to use when constructing the histogram

    Output:
    -----------
         avg: optimal TEC estimate [TECU]
         low: TEC one standard deviation lower than the optimal [TECU]
        high: TEC one standard deviation higher than the optimal [TECU]
    '''

    import numpy as np

    hist, edges = np.histogram(data, bins=b)
    cumhist = np.cumsum(hist)
    x = np.argmin(np.abs((cumhist / max(cumhist)) - 0.50))
    y = np.argmin(np.abs((cumhist / max(cumhist)) - 0.16))
    z = np.argmin(np.abs((cumhist / max(cumhist)) - 0.84))
    avgA = edges[x]
    lowA = edges[y]
    highA = edges[z]
    avgB = edges[x + 1]
    lowB = edges[y + 1]
    highB = edges[z + 1]
    avg = (avgA + (avgB - avgA) / 2) / 1E16
    low = (lowA + (lowB - lowA) / 2) / 1E16
    high = (highA + (highB - highA) / 2) / 1E16

    return avg, low, high

def sharad_trim(sha_sp, marb1_sp, marb2_sp, method, number):
    '''
    Trim the number of SHARAD traces to use in the comparison in an effort to
    try and speed the analysis up. The number of SHARAD traces to be used in
    the analysis can be defined in different ways;
    Method A - will be trim SHARAD to be the same length as the number of
               MARSIS traces but will choose which SHARAD traces will be kept
               randomly.
    Method B - Will simply choose a defined number of SHARAD traces
               - If the SHARAD dataset is not as long as the defined number
                 the input dataset will just be output again

    Input:
    -----------
         sha_sp: vector of SHARAD surface picks
       marb1_sp: vector of MARSIS channel 1 surface picks
       marb2_sp: vector of MARSIS channel 2 surface picks
         method: user input defining whether Method A or B is followed
         number: number of SHARAD traces to trim the dataset down to

    Output:
    -----------
         out: vector of trimmed SHARAD surface picks
    '''

    import numpy as np
    import numpy.random

    if method == 'A':
        # define the number of existing MARSIS surface picks
        num_mar = len(marb1_sp) - len(np.argwhere(np.isnan(marb1_sp)))
        num_mar = num_mar + (len(marb2_sp) - len(np.argwhere(np.isnan(marb2_sp))))
    elif method == 'B':
        num_mar = number

    if num_mar < len(sha_sp) - len(np.argwhere(np.isnan(sha_sp))):
        # extract all indices where SHARAD surfpicks exist
        ind = np.argwhere(np.isfinite(sha_sp))
        ind = ind[:, 0]
        # define unique indices to be kept randomly from the surfpick index list
        ind2 = np.sort(numpy.random.choice(ind, (num_mar, ), replace=False))
        # create the output
        out = np.full((len(sha_sp), 1), np.nan)
        # populate the output with SHARAD surfpicks
        out[ind2, 0] = sha_sp[ind2, 0]
    else:
        out = sha_sp

    return out

def trc_align(data, rxwin, scalt, instrument, marsis_band):
    '''
    Relative trace alignment algorithm to properly align individual range lines
    within the range such that the surface can be accuately picked.

    Input:
    -----------
            data: array of complex-valued range lines expressed in the
                  slow-time/frequency domain.
                  - range lines organized by row
                  - frequency samples organized by column
           rxwin: vector of receive window opening times
                  - expressed in seconds for SHARAD
                  - expressed in samples for MARSIS
           scalt: vector of spacecraft altitudes [km]
      instrument: specific instrument used in data collection. required to set
                  specific details of the range compression
     marsis_band: identification flag for which MARSIS channel is being worked
                   - either '1' or '2'

    Output:
    -----------
        out: array of aligned range lines expressed in the slow-time/fast-time
             domain.
    '''

    import numpy as np

    # set parameters for each instrument
    if instrument == 'SHARAD':
        n = 3600
        dt = 0.0375E-6
    elif instrument == 'MARSIS':
        n = 512
        dt = 1 / 1.4E6

    # define the times by which the individual traces must be shifted
    if instrument == 'MARSIS' and marsis_band == '1':
        rx = rxwin * (1 / 2.8E6) - (2000 * (scalt - 25) / 299792458 )
    elif instrument == 'MARSIS' and marsis_band == '2':
        rx = rxwin * (1 / 2.8E6) - 450E-6 - (2000 * (scalt - 25) / 299792458) 
    elif instrument == 'SHARAD':
        rx = rxwin

    # define the frequency vector
    f = np.fft.fftfreq(n, dt)

    # define the output
    out = np.zeros((len(data), n), dtype=complex)

    # apply phase shift
    for jj in range(len(data)):
        temp_F1 = np.multiply(np.fft.fft(data[jj], norm='ortho'),
                              np.exp(-1j * 2 * np.pi * rx[jj] * f))
        out[jj] = np.fft.ifft(temp_F1, norm='ortho')

    return out

def iau2000_ellipsoid_radius(lat):
    '''
    Algorithm for determining martian IAU2000 ellipsoid radius at a particular
    latitude angle.

    Inputs:
    -------------
       latitude: latitude at which we want to extract the martian ellipsoid
                 radius [deg]

    Outputs:
    -------------
      output is the radius of the martian ellipsoid [m]
    '''

    import numpy as np

    # set the radii for Mars
    a = 3396.19E3        # equatorial martian ellipsoid radius [m]
    b = 3376.20E3        # polar martian ellipsoid radius
    if lat < 0:
        lat = np.abs(lat)

    # define latitude in both degrees and radians
    lat_rad = np.flipud(np.linspace(0, np.pi / 2, 90 * 1000))
    lat_deg = np.flipud(np.linspace(0, 90, 90 * 1000))

    # define the radius as a function of latitude (absolute value)
    tempA = np.square(b) * np.square(np.cos(lat_rad))
    tempB = np.square(a) * np.square(np.sin(lat_rad))
    rad = (a * b) / np.sqrt(tempA + tempB)

    # find the index closest to the desired latitude
    ind = np.argwhere(np.abs(lat_deg - lat) == np.min(np.abs(lat_deg - lat)))
    rad = rad[ind]

    return rad

def normalize_mola(radi, bounds, ddeg=64):
    '''
    Algorithm for normalizing MOLA radii datasets to the IAU2000 reference
    ellipsoid to create a topographic dataset. Algorithm cannot handle a mola
    array that encompasses portions of both the northern and southern
    hemisphere.

    Inputs:
    -------------
         radi: array of MOLA radius measurements [km]
       bounds: minimum and maximum latitude and longitude bounds of the area
               covered by the mola array [deg]
         ddeg: resolution of the MOLA dataset [deg]

    Outputs:
    -------------
      output is MOLA topography relative to the IAU2000 reference ellipsoid
             [km]
    '''

    import numpy as np

    # add back planetary radius to radi
    true_radi = radi + 3396E3

    # derive IAU2000 ellipsoid radii at each latitude position covered by MOLA
    if bounds[0] >= 0 and bounds[1] >= 0:
        mola_lat = np.flipud(np.arange(bounds[0], bounds[1], 1 / ddeg))
    else:
        mola_lat = -1 * np.arange(bounds[0], bounds[1], 1 / ddeg)
    iau = np.zeros((len(mola_lat), 1), dtype=float)
    for ii in range(len(iau)):
        iau[ii] = iau2000_ellipsoid_radius(mola_lat[ii])

    # remove IAU2000 ellipsoid from radii
    iau_topo = np.zeros((np.size(radi, axis=0), np.size(radi, axis=1)),
                        dtype=float)
    for ii in range(np.size(iau_topo, axis=1)):
        iau_topo[:, ii] = true_radi[:, ii] - iau[:, 0]

    return iau_topo

def mola2iautopo(dtm, lat_bounds):
    '''
    Algorithm for creating a topographic map relative to the IAU2000 martian
    reference ellipsoid from a MOLA martian surface radii dtm.

    Inputs:
    -------------
               dtm: MOLA martian surface radius dtm [km]
        lat_bounds: lower (most southern) and upper (most northen) latitudes of
                    the MOLA dtm [deg]

    Outputs:
    -------------
      output is a topographic dtm normalized to the IAU2000 martian ellipsoid
         [km]
    '''

    import numpy as np

    # define the latitude coordinates of the DTM grid
    lat = np.linspace(lat_bounds[0], lat_bounds[1], np.size(dtm, axis=0))

    # extract an IAU2000 ellipsoid radii for each latitude
    iau_rad = np.zeros((len(lat), 1), dtype=float)
    for ii in range(len(iau_rad)):
        iau_rad[ii, 0] = iau2000_ellipsoid_radius(lat[ii])

    # produce an ellipsoid map
    iau_map = np.tile(np.flipud(iau_rad), (1, np.size(dtm, axis=1)))

    # produce a difference map between the reference spheroid and the ellipsoid
    diff_map = 3396E3 - iau_map

    # create topographic map
    dtm = dtm + diff_map

    return dtm

def altimetry_error(scalt, topo, surf_pk, rx, TECU, instrument, mar_channel='1', mar_band='1'):
    '''
    Algorithm to compare the one-way distance to the Martian surface determined
    from the spacecraft altitude and MOLA to the one-way distance determined
    from the surface pick. This might shed some light on the size of the error
    bars we find in the TECU estimates.

    Inputs:
    -------------
           scalt: spacecraft altitude as read directly from the
                  auxiliary/geometry files associated with each instrument [km]
            topo: nadir topography (relative to IAU2000) for each range line
                  where a surface echo was picked [m]
         surf_pk: vector of surface picks
                  - expressed in samples for MARSIS
                  - expressed in time for SHARAD
              rx: vector of receive window opening times [s]
            TECU: optimal ionosphere TECU determined during the iterative
                  assessment
      instrument: string identifying which instrument is being analyzed
     mar_channel: if working with MARSIS data, need to know which channel it is
        mar_band: vector identifying the MARSIS centre frequency for each
                  range line

    Outputs:
    -------------
      output is a vector of the differences between the distance to the surface
      as defined from the spacecraft altitude and MOLA and the picked surface
      from the radargram
    '''

    import numpy as np

    # define the one-way distances to the surface based on spacecraft altitude
    # and MOLA
    if instrument == 'MARSIS':
        scalt = np.transpose(np.array(scalt, ndmin=2))
    geo_distance = scalt - (topo / 1000)

    # define the one-way distances to the surface based on surface pick
    pk_distance = np.zeros((len(scalt), 1), dtype=float)
    for ii in range(len(surf_pk)):
        if np.isnan(surf_pk[ii]) == False:
            # define effective time to the surface based on picked echo
            # TODO: GNG: make this a dict/list LUT
            if instrument == 'MARSIS':
                if mar_channel == '1':
                    Te_rx = (rx[ii] * (1 / 2.8E6))
                    if int(mar_band[ii]) == 0: f = 4.0E6
                    elif int(mar_band[ii]) == 1: f = 1.8E6
                    elif int(mar_band[ii]) == 2: f = 3.0E6
                    elif int(mar_band[ii]) == 3: f = 5.0E6
                elif mar_channel == '2':
                    Te_rx = ((rx[ii] - 450E-6 / (1 / 2.8E6)) * (1 / 2.8E6))
                    if int(mar_band[ii]) == 0: f = 1.8E6
                    elif int(mar_band[ii]) == 1: f = 4.0E6
                    elif int(mar_band[ii]) == 2: f = 3.0E6
                    elif int(mar_band[ii]) == 3: f = 5.0E6
                Te = Te_rx + surf_pk[ii] * (1 / 1.4E6)
            elif instrument == 'SHARAD':
                Te = rx[ii] + surf_pk[ii] * 0.0375E-6
                f = 20E6
            # correct for bulk delay due to ionosphere
            Te = Te - (1.69E-6 * TECU * 1E16 / (2 * np.pi * np.square(f)))
            # convert to one-way distance
            pk_distance[ii] = (299792458 * (Te / 2)) / 1000
        else:
            pk_distance[ii] = np.nan

    # calculate the difference in distances
    output = geo_distance - pk_distance

    return output
