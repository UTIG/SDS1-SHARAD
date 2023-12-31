#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge (JPL), Gregor.B.Steinbruegge@jpl.nasa.gov']

__version__ = '1.2'
__history__ = {
    '1.0':
        {'date': 'March 05, 2019',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'Initial Release.'},
    '1.1':
        {'date': 'September 16, 2020',
         'author': 'Gregor Steinbruegge, Stanford',
         'info': 'Added Zero-Doppler Filter'},
    '1.2':
        {'date': 'October 05, 2021',
         'author': 'Gregor Steinbruegge, JPL',
         'info': 'Checks for corrupted data & wrong range window start.'}}


# TODO:
# reorder imports
# send final copy back to gregor
import time
from math import tan, pi, erf, sqrt
import logging
import sys
import os
import numpy as np
from scipy.constants import c
import spiceypy as spice
import pandas as pd
from scipy.optimize import least_squares

# Imports from using pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc import prog as prg
from misc import coord as crd
import cmp.pds3lbl as pds3


def beta5_altimetry(cmp_path, science_path, label_science, label_aux,
                    use_spice=False, ft_avg=10,
                    max_slope=25, noise_scale=20, fix_pri=None, fine=True,
                    idx_start=0, idx_end=None):

    """
    Computes altimetry profile based on Steinbrügge et al. (in review)
    Method is the Beta5 model based on ocean altimetry.

    Input
    -----
    cmp_path: string
        Path to pulse compressed data. h5 file expected
    science_path: string
        Path to EDR science data
    label_path: string
        Path to science label of EDR data
    label_aux: string
        Path to auxillary label file
    idx_start (optional): integer
        Start index for processing track. Default is 0.
    idx_end (optional): integer
        End index for processing track. Default is None,
        which processes to the end of the track.
    use_spice (optional): boolean
        Specifies is spacecraft position is taken from the EDR
        data or if it is re-computed using a spice kernel.
        IMPORTANT NOTE: If use_spice is set to True then, a valid
                        kernel must be furnished!
    ft_avg(optional): integer
        Window for fast time averaging [TODO: units]. Set to None to turn off.
        Default is 10.
    max_slope (optional): double
        Maximum slope, in degrees, to be considered in coherent averaging.
        Default is 25 degrees.
    noise_scale (optional): double
        Scaling factor for the rms noise used within the threshold
        detection. Default is set to 20.
    fix_pri (optional): integer
        Pulse repetition interval code. If None, it will be taken from
        the input data, otherwise it can be set to a fixed code.
        Fixing it avoids reading the bitcolumns in the input, making
        the reading faster.
    fine (optional): boolean
        Apply fine detection with beta-5 model fit. Default is True.

    Output
    ------
    et: double
        Ephemeris time
    lat: double
        Latitude
    lon: double
        Longitude
    r: double
        Radius from CoM
    avg: double (array)
        Unfocussed SAR radargram (smoothed in fast time)

    """
    MB = 1024*1024
    time0 = time.time()
    #============================
    # Read and prepare input data
    #============================

    # Read input data
    dbc = fix_pri is None
    data = pds3.read_science(science_path, label_science, science=True, bc=dbc)
    logging.debug("Size of 'edr' data: {:0.2f} MB".format(sys.getsizeof(data)/MB))
    aux = pds3.read_science(science_path.replace('_s.dat', '_a.dat'),
                            label_aux, science=False, bc=False)
    logging.debug("Size of 'aux' data: {:0.2f} MB".format(sys.getsizeof(aux)/MB))
    re = pd.read_hdf(cmp_path, key='real').values[idx_start:idx_end]
    im = pd.read_hdf(cmp_path, key='imag').values[idx_start:idx_end]
    cmp_track = np.empty(re.size,dtype=np.complex64)
    cmp_track = re+1j*im
    logging.debug("Size of 'cmp' data: {:0.2f} MB".format(sys.getsizeof(cmp_track)/MB))
    tecu_filename = cmp_path.replace('.h5', '_TECU.txt')
    tecu = np.genfromtxt(tecu_filename)[idx_start:idx_end, 0]

    time1 = time.time()
    logging.debug("Read input elapsed time: {:0.2f} sec".format(time1-time0))
    time0 = time1

    # Get Range window start
    range_window_start = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]

    # Compute or read S/C position
    ets = aux['EPHEMERIS_TIME'].values[idx_start:idx_end]
    if use_spice is True:
        sc = np.empty(len(ets))
        lat = np.empty(len(ets))
        lon = np.empty(len(ets))
        for i in range(len(ets)):
            scpos, lt = spice.spkgeo(-74, ets[i], 'J2000', 4)
            llr = crd.cart2sph(scpos[0:3])[0]
            lat[i] = llr[0]
            lon[i] = llr[1]
            sc[i] = np.linalg.norm(scpos[0:3])
    else:
        sc_x = aux['X_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc_y = aux['Y_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc_z = aux['Z_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc = np.sqrt(sc_x**2+sc_y**2+sc_z**2)
        lon = aux['SUB_SC_EAST_LONGITUDE'].values[idx_start:idx_end]
        lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE'].values[idx_start:idx_end]

    # Code mapping PRI codes to actual pulse repetition intervals
    pri_table = {
        1: 1428E-6, 2: 1429E-6,
        3: 1290E-6, 4: 2856E-6,
        5: 2984E-6, 6: 2580E-6
    }

    pri_code = data['PULSE_REPETITION_INTERVAL'].values[idx_start:idx_end]

    pri = np.array([pri_table.get(x, 1428E-6) for x in pri_code])

    del data

    time1 = time.time()
    logging.debug("Spice Geometry calculations: {:0.2f} sec".format(time1-time0))
    time0 = time1

    # Check for too small range window start values   
    idx_valid = np.where(range_window_start>1000)
    
    # Identify corrupted data
    corrupted_flag = aux['CORRUPTED_DATA_FLAG'].values[idx_start:idx_end]
    corrupted_idx = np.where(corrupted_flag == 1)[0]

    # Calculate offsets for radargram
    sc_cor = np.array(2000*sc/c/0.0375E-6).astype(int)
    phase = -sc_cor+range_window_start
    tx0 = int(min(phase[idx_valid]))
    shift_param = (phase-tx0) % 3600


    time1 = time.time()
    logging.debug("radargram offsets get shot freq: {:0.2f} sec".format(time1-time0))
    time0 = time1

    # Compute SAR apertures for coherent and incoherent stacking
    # Note the window for coherent stacking is by default set to preserve slopes
    # up to 25deg. For SHARAD data this means that generally no incoherent stacking
    # is performed, i.e. coh_window = 1. However, with variable observing geometries
    # during Europa flybys, REASON data might allow for coherent stacking.
    sc_alt = aux['SPACECRAFT_ALTITUDE'].values[idx_start:idx_end]*1000
    vel_t = aux['MARS_SC_TANGENTIAL_VELOCITY'].values[idx_start:idx_end]*1000
    # catch zero-velocities
    idxn0 = np.where(abs(vel_t)>0)
    fresnel = np.sqrt(sc_alt[idxn0]*c/10E6+(c/(20E6))**2)
    sar_window = int(np.mean(2*fresnel/vel_t[idxn0]/pri[idxn0])/2)
    coh_window = int(np.mean((c/20E6/4/tan(max_slope*pi/180)/vel_t[idxn0]/pri[idxn0])))

    time1 = time.time()
    logging.debug("Compute SAR apertures: {:0.2f} sec".format(time1-time0))
    time0 = time1

    del aux

    #========================
    # Actual pulse processing
    #========================

    # Zero Doppler Filter
    dp_wdw = 30
    doppler = np.copy(cmp_track)
    for j in range(dp_wdw,len(cmp_track)-dp_wdw):
        doppler_r = np.fft.fft(cmp_track[j-dp_wdw:j+dp_wdw],axis=0)
        doppler[j] = doppler_r[0]

    # Perform smoothing of the waveform aka averaging in fast time
    if ft_avg is not None:
        wvfrm = np.empty((len(doppler), 3600), dtype=np.complex)
        for i in range(len(doppler)):
            rmean = running_mean(doppler[i], ft_avg)
            wvfrm[i] = rmean - np.mean(rmean)
            #wvfrm[i] = running_mean(np.abs(cmp_track[i]),10)
            #wvfrm[i] -= np.mean(wvfrm[i])
            wvfrm[i, :10] = wvfrm[i, 20:30]
            wvfrm[i, -10:] = wvfrm[i, -30:-20]
    else:
        wvfrm = cmp_track

    del cmp_track
    time1 = time.time()
    logging.debug("Waveform smoothing: {:0.2f} sec".format(time1-time0))
    time0 = time1

    # Construct radargram
    radargram = np.zeros((len(wvfrm), 3600), dtype=np.complex)
    for rec in range(len(wvfrm)):
        radargram[rec] = np.roll(wvfrm[rec],int(phase[rec]-tx0))
        #radargram[rec] = np.roll(wvfrm[rec], int(phase[rec + idx_start] - tx0))

    logging.debug("Size of radargram: {:0.2f} MB".format(sys.getsizeof(radargram)/MB))
    del wvfrm
    
    # Perform averaging in slow time. Pulse is averaged over the 1/4 the
    # distance of the first pulse-limited footprint and according to max
    # slope specification.
    max_window = max(coh_window, sar_window)
    radargram_ext = np.empty((len(radargram) + 2 * max_window, 3600),
                             dtype=np.complex)

    # For zero Doppler no coherent averaging
    for i in range(3600):
        radargram_ext[:, i] = np.pad(radargram[:,i],
                                     (max_window, max_window), 'edge')

    avg = np.empty((len(radargram), 3600))
    if sar_window > 1:
        for i in range(3600):
            avg[:, i] = abs(running_mean(abs(radargram_ext[:, i]), sar_window))[max_window:-max_window]
    else:
        avg = abs(radargram_ext)

    logging.debug("Size of 'avg' data: {:0.2f} MB".format(sys.getsizeof(avg)/MB))
    del radargram_ext

    """ 
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(12,4))
    plt.imshow(np.transpose(20*np.log10(abs(avg[:,0:1500])+1e-4)),aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('[dB]')
    plt.show()
    """
    """ 
    import cmp.plotting
    print('control')
    rx_window_start = data['RECEIVE_WINDOW_OPENING_TIME']
    tx0=data['RECEIVE_WINDOW_OPENING_TIME'][0]
    tx=np.empty(len(data))
    for rec in range(len(data)):
        tx[rec]=data['RECEIVE_WINDOW_OPENING_TIME'][rec]-tx0
    cmp.plotting.plot_radargram(10*np.log10(np.abs(avg)**2),tx,samples=3600)
    """

    # Coarse detection
    coarse = np.zeros(len(avg), dtype=int)
    deriv = np.zeros((len(avg), 3599))

    for i in range(len(avg)):
        deriv[i] = np.abs(np.diff(avg[i]))

    # TODO: does this yield to parallel prefix sum or vectorizing?
    noise = np.sqrt(np.var(deriv[:, -1000:], axis=1))*noise_scale
    for i in range(len(deriv)):
    
        if i in corrupted_idx:
            coarse[i] = 0
            continue
        #found = False
        j0 = int(shift_param[i] + 20)
        j1 = int(min(shift_param[i] + 1020, len(deriv[i])))
        # Find the noise level to set the threshold to, and set the starting
        # lvl to that, to accelerate the search.
        # Round up to the next highest level
        if not np.isnan(noise[i]) and noise[i] > 0:
            lvlstart = np.ceil(np.max(deriv[i]) / noise[i] * 10.0) / 10.0
            lvlstart = min(max(lvlstart, 0.1), 1.0)
        else:
            lvlstart = 1.0
        #lvlstart = 1.0
        #logging.debug("lvlstart={:f}".format(lvlstart))

        for lvl in np.arange(lvlstart, 0, -0.1):

            noise_threshold = noise[i]*lvl
            """
            for j in range(j0, j1):
                if deriv[i, j] > noise_threshold:
                    coarse[i] = j
                    found = True
                    break
            if found == True:
                break
            """
            idxes = np.argwhere(deriv[i][j0:j1] > noise_threshold)
            if len(idxes) > 0:
                coarse[i] = idxes[0] + j0
                #found = True
                break
 
        time1 = time.time()
    logging.debug("Coarse detection: {:0.2f} sec".format(time1-time0))
    time0 = time1

    del deriv
    # Perform least-squares fit of waveform according to beta-5 re-tracking
    # model
    delta = np.zeros(len(avg))
    snr = np.zeros(len(avg))
    if fine is True:
        b3 = 100
        b4 = 2
        b5 = 3E-2
        #p = prg.Prog(len(avg))
        for i in range(len(avg)):
            #p.print_Prog(i)
            
            if i in corrupted_idx:
                delta[i] = 0
                snr[i]
                continue
        
            wdw = 10*np.log10(avg[i, max(coarse[i]-100, 0):min(coarse[i]+100, len(avg[i]))])
            b1 = np.mean(wdw[0:128])
            b2 = max(wdw)-b1
            res = least_squares(model5, [b1, b2, b3, b4, b5], args=wdw,
                                bounds=([-500, 0, 0, 0, 0],
                                        [np.inf, np.inf, np.inf, np.inf, 1]))
            delta[i] = res.x[2]+max(coarse[i]-100, 0)
            snr[i] = res.x[1]
    else:
        delta = coarse

    # Ionospheric delay
    disp = 1.69E-6/20E6*tecu*1E+16
    dt_ion = disp/(2*np.pi*20E6)
    # Time-of-Flight (ToF)
    tx = (range_window_start+delta-phase+tx0)*0.0375E-6+pri-11.98E-6-dt_ion
    # One-way range in km
    d = tx*c/2000

    # Elevation from Mars CoM
    r = sc-d

    # GNG TODO: Does pandas want this to be a np array or could it be a list of lists?
    # Seems like it just wants to be a python list.
    # https://www.geeksforgeeks.org/python-pandas-dataframe/#Basics

    spots = np.empty((len(r), 8))
    columns = ['et', 'spot_lat', 'spot_lon', 'spot_radius', \
               'idx_coarse', 'idx_fine', 'range', 'snr']
        
    for i in range(len(r)):
        if i in corrupted_idx:
            spots[i, :] = [ets[i], lat[i], lon[i], -1, -1, -1, -1, -1]
        else:
            spots[i, :] = [ets[i], lat[i], lon[i], r[i], \
                       (coarse-shift_param)[i], (delta-shift_param)[i], r[i], snr[i]]

    df = pd.DataFrame(spots, columns=columns)
    time1 = time.time()
    logging.debug("LSQ, Frame Conversion, DataFrame: {:0.2f} sec".format(time1-time0))
    time0 = time1

    return df

def running_mean(x, N):
    res = np.zeros(len(x), dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0), dtype=x.dtype)
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

def model5(beta, *wvfrm):
    t = np.arange(len(wvfrm))
    erf_v = np.vectorize(erf)
    Q = t - (beta[2] + 0.5*beta[3])
    Q[np.where(t < (beta[2] - 2 * beta[3]))] = 0
    y = beta[0] + \
        beta[1] * np.exp(-beta[4]*Q) \
                * 0.5 * (1 + erf_v((t - beta[2])/(sqrt(2) * beta[3])))
    return y - wvfrm

