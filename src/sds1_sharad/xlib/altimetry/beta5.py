#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge (JPL), Gregor.B.Steinbruegge@jpl.nasa.gov']

__version__ = '1.3'
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
         'info': 'Checks for corrupted data & wrong range window start.'},
    '1.3':
        {'date': 'February 8, 2023',
         'author': 'Gregory Ng, UTIG',
         'info': 'Rewrite in functional iterator style that operates per-trace and improves memory efficiency'},
}

# TODO:
# reorder imports
# send final copy back to gregor
import time
from math import tan, pi, erf, sqrt
import logging
import sys
import os
import itertools
import multiprocessing
import tempfile
from pathlib import Path
import json

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





def beta5_altimetry(cmp_path: str, science_path: str, label_science: str, label_aux: str,
                    use_spice=False, ft_avg: int=10,
                    max_slope:float=25, noise_scale:float=20, fix_pri=None,
                    finethreads:int=1,
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
    finethreads: int
        Apply fine detection with beta-5 model fit, and how many parallel processes
        0 = Don't do fine detection (previously fine=False)
        1 = do fine detection without multiprocessing
        2+ use multiprocessing

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
    #dbc = fix_pri is None
    scireader = pds3.SHARADDataReader(label_science, science_path)
    data = scireader.arr
    bit_data = scireader.get_bitcolumns()
    logging.debug("Size of 'edr' data: %0.2f MB, dimensions %r", sys.getsizeof(data)/MB, data.shape)
    auxdatapath = science_path.replace('_s.dat', '_a.dat')
    aux = pds3.read_science(auxdatapath, label_aux)
    logging.debug("Size of 'aux' data: %0.2f MB, dimensions %r", sys.getsizeof(aux)/MB, aux.shape)

    p1 = Path(cmp_path)
    tecu_filename = Path(cmp_path).with_stem(p1.stem + '_TECU').with_suffix('.txt')
    tecu = np.genfromtxt(tecu_filename)[idx_start:idx_end, 0]

    time1 = time.time()
    logging.debug("Read input elapsed time: %0.2f sec", time1-time0)
    time0 = time1

    # Get Range window start
    range_window_start = data['RECEIVE_WINDOW_OPENING_TIME'][idx_start:idx_end]

    # Compute or read S/C position
    ets = aux['EPHEMERIS_TIME'][idx_start:idx_end]
    if use_spice:
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
        sc_x = aux['X_MARS_SC_POSITION_VECTOR'][idx_start:idx_end]
        sc_y = aux['Y_MARS_SC_POSITION_VECTOR'][idx_start:idx_end]
        sc_z = aux['Z_MARS_SC_POSITION_VECTOR'][idx_start:idx_end]
        sc = np.sqrt(sc_x**2+sc_y**2+sc_z**2)
        lon = aux['SUB_SC_EAST_LONGITUDE'][idx_start:idx_end]
        lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE'][idx_start:idx_end]

    # Code mapping PRI codes to actual pulse repetition intervals
    pri_table = {
        1: 1428E-6, 2: 1429E-6,
        3: 1290E-6, 4: 2856E-6,
        5: 2984E-6, 6: 2580E-6
    }

    pri_code = bit_data['PULSE_REPETITION_INTERVAL'][idx_start:idx_end]

    pri = np.array([pri_table.get(x, 1428E-6) for x in pri_code])

    del data

    time1 = time.time()
    logging.debug("Spice Geometry calculations: %0.2f sec", time1-time0)
    time0 = time1

    # Check for too small range window start values   
    idx_valid = np.where(range_window_start>1000)
    
    # Identify corrupted data
    corrupted_flag = aux['CORRUPTED_DATA_FLAG'][idx_start:idx_end]
    corrupted_idx = np.where(corrupted_flag == 1)[0]

    # Calculate offsets for radargram
    sc_cor = np.array(2000*sc/c/0.0375E-6).astype(int)
    phase = -sc_cor+range_window_start
    tx0 = int(min(phase[idx_valid]))
    shift_param = (phase-tx0) % 3600


    time1 = time.time()
    logging.debug("radargram offsets get shot freq: %0.2f sec", time1-time0)
    time0 = time1

    # Compute SAR apertures for coherent and incoherent stacking
    # Note the window for coherent stacking is by default set to preserve slopes
    # up to 25deg. For SHARAD data this means that generally no incoherent stacking
    # is performed, i.e. coh_window = 1. However, with variable observing geometries
    # during Europa flybys, REASON data might allow for coherent stacking.
    sc_alt = aux['SPACECRAFT_ALTITUDE'][idx_start:idx_end]*1000
    vel_t = aux['MARS_SC_TANGENTIAL_VELOCITY'][idx_start:idx_end]*1000
    iiend = aux.shape[0] if idx_end is None else idx_end
    del aux
    # catch zero-velocities
    idxn0 = np.where(abs(vel_t)>0)
    fresnel = np.sqrt(sc_alt[idxn0]*c/10E6+(c/(20E6))**2)
    sar_window = int(np.mean(2*fresnel/vel_t[idxn0]/pri[idxn0])/2)
    coh_window = int(np.mean((c/20E6/4/tan(max_slope*pi/180)/vel_t[idxn0]/pri[idxn0])))
    logging.debug("sar_window=%d coh_window=%d ntraces=%d",
                  sar_window, coh_window, iiend)

    # For figuring out how much padding we'll need
    max_window = max(coh_window, sar_window)


    #time1 = time.time()
    #logging.debug("Compute SAR apertures: %0.2f sec", time1-time0)
    #time0 = time1


    #========================
    # Actual pulse processing
    #========================


    logging.debug("cmp_path=%s", cmp_path)
    read_gen = gen_hdf_complex_traces(cmp_path, idx_start, idx_end)


    if ft_avg is None:
        wvfrm_gen = trace_gen(read_gen)
    else:
        wvfrm_gen = zero_doppler_filter(read_gen, ft_avg, ntraces=iiend)

    phaseroll_gen = roll_radar_phase(wvfrm_gen, phase, tx0)
    #phaseroll_gen = map(lambda arr: arr.astype(np.complex64), phaseroll_gen0)


    # Pulse is averaged over the 1/4 the
    # distance of the first pulse-limited footprint and according to max
    # slope specification.
    # Slow time averaging could use some padding


    gen_sta = slow_time_averaging_gen(phaseroll_gen, coh_window, sar_window, iiend)
    #gen_sta = map(lambda arr: arr.astype(np.float64), gen_sta0)

    coarse_gen = coarse_detection_gen(gen_sta, noise_scale, shift_param, corrupted_idx)

    #time1 = time.time()
    #logging.debug("Coarse detection: %0.2f sec", time1-time0)
    #time0 = time1
    if finethreads > 0:
        coarse, delta, snr = fine_tracking(coarse_gen, ntraces=iiend, finethreads=finethreads)
    else:
        coarse = np.empty((iiend,), dtype=int)
        for ii, _, x in coarse_gen:
            if x is None:
                coarse[ii] = 0
            else:
                coarse[ii] = x

        delta, snr = coarse, np.zeros((iiend,))

    time1 = time.time()
    logging.debug("Pulse processing: %0.2f sec", time1-time0)
    time0 = time1

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
    #np.save('beta5.npy', spots) # for debugging
    time1 = time.time()
    logging.debug("LSQ, Frame Conversion, DataFrame: %0.2f sec", time1-time0)
    time0 = time1

    return df

def gen_hdf_complex_traces(cmp_path, idx_start, idx_end):
    """ return an iterator that generates complex traces from HDF5 file
    """
    # TODO: can we make iterator==True to reduce memory? (doesn't work as advertised)
    kwargs = {'path_or_buf': cmp_path, 'iterator': False}

    if cmp_path.endswith('.i'):
        json_path = Path(cmp_path).with_suffix('.json')
        with json_path.open('rt') as fhjson:
            jinfo = json.load(fhjson)
        shape1 = tuple(jinfo['shape'])
        assert len(shape1) == 3, "Expecting a 3D integer radargram (shape=%r)" % (shape1,)
        buf = np.memmap(cmp_path, mode='r', dtype=np.int16, shape=shape1)
        for trace in buf:
            t = np.empty((shape1[1],), dtype=np.complex64)
            t.real = trace[:, 0]
            t.imag = trace[:, 1]
            yield t
            #yield trace[:, 0] + 1j*trace[:, 1]
    elif cmp_path.endswith('.h5'):
        re_iter = pd.read_hdf(key='real', **kwargs)
        im_iter = pd.read_hdf(key='imag', **kwargs)
        for re, im in zip(re_iter.values[idx_start:idx_end], im_iter.values[idx_start:idx_end]):
            yield (re + 1j*im) #.astype(np.complex64)

def trace_gen(radargram: np.ndarray):
    for trace in radargram:
        yield trace


def gen_doppler_trace_buffered(gen_radargram, dp_wdw: int, ntraces: int):
    """ calculate the zero doppler for the given trace in the radargram
    and return them.
    We maintain a circular buffer in order to do the FFTs
    """
    if ntraces == 0:
        return
    assert ntraces > 2*dp_wdw, "Hasn't been tested with short transects!"
    buffer = None
    bidx = np.zeros((2*dp_wdw,), dtype=int)
    tracenum0 = -1
    #assert dp_wdw == len(buffer[dp_wdw:])
    for ii, trace in enumerate(gen_radargram):
        if buffer is None:
            nsamples = len(trace)
            buffer = np.empty((2*dp_wdw, nsamples), dtype=complex)

        # Shift data in buffer
        buffer[0:-1, :] = buffer[1:, :]
        buffer[-1] = trace
        bidx[0:-1] = bidx[1:]
        bidx[-1] = ii
        if ii < dp_wdw:
            tracenum = ii
            yield tracenum, trace # return unmodified trace
            tracenum0 = tracenum
        elif dp_wdw <= ii < 2*dp_wdw - 1:
            pass # keep filling buffer but don't yield anything
            #assert ii+1 < 2*dp_wdw
        elif ii < ntraces-1: # condition emulates behavior in calc_doppler_trace
            #assert ii+1 >= 2*dp_wdw

            # If it's in the middle, do an fft and get the zero doppler bin
            doppler_r = np.fft.fft(buffer, axis=0)
            tracenum = ii - dp_wdw + 1
            #assert tracenum < (ntraces - dp_wdw)
            #assert tracenum0 + 1 == tracenum

            yield tracenum, doppler_r[0]
            tracenum0 = tracenum

    # Return unmodified traces at the end
    assert tracenum0+1 == ntraces - dp_wdw, "tracenum0=%d ntraces=%d dp_wdw=%d" % (tracenum0, ntraces, dp_wdw)
    assert dp_wdw == len(buffer[dp_wdw:])
    for tracenum, trace in enumerate(buffer[dp_wdw:], start=ntraces - dp_wdw):
        #assert tracenum >= (ntraces - dp_wdw)
        yield tracenum, trace

def zero_doppler_filter(gen_radargram, ft_avg: int, ntraces: int):
    """ Zero Doppler Filter
    TODO: save the doppler array to a memmapped array to a temp directory so that
    it can be paged out

    do the roll in the same operation
    """
    assert ft_avg > 0
    dp_wdw = 30

    buffer = None

    # Perform smoothing of the waveform aka averaging in fast time
    #wvfrm = np.empty(cmp_track.shape, dtype=np.complex64)
    #for i, trace in enumerate(radargram):
    gen_doppler = gen_doppler_trace_buffered(gen_radargram, dp_wdw, ntraces)
    for i, (jj, doppler_r) in enumerate(gen_doppler):
        if buffer is None:
            nsamples = len(doppler_r)
            buffer = np.empty((2*dp_wdw, nsamples), dtype=complex)

        rmean = running_mean(doppler_r, ft_avg)
        trace = rmean - np.mean(rmean)
        #wvfrm[i] = running_mean(np.abs(cmp_track[i]),10)
        #wvfrm[i] -= np.mean(wvfrm[i])
        trace[:10] = trace[20:30]
        trace[-10:] = trace[-30:-20]
        yield trace
        #wvfrm[i] = trace


def roll_radar_phase(wvfrm_gen, phase: np.array, tx0: int):
    # for rec in range(len(wvfrm)):
    for trace, trace_phase in zip(wvfrm_gen, phase):
        shiftn = int(trace_phase - tx0)
        yield np.roll(trace, shiftn)




def pad_radargram(gen_radargram, pad_pre, pad_post, mode='edge'):
    """ Pad radargram by duplicating the edge trace
    To be equivalent to np.pad(radargram, padding, 'edge')
    """
    trace = None
    for ii, trace in enumerate(gen_radargram):
        if ii == 0:
            # emit padding
            for i0 in range(pad_pre):
                if mode == 'zeros':
                    yield np.zeros_like(trace)
                elif mode == 'edge':
                    yield trace
        yield trace

    if trace is not None:
        for i0 in range(pad_post):
            if mode == 'zeros':
                yield np.zeros_like(trace)
            elif mode == 'edge':
                yield trace

def cumsum_gen(traces_gen):
    """ Compute and return a cumulative sum generator, inserting """
    accum = None
    for ii, trace in enumerate(traces_gen):
        if accum is None:
            accum = np.zeros_like(trace)

        accum = accum + trace
        yield accum

def running_mean_gen(radargram, N: int):
    """ Generator-style running mean, using cumulative sum
    TODO: compare non-cumsum accuracy 

    # Implements generator-based version of the following:
    res = np.zeros(len(x), dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0), dtype=x.dtype)
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

    """
    buffer = [] # Circular buffer of output values
    csumbuffer = [] # Circular buffer of cumulative sums

    n2a, n2b = (N // 2), (N - ((N // 2)+1))
    gen1 = cumsum_gen(radargram)
    z0, sumtrace = None, None
    for ii, sumtrace in enumerate(gen1):
        if ii == 0:
            z0 = np.zeros_like(sumtrace)
            # yield an extra zero for np.insert(x, 0, 0)
            csumbuffer.append(z0)

        if ii < n2a: # yield for indices from res[0:N//2]
            yield z0

        csumbuffer.append(sumtrace)

        if len(csumbuffer) > N: # really  len(csumbuffer) == N
            res = (csumbuffer[-1] - csumbuffer.pop(0)) / N
            yield res

    if sumtrace is not None:
        for _ in range(N//2):
            yield z0 # np.zeros_like(sumtrace)


def skip_traces(gen_radargram, skip_pre: int, skip_post: int):
    assert skip_pre >= 0 and skip_post >= 0
    buffer = []
    for ii, trace in enumerate(gen_radargram):
        if ii < skip_pre:
            continue
        # Return the last item if we know it's not within the last skip_post traces
        buffer.append(trace)
        if len(buffer) > skip_post:
            yield buffer.pop(0) # dequeue first item and return the trace


def slow_time_averaging_gen(radargram, coh_window: int, sar_window: int, ntraces: int):
    # Perform averaging in slow time.
    max_window = max(coh_window, sar_window)

    if sar_window > 1:
        gen_radargram_ext = pad_radargram(radargram, max_window, max_window, 'edge')
        gen_abs = map(np.abs, gen_radargram_ext)
        gen_sum = running_mean_gen(gen_abs, sar_window)
        gen4 = skip_traces(gen_sum, max_window, max_window)

        for trace in gen4:
            yield trace
    else:
        for ii, trace in enumerate(radargram):
            yield np.abs(trace)




def coarse_detection_gen(avg: np.ndarray, noise_scale: float, shift_param, corrupted_idx):
    """ A streamed version of coarse detection that returns the location of the detection
    as well as the raw trace 
    In the beginning it will buffer the result
    """

    for i, (trace, shift_param0) in enumerate(zip(avg, shift_param)):
        yielded = False
        if i in corrupted_idx:
            yield i, trace, None #coarse0
            continue

        deriv0 = np.abs(np.diff(trace))
        noise0 = np.sqrt(np.var(deriv0[-1000:])) * noise_scale

        j0 = int(shift_param0 + 20)
        j1 = int(min(shift_param0 + 1020, len(deriv0)))

        # Find the noise level to set the threshold to, and set the starting
        # lvl to that, to accelerate the search.
        # Round up to the next highest level
        if not np.isnan(noise0) and noise0 > 0:
            lvlstart = np.ceil(np.max(deriv0) / noise0 * 10.0) / 10.0
            lvlstart = min(max(lvlstart, 0.1), 1.0)
        else:
            lvlstart = 1.0

        for lvl in np.arange(lvlstart, 0, -0.1):
            noise_threshold = noise0*lvl
            idxes = np.argwhere(deriv0[j0:j1] > noise_threshold)
            if len(idxes) > 0:
                coarse0 = idxes[0][0] + j0
                yield i, trace, coarse0
                yielded = True
                break

        if not yielded:
            yield i, trace, None

def fine_tracking(coarse_gen, ntraces: int, finethreads:int=1):
    """ Perform least-squares fit of waveform according to beta-5 re-tracking model
    avg: ntraces x nsamples radargram
    coarse: ntraces x 1 integer array of coarse surface detection
    corrupted_idx: list of corrupted trace indexes. Skip these traces
    """
    coarse = np.zeros(ntraces, dtype=int)
    delta = np.zeros(ntraces)
    snr = np.zeros(ntraces)

    paramgen = fine_tracking_params(coarse_gen)
    time0 = time.time()
    paramgen = list(paramgen)
    time1 = time.time()
    logging.debug("Fine tracking part 1 time: %0.2f seconds", time1 - time0)
    time0 = time1

    if finethreads <= 1:
        for i, (coarse0, delta0, snr0) in enumerate(itertools.starmap(fine_tracking_trace, paramgen)):
            if delta0 is None and snr0 is None:
                continue
            coarse[i], delta[i], snr[i] = coarse0, delta0, snr0
    else:
        with multiprocessing.Pool(processes=finethreads) as pool:
            for i, (coarse0, delta0, snr0) in enumerate(pool.starmap(fine_tracking_trace, paramgen)):
                if delta0 is None and snr0 is None:
                    continue
                coarse[i], delta[i], snr[i] = coarse0, delta0, snr0

    time1 = time.time()
    logging.debug("Fine tracking part 2 time: %0.2f seconds", time1 - time0)
    time0 = time1

    return coarse, delta, snr

def fine_tracking_params(coarse_gen):
    b3 = 100
    b4 = 2
    b5 = 3E-2

    for i, trace, coarse0 in coarse_gen:
        if coarse0 is None:# or i in corrupted_idx:
            yield coarse0, None, None, None, None, None
            continue

        idx0 = max(coarse0-100, 0)
        idx1 = min(coarse0+100, len(trace))
        window = trace[idx0:idx1]
        # We should be able to run this through multiprocessing and we don't have to pass
        # the entire radargram through
        yield coarse0, window, idx0, b3, b4, b5



def fine_tracking_trace(coarse0, window_samples: np.ndarray, idx0: int, b3: float, b4: float, b5: float):
    """ Fit the altimetry model to the relevant window in the trace """
    if window_samples is None:
        return coarse0, None, None

    wdw = 10*np.log10(window_samples)
    b1 = np.mean(wdw[0:128])
    b2 = max(wdw)-b1
    res = least_squares(model5, [b1, b2, b3, b4, b5], args=wdw,
                        bounds=([-500, 0, 0, 0, 0],
                                [np.inf, np.inf, np.inf, np.inf, 1]))

    delta = res.x[2] + idx0
    snr = res.x[1]
    return coarse0, delta, snr

def running_mean(x, N: int):
    #logging.debug("Running_mean(x.shape=%r, N=%d)", x.shape, N)
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

