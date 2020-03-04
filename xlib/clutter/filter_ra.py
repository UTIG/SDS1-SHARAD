#!/usr/bin/env python

"""
To run self-test, use this command:
./filter_ra.py --selftest 1 1 1 1 1 1 1

This module implements a step that takes a raw HiCARS/MARFA bxds,
which is evenly-sampled in time, along with geometry metadata (positions of samples),
and outputs a data product interpolated to evenly-sampled in space (distance).

Along-track filtering is implemented prior to resampling by distance, which is
technically incorrect, but for most data to date, this is well-tolerated and
helps to reduce aliasing and noise amplification due to noisy input data.

Radar records are processed in blocks to limit memory usage, and blocks are
overlapped when read and filtered, to reduce edge effects.

2020-02-14: 
GNG modified this script to rearrange how tears are processed.
Previously, tears were handled in either two or four sections, depending on how 
you look at it, but this entire code was moved to a function called fill_gap_gen.

"""

# TODO: make compatible with python3
import argparse
import sys
from collections import namedtuple
import math
import logging
import os

import numpy as np
from scipy.signal import detrend
import pyfftw
pyfftw.interfaces.cache.enable()
#import pcheck

WAIS=os.getenv('WAIS', '/disk/kea/WAIS')

import unfoc_KMS2 as unfoc

# Issue a warning if we attempt to create a gap of length 0 
WARN_GAP0 = False

def quad3(X,X1,X2,X3,P1,P2,P3):
    """ Quadratic interpolation with three input points 
    Lagrange form """
    XX1 = X-X1
    XX2 = X-X2
    XX3 = X-X3
    X1X2 = X1-X2
    X2X3 = X2-X3
    X3X1 = X3-X1
    A = - (XX2*XX3)/(X1X2*X3X1)
    B = - (XX1*XX3)/(X1X2*X2X3)
    C = - (XX1*XX2)/(X3X1*X2X3)
    return A*P1 + B*P2 + C*P3


def main():
    parser = argparse.ArgumentParser(description='2D filtering of data in preparation for focusing.')

    parser.add_argument('DX', help='Along-track resampling interval', type=float)
    parser.add_argument('MS', help='Number of samples per record (e.g., 3200)', type=int)
    parser.add_argument('NR', help='Number of records output per along-track filtering block', type=int)
    parser.add_argument('NRr', help='Number of records in block overlap section needed '
                        'to ensure smoothness between blocks.  Blocks overlap by (NRr//2) '
                        'records at beginning and end.  (NRr = 100 appears to work pretty '
                        'well.) (Should be even number, otherwise it is forced even.)', type=int)
    parser.add_argument('InName', help='Filename of data input file', type=str)
    parser.add_argument('OutName', help='Filename of filtered/resampled output file', type=str)
    parser.add_argument('--geopath', help='path to geo files', default='.', type=str)
    parser.add_argument('channel', help='The channel number to produce', type=int)
    parser.add_argument('--undersamp', help='let the processor know if data was undersampled', action='store_true')
    parser.add_argument('--combined', help='let the processor know if channels need to be combined', action='store_true')
    parser.add_argument('--selftest', help='Run self-test routines', action='store_true')
    parser.add_argument('--nofilter2d', help='Disable 2d filtering', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
    args = parser.parse_args()

    # TODO: use args vars directly
    DX = args.DX
    MS = args.MS
    NR = args.NR
    NRr = args.NRr
    InName = args.InName
    OutName = args.OutName
    channel = args.channel
    undersamp = args.undersamp
    combined = args.combined
    
    fmt = "%(levelname)-8s: %(message)s"
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout, format=fmt)
    
    if args.selftest:
        sys.exit(test())
        

    # TODO: is_marfa() function with set as input
    # Test to see if this system is MARFA
    #MARFA=['JKB2j','MKB2l','JKB2m','JKB2n','JKB2o','JKB2r','JKB2t']
    InNameArray = InName.split("/")
    #Platform=InNameArray[1]
    #MARFA_CHECK=Platform in MARFA

    if (args.combined):
        logging.info("filter: MARFA")


    # Load files:
    # Nc = indexes of original points closest to new resampling points
    # Xo = original along-track distances (First distance must be zero.)
    # NRt = record numbers of the data tears.  An effective data tear
    #       occurs at the very last record.
    geo_path = args.geopath

    with open(OutName, "wb") as OFD:
        for signalim in filter_ra_gen(InName, geo_path, DX, MS, NR, NRr, channel, 
                                  undersamp=undersamp, combined=combined, filter2d=(not args.nofilter2d)):
            np.real(signalim).astype('<i2').T.tofile(OFD)

def read_stackgen_block(stackgen, MS, NumRead, NB, filename="?", short_read_severity=logging.INFO):
    block = np.empty((MS, NumRead))
    for i in range(NumRead):
        try:
            trace = next(stackgen)
        except StopIteration:
            # It's fine to have a StopIteration here,
            # because we won't always have an even number of stacks.
            # TODO: check that we have fully-formed traces.
            msg = "Short read (stackgen NB={:d} i={:d} bytes={:d} of {:d})".format(NB, i, MS, MS)
            logging.log(short_read_severity, msg)
            break #raise StopIteration(msg)
        block[:, i] = trace.data[0:MS]
    return block
    
def read_file_block(IFD, MS, NumRead, NB, filename="?", short_read_severity=logging.WARNING):
    block = np.empty((MS, NumRead))
    for i in range(NumRead):
        data = np.fromfile(IFD, "<i2", MS)
        if (data.size < MS):
            msg = "Short read (filegen NB={:d} i={:d} bytes={:d} of {:d})".format(NB, i, data.size, MS)
            logging.log(short_read_severity, msg)
            raise StopIteration(msg)
        block[:,i] = data
    return block

def filter_ra_length(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm=None, debug=False):
    """ Calculate the length of output data in records, from the metadata Nc.
    """
    len_output = len(np.fromfile(os.path.join(geo_path, "Nc"), dtype=int, sep=" "))

    if debug:
        idx = 0
        logging.debug("Start calculating length for " + geo_path)
        for signalim in filter_ra_gen(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm,
                                  filter2d=False, resample=False):
            assert len(signalim.shape) >= 2
            idx += signalim.shape[1]
        logging.debug("Finished calculating length for " + geo_path)

        if len_output != idx:
            logging.warning("Total number of records output by filter_ra does not match "
                            " expected: expected={:d} actual={:d}".format(len_output, idx))
            logging.warning("filter_ra_length input file: " + geo_path)
        return idx
    else:
        return len_output

def filter_ra(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm=None,
              undersamp=False, combined=False, filter2d=True, resample=True, trim=None):
    """ Generate a filtered data file and output to a numpy array
    See filter_ra_gen for parameter information.
    """

    if trim is not None and trim[1] is not None:
        MS1 = trim[1] - trim[0]
    else:
        MS1 = MS
        trim = (0, MS)

    len_output = filter_ra_length(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm=snm)
    signalout = np.empty((MS1, len_output), dtype=complex)

    idx = 0
    for signalim in filter_ra_gen(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm,
                              undersamp, combined, filter2d, resample, trim=trim):
        assert len(signalim.shape) >= 2
        signalout[trim[0]:trim[1], idx:(idx + signalim.shape[1])] = signalim
        idx += signalim.shape[1]
    assert idx == len_output
    return signalout



def test():
    """ Run self-tests """
    test_fill_gap()

    bxds_input = '/disk/kea/WAIS/orig/xlob/AGAW/JKB2k/JEVANSb/RADnh3/bxds'
    geopath = '/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/AGAW/JKB2k/JEVANSb'

    x = filter_ra(bxds_input, geopath, 1, 3200, 1000, 100, 5)
    assert x is not None
    assert x.shape[0] == 3200 and x.shape[1] > 0

    bxds_input = '/disk/kea/WAIS/orig/xlob/ICP4/JKB2g/F19T03a/RADnh3/bxds'
    geopath = '/disk/kea/WAIS/targ/xtra/ICP4/FOC/Best_Versions/S2_FIL/ICP4/JKB2g/F19T03a'
    assert x is not None
    assert x.shape[0] == 3200 and x.shape[1] > 0
    del x

    logging.info("Processing bxds    " + bxds_input)
    logging.info("Processing geopath " + geopath)
    x = filter_ra(bxds_input, geopath, 1, 3200, 1000, 100, 5, filter2d=False)
    assert x is not None
    del x

    # This one has 49 tears and a short read at the end
    bxds_input = "/disk/kea/WAIS/orig/xlob/DEV/JKB2t/Y49a/RADnh5/bxds"
    geopath = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV/JKB2t/Y49a'

    logging.info("Processing bxds    " + bxds_input)
    logging.info("Processing geopath " + geopath)
    for x in filter_ra_gen(bxds_input, geopath, 1, 3200, 1000, 100, 5, filter2d=False):
        assert x is not None
        assert x.shape[0] == 3200 and x.shape[1] > 0

    # Run self-tests with a trim length
    logging.info("Processing trim bxds    " + bxds_input)
    logging.info("Processing trim geopath " + geopath)
    trim = (10, 58)
    for x in filter_ra_gen(bxds_input, geopath, 1, 3200, 1000, 100, 5, filter2d=False, trim=trim):
        assert x is not None
        assert x.shape[0] == trim[1] - trim[0]
        assert x.shape[1] > 0


    return 0

def make_range_filter(MS, NRb):
    """ 
    Generate 2D filter
    MS: Number of fast time samples
    NRb = Number of records included in each along-track filtering block
    
    returns:
    2D array to be used for filtering
    """
    # --------------------------------------------------
    # Define Range Filtering.
    # Tr = Range sampling time (0.02 microseconds; 50 MHz sampling)
    Tr = 0.02
    FilterR = np.zeros([MS, 1], complex)
    Freq1 = 02.5
    Freq2 = 17.5
    M1 = int(math.floor((Freq1*Tr*MS)+0.5)) + 1
    M2 = int(math.floor((Freq2*Tr*MS)+0.5)) + 1
    BW = M2 - M1
    Hanning = np.reshape(np.sin(np.linspace(0.0,1.0,BW+1) * np.pi),(-1,1))
    FilterR[M1-1:M2] = Hanning
    FilterR[MS+2-M2-1:MS+2-M1] = Hanning
    # Define Along-Track Filtering.
    # Ta = Along-track sampling time (0.0025 s; 400 Hz sampling)
    Ta = 0.0025
    FilterA = np.zeros([NRb,1], complex)
    Freq1 = 35.0
    Freq2 = 40.0
    N1 = int(math.floor((Freq1*Ta*NRb)+0.5)) + 1
    N2 = int(math.floor((Freq2*Ta*NRb)+0.5)) + 1
    BW = N2 - N1
    Hanning = np.reshape(0.5 + 0.5*np.cos(np.linspace(0.0,1.0, BW+1) * np.pi),(-1,1))
    FilterA[N1-1:N2] = Hanning
    FilterA[NRb+2-N2-1:NRb+2-N1] = 1.0-Hanning
    FilterA[0:N1-1] = 1.0
    FilterA[NRb+3-N1-1:NRb] = 1.0
    # Combine into 2D Filter
    Filter = FilterR * FilterA.conj().transpose()
    #--------------------------------------------------------
    return Filter

def fill_gap_gen(rec0_idx, rec0, rec1_idx, rec1, winsize, blocksize=None):
    """
    
    yield numpy arrays to fill a gap (records between two segments).
    
    The gap is filled with silence (zeros), but a cosine taper is applied for the
    transition between the last sample of the previous segment and the silence,
    and a cosine taper is applied between the silence and the first sample of the
    next segment.
    
    This smooths the transition between data and silence and prevents undesired
    ringing effects.
    
    rec0_idx: global index for radar record rec0
    rec0    : a 1D vector containing the last radar record of previous segment.
              This record is used to produce the lead-in cosine taper.
              This is expected to be a complex 1D numpy array.
    rec1_idx: global index for radar record rec1
    rec1    : a 1D vector containing a radar record at the beginning of the next segment
              This record is used to produce the lead-out cosine taper.
              This is expected to be a complex 1D numpy array.
    winsize: length of the cosine taper at one end of the gap, in records.
             For example, a value of 10 makes a cosine taper of up to 10 records
             at the beginning of the gap, and a cosine taper of up to 10 records
             at the end of the gap.
             If the gap is less than 2*winsize records, the
             the cosine tapers will be superimposed and cross-faded on each other.

    blocksize: Blocksize limits the size of the max array to be generated.
               For now, blocksize==None is the only supported value, which 
               generates the entire gap as one numpy array.
    """

    assert rec0_idx < rec1_idx
    assert len(rec0.shape) == 1 and len(rec1.shape) == 1
    assert len(rec0) == len(rec1)
    assert blocksize is None or blocksize > 0
    assert winsize > 0
    # Length of the gap in units of records
    nrecs = rec1_idx - rec0_idx - 1

    if nrecs == 0:
        if WARN_GAP0:
            logging.log(level, "Gap length of 0: previous segment ended at {:d}, "
                        "next segment begins at {:d}".format(rec0_idx, rec1_idx))
        # Yield no records
        return

    # Don't include value 0, so that the window is symmetric and minimal.
    x = np.arange(1, rec1_idx - rec0_idx)
    if nrecs < winsize: # cross-fade
        # Gap is too small to apply a cosine taper of the desired length.
        w0 = x / nrecs # lead-in weights
        #w1 = 1.0 - w0  # lead-out weights
    else: # cosine taper
        # lead-in weights
        # w0 = np.maximum(0, winsize - x) # (test, triangle)
        w0 = 0.5*np.cos((np.pi/winsize)*(np.minimum(winsize, x))) + 0.5
        # lead-out weights
        # w1 = np.maximum(0, winsize + x - nrecs) # (test, triangle)
        # w1 = 0.5*np.cos((np.pi/winsize)*(np.minimum(winsize, nrecs - x + 1))) + 0.5
    # Element-wise sum is less than or equal to 1
    # assert np.max(np.abs(w0) + np.abs(w1)) <= 1.0

    if blocksize is None or nrecs <= blocksize:
        # this is inefficient, but very concise:
        # signal = np.outer(rec0, w0) + np.outer(rec1, w1)
        signal = np.zeros((rec0.shape[0], nrecs), dtype=complex)
        # Iterate through the window size or half the gap length, whichever is smaller.
        for x0 in range(min(winsize, int(np.ceil(nrecs / 2.0)))):
            x1 = nrecs - x0 - 1
            signal[:, x0] = rec0*w0[x0] + rec1*w0[x1]
            signal[:, x1] = rec0*w0[x1] + rec1*w0[x0]
        yield signal
    else:
        # divide and round up
        #logging.info("nrecs={:d} blocksize={:d}".format(nrecs, blocksize))
        nblocks = int(((nrecs + (blocksize-1)) / blocksize))
        for nblock in range(nblocks):
            n0 = nblock * blocksize
            n1 = min((nblock + 1) * blocksize, nrecs)
            signal = np.zeros((rec0.shape[0], n1 - n0))
            
            if n0 <= winsize or nrecs - winsize <= n1:
                # If the range is within the window edges of the gap, process.
                # but of course adjust range if beneficial.
                m0 = max(n0, nrecs - winsize) if n0 > winsize else n0
                m1 = min(winsize, n1) if nrecs - winsize > n1 else n1

                for x0 in range(m0, m1):
                    x1 = nrecs - x0 - 1
                    signal[:, x0 - n0] = rec0*w0[x0] + rec1*w0[x1]
            # else this signal is just zeros.
            yield signal


def test_fill_gap(b_run_timing=False):
    global WARN_GAP0
    # MS = 3200
    # data1 = np.ones((MS,))
    # data2 = 2*np.ones((MS,))
    data_a = np.array([1, 2, 0, 3, 4, 5])
    data_b = np.array([5, 4, 0, 3, 2, 1])
    
    for arr in fill_gap_gen(999, data_a, 1015, data_b, 5):
        for i, rec in enumerate(arr.transpose()):
            pass #print(i, i+999+1, rec.transpose())

    for data1, data2 in ((data_a, data_b), (data_a, data_a)):
        # Short gaps
        for gaplen in (0, 1, 2, 3, 5, 10, 100):
            WARN_GAP0 = not gaplen == 0
            for windowsize in (1, 2, 3, 5, 7, 11, 13):
                x0 = 1
                x1 = x0 + gaplen + 1
                #logging.debug("gaplen={:d} windowsize={:d}".format(gaplen, windowsize))
                arr1 = None
                arr_gen = fill_gap_gen(x0, data1, x1, data2, winsize=windowsize, blocksize=None)
                arr1 = next(arr_gen, None)
                assert next(arr_gen, None) is None # no more items left

                if gaplen == 0:
                    assert arr1 is None
                else:
                    # Check contents for symmetry
                    assert arr1.shape == (len(data1), gaplen)
                
                    if not np.any(data1) and not np.any(data2):
                        # If both boundary records are zero, output should be zero
                        assert not np.any(arr1)
                
                    # assert that the flip is the same as when you reverse the elements.
                    if (data1 == data2).all():
                        assert (np.flip(arr1, axis=1) == arr1).all()
                    elif (data1.astype(bool) == data2.astype(bool)).all():
                        # even if not equal, then at least the same entries will be nonzero
                        assert (arr1.astype(bool) == np.flip(arr1.astype(bool), axis=1)).all()
                        
                    if np.any(data1) and np.any(data2): #if both ends are nonzero,
                        # assert that all middle columns are symmetrically zero.
                        assert (np.all(arr1.astype(bool), axis=0) == np.all(np.flip(arr1.astype(bool), axis=1), axis=0)).all()

                # Check equivalence of blocked output
                for blocksize in (1, 2, 3, 5, 11, 13, 10, 100, 101, 199, 200, 1000, 1001, 1024):
                    idx = 0
                    for arr in fill_gap_gen(x0, data1, x1, data2, winsize=windowsize, blocksize=blocksize):
                        try:
                            assert arr.shape[1] > 0
                            assert (arr1[:, idx:(idx + arr.shape[1])] == arr).all()
                            idx += arr.shape[1]
                        except AssertionError:
                            arr1shape = str(arr1.shape) if arr1 is not None else "None"
                            
                            logging.error("blocksize={:d} gaplen={:d} windowsize={:d}, arr1.shape={:s} "
                                          "idx={:d}".format(blocksize, gaplen, windowsize, arr1shape, idx))
                            logging.error("arr1={:s}\narr={:s}".format(str(arr1), str(arr)))
                            raise

                    if arr1 is None:
                        assert gaplen == 0
                        assert idx == 0
                    else:
                        assert idx == arr1.shape[1] # all entries were filled
    # Reset gap warning
    WARN_GAP0 = True

    ##############################################
    #data1 = np.array([1, 2, 0, 3, 4, 5])
    #data2 = np.array([5, 4, 0, 3, 2, 1])
    # Try a really long gap    for data1, data2 in ((data_a, data_b), (data_a, data_a)):
    # Short gaps
    #for blocksize in (1, 2, 3, 5, 11, 13, 10, 100, 101, 199, 200, 1000, 1001, 1024):
    #    for gaplen in (0, 1, 2, 3, 5, 10, 100, 200, 500, 1000):
    #        for windowsize in (1, 2, 3, 5, 7, 11, 13):
    #            x0 = 1
    #            x1 = x0 + gaplen + 1

    #            arr_gen = fill_gap_gen(x0, data1, x1, data2, winsize=windowsize, blocksize=None)
    #            arr1 = next(arr_gen, None)
    #            assert next(arr_gen, None) is None # no more items left
                
    ntrials = 1
    logging.info("Doing {:d} timed trials of fill_gap".format(ntrials))
            
    data1 = np.arange(0, 3200)
    data2 = np.arange(3200, 0, -1)
    ntrials = 100 if b_run_timing else 1
    for trial in range(ntrials):
        # Try a really long gap
        for gaplen in (1000, 1001, 1999, 2000):
            for windowsize in (10, 11):
                x0 = 1
                x1 = x0 + gaplen + 1

                arr_gen = fill_gap_gen(x0, data1, x1, data2, winsize=windowsize, blocksize=None)
                arr = next(arr_gen, None)
                assert next(arr_gen, None) is None # no more items left
                assert arr.shape == (len(data1), gaplen)

                for blocksize in (1, 2, 13, 101, 200, 1000, 1001, 1024):
                    idx=0
                    for arr in fill_gap_gen(x0, data1, x1, data2, winsize=windowsize, blocksize=blocksize):
                        assert arr.shape[1] > 0
                        idx += arr.shape[1]
                    assert idx == gaplen
    logging.info("Done timed trials of fill_gap")

              
def filter_ra_gen(bxds_input, geo_path, DX, MS, NR, NRr, channel, snm=None,
                  undersamp=False, combined=False, filter2d=True, resample=True, trim=[None, None, None, None]):
    """ Filter a bxds file to doppler filtering and resampling to equal distances
    (with spacing DX).  
    
    Data stored in file bxds_input is assumed to be in one or more contiguous
    *segments*.  These segments are read in as one or more blocks.
    Input blocks are filtered to produce output blocks, and blocks are read
    overlapping.

    
    DX: distance in meters
    MS: number of samples per record
    NR: maximum width (in records) of blocks to be processed
    NRr: Total overlap (in records) per block. If this value is not even, it will
         be incremented to the next even value.
    channel: channel number to read
    combined: Combine two input channels before processing.
    snm: stream name of data type. None to autodetect, or e.g., 'RADnh3', 'RADnh5'
    filter2d: Perform 2d filtering (default true) -- normally this is only disabled
              for debugging or testing.  If true, don't do any doppler filtering.
              This is useful for computing total output data size.

    resample: Perform spatial resampling using quadratic interpolation.  
              (default true) -- normally this is only disabled
              for debugging or testing.
    
    
    """
              
    # Number of output blocks
    out = 0

    # Nc is an array of one-based indices to the nearest record in bxds corresponding
    # to each output record in the distance-resampled array.
    # therefore it should be increasing, 
    
    Nc = np.fromfile(os.path.join(geo_path, "Nc"), dtype=int, sep=" ")
    # Xo is an array of the along-track position of each record in the bxds, in meters.
    # by definition, the length of Xo should be the same as the length of bxds
    Xo = np.fromfile(os.path.join(geo_path, "Xo"), sep=" ")
    # NRt is an array containing the index of the last record in a contiguous segment
    # i.e., it is the index where a tear occurs
    # by definition, the last value in NRt is the same as the length of Xo
    NRt = np.fromfile(os.path.join(geo_path, "NRt"), dtype=int, sep=" ")
    # Insert a zero at the beginning which simplifies some logic later
    NRt = np.insert(NRt, 0, 0)

    # NumTears = Number of data tears
    NumTears = len(NRt) - 1
    
    # x positions of output
    # Perform some checks on Nc
    try:
        #This isn't necessarily true. assert Nc[-1] == len(Xo)
        assert Nc[0] == 1
    except AssertionError as e:
        logging.error("metadata malformed: Nc[0]={:d} Nc[-1]={:d} len(Xo)={:d}".format(Nc[0], Nc[-1], len(Xo)))
        # print the error but don't quit

    # Perform some simple error checks on Xo vs NRt
    if np.abs(Xo[0]) > 1e-3: #pragma: no cover
        raise ValueError("filterRA: Distance Xo[0] must start from zero.")


    if (len(Xo) != NRt[NumTears]): #pragma: no cover
        raise ValueError("filterRA: Distances (Xo) inconsistent with number of records (NRt).")

    logging.debug("FILL: Expecting fill from [{:5d}, {:5d})".format(0, len(Nc)))

    # Force NRr to be an even number
    NRr += NRr % 2

    # Check for single segment, single block case and force variables accordingly.
    if ((NumTears == 1) and (NR > NRt[1])):
        NR = NRt[1]
        NRr = 0

    # NRb = Number of records included in each along-track filtering block
    NRb = NR + NRr


    if trim is not None and trim[1] is not None:
        MS1 = trim[1] - trim[0]
        ftrim = trim[0:2]
    else:
        MS1 = MS
        ftrim = (0, MS)


    Filter = make_range_filter(MS1, NRb)


    # Detect if Hicars by stream name, getting directory name.
    # Find filename, HiCARS1 has bxds[C], HiCARS2 is just bxds
    logging.debug("bxds_input={:s}".format(bxds_input))
    logging.debug("snm={:s}".format(str(snm)))
    if snm is None:
        snm = os.path.basename(os.path.dirname(bxds_input))
    if snm == 'RADjh1':
        HiCARS = 1
    else:
        assert snm == 'RADnh3' or snm == 'RADnh5'
        HiCARS = 2

    # Open Input file if HiCARS1. 
    IFD = None
    f_read_block = None # function pointer to read blocks
    block_source = None # block generator for f_read_block
    if HiCARS == 1:
        # Directly open the file
        IFD = open(bxds_input, "rb")
        f_read_block = read_file_block
        block_source = IFD
    else: # HiCARS == 2
        # Construct the unfoc processor if HiCARS2
        if (channel in [1, 2]):
            if args.combined:
                logging.debug("filter: Combining channels to make channel {:d}".format(channel))
                channel_specs = unfoc.parse_channels('[1,%d,1,%d,1]' % (channel, channel+2))
            else:
                logging.debug("filter: Single channel {:d}".format(channel))
                channel_specs = unfoc.parse_channels('[1,%d,1,0,0]' % channel)
        elif (channel in [5, 6, 7, 8]):
            channel_specs = unfoc.parse_channels('[1,%d,1,0,0]' % (channel-4))
        else: #pragma: no cover
            raise ValueError("filterRA: illegal channel number requested")
        tracegen = unfoc.read_RADnhx_gen(bxds_input, channel_specs)
        stackgen = unfoc.stacks_gen(tracegen, channel_specs, 1)
        f_read_block = read_stackgen_block
        block_source = stackgen

    rec_prev = None
    rec_prev_idx = None
    # Start processing. NT is the index of the current segment
    # NT0 is the zero-based NT index
    for NT0 in range(NumTears):

        # NRs = Number of records to process up to the next data tear,
        # or to the last position available
        assert NRt[NT0 + 1] > NRt[NT0]
        NRs = min(len(Xo), NRt[NT0 + 1]) - min(len(Xo), NRt[NT0])
        # The positions end before the end of this segment. This is
        # unexpected and probably means some bxds data is missing.
        if len(Xo) < NRt[NT0 + 1]:
            msg = "Not enough positions for segment {:d} of {:d} (radar records {:d} to {:d}): " \
                  "Xo has {:d} positions.".format(NT0, NumTears, NRt[NT0], NRt[NT0 + 1], len(Xo))
            if len(Xo) < NRt[NT0]:
                logging.warning(msg)
                break # quit. Will quitting cause problems?
            else:
                logging.info(msg)


        # NumNBlocks = Number of along-track blocks
        NumNBlocks = max(1, int(math.floor((NRs+NR-1-NRr//2)/NR)))
        if (NumNBlocks == 0):
            logging.error("filter: NumNBlocks={:d}".format(NumNBlocks))
            logging.error("filter: NRs=%d NR=%d NRr=%d" % (NRs,NR,NRr))
            raise ValueError("NumNBlocks=0")
        logging.debug("filter: NumNBlocks={:d}".format(NumNBlocks))

        # Start processing along-track blocks.
        # NB0 is zero-based NB indexing variable
        for NB0 in range(NumNBlocks):

            # Display loop counters for progress reporting.
            # logging.debug("filter: Current file: bxds_input=" + bxds_input)
            # Contiguous segment and block within segment
            #logging.debug("filter: Current segment: NT={:d} of {:d}; "
            #              "NB={:d} of {:d}".format((NT0+1), NumTears, NB0 + 1, NumNBlocks))

            # NRp = Number of records read in previous block
            if NB0 > 0:
                NRp = NumRead


            # NumRead = Number of new records to read
            if (NB0 == 0) and (NB0 == NumNBlocks - 1):   # Fix for very short data sections
                NumRead = NRs
            elif NB0 == 0:
                NumRead = NR + (NRr//2)
            elif NB0 == NumNBlocks - 1:
                NumRead = NRs - (NB0*NR) - (NRr//2)
            else:
                NumRead = NR


            # NGPri = Number of initial (start) record being processed this block
            # NGPrf = Number of  final  (stop)  record being processed this block
            # NOTE: NGPri and NGPrf are in the global index system, where "global" refers
            #       to the full set of records.
            # These variables are not used anywhere else in this code,
            # but they are output here for progress reporting.

            NGPri = NRt[NT0] + max(0, (NB0*NR) - (NRr//2)) + 1
            NGPrf = NRt[NT0] + ((NB0+1)*NR) + (NRr//2)

            if (NB0 == NumNBlocks - 1):
                NGPrf = NRt[NT0] + NRs

            # NGWri = Number of initial (start) record for controlling output on this processed block
            # NGWrf = Number of  final  (stop)  record for controlling output on this processed block
            # NOTE: NGWri and NGWrf are in the global index system, where "global" refers
            #       to the full set of records.
            # These variables are output for progress reporting.

            NGWri = NRt[NT0] + (NB0*NR) + 1
            NGWrf = NRt[NT0] + ((NB0+1)*NR)
            if (NB0+1) == NumNBlocks:
                NGWrf = NRt[NT0] + NRs


            #logging.debug("filter: Block range input  is "
            #            "NGPri={:d} to NGPrf={:d}".format(NGPri, NGPrf))

            #logging.debug("filter: Block range output is NGWri={:d} to NGWrf={:d}".format(NGWri, NGWrf))
            logging.debug("filter: Block range input NGPri={:d} to NGPrf={:d}, output "
                          "NGWri={:d} to NGWrf={:d}".format(NGPri, NGPrf, NGWri, NGWrf))

            # Read Data and define signal.
            # Pad (NRr//2) overlap region with first/last records on first/last blocks.
            if NB0 == 0:
                # NOTE GNG: for not the first gap, we could try crossfading data in
                S = f_read_block(block_source, MS, NumRead, (NB0+1), bxds_input)
                # Initialize enough space for number of blocks to read (NumRead), 
                # plus overlap at the beginning (NRr//2).
                signal = np.empty((MS1,int(NRr//2+NumRead)))
                # Pad the beginning of the block with the first block read
                for N in range(NRr//2):
                    signal[:, N] = S[ftrim[0]:ftrim[1], 0]
                # Insert the rest of the signal into the block
                signal[:, (NRr//2):(NRr//2)+NumRead] = S[ftrim[0]:ftrim[1], :]
            else:
                # If not the first block in the segment, then
                # Initialize enough space for number of blocks to read (NumRead)
                # plus overlap at beginning (NRr//2) and at end (NRr//2)
                signal = np.empty((MS1, NRr+NumRead))
                # Copy previous overlap
                signal[:, 0:NRr] = S[ftrim[0]:ftrim[1], NRp-NRr:NRp]
                S = f_read_block(block_source, MS, NumRead, (NB0+1), bxds_input)
                signal[:, NRr:NRr+NumRead] = S[ftrim[0]:ftrim[1], :]

            if ((NB0 > 0) and ((NB0+1) == NumNBlocks)):
                # If this is the last block of a multi-block segment
                # then resize for padding (why is resize necessary?)
                signal = np.resize(signal, (MS1, NRb))
                # and fill these records with the last record.
                for N in range(NRr+NumRead, NRb):
                    signal[:, N] = S[ftrim[0]:ftrim[1], NumRead-1]

            #pcheck.pcheck(signal, "signal")
  
            # Clear top samples
            if HiCARS == 2:
                # HiCARS2
                signal[0:250, :] = 0
            else: 
                # HiCARS1
                signal[0:50, :] = 0


            if filter2d:
                #find peak energy below 250 samples
                # No more shifting
                #[m,im]=max(signal);
                #shifter=round(mean(im));
                #signal=shift(signal,-shifter);
                # Filter signal in the following steps:
                #      2D FFT
                #      Interpolate on sampling frequency and harmonic
                #      Filter as a Dot-Product
                #      2D IFFT
                F = pyfftw.interfaces.numpy_fft.fft2(detrend(signal, 0),[MS1, NRb])

                if not undersamp:
                    # Don't cinterp downcoverted radars
                    # New LO leakage removal code (measure on bottom 1/4 of the data)
                    # This probably only works on 3200 sample data and a 10 MHz leak
                    Fs = pyfftw.interfaces.numpy_fft.fft2(signal[MS1-800-1:MS1-1],[800-1,NRb])
                    #FIXME I think Matlab dfts and python dfts look different
                    F[2561-1,:] = F[2561-1,:] - 4*Fs[641-1,:]

                F = Filter * F
                signal = pyfftw.interfaces.numpy_fft.ifft2(F,[MS1,NRb]) # .astype(int)

                # No more shifting
                #signal = shift(signal,shifter);
                del F

            # Get global record resampling indices.
            # GNG: the indices of the output records
            # Nii = Number of initial (start) new/resampled record to be output this processed block.
            # Nif = Number of  final  (stop)  new/resampled record to be output this processed block.
            # These variables are output for progress reporting.
            if NB0 == 0:
                # if the first block in a segment (not necessarily the first segment), calculate the initial position as
                # the position reported in Xo, but add just under a meter to it
                # GNG: is there a reason why this isn't a ceiling? perhaps
                # Nii = int(math.ceil(Xo[NGWri-1]/DX)) + 1
                # The floor appears to exist in the previous python version and matlab.
                Nii = int(math.floor((Xo[NGWri-1]/DX)+0.99999)) + 1
                
                # in the case where this is the first block in the segment,  
                # Nii could be larger than NiF, if the block extent is less than 1 meter. (0.99999)
            else:
                # previous plus 1. easy
                Nii = Nif + 1
                
            # Convert the location into an output index.
            Nif = int(math.floor(Xo[NGWrf-1]/DX)) + 1
            output_block_len = Nif - Nii + 1
            #Nii
            #Nif
            # output x positions
            logging.debug("filter: Xo[NGWri-1]={:0.2f} and Xo[NGWrf-1]={:0.2f}".format(Xo[NGWri-1], Xo[NGWrf-1]))
            #logging.debug("filter: Resampled record range for this block: Nii={:d} to Nif={:d}".format(Nii, Nif))



            # --------------------------------------------------
            # Interpolate filtered signal to resampling points.
            # inputs: signal, MS, Nif, Nii, Nc, Xo, NGWri, NRr
            # output: signali -- interpolated signal


            if Nif < Nii:
                logging.warning("filter: Nif < Nii! Block range output is NGWri={:d} to NGWrf={:d}".format(NGWri, NGWrf))
                logging.warning("filter: Skipping this block")
                continue



            signali = np.empty((MS1, Nif-Nii+1), complex)
            if resample:
                # gotta have at least 3 samples.
                # assert signal.shape[1] >= 3
                for j0, Ni in enumerate(range(Nii, Nif+1)):
                    # Get the index of the nearest record in the input, but
                    # saturate to make sure we don't go off the end of the array
                    # when getting neighbors.
                    # GNG: these also seem like hacks to keep it from going off the end.
                    Nci = Nc[Ni-1]
                    Nci = max(Nci, 2)
                    Nci = min(Nci, len(Xo)-1)
                    # Get the desired x position of this sample
                    X = (Ni-1)*DX
                    # Get x coordinates of neighboring samples
                    X1 = Xo[Nci-1-1]
                    X2 = Xo[Nci-1]
                    X3 = Xo[Nci+1-1]
                    # Get signal values at these positions
                    # Note: there appears to be nothing to guarantee that
                    #signal = np.empty((MS1, NRr+NumRead))
                    # the right thing to do is to read forward so you can access it.
                    # i1+1 < signal.shape[1]
                    # NGWri = NRt[NT-1-1] + ((NB-1)*NR) + 1
                    # NRr / 2 = 50
                    i1 = Nci-NGWri+(NRr//2)
                    #logging.debug("NT={:4d} NB={:2d} Ni={:4d} j0={:d} i1={:d}".format((NT0+1), NB, Ni, j0, i1))
                    if i1+1 >= signal.shape[1]:
                        #logging.error("Nci-NGWri+(NRr//2) + 1 >= signal.shape[1] ({:d} >= {:d})".format(i1 + 1, signal.shape[1]))
                        #logging.error("Ni={:d} Nc[Ni-1]={:d} Nci={:d} len(Xo)={:d} ".format(Ni, Nc[Ni-1], Nci, len(Xo)))
                        # GNG This is a hack to keep it from going off the end.
                        i1 -= 1
                    P1 = signal[:, i1-1]
                    P2 = signal[:, i1]
                    P3 = signal[:, i1+1]
                    # Compute interpolated signal
                    # assert Ni - Nii == j0
                    signali[:, Ni-Nii] = quad3(X, X1, X2, X3, P1, P2, P3)
            else:
                # Get the nearest neighbor.
                #for Ni in range(Nii, Nif+1):
                #    P2idx = (Nc[Ni-1]-NGWri+(NRr//2)+1)-1
                #    signali[:, Ni-Nii] = signal[:, P2idx]
                # Alternatively for testing we fill with zeros.
                signali.fill(0.0)
            # --------------------------------------------------
            
            # If this is the first block and not the segment, then output a gap.
            if NT0 > 0 and NB0 == 0:
                assert rec_prev is not None and rec_prev_idx is not None
                
                logging.debug("filter: FILL A NT={:d}/{:d} NB={:d}/{:d} Ni=[{:5d}, {:5d})".format(NT0+1, \
                              NumTears, NB0+1, NumNBlocks, rec_prev_idx + 1, Nii))
                for ngapblock, sig_gap in enumerate(fill_gap_gen(rec_prev_idx, rec_prev, Nii, signali[:, 0], winsize=10)):
                    yield sig_gap
                    out += sig_gap.shape[1]

            rec_prev_idx = Nif # TODO: is this right?
            rec_prev = signali[:, -1]

            logging.debug("filter: FILL C NT={:d}/{:d} NB={:d}/{:d} Ni=[{:5d}, {:5d})".format(NT0+1, \
                          NumTears, NB0+1, NumNBlocks, rec_prev_idx + 1, Nii))
            yield signali #signali.astype('<i2').T.tofile(OFD)
            out += signali.shape[1]

            if out != Nif:
                logging.warning("filter: Block processing mismatch: Nif=%d out=%d "
                                "(diff=%d) shape=%d" % (Nif, out, Nif - out, signali.shape[1]))



    try:
        assert out == len(Nc)
    except AssertionError:
        logging.error("Output length doesn't match! out={:d} len(Nc)={:d} diff={:d}".format(out, len(Nc), len(Nc) - out))
        #raise
    logging.debug("Output {:d} blocks".format(out))
    if IFD:
        IFD.close()



if __name__ == "__main__":
    main()


