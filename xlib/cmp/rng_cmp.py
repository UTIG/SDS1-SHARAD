#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu',
               'Kirk Scanlan, kirk.scanlan@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'August 15, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'
                 'CMP Library is a collection of functions used'
                 'for the pulse compression of SHARAD data and'
                 'to correct for the ionospheric distortion'}}

import os
import sys
import logging
from typing import List
import traceback

import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

sys.path.insert(1, os.path.dirname(__file__))
import pds3lbl as pds3



def us_refchirp(tec_custom=None, maxTECU=1, resolution=50):
    """
    This subroutine creates a menu of SHARAD reference chirps used for
    pulse compression as a function of different total electron
    content values. One total electron content unit (TECU)
    is TEC x 10^-16. TEC is correlated with the E value
    by TEC = E/0.29. For reference see
    [1] Campbell et al. 2011 doi: 10.1109/LGRS.2011.2143692
    [2] Campbell and Watters 2016 doi: 10.1002/2015JE004917

    By default, if tec is None, create a menu of reference chirps
    using the max_tecu and resolution arguments to create chirps
    corresponding to TEC values from 0 to maxTECU.

    If no Ionosphere is present, TECU is set to 0.

    TODO: this is kind of weirdly formatted since maxTECU
    could be a float and resolution is an int.  Maybe we're missing
    something?

    Input:
    -----------
        tec_custom: float
            With default value (None), create a menu of chirps
            If specified, use this specific value of TEC to create one refchirp
            Use a value of 0 for no ionospheric correction.
        maxTECU (optional): Maximum TECU value to be expected.
                            Default value is 1.
        resolution: int
            number of reference chirps to produce (if tec_custom is None)

    Output:
    -----------
        fs: Set of filter functions (reference chirps)
            suitable for pulse compression
    """

    # Parameters
    fl = 15E+6   # Sharad lower frequency 15 MHz
    a = (10E+6/85.05E-6) # Frequency rate 10 MHz/85 mus
    t = np.arange(0, 85.05E-6, 0.0375E-6) # Times in rng window

    # Chirp can be expressed as an instantaneous angular frequency ([2] eq 2)
    phi = 2*np.pi*(t*fl+a/2*t**2)
    nsamples = 3600

    if tec_custom is None:
        # Initialize filter array
        fs = np.empty((maxTECU*resolution, nsamples), dtype=complex)
        for i in range(0, maxTECU*resolution):
            # Calculate empiric E value
            E = i*1E+16/(0.29*resolution)
            fs[i] = iono_ref_chirp(E, fl, a, t, phi, nsamples)
    else:
        E = tec_custom*1E+16/(0.29*resolution)
        fs = iono_ref_chirp(E, fl, a, t, phi, nsamples)

    return fs

def iono_ref_chirp(E: float, fl: float, a: np.ndarray, t: np.ndarray, phi: np.ndarray, nsamples: int):
    """ Calculate ionospherically-distorted reference chirp
    by TEC = E/0.29. For reference see
    [1] Campbell et al. 2011 doi: 10.1109/LGRS.2011.2143692
    [2] Campbell and Watters 2016 doi: 10.1002/2015JE004917
    """
    # Compute phase shift due to ionospheric distortion ([2] eq (7))
    # Use a single float if no correction
    phase = E*(fl+a*t)**(-1.93) if (E != 0.) else 0.

    # Compute distorted chirp ([2] eq (3))
    C = -np.sin(phi-phase)+1j*np.cos(phi-phase)
    # Pad to requested number of samples
    ref_chirp = np.pad(C, (nsamples-len(C), 0),
                       'constant', constant_values=0)
    # Chirp needs to be flipped due to SHARAD aliasing
    ref_chirp = np.fft.fft(np.flipud(ref_chirp))
    return ref_chirp




def us_rng_cmp(data, chirp_filter=True, iono=True, maxTECU=1, resolution=50,
               debug=True):
    """
    Performs the range compression according to the Bruce Campbell
    method. In case of ionosphere it tries to find the optimal
    TEC expressed by the empiric factor E to get the best SNR return

    Input:
    -----------
        data:       Track to be compressed [len(track) x 3600 samples]
        chirp_filter: If a filter is applied to the reference chirps
        iono (optional): If ionospheric correction is needed
        maxTECU:    Maximum TECU = TEC x 10E-16 to be expected
        resolution: In how many steps the TECU shall be tested
    Output:
    -----------
        E:         Optimal E value found
        dechirped: Pulse compressed with optimal E value
    """
    # Compute list of reference chirps
    tec_custom = None if iono else 0.
    fs = us_refchirp(tec_custom, resolution=resolution, maxTECU=maxTECU)
    if iono:
        #hammingf = Hamming(15E6, 25E6)
        csnr = np.empty((len(fs), len(data)))
        # Perform range compression per filter and record SNR
        fftdata = np.fft.fft(data)
        # apply frequency domain filter if desired
        # (combine it with the original fft data)
        if chirp_filter:
            fftdata *= Hamming(15E6, 25E6)

        for i, chirp in enumerate(fs):
            product = fftdata*np.conj(chirp)
            dechirped = np.fft.ifft(product)
            # Noise is recorded within first 266 samples
            var = np.var(dechirped[:, 0:266], axis=1)
            # Signal is the maximum amplitude (GNG: should this be np.abs?)
            maxi = np.max(abs(dechirped), axis=1)
            csnr[i] = maxi**2/var

        Emax = np.argmax(csnr, axis=0)
        # Create a histogram of SNR maximizing E's
        hist, edges = np.histogram(Emax, bins=resolution*maxTECU)
        if debug:
            pass #plt.bar(np.arange(len(hist)), hist)
            #plt.show()

        # Fit histogram by a Gauss function
        try:
            opt, cov = curve_fit(Gaussian, np.arange(len(hist)),
                                 hist,
                                 p0=[len(data)/2, resolution*maxTECU/2, 20])
        except:
            opt, cov = [-1, 0, -1], [-1, -1, -1]
            pass #raise # Raise an error to get a more specific error

        if debug: # pragma: no cover
            logging.debug('Gauss fit opt/cov: %r %r', opt, cov)

        x0 = min(49, max(0, opt[1]))
        E = x0/maxTECU/resolution
        sigma = opt[2]/maxTECU/resolution

        # Pulse compress whole track with optimal E
        fs = us_refchirp(resolution=resolution, tec_custom=x0)

        product = fftdata*np.conj(fs)
        dechirped = np.fft.ifft(product)

    else:
        E, sigma = 0, 0
        product = np.fft.fft(data) * np.conj(fs)
        del fs
        # apply frequency domain filter if desired
        if chirp_filter:
            product *= Hamming(15E6, 25E6)
        dechirped = np.fft.ifft(product)
        if debug:
            pass #plt.show()
    return E, sigma, dechirped

def Gaussian(x, a, x0, sigma):
    """
    Simple Gaussian distribution
    This function is used internally for curve fitting.

    Input:
    -----------
        x: Input array of x-values for Gaussian to be evaluated
        a: Amplitude at x0
       x0: Center of Gaussian
    sigma: Standard deviation of Guassian

    Output:
    -----------
        Gaussian at x values

    """
    return a*np.exp(-(x - x0)**2/(2*sigma**2))

def Hamming(Fl, Fh):
    """
    Create a frequency domain Hamming filter

    Input:
    -----------
       Fl: lower cutoff frequency for the Hamming filter
       Fh: upper cutoff frequency for the Hamming filter

    Output:
    -----------
        Frequency domain Hamming filter
    """

    min_freq = int(round((Fl) * 3600 / (1/0.0375E-6)))
    max_freq = int(round((Fh) * 3600 / (1/0.0375E-6)))
    dfreq = max_freq - min_freq + 1
    hamming = np.sin(np.linspace(0, 1, num=dfreq) * np.pi)
    hfilter = np.flipud(np.hstack((np.zeros(min_freq), hamming,
                                   np.zeros(3600 - min_freq - hamming.size))))
    return hfilter

def decompress_sci_data(data, compression:str, presum: int, bps, sdi):
    """
    Decompress the science data in-place according to the
    SHARAD interface specification.

    Input:
    -----------
        data: data to be decompressed in place
        compression: type of compression.
                     use 'static' or 'dynamic'
        presum: Onboard presumming parameter
        bps: Compression parameter
        sdi: compression parameter
    Output:
    -----------
        (none)
    """

    decomp = decompression_scale_factors(compression, presum, bps, sdi)
    data *= decomp


def decompression_scale_factors(compression:str, presum, bps, sdi):
    """
    Return the decompression scaling factors according to the
    SHARAD interface specification.

    Input:
    -----------
        data: data to be decompressed in place
        compression: type of compression.
                     use 'static' or 'dynamic'
        presum: Onboard presumming parameter
        bps: Compression parameter
        sdi: compression parameter
    Output:
    -----------
        coefficient (for static scaling) or numpy array
    """
    if compression == 'static': # Static scaling
        L = np.ceil(np.log2(int(presum)))
        R = bps
        S = L - R + 8
        N = presum
        decomp = np.power(2, S) / N
    elif compression == 'dynamic': # pragma: no cover
        # dynamic scaling
        raise NotImplementedError('Dynamic scaling is not yet implemented and tested')
        N = presum
        if sdi <= 5:
            S = sdi
        elif 5 < sdi <= 16:
            S = sdi - 6
        elif sdi > 16:
            S = sdi - 16
        #data *= (np.power(2, S) / N)
        decomp = np.power(2, S) / N
    else:
        raise ValueError('Compression type %r not understood' % compression)
    logging.debug("decomp %r", decomp)
    return decomp


def calculate_chunks(tlp: List[float], chunklen_km: float):
    """ Calculate dimensions
    tlp is along-track distance for the track
    chunklen_km is nominal length of each chunk
    """
    chunks = []
    tlp0 = tlp[0]
    i0 = 0
    for i, tlp1 in enumerate(tlp):
        if tlp1 > tlp0 + chunklen_km:
            chunks.append([i0, i])
            i0 = i
            tlp0 = tlp1
    if not chunks: # if chunks is empty
        chunks = [(0, len(tlp))]
    elif (tlp[-1]-tlp[i0]) >= (chunklen_km/2.):
        chunks.append([i0, len(tlp)]) # add a new chunk
    else: # append this chunk to the previous chunk
        chunks[-1][1] = len(tlp)
    return chunks #chunks = np.array(chunks)

def parse_decompress_parameters(sci_data, idx_start: int):
    compression = 'static' if (sci_data['COMPRESSION_SELECTION'][idx_start] == 0) else 'dynamic'

    # Tracking presumming table. Converts TPS field into presumming count
    tps_table = (1, 2, 3, 4, 8, 16, 32, 64)
    tps = sci_data['TRACKING_PRE_SUMMING'][idx_start]
    assert tps >= 0 and tps <= 7
    presum = tps_table[tps]
    sdi = sci_data['SDI_BIT_FIELD'][idx_start]
    bps = 8

    return compression, presum, bps, sdi



def read_and_chunk_radar(aux_data_path:str, sci_label_path:str, aux_label_path:str, idx_start:int, idx_end:int, taskname, chunklen_km=30.):
    """ Read radargram and related files and return an array with science data
    TODO: should we start with a SHARAD root and a relative path (that we can get
    from the PDS table?

    return value: tuple of two values
    data: numpy array with complex radar data, decompressed
    chunks: An array of indexes describing radar chunks
    """

    # Load data
    science_path = aux_data_path.replace('_a.dat', '_s.dat')
    data = pds3.read_science(science_path, sci_label_path, science=True)
    aux  = pds3.read_science(aux_data_path, aux_label_path, science=False)

    # Array of indices to be processed
    if idx_start is None or idx_end is None:
        idx_start, idx_end = 0, len(data)
    idx = np.arange(idx_start, idx_end)
    logging.debug('%s: Length of track: %d traces', taskname, len(idx))

    raw_data = chop_raw_data(data, idx)

    compression, presum, bps, sdi = parse_decompress_parameters(data, idx_start)
    decomp = decompression_scale_factors(compression, presum, bps, sdi)

    # Decompress the data
    #decompress_sci_data(raw_data, compression, presum, bps, sdi)
    # Get groundtrack distance and define 30 km chunks
    tlp = list(data['TLP_INTERPOLATE'][idx_start:idx_end])
    chunks = calculate_chunks(tlp, chunklen_km)
    logging.debug('%s: chunked into %d pieces', taskname, len(chunks))

    # TODO: slice sza out to reduce memory usge?
    #sza = aux_data['SOLAR_ZENITH_ANGLE']
    return raw_data, decomp, chunks, aux, idx_start, idx_end


def compress_chunks(raw_data: np.ndarray, decomp, chunks, aux, chirp_filter: bool, verbose: bool, idx_start:int, idx_end:int, taskname):
    """ Perform pulse compression on all the chunks """
    # TODO: E_track can just be a list of tuples
    E_track = np.empty((idx_end-idx_start, 2))

    # TODO: memmap?
    cmp_track = np.empty(raw_data.shape + (2,), dtype=np.int16) # real and imaginary

    # Compress the data chunkwise and reconstruct
    for i, (start, end) in enumerate(chunks):
        decompress_chunk = np.empty((end-start, raw_data.shape[1]), dtype=complex)
        decompress_chunk[...] = raw_data[start:end]
        decompress_chunk *= decomp


        #check if ionospheric correction is needed
        iono_check = np.where(aux['SOLAR_ZENITH_ANGLE'][start:end]<100)[0]
        b_iono = len(iono_check) != 0
        minsza = min(aux['SOLAR_ZENITH_ANGLE'][start:end])
        logging.debug('%s: chunk %3d/%3d Minimum SZA: %0.3f  Ionospheric Correction: %r',
            taskname, i, len(chunks), minsza, b_iono)

        E, sigma, cmp_data = us_rng_cmp(decompress_chunk, chirp_filter=chirp_filter,
                                        iono=b_iono, debug=verbose)

        cmp_track[start:end, :, 0] = np.round(cmp_data.real)
        cmp_track[start:end, :, 1] = np.round(cmp_data.imag)
        E_track[start:end, 0] = E
        E_track[start:end, 1] = sigma
 
    return cmp_track, E_track


def save_cmp_files(h5_filename:str, tecu_filename:str, cmp_track, E_track, taskname:str, save_np=False):
    # Assume these two files are being put into the same directory
    outdir = os.path.dirname(h5_filename)
    logging.debug('%s: Saving to folder: %s', taskname, outdir)
    os.makedirs(outdir, exist_ok=True)

    # restructure of data save
    dfreal = pd.DataFrame(cmp_track[:, :, 0])
    dfreal.to_hdf(h5_filename, key='real', complib='blosc:lz4', complevel=6)
    del dfreal
    dfimag = pd.DataFrame(cmp_track[:, :, 1])
    dfimag.to_hdf(h5_filename, key='imag', complib='blosc:lz4', complevel=6)
    del dfimag
    if save_np:
        np.savez(os.path.splitext(h5_filename)[0] + "_int16.npz",
                 real=cmp_track[:, :, 0], imag=cmp_track[:, :, 1])
    os.makedirs(os.path.dirname(tecu_filename), exist_ok=True)
    np.savetxt(tecu_filename, E_track)



def test_cmp_processor(infile, outdir, idx_start=None, idx_end=None,
                       taskname="TaskXXX", sharad_root=None, chirp_filter=True, verbose=False
                       ):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual tracks and returns pulse compressed data.

    Input:
    -----------
      infile       : Path to track file to be processed.
      outdir       : Path to directory to write output data
      idx_start    : Start index for processing.
      idx_end      : End index for processing.
      chirp_filter : Apply a filter to the reference chirp
      verbose      : Gives feedback in the terminal.

    Output:
    -----------
      E          : Optimal E value
      cmp_pulses : Compressed pulses

    """

    try:
        time_start = time.time()

        # Get science data structure
        label_path = os.path.join(sharad_root, 'mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(sharad_root, 'mrosh_0004/label/auxiliary.fmt')

        decompressed, decomp, chunks, aux, idx_start, idx_end = read_and_chunk_radar(infile, label_path, aux_path, idx_start, idx_end, taskname)

        cmp_track, E_track = compress_chunks(decompressed, decomp, chunks, aux, chirp_filter, verbose, idx_start, idx_end, taskname)

        if outdir != "":
            data_file   = os.path.basename(infile)
            outfilebase = data_file.replace('.dat', '.h5')
            h5_filename = os.path.join(outdir, outfilebase)
            outfilebase = data_file.replace('.dat', '_TECU.txt')
            tecu_filename = os.path.join(outdir, outfilebase)

            save_cmp_files(h5_filename, tecu_filename, cmp_track, E_track, taskname, save_np=True)

    except Exception as e: # pragma: no cover
        logging.error('%s: Error processing file %s', taskname, infile)
        for line in traceback.format_exc().split("\n"):
            logging.error('%s: %s', taskname, line)
        return 1
    logging.info('%s: Success processing file %s', taskname, infile)
    return 0

def chop_raw_data(data, idx, nsamples=3600):
    raw_data = np.empty((len(idx), nsamples), dtype=data['sample0'].dtype)
    for j in range(nsamples):
        k = 'sample' + str(j)
        raw_data[:, j] = data[k][idx].values
    return raw_data


def main():

    parser = argparse.ArgumentParser(description='Range compression library')
    parser.add_argument('-o', '--output', default='./rng_cmp_data',
                        help="Output base directory")
    #parser.add_argument('--ofmt', default='npy',
    #                    choices=('hdf5', 'npy', 'none'),
    #                    help="Output file format")

    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default=None, #"elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="rng_cmp: [%(levelname)-7s] %(message)s")

    SDS = os.getenv('SDS', '/disk/kea/SDS')
    sharad_root = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR')

    # Read lookup table associating gob's with tracks
    if args.tracklist is None:
        lookup = (
            os.path.join(sharad_root, 'mrosh_0001/data/edr19xxx/edr1920301/e_1920301_001_ss04_700_a_a.dat'),
            #os.path.join(sharad_root, 'mrosh_0001/data/edr03xxx/edr0336603/e_0336603_001_ss19_700_a_a.dat'),
            #os.path.join(sharad_root, 'mrosh_0004/data/rm286/edr5272901/e_5272901_001_ss19_700_a_a.dat'),
            #os.path.join(sharad_root, 'mrosh_0004/data/rm278/edr5116601/e_5116601_001_ss19_700_a_a.dat'),
            #os.path.join(sharad_root, 'mrosh_0004/data/rm184/edr3434001/e_3434001_001_ss19_700_a_a.dat'),
        )
    else:
        lookup = np.genfromtxt(args.tracklist, dtype='str')

    # Build list of processes
    logging.debug("Building task list for test")
    process_list = []

    logging.debug("Base output directory: " + args.output)
    for i, infile in enumerate(lookup):
        path_file = os.path.dirname(os.path.relpath(infile, sharad_root))
        data_file = os.path.basename(infile)
        outdir    = os.path.join(args.output, path_file, 'ion')

        logging.debug("Adding " + infile)
        process_list.append([infile, outdir, None, None, "Task{:03d}".format(i+1)])

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        return 0

    logging.info("Start processing %d tracks", len(process_list))

    start_time = time.time()
    named_params = {
        'chirp_filter':True,
        'verbose':args.verbose,
        'sharad_root': sharad_root,
        }
    # Single processing (for profiling)
    for t in process_list:
        test_cmp_processor(*t, **named_params)
    logging.info("Done in %0.2f seconds",time.time() - start_time)


if __name__ == "__main__":
    # execute and import only if run as a script
    import argparse
    import time
    main()

