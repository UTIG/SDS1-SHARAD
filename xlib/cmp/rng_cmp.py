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

import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def us_refchirp(iono=True, custom=None, maxTECU=1, resolution=50):
    """
    This subroutine creates SHARAD reference chirps used for
    pulse compression as a function of different TECU values.
    TECU is TEC x 10^-16. TEC is correlated with the E value
    by TEC = E/0.29. For reference see
    Campbell et al. 2011 doi: 10.1109/LGRS.2011.2143692
    Campbell and Watters 2016 doi: 10.1002/2015JE004917
    If no Ionosphere is present TECU is set to 0.

    Input:
    -----------
        maxTECU (optional): Maximum TECU value to be expected.
                            Default value is 1.

    Output:
    -----------
        fs: Set of filter functions (reference chirps)
            suitable for pulse compression
    """

    # Parameters
    fl = 15E+6   # Sharad lower frequency 15 MHz
    a = (10E+6/85.05E-6) # Frequency rate 10 MHz/85 mus
    t = np.arange(0, 85.05E-6, 0.0375E-6) # Times in rng window

    # Chirp can beexpressed as an instantaneous angular frequency
    phi = 2*np.pi*(t*fl+a/2*t**2)

    if iono:
        if custom is None:
            # Initialize filter array
            fs = np.empty((maxTECU*resolution, 3600), dtype=np.complex_)
            for i in range(0, maxTECU*resolution):
                # Calculate empiric E value
                E = i*1E+16/(0.29*resolution)
                # Compute phase shift due to ionospheric distortion
                phase = E*(fl+a*t)**(-1.93)
                # Compute distorted chirp
                C = -np.sin(phi-phase)+1j*np.cos(phi-phase)
                # Pad to 3600 samples
                ref_chirp = np.pad(C, (3600-len(C), 0),
                                   'constant', constant_values=0)
                # Chirp needs to be flipped due to SHARAD aliasing
                ref_chirp = np.fft.fft(np.flipud(ref_chirp))
                fs[i] = ref_chirp
        else:
            E = custom*1E+16/(0.29*resolution)
            # Compute phase shift due to ionospheric distortion
            phase = E*(fl+a*t)**(-1.93)
            # Compute distorted chirp
            C = -np.sin(phi-phase)+1j*np.cos(phi-phase)
            # Pad to 3600 samples
            ref_chirp = np.pad(C, (3600-len(C), 0),
                               'constant', constant_values=0)
            # Chirp needs to be flipped due to SHARAD aliasing
            fs = np.fft.fft(np.flipud(ref_chirp))
    else:
        # Without ionosphere - no phase
        C = -np.sin(phi)+1j*np.cos(phi)
        # Pad to 3600 samples
        ref_chirp = np.pad(C, (3600 - len(C), 0),
                           'constant', constant_values=0)
        # Flip and fft
        fs = np.fft.fft(np.flipud(ref_chirp))

    return fs


def us_rng_cmp(data, chirp_filter=True, iono=True, maxTECU=1, resolution=50,
               debug=True):
    """
    Performs the range compression according to the Bruce Campbell
    method. In case of ionosphere it tries to find the optimal
    TEC expressed by the empiric factor E to get the best SNR return

    Input:
    -----------
        raw_data:       Track to be compressed [len(track) x 3600 samples]
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
    fs = us_refchirp(iono, resolution=resolution,
                     maxTECU=maxTECU)
    if iono:
        #hammingf = Hamming(15E6, 25E6)
        csnr = np.empty((len(fs), len(data)))
        # Perform range compression per filter and record SNR
        fftdata = np.fft.fft(data)
        # apply frequency domain filter if desired
        # (combine it with the original fft data)
        if chirp_filter:
            fftdata = fftdata * Hamming(15E6, 25E6)

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
            opt = [-1, 0, -1]
            cov = [-1, -1, -1]

        if debug:
            print('Gauss fit opt/cov:', opt, cov)

        x0 = min(49, max(0, opt[1]))
        E = x0/maxTECU/resolution
        sigma = opt[2]/maxTECU/resolution

        # Pulse compress whole track with optimal E
        fs = us_refchirp(iono, resolution=resolution,
                         custom=x0)

        product = fftdata*np.conj(fs)
        dechirped = np.fft.ifft(product)

    else:
        E = 0
        sigma = 0
        # GNG: should this be * or np.multiply?
        product = np.fft.fft(data)*np.conj(fs)
        # apply frequency domain filter if desired
        if chirp_filter:
            product = np.multiply(product, Hamming(15E6, 25E6))
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

def decompress_sci_data(data, compression, presum, bps, SDI):
    """
    Decompress the science data according to the
    SHARAD interface specification.

    Input:
    -----------
        data: data to be decompressed
        compression: type of compression.
                     use 'static' or 'dynamic'
        presum: Onboard presumming parameter
        bps: Compression parameter
        SDI: compression parameter
    Output:
    -----------
        Decompressed data
    """

    #TODO: only static decompression is currently implemented
    #      implement dynamic decompression

    if compression == 'static' or compression == 'dynamic':
        if compression == 'static': # Static scaling
            L = np.ceil(np.log2(int(presum)))
            R = bps
            S = L - R + 8
            N = presum
            decomp = np.power(2, S) / N
            decompressed_data = data * decomp
        # dynamic currently disabled!
        elif compression == True:#dynamic scaling
            N = presum
            if SDI <= 5:
                S = SDI
            elif 5 < SDI <= 16:
                S = SDI - 6
            elif SDI > 16:
                S = SDI - 16
            decompressed_data = data * (np.power(2, S) / N)
        return decompressed_data
    else:
        # TODO: logging, should this be an exception?
        print('Decompression Error: Compression Type {}'
              ' not understood'.format(compression))
    return



def test_cmp_processor(infile, outdir, idx_start=None, idx_end=None,
                       taskname="TaskXXX", chrp_filt=True, verbose=False,
                       saving='hdf5'):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual tracks and returns pulse compressed data.

    Input:
    -----------
      infile    : Path to track file to be processed.
      outdir    : Path to directory to write output data
      idx_start : Start index for processing.
      idx_end   : End index for processing.
      chrp_filt : Apply a filter to the reference chirp
      verbose   : Gives feedback in the terminal.

    Output:
    -----------
      E          : Optimal E value
      cmp_pulses : Compressed pulses

    """

    try:
        time_start = time.time()
        # Get science data structure
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        # Load data
        science_path = infile.replace('_a.dat', '_s.dat')
        data = pds3.read_science(science_path, label_path, science=True)
        aux  = pds3.read_science(infile      , aux_path,   science=False)

        stamp1 = time.time() - time_start
        logging.debug("{:s}: data loaded in {:0.1f} seconds".format(taskname, stamp1))

        # Array of indices to be processed
        if idx_start is None or idx_end is None:
            idx_start = 0
            idx_end = len(data)
        idx = np.arange(idx_start, idx_end)

        logging.debug('{:s}: Length of track: {:d}'.format(taskname, len(idx)))

        # Chop raw data
        raw_data = chop_raw_data(data, idx, idx_start)

        compression = 'static' if (data['COMPRESSION_SELECTION'][idx_start] == 0) else 'dynamic'

        # Tracking presumming table. Converts TPS field into presumming count
        tps_table = (1, 2, 3, 4, 8, 16, 32, 64)
        tps = data['TRACKING_PRE_SUMMING'][idx_start]
        assert tps >= 0 and tps <= 7
        presum = tps_table[tps]
        SDI = data['SDI_BIT_FIELD'][idx_start]
        bps = 8

        # Decompress the data
        decompressed = decompress_sci_data(raw_data, compression, presum, bps, SDI)
        # TODO: E_track can just be a list of tuples
        E_track = np.empty((idx_end-idx_start, 2))
        # Get groundtrack distance and define 30 km chunks
        tlp = np.array(data['TLP_INTERPOLATE'][idx_start:idx_end])
        tlp0 = tlp[0]
        chunks = []
        i0 = 0
        for i in range(len(tlp)):
            if tlp[i] > tlp0 + 30:
                chunks.append([i0, i])
                i0 = i
                tlp0 = tlp[i]

        if len(chunks) == 0:
            chunks.append([0, idx_end-idx_start])

        if (tlp[-1] - tlp[i0]) >= 15:
            chunks.append([i0, idx_end-idx_start])
        else:
            chunks[-1][1] = idx_end - idx_start

        logging.debug('{:s}: Made {:d} chunks'.format(taskname, len(chunks)))
        # Compress the data chunkwise and reconstruct


        list_cmp_track = []
        for i, chunk in enumerate(chunks):
            start, end = chunks[i]

            #check if ionospheric correction is needed
            iono_check = np.where(aux['SOLAR_ZENITH_ANGLE'][start:end] < 100)[0]
            b_iono = len(iono_check) != 0
            minsza = min(aux['SOLAR_ZENITH_ANGLE'][start:end])
            logging.debug('{:s}: chunk {:03d}/{:03d} Minimum SZA: {:6.2f} '
                          ' Ionospheric Correction: {!r}'.format(
                          taskname, i, len(chunks), minsza, b_iono))

            E, sigma, cmp_data = us_rng_cmp(decompressed[start:end],
                                            chirp_filter=chrp_filt,
                                            iono=b_iono, debug=verbose)
            list_cmp_track.append(cmp_data)
            E_track[start:end, 0] = E
            E_track[start:end, 1] = sigma

        cmp_track = np.vstack(list_cmp_track)
        list_cmp_track = None  # free memory


        stamp3 = time.time() - time_start - stamp1
        logging.debug('{:s} Data compressed in {:0.2f} seconds'.format(taskname, stamp3))

        if saving and outdir != "":
            data_file   = os.path.basename(infile)
            outfilebase = data_file.replace('.dat', '.h5')
            outfile     = os.path.join(outdir, outfilebase)

            logging.debug('{:s}: Saving to {:s}'.format(taskname, outdir))
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # restructure of data save
            real = np.array(np.round(cmp_track.real), dtype=np.int16)
            imag = np.array(np.round(cmp_track.imag), dtype=np.int16)
            cmp_track = None # free memory
            if saving == 'hdf5':
                pd.DataFrame(real).to_hdf(outfile, key='real', complib='blosc:lz4', complevel=6)
                pd.DataFrame(imag).to_hdf(outfile, key='imag', complib='blosc:lz4', complevel=6)
                logging.info("{:s}: Wrote {:s}".format(taskname, outfile))
            elif saving == 'npy':
                # Round it just like in an hdf5 and save as side-by-side arrays
                cmp_track = np.vstack([real, imag])
                basename = data_file.replace('.dat', '.npy')
                outfile = os.path.join(outdir, basename)
                np.save(outfile, cmp_track)
                logging.info("{:s}: Wrote {:s}".format(taskname, outfile))
            elif saving == 'none':
                pass
            else:
                logging.error("{:s}: Unrecognized output format '{:s}'".format(taskname, saving))

            basename = data_file.replace('.dat', '_TECU.txt')
            outfile_TECU = os.path.join(outdir, basename)
            np.savetxt(outfile_TECU, E_track)
            logging.info("{:s}: Wrote {:s}".format(taskname, outfile_TECU))


    except Exception as e:

        logging.error('{:s}: Error processing file {:s}'.format(taskname, infile))
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line))
        return 1
    logging.info('{:s}: Success processing file {:s}'.format(taskname, infile))
    return 0

def chop_raw_data(data, idx, idx_start):
    raw_data = np.zeros((len(idx), 3600), dtype=np.complex)
    for j in range(3600):
        k = 'sample' + str(j)
        raw_data[:, j] = data[k][idx].values
    return raw_data


def main():

    parser = argparse.ArgumentParser(description='Range compression library')
    parser.add_argument('-o', '--output', default='./rng_cmp_data',
                        help="Output base directory")
    parser.add_argument('--ofmt', default='npy',
                        choices=('hdf5', 'npy', 'none'),
                        help="Output file format")

    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default=None, #"elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="rng_cmp: [%(levelname)-7s] %(message)s")


    # Read lookup table associating gob's with tracks
    if args.tracklist is None:
        root = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR'
        lookup = (
            root + '/mrosh_0001/data/edr03xxx/edr0336603/e_0336603_001_ss19_700_a_a.dat',
            root + '/mrosh_0004/data/rm286/edr5272901/e_5272901_001_ss19_700_a_a.dat',
            root + '/mrosh_0004/data/rm278/edr5116601/e_5116601_001_ss19_700_a_a.dat',
            root + '/mrosh_0004/data/rm184/edr3434001/e_3434001_001_ss19_700_a_a.dat',
        )
    else:
        lookup = np.genfromtxt(args.tracklist, dtype='str')

    # Build list of processes
    logging.info("Building task list for test")
    process_list = []
    path_outroot = args.output

    logging.debug("Base output directory: " + path_outroot)
    for i, infile in enumerate(lookup):
        path_file = infile.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = os.path.basename(path_file)
        path_file = os.path.dirname(path_file)
        outdir    = os.path.join(path_outroot, path_file, 'ion')

        logging.debug("Adding " + infile)
        process_list.append([infile, outdir, None, None, "Task{:03d}".format(i+1)])

    if args.maxtracks > 0:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)))

    start_time = time.time()
    named_params = {'saving':args.ofmt, 
                    'chrp_filt':True, 'verbose':args.verbose}
    # Single processing (for profiling)
    for t in process_list:
        test_cmp_processor(*t, **named_params)
    logging.info("Done in {:0.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    # execute and import only if run as a script
    import argparse
    import sys
    import logging
    import os
    import time
    import traceback

    import pandas as pd

    import pds3lbl as pds3


    main()

