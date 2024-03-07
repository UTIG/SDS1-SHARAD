#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu',
               'Kirk Scanlan, kirk.scanlan@gmail.com']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'August 15 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'August 23 2018',
         'info': 'Added pulse decompression and 6bit data'},
    '1.2':
        {'date': 'October 16 2018',
         'info': 'Modified data saving method from .npy to .h5'}}

import sys
import os
import time
import logging
import argparse
import warnings
import multiprocessing
import traceback
import importlib.util
import numpy as np
#from scipy.optimize import curve_fit
import pandas as pd

# TODO: make this import more robust to allow this script
# TODO: to be run from outside the SHARAD directory
sys.path.insert(0, '../xlib')
#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting
import cmp.rng_cmp

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



def cmp_processor(infile, outdir, idx_start=None, idx_end=None, taskname="TaskXXX", radargram=True,
                  chrp_filt=True, verbose=False, saving=False, SDS=None):
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
    assert SDS is not None
    try:
    #if chrp_filt:
        time_start = time.time()
        # Get science data structure
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')
        # Load data
        science_path = infile.replace('_a.dat', '_s.dat')
        data = pds3.read_science(science_path, label_path, science=True)
        aux  = pds3.read_science(infile      , aux_path,   science=False)

        stamp1 = time.time()-time_start
        logging.debug("%s: data loaded in %0.1f seconds", taskname, stamp1)

        # Array of indices to be processed
        if idx_start is None or idx_end is None:
            idx_start, idx_end = 0, len(data)
        idx = np.arange(idx_start, idx_end)

        logging.debug('%s: Length of track: %d', taskname, len(idx))

        # Chop raw data
        nsamples = 3600
        raw_data = np.empty((len(idx), nsamples), dtype=np.complex128)
        for j in range(nsamples):
            raw_data[idx-idx_start, j] = data['sample'+str(j)][idx].values

        if data['COMPRESSION_SELECTION'][idx_start] == 0:
            compression = 'static'
        else:
            compression = 'dynamic'
        tps = data['TRACKING_PRE_SUMMING'][idx_start]
        presum_table = (1, 2, 3, 4, 8, 16, 32, 64)
        assert 0 <= tps <= 7
        presum = presum_table[tps]
        sdi = data['SDI_BIT_FIELD'][idx_start]
        bps = 8

        # Decompress the data
        decompressed = cmp.rng_cmp.decompress_sci_data(raw_data, compression, presum, bps, sdi)
        del raw_data
        E_track = np.empty((idx_end-idx_start, 2))
        # Get groundtrack distance and define 30 km chunks
        tlp = np.array(data['TLP_INTERPOLATE'][idx_start:idx_end])
        chunks = calculate_chunks(tlp, 30.)
        logging.debug('%s: chunked into %d pieces', taskname, len(chunks))
        cmp_track = np.empty(decompressed.shape + (2,), dtype=np.int16) # real and imaginary
        # Compress the data chunkwise and reconstruct
        for i, (start, end) in enumerate(chunks):
            #check if ionospheric correction is needed
            iono_check = np.where(aux['SOLAR_ZENITH_ANGLE'][start:end]<100)[0]
            b_iono = len(iono_check) != 0
            minsza = min(aux['SOLAR_ZENITH_ANGLE'][start:end])
            logging.debug('%s: chunk %03d/%03d Minimum SZA: %0.3f  Ionospheric Correction: %r',
                taskname, i, len(chunks), minsza, b_iono)

            # GNG: These concats seem relatively expensive.
            E, sigma, cmp_data = cmp.rng_cmp.us_rng_cmp(decompressed[start:end],\
                                 chirp_filter=chrp_filt, iono=b_iono, debug=verbose)

            cmp_track[start:end, :, 0] = np.round(cmp_data.real)
            cmp_track[start:end, :, 1] = np.round(cmp_data.imag)
            E_track[start:end, 0] = E
            E_track[start:end, 1] = sigma

        stamp3 = time.time() - time_start - stamp1
        logging.debug('%s Data compressed in %0.2f seconds', taskname, stamp3)

        if saving:
            outfile, outfile_tecu = output_filenames(infile, outdir)
            logging.debug('%s: Saving to folder: %s', taskname,outdir)
            os.makedirs(outdir, exist_ok=True)

            # restructure of data save
            dfreal = pd.DataFrame(cmp_track[:, :, 0])
            dfreal.to_hdf(outfile, key='real', complib='blosc:lz4', complevel=6)
            del dfreal
            dfimag = pd.DataFrame(cmp_track[:, :, 1])
            dfimag.to_hdf(outfile, key='imag', complib='blosc:lz4', complevel=6)
            del dfimag
            #np.savez(outfile + "_int16.npz", real=cmp_track[:, :, 0], imag=cmp_track[:, :, 1])
            np.savetxt(outfile_tecu, E_track)

        if radargram:
            # Plot a radargram
            rx_window_start = data['RECEIVE_WINDOW_OPENING_TIME'][idx]
            tx0 = data['RECEIVE_WINDOW_OPENING_TIME'][0]
            tx = np.empty(len(data))
            # GNG: this seems likely to be wrong. Should be:
            #for rec in range(len(data['RECEIVE_WINDOW_OPENING_TIME'])):
            for rec in range(len(data)):
                tx[rec] = data['RECEIVE_WINDOW_OPENING_TIME'][rec]-tx0
            cmp.plotting.plot_radargram(cmp_track, tx, samples=nsamples)

    except Exception: # pylint: disable=W0703
        logging.error('%s: Error processing file %s', taskname, infile)
        for line in traceback.format_exc().split("\n"):
            logging.error('%s: %s', taskname, line)
        return 1
    logging.info('%s: Success processing file %s', taskname, infile)
    return 0

def calculate_chunks(tlp, chunklen_km):
    chunks = []
    tlp0 = tlp[0]
    i0 = 0
    for i, tlp1 in enumerate(tlp):
        if tlp1 > tlp0 + chunklen_km:
            chunks.append([i0, i])
            i0 = i
            tlp0 = tlp1
    if not chunks: # if chunks is empty
        chunks = [0, len(tlp)]
    elif (tlp[-1]-tlp[i0]) >= (chunklen_km/2.):
        chunks.append([i0, len(tlp)]) # add a new chunk
    else: # append this chunk to the previous chunk
        chunks[-1][1] = len(tlp)
    return chunks #chunks = np.array(chunks)



def output_filenames(infile: str, outdir: str):
    data_file   = os.path.basename(infile)
    outfilebase = data_file.replace('_a.dat', '_s.h5')
    outfile     = os.path.join(outdir, outfilebase)
    tecubase = data_file.replace('_a.dat','_s_TECU.txt')
    outfile_tecu = os.path.join(outdir, tecubase)
    return outfile, outfile_tecu

def all_outputs_updated(infile: str, outdir: str):
    """ For now, just check that outputs exist.
    Later, we can calculate whether the modified timestamps
    are also correct """
    for outfile in output_filenames(infile, outdir):
        if not os.path.exists(outfile):
            return False
        logging.debug("%s exists", outfile)
    return True


def main():
    parser = argparse.ArgumentParser(description='Run range compression')
    parser.add_argument('-o', '--output', default=None,
                        help="Output base directory")

    parser.add_argument('--overwrite',  action="store_true",
                        help="Overwrite outputs even if they exist")
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")
    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                        help="Root directory (default: environment variable SDS)")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_rng_cmp: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/cmp')

    # Set number of cores
    nb_cores = args.jobs

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    with open(args.tracklist, 'r') as fin:
        lookup = [line.strip() for line in fin if line.strip() != '']
    logging.debug(lookup)

    # Build list of processes
    logging.info("Building task list")
    process_list = []
    path_outroot = args.output
    path_inroot = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')

    logging.debug("Base output directory: %s", path_outroot)

    for i, infile in enumerate(lookup):
        # check if file has already been processed
        path_file = os.path.dirname(os.path.relpath(infile, path_inroot))
        outdir    = os.path.join(path_outroot, path_file, 'ion')

        if (not args.overwrite) and all_outputs_updated(infile, outdir):
            logging.debug('File already processed. Skipping ' + infile)
            continue
        logging.debug("Adding %s", infile)
        process_list.append([infile, outdir, None, None, "Task{:03d}".format(i+1)])

    #h5file.close()
    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing %d tracks", len(process_list))

    start_time = time.time()

    named_params = {'saving':True, 'chrp_filt':True, 'verbose':args.verbose, 'radargram':False, 'SDS': args.SDS}

    if nb_cores <= 1:
        # Single processing (for profiling)
        for t in process_list:
            cmp_processor(*t, **named_params)
    else:
        # Multiprocessing
        with multiprocessing.Pool(nb_cores) as pool:
            results = [pool.apply_async(cmp_processor, t, \
                       named_params) for t in process_list]

            for i, result in enumerate(results):
                dummy = result.get()
                if dummy == 1:
                    logging.error("%s: Problem running pulse compression",process_list[i][4])
                else:
                    logging.info("%s: Finished pulse compression", process_list[i][4])

    logging.info("Done in %0.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    # execute only if run as a script
    main()
