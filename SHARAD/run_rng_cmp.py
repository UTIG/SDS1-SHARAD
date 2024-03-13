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

import re
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
from typing import Dict, List, Tuple, Callable

from SHARADEnv import SHARADFiles
#from run_altimetry import read_tracklistfile, filename_to_productid

CODEPATH = os.path.dirname(__file__)
p1 = os.path.abspath(os.path.join(CODEPATH, "../xlib"))
sys.path.insert(1, p1)

#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting
import cmp.rng_cmp
import misc.fileproc as fileproc


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



def cmp_processor(infile, outfiles: Dict[str, str], SDS: str, idx_start=None, idx_end=None, taskname="TaskXXX",
                  radargram=True, chrp_filt=True, verbose=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual tracks and returns pulse compressed data.

    Input:
    -----------
      infile    : Path to track file to be processed.
      outfiles  : A dictionary with keys 'cmp_h5' and 'cmp_tecu' giving paths to files to create
                  If outfiles is None, don't save any files
      SDS       : SDS root path
      idx_start : Start index for processing.
      idx_end   : End index for processing.
      chrp_filt : Apply a filter to the reference chirp
      radargram : Plot radargrams interactively
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

        if outfiles is not None:
            # Assume these two files are being put into the same directory
            outdir = os.path.dirname(outfiles['cmp_h5'])
            logging.debug('%s: Saving to folder: %s', taskname, outdir)
            os.makedirs(outdir, exist_ok=True)

            # restructure of data save
            dfreal = pd.DataFrame(cmp_track[:, :, 0])
            dfreal.to_hdf(outfiles['cmp_h5'], key='real', complib='blosc:lz4', complevel=6)
            del dfreal
            dfimag = pd.DataFrame(cmp_track[:, :, 1])
            dfimag.to_hdf(outfiles['cmp_h5'], key='imag', complib='blosc:lz4', complevel=6)
            del dfimag
            #np.savez(outfile + "_int16.npz", real=cmp_track[:, :, 0], imag=cmp_track[:, :, 1])
            np.savetxt(outfiles['cmp_tecu'], E_track)

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



def add_standard_args(parser, script=None):
    """ Add standard script arguments to the args list """

    if script not in ['rsr', 'srf']:
        # These scripts haven't been updated for these args
        parser.add_argument('product_ids', nargs='*',
                            help='SHARAD product IDs to process')
        parser.add_argument('-o', '--output', default=None,
                            help="Output base directory")
        parser.add_argument('--overwrite',  action="store_true",
                            help="Overwrite outputs even if they exist")
        parser.add_argument('--tracklist', default=None,#"elysium.txt",
                            help="List of track data files or product IDs to process")
        parser.add_argument('--maxtracks', type=int, default=0,
                            help="Maximum number of tracks to process")

    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                        help="Root directory (default: environment variable SDS)")


def main():
    parser = argparse.ArgumentParser(description='Run range compression')

    add_standard_args(parser)
    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_rng_cmp: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')


    # Build list of processes
    logging.debug("Building task list")
    # File location calculator
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist = process_product_args(args.product_ids, args.tracklist, sfiles)
    assert productlist, "No files to process"


    process_list = []
    for tasknum, product_id in enumerate(productlist, start=1):
        infiles = sfiles.product_paths('edr', product_id)
        outfiles = sfiles.product_paths('cmp', product_id)

        if not should_process_products(product_id, infiles,  outfiles, args.overwrite):
            continue

        logging.debug("Adding %s", product_id)
        process_list.append({
            'infile': infiles['edr_aux'],
            'outfiles': outfiles,
            'SDS': args.SDS,
            'idx_start': None,
            'idx_end': None,
            'taskname': "Task{:03d}".format(tasknum),
            'chrp_filt': True,
            'verbose': args.verbose,
            'radargram': False,
        })


    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        return 0

    run_jobs(cmp_processor, process_list, args.jobs)



def process_product_args(product_ids: List[str], tracklistfile: str, sfiles: SHARADFiles):
    """ Assemble a list of desired product IDs from the
    product ID argument and the file of tracks to process """
    productlist = []
    # Add tracks from command line
    if product_ids:
        productlist += product_ids

    # Add tracks from file
    if tracklistfile is not None:
        productids = map(filename_to_productid, read_tracklistfile(tracklistfile))
        productlist += list(productids)

    f_unknown = lambda p: p not in sfiles.product_id_index
    unknown_product_ids = list(filter(f_unknown, productlist))
    if unknown_product_ids:
        s = ' '.join(unknown_product_ids)
        raise KeyError("Unknown product IDs %s" % s)

    return productlist

def should_process_products(product_id, infiles: Dict[str, str], outfiles: Dict[str, str], overwrite: bool):
    filestatus = fileproc.file_processing_status(infiles.values(), outfiles.values())
    if (not overwrite) and filestatus[1] == 'output_ok':
        logging.info("Not adding %s to jobs. output is up to date", product_id)
        return False
    elif filestatus[0] == 'input_missing':
        logging.info("Not adding %s to jobs. One or more inputs is missing", product_id)
        return False
    return True



def run_jobs(f_processor: Callable, jobs: List[Dict[str, str]], nb_cores: int):
    logging.info("Start processing %d tracks", len(jobs))
    start_time = time.time()

    if nb_cores <= 1 or len(jobs) <= 1:
        # Single processing (for profiling)
        for t in jobs:
            f_processor(**t)
    else:
        # Multiprocessing
        with multiprocessing.Pool(nb_cores) as pool:
            results = [pool.apply_async(f_processor, [], t) for t in jobs]

            for i, result in enumerate(results, start=1):
                _ = result.get()
                logging.info("Finished task %d of %d", i, len(jobs))
                #if dummy == 1:
                #    logging.error("%s: Problem running pulse compression",process_list[i][4])
                #else:
                #    logging.info("%s: Finished pulse compression", process_list[i][4])

    logging.info("Done in %0.2f seconds", time.time() - start_time)

def read_tracklistfile(trackfile: str):
    """ Read a track list and return data structures about desired products
    TODO: unify with the read_tracklist functions in other code
    """
    with open(trackfile, 'rt') as flist:
        for linenum, path in enumerate(flist, start=1):
            path = path.strip()
            if not path or path.startswith('#'):
                continue
            yield path

def filename_to_productid(filename: str):
    """ Converts a filename to a productid.  If the
    string is already a product ID, then pass it through
    without complaining """
    if re.match(r'^e_\d{7}_\d{3}_\w+_\d+_\w$', filename):
        return filename # It's a product ID
    name = os.path.basename(filename)
    assert name.startswith('e_'), "Unexpected filename %s" % filename
    assert name.endswith('_a.dat'), "Unexpected filename %s" % filename
    return name.replace('_a.dat', '')



if __name__ == "__main__":
    # execute only if run as a script
    main()
