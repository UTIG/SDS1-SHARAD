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

CODEPATH = os.path.dirname(__file__)
p1 = os.path.abspath(os.path.join(CODEPATH, "../xlib"))
sys.path.insert(1, p1)

#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting
from cmp.rng_cmp import read_and_chunk_radar, compress_chunks, save_cmp_files
import misc.fileproc as fileproc


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



def cmp_processor(infile, outfiles: Dict[str, str], sharad_root: str, idx_start=None, idx_end=None, taskname="TaskXXX",
                  radargram=False, chrp_filt=True, verbose=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual tracks and returns pulse compressed data.

    Input:
    -----------
      infile    : Path to track file to be processed.
      outfiles  : A dictionary with keys 'cmp_h5' and 'cmp_tecu' giving paths to files to create
                  If outfiles is None, don't save any files
      sharad_root : SHARAD EDR data repository root path
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
    try:
        # Get science data structure
        label_path = os.path.join(sharad_root, 'EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(sharad_root, 'EDR/mrosh_0004/label/auxiliary.fmt')

        decompressed, decomp, chunks, aux, idx_start, idx_end = read_and_chunk_radar(infile, label_path, aux_path, idx_start, idx_end, taskname)

        cmp_track, E_track = compress_chunks(decompressed, decomp, chunks, aux, chrp_filt, verbose, idx_start, idx_end, taskname)

        if outfiles is not None:
            save_cmp_files(outfiles['cmp_h5'], outfiles['cmp_tecu'], cmp_track, E_track, taskname)

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




def add_standard_args(parser, script=None):
    """ Add standard script arguments to the args list """

    if script in ['rsr', 'srf']:
        parser.add_argument('orbits', metavar='orbit', nargs='*',
                            help='SHARAD orbit/product IDs to process.'
                            'If "all", processes all orbits')
    else:
        parser.add_argument('product_ids', nargs='*',
                            help='SHARAD orbit/product IDs to process')

    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")

    parser.add_argument('-o', '--output', default=None,
                        help="Output base directory")

    parser.add_argument('--tracklist', '--orbitlist', default=None,#"elysium.txt",
                        help="List of track data files or product IDs to process")

    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")

    if script != 'pipeline':
        # pipeline has its own mutually exclusive group for verbose
        parser.add_argument('-v', '--verbose', action="store_true",
                            help="Display verbose output")
        # it also has ignoretimes (which we haven't changed over to 'overwrite' yet)
        parser.add_argument('--overwrite',  action="store_true",
                            help="Overwrite outputs even if they exist")

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
            'sharad_root': sharad_root,
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

def should_process_products(product_id, infiles: Dict[str, str], outfiles: Dict[str, str], overwrite: bool, loglevel=logging.INFO):
    filestatus = fileproc.file_processing_status(infiles.values(), outfiles.values())
    if (not overwrite) and filestatus[1] == 'output_ok':
        logging.log(loglevel, "Not adding %s to jobs. output is up to date", product_id)
        return False
    elif filestatus[0] == 'input_missing':
        logging.log(loglevel, "Not adding %s to jobs. One or more inputs is missing", product_id)
        return False
    return True



def run_jobs(f_processor: Callable, jobs: List[Dict[str, str]], nb_cores: int):
    logging.info("Start processing %d tracks", len(jobs))
    start_time = time.time()

    if nb_cores <= 1 or len(jobs) <= 1:
        # Single processing (for profiling)
        for ii, t in enumerate(jobs, start=1):
            f_processor(**t)
            logging.info("Finished task %d of %d", ii, len(jobs))
    else:
        # non-reentrant hack but hopefully that's ok
        global F_PROCESSOR
        F_PROCESSOR = f_processor
        with multiprocessing.Pool(nb_cores) as pool:
            gen_jobs = zip(range(len(jobs)), jobs)
            for ii, res in enumerate(pool.imap_unordered(f_processor_mp, gen_jobs), start=1):
                logging.info("Finished task %d (%d of %d remaining)",
                             res[0]+1, len(jobs) - ii, len(jobs))

    logging.info("Done in %0.2f seconds", time.time() - start_time)

def f_processor_mp(t):
    return (t[0], F_PROCESSOR(**t[1]))

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
