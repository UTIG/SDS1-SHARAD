#!/usr/bin/env python3

"""
Extracted cluttergram production
"""

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'April 15 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}

import sys
import os
import multiprocessing
import time
import logging
import argparse
import traceback
from typing import List
from pathlib import Path

import numpy as np
import spiceypy as spice
import pandas as pd
import matplotlib.pyplot as plt

from run_rng_cmp import run_jobs, process_product_args,\
                        should_process_products, add_standard_args, filename_to_productid

sys.path.append('../xlib')
#import misc.prog as prog
#import misc.hdf as hdf

import rng.icd as icd
from rng.icsim import calc_roi
from SHARADEnv import SHARADEnv, SHARADFiles

def main():
    global args
    parser = argparse.ArgumentParser(description='Run SHARAD cluttergrams')
    #parser.add_argument('--qcdir', help="Quality control output directory")
    #parser.add_argument('--ofmt', default='hdf5', choices=('hdf5','csv','none'),
    #                    help="Output file format")
    parser.add_argument('--maxechoes', type=int, default=0, help="Maximum number of echoes to process in a track")

    add_standard_args(parser)

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/rng')

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_clutter: [%(levelname)-7s] %(message)s")

    # Build list of processes
    logging.info('build task list')
    # File location calculator
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist = process_product_args_ranging(args.product_ids, args.tracklist, sfiles)
    logging.debug(repr(productlist))
    process_list = []
    for tasknum, (product_id, idx_start, idx_end) in enumerate(productlist, start=1):
        idx_start1, idx_end1, _ = calc_roi(idx_start, idx_end, args.maxechoes)
        infiles = sfiles.product_paths('edr', product_id)
        #infiles.update(sfiles.product_paths('cmp', product_id))
        outfiles = sfiles.product_paths('clu', product_id, start=idx_start1, end=idx_end1)

        if not should_process_products(product_id, infiles, outfiles, args.overwrite):
            continue

        process_list.append({
            'inpath' : infiles['edr_aux'],
            'clutterfile': outfiles['clu_rad'],
            'product_id': product_id,
            'sfiles': sfiles,
            'tasknum': tasknum,
            'maxechoes': args.maxechoes,
            'idx_start': idx_start,
            'idx_end': idx_end,
            'SDS': args.SDS,
            })
        logging.debug("[%d] %s", tasknum, str(process_list[-1]))
        tasknum += 1

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        return

    run_jobs(process_clutter, process_list, args.jobs)

def read_tracklist_ranging(tracklistfile: str):
    """ Read a ranging tracklist, which differs from a regular track list in that it
    must contain two additional fields: index values that limit the range to be processed """
    count = 0
    with open(tracklistfile, 'rt') as flist:
        for line in flist:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            path, idx_start, idx_end = line.split()
            yield path.strip(), int(idx_start), int(idx_end)
            count += 1

    #assert count % 2 == 0, "%s doesn't contain an even number of tracks"

def process_product_args_ranging(product_ids: List[str], tracklistfile: str, sfiles: SHARADFiles):
    """ Assemble list of desired product IDs from the product ID argument and
    the file of tracks to process
    product IDs must be a separated list of product ID, number, number"""
    productlist = []
    if product_ids:
        for arg in product_ids:
            try:
                product_id, idx_start, idx_end = arg.split(',')
            except ValueError:
                raise ValueError("Expected argument of form 'product_id,idx_start,idx_end'")
            productlist.append((product_id, int(idx_start), int(idx_end)))

    # Add tracks from file
    logging.debug("tracklistfile=%s", tracklistfile)
    if tracklistfile is not None:
        for filename, idx_start, idx_end in read_tracklist_ranging(tracklistfile):
            productlist.append((filename_to_productid(filename), idx_start, idx_end))

    f_unknown = lambda p: p[0] not in sfiles.product_id_index
    unknown_product_ids = list(filter(f_unknown, productlist))
    if unknown_product_ids:
        s = ' '.join([p[0] for p in unknown_product_ids])
        raise KeyError("Unknown product IDs %s" % s)

    return productlist

def process_clutter(inpath, clutterfile, product_id: str, sfiles: SHARADFiles, SDS: str, idx_start=None, idx_end=None, tasknum=0,
                maxechoes=0, debug=False):

    """
    TODO: describe input and output parameters
    inpath : path to source data file, relative to EDR path root
    idx_start, idx_end: index range of track to process
    save_format: Unused?
    tasknum: Task number to uniquely identify this job for multiprocessing
    clutterfile: path to clutter simulation file, if already completed. None = run clutter sim
    b_noprogress: if True, don't print status progress bar
    bplot: if True, show qc plots
    maxechoes: ?

    output:
    If an error, a tuple with None
    """

    try:
        taskname = "task{:03d}".format(tasknum)
        dtm_path = os.path.join(SDS, 'orig/supl/hrsc/MC11E11_HRDTMSP.dt5.tiff')
        assert os.path.exists(dtm_path)
        #cmp_path = sfiles.cmp_product_paths(product_id)['cmp_rad']
        edr_paths = sfiles.edr_product_paths(product_id)
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_label = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

        science_path = edr_paths['edr_sci'] #inpath.replace('_a.dat','_s.dat')
        #co_data = 10
        # TODO: convert idx_start, idx_end to slice
        sim = icd.gen_cluttergram(clutterfile, dtm_path, science_path, label_path,
                inpath, aux_label,
                idx_start, idx_end, debug=debug,
                ipl=True,
                do_progress=False, maxechoes=maxechoes)


    except Exception:
        taskname = "task{:03d}".format(tasknum)
        for line in traceback.format_exc().split("\n"):
            logging.error('%s: %s', taskname, line)

if __name__ == "__main__":
    # execute only if run as a script
    main()

