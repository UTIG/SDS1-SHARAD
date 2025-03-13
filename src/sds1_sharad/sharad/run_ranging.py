#!/usr/bin/env python3

""" 
run_ranging uses the common argument processing system,
with the exception that the command line orbit names require
trace numbers to describe the first and last trace to be processed,
as is required in run_clutter.
The tracklist format similarly requires a space-delimited field
similarly of the orbit name or data file name, the and start and
end trace numbers.

TODO: unify this so the track file also requires comma-separated.
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
import json
from pathlib import Path
from typing import List

import numpy as np
import spiceypy as spice
import pandas as pd
import matplotlib.pyplot as plt

p1 = Path(__file__).parent
sys.path.insert(1, str(p1.resolve()))

from run_rng_cmp import run_jobs,\
                        should_process_products, add_standard_args

from run_clutter import process_product_args_ranging

sys.path.append('../xlib')

from sharad.sharadenv import SHARADFiles

import rng.icd as icd

def main():
    parser = argparse.ArgumentParser(description='Run SHARAD ranging processing')
    parser.add_argument('--qcdir', help="Quality control output directory")
    parser.add_argument('--noprogress', action="store_true", help="don't show progress")
    parser.add_argument('--maxechoes', type=int, default=0, help="Maximum number of echoes to process in a track")
    parser.add_argument('--showqc', action='store_true', help='Show QC plots interactively')

    add_standard_args(parser)

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/rng')

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_ranging: [%(levelname)-7s] %(message)s")

    # Build list of processes
    logging.info('build task list')

    # File location calculator 
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist = process_product_args_ranging(args.product_ids, args.tracklist, sfiles)
    process_list = []

    tasknum = 1
    b_noprogress = True if args.noprogress else False
    all_tasks = []

    for tasknum, (product_id, idx_start, idx_end) in enumerate(productlist, start=1):
        if args.maxtracks > 0 and len(all_tasks) >= args.maxtracks:
            break
        logging.debug("pid: %s", product_id)
        infiles = sfiles.product_paths('edr', product_id)
        infiles.update(sfiles.product_paths('cmp', product_id))
        infiles.update(sfiles.product_paths('clu', product_id))
        outfiles = sfiles.product_paths('rng', product_id)

        b_process = should_process_products(product_id, infiles, outfiles, args.overwrite)

        clutterfile = infiles['clu_rad']

        task = {
            'infiles': infiles,
            'outfiles': outfiles,
            'idx_start' : idx_start,
            'idx_end' : idx_end,
            'SDS': args.SDS,
            'tasknum': tasknum,
            'b_noprogress': b_noprogress,
            'maxechoes': args.maxechoes,
        }
        all_tasks.append(task)
        if b_process:
            process_list.append(task)


    if len(all_tasks) % 2 != 0:
        raise ValueError("Must have an even number of tracks for crossovers!")

    if args.dryrun:
        logging.debug("process_list: %r", process_list)
        return

    # We could return the results but save intermediate product anyways
    run_jobs(process_rng, process_list, args.jobs)


    """When using multiprocessing the
    results are returned in random order. Since always two tracks have to be compared
    per crossover I found it convenient for testing if the output is listed in the
    same order as the input, although this is not strictly necessary. But the first
    two tracks are then x-over 1, track 3 and 4 x-over 2, and so on... The error
    seems to happen if an error at some track occurs. Then he cannot find it in the
    input.

    Why these specific tracks are not going through needs further investigation. There
    seems to be some array length issue.
    """
    rngfiles = [params['outfiles']['rng_json'] for params in process_list]
    crossover_processing(rngfiles, args.qcdir, args.showqc)

def crossover_processing(rngfiles: List[str], qcdir: str, showqc: bool):
    logging.info("Running crossover processing")
    logging.debug("starting with rng files: %r", rngfiles)
    out = np.zeros((len(rngfiles), 2))
    # Sort results
    try:
        for ii, rngjson in enumerate(rngfiles):
            with open(rngjson, 'rt') as fin:
                result = json.load(fin)
                out[ii, :] = result['result'][0:2]
    except:
        logging.error("Can't do crossover processing. Not all tasks have succeeded")
        raise

    delta_ranges = out[0::2, 1] - out[1::2, 1]
    logging.debug("rms=%0.4f", np.nanstd(delta_ranges))

    import matplotlib
    font = {'family' : 'serif',
            'size'   : 24}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.scatter(np.arange(0,len(delta_ranges)), delta_ranges, s=30)
    ax.set_ylim(-150, 150)
    ax.set_xlabel('Xover Number')
    ax.set_ylabel('Ranging Residual [m]')
    plt.grid()
    plt.tight_layout()

    if qcdir:
        # Save debugging outputs to qc directory if requested
        qcplotfile = os.path.join(qcdir, 'ranging_result.pdf')
        logging.info("Saving qc to %s", qcplotfile)
        os.makedirs(qcdir, exist_ok=True)
        plt.savefig(qcplotfile)

    if showqc:
        plt.show()


def process_rng(infiles, outfiles, SDS: str, idx_start=None, idx_end=None, tasknum=0,
                b_noprogress=False, bplot=False, maxechoes=0):

    """
    TODO: describe input and output parameters
    infiles : dictionary of products required as inputs
    outfiles : dictionary of output product file(s)
    idx_start, idx_end: index range of track to process
    tasknum: Task number to uniquely identify this job for multiprocessing
    b_noprogress: if True, don't print status progress bar
    bplot: if True, show qc plots
    maxechoes: ?

    output:
    If an error, a tuple with None
    """

    try:
        taskname = "task{:03d}".format(tasknum)
        # Relative path to this file
        inpath = infiles['edr_aux']
        fname = os.path.basename(inpath)
        product_id = Path(infiles['edr_lbl']).stem
        # Relative directory of this file
        cmp_path = infiles['cmp_rad']
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_label = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

        science_path = infiles['edr_sci']
        clutterfile = infiles['clu_rad']

        logging.info("%s: Reading %s", taskname, cmp_path)
        co_data = 10


        # Load cluttergram
        sim = None
        if clutterfile.endswith('.npz'):
            with np.load(clutterfile) as data:
                sim = data['sim']
            # TODO: assert idx_start, idx_end match in json requested
        elif clutterfile.endswith('.npy'):
            sim = np.load(clutterfile)
        assert sim is not None, "Don't know how to read %s" % (clutterfile,)

        # After we get memory usage under control and start memmapping
        # the radargrams, we should be able to start dispatching
        # coregistration jobs
        
        result = np.zeros((20, 3))
        for co_sim in range(12, 25):
            logging.debug("%s: icd_ranging(co_sim=%d)", taskname, co_sim)

            # icd_ranging_cg3 returns delta, dz, min(md)
            result[co_sim-5] = icd.icd_ranging_cg3\
                              (cmp_path, None, science_path, label_path, \
                               inpath, aux_label,
                               idx_start, idx_end, debug=False,
                               ipl=True, co_sim=co_sim, co_data=co_data,
                               window=50, sim=sim,
                               do_progress=not b_noprogress, maxechoes=maxechoes)
            j = co_sim - 5
            logging.info("%s: result[%d] = %f %f %f", taskname, j, \
               result[j][0], result[j][1], result[j][2])

        if bplot:
            plt.scatter(np.arange(1,25,1), result[:,0], s=30)
            plt.title('process_rng result[:, 0] for {:s}'.format(taskname))
            plt.show()
            plt.scatter(np.arange(1,25,1), result[:,2], s=30)
            plt.title('process_rng result[:, 2] for {:s}'.format(taskname))
            plt.show()
        amin = np.argmin(result[:,2])
        min_result = result[amin]
        logging.info("min=%d", amin)

        res1 = {
            'inpath': inpath,
            'product_id': product_id,
            'tasknum': tasknum,
            'result': result[amin].tolist(),
            'all_results': result.tolist(),
        }
        os.makedirs(os.path.dirname(outfiles['rng_json']), exist_ok=True)
        with open(outfiles['rng_json'], 'wt') as fhout:
            json.dump(res1, fhout) #, indent="\t")
        logging.info("Wrote %s", outfiles['rng_json'])
        return res1

    except Exception:
        taskname = "task{:03d}".format(tasknum)
        for line in traceback.format_exc().split("\n"):
            logging.error('%s: %s', taskname, line)

if __name__ == "__main__":
    # execute only if run as a script
    main()

