#!/usr/bin/env python3
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

import numpy as np
import spiceypy as spice
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../xlib')
#import misc.prog as prog
import misc.hdf as hdf

import rng.icd as icd

DO_CLUTTER_ONLY = False

def main():
    global args, DO_CLUTTER_ONLY
    parser = argparse.ArgumentParser(description='Run SHARAD ranging processing')
    parser.add_argument('-o','--output', help="Output base directory")
    parser.add_argument('--qcdir', help="Quality control output directory")
    parser.add_argument('--ofmt', default='hdf5', choices=('hdf5','csv','none'),
                        help="Output file format")
    parser.add_argument('--tracklist', default="xover_idx.dat",
        help="List of tracks with xover points to process")
    parser.add_argument('--clutterpath', default=None, help="Cluttergram path")
    parser.add_argument('--noprogress', action="store_true", help="don't show progress")
    parser.add_argument('-j','--jobs', type=int, default=4, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-d','--debug', action="store_true", help="Display debugging plots")
    parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    parser.add_argument('--maxtracks', type=int, default=0, help="Maximum number of tracks to process")
    parser.add_argument('--maxechoes', type=int, default=0, help="Maximum number of echoes to process in a track")
    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                        help="Root directory (default: environment variable SDS)")
    parser.add_argument('--clutteronly', action="store_true", help="Cluttergram simulation only")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/rng')

    DO_CLUTTER_ONLY = args.clutteronly

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_ranging: [%(levelname)-7s] %(message)s")

    # Build list of processes
    logging.info('build task list')
    process_list = []
    path_root = os.path.join(args.SDS, 'targ/xtra/SHARAD/cmp/')
    path_edr = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')
    path_out = args.output

    ext = {'hdf5':'.h5','csv':'.csv', 'none':''}

    tasknum = 1
    b_noprogress = True if args.noprogress else False
    with open(args.tracklist, 'r') as flist:
        for line in flist:
            if line.strip().startswith('#'):
                continue
            print(line)
            path, idx_start, idx_end = line.split()
            path = path.rstrip()
            relpath = os.path.dirname(os.path.relpath(path, path_edr))
            path_file = os.path.relpath(path, path_edr)
            data_file = os.path.basename(path)
            outfile = os.path.join(path_out, relpath, 'icd', data_file.replace('.dat', ext[args.ofmt]))
            clutterfile = os.path.join(path_out, relpath, 'icd', data_file.replace('.dat', '.cluttergram.npy'))

            if not os.path.exists(outfile):
                process_list.append({
                    'inpath' : path,
                    #'outputfile' : outfile,
                    'idx_start' : idx_start,
                    'idx_end' : idx_end,
                    'SDS': args.SDS,
                    'save_format' : args.ofmt,
                    'tasknum': tasknum,
                    'clutterfile': clutterfile,
                    'b_noprogress': b_noprogress,
                    'maxechoes': args.maxechoes,
                    })
                logging.debug("[%d] %s", tasknum, str(process_list[-1]))
                tasknum += 1

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        logging.info("Limiting to first %d tracks", args.maxtracks)
        process_list = process_list[0:args.maxtracks]

    if len(process_list) % 2 != 0:
        raise ValueError("Must have an even number of tracks for crossovers!")

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing %d tracks", len(process_list))

    rlist = [] # result list
    if args.jobs <= 1:
        for i, t in enumerate(process_list, start=1):
            result = process_rng(**t)
            rlist.append(result) # tuple of two numbers
            logging.info("Finished task %d of %d", i, len(process_list))
    else:
        pool = multiprocessing.Pool(args.jobs)
        results = [pool.apply_async(process_rng, [], t) for t in process_list]
        for i, result in enumerate(results, start=1):
            rlist.append(result.get())
            logging.info("Finished task %d of %d", i, len(process_list))
    logging.info('done with tasks.')


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

    out = np.zeros((len(rlist), 2))
    # Sort results
    for result in rlist:
        ii = result['tasknum'] - 1
        out[ii, :] = result['result'][0:2]

    delta_ranges = out[0::2, 1] - out[1::2, 1]
    logging.debug("rms={:0.4f}".format(np.nanstd(delta_ranges)))

    import matplotlib
    font = {'family' : 'serif',
            'size'   : 24}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(10,10))
    rlist = np.array(rlist)
    plt.scatter(np.arange(0,len(delta_ranges)), delta_ranges, s=30)
    ax.set_ylim(-150, 150)
    ax.set_xlabel('Xover Number')
    ax.set_ylabel('Ranging Residual [m]')
    plt.grid()
    plt.tight_layout()

    if args.qcdir:
        # Save debugging outputs to qc directory if requested
        logging.debug("Saving qc to " + args.qcdir)
        os.makedirs(args.qcdir, exist_ok=True)
        np.save(os.path.join(args.qcdir, 'ranging_result.npy'), rlist)
        plt.savefig(os.path.join(args.qcdir, 'ranging_result.pdf'))

    plt.show()


def process_rng(inpath, SDS, idx_start=None, idx_end=None, save_format='', tasknum=0, clutterfile=None,
                b_noprogress=False, bplot=False, maxechoes=0):

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
        # create cmp path
        path_root_rng = os.path.join(SDS, 'targ/xtra/SHARAD/rng/')
        path_root_cmp = os.path.join(SDS, 'targ/xtra/SHARAD/cmp/')
        path_root_edr = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')
        dtm_path = os.path.join(SDS, 'orig/supl/hrsc/MC11E11_HRDTMSP.dt5.tiff')
        # Relative path to this file
        fname = os.path.basename(inpath)
        obn = fname[2:9] # orbit name
        # Relative directory of this file
        reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
        logging.debug("inpath: " + inpath)
        logging.debug("reldir: " + reldir)
        logging.debug("path_root_edr: " + path_root_edr)
        cmp_path = os.path.join(path_root_cmp, reldir, 'ion', fname.replace('_a.dat','_s.h5') )
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_label = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

        science_path = inpath.replace('_a.dat','_s.dat')

        if not os.path.exists(cmp_path):
            logging.warning(cmp_path + " does not exist")
        science_path = inpath.replace('_a.dat','_s.dat')
        if not os.path.exists(cmp_path):
            logging.warning(cmp_path + " does not exist")
            return 0

        logging.info("%s: Reading %s", taskname, cmp_path)
        co_data = 10

        if os.path.exists(clutterfile):
            # Read from this file
            clutter_save_path = None
            clutter_load_path = clutterfile
        else:
            # Write to this file
            clutter_load_path = None
            clutter_save_path = clutterfile
            clutter_save_dir = os.path.dirname(clutterfile)
            os.makedirs(clutter_save_dir, exist_ok=True)


        sim = icd.gen_or_load_cluttergram(cmp_path, dtm_path, science_path, label_path,
                inpath, aux_label,
                int(idx_start), int(idx_end), debug=args.debug,
                ipl=True, #co_sim=co_sim, co_data=co_data,
                #window=50, sim=sim,
                cluttergram_path=clutter_load_path, save_clutter_path=clutter_save_path,
                do_progress=not b_noprogress, maxechoes=maxechoes)

        if DO_CLUTTER_ONLY:
            raise Exception("Done with clutter simulation.")

        result = np.zeros((20, 3))
        for co_sim in range(12, 25):

            logging.debug("%s: icd_ranging(co_sim=%d)", taskname, co_sim)
            #ranging_func = icd.icd_ranging_3

            # icd_ranging_cg3 returns delta, dz, min(md)
            result[co_sim-5] = icd.icd_ranging_cg3\
                              (cmp_path, dtm_path, science_path, label_path, \
                               inpath, aux_label,
                               int(idx_start), int(idx_end), debug=args.debug,
                               ipl=True, co_sim=co_sim, co_data=co_data,
                               window=50, sim=sim,
                               #cluttergram_path=clutter_load_path, save_clutter_path=clutter_save_path,
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
        print("min: " + str(amin))

        return {
            'inpath': inpath,
            'obn': obn,
            'tasknum': tasknum,
            'result': result[amin]
        }
        #return (obn, min_result[0], min_result[1])

    except Exception:
        taskname = "task{:03d}".format(tasknum)
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line) )
        #return (None, None)
        return {
            'inpath': inpath,
            'obn': obn,
            'tasknum': tasknum,
            'result': np.array([float('nan'), float('nan')])
        }

if __name__ == "__main__":
    # execute only if run as a script
    main()

