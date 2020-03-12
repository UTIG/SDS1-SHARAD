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


def main():
    global args
    parser = argparse.ArgumentParser(description='Run SHARAD ranging processing')
    parser.add_argument('-o','--output', help="Output base directory")
    parser.add_argument('--ofmt', default='hdf5', choices=('hdf5','csv','none'),
                        help="Output file format")
    parser.add_argument('--tracklist', default="xover_idx.dat",
        help="List of tracks with xover points to process")
    parser.add_argument('--clutterpath', default=None, help="Cluttergram path")
    parser.add_argument('--noprogress', action="store_true", help="don't show progress")
    parser.add_argument('-j','--jobs', type=int, default=4, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    parser.add_argument('--maxtracks', type=int, default=0, help="Maximum number of tracks to process")
    parser.add_argument('--SDS', default=os.getenv('SDS'), help="Override SDS environment variable")
    
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/rng')


    loglevel=logging.DEBUG if args.verbose else logging.INFO
 
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_ranging: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs

    # Build list of processes
    logging.info('build task list')
    process_list = []
    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
    path_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    path_out = args.output

    ext = {'hdf5':'.h5','csv':'.csv', 'none':''}

    tasknum = 1
    b_noprogress = True if args.noprogress else False
    with open(args.tracklist, 'r') as flist:
        for line in flist:
            if line.strip().startswith('#'):
                continue
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
                    'save_format' : args.ofmt,
                    'tasknum': tasknum,
                    'clutterfile': clutterfile,
                    'b_noprogress': b_noprogress,
                    })
                logging.debug("[{:d}] {:s}".format(tasknum, str(process_list[-1])))
                tasknum += 1

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        logging.info("Limiting to first {:d} tracks".format(args.maxtracks))
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)) )

    rlist = []
    if nb_cores <= 1:
        for i,t in enumerate(process_list):
            result = process_rng(**t)
            rlist.append(result)
            logging.info("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    else:
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(process_rng, [], t) for t in process_list]
        for i,result in enumerate(results):
            rlist.append(result.get())
            logging.info("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    logging.info('done')

    rlist = np.array(rlist)
    out = np.zeros((len(rlist),2))
    # Sort results
    if nb_cores > 1:
        for i in range(len(rlist)):
            for j in range(len(process_list)):
                if str(int(rlist[i,0])) in process_list[j]['inpath']:
                    out[j] = rlist[i,1:3]
    else: out = rlist[:,1:3]

    print(out)
    delta_ranges = out[0::2,1] - out[1::2,1] 
    print('rms', np.sqrt(np.var(delta_ranges)))
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
    #plt.savefig('ranging_result.pdf')
    plt.tight_layout()
    plt.show()
    
    np.save('ranging_result.npy', rlist)

def process_rng(inpath, idx_start=None, idx_end=None, save_format='', tasknum=0, clutterfile=None,
                b_noprogress=False, bplot=False):
    try:
        obn = 0 # Suppressing undefined variable error.
        # create cmp path
        path_root_rng = '/disk/kea/SDS/targ/xtra/SHARAD/rng/'
        path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_root_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
        dtm_path = '/disk/daedalus/sds/orig/supl/hrsc/MC11E11_HRDTMSP.dt5.tiff'
        # Relative path to this file
        fname = os.path.basename(inpath)

        # Relative directory of this file
        reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
        logging.debug("inpath: " + inpath)
        logging.debug("reldir: " + reldir)
        logging.debug("path_root_edr: " + path_root_edr)
        cmp_path = os.path.join(path_root_cmp, reldir, 'ion', fname.replace('_a.dat','_s.h5') )
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_label = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'

        science_path = inpath.replace('_a.dat','_s.dat')

        if not os.path.exists(cmp_path): 
            logging.warning(cmp_path + " does not exist")
        science_path=inpath.replace('_a.dat','_s.dat')
        if not os.path.exists(cmp_path): 
            logging.warning(cmp_path + " does not exist")
            return 0

        logging.info("Reading " + cmp_path)
        co_data = 10
        
        result = np.zeros((20,3))
        for co_sim in range(12,25):

            if os.path.exists(clutterfile):
                # Read from this file
                clutter_save_path = None
                clutter_load_path = clutterfile
            else:
                # Write to this file
                clutter_load_path = None
                clutter_save_path = clutterfile
                clutter_save_dir = os.path.dirname(clutterfile)
                if not os.path.exists(clutter_save_dir):
                    os.makedirs(clutter_save_dir)

            result[co_sim-5] = icd.icd_ranging(cmp_path, dtm_path, science_path, label_path,
                                 inpath, aux_label,
                                 int(idx_start), int(idx_end), debug=False,
                                 ipl=True, co_sim=co_sim, co_data=co_data,
                                 window=50, 
                                 cluttergram_path=clutter_load_path, save_clutter_path=clutter_save_path,
                                 do_progress=not b_noprogress)


        if bplot:
            plt.scatter(np.arange(1,25,1), result[:,0], s=30)
            plt.show()
            plt.scatter(np.arange(1,25,1), result[:,2], s=30)
            plt.show()

        min_result = result[np.argmin(result[:,2])]
        print(np.argmin(result[:, 2]))
        return [obn, min_result[0], min_result[1]]

    except Exception:
        taskname = "task{:03d}".format(tasknum)
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line) )
        return 1

if __name__ == "__main__":
    # execute only if run as a script
    main()

