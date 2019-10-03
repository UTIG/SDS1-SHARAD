#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 6 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}

import sys
import os
import multiprocessing
import logging
import argparse

import numpy as np
import spiceypy as spice
import pandas as pd
import traceback

sys.path.append('../xlib/altimetry')
#import misc.prog as prog
import misc.hdf as hdf
import matplotlib.pyplot as plt
#import altimetry.beta5 as b5
import beta5 as b5
#import b5int as b5

def main():
    desc = 'Run SHARAD altimetry processing'
    default_output_path = '/disk/kea/SDS/targ/xtra/SHARAD/alt'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-o', '--output', default=default_output_path,
                        help="Output base directory")
    parser.add_argument('--ofmt', default='hdf5',
                        choices=('hdf5', 'csv', 'none'),
                        help="Output file format")
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="EDR_NorthPole_Path.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_altimetry: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs
    kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
    spice.furnsh(kernel_path)

    # Build list of processes
    logging.info('build task list')
    process_list = []
    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
    path_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'

    ext = {'hdf5':'.h5', 'csv':'.csv', 'none':''}

    with open(args.tracklist, 'r') as flist:
        for i, path in enumerate(flist):
            path = path.rstrip()
            relpath = os.path.dirname(os.path.relpath(path, path_edr))
            path_file = os.path.relpath(path, path_edr)
            data_file = os.path.basename(path)
            outfile = os.path.join(args.output, relpath, 'beta5',
                                   data_file.replace('.dat', '.h5'))

            if not os.path.exists(outfile) and not os.path.exists(outfile.replace('_a_a.h5', '_a_s.h5')):
                process_list.append({
                    'inpath': path,
                    'outfile': outfile,
                    'idx_start': 0,
                    'idx_end': None,
                    'save_format': args.ofmt})
                logging.debug("[{:d}] {:s}".format(i+1, str(process_list[-1])))

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)))

    if nb_cores <= 1:
        for i, t in enumerate(process_list):
            result = alt_processor(**t)
            print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    else:
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(alt_processor, [], t) for t in process_list]
        for i, result in enumerate(results):
            flag = result.get() # Wait for result
            print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    print('done')


def alt_processor(inpath, outfile, idx_start=0, idx_end=None, save_format=''):
    try:
        # create cmp path
        path_root_alt = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
        path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_root_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
        # Relative path to this file
        fname = os.path.basename(inpath)
        obn = fname[2:9] # orbit name
        # Relative directory of this file
        reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
        logging.debug("inpath: " + inpath)
        logging.debug("reldir: " + reldir)
        logging.debug("path_root_edr: " + path_root_edr)
        cmp_path = os.path.join(path_root_cmp, reldir, 'ion',
                                fname.replace('_a.dat', '_s.h5'))
        #cmp_path = path_root_cmp+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'

        science_path = inpath.replace('_a.dat', '_s.dat')
        if not os.path.exists(cmp_path):
            cmp_path = cmp_path.replace('_s.h5', '_a.h5')
            if not os.path.exists(cmp_path):
                logging.warning(cmp_path + " does not exist")
                return 0

        logging.info("Reading " + cmp_path)
        result = b5.beta5_altimetry(cmp_path, science_path, label_path, aux_path,
                                    idx_start=idx_start, idx_end=idx_end,
                                    use_spice=False, ft_avg=10, max_slope=25,
                                    noise_scale=20, fix_pri=1, fine=True)

        #plt.plot(result['spot_radius'])
        #plt.show()

        if save_format == '' or save_format == 'none':
            return 0

        logging.info("Writing to " + outfile)
        outputdir = os.path.dirname(outfile)

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if save_format == 'hdf5':
            orbit_data = {'orbit'+str(obn): result}
            #os.system("rm " + outfile)
            h5 = hdf.hdf(outfile, mode='w')
            h5.save_dict('beta5', orbit_data)
            h5.close()
        elif save_format == 'csv':
            #fname1 = fname.replace('_a.dat', '.csv.gz')
            #outfile = os.path.join(path_root_alt, reldir, 'beta5',fname1)

            df.to_csv(outfile)
        else:
            logging.warning("Unknown output format '{:s}'".format(save_format))
            return 1
        return 0
    except Exception: # pragma: no cover
        taskname = "error"
        for line in traceback.format_exc().split("\n"):
            print('{:s}: {:s}'.format(taskname, line))
        return 1


if __name__ == "__main__":
    # execute only if run as a script
    main()


