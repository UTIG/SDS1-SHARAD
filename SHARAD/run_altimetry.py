#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 6 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'February 19, 2024',
         'author': 'Gregory Ng, UTIG',
         'info': 'Implement fine tracking multiprocessing'},
    
}
           
"""

On freeze, I think we can run run_altimetry with a -j flag of 8 and an
--finethreads flag of 8. This will set typical CPU utilization somewhere
between 8 and 64 cores as it goes through the altimetry processing. During
most of the radar processing it uses 1 job per transect, and then only during
the fine tracking phase will it use 8 jobs per transect. The fine tracking
phase takes about 10 times longer than all the previous phases on freeze.

Increasing the -j flag increases peak memory usage and disk read bandwidth
requirements (because we're reading multiple transects in parallel), while
increasing --finejobs will have no effect on peak memory usage (because at
this stage we're only carrying a small window of samples around the surface).
However, very high values of finejobs have a quickly diminishing return on
processing time.

This should be compatible with efficient processing on freeze and on tacc. On
TACC, this works, but if we have 128 cores to work with, then maybe we could
bump this to {j=8, finejobs=16} or {j=8,finejobs=32}? Or push your luck with
{j=16,finejobs=8}?

However one problem is that you can't multiprocess inside of other multiprocessing

Here is some math to support this:

Memory profiing showed that there was a baseline python process VmSize of
about 2 GB (to load all the python code) and then memory went up commensurate
with intermediate variables. For a 1.5 GB input file you could budget maybe +3
GB? So we could budget memory usage of 2.0 + 3.0 *j gigabytes, where 'j' is
the number specified in the -j flag to run_altimetry.py

So that is to say that when we run on TACC, I think we should limit the number
of parallel jobs (-j flag) to no more than maybe 3 or 4 for vm-small. This
math says that you could do up to 63 jobs on the regular sized machine under
/ultra-ideal/ circumstances?

"""




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
sys.path.append('../xlib')
#import misc.prog as prog
import misc.hdf as hdf
import matplotlib.pyplot as plt
#import altimetry.beta5 as b5
import beta5 as b5
#import b5int as b5

def main():
    desc = 'Run SHARAD altimetry processing'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-o', '--output', default=None,
                        help="Output base directory")
    parser.add_argument('--ofmt', default='hdf5',
                        choices=('hdf5', 'csv', 'none'),
                        help="Output file format")
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of orbit jobs to process simultaneously")
    parser.add_argument('--finethreads', type=int, default=1,
                        help="Number of threads to use when processing fine tracking")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="EDR_NorthPole_Path.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")

    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                        help="Root directory (default: environment variable SDS")

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_altimetry: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD/alt')

    # Set number of cores
    nb_cores = args.jobs
    kernel_path = os.path.join(args.SDS, 'orig/supl/kernels/mro/mro_v01.tm')
    spice.furnsh(kernel_path)

    # Build list of processes
    logging.info('build task list')
    process_list = []
    path_root = os.path.join(args.SDS, 'targ/xtra/SHARAD/cmp/')
    path_edr = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')

    ext = {'hdf5':'.h5', 'csv':'.csv', 'none':''}

    with open(args.tracklist, 'rt') as flist:
        i = 0
        for path in flist:
            if not path or path.strip().startswith('#'):
                continue
            path = path.rstrip()
            relpath = os.path.dirname(os.path.relpath(path, path_edr))
            path_file = os.path.relpath(path, path_edr)
            data_file = os.path.basename(path)
            outfile = os.path.join(args.output, relpath, 'beta5',
                                   data_file.replace('.dat', '.h5'))

            if os.path.exists(outfile):
                logging.info("Not adding %s to jobs.  %s already exists", path, outfile)
                continue
            outfile2 = outfile.replace('_a_a.h5', '_a_s.h5')
            if os.path.exists(outfile2):
                logging.info("Not adding %s to jobs.  %s already exists", path, outfile2)
                continue

            process_list.append({
                'inpath': path,
                'outfile': outfile,
                'SDS': args.SDS,
                'idx_start': 0,
                'idx_end': None,
                'finethreads': args.finethreads,
                'save_format': args.ofmt})
            logging.debug("[%d] %s", i+1, str(process_list[-1]))
            i += 1

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing %d tracks", len(process_list))

    if nb_cores <= 1:
        for i, t in enumerate(process_list):
            result = alt_processor(**t)
            print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    else:
        with multiprocessing.Pool(nb_cores) as pool:
            results = [pool.apply_async(alt_processor, [], t) for t in process_list]
            for i, result in enumerate(results):
                flag = result.get() # Wait for result
                print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    print('done')


def alt_processor(inpath:str, outfile: str, SDS: str, finethreads: int, idx_start=0, idx_end=None, save_format=''):
    try:
        # create cmp path
        path_root_alt = os.path.join(SDS, 'targ/xtra/SHARAD/alt/')
        path_root_cmp = os.path.join(SDS, 'targ/xtra/SHARAD/cmp/')
        path_root_edr = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')
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
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

        science_path = inpath.replace('_a.dat', '_s.dat')
        if not os.path.exists(cmp_path):
            cmp_path0 = cmp_path
            cmp_path = cmp_path.replace('_s.h5', '_a.h5')
            if not os.path.exists(cmp_path):
                logging.warning(cmp_path0 + " does not exist")
                logging.warning(cmp_path + " does not exist")
                return 0

        logging.info("Reading " + cmp_path)
        result = b5.beta5_altimetry(cmp_path, science_path, label_path, aux_path,
                                    idx_start=idx_start, idx_end=idx_end,
                                    use_spice=False, ft_avg=10, max_slope=25,
                                    noise_scale=20, finethreads=finethreads)

        #plt.plot(result['spot_radius'])
        #plt.show()

        if save_format in ('', 'none'):
            return 0

        logging.info("Writing to " + outfile)
        outputdir = os.path.dirname(outfile)

        os.makedirs(outputdir, exist_ok=True)

        if save_format == 'hdf5':
            orbit_data = {'orbit'+str(obn): result}
            #os.system("rm " + outfile)
            with hdf.hdf(outfile, mode='w') as h5:
                h5.save_dict('beta5', orbit_data)
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


