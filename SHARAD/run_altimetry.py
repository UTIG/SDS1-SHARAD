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
from typing import List

import numpy as np
import spiceypy as spice
import pandas as pd
import traceback

from run_rng_cmp import read_tracklistfile, filename_to_productid, run_jobs,\
    process_product_args, should_process_products, \
    add_standard_args

sys.path.append('../xlib')
#import misc.prog as prog
import misc.hdf as hdf
import matplotlib.pyplot as plt
#import altimetry.beta5 as b5
import altimetry.beta5 as b5

from SHARADEnv import SHARADFiles
import misc.fileproc as fileproc

def main():
    desc = 'Run SHARAD altimetry processing'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ofmt', default='hdf5',
                        choices=('hdf5', 'csv', 'none'),
                        help="Output file format")
    parser.add_argument('--finethreads', type=int, default=1,
                        help="Number of threads to use when processing fine tracking")
    add_standard_args(parser)

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_altimetry: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')

    logging.debug('build task list')
    # File location calculator
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist  = process_product_args(args.product_ids, args.tracklist, sfiles)
    assert productlist, "No files to process"

    process_list = []
    for tracknum, product_id in enumerate(productlist):
        infiles = sfiles.cmp_product_paths(product_id)
        infiles.update(sfiles.edr_product_paths(product_id))
        outfiles = sfiles.alt_product_paths(product_id)

        if not should_process_products(product_id, infiles,  outfiles, args.overwrite):
            continue

        logging.debug("Adding %s", product_id)
        process_list.append({
            #'product_id': product_id,
            'cmp_path': infiles['cmp_h5'],
            'edr_sci': infiles['edr_sci'],
            'outfile': outfiles['alt_h5'],
            'SDS': args.SDS,
            'idx_start': 0,
            'idx_end': None,
            'finethreads': args.finethreads,
            'save_format': args.ofmt})

    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        return 0

    kernel_path = os.path.join(args.SDS, 'orig/supl/kernels/mro/mro_v01.tm')
    spice.furnsh(kernel_path)

    run_jobs(alt_processor, process_list, args.jobs)


def alt_processor(cmp_path: str, edr_sci: str, outfile: str, SDS: str, finethreads: int, idx_start=0, idx_end=None, save_format=''):
    """
    Parameters:
    cmp_path - path to hdf5 output of range compression
    edr_sci - path to EDR science data file
    outfile - output altimetry file (usually HDF5)
    SDS - SDS environment variable
    finethreads - number of processes to use during fine tracking (0 or 1 to disable)
    idx_start, idx_end: trace indexes to process
    """
    try:
        label_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

        # This should have already been checked on the outside
        assert os.path.exists(edr_sci)
        logging.info("Reading cmp " + cmp_path)
        logging.info("Reading sci " + edr_sci)
        result = b5.beta5_altimetry(cmp_path, edr_sci, label_path, aux_path,
                                    idx_start=idx_start, idx_end=idx_end,
                                    use_spice=False, ft_avg=10, max_slope=25,
                                    noise_scale=20, finethreads=finethreads)

        #plt.plot(result['spot_radius'])
        #plt.show()

        if save_format in ('', 'none'):
            return 0

        logging.info("Writing to %s", outfile)
        outputdir = os.path.dirname(outfile)

        os.makedirs(outputdir, exist_ok=True)

        if save_format == 'hdf5':
            fname = os.path.basename(edr_sci)
            obn = fname[2:9] # orbit name (transaction ID)
            orbit_data = {'orbit'+str(obn): result}
            #os.system("rm " + outfile)
            with hdf.hdf(outfile, mode='w') as h5:
                h5.save_dict('beta5', orbit_data)
        elif save_format == 'csv':
            #fname1 = fname.replace('_a.dat', '.csv.gz')
            #outfile = os.path.join(path_root_alt, reldir, 'beta5',fname1)
            outfile = outfile.replace('.dat', '.csv')
            df.to_csv(outfile)
        else:
            logging.warning("Unknown output format '%s'", save_format)
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


