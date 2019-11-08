#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'October 9 2019',
         'author': 'Scott Kempf, UTIG',
         'info': 'First release.'}}

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

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

Processors = [
    [("Name","Run Range Compression"),
     ("Input","_a.dat"),
     ("Input","_s.dat"),
     ("Processor", "run_rng_cmp.py"), ("Library", "xlib/cmp/pds3lbl.py"),
     ("Library", "xlib/cmp/plotting.py"), ("Library", "xlib/cmp/rng_cmp.py"),
     ("Outdir", "ion"),
     ("Output", "_s.h5"),
     ("Output", "_s_TECU.txt")
    ],
    [("Name","Run Altimetry"),
     ("Indir", "ion"),
     ("Input", "_s.h5"),
     ("Input", "_s_TECU.txt"),
     ("Processor", "run_altimetry.py"),
     ("Library", "xlib/cmp/pds3lbl.py"), ("Library", "xlib/altimetry/beta5.py"),
     ("Outdir", "beta5"),
     ("Output", ".h5")
    ]
]

def getmtime(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = -1
    return (mtime,path)


def main():
    parser = argparse.ArgumentParser(description='SHARAD Pipeline')
    parser.add_argument('-o', '--output', default='/disk/kea/SDS/targ/xtra/SHARAD',
                        help="Output base directory")

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

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")

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

    logging.debug("Base output directory: " + path_outroot)

    SHARADroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    for prod in Processors:
        for i, infile in enumerate(lookup):
            path_file = infile.replace(SHARADroot, '')
            data_file = os.path.basename(path_file).replace('_a.dat', '')
            path_file = os.path.dirname(path_file)
            root_file,ext_file = os.path.splitext(data_file)
            intimes = []
            outtimes = []
            for attr in prod:
                if (attr[0] == "Name"):
                    logging.info("Considering: " + attr[1])
                if (attr[0] == "Indir"):
                    indir = os.path.join(path_outroot, path_file, attr[1])
                if (attr[0] == "Input"):
                    # FIXME: This might be better if absolute paths are detected
                    if (attr[1] == '_a.dat' or attr[1] == '_s.dat'):
                         infile = os.path.join(SHARADroot,path_file,data_file+attr[1])
                    else:
                         infile = os.path.join(indir, data_file+attr[1])
                    intimes.append(getmtime(infile))
                if (attr[0] == "Processor"):
                    proc = attr[1]
                    intimes.append(getmtime(attr[1]))
                if (attr[0] == "Library"):
                    libfile = os.path.join('../',attr[1])
                    intimes.append(getmtime(libfile))
                if (attr[0] == "Outdir"):
                    outdir = os.path.join(path_outroot, path_file, attr[1])
                if (attr[0] == "Output"):
                    output = os.path.join(outdir, data_file+attr[1])
                    outtimes.append(getmtime((output)))
            intimes.sort(key = lambda x: x[0])
            outtimes.sort(key = lambda x: x[0])
            if (intimes[0][0] == -1):
                print('Input missing.')
            elif (intimes[-1][0] < outtimes[0][0]):
                print('Up to date.')
            else:
                print('Ready to process.')
                logging.debug("Adding " + infile)
                process_list.append(infile)
        else:
            logging.debug('File already processed. Skipping ' + infile)

    sys.exit(0)
    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)))

    start_time = time.time()

    named_params = {'saving':True, 'chrp_filt':True, 'verbose':args.verbose, 'radargram':False}

    if nb_cores <= 1:
        # Single processing (for profiling)
        for t in process_list:
            cmp_processor(*t, **named_params)
    else:
        # Multiprocessing
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(cmp_processor, t, \
                   named_params) for t in process_list]

        for i, result in enumerate(results):
            dummy = result.get()
            if dummy == 1:
                logging.error("{:s}: Problem running pulse compression".format(process_list[i][4]))
            else:
                logging.info("{:s}: Finished pulse compression".format(process_list[i][4]))

    logging.info("Done in {:0.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    # execute only if run as a script
    main()