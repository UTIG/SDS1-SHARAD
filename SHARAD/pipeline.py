#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'October 9 2019',
         'author': 'Scott Kempf, UTIG',
         'info': 'First release.'}}

# TODO: handle "rng", "srf", "sza"
# TODO: Call processors.
# TODO: Parameters for SAR processing (these could change the output path).
# TODO: Manual vs automatic pipeline
# TODO: Parallelism

import sys
import os
import time
import logging
import argparse
import warnings
import multiprocessing
import traceback
import importlib.util
import re
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
     ("Prefix", "cmp"),
     ("Outdir", "ion"),
     ("Output", "_s.h5"),
     ("Output", "_s_TECU.txt")
    ],
    [("Name","Run Altimetry"),
     ("InPrefix", "cmp"),
     ("Indir", "ion"),
     ("Input", "_s.h5"),
     ("Input", "_s_TECU.txt"),
     ("Processor", "run_altimetry.py"),
     ("Library", "xlib/cmp/pds3lbl.py"), ("Library", "xlib/altimetry/beta5.py"),
     ("Prefix", "alt"),
     ("Outdir", "beta5"),
     ("Output", "_a.h5")
    ],
    [("Name","Run RSR"),
     ("InPrefix", "cmp"),
     ("Indir", "ion"),
     ("Input", "_s.h5"),
     ("Processor", "run_rsr.py"),
     ("Library", "xlib/rsr/Classdef.py"), ("Library", "xlib/rsr/detect.py"),
     ("Library", "xlib/rsr/fit.py"), ("Library", "xlib/rsr/invert.py"),
     ("Library", "xlib/rsr/pdf.py"), ("Library", "xlib/rsr/run.py"),
     ("Library", "xlib/rsr/utils.py"), ("Library", "xlib/subradar/Classdef.py"),
     ("Library", "xlib/subradar/iem.py"),
     ("Library", "xlib/subradar/invert.py"),
     ("Library", "xlib/subradar/roughness.py"),
     ("Library", "xlib/subradar/utils.py"), ("Library", "SHARAD/SHARADEnv.py"),
     ("Prefix", "rsr"),
     ("Outdir", "cmp"),
     #("Outrsr", "rsr_%s.npy")
     ("Output", ".txt")
    ],
    [("Name","Run SAR"),
     ("InPrefix", "cmp"),
     ("Indir", "ion"),
     ("Input", "_s.h5"),
     ("Input", "_s_TECU.txt"),
     ("Processor", "run_sar2.py"),
     ("Library", "xlib/sar/sar.py"), ("Library", "xlib/sar/smooth.py"),
     ("Library", "xlib/cmp/pds3lbl.py"), ("Library", "xlib/altimetry/beta5.py"),
     ("Prefix", "foc"),
     ("Outdir", "5m/5 range lines/40km"),
     ("Output", "_s.h5")
    ],
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
    parser.add_argument('--ignorelibs', action='store_true',
                        help="Do not check times on libraries")
    parser.add_argument('--ignoretimes', action='store_true',
                        help="Do not any times")

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
            # FIXME error checking is needed here
            m = re.search('edr(\d*)/', infile)
            orbit = m.group(1)
            intimes = []
            outtimes = []
            prefix = ""
            inprefix = ""
            for attr in prod:
                if (attr[0] == "Name"):
                    logging.info("Considering: " + attr[1])
                if (attr[0] == "InPrefix"):
                    prefix = attr[1] + '/'
                if (attr[0] == "Indir"):
                    indir = os.path.join(path_outroot, prefix, path_file, attr[1])
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
                    if (not args.ignorelibs):
                        libfile = os.path.join('../',attr[1])
                        intimes.append(getmtime(libfile))
                if (attr[0] == "Prefix"):
                    # Must come before Outdir
                    prefix = attr[1] + '/'
                if (attr[0] == "Outdir"):
                    # Must come before Output
                    outdir = os.path.join(path_outroot, prefix, path_file, attr[1])
                if (attr[0] == "Output"):
                    output = os.path.join(outdir, data_file+attr[1])
                    outtimes.append(getmtime((output)))
                if (attr[0] == "Outrsr"):
                    # Ugly special case for RSR
                    # FIXME these should probably be in outdir not path_outroot
                    output = os.path.join(path_outroot, attr[1] % orbit)
                    outtimes.append(getmtime((output)))
            if (len(intimes) == 0):
                logging.error("No inputs for process")
            intimes.sort(key = lambda x: x[0])
            if (len(outtimes) == 0):
                logging.error("No outputs for process")
            outtimes.sort(key = lambda x: x[0])
            if (intimes[0][0] == -1):
                print('Input missing.')
            elif (intimes[-1][0] < outtimes[0][0]):
                print('Up to date.')
            else:
                if not args.ignoretimes or outtimes[0][0] == -1:
                    if (outtimes[0][0] == -1):
                        print('Ready to process (no output).')
                    else:
                        print('Ready to process (old output).')
                    print(output)
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
