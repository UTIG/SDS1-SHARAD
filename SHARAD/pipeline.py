#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.1'
__history__ = {
    '1.1':
        {'date': 'July 7 2020',
         'author': 'Scott Kempf, UTIG',
         'info': 'Detection is working.'},
    '1.0':
        {'date': 'October 9 2019',
         'author': 'Scott Kempf, UTIG',
         'info': 'First release.'}
}

# TODO: manual step mode
# TODO: handle "srf"
# TODO: Call processors: test...
# TODO: Call processors: rng_cmp: works
# TODO: Call processors: altim: nothing runs, tracklist matches input
# TODO: Call processors: rsr: fix
# TODO: Call processors: sar2: works
# TODO: Parameters for SAR processing (these could change the output path).
# TODO: Manual vs automatic pipeline
# TODO: Sandbox mode (input form sandbox is issue)
# TODO: Single step
# TODO: Use max tracks arg
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
import subprocess

# TODO: make this import more robust to allow this script
# TODO: to be run from outside the SHARAD directory
sys.path.insert(0, '../xlib')
#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting
import tempfile

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
    [("Name","Run Ranging"),
     ("InPrefix", "cmp"),
     ("Indir", "ion"),
     ("Input", "_s.h5"),
     ("Input", "_s_TECU.txt"),
     ("Processor", "run_ranging.py"),
     ("Library", "xlib/misc/hdf.py"), ("Library", "xlib/rng/icd.py"),
     ("Prefix", "rng"),
     ("Outdir", "icd"),
     ("Output", "_a.cluttergram.npy")
    ],
]

def temptracklist(infile):
    logging.debug("Writing Temp File");
    logging.debug(infile);
    temp = tempfile.NamedTemporaryFile(mode='w+',delete=False)
    temp.write(infile+'\n')
    temp.close()
    return temp.name

def getmtime(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = -1
    return mtime,path

def manual(cmd, infile, outputs):
    import getch
    print('Trackline: ' + infile);
    print('Command: ' + ' '.join(cmd));
    print('(Y)es, (N)o, (Q)uit?', end=' ', flush=True);
    c = getch.getch()
    print(c)
    while c not in 'YyNnQq':
        c = getch.getch()
        print(c)
    if c in 'Qq':
        sys.exit(0)
    if c in 'Yy':
        subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='SHARAD Pipeline')
    parser.add_argument('-o', '--output', default='/disk/kea/SDS/targ/xtra/SHARAD',
                        help="Output base directory")

    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-m', '--manual', action="store_true",
                        help="Prompt before running processors")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run (NOT WORKING)")
    parser.add_argument('--tracklist', default="elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process (NOT WORKING)")
    parser.add_argument('-1', '--once', action="store_true",
                        help="Just run one processor and exit.")
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
        indir = ''
        proc = ''
        outdir = ''
        for i, infile in enumerate(lookup):
            path_file = infile.replace(SHARADroot, '')
            data_file = os.path.basename(path_file).replace('_a.dat', '')
            path_file = os.path.dirname(path_file)
            root_file,ext_file = os.path.splitext(data_file)
            # FIXME error checking is needed here
            m = re.search('edr(\d*)/', infile)
            orbit = m.group(1)
            intimes = []
            outputs = []
            outtimes = []
            prefix = ''
            inprefix = ''
            for attr in prod:
                if (attr[0] == "Name"):
                    logging.info("Considering: " + attr[1])
            for attr in prod:
                if (attr[0] == "InPrefix"):
                    prefix = attr[1] + '/'
            for attr in prod:
                if (attr[0] == "Indir"):
                    indir = os.path.join(path_outroot, prefix, path_file, attr[1])
            for attr in prod:
                if (attr[0] == "Input"):
                    # FIXME: This might be better if absolute paths are detected
                    if (attr[1] == '_a.dat' or attr[1] == '_s.dat'):
                         oneinput = os.path.join(SHARADroot,path_file,data_file+attr[1])
                    else:
                         oneinput = os.path.join(indir, data_file+attr[1])
                    intimes.append(getmtime(oneinput))
            for attr in prod:
                if (attr[0] == "Processor"):
                    proc = attr[1]
                    intimes.append(getmtime(attr[1]))
            for attr in prod:
                if (attr[0] == "Library"):
                    if (not args.ignorelibs):
                        libfile = os.path.join('../',attr[1])
                        intimes.append(getmtime(libfile))
            for attr in prod:
                if (attr[0] == "Prefix"):
                    # Must come before Outdir
                    prefix = attr[1] + '/'
            for attr in prod:
                if (attr[0] == "Outdir"):
                    # Must come before Output
                    outdir = os.path.join(path_outroot, prefix, path_file, attr[1])
                if (attr[0] == "Output"):
                    output = os.path.join(outdir, data_file+attr[1])
                    outputs.append(output)
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
                        logging.info('Ready to process (no output file).')
                    else:
                        logging.info('Ready to process (old output file).')
                        logging.info('Deleting old files.')
                        for output in outputs:
                            os.unlink(output)
                            try:
                                os.rmdir(os.path.dirname(output))
                            except OSError:
                                # We don't care if it fails a few times
                                pass
                    logging.info(output)
                    logging.debug("Processing " + infile)
                    temp = temptracklist(infile)
                    logging.info("Invoking: " + './' + proc + ' --tracklist ' + temp + ' -o ' + os.path.join(path_outroot,prefix))
                    if args.dryrun:
                        logging.info("Dryrun, quiting.");
                        sys.exit(0)
                    cmd = ['./' + proc, '--tracklist', temp, '-o', os.path.join(path_outroot,prefix), '-v']
                    if args.manual:
                        manual(cmd, infile, outputs)
                    else:
                        subprocess.run(cmd)
                    os.unlink(temp)
                    if args.once:
                        logging.info("Only one process request.  Quiting.")
                        sys.exit(1)

        else:
            logging.debug('File already processed. Skipping ' + infile)

    sys.exit(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()
