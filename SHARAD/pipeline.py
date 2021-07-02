#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.2'
__history__ = {
    '1.2':
        {'date': 'March 16 2021',
         'author': 'Scott Kempf, UTIG',
         'info': 'Processing is working.'},
    '1.1':
        {'date': 'July 7 2020',
         'author': 'Scott Kempf, UTIG',
         'info': 'Detection is working.'},
    '1.0':
        {'date': 'October 9 2019',
         'author': 'Scott Kempf, UTIG',
         'info': 'First release.'}
}

# TODO: Parameters for SAR processing (these could change the output path).
# TODO: Sandbox mode (input from sandbox is issue)
# TODO: Use max tracks arg
# TODO: Parallelism



"""

SYNOPSIS

pipeline.py orchestrates the running of SDS data processing scripts based on 
looking for the expected inputs and corresponding output files.

To run pipeline.py, you usually provide it with a list of orbits to examine.

To simply see what would be run, run with the -n option.

./pipeline.py -i elysium.txt -n

"""


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
import subprocess
import tempfile



PROCESSORS = [
    {
        "Name": "Run Range Compression",
        "InPrefix": '',
        "Indir": '',
        "Inputs": ["_a.dat", "_s.dat"],
        "Processor": "run_rng_cmp.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/cmp/plotting.py", "xlib/cmp/rng_cmp.py"],
        "OutPrefix": "cmp",
        "Outdir": "ion",
        "Outputs": ["_s.h5", "_s_TECU.txt"],
    },
    {
        "Name": "Run Altimetry",
        "InPrefix": "cmp",
        "Indir": "ion",
        "Inputs": ["_s.h5", "_s_TECU.txt"],
        "Processor": "run_altimetry.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        "OutPrefix": "alt",
        "Outdir": "beta5",
        "Outputs": ["_a.h5"],
    },
    {
        "Name": "Run RSR",
        "InPrefix": "cmp",
        "Indir": "ion",
        "Inputs": ["_s.h5"],
        "Processor": "run_rsr.py",
        # The libraries for rsr and subradar are no longer in the repository; they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "OutPrefix": "rsr",
        "Outdir": "cmp",
        "Outputs": [".txt"],
    },
    {
        "Name": "Run SAR",
        "InPrefix": "cmp",
        "Indir": "ion",
        "Inputs": ["_s.h5", "_s_TECU.txt"],
        "Processor": "run_sar2.py",
        "Libraries": ["xlib/sar/sar.py", "xlib/sar/smooth.py",
                      "xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        "OutPrefix": "foc",
        # TODO: GNG -- suggest spaces become underscores
        # Other flags might result in other outputs.
        # option makes sense.
        "Outdir": "5m/5 range lines/40km",
        "Outputs": ["_s.h5"],
    },
    # Run ranging needs crossovers, so needs a track file with pairs of tracks
    # and record numbers.  This is a special data product so we can't run it
    # automatically.
    #{
    #    "Name": "Run Ranging",
    #    "InPrefix": "cmp",
    #    "Indir": "ion",
    #    "Inputs": ["_s.h5", "_s_TECU.txt"],
    #    "Processor": "run_ranging.py",
    #    "Libraries": ["xlib/misc/hdf.py", "xlib/rng/icd.py"],
    #    "OutPrefix": "rng",
    #    "Outdir": "icd",
    #    "Outputs": ["_a.cluttergram.npy"],
    #},
]

def temptracklist(infile):
    """ Create a temporary file with one track in it,
    and return the path to it """
    logging.debug("Writing temporary track list for input file:");
    logging.debug(infile);
    temp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    temp.write(infile+'\n')
    temp.close()
    return temp.name

def getmtime(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = -1
    return mtime, path

def manual(cmd, infile):
    """ Interactively prompt whether to run a command """
    import getch
    print('Trackline: ' + infile);
    print('Command: ' + ' '.join(cmd));
    c = ' '
    while c not in 'ynq':
        print('(Y)es, (N)o, (Q)uit?', end=' ', flush=True);
        c = getch.getch().lower()
        print(c)
    if c == 'q':
        sys.exit(0)
    if c == 'y':
        subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='SHARAD Pipeline')
    parser.add_argument('-o', '--output', default='/disk/kea/SDS/targ/xtra/SHARAD',
                        help="Output base directory")

    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing (currently ignored)")

    grp_verb = parser.add_mutually_exclusive_group()
    grp_verb.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output in pipeline script")
    grp_verb.add_argument('-vv', action="store_true",
                        help="Display verbose output in pipeline script and subprocesses")


    parser.add_argument('-m', '--manual', action="store_true",
                        help="Prompt before running processors")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxrequests', type=int, default=0,
                        help="Maximum number of processing requests to process")
    # Use --maxrequests 1 instead of -1
    #parser.add_argument('-1', '--once', action="store_true",
    #                    help="Just run one processor and exit.")
    parser.add_argument('--ignorelibs', action='store_true',
                        help="Do not check times on libraries")
    parser.add_argument('--ignoretimes', action='store_true',
                        help="Do not any times")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if (args.verbose or args.vv) else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")

    with open(args.tracklist, 'r') as fin:
        lookup = [line.strip() for line in fin if line.strip() != '']
    logging.debug(lookup)

    # Build list of processes
    logging.info("Building task list")
    process_list = []
    path_outroot = args.output

    logging.debug("Base output directory: %s", path_outroot)

    nrequests = 0
    SHARADroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    for prod in PROCESSORS:
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
            #prefix = ''
            #inprefix = ''
            logging.info("Considering: %s", prod['Name'])
            #prefix = prod['InPrefix'] + '/'
            indir = os.path.join(path_outroot, prod['InPrefix'], path_file, prod['Indir']) 

            # Get the modification times for the input files
            for suffix in prod['Inputs']:
                # FIXME: This might be better if absolute paths are detected
                if suffix == '_a.dat' or suffix == '_s.dat':
                    oneinput = os.path.join(SHARADroot, path_file, data_file + suffix)
                else:
                    oneinput = os.path.join(indir, data_file + suffix)
                intimes.append(getmtime(oneinput))

            # Get modification time for the processor file and known input modules
            intimes.append(getmtime(prod['Processor']))

            for libname in prod['Libraries']:
                intimes.append(getmtime(os.path.join('..', libname)))

            prefix = prod['OutPrefix']
            outdir = os.path.join(path_outroot, prod['OutPrefix'], path_file, prod['Outdir'])

            for o in prod['Outputs']:
                output = os.path.join(outdir, data_file + o)
                outputs.append(output)
                outtimes.append(getmtime(output))

            if len(intimes) == 0:
                logging.error("No inputs for process")
            intimes.sort()

            if len(outtimes) == 0:
                logging.error("No outputs for process")
            outtimes.sort()

            # Print considered input files for equivalence checking
            for x in intimes:
                logging.debug("INPUT:  %0.0f %s", *x)
            for x in outtimes:
                logging.debug("OUTPUT: %0.0f %s", *x)


            if intimes[0][0] == -1:
                # TODO: logging
                print('Input missing.')
            elif intimes[-1][0] < outtimes[0][0]:
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
                    logging.info("Output: %s", output)
                    logging.debug("Processing %s", infile)
                    if prod['Processor'] == "run_rsr.py":
                        temp = None
                        cmd = ['./' + prod['Processor'], '-o', os.path.join(path_outroot, prod['OutPrefix']), orbit]
                    else:
                        temp = temptracklist(infile)
                        cmd = ['./' + prod['Processor'], '--tracklist', temp, '-o', os.path.join(path_outroot,prefix)]

                    if args.vv:
                        # Run processor with a verbose flag
                        cmd.append('-v')

                    logging.info("Invoking: %s", ' '.join(cmd))
                    if args.manual:
                        manual(cmd, infile)
                    elif args.dryrun:
                        logging.info("Dryrun")
                    else:
                        subprocess.run(cmd)
                    if temp: # Remove temp file if it was created
                        os.unlink(temp)

                    nrequests += 1
                    if args.maxrequests > 0 and nrequests >= args.maxrequests:
                        logging.info("Only process %d requests.  Quitting.", args.maxrequests)
                        return 0

        else:
            logging.debug('File already processed. Skipping %s', infile)
    return 0

if __name__ == "__main__":
    # execute only if run as a script
    sys.exit(main())
