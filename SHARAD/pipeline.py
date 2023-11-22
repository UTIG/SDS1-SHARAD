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
# TODO: --ignoretimes option. Should we call this --force ?


"""

SYNOPSIS

pipeline.py orchestrates the running of SDS data processing scripts based on
looking for the expected inputs and corresponding output files.

To run pipeline.py, you usually provide it with a list of input files to examine.

To only see what would be run, run with the -n option.

./pipeline.py -i elysium.txt -n

To run interactively and prompt before executing each subprocess, run with -m

./pipeline.py -i elysium.txt -m

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
import psutil


PROCESSORS = [
    {
        "Name": "Run Range Compression",
        "InPrefix": '',
        "Inputs": ["{0[orig_root]}/{0[path_file]}/{0[data_file]}_a.dat",
                    "{0[orig_root]}/{0[path_file]}/{0[data_file]}_s.dat"],

        "Processor": "run_rng_cmp.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/cmp/plotting.py", "xlib/cmp/rng_cmp.py"],
        "Outputs": [
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt",
                    ]

    },
    {
        "Name": "Run Altimetry",
        "InPrefix": "cmp",
        # Uses the outputs from run_rng_cmp.py
        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
        "Processor": "run_altimetry.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        "Outputs": ["{0[targ_root]}/alt/{0[path_file]}/beta5/{0[data_file]}_a.h5"],
    },
    {
        "Name": "Run Surface",
        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5"],
        "Processor": "run_surface.py",
        # The libraries for rsr and subradar are no longer in the repository; they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "OutPrefix": "srf",
        "Outputs": ["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
    },
    {
        "Name": "Run RSR",
        "Inputs": ["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
        "Processor": "run_rsr.py",
        # The libraries for rsr and subradar are no longer in the repository; they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "OutPrefix": "rsr",
        "Outputs": ["{0[targ_root]}/rsr/{0[path_file]}/cmp/{0[data_file]}.txt"],
    },
#    Don't do SAR on TACC
#    {
#        "Name": "Run SAR",
#        "InPrefix": "cmp",
#        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
#                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
#        "Processor": "run_sar2.py",
#        "Libraries": ["xlib/sar/sar.py", "xlib/sar/smooth.py",
#                      "xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        #"OutPrefix": "foc",
        # TODO: GNG -- suggest spaces become underscores
        # Other flags might result in other outputs.  For these we need a different command line
        # option makes sense.
#        "Outputs": ["{0[targ_root]}/foc/{0[path_file]}/5m/5 range lines/40km/{0[data_file]}_s.h5"],
#    },
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

def run_command(cmd, output):
    output = output.replace("targ", "note")
    if (not os.path.exists(os.path.dirname(output))):
        os.makedirs(os.path.dirname(output))
    out = open(output+".stdout", "w")
    err = open(output+".stderr", "w")
    return subprocess.run(cmd, stdout=out, stderr=err)

def temptracklist(infile):
    """ Create a temporary file with one track in it,
    and return the path to it """
    logging.debug("Writing temporary track list for input file:");
    logging.debug(infile);
    temp = tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix='pipeline_tracklist_')
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

def read_tracklist(filename, SHARADroot='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'):
    """ Read the tracklist and parse for relevant information
    The tracklist contains a list of "orig" input filenames.
    """
    list_items = []
    with open(filename, 'rt') as fin:
        for line in fin:
            infile = line.strip()
            if not infile:
                continue

            path_file = os.path.relpath(infile, SHARADroot)
            #path_file = infile.replace(SHARADroot, '')
            assert path_file.endswith('_a.dat')
            data_file = os.path.basename(path_file).replace('_a.dat', '')
            path_file = os.path.dirname(path_file)
            #root_file, ext_file = os.path.splitext(data_file)
            m = re.search('(e_\d{7}_\d{3}_ss\d{2}_\d{3}_a)', infile)
            assert m # FIXME error checking is needed here
            orbit = m.group(1)

            item = { # variables for input/output file calculation
                'infile': infile,
                'path_file': path_file, # Relative path to file
                # basename for data file (excluding path and suffix)
                'data_file': data_file,
                'orbit': orbit,
            }
            list_items.append(item)
    return list_items


def main():
    parser = argparse.ArgumentParser(description='SHARAD Pipeline')
    parser.add_argument('-o', '--output', default='/disk/kea/SDS/targ/xtra/SHARAD',
                        help="Output base directory")

    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help="Number of jobs (cores) to use for processing")

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
    parser.add_argument('-r', '--maxrequests', type=int, default=0,
                        help="Maximum number of processing requests to process")
    # Use --maxrequests 1 instead of -1
    #parser.add_argument('-1', '--once', action="store_true",
    #                    help="Just run one processor and exit.")
    parser.add_argument('--ignorelibs', action='store_true',
                        help="Do not check times on libraries")
    parser.add_argument('--ignoretimes', action='store_true',
                        help="Do not check any times")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if (args.verbose or args.vv) else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")

    lookup = read_tracklist(args.tracklist)


    # Build list of processes
    logging.info("Building task list")
    process_list = []
    # strip trailing slash
    args.output = args.output.rstrip('/')
    path_outroot = args.output

    logging.debug("Base output directory: %s", path_outroot)

    nrequests = 0
    SHARADroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    for prod in PROCESSORS:
        indir = ''
        proc = ''
        results = []
        #outdir = ''
        for lookup_line, item in enumerate(lookup, start=1):
            infile = item['infile']
            ivars = item.copy()
            ivars['orig_root'] = SHARADroot.rstrip('/')
            ivars['targ_root'] = args.output

            intimes = [] # input file names and modification times
            outtimes = [] # output file names and modification times
            logging.info("Considering %s track %d", prod['Processor'], lookup_line)

            # Get the modification times for the input files
            for fmtstr in prod['Inputs']:
                oneinput = fmtstr.format(ivars)
                intimes.append(getmtime(oneinput))

            if not args.ignorelibs:
                # Get modification time for the processor file and known input modules
                intimes.append(getmtime(prod['Processor']))

                for libname in prod['Libraries']:
                    intimes.append(getmtime(os.path.join('..', libname)))
            else:
                logging.debug("Not checking code modification times")

            for fmtstr in prod['Outputs']:
                output = fmtstr.format(ivars) #output = os.path.join(outdir, data_file + o)
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


            if intimes[0][0] == -1: # Missing input files
                for tim, filename in intimes:
                    if tim == -1:
                        logging.info("%s is missing input %s", prod['Name'], filename)
            elif intimes[-1][0] < outtimes[0][0]:
                logging.info("Up to date.")
            else:
                if not args.ignoretimes or outtimes[0][0] == -1:
                    if (outtimes[0][0] == -1):
                        logging.info('Ready to process (no output file).')
                    else:
                        logging.info('Ready to process (old output file).')
                        logging.info('Deleting old files.')
                        for _, output in outtimes:
                            os.unlink(output)
                            try:
                                os.rmdir(os.path.dirname(output))
                            except OSError:
                                # We don't care if it fails a few times
                                pass
                    logging.info("Output: %s", output)
                    logging.debug("Processing %s", infile)
                    # TODO: move this over to the dict
                    pool = multiprocessing.Pool(args.jobs)
                    if prod['Processor'] in ["run_rsr.py", "run_surface.py"]:
                        temp = None
                        cmd = ['./' + prod['Processor'], item['orbit'], '-j 1']
                    else:
                        temp = temptracklist(infile)
                        cmd = ['./' + prod['Processor'], '--tracklist', temp, '-j 1']
                        if prod['Processor'] == "run_sar2.py":
                            # Add targ path which it uses to find input files
                            cmd += ['-i', ivars['targ_root']]

                    if args.vv:
                        # Run processor with a verbose flag
                        cmd.append('-v')

                    logging.info("Invoking: %s", ' '.join(cmd))
                    if args.manual:
                        manual(cmd, infile)
                    elif args.dryrun:
                        logging.info("Dryrun")
                    else:
                        results.append(pool.apply_async(run_command, [cmd, output]))
                        #subprocess.run(cmd)
                    if False: # Remove temp file if it was created
                        # Don't remove it before it is used!
                        os.unlink(temp)

                    nrequests += 1
                    if args.maxrequests > 0 and nrequests >= args.maxrequests:
                        logging.info("Only process %d requests.  Finishing.", args.maxrequests)
                        # Wait for everything using this processor to finish
                        for result in results:
                            result.get()
                        logging.info("Only process %d requests.  Quitting.", args.maxrequests)
                        return 0

        # FIXME; This else got lost somehow.
        #else:
            #logging.debug('File already processed. Skipping %s', infile)

        # Wait for everything using this processor to finish
        logging.info("Waiting for " + prod["Name"] + " to finish.");
        for result in results:
            logging.debug("CPU percent: " + str(psutil.cpu_percent()))
            result.get()
    logging.info("All done.");
    return 0

if __name__ == "__main__":
    # execute only if run as a script
    sys.exit(main())
