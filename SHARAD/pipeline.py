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
from collections import namedtuple
from typing import List

#import psutil
FileInfo = namedtuple('FileInfo', 'mtime path')

# Assumes dictionary is ordered (python3.7)
assert sys.version_info[0:2] >= (3,7), "Dictionaries aren't ordered!"
# key is previous 'OutPrefix'
PROCESSORS = {
    'cmp': {
        "Name": "Run Range Compression",
        "InPrefix": '',
        "Inputs": ["{0[orig_root]}/{0[path_file]}/{0[data_file]}_a.dat",
                    "{0[orig_root]}/{0[path_file]}/{0[data_file]}_s.dat"],

        "Processor": "run_rng_cmp.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/cmp/plotting.py", "xlib/cmp/rng_cmp.py"],
        "Outputs": [
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt",
                    ],
    },
    'alt': {
        "Name": "Run Altimetry",
        "InPrefix": "cmp",
        # Uses the outputs from run_rng_cmp.py
        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
        "Processor": "run_altimetry.py",
        "args": ['--finethreads', '8'],
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        "Outputs": ["{0[targ_root]}/alt/{0[path_file]}/beta5/{0[data_file]}_a.h5"],
    },
    # Note: it is super inefficient to run multiple copies of run_surface
    # because run_surface spends most of its time indexing the sharad environment
    'srf': {
        "Name": "Run Surface",
        "InPrefix": "alt",
        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5"],
        "Processor": "run_surface.py",
        # The libraries for rsr and subradar are no longer in the repository; they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "Outputs": ["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
    },
    'rsr': {
        "Name": "Run RSR",
        "InPrefix": "srf",
        "Inputs": ["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
        "Processor": "run_rsr.py",
        # The libraries for rsr and subradar are no longer in the repository; they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "Outputs": ["{0[targ_root]}/rsr/{0[path_file]}/cmp/{0[data_file]}.txt"],
    },
    'foc': {
        "Name": "Run SAR",
        "InPrefix": "cmp",
        "Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
        "Processor": "run_sar2.py",
        "Libraries": ["xlib/sar/sar.py", "xlib/sar/smooth.py",
                      "xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        # TODO: GNG -- suggest spaces become underscores
        # Other flags might result in other outputs.  For these we need a different command line
        # option makes sense.
        "Outputs": ["{0[targ_root]}/foc/{0[path_file]}/5m/5 range lines/40km/{0[data_file]}_s.h5"],
    },
    # Run ranging needs crossovers, so needs a track file with pairs of tracks
    # and record numbers.  This is a special data product so we can't run it
    # automatically.
    'rng': {
        "Name": "Run Ranging",
        "InPrefix": "cmp",
        "Indir": "ion",
        "Inputs": ["_s.h5", "_s_TECU.txt"],
        "Processor": "run_ranging.py",
        "Libraries": ["xlib/misc/hdf.py", "xlib/rng/icd.py"],
        "Outdir": "icd",
        "Outputs": ["_a.cluttergram.npy"],
    },
}

def delete_files(filelist: List[str], missing_ok=True):
    for file in filelist:
        try:
            if file is not None:
                os.unlink(file)
        except OSError:
            if not missing_ok:
                raise
def run_command(tasknum: int, cmd: str, output: str, delete_before: List[str], delete_after: List[str]):
    output = output.replace("/targ/", "/note/")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    delete_files(delete_before)
    logging.debug("CMD: %s", repr(cmd))
    logging.debug("Task %d writing to %s.stdout", tasknum, output)
    logging.debug("Task %d writing to %s.stderr", tasknum, output)
    with open(output+".stdout", "w") as outfh, \
         open(output+".stderr", "w") as errfh:
        res = subprocess.run(cmd, stdout=outfh, stderr=errfh)

    if res.returncode == 0:
        delete_files(delete_after)
    logging.debug("Task %d done res=%d", tasknum, res.returncode)
    return res

def temptracklist(infile: str, tasknum: int):
    """ Create a temporary file with one track in it,
    and return the path to it """
    logging.debug("Task %d writing temporary track list for input file:", tasknum);
    logging.debug(infile);
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix='pipeline_tracklist_') as fhtemp:
        print(infile, file=fhtemp)
        return fhtemp.name

def getmtime(path):
    try:
        mtime = int(os.path.getmtime(path))
    except OSError: # file doesn't exist
        mtime = -1
    return FileInfo(mtime, path)

def manual(cmd, infile): # pragma: no cover
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

def read_tracklist(filename: str, sharad_root: str):
    """ Read the tracklist and parse for relevant information
    The tracklist contains a list of "orig" input filenames.
    """
    list_items = []
    with open(filename, 'rt') as fin:
        for line in fin:
            infile = line.strip()
            if not infile or infile.startswith('#'):
                continue

            path_file = os.path.relpath(infile, sharad_root)
            #path_file = infile.replace(sharad_root, '')
            assert path_file.endswith('_a.dat'), path_file
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
    parser.add_argument('tasks', nargs='*', help="Tasks to run",
                        default=['rsr', 'foc'])
    parser.add_argument('-o', '--output', default=None,
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

    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                        help="Root directory (default: environment variable SDS)")

    parser.add_argument('-r', '--maxrequests', type=int, default=0,
                        help="Maximum number of processing requests to process")
    parser.add_argument('--ignorelibs', action='store_true',
                        help="Do not check times on processing software (libraries)")
    parser.add_argument('--ignoretimes', action='store_true',
                        help="Process data regardless of input/output file times")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if (args.verbose or args.vv) else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")


    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')
    # strip trailing slash
    args.output = args.output.rstrip('/')

    # Root ORIG directory
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD/EDR/')
    lookup = read_tracklist(args.tracklist, sharad_root=sharad_root)


    # Build list of processes
    logging.info("Building task list")
    process_list = []

    logging.debug("Base output directory: %s", args.output)

    tasks = build_task_order(args.tasks, PROCESSORS)
    cwd = os.path.dirname(__file__)
    if cwd == '':
        cwd = '.'

    logging.debug("Tasks: %r", tasks)
    for outprefix in tasks:
        prod = PROCESSORS[outprefix]
        indir = ''
        proc = ''
        jobs = []
        results = []
        #outdir = ''
        for lookup_line, item in enumerate(lookup, start=1):
            infile = item['infile']
            ivars = item.copy()
            ivars['orig_root'] = sharad_root.rstrip('/')
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

            if not args.ignoretimes and is_output_updated(intimes, outtimes, prod["Name"]):
                logging.debug("Outputs are up to date")
                continue

            # Files to delete before running task
            delete_before = [x.path for x in outtimes]

            # TODO: move this over to the dict
            if prod['Processor'] in ["run_rsr.py", "run_surface.py"]:
                tempfile1 = None
                cmd = [os.path.join(cwd, prod['Processor']), item['orbit'], '--delete', '-j', '1']
            else:
                tempfile1 = temptracklist(infile, lookup_line)
                cmd = [os.path.join(cwd, prod['Processor']), '--tracklist', tempfile1, '-j', '1']
                if prod['Processor'] == "run_sar2.py":
                    # Add targ path which it uses to find input files
                    cmd += ['-i', ivars['targ_root']]

            if 'args' in prod:
                cmd += prod['args']

            if args.vv:
                # Run processor with a verbose flag
                cmd.append('-v')

            logging.info("Task %d: %r", lookup_line, cmd)
            jobs.append((run_command, [lookup_line, cmd, output, delete_before, [tempfile1]]))
            # Limit number of jobs as requested on command line
            if args.maxrequests > 0 and len(jobs) >= args.maxrequests:
                break
        # end for lookup (end create job list)

        if args.dryrun:
            logging.info("Dryrun")
            continue # don't run


        # Now that job list has been created, execute it either single-threaded
        # or using multiprocessing
        if args.jobs <= 1 or len(jobs) <= 1:
            for f_run, runargs in jobs:
                f_run(*runargs)
        else:
            # TODO: sort jobs from longest to shortest to improve parallelism,
            # or allow the processor to do this
            with multiprocessing.Pool(args.jobs) as pool:
                results = []
                # TODO: map_async or starmap?
                for f_run, runargs in jobs:
                    results.append(pool.apply_async(f_run, runargs))
                    #results.append(pool.apply_async(run_command, [cmd, output]))

                # Wait for everything using this processor to finish
                logging.info("Waiting for " + prod["Name"] + " to finish.");
                for n, result in enumerate(results, start=1):
                    #logging.debug("CPU percent: " + str(psutil.cpu_percent()))
                    logging.debug("Result %d of %d done", n, len(jobs))
                    logging.debug("Command %d was %r", n, jobs[n-1][1])
                    result.get()
        logging.info("All done.");
    return 0

def build_task_order(tasks, processors):
    """ Given a list of task names from the command line,
    get the list of all the tasks to be done.
    This currently assumes just one prerequisite, but that would be
    easy to fix.
    Tasks are output in an order that processes one final product first then
    goes to the next "final" product.
    """
    # Construct list of dependencies in reverse order
    tasks_out1 = []

    for task in reversed(tasks):
        tasks_out1.append(task)
        print("task=", task)
        prereq = processors[task]['InPrefix']
        assert isinstance(prereq, str), prereq
        while prereq != '': # assumes no cycles or branches in dependencies
            tasks_out1.append(prereq)
            try:
                prereq = processors[prereq]['InPrefix']
            except KeyError:
                print(prereq)
                raise
            assert isinstance(prereq, str), prereq

    tasks_done = set()
    tasks_out2 = []
    for task in reversed(tasks_out1): # return tasks to forward order
        if task not in tasks_done:
            tasks_done.add(task)
            tasks_out2.append(task)
    return tasks_out2

def delete_output_files(outtimes):
    """ Delete the output files if they exist,
    and also remove parent directories """
    logging.info('Deleting old files.')
    for mtime, output in outtimes:
        if mtime == -1:
            continue
        os.unlink(output)
        try:
            os.rmdir(os.path.dirname(output))
        except OSError:
            pass # We don't care if it fails a few times

def is_output_updated(intimes, outtimes, name):
    """ Compares input times to output times, and
    returns true if all output times are later (greater) than
    all input times and all output files exist
    Assume times of -1 or None mean the file is missing.
    If any input files are missing, returns True
    """
    if len(intimes) == 0:
        logging.error("No inputs for process")
    intimes.sort()

    if len(outtimes) == 0:
        logging.error("No outputs for process")
    outtimes.sort()

    # Print considered input files for equivalence checking
    for x in intimes:
        logging.debug("INPUT:  %r %s", *x)
    for x in outtimes:
        logging.debug("OUTPUT: %r %s", *x)


    if intimes[0].mtime == -1: # Missing input files
        for finfo in intimes:
            if finfo.mtime == -1:
                logging.info("%s is missing input %s", name, finfo.path)
        return False
    elif (outtimes[0].mtime != -1) and (intimes[-1].mtime < outtimes[0].mtime):
        logging.info("Output is up to date.")
        return True

    if outtimes[0].mtime == -1:
        # There is at least one missing file
        logging.info('Ready to process (no output file).')
        return False

    logging.info('Ready to process (old output file).')

    logging.info('Deleting old files.')
    for _, output in outtimes:
        os.unlink(output)
        try:
            os.rmdir(os.path.dirname(output))
        except OSError:
            # We don't care if it fails a few times
            pass

    return False


if __name__ == "__main__":
    # execute only if run as a script
    sys.exit(main())
