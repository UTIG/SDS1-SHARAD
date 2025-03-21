#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.3'
__history__ = {
    '1.3':
        {'date': 'March 12 2024',
         'author': 'Gregory Ng, UTIG',
         'info': 'Reworking file calculations and script integration'},
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
# TODO: --ignoretimes option. Should we call this --force or --overwrite ?
# TODO: Update processing scripts to accept a list of product IDs on the command line,
#       so we don't have to do this temporary file business
# TODO: rewrite this pipeline script to accept either a list of files or
#       just the list of sharad product IDs.

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
import logging
import argparse
import subprocess
import tempfile
from typing import List, Dict, Any
from pathlib import Path

p1 = Path(__file__).parent
sys.path.insert(1, str(p1.resolve()))

from run_rng_cmp import run_jobs, add_standard_args, process_product_args

p1 = Path(__file__).parent / '..' / 'xlib'
sys.path.insert(1, str(p1.resolve()))

from sharad.sharadenv import SHARADFiles

from misc.fileproc import file_processing_status, delete_files


TARGDIR_TYPICAL = None
#import psutil

# Assumes dictionary is ordered (python3.7)
assert sys.version_info[0:2] >= (3,7), "Dictionaries aren't ordered!"
# key is previous 'OutPrefix'
PROCESSORS = {
    'cmp': {
        "Name": "Run Range Compression",
        "InPrefix": '',
        #"Inputs": ["{0[orig_root]}/{0[path_file]}/{0[data_file]}_a.dat",
        #            "{0[orig_root]}/{0[path_file]}/{0[data_file]}_s.dat"],
        "Inputs": ["edr"],

        "Processor": "run_rng_cmp.py",
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/cmp/plotting.py", "xlib/cmp/rng_cmp.py"],
        "Outputs": [
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
                    "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt",
                    ],
        "internal_parallelism": 1,
    },
    'alt': {
        "Name": "Run Altimetry",
        "InPrefix": "cmp",
        # Uses the outputs from run_rng_cmp.py
        # TODO: it also uses the science EDR, but should we include that here?
        #"Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
        #            "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
        "Inputs": ["cmp"],
        "Processor": "run_altimetry.py",
        "args": ['--finethreads', '4'],
        "Libraries": ["xlib/cmp/pds3lbl.py", "xlib/altimetry/beta5.py"],
        "Outputs": ["{0[targ_root]}/alt/{0[path_file]}/beta5/{0[data_file]}_a.h5"],
        "internal_parallelism": 4, # --finethreads
    },
    # Note: it is super inefficient to run multiple copies of run_surface
    # because run_surface spends most of its time indexing the sharad environment
    'srf': {
        "Name": "Run Surface",
        "InPrefix": "alt",
        "Inputs": ["alt", "cmp"],
        #"Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
        #           "{0[targ_root]}/alt/{0[path_file]}/beta5/{0[data_file]}_a.h5"],
        "Processor": "run_surface.py",
        # The libraries for rsr and subradar are no longer in the repository;
        # they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "Outputs": ["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
        "internal_parallelism": 1,
    },
    'rsr': {
        "Name": "Run RSR",
        "InPrefix": "srf",
        "Inputs": ["srf"], #["{0[targ_root]}/srf/{0[path_file]}/cmp/{0[data_file]}.txt"],
        "Processor": "run_rsr.py",
        # run_rsr can use multiple threads at the same time because
        # the parallelism is at a lower level
        "args": ['-j', '4'],
        # The libraries for rsr and subradar are no longer in the repository;
        # they are a pip package.
        "Libraries": ["SHARAD/SHARADEnv.py"],
        "Outputs": ["{0[targ_root]}/rsr/{0[path_file]}/cmp/{0[data_file]}.txt"],
        "internal_parallelism": 4,
    },
    'foc': {
        "Name": "Run SAR",
        "InPrefix": "cmp",
        "Inputs": ["cmp"],
        #"Inputs": ["{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s.h5",
        #            "{0[targ_root]}/cmp/{0[path_file]}/ion/{0[data_file]}_s_TECU.txt"],
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


def run_command(product_id: str, tasknum: int, cmd: List[str],
                output: str, delete_before: List[str], delete_after: List[str]):
    logdir = os.path.dirname(output)
    assert '/targ/' in logdir, "logdir=" + logdir
    logdir = logdir.replace('/targ/', '/note/')
    logfile = os.path.join(logdir, product_id)
    os.makedirs(logdir, exist_ok=True)
    delete_files(delete_before)
    logging.debug("CMD: %s", ' '.join(cmd))
    logging.debug("Task %d writing to %s.stdout", tasknum, logfile)
    logging.debug("Task %d writing to %s.stderr", tasknum, logfile)
    with open(logfile+".cmd", "wt") as outfh:
        print(' '.join(cmd), file=outfh)
    with open(logfile+".stdout", "w") as outfh, \
         open(logfile+".stderr", "w") as errfh:
        res = subprocess.run(cmd, stdout=outfh, stderr=errfh)

    if res.returncode == 0:
        delete_files(delete_after)
    logging.debug("Task %d done res=%d", tasknum, res.returncode)
    return res

def temptracklist(infile: str):
    """ Create a temporary file with one track in it,
    and return the path to it """
    logging.debug("Writing temporary track list for input file: %s", infile)
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix='pipeline_tracklist_') as fhtemp:
        print(infile, file=fhtemp)
        return fhtemp.name


def manual(cmd, infile): # pragma: no cover
    """ Interactively prompt whether to run a command """
    import getch
    print('Trackline: ' + infile)
    print('Command: ' + ' '.join(cmd))
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
    parser.add_argument('--tasks', default='rsr,foc', help="Comma-separated list of tasks to run")

    add_standard_args(parser, script='pipeline')

    grp_verb = parser.add_mutually_exclusive_group()
    grp_verb.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output in pipeline script")
    grp_verb.add_argument('-vv', action="store_true",
                        help="Display verbose output in pipeline script and subprocesses")


    parser.add_argument('-m', '--manual', action="store_true",
                        help="Prompt before running processors")

    parser.add_argument('--ignorelibs', action='store_true',
                        help="Do not check times on processing software (libraries)")
    parser.add_argument('--ignoretimes', action='store_true',
                        help="Process data regardless of input/output file times")

    args = parser.parse_args()

    loglevel = logging.DEBUG if (args.verbose or args.vv) else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")

    global TARGDIR_TYPICAL
    TARGDIR_TYPICAL = os.path.join(args.SDS, 'targ/xtra/SHARAD')
    if args.output is None:
        args.output = TARGDIR_TYPICAL
    # strip trailing slash
    args.output = args.output.rstrip('/')
    assert '/targ/' in args.output, \
           "output directory must contain /targ/ (typically %r)" % TARGDIR_TYPICAL

    # Root ORIG directory
    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist = process_product_args(args.product_ids, args.tracklist, sfiles)

    # Build list of processes
    logging.info("Building task list")
    process_list = []

    logging.debug("Base output directory: %s", args.output)

    args.tasks = args.tasks.split(',') # change to a list of strings
    tasks = build_task_order(args.tasks, PROCESSORS)

    logging.debug("Tasks: %r", tasks)
    for outprefix in tasks:
        prod = PROCESSORS[outprefix]
        process_list = []
        for productnum, product_id in enumerate(productlist, start=1):
            # Limit number of jobs as requested on command line
            if args.maxtracks > 0 and productnum > args.maxtracks:
                break

            logging.debug("%s track %d: %s", prod['Processor'], productnum, product_id)
            infiles, outfiles = calculate_input_output_files(product_id, prod, not args.ignorelibs, sharad_root, args.output, outprefix, sfiles)
            cmd, tempfile1, filestatus = build_command(product_id, prod, infiles, outfiles, not args.ignoretimes,
                                                       args.vv, args.output)

            if cmd is None:
                logging.info("%s track %d: skipping %s (%s)", outprefix, productnum, product_id, ' '.join(filestatus))
                continue

            logging.info("%s track %d: %s %s", outprefix, productnum, product_id, ' '.join(cmd))
            process_list.append({
                'product_id': product_id,
                'tasknum': productnum,
                'cmd': cmd,
                'output': outfiles[0],
                'delete_before': outfiles,
                'delete_after': [tempfile1],
            })

        # end for lookup (end create job list)

        if args.dryrun:
            logging.info("Dryrun done for task %s", outprefix)
            continue # don't run

        njobs = args.jobs // prod.get('internal_parallelism', 1)
        run_jobs(run_command, process_list, njobs)
        logging.info("All done with %s.", outprefix)
    return 0

def calculate_input_output_files(product_id: str, prod: Dict[str, Any], include_libraries: bool,
                                 sharad_root: str, data_path: str, outprefix: str, sfiles: SHARADFiles):
    """ Figure out the required input and output files for this processing step
    Parameters:
    item
    prod is the desired processing step
    include_libraries - if True, include code in list of input files

    returns two lists, infiles, and outfiles listing the input and output files
    for this processor
    """
    infiles = []

    # Get the paths for the input files
    for inprefix in prod['Inputs']:
        assert len(inprefix) == 3, "Unexpected input prefix " + inprefix
        infiles.extend(sfiles.product_paths(inprefix, product_id).values())

    if include_libraries: #not args.ignorelibs:
        # Include the processor file and known input modules
        codedir = os.path.dirname(__file__)
        infiles.append(os.path.abspath(os.path.join(codedir, prod['Processor'])))
        codedir = os.path.abspath(os.path.join(codedir, '..'))
        for libname in prod['Libraries']:
            infiles.append(os.path.join(codedir, libname))

    outfiles = list(sfiles.product_paths(outprefix, product_id).values())

    for f in infiles:
        logging.debug("Input:  %s", f)
    for f in outfiles:
        logging.debug("Output: %s", f)

    return infiles, outfiles

def build_command(product_id: str, prod: Dict[str, Any],
                  infiles: List[str], outfiles: List[str],
                  check_mtimes: bool,
                  verbose_processor: bool,
                  data_path: str):
    """ Build command line arguments for the processor
    prod is a dictionary from the processor information
    returns:
    cmd - the command arguments as an List of strings
    tempfile1 - temporary file created (None if no file was created)
    """

    filestatus =  file_processing_status(infiles, outfiles, check_mtimes=check_mtimes)

    if filestatus == ('input_ok', 'output_ok'):
        logging.debug("Outputs are up to date for %s", product_id)
        return None, None, filestatus

    if filestatus[0] == 'input_missing' and filestatus[1] != 'output_ok':
        logging.debug("Missing some input, can't process")
        return None, None, filestatus

    proc = os.path.join(os.path.dirname(__file__), prod['Processor'])
    proc = os.path.abspath(proc)


    # Build the commmand
    # TODO: write a function
    # TODO: move this over to the dict so we don't need an if statement here
    cmd = ['python3']
    if prod['Processor'] in ['run_surface.py']:
        tempfile1 = None
        cmd += [proc, '--overwrite', '-j', '1', product_id]
    elif prod['Processor'] in ['run_rsr.py']: # no -j option for run_rsr, that's in args
        tempfile1 = None
        cmd += [proc, '--overwrite', product_id]
    elif prod['Processor'] in ['run_rng_cmp.py', 'run_altimetry.py']:
        tempfile1 = None
        cmd += [proc, '--overwrite', '-j', '1', product_id]
    else:
        tempfile1 = temptracklist(product_id)
        cmd += [proc, '--tracklist', tempfile1, '-j', '1']
        if prod['Processor'] == "run_sar2.py":
            # Add targ path which it uses to find input files
            cmd += ['-i', data_path]

    # Pass the targ directory to the lower level script
    # if we requested something different
    if data_path != TARGDIR_TYPICAL:
        cmd = cmd[0:2] + ['-o', data_path] + cmd[1:]

    if 'args' in prod:
        cmd += prod['args']

    if verbose_processor:
        # Run processor with a verbose flag
        cmd.append('-v')

    return cmd, tempfile1, filestatus


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





if __name__ == "__main__":
    # execute only if run as a script
    sys.exit(main())
