#!/usr/bin/env python3

""" Functions for calculating whether input and output files are up to date,
and deleting outputs """

__authors__ = ['Gregory Ng, ngg at utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'March 10, 2024',
         'author': 'Gregory Ng, UTIG',
         'info': 'Initial version'},
}


import os
import logging
from typing import List
from collections import namedtuple


FileInfo = namedtuple('FileInfo', 'mtime path')


def file_processing_status(infiles: List[str], outfiles: List[str], check_mtimes=True):
    """ Compares input files and output files and returns a tuple of two strings
    describing input and output processing status
    If check_mtimes is True, then all output file modification times must be newer than all
    input modification times  and their output times are newer than all input files

    parameters:
    infiles: a list of strings giving paths to infiles
    outfiles: a list of strings giving paths to output files

    input_status:
    Is one of two choices
    'input_missing' - returned if any of the input files do not exist
    'input_ok' - returned if all input files exist

    output_status:
    'output_missing' - returned if any of the output files do not exist
    'output_stale' - returned if file modification time checking is enabled and
                     any of the output files are older than any of the inputs
    'output_exists' - returned if file modification time checking is disabled, any input files
                     are missing, and all output files exist and are newer than
                     any input files present
    'output_ok' - otherwise (all input and output files exist, and 
                  either modification time checking is disabled
                  or all output files are newer than input files

    Assume times of -1 or None mean the file is missing.
    If any input files are missing, returns True

    """

    assert len(infiles) > 0, "No input files specified"
    assert len(outfiles) > 0, "No output files specified"

    # Get existence and times of input files
    intimes = sorted(map(getmtime, infiles))
    outtimes = sorted(map(getmtime, outfiles))

    if intimes[0].mtime == -1: # Missing input files. Throw exception
        missing = list(filter(lambda finfo: finfo.mtime == -1, intimes))
        for finfo in missing:
            logging.debug("Missing input %s", finfo.path)
        input_status = 'input_missing'
    else:
        input_status = 'input_ok'


    if outtimes[0].mtime == -1:
        # At least one of the output files doesn't exist
        output_status = 'output_missing'
    else:
        # That newest input time is older than oldest output time
        # We know that all files exist
        if input_status == 'input_missing':
            # mtime comparison may be inaccurate
            if check_mtimes:
                output_status = 'output_exists'
            else:
                output_status = 'output_ok'
        else:
            output_ok = (not check_mtimes) or (intimes[-1].mtime < outtimes[0].mtime)
            output_status = 'output_ok' if output_ok else 'output_stale'

    return input_status, output_status


def delete_files(files, delete_parents=True):
    """ Delete the output files if they exist,
    and also remove parent directories """
    for f in files:
        if f is None:
            continue
        if os.path.exists(f):
            os.unlink(f)
        if delete_parents:
            try:
                os.rmdir(os.path.dirname(f))
            except OSError:
                pass # We don't care if it fails a few times


def all_files_exist(files: List[str]):
    """ Return true if all files in the list exist """
    for f in files:
        if not os.path.exists(f):
            return False
    return True

def getmtime(path):
    """ Get file modification time or -1 if file does not exist
    Using the value -1 allows us to sort nonexistent files to the beginning
    """
    try:
        mtime = int(os.path.getmtime(path))
    except OSError: # file doesn't exist
        mtime = -1
    return FileInfo(mtime, path)

