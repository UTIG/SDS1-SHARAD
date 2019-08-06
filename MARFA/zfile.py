#!/usr/bin/env python3

# Read ztim formatted binary and ascii data
# Was previously known as read_ztim.py
# Adapted for compatibility with python2.7 or python3

# Note: There is a prety good implementation of zread in
# $WAIS/syst/linux/src/deva/zutils.py
# that combines reading and writing in pure python

import re
import logging
import datetime
import collections
import traceback
import glob

import numpy as np

Ztim = collections.namedtuple('Ztim', ('year', 'doy', 'isec'))



epoch_J2000 = datetime.datetime(2000,1,1,12,0,0,0)
epoch_unix  = datetime.datetime(1970,1,1,0,0,0,0) # aka epoch posix


def read_ztim(filename, field_names=None):
    '''
    Assumes that filename is a binary zfile. Reads in the values, returns
    as a numpy.recarray, indexed by the provided field names. 
    Default names for data fields are ['d0', 'd1', ...]
    '''
    ztim_format = 'S1,i2,i2,i4'
    ztim_names = ['PLUS', 'YEAR', 'DOY', 'itim']
    field_format = ',f8'
    

    with open(filename, 'rb') as zfile:
        # The format will be something like:
        # 'zfil1zeeee\n'
        # where the number of e's indicates how many fields after the ztim.
        zformat = zfile.readline();
        if not zformat.startswith(b'zfil1z'):
            logging.error("read_ztim: Input is not a zfile!")
            return None

        num_fields = len(zformat.lstrip(b'zfil1z').rstrip())
        zformat = ztim_format + num_fields * field_format
        print(zformat)

        if field_names is None:
            field_names = ['d'+str(elem) for elem in range(num_fields)]
        print(field_names)

        if len(field_names) != num_fields:
            logging.warning("read_ztim: Input field names of wrong length! zformat is %r, names are %r in file %s" 
                   % (zformat, field_names, filename))


        # QUESTION: I'm still not sure what the desired format should be. 
        # (I'd rather a pure array that I can slice in 2 dimensions, w/o '+')
        #logging.debug( "{:s},{:s},{:s}".format(str(zfile), str(zformat), ",".join(field_names)) )
        data = np.core.records.fromfile(zfile, formats=zformat, names=ztim_names + field_names, aligned=True, byteorder='>')
    return data


##################################################
# Routines for reading a text ztim file
##################################################

def read_ztim_text(filename):
    """ Read a text file and generate tuples for each ztim
    Example ztim string: (2016, 13, 357433923)
    This string represents year, day of year, and 1/10000 seconds of day
    Returns a tuple of year, day of year, and 1/10000 seconds of day
    """
    pat_ztim = re.compile(r'\((\d+),\s*(\d+),\s*(\d+)\)')
    with open(filename, "r") as fin:
        for i,line in enumerate(fin):
            m = pat_ztim.match(line)
            if not m:
                logging.warning("Malformed ztim '{:s}' at {:s} line {d}".format(line.strip(), filename, i+1))
                yield (float('nan'), float('nan'), float('nan'))
                continue
            # Year, day of year,1/10000 seconds of day
            yield Ztim._make([ int(s) for s in m.group(1, 2, 3) ])

def read_ztim_text_ascolumns(filename):
    """
    Read a text file of ztims, but return in a column-type output.
    The result is three lists of year, day of year, and second of day
    """
    return zip( * read_ztim_text(filename)  )


def parse_ztim_str(s):
    pat_ztim = re.compile(r'\((\d+),\s*(\d+),\s*(\d+)\)')
    m = pat_ztim.match(s)
    if not m:
        logging.warning("Malformed ztim '{:s}'".format(s.strip()))
        return (0, 0, float('nan'))
    # Year, day of year,1/1000 seconds of day
    fields = [ int(s) for s in m.group(1, 2, 3) ]
    return tuple(fields)

def get_ztim_range_posix(filename):
    first,last = get_ztim_range(filename)
    return (ztim_to_posix(first),ztim_to_posix(last))

def get_ztim_range(filename, head_index=0, tail_index=-1):
    """ Get the first and last ztims, and return as a timestamp 
    head_index specifies which line to use for the range.
    0 is the first line, 1 is the second line, etc.

    tail_index specifies which line to use for the last ztim.
    -1 is the last line, - is the second-to-last line, etc.
    """
    with open(filename, 'rb') as fh:
        for i in range(0,head_index+1):
            line1 = next(fh)
        first = parse_ztim_str(line1.decode())

        fh.seek(0, 2) # seek to end of file
        # Seek far enough back that we get all lines, but not before beginning of file
        fh.seek(max(22*tail_index-50, -fh.tell()), 2)
        last = parse_ztim_str(fh.readlines()[tail_index].decode())

    return (first,last)


def ztim_to_seconds(ztim, epoch=epoch_J2000):
    """ 
    Convert a ztim tuple (year, day of year, 1/10000 seconds of day)
    to the number of seconds since a given epoch.
    epoch is a datetime.datetime object representing the start of the epoch.
    By default, uses the J2000 epoch
    """
    #tz_utc=None

    # TODO: accelerate and store the result in a dict
    datetime_yyyy = datetime.datetime(ztim[0],1,1,0,0,0)
    # round() might reduce floating point precision problems.
    secs_yyyy = (datetime_yyyy - epoch).total_seconds()
    # Don't forget to subtract 1 since the first day of year is day 1, not day 0
    return secs_yyyy + ((ztim[1]-1) * 86400.0 + ztim[2] / 10000.0)


def ztim_to_j2000(ztim):
    """
    Given a ztim tuple of (year, day of year, 1/10000 seconds of day),
    return J2000 time (seconds since 12:00 January 1, 2000)
    """
    global epoch_J2000
    return ztim_to_seconds(ztim, epoch_J2000)

def ztim_to_posix(ztim):
    # Alias for function in deva
    global epoch_unix
    return ztim_to_seconds(ztim, epoch_unix)

def test_read_ztim_bin():
    files = glob.glob('/disk/kea/WAIS/targ/tpro/MBL/*/*/*/ztim_llz_bedelv.bin')
    logging.info("Found {:d} ztim files".format(len(files)) )
    for filename in files:
        test_read_one_zfile(filename)

def test_read_one_zfile(filename):
    try:
        logging.debug("Reading "+filename)
#        field_names=['xx','yy','aircraft_elevation','roll_angle','pitch_angle',
#                     'heading_angle','position_error','EW_acceleration',
#                     'NS_acceleration','z_acceleration']
        data=read_ztim(filename)
        print(data[3])
#        return data
    except Exception as e:
        logging.error('Error processing file {:s}'.format(filename))
        for line in traceback.format_exc().split("\n"):
            logging.error(line)


def test_zfile():
    logging.debug("called zfile w/: %r" % (sys.argv))
    field_names=['xx','yy','aircraft_elevation','roll_angle','pitch_angle','heading_angle','position_error','EW_acceleration','NS_acceleration','vertical_acceleration']

    data = read_ztim(sys.argv[1], field_names)
    logging.debug(data)
    data2 = read_ztim(sys.argv[1])
    logging.debug(data2)


def main():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
        format="zfile: [%(levelname)-7s] %(message)s")

#    # Test ascii generator
#    infile='/disk/kea/WAIS/targ/artl/onset-nature/norm/RTZ6/TF09/TT21a/LAS_SJBa/syn_ztim'
#    for ztim in read_ztim_text(infile):
#        j2000 = ztim_to_j2000(ztim)
#        unixt = ztim_to_seconds(ztim, epoch_unix)
#        #print(ztim,j2000, unixt)
#
#    # Test column generator
#    yyyy, doy, sod = read_ztim_text_ascolumns(infile)
#    test_read_ztim_bin()

    filename='/disk/kea/WAIS/targ/treg/NAQLK/JKB2j/ZY1b/TRJ_JKB0/ztim_llzrphsaaa.bin'
    read_ztim(filename)



if __name__=="__main__":
    import sys
    main()





