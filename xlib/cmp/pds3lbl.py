#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'June 26, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'July 12, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'Changed the binary file to be read'
                 'by numpy and only the bitstrings by the bitstring package.'
                 'This significantly speeds up the process'}}

import glob
import os
import pickle
import logging
import gzip
import sys
import json

import pandas as pd
import numpy as np
import bitstring as bs
import pvl

G_DEBUG = False

def read_science(data_path, label_path, science=True, bc=True):

    """
    Routine to read the pds3 label files and return the corresponding
    data structure. Structure is returned as dictionary

    Input:
    ----------------
    label_path: path to pds3 label file
    data_path: path to pds3 science or aux data file
    science (optional): Set to True when it is science data and False
                        if aux data
    bc: Should it read the bit columns as well. Default is True.

    Output:
    ----------------
    Dictionary containing the data from the science file specified in the
    label file.
    """

    logging.debug("pds3lbl::read_science reading data  " + data_path)
    logging.debug("pds3lbl::read_science reading label " + label_path)

    # Dictionary for PDS3 data types: Note that floats are currently limited
    # to 64 bit. Higher precisions will be returned as bitstrings. Dates are
    # translated to integer values :)

    PDS3_DATA_TYPE_TO_DTYPE = {
        'DATE':                 'S',
        'MSB_BIT_STRING':       'V',
        'BOOLEAN' :             '?',
        'IEEE_REAL':            '>f',
        'SUN_REAL':             '>f',
        'MAC_REAL':             '>f',
        'LSB_INTEGER':          '<i',
        'MAC_INTEGER':          '>i',
        'MSB_INTEGER':          '>i',
        'PC_INTEGER':           '<i',
        'SUN_INTEGER':          '>i',
        'VAX_INTEGER':          '<i',
        'LSB_UNSIGNED_INTEGER': '<u',
        'MAC_UNSIGNED_INTEGER': '>u',
        'MSB_UNSIGNED_INTEGER': '>u',
        'SUN_UNSIGNED_INTEGER': '>u',
        'PC_UNSIGNED_INTEGER':  '<u',
        'VAX_UNSIGNED_INTEGER': '<u',
    }

    PDS3_DATA_TYPE_TO_BS = {
        'DATE' :                'int',
        'BOOLEAN' :             'bool',
        'MSB_BITSTRING':        'bits',
        'IEEE_REAL':            'float',
        'MAC_REAL':             'float',
        'SUN_REAL':             'float',
        'LSB_INTEGER':          'int',
        'MSB_INTEGER':          'int',
        'PC_INTEGER':           'int',
        'SUN_INTEGER':          'int',
        'SUN_UNSIGNED_INTEGER': 'int',
        'VAX_INTEGER':          'int',
        'VAX_UNSIGNED_INTEGER': 'int',
        'LSB_UNSIGNED_INTEGER': 'uint',
        'MAC_INTEGER':          'uint',
        'MAC_UNSIGNED_INTEGER': 'uint',
        'MSB_UNSIGNED_INTEGER': 'uint',
        'PC_UNSIGNED_INTEGER':  'uint'
    }

    # List of Mode ID's as specified in the EDR interface document
    bits8 = ['SS1', 'SS01', 'SS4', 'SS04', 'SS7', 'SS07', 'SS10', 'SS13', 'SS16', 'SS19']
    bits6 = ['SS2', 'SS02', 'SS5', 'SS05', 'SS8', 'SS08', 'SS11', 'SS14', 'SS17', 'SS20']
    bits4 = ['SS3', 'SS03', 'SS6', 'SS06', 'SS9', 'SS09', 'SS12', 'SS15', 'SS18', 'SS21']

    # Decode label file structure
    label = pvl.load(label_path)
    dtype = []
    bitcolumns = []
    spare = 0
    # Decode data type of each column
    for column in label:
        name = column[1]['NAME']
        # Identify Spares
        if 'SPARE' in name:
            name = 'SPARE'+str(spare)
            dty = 'S'+str(column[1]['BYTES'])
            dtype.append((name, dty))
            spare += 1
        else:
            # Check if column consists of sub columns:
            # If yes read it as string and evaluate it later
            if 'MSB_BIT_STRING' in column[1]['DATA_TYPE']:
                bitcolumns.append([name, column[1], column[1]['BYTES']])
            # And add datatype to decoding list
            dty = PDS3_DATA_TYPE_TO_DTYPE[column[1]['DATA_TYPE']]+str(
                column[1]['BYTES'])
            # Numpy is not able to read 24 bit or 28 / 32 byte values
            # These are split into readable formats
            # TODO: Reconstruct them into one value before returning it
            #       to the user
            if 'u3' in dty:
                dtype.append((name+'_8b', 'u2'))
                dtype.append((name+'_4b', 'u1'))
            elif 'f28' in dty:
                dtype.append((name+'_16b', 'f16'))
                dtype.append((name+'_8b', 'f8'))
                dtype.append((name+'_4b', 'f4'))
            elif 'f32' in dty:
                dtype.append((name+'_16b1', 'f16'))
                dtype.append((name+'_16b2', 'f16'))
            elif "i23" in str(dty):
                dtype.append((name, 'S23'))
            else:
                dtype.append((name, dty))

    # Open label file corresponding to science file
    # Unfortunately labels are not labeled consistently
    # It first tries the regular file and then looks for other ones
    # in the respective data folder
    # GNG QUESTION: should this thing "break" on a successful load?
    try:
        if science:
            science_label = pvl.load(data_path.replace('_a_s.dat', '_a.lbl'))
        else:
            science_label = pvl.load(data_path.replace('_a_a.dat', '_a.lbl'))
    except Exception: # TODO: be more specific: FileNotFound exception?
        new_path, _ = data_path.rsplit('/', 1)
        os.chdir(new_path)
        for filename in glob.glob("*.LBL"):
            science_label = pvl.load(filename)
        for filename in glob.glob("*.lbl"):
            science_label = pvl.load(filename)

    # read number of range lines and science mode from label file
    rows = science_label['FILE']['SCIENCE_TELEMETRY_TABLE']['ROWS']

    if science:
        mode_id = science_label['FILE']['INSTRUMENT_MODE_ID']
        if mode_id in bits8: pseudo_samples = 3600
        elif mode_id in bits6: pseudo_samples = 2700
        elif mode_id in bits4: pseudo_samples = 1800
        else:
            # GNG: TODO: raise an exception?
            print('Error while reading science label! Invalid mode id', mode_id)
            return 0

        # If it is science data, this has to be added here since it is not
        # contained in the science ancilliary label file
        if pseudo_samples == 3600:
            for i in range(0, 3600):
                dtype.append(('sample'+str(i), 'b'))
        else:
            dtype.append(('samples', str(pseudo_samples)+'B'))


    # Read science data file
    fil = glob.glob(data_path)[0]
    dtype_r = np.dtype(dtype)
    arr = np.fromfile(fil, dtype=dtype_r)
    if science:
        if pseudo_samples < 3600:
            dtype_pd = np.dtype([('samples', 'V'+str(pseudo_samples))
                if x == ('samples', str(pseudo_samples) + 'B') else x for x in dtype])
            arr_pd = np.fromfile(fil, dtype=dtype_pd)
            dfr = pd.DataFrame(arr_pd)
        else:
            dfr = pd.DataFrame(arr)
    else:
        dfr = pd.DataFrame(arr)
    out = np.reshape(arr, [1, rows])[0]

    del arr
    if science:
        if pseudo_samples < 3600: del arr_pd

    # Convert 6 and 4 bit samples
    if science and pseudo_samples < 3600:
        s = out['samples']
        conv = np.empty((len(s), 3600), dtype='i1')
        if pseudo_samples == 2700:
            for j in range(len(s)):
                conv[j] = [x for y in [
                    [s[j][i]>>2,
                   ((s[j][i] << 4) & 0x3f) | s[j][i+1] >> 4,
                   ((s[j][i+1] << 2) & 0x3f) | s[j][i+2] >> 6,
                     s[j][i+2] & 0x3f] for i in range(0, 2700, 3)] for x in y]
            for i in range(0, 3600):
                dfr['sample' + str(i)] = pd.Series(conv[:, i], index=dfr.index)
        elif pseudo_samples == 1800:
            for j in range(len(s)):
                conv[j] = [x for y in [[s[j][i] >> 4, s[j][i] & 0xf]
                          for i in range(1800)] for x in y]
            for i in range(0, 3600):
                dfr['sample'+str(i)] = pd.Series(conv, index=dfr.index)
        else:
            # GNG: TODO: raise an exception?
            print('This error should not occur. Something horribly went wrong')
            return 0
    if bc:
        # Replace the bitstrings
        # Read bitcolumns. These have been previously saved in np.void format
        # and are now converted into bitstrings which are evaluated bit per bit.

        for bcl in bitcolumns:
            # A list of bitarray objects for data
            bitdata = [bs.ConstBitStream(bs1.tobytes()) for bs1 in out[bcl[0]]]

            for sub in bcl[1]:
                # GNG: TODO: should this be?
                # if sub[0] == 'BIT_COLUMN':
                if 'BIT_COLUMN' not in sub:
                    continue

                name = sub[1]['NAME']
                # Select data type from dictionary if field is not a spare
                if 'SPARE' in name:
                    continue

                nb_bits   = sub[1]['BITS']
                start_bit = sub[1]['START_BIT'] - 1
                dtype = PDS3_DATA_TYPE_TO_BS[sub[1]['BIT_DATA_TYPE']]\
                        +':'+str(nb_bits)
                if 'BOOLEAN' in sub[1]['BIT_DATA_TYPE']:
                    dtype = 'bool'
                if G_DEBUG:
                    logging.debug("start_bit={:d} nb_bits={:d} dtype=s"\
                                  .format(start_bit, nb_bits, dtype))

                conv = np.array([bit_select2(bits, start_bit, dtype) for bits in bitdata])
                dfr[name] = pd.Series(conv, index=dfr.index)
    return dfr

def tobit(string):
    """
    This subroutine converts bits from np.void into a bitstream

    Input:
    -----------
        string: np.void string

    Output:
    -----------
        bitStream
    """
    return bs.BitArray(string.tobytes()).bin

def bit_select(bits, start, num, form):
    """
    This subroutine selects bits from a bitstream and converts
    them into a new format [GNG: what new format?]

    Input:
    -----------
        bits: bitstream
        start: start_value to read in stream
        num: number of bits to read
        form: return formats
    Output:
    -----------
        list with data records
    """
    # GNG: does this modify the input parameter?
    bits = bits.decode('utf-8')
    return bs.ConstBitStream('0b'+str(bits[start:(start+num)]
                                     )).readlist(form)[0]

def bit_select2(bits, pos, form):
    """
    This subroutine selects bits from a BitStream object and converts them
    to a numeric with the requested format.  The length of the bit field
    is encoded in the "form" field

    Input:
    -----------
        bits: bistring Bits object representing a bit array
        pos: start index (lowest numbered index) of the field
        form: return format
    Output:
    -----------
        numeric data
    """
    bits.pos = pos
    return bits.read(form)

def read_raw(path):
    """
    This routine reads binary files for pulse compressed SHARAD data as
    provided by Bruce Campbell.

    Input:
    -----------
        path: Path to the according file.
    Output:
    -----------
        numpy array with data records
    """
    fil = glob.glob(path)[0]
    file_size = os.stat(fil).st_size
    rows = 7200
    columns = int(file_size*8/32/rows)
    arr = np.fromfile(fil, dtype='<f4')
    out = np.reshape(arr, [columns, rows])
    return out

def read_refchirp(path):
    """
    This routine reads binary files with reference chirps for SHARAD as
    given on PDS by the italian team. Returns a numpy array with 2048
    complex value samples.

    Input:
    -----------
        path: Path to the according file.
    Output:
        numpy array with 2048 samples. Complex values.
    """
    fil = glob.glob(path)[0]
    arr = np.fromfile(fil, dtype=np.float)
    return arr

def test_write(datafile, labelfile, goldfile):
    """ Test writing pdsdata """
    pdsdata = read_science(datafile, labelfile)
    with gzip.open(goldfile, "wb") as  fout:
        pickle.dump(pdsdata, fout)

def test_cmp_gold(datafile, labelfile, goldfile):
    """ compare one file to another """
    pdsdata1 = read_science(datafile, labelfile)
    with gzip.open(goldfile, "rb") as  fin:
        pdsdata2 = json.load(fin)

    assert cmp(pdsdata1, pdsdata2)


def test1(outputdir='.'):
    """ Test basic PDS file read functionality """
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Test pds3lbl on a known piece of data to assure correct output
    datafile = "/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/data/rm184/edr3434001/e_3434001_001_ss19_700_a_s.dat"
    labelfile = "/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt"
    #outfile="rm184_edr3434001.gold.pickle.gz"
    #pdsdata1 = read_science(datafile, labelfile)
    outfile = "rm184_edr3434001.pickle.gz"
    test_write(datafile, labelfile, outfile)
    #test_cmp_gold(datafile, labelfile, goldfile)
    # It would be nice to have something to compare with but oh well



def main():
    """ main function """
    global G_DEBUG
    G_DEBUG = True

    parser = argparse.ArgumentParser(description='Planetary Data System 3 Labels')
    parser.add_argument('-o', '--output', default='.', help="Output directory")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")

    args = parser.parse_args()


    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pds3lbl: [%(levelname)-7s] %(message)s")

    test1(args.output)

if __name__ == "__main__":
    # execute only if run as a script
    import argparse
    main()
