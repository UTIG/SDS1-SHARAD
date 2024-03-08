#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.2'
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
                 'This significantly speeds up the process'},
    '1.2':
        {'date': 'March 8, 2024',
         'author': 'Gregory Ng, UTIG',
         'info':  'Switch from bitstring to bitstruct package'},
}

import glob
import os
import logging
import sys
from warnings import simplefilter

import pandas as pd
import numpy as np
import bitstruct
import pvl

# Ignore PerformanceWarning in pandas 2.0.0 and higher
# https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

    logging.debug("pds3lbl::read_science science=%r bc=%r", science, bc)
    logging.debug("pds3lbl::read_science reading data  %s", data_path)
    logging.debug("pds3lbl::read_science reading label %s", label_path)


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
    if science:
        science_label_filename = data_path.replace('_a_s.dat', '_a.lbl')
    else:
        science_label_filename = data_path.replace('_a_a.dat', '_a.lbl')
    # Try alternate labelnames
    if not os.path.exists(science_label_filename):
        dir1 = os.path.dirname(data_path)
        pat1 = os.path.join(dir1, "*.LBL")
        pat2 = os.path.join(dir1, "*.lbl")
        labellist = glob.glob(pat1) + glob.glob(pat2)
        if not labellist:
            raise FileNotFoundError("Can't find science label for %s" % data_path)
        science_label_filename = labellist[0]

    logging.debug("Read science label from %s", science_label_filename)
    science_label = pvl.load(science_label_filename)


    # read number of range lines and science mode from label file
    rows = science_label['FILE']['SCIENCE_TELEMETRY_TABLE']['ROWS']

    if science:
        mode_id = science_label['FILE']['INSTRUMENT_MODE_ID']
        if mode_id in bits8:
            pseudo_samples = 3600
        elif mode_id in bits6:
            pseudo_samples = 2700
        elif mode_id in bits4:
            pseudo_samples = 1800
        else: # pragma: no cover
            raise ValueError('Error while reading science label! Invalid mode id %d' % mode_id)

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

    # Convert 6 and 4 bit samples
    if science and pseudo_samples < 3600:
        del arr_pd
        s = out['samples']
        conv = np.empty((len(s), 3600), dtype='i1')
        if pseudo_samples == 2700: # 6-bit
            conv[:, 0:3600:4] = (                                (s[:, 0:2700:3] >> 2))
            conv[:, 1:3600:4] = ((s[:, 0:2700:3] << 4) & 0x3f) | (s[:, 1:2700:3] >> 4)
            conv[:, 2:3600:4] = ((s[:, 1:2700:3] << 2) & 0x3f) | (s[:, 2:2700:3] >> 6)
            conv[:, 3:3600:4] = ((s[:, 2:2700:3] & 0x3f))

            for i in range(3600):
                dfr['sample' + str(i)] = pd.Series(conv[:, i], index=dfr.index)
        elif pseudo_samples == 1800: # 4-bit
            conv[:, 0:3600:2] = s[:, 0:1800] >> 4
            conv[:, 1:3600:2] = s[:, 0:1800] & 0xff
            for i in range(3600):
                dfr['sample'+str(i)] = pd.Series(conv[:, i], index=dfr.index)
        else: # pragma: no cover
            raise ValueError("Unexpected value for pseudo_samples = %d" % pseudo_samples)

    if bc:
        # Replace the packed bitfields with individual arrays
        # Read bitcolumns. These have been previously saved in np.void format
        #for bcl in bitcolumns:
        for name, pvlobj, _ in bitcolumns:
            # A list of bitarray objects for data
            bitstruct_parser, npdtype = build_bitstruct(pvlobj)
            # Build a numpy structured array
            barr = np.empty((len(out[name]),), dtype=npdtype)
            for ii, bs1 in enumerate(out[name]):
                barr[ii] = bitstruct_parser.unpack(bs1.tobytes())

            for name1, _ in npdtype:
                dfr[name1] = pd.Series(barr[name1], index=dfr.index)

    return dfr


def build_bitstruct(pvlobj: pvl.PVLObject):
    """ From the pvl specification, build a bitstruct
    format string to be used with the bitstruct.unpack function """

    fmttokens, npdtype = [], []
    end_bit_prev = 0
    for sub in pvlobj:
        if 'BIT_COLUMN' not in sub:
            continue

        name = sub[1]['NAME']
        nb_bits   = sub[1]['BITS']
        start_bit = sub[1]['START_BIT'] - 1
        assert end_bit_prev == start_bit, "Unaccounted bits: %d != %d" % (end_bit_prev, start_bit)
        end_bit_prev = start_bit + nb_bits

        #dtype = PDS3_DATA_TYPE_TO_BS[sub[1]['BIT_DATA_TYPE']] + str(nb_bits)
        if name.startswith('SPARE'):
            dtype_bitstruct = 'p' # skip as padding
        else:
            dtype_bitstruct = PDS3_DATA_TYPE_TO_BS[sub[1]['BIT_DATA_TYPE']][0]
            npdtype.append((name, np.int64))

        dtype_bitstruct += str(nb_bits) # append length
        fmttokens.append(dtype_bitstruct)

    cf = bitstruct.compile(''.join(fmttokens))

    return cf, npdtype


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
    return np.reshape(arr, [columns, rows])

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
    return np.fromfile(fil, dtype=float)

