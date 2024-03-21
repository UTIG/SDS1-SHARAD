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
import mmap
import traceback

import numpy as np
import bitstruct
import pvl


# Dictionary for PDS3 data types: Note that floats are currently limited
# to 64 bit. Higher precisions will be returned as bitstrings. Dates are
# translated to integer values :)

PDS3_DATA_TYPE_TO_DTYPE = {
    'DATE':                 'S',
    'MSB_BIT_STRING':       'V',
    'BOOLEAN' :             'i', #'?',
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


def read_science(data_path, label_path, science=None):

    """
    Routine to read the pds3 label files and return the corresponding
    data structure. Structure is returned as a pandas DataFrame

    Input:
    ----------------
    data_path: path to pds3 science or aux data file
    label_path: path to pds3 label file
    science (optional): Set to True when it is science data and False
                        if aux data

    Output:
    ----------------
    Numpy structured table of data in science telemetry table
    """
    if science is not None:
        logging.warning("science boolean argument deprecated")
        traceback.print_stack(file=sys.stdout)

    return SHARADDataReader(label_path, data_path=data_path).arr



def build_bitstruct(pvlobj: pvl.PVLObject):
    """ From the pvl specification, gather all the packed bit fields
    and build structures necessary to parse them using bitstruct and
    put them all into one numpy structured array.  build a bitstruct
    format string to be used with the bitstruct.unpack function """

    fmttokens, npdtype = [], []
    end_bit_prev = 0
    for sub in pvlobj:
        if 'BIT_COLUMN' not in sub:
            continue

        name = sub[1]['NAME']
        nb_bits   = sub[1]['BITS']
        start_bit = sub[1]['START_BIT'] - 1
        bdtype = sub[1]['BIT_DATA_TYPE']
        assert end_bit_prev == start_bit, "Unaccounted bits: %d != %d" % (end_bit_prev, start_bit)
        end_bit_prev = start_bit + nb_bits

        #dtype = PDS3_DATA_TYPE_TO_BS[sub[1]['BIT_DATA_TYPE']] + str(nb_bits)
        if name.startswith('SPARE'):
            dtype_bitstruct = 'p' # skip as padding
        else:
            dtype_bitstruct = PDS3_DATA_TYPE_TO_BS[bdtype][0]
            nbytes = npdtype_length(nb_bits)
            npdtype.append((name, PDS3_DATA_TYPE_TO_DTYPE[bdtype] + str(nbytes)))

        dtype_bitstruct += str(nb_bits) # append length
        fmttokens.append(dtype_bitstruct)

    cf = bitstruct.compile(''.join(fmttokens))

    return cf, npdtype

def npdtype_length(nbits: int):
    """ Get a valid number of bytes for numpy dtypes from the number of bits """
    for dtypebits in (8, 16, 32, 64):
        if nbits <= dtypebits:
            return dtypebits // 8
    raise ValueError("Unexpected number of bits %d" % nbits)


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


class SHARADDataReader:
    """ read the label and cache it so you don't have to read it again
    TODO: separate the science into two different classes
    """
    # List of Mode ID's as specified in the EDR interface document
    BITS8 = ('SS1', 'SS01', 'SS4', 'SS04', 'SS7', 'SS07', 'SS10', 'SS13', 'SS16', 'SS19')
    BITS6 = ('SS2', 'SS02', 'SS5', 'SS05', 'SS8', 'SS08', 'SS11', 'SS14', 'SS17', 'SS20')
    BITS4 = ('SS3', 'SS03', 'SS6', 'SS06', 'SS9', 'SS09', 'SS12', 'SS15', 'SS18', 'SS21')


    def __init__(self, label_path: str, data_path=None):
        self.read_global_label(label_path)
        if data_path is not None:
            self.read_data(data_path)

    def read_global_label(self, label_path: str):
        """
        Routine to read the global SHARAD pds3 label files and cache its structure

        Input:
        ----------------
        label_path: path to pds3 label file
        data_path: path to pds3 science or aux data file

        """

        logging.debug("read_global_label %s", label_path)
        # Decode label file structure
        label = pvl.load(label_path)

        dtype = []
        bitcolumns = []
        spare = 0
        # Decode data type of each column
        for column in label:
            name = column[1]['NAME']
            # Identify Spares
            if name.startswith('SPARE'):
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

        self.dtype_global = dtype
        self.bitparsers = []

        self.all_bit_dtypes = []
        for name, pvlobj, _ in bitcolumns:
            assert name != 'sample_bytes', "Wasn't expecting this to be a bitcolumn"
            # A list of bitarray objects for data
            bitstruct_parser, npdtype = build_bitstruct(pvlobj)
            self.bitparsers.append((name, bitstruct_parser, npdtype))
            self.all_bit_dtypes.extend(npdtype)
        self.all_bit_dtypes = np.dtype(self.all_bit_dtypes)

    @staticmethod
    def label_path_from_data(data_path: str):
        """ Calculate the path and search for label if given a path to
        a data file
        return the path and whether this is a science telemetry file or not
        """
        # Open label file corresponding to science file
        # Unfortunately labels are not labeled consistently
        # It first tries the regular file and then looks for other ones
        # in the respective data folder
        # GNG QUESTION: should this thing "break" on a successful load?
        if data_path.endswith('_a_s.dat'):
            science = True
            science_label_filename = data_path.replace('_a_s.dat', '_a.lbl')
        elif data_path.endswith('_a_a.dat'):
            science = False
            science_label_filename = data_path.replace('_a_a.dat', '_a.lbl')
        else:
            raise ValueError("Unrecognized path '%s'" % data_path)

        # Try alternate labelnames
        if not os.path.exists(science_label_filename):
            dir1 = os.path.dirname(data_path)
            pat1 = os.path.join(dir1, "*.LBL")
            pat2 = os.path.join(dir1, "*.lbl")
            labellist = glob.glob(pat1) + glob.glob(pat2)
            if not labellist:
                raise FileNotFoundError("Can't find science label for %s" % data_path)
            science_label_filename = labellist[0]
        return science_label_filename, science

    def read_product_label(self, data_path: str):
        """ Construct the dtype for the product starting with the
        global label """
        # copy of the member dtype
        dtype = list(self.dtype_global)
        label_filename, science = SHARADDataReader.label_path_from_data(data_path)
        self.science = science
        logging.debug("Read product label from %s", label_filename)
        science_label = pvl.load(label_filename)


        # read number of range lines and science mode from label file
        self.rows = science_label['FILE']['SCIENCE_TELEMETRY_TABLE']['ROWS']

        if science:
            mode_id = science_label['FILE']['INSTRUMENT_MODE_ID']
            if mode_id in self.BITS8:
                self.pseudo_samples = 3600
            elif mode_id in self.BITS6:
                self.pseudo_samples = 2700
            elif mode_id in self.BITS4:
                self.pseudo_samples = 1800
            else: # pragma: no cover
                raise ValueError('Error while reading science label! Invalid mode id %d' % mode_id)

            # If it is science data, this has to be added here since it is not
            # contained in the science ancilliary label file
            dtype.append(('sample_bytes', str(pseudo_samples)+'b')) # signed bytes

        else:
            self.pseudo_samples = None
        self.dtype_local = dtype
        return dtype

    def read_data(self, data_path: str):
        """
        Routine to read the data files and return the corresponding
        data structure. Structure is returned as dictionary
        and saved into the member attribute self.arr

        Input:
        ----------------
        data_path: path to pds3 science or aux data file

        Output:
        ----------------
        Numpy structured array with raw data fields from science telemetry table
        """

        dtype = self.read_product_label(data_path)

        # Read science data file
        fil = glob.glob(data_path)[0]
        self.arr = np.memmap(fil, dtype=np.dtype(dtype), mode='r')
        logging.debug("arr.shape=%r len(dtype)=%r", self.arr.shape, len(dtype))
        assert self.rows == len(self.arr), "%s doesn't have expected length" % (fil,)
        return self.arr

    def get_radar_samples(self, arr_out=None, idxes=None):
        """ 
        Decode and return radar samples from the science telemetry table

        Parameters:
        arr_out: (optional)
        If provided, write radar data to this ndarray
        this should be a 2D array of size equal to the length of the slice requested
        and 3600 samples and accept signed integers.

        idxes: slice object for traces to extract
        default=None returns all traces

        Return value:
        -----------------------------
        radar samples array
        """

        assert self.science, "Not a science label"

        if arr_out is None:
            arr_out = np.empty((len(self.arr), 3600), dtype='i1')

        arr = self.arr
        assert len(arr_out.shape) == 2 and arr_out.shape[1] == 3600, "Output array not expected shape"
        #arr_out = np.empty((len(arr), 3600), dtype='i1')
        assert arr['sample_bytes'].dtype == np.dtype('b'), "Unexpected dtype: %r" % s.dtype

        if idxes is None:
            idxes = slice(None)

        s = arr['sample_bytes']
        if self.pseudo_samples == 3600: # 8-bit
            arr_out[...] = s[idxes]
        elif self.pseudo_samples == 2700: # 6-bit
            # bits 7:2 of s[0] become arr_out[0]
            arr_out[:, 0:3600:4] = s[idxes, 0:2700:3] >> 2
            # bits 1:0 of s[0] (hi2) and 7:4 of s[1] become arr_out[1]
            # mask then sign extend
            hi2 = ((s[idxes, 0:2700:3] & 0x3) << 6) >> 2
            lo4 = ( s[idxes, 1:2700:3] >> 4) & 0x0f
            arr_out[:, 1:3600:4] = hi2 | lo4
            del hi2, lo4
            # bits 3:0 of s[1] get sign extended and bits 7:6 of s[2]
            hi4 = ((s[idxes, 1:2700:3] & 0x0f) << 4) >> 2
            lo2 = (s[idxes, 2:2700:3] >> 6) & 0x03
            arr_out[:, 2:3600:4] = hi4 | lo2
            del hi4, lo2
            # Sign extend bits 5:0 of s[3]
            arr_out[:, 3:3600:4] = (s[idxes, 2:2700:3] << 2) >> 2
        elif self.pseudo_samples == 1800: # 4-bit
            # Sign extend everything
            arr_out[:, 0:3600:2] = s[idxes, 0:1800] >> 4
            arr_out[:, 1:3600:2] = (s[idxes, 0:1800] << 4) >> 4
        else: # pragma: no cover
            raise ValueError("Unexpected value for pseudo_samples = %d" % self.pseudo_samples)
        return arr_out


    def get_bitcolumns(self):
        """ Parse the packed bit columns into individually-accessible fields
        in a numpy structured array.
        These have been previously saved in np.void format
        """
        bitarr = np.empty((len(self.arr),),  dtype=self.all_bit_dtypes)
        for name, bitstruct_parser, npdtype in self.bitparsers:
            # Build a numpy structured array
            for ii, bs1 in enumerate(self.arr[name]):
                row = bitstruct_parser.unpack(bs1.tobytes())
                for (bname, _), v in zip(npdtype, row):
                    bitarr[bname][ii] = v

        return bitarr
