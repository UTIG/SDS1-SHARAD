#!/usr/bin/env python3

"""
test hdf.py with some examples

"""

import argparse
import logging
import sys
import os
import glob

import pandas as pd

import hdf



def test_read(inputpath='/disk/kea/SDS/targ/xtra/SHARAD/cmp', maxfiles=100):
    logging.info("test_read()")
    pat = os.path.join(inputpath, '*/*/*/*/*/*.h5')
    h5files = glob.glob(pat)
    logging.info("Found %d hdf5 files", len(h5files))
    if len(h5files) > maxfiles:
        h5files = h5files[0:maxfiles]

    for filename in h5files:
        with hdf.hdf(filename, mode='r') as hdf5:
            pass


def test_write(outputdir="."):
    """ Test writing an HDF5 file from some pandas data """
    logging.info("test_write()")
    outputfile = os.path.join(outputdir, "test_write1.h5")
    test_data = {'dict1': make_pandas_data()}

    os.makedirs(outputdir, exist_ok=True)

    # demonstrate in a standard context open/close syntax
    hdf5 = hdf.hdf(outputfile, mode='w')
    hdf5.save_dict('dict1', test_data)
    hdf5.close()

    outputfile = os.path.join(outputdir, "test_write2.h5")
    # Now demonstrate using it in a 'with' context
    with hdf.hdf(outputfile, mode='w') as hdf5:
        hdf5.save_dict('dict1', test_data)


def make_pandas_data():
    """ Generate some arbitrary pandas data """
    # GNG TODO: Does pandas want this to be a np array or could it be a
    # list of lists?
    # Seems like it just wants to be a python list.
    # https://www.geeksforgeeks.org/python-pandas-dataframe/#Basics
    spots = []
    columns = ['et', 'spot_lat', 'spot_lon', 'spot_radius',
               'idx_coarse', 'idx_fine', 'range']
    for i in range(1000):
        row = (i*0.1, -115, 60, 30, i // 100, i % 100, i*20)
        spots.append(row)
    return pd.DataFrame(spots, columns=columns)



def main():
    """ Demonstrate opening of an HDF5 file """
    parser = argparse.ArgumentParser(description='Test hdf5 class')
    parser.add_argument('-o', '--output', default='.', help="Output directory")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")

    args = parser.parse_args()


    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="hdf: [%(levelname)-7s] %(message)s")

    logging.info("Starting HDF testing")
    test_read()
    test_write(args.output)


if __name__ == "__main__":
    # execute only if run as a script
    main()


