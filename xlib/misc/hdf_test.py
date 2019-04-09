#!/usr/bin/env python3


import argparse
import logging
import sys
import os



import pandas as pd
import numpy as np

import hdf


def find_h5files():
    #/disk/kea/SDS/targ/xtra/SHARAD/cmp/
    pass

def test1(inputfile=None):
    # Show it being used in a typical context
    #h5 = hdf.hdf()
    pass

def test_write(outputdir="."):
    outputfile = os.path.join(outputdir, "test_write1.h5")
    test_data = {'dict1': make_pandas_data()}

    # demonstrate in a standard context open/close syntax
    h5 = hdf.hdf(outputfile, mode='w')
    h5.save_dict('dict1', test_data)
    h5.close()

    outputfile = os.path.join(outputdir, "test_write2.h5")
    # Now demonstrate using it in a 'with' context
    with hdf.hdf(outputfile, mode='w') as h5:
        h5.save_dict('dict1', test_data)


def make_pandas_data():
    # GNG TODO: Does pandas want this to be a np array or could it be a list of lists?
    # Seems like it just wants to be a python list.
    # https://www.geeksforgeeks.org/python-pandas-dataframe/#Basics
    spots = []
    columns = ['et', 'spot_lat', 'spot_lon', 'spot_radius', 'idx_coarse', 'idx_fine', 'range']
    for i in range(1000):
        row = (i*0.1, -115, 60, 30, i // 100, i % 100, i*20)
        spots.append(row)
    df = pd.DataFrame(spots, columns=columns)
    return df



def main():
    # Demonstrate opening of an HDF5 file
    # TODO: improve description
    parser = argparse.ArgumentParser(description='Test hdf5 class')
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")

    args = parser.parse_args()


    loglevel=logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="hdf: [%(levelname)-7s] %(message)s")

    logging.info("Starting HDF testing")
    test1()
    test_write()


if __name__ == "__main__":
    import argparse
    # execute only if run as a script
    main()


