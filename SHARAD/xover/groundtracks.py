#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'July 30, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'March 26, 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'Reorganized code.'}
}

"""
This is a short script to extract the ground tracks in lat/lon from
each record within the SHARAD data set.

The area is limited to the HRSC MC11E DTM.
"""

import os
#import sys
#import glob

import numpy as np
import pandas as pd
#import pvl

import get_et

def main():

    """ get a list of all files in sharad data folder
    identify data records and corresponding label files """
    # TODO: GNG this root path seems wrong now (path not found)
    raw = '/disk/kea/SDS/orig/supl/SHARAD/raw/'
    records = []
    lbls = []
    for path, _, files in os.walk(raw):
        for name in files:
            filename = os.path.join(path, name)
            if filename.endswith('_a_a.dat'):
                records.append(filename)
                lbls.append(filename.replace('_a_a.dat', '_a.lbl'))

    # path to science auxillary label file
    lbl_file = os.path.join(raw, 'mrosh_0001/label/auxiliary.fmt')

    #p=prog.Prog(int(len(records)),step=0.1)
    print("Found {:d} record files in {:s}".format(len(records), raw))

    save_path = 'mc11e_full.h5'
    data_columns = ['idx_start', 'idx_end']
    for i, record in enumerate(records):
        #p.print_Prog(int(i))
        rec = get_et.read_science_np(lbl_file, record)[0]
        et = np.empty(len(rec), dtype=np.double)
        lat = np.empty(len(rec), dtype=np.double)
        lon = np.empty(len(rec), dtype=np.double)
        onb = np.empty(len(rec), dtype=np.int)
        # NOTE: could use list comprehensions here
        # et = np.array([ row[2] for row in rec ])
        for j, col in enumerate(rec):
            et[j] = col[2]
            onb[j] = col[5]
            lat[j] = col[11]
            lon[j] = col[10]
        indx = np.where((lat > 0) & (lat < 30) &
                        (lon > 337.5) & (lon < 359.9))[0]
        if len(indx) > 2:
            print("record[{:3d}]: Saving to {:s}".format(i, save_path))
            df = pd.DataFrame({'idx_start': indx[0], 'idx_end': indx[-1]})
            df.to_hdf(save_path, records[i], format='table',
                      data_columns=data_columns)
    print('tracks done')

if __name__ == "__main__":
    # execute only if run as a script
    main()
