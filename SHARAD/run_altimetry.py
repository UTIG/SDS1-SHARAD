#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 6 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}

import sys
import os
import importlib.util
import multiprocessing
import time
import logging

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import pandas as pd
import traceback

sys.path.append('../xlib')
#import misc.prog as prog
import misc.hdf as hdf

import altimetry.beta5 as b5
import gzip


def main():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
        format="run_altimetry: [%(levelname)-7s] %(message)s")


    # Set number of cores
    nb_cores = 1
    kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
    spice.furnsh(kernel_path)

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    #tracklist = 'lookup.txt',dtype='str')
    #tracklist = 'EDR_Cyril_SouthPole_Path.txt', dtype = 'str')
    #tracklist = 'EDR_Cyril_Path.txt', dtype = 'str')
    tracklist = 'EDR_NorthPole_Path.txt'

    # Build list of processes
    print('build task list')
    process_list=[]
    #p=prog.Prog(len(keys))
    #p = prog.Prog(len(lookup))
    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
    path_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    #path_out = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
    path_out = './alt/'


    #for orbit in keys:
    with  open(tracklist, 'r') as flist:
        for i,path in enumerate(flist):
            path = path.rstrip()
            #p.print_Prog(i)
            #gob = int(orbit.replace('/orbit', ''))
            #path = lookup[gob]
            #path ='/disk/daedalus/sds/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr10xxx/edr1058901/e_1058901_001_ss19_700_a_a.dat'
            #idx_start = h5file[orbit]['idx_start'][0]
            #idx_end = h5file[orbit]['idx_end'][0]
            relpath = os.path.dirname(os.path.relpath(path, path_edr))
            path_file = os.path.relpath(path, path_edr)
            data_file = os.path.basename(path)
            #outfile = os.path.join(path_out, path_file, 'beta5', data_file.replace('.dat', '.npy'))
            outfile = os.path.join(path_out, relpath, 'beta5', data_file.replace('.dat', '.csv'))

            if not os.path.exists(outfile):
                process_list.append({
                    'inpath' : path, 
                    'outputfile' : outfile, 
                    'idx_start' : None,
                    'idx_end' : None,
                    'save_format' : 'hdf5'})

    #p.close_Prog()

    # Limit to first 20 tracks
    process_list = process_list[0:20]

    print('start processing',len(process_list),'tracks')
    #sys.exit(1)

    if nb_cores <= 1:
        for i,t in enumerate(process_list):
            result = alt_processor(**t)
            print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    else:
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(alt_processor, None, t) for t in process_list]
        for i,result in enumerate(results):
            flag = result.get()
            print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
        print('done')



def alt_processor(inpath, outputfile, idx_start=None, idx_end=None, save_format=''):
    try:
        # create cmp path
        path_root_alt = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
        path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_root_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
        # Relative path to this file
        fname = os.path.basename(inpath)
        obn = fname[2:9] # orbit name
        # Relative directory of this file
        reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
        logging.debug("inpath: " + inpath)
        logging.debug("reldir: " + reldir)
        logging.debug("path_root_edr: " + path_root_edr)
        cmp_path = os.path.join(path_root_cmp, reldir, 'ion', fname.replace('_a.dat','_s.h5') )
        #cmp_path = path_root_cmp+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        science_path = inpath.replace('_a.dat','_s.dat')

        if not os.path.exists(cmp_path): 
            logging.warning(cmp_path + " does not exist")
            return 0

        logging.info("Reading " + cmp_path)
        df = b5.beta5_altimetry(cmp_path, science_path, label_path, aux_path,
                                idx_start=0, idx_end=None, use_spice=False, ft_avg=10,
                                max_slope=25, noise_scale=20, fix_pri=1, fine=False)

        #new_path = path_root_alt+path_file+'beta5/'
        if save_format == 'hdf5':
            outfile='north_pole_beta5.h5'
            logging.info("Writing to " + outfile)
            h5 = hdf.hdf(outfile, mode='a')
            orbit_data = {obn: df}
            h5.save_dict('sharad', orbit_data)
            h5.close()
        elif save_format == 'csv':
            #fname1 = fname.replace('_a.dat', '.csv.gz')
            #outfile = os.path.join(path_root_alt, reldir, 'beta5',fname1)
            logging.info("Writing to " + outputfile)
            if not os.path.exists( os.path.dirname(outputfile) ):
                os.makedirs( os.path.dirname(outputfile) )

            df.to_csv(outputfile)
        elif save_format == '':
            pass
        else:
            logging.warning("Unrecognized output format '{:s}'".format(save_format))

    except Exception as e:
        taskname="error"
        for line in traceback.format_exc().split("\n"):
            print('{:s}: {:s}'.format(taskname, line) )
        return 1
    

if __name__ == "__main__":
    # execute only if run as a script
    main()

