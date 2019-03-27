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

sys.path.append('../xlib')
#import misc.prog as prog
import misc.hdf as hdf

import altimetry.beta5 as b5



def alt_processor(path, idx_start=None, idx_end=None, b_save=False):

    try:
        # create cmp path
        path_root_alt = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
        path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = path_file.split('/')[-1]
        obn = data_file[2:9]
        path_file = path_file.replace(data_file,'')
        cmp_path = path_root_cmp+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        science_path=path.replace('_a.dat','_s.dat')
        if os.path.exists(cmp_path): 
            result = b5.beta5_altimetry(cmp_path, science_path, label_path, aux_path,
                                     idx_start=0, idx_end=None, use_spice=False, ft_avg=10,
                                     max_slope=25, noise_scale=20, fix_pri=1, fine=False)

            #new_path = path_root_alt+path_file+'beta5/'
            if b_save:
                h5 = hdf.hdf('north_pole_beta5.h5', mode='a')
                orbit_data = {obn: result}
                h5.save_dict('sharad', orbit_data)
                h5.close()  
        else:
            print('warning',cmp_path,'does not exist')
            return 0
    except Exception as e:
        print(e)

# Set number of cores
nb_cores = 8
kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
spice.furnsh(kernel_path)

# Read lookup table associating gob's with tracks
#h5file = pd.HDFStore('mc11e_spice.h5')
#keys = h5file.keys()
#lookup = np.genfromtxt('lookup.txt',dtype='str')
#lookup = np.genfromtxt('EDR_Cyril_SouthPole_Path.txt', dtype = 'str')
#lookup = np.genfromtxt('EDR_Cyril_Path.txt', dtype = 'str')
lookup = np.genfromtxt('EDR_NorthPole_Path.txt', dtype = 'str')

# Build list of processes
print('build task list')
process_list=[]
#p=prog.Prog(len(keys))
#p = prog.Prog(len(lookup))
i=0
path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'

#for orbit in keys:
for path in lookup:
    #p.print_Prog(i)
    #gob = int(orbit.replace('/orbit', ''))
    #path = lookup[gob]
    #path ='/disk/daedalus/sds/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr10xxx/edr1058901/e_1058901_001_ss19_700_a_a.dat'
    #idx_start = h5file[orbit]['idx_start'][0]
    #idx_end = h5file[orbit]['idx_end'][0]
    path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
    data_file = path_file.split('/')[-1]

    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
    new_path = path_root+path_file+'beta5/'
    if not os.path.exists(new_path+data_file.replace('.dat','.npy')):
        process_list.append([path,None,None])
        i+=1
#p.close_Prog()
#h5file.close()

print('start processing',len(process_list),'tracks')


if nb_cores <= 1:
    for t in process_list:
        result = alt_processor(*t)
else:
    pool = multiprocessing.Pool(nb_cores)
    results = [pool.apply_async(alt_processor, t) for t in process_list]
    for i,result in enumerate(results):
        flag = result.get()
        print("Finished task {:d} of {:d}".format(i+1, len(process_list)))
    print('done')
