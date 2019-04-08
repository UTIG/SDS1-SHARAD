__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 6 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}


def alt_processor(path, idx_start=None, idx_end=None):

    import altimetry.beta5 as b5
    import numpy as np
    import os

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
                                        max_slope=25, noise_scale=20, fix_pri=1, fine=True)

            new_path = path_root_alt+path_file+'beta5/'
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            h5 = hdf.hdf(new_path + data_file.replace('.dat','.h5'), mode='w')
            orbit_data = {obn: result}
            h5.save_dict('beta5', orbit_data)
            h5.close()  

        else:
            print('warning',cmp_path,'does not exist')
            return 0
    except Exception as e:
        print(e)

import sys
import os
import numpy as np
import importlib.util
import pandas as pd
import multiprocessing
import time
import logging
import matplotlib.pyplot as plt
import spiceypy as spice
import misc.prog as prog
import misc.hdf as hdf

# Set number of cores
nb_cores = 8
kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
spice.furnsh(kernel_path)

# Read lookup table associating gob's with tracks
#h5file = pd.HDFStore('mc11e_spice.h5')
#keys = h5file.keys()
#lookup = np.genfromtxt('lookup.txt',dtype='str')
#lookup = np.genfromtxt('EDR_Cyril_SouthPole_Path.txt', dtype = 'str')
lookup = np.genfromtxt('EDR_Cyril_Path.txt', dtype = 'str')
#lookup = np.genfromtxt('EDR_NorthPole_Path.txt', dtype = 'str')

# Build list of processes
print('build task list')
process_list=[]
#p=prog.Prog(len(keys))
p = prog.Prog(len(lookup))
i=0
path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'

#for orbit in keys:
for path in lookup:
    p.print_Prog(i)
    #gob = int(orbit.replace('/orbit', ''))
    #path = lookup[gob]
    #path ='/disk/daedalus/sds/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr10xxx/edr1058901/e_1058901_001_ss19_700_a_a.dat'
    #path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr17xxx/edr1748102/e_1748102_001_ss19_700_a_a.dat'

    #idx_start = h5file[orbit]['idx_start'][0]
    #idx_end = h5file[orbit]['idx_end'][0]
    path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
    data_file = path_file.split('/')[-1]

    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
    new_path = path_root+path_file+'beta5/'
    if not os.path.exists(new_path+data_file.replace('.dat','.h5')):
        process_list.append([path,None,None])
        i+=1
    else:
        print('folder ' + new_path + ' already exists')
p.close_Prog()

print('start processing',len(process_list),'tracks')

pool = multiprocessing.Pool(nb_cores)
results = [pool.apply_async(alt_processor, t) for t in process_list]

p = prog.Prog(len(process_list))
i=0
for result in results:
    p.print_Prog(i)
    flag = result.get()
    i+=1
print('done')
