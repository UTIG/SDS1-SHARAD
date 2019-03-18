__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 13 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}


def sar_processor(path, idx_start=None, idx_end=None,
                  sar_window=200):

    import sar.sar as sar
    import numpy
    import matplotlib.pyplot as plt
    import cmp.pds3lbl as pds3
    from scipy.constants import c

    kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
    # create cmp path
    path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
    data_file = path_file.split('/')[-1]
    path_file = path_file.replace(data_file,'')
    cmp_path = path_root+path_file+'ion/'+data_file.replace('_a.dat','_s.npy')
    label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
    aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
    science_path=path.replace('_a.dat','_s.dat')

    cmp_track=np.load(cmp_path)
    if idx_start is not None:
        idx_start = max(0,idx_start)
    else:
        idx_start = 0
    if idx_end is not None:
        idx_end = min(len(cmp_track),idx_end)
    else:
        idx_end = len(cmp_track)

    cmp_track = cmp_track[idx_start:idx_end]
    data = pds3.read_science(science_path, label_path, science=True, bc=False)[idx_start:idx_end]
    aux = pds3.read_science(science_path.replace('_s.dat','_a.dat'), aux_path, science=False, bc=False)[idx_start:idx_end]

    pri_code = 1
    #pri_code = data['PULSE_REPETITION_INTERVAL'][0]                   
    if pri_code == 1:   pri = 1428E-6
    elif pri_code == 2: pri = 1429E-6
    elif pri_code == 3: pri = 1290E-6
    elif pri_code == 4: pri = 2856E-6
    elif pri_code == 5: pri = 2984E-6
    elif pri_code == 6: pri = 2580E-6
    else: pri = 0

    # S/C position
    et = aux['EPHEMERIS_TIME'].as_matrix()
    dpst = 460
    La = 8.774
    dBW = 0.4
    tlp = data['TLP_INTERPOLATE'].as_matrix() 
    scrad = data['RADIUS_INTERPOLATE'].as_matrix()
    tpgpy = data['TOPOGRAPHY'].as_matrix()
    tof = 2000*(scrad-min(scrad))/c
    rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].as_matrix()*0.0375E-6+pri-11.98E-6-tof
    vt = data['TANGENTIAL_VELOCITY_INTERPOLATE'].as_matrix()
    
    """
    sc = np.empty(len(ets))
    i = 0
    for et in ets:
        scpos, lt = spice.spkgeo(-74,et,'J2000',4)
        sc[i] = np.linalg.norm(scpos[0:3])
        i+=1
    """

    rl, pst_trc = sar.delay_doppler(cmp_track, dpst, La, dBW, tlp, et, scrad, tpgpy, rxwot, vt)
    np.save('sar_test.npy',rl)
    plt.style.use('dark_background')
    plt.imshow(rl.transpose(),cmap='binary_r',aspect='auto')
    plt.show()
    
    return 0

import sys
import os
import numpy as np
import importlib.util
import pandas as pd
import multiprocessing
import time
import logging
import misc.prog as prog
import misc.hdf as hdf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Set number of cores
nb_cores = 1

# Read lookup table associating gob's with tracks
h5file = pd.HDFStore('mc11e_spice.h5')
keys = h5file.keys()
lookup = np.genfromtxt('lookup.txt',dtype='str')
#lookup = np.genfromtxt('EDR_NorthPole_Path.txt', dtype = 'str')

# Build list of processes
print('build task list')
process_list=[]
p=prog.Prog(len(keys))
i=0
path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
for orbit in keys:
    p.print_Prog(i)
    gob = int(orbit.replace('/orbit', ''))
    path = lookup[gob]

    idx_start = h5file[orbit]['idx_start'][0]
    idx_end = h5file[orbit]['idx_end'][0]

    process_list.append([path,None,None])# idx_start, idx_end])
    i+=1
    break
p.close_Prog()
h5file.close()

print('start processing',len(process_list),'tracks')

pool = multiprocessing.Pool(nb_cores)
results = [pool.apply_async(sar_processor, t) for t in process_list]

p = prog.Prog(len(process_list))
i=0
for result in results:
    p.print_Prog(i)
    dummy = result.get()
    i+=1
print('done')
