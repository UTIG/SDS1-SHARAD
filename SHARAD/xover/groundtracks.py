#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'July 30, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'}}

"""
This is a short script to extract the ground tracks in lat/lon from
each record within the SHARAD data set. The area is limited to the HRSC MC11E DTM.
"""

import sys
import os
import numpy as np
import importlib.util
import pandas as pd

def read_science_np(label_path,data_path):
    import pvl
    import numpy as np
    import glob
    import os

    PDS3_DATA_TYPE_TO_DTYPE = {
    'DATE' : '>i',
    'BOOLEAN' : '?',
    'IEEE_REAL': '>f',
    'LSB_INTEGER': '<i',
    'LSB_UNSIGNED_INTEGER': '<u',
    'MAC_INTEGER': '>i',
    'MAC_REAL': '>f',
    'MAC_UNSIGNED_INTEGER': '>u',
    'MSB_UNSIGNED_INTEGER': '>u',
    'MSB_INTEGER': '>i',
    'PC_INTEGER': '<i',
    'PC_UNSIGNED_INTEGER': '<u',
    'SUN_INTEGER': '>i',
    'SUN_REAL': '>f',
    'SUN_UNSIGNED_INTEGER': '>u',
    'VAX_INTEGER': '<i',
    'VAX_UNSIGNED_INTEGER': '<u',
}

    label=pvl.load(label_path)
    dtype=[]
    #nlist=[]
    spare=0
    for column in label:
        name=column[1]['NAME']
        if 'SPARE' in name:
            name='SPARE'+str(spare)
            spare+=1
        if 'MSB_BIT_STRING' in column[1]['DATA_TYPE']:
            for sub in column[1]:
                if 'BIT_COLUMN' in sub:
                    name=sub[1]['NAME']
                    if 'PULSE_REPETITION_INTERVAL' in name:
                        dtype.append('<u1')
                        #nlist.append(name)
                        dtype.append('S15')
                        #nlist.append('MSB_BITSTRING')                   
                    elif 'SCIENTIFIC_DATA_TYPE' in name:
                         dtype.append('S2')
                         #nlist.append('PACKET')                       

        else:
            dty=PDS3_DATA_TYPE_TO_DTYPE[column[1]['DATA_TYPE']]+str(
                    column[1]['BYTES'])
            if '<u3' in dty:
                dtype.append('<u2')
                #nlist.append(name+'_MSB')
                dtype.append('<u1')
                #nlist.append(name+'LSB')
            elif '>u3' in dty:
                dtype.append('>u2')
                #nlist.append(name+'MSB')
                dtype.append('>u1')
                #nlist.append(name+'LSB') 
            elif "f28" in str(dty):
                dtype.append('<f16')
                #nlist.append(name+'_MSB')   
                dtype.append('<f8')
                #nlist.append(name+'_LSB')
                dtype.append('<f4')
                #nlist.append(name+'_LSB2')              
            elif "f32" in str(dty):
                dtype.append('<f16')
                #nlist.append(name+'_MSB')    
                dtype.append('<f16')
                #nlist.append(name+'_LSB')
            elif "i23" in str(dty):
                dtype.append('S23')
            else:
                dtype.append(dty)
                #nlist.append(name)
    #for i in range(0,3600):
    #    dtype.append('b')
    #    #nlist.append('sample'+str(i))     
    dstr=dtype[0]
    #nstr=nlist[0]
    for d in range(1,len(dtype)):
        dstr=dstr+','+dtype[d]
        #nstr=nstr+','+nstr[d]
    dtype = np.dtype(dstr)

    # Open label file corresponding to science file
    try:
        science_label = pvl.load(data_path.replace('_a_a.dat','_a.lbl'))
    except:
        new_path, filename = data_path.rsplit('/', 1)
        os.chdir(new_path)
        for file in glob.glob("*.LBL"):
            science_label = pvl.load(file)
        for file in glob.glob("*.lbl"):
            science_label = pvl.load(file)
    rows=science_label['FILE']['SCIENCE_TELEMETRY_TABLE']['ROWS']
    #print(dtype.itemsize) #3787 bytes
    fil = glob.glob(data_path)[0]
    columns = 1
    a = np.fromfile(fil, dtype=dtype)
    #print a
    out = np.reshape(a, [columns, rows])
    return out

spec = importlib.util.spec_from_file_location('prog','/disk/kea/SDS/code/xtra/MISC/prog.py')
prog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prog)

# get a list of all files in sharad data folder
file_list=[]
raw='/disk/kea/SDS/orig/supl/SHARAD/raw/'
for path, subdirs, files in os.walk(raw):
    for name in files:
        file_list.append(os.path.join(path, name))

# identify data records and corresponding label files
records=[]
lbls=[]
for f in file_list:
    if '_a_a.dat' in f:
        records.append(f)
        lbls.append(f.replace('_a_a.dat','_a.lbl'))

# path to science auxillary label file
lbl_file=raw+'/mrosh_0001/label/auxiliary.fmt'

p=prog.Prog(int(len(records)),step=0.1)
print (len(records))

save_path='mc11e_full.h5'
data_columns = ['idx_start','idx_end']
for i in range(len(records)):
    p.print_Prog(int(i))
    rec=read_science_np(lbl_file,records[i])[0]
    et = np.empty(len(rec), dtype = np.double)
    lat = np.empty(len(rec), dtype = np.double)
    lon = np.empty(len(rec), dtype = np.double)
    onb = np.empty(len(rec), dtype = np.int)
    for j in range(len(rec)):
        et[j]=rec[j][2]
        onb[j]=rec[j][5]
        lat[j]=rec[j][11]
        lon[j]=rec[j][10]
    indx = np.where((lat>0) & (lat<30) & (lon>337.5) & (lon<359.9))[0]
    if len(indx)>2:  
        df = pd.DataFrame({'idx_start': indx[0], 'idx_end': indx[-1]})
        df.to_hdf(save_path,records[i],format='table',data_columns=data_columns)
print('tracks done')
