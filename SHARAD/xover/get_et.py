#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'October 29, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'March 26, 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'Reorganized code.'}
}

"""
This is a short script to extract only the ephemeris time from
each record within the SHARAD data set.
These times are mainly used for the cross-over search.
"""

import sys
import os
import glob

import numpy as np
import pvl
sys.path.append("../../xlib/cmp")
import pds3lbl as pds3

def read_science_np(label_path, data_path):

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

    label = pvl.load(label_path)
    dtype = []
    #nlist=[]
    spare = 0
    for column in label:
        name = column[1]['NAME']
        if 'SPARE' in name:
            name = 'SPARE'+str(spare)
            spare += 1
        if 'MSB_BIT_STRING' in column[1]['DATA_TYPE']:
            for sub in column[1]:
                if 'BIT_COLUMN' in sub:
                    name = sub[1]['NAME']
                    if 'PULSE_REPETITION_INTERVAL' in name:
                        dtype.append('<u1')
                        #nlist.append(name)
                        dtype.append('S15')
                        #nlist.append('MSB_BITSTRING')
                    elif 'SCIENTIFIC_DATA_TYPE' in name:
                        dtype.append('S2')
                         #nlist.append('PACKET')

        else:
            dty = PDS3_DATA_TYPE_TO_DTYPE[column[1]['DATA_TYPE']] + \
                  str(column[1]['BYTES'])
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
    dstr = dtype[0]
    #nstr=nlist[0]
    for d in range(1, len(dtype)):
        dstr = dstr + ',' + dtype[d]
        #nstr=nstr+','+nstr[d]
    dtype = np.dtype(dstr)

    # Open label file corresponding to science file
    try:
        science_label = pvl.load(data_path.replace('_a_a.dat', '_a.lbl'))
    except:
        new_path, filename = data_path.rsplit('/', 1)
        os.chdir(new_path)
        for file in glob.glob("*.LBL"):
            science_label = pvl.load(file)
        for file in glob.glob("*.lbl"):
            science_label = pvl.load(file)
    rows = science_label['FILE']['SCIENCE_TELEMETRY_TABLE']['ROWS']
    #print(dtype.itemsize) #3787 bytes
    fil = glob.glob(data_path)[0]
    columns = 1
    a = np.fromfile(fil, dtype=dtype)
    #print a
    out = np.reshape(a, [columns, rows])
    return out



def main():
    # get a list of all files in sharad data folder
    file_list = []
    # identify data records and corresponding label files
    raw = '/disk/kea/SDS/orig/supl/SHARAD/raw/'
    records = []
    lbls = []
    for path, subdirs, files in os.walk(raw):
        for name in files:
            f = os.path.join(path, name)
            if f.endswith('_a_a.dat'):
                #if '1748102' in f or '1855601' in f:
                #print(f)
                records.append(f)
                lbls.append(f.replace('_a_a.dat', '_a.lbl'))


    #np.savetxt('lookup.txt',np.array(records),fmt='%s')
    #quit()
    # path to science auxillary label file
    lbl_file = os.path.join(raw, 'mrosh_0001/label/auxiliary.fmt')

    #p=prog.Prog(int(len(records)))
    print("Found {:d} record files in {:s}".format(len(records), raw))
    for i, record in enumerate(records):
        #p.print_Prog(int(i))
        rec = pds3.read_science(lbl_file, record, science=False)
        rec2 = read_science_np(lbl_file, record)[0]
        et = rec['EPHEMERIS_TIME']#np.zeros(len(rec))
        print(rec['SCET_BLOCK_WHOLE'])
        #for j in range(len(rec)):
        #    et[j]=np.double(rec[j][2])#+np.double(rec[j][1])/(2.0**16)
        #    print (et[j])
        quit()
        #np.save('/disk/kea/SDS/code/xtra/SHARAD/XOVER/mc11_'+str(i).zfill(5),et)


if __name__ == "__main__":
    # execute only if run as a script
    main()

