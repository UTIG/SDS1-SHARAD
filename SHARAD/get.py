# Tools to explore SHARAD data

__authors__ = ['Cyril Grima, cyril.grima@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'November 30 2018',
         'author': 'Cyril Grima, UTIG',
         'info': 'First release.'}}

import os
import sys
import glob
import string
import numpy as np
import pandas as pd
sys.path.append('../xlib')
sys.path.append('../xlib/rdr')
#import solar_longitude
import cmp.pds3lbl as pds3


def params():
    """Get various parameters defining the dataset
    """
    out = {'code_path': os.getcwd()}
    out['data_path'] = '/disk/daedalus/sds/targ/xtra/SHARAD'
    out['data_product'] = os.listdir(out['data_path'])
    for i in out['data_product']:
        out[i + '_path'] = out['data_path'] + '/' + i
    out['orig_path'] = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'
    out['edr_path'] = out['orig_path'] + '/EDR'
    out['orig_product'] = os.listdir(out['orig_path'])
    for i in out['orig_product']:
        out[i + '_path'] = out['orig_path'] + '/' + i
    out['orbit_full'] = [i.split('/')[-1].split('.')[0]
                         for i in glob.glob(out['EDR_path'] + '/*/data/*/*/*.lbl')]
    out['orbit'] = [i.split('_')[1] for i in out['orbit_full']]
    out['orbit_path'] = ['/'.join(i.split('/')[-5:-1])
                         for i in glob.glob(out['EDR_path'] + '/*/data/*/*/*.lbl')]
    out['dataset'] = out['data_path'].split('/')[-1]

    out['mrosh'] = [i.split('/')[0] for i in out['orbit_path']]
    return out


def orbit2full(orbit):
    """Output the full orbit name(s) avaialble for a given orbit"""
    p = params()
    k = p['orbit'].index(orbit)
    out = p['orbit_full'][k]
    return out


def check(path, verbose=True):
    """Check if file/path exist
    """
    out = os.path.exists(path)
    if out is False:
        message = 'MISSING: ' + path
        if verbose is True:
            print(message)
    return out


def aux(orbit):
    """Output content of the auxilliary file for a given orbit
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = '/'.join( [p['edr_path'], p['orbit_path'][k], orbit_full + '_a.dat'] )
    foo = check(fil)
    # READ AUX FILE
    dtype = np.dtype([
        ("SCET_BLOCK_WHOLE", '>u4'),
        ("SCET_BLOCK_FRAC", '>u2'),
        ("EPHEMERIS_TIME", '>i8'),
        ("GEOMETRY_EPOCH", 'S23'),
        ("SOLAR_LONGITUDE", '>f8'),
        ("ORBIT_NUMBER", '>i4'),
        ("X_MARS_SC_POSITION_VECTOR", '>f8'),
        ("Y_MARS_SC_POSITION_VECTOR", '>f8'),
        ("Z_MARS_SC_POSITION_VECTOR", '>f8'),
        ("SPACECRAFT_ALTITUDE", '>f8'),
        ("SUB_SC_EAST_LONGITUDE", '>f8'),
        ("SUB_SC_PLANETOCENTRIC_LATITUDE", '>f8'),
        ("SUB_SC_PLANETOGRAPHIC_LATITUDE", '>f8'),
        ("X_MARS_SC_VELOCITY_VECTOR", '>f8'),
        ("Y_MARS_SC_VELOCITY_VECTOR", '>f8'),
        ("Z_MARS_SC_VELOCITY_VECTOR", '>f8'),
        ("MARS_SC_RADIAL_VELOCITY", '>f8'),
        ("MARS_SC_TANGENTIAL_VELOCITY", '>f8'),
        ("LOCAL_TRUE_SOLAR_TIME", '>f8'),
        ("SOLAR_ZENITH_ANGLE", '>f8'),
        ("SC_PITCH_ANGLE", '>f8'),
        ("SC_YAW_ANGLE", '>f8'),
        ("SC_ROLL_ANGLE", '>f8'),
        ("MRO_SAMX_INNER_GIMBAL_ANGLE", '>f8'),
        ("MRO_SAMX_OUTER_GIMBAL_ANGLE", '>f8'),
        ("MRO_SAPX_INNER_GIMBAL_ANGLE", '>f8'),
        ("MRO_SAPX_OUTER_GIMBAL_ANGLE", '>f8'),
        ("MRO_HGA_INNER_GIMBAL_ANGLE", '>f8'),
        ("MRO_HGA_OUTER_GIMBAL_ANGLE", '>f8'),
        ("DES_TEMP", '>f4'),
        ("DES_5V", '>f4'),
        ("DES_12V", '>f4'),
        ("DES_2V5", '>f4'),
        ("RX_TEMP", '>f4'),
        ("TX_TEMP", '>f4'),
        ("TX_LEV", '>f4'),
        ("TX_CURR", '>f4'),
        ("CORRUPTED_DATA_FLAG1", 'B'),
        ("CORRUPTED_DATA_FLAG2", 'B'),
    ])
    f = open(fil, 'r')
    out = np.fromfile(f, dtype = dtype, count = -1)
    f.close()
    return out


def alt(orbit, typ='deriv'):
    """Output data processed and archived by the altimetry processor
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['alt_path'], p['orbit_path'][k], typ, '*'] ) )[0]
    foo = check(fil)
    out = np.load(fil)
    return out


def tec(orbit, typ='ion'):
    """Output TEC data
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*TECU.txt']  )  )[0]
    foo = check(fil)
    out = np.loadtxt(fil)
    return out


def cmp(orbit, typ='ion'):
    """Output data processed and archived by the CMP processor
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*.h5']   )   )[0]
    foo = check(fil)
    re = pd.read_hdf(fil,key='real').values
    im = pd.read_hdf(fil,key='imag').values
    out = re + 1j*im
    return out


def srf(orbit, typ='cmp'):
    """Output data processed and archived by the altimetry processor
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['srf_path'], p['orbit_path'][k], typ, '*']  )  )[0]
    foo = check(fil)
    out = np.load(fil)
    return out


def my(orbit):
    """Output martian year for a given orbit (gives the MY at the beginning of the orbit)
    """
    p = params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    a = aux(orbit_full)['GEOMETRY_EPOCH']

    yr = np.int( [np.str(a[0])][0][2:6] )
    mnth = np.int( [np.str(a[0])][0][7:9] )
    dy = np.int( [np.str(a[0])][0][10:12] )
    hr = np.int([np.str(a[0])][0][13:15])
    mnt = np.int([np.str(a[0])][0][16:18])
    scnd = np.int([np.str(a[0])][0][19:21])

    MY, Ls = solar_longitude.Ls(yr, mnth, dy, hr, mnt, scnd)
    return MY

