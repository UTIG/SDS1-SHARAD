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

import logging

import numpy as np
import pandas as pd

# TODO GNG: not sure if this does anything
sys.path.append('../xlib')
sys.path.append('../xlib/rdr')


# TODO: Improve globs to assert that there is only one file matching the pattern

def params():
    """Get various parameters defining the dataset
    """
    # GNG TODO: make this an object so that we can override params, then pass it in.
    # GNG TODO: make this code work even if we're not executing from the current directory.
    # Should use __file__
    out = {'code_path': os.getcwd()}
    # GNG TODO: make this use SDS environment variable
    # GNG TODO: convert this to use os.path
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


def orbit2full(orbit, p=params()):
    # prefer not to use "2" to make code more internationally readable and
    # less biased toward native English speakers
    logging.warning("Use of 'orbit2full()' is deprecated. Please use 'orbit_to_full()'")
    return orbit_to_full(orbit, p)

def orbit_to_full(orbit, p=params()):
    """Output the full orbit name(s) avaialble for a given orbit"""
    k = p['orbit'].index(orbit)
    return p['orbit_full'][k]


def check(path, verbose=True):
    """Check if file/path exist
    """
    path_exists = os.path.exists(path)
    if verbose and not path_exists:
        print('MISSING: ' + path)
    return path_exists


def aux(orbit, p=params()):
    """Output content of the auxilliary file for a given orbit
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = os.path.join(p['edr_path'], p['orbit_path'][k], orbit_full + '_a.dat' )
    foo = check(fil)
    # TODO GNG: this definition can be made global in the module
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
    with open(fil, 'r') as f:
        out = np.fromfile(f, dtype = dtype, count = -1)
    return out


def alt(orbit, typ='deriv', p=params()):
    """Output data processed and archived by the altimetry processor
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['alt_path'], p['orbit_path'][k], typ, '*'] ) )[0]
    foo = check(fil)
    out = np.load(fil)
    return out


def tec(orbit, typ='ion', p=params()):
    """Output TEC data
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*TECU.txt']  )  )[0]
    foo = check(fil)
    out = np.loadtxt(fil)
    return out


def cmp(orbit, typ='ion', p=params()):
    """Output data processed and archived by the CMP processor
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*.h5']   )   )[0]
    foo = check(fil)
    re = pd.read_hdf(fil,key='real').values
    im = pd.read_hdf(fil,key='imag').values
    out = re + 1j*im
    return out
    

def srf(orbit, typ='cmp', p=params()):
    """Output data processed and archived by the altimetry processor
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['srf_path'], p['orbit_path'][k], typ, '*']  )  )[0]
    foo = check(fil)
    out = np.load(fil)
    return out


def my(orbit, p=params()):
    """Output martian year for a given orbit (gives the MY at the beginning of the orbit)
    """
    orbit_full = orbit if orbit.find('_') is 1 else orbit2full(orbit)
    a = aux(orbit_full)['GEOMETRY_EPOCH']

    timestr = [np.str(a[0])][0]

    yr   = np.int(timestr[ 2: 6])
    mnth = np.int(timestr[ 7: 9])
    dy   = np.int(timestr[10:12])
    hr   = np.int(timestr[13:15])
    mnt  = np.int(timestr[16:18])
    scnd = np.int(timestr[19:21])

    MY, Ls = solar_longitude.Ls(yr, mnth, dy, hr, mnt, scnd)
    return MY

