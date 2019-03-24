#!/usr/bin/env python3  
# Tools to explore SHARAD data

__authors__ = ['Cyril Grima, cyril.grima@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'November 30 2018',
         'author': 'Cyril Grima, UTIG',
         'info': 'First release.'}}

"""
# Propose calling this SDSEnv

sdsenv = SDSEnv(name='SHARAD')

sdsenv.orbit_data(orbitname)

sdsenv.alt_filename()
sdsenv.alt_data()

"""



import os
import sys
import glob
import string
import traceback
import logging
import time
import re

import numpy as np
import pandas as pd

codepath=os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(codepath, "../xlib")) )
#sys.path.append(os.path.normpath(os.path.join(codepath, "../xlib/rdr")))
from rdr import solar_longitude

#2006-12-06T02:22:01.951
p_timestamp = re.compile("(\d\d\d\d)-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.(\d\d\d)")

# TODO GNG: this definition can be made global in the module
# READ AUX FILE
aux_dtype = np.dtype([
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



def make_orbit_info(f):
    orbit_name1, orbit_ext = os.path.splitext(os.path.basename(f))
    orbit = orbit_name1.split('_')[1]
    # orbit_path = os.path.dirname(f)
    orbit_path = os.path.join(* f.split('/')[-5:-1])

    return orbit, {
        'name': orbit_name1, # full basic orbit name (excluding .lbl; this is orbit_full)
        'file': f,           # full filename
        'path': orbit_path,  # relative path to the file
    }

# TODO GNG: Improve globs to assert that there is only one file matching the pattern

# TODO GNG: Propose class SDSEnv
def params(data_path='/disk/daedalus/sds/targ/xtra/SHARAD', orig_path='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'):
    """Get various parameters defining the dataset
    """
    # GNG TODO: make this an object so that we can override params, then pass it in.
    # GNG TODO: make this code work even if we're not executing from the current directory.
    # Should use __file__

    out = {'code_path': os.path.abspath( os.path.dirname(__file__))}
    # GNG TODO: make this use SDS environment variable
    # GNG TODO: convert this to use os.path
    out['data_path'] = data_path
    out['data_product'] = os.listdir(out['data_path'])
    for s in out['data_product']:
        out[s + '_path'] = os.path.join(out['data_path'], s)
    out['orig_path'] = orig_path
    out['edr_path'] = os.path.join(out['orig_path'], 'EDR')
    logging.debug("edr_path: " + out['edr_path'])
    out['orig_product'] = os.listdir(out['orig_path'])
    for s in out['orig_product']:
        out[s + '_path'] = os.path.join(out['orig_path'] ,  s)
    # TODO: turn this into a dict, and make a search function called get_orbit?
    # Turn this into a normal iteration and generate lists

    label_files =  glob.glob(os.path.join(out['EDR_path'], '*/data/*/*/*.lbl'))
    logging.debug("Found {:d} label files".format(len(label_files)))

    # For each label file, get the full name of the basename everything before the extension
    out['orbit_full'] = [f.split('/')[-1].split('.')[0]
                         for f in label_files]
    # For each full orbit name, extract just the orbit and put that into a list
    out['orbit'] = [s.split('_')[1] for s in out['orbit_full']]
    # Extract the orbit path also as part of that name
    out['orbit_path'] = ['/'.join(f.split('/')[-5:-1])
                         for f in label_files]

    # TODO: allow multiple files to map to one orbit name
    dict_orbitinfo = {} # map orbit name prefix to full orbit name
    for f in label_files:
        orbit, orbitinfo = make_orbit_info(f)

        if orbit not in dict_orbitinfo:
            dict_orbitinfo[orbit] = []

        dict_orbitinfo[orbit].append(orbitinfo)

    out['orbit_info'] = dict_orbitinfo
    
    out['dataset'] = os.path.basename(out['data_path'])
    logging.debug("dataset: " + out['dataset'])
    out['mrosh'] = [i.split('/')[0] for i in out['orbit_path']]
    return out

# TODO GNG: Propose making a member of SDSEnv
def orbit2full(orbit,p=None):
    # prefer not to use "2" to make code more internationally readable and
    # less biased toward native English speakers
    if p is None:
        p=params()
    logging.warning("Use of 'orbit2full()' is deprecated. Please use 'orbit_to_full()'")
    return orbit_to_full(orbit, p)

# TODO GNG: Propose making a member of SDSEnv
def orbit_to_full(orbit, p=None):
    """ 
    Output the full orbit name(s) available for a given orbit
    input "orbit" may either be the short orbit name (such as 0704902)
    or the full orbit name (such as e_0704902_001_ss05_700_a)
    If the short orbit name, return a list of all orbits matching that orbit

    """
    if p is None:
        p=params()

    return get_orbit_info(orbit, p)[0].get('name', None)


def get_orbit_info(orbit, p=None):
    if p is None:
        p=params()

    # Check if this is a short orbit name or a full orbit name
    if '_' in orbit:
        orbit_fullname = orbit
        orbit = orbit_fullname.split('_')[1]
    else:
        orbit_fullname = ''

    try:
        if orbit_fullname != '':
            # TODO: replace with filter()
            list_ret = []
            for x in p['orbit_info'][orbit]:
                if x['name'] == orbit_fullname:
                    list_ret.append(x)
            return list_ret
        else:
            return p['orbit_info'][orbit]
        
    except KeyError as e:
        return [{}]


def check(path, verbose=True):
    """Check if file/path exist
    """
    path_exists = os.path.exists(path)
    if verbose and not path_exists:
        print('MISSING: ' + path)
    return path_exists


# TODO GNG: Propose making a member of SDSEnv
def aux(orbit, p=None, count=-1):
    """Output content of the auxilliary file for a given orbit
    """
    global aux_dtype
    if p is None:
        p=params()

    #logging.debug("getting info for orbit {:s}".format(orbit))
    list_orbit_info = get_orbit_info(orbit, p)

    nitems = len( list_orbit_info )
    if nitems > 1:
       logging.warning("Orbit {:s} has {:d} files".format(orbit, nitems))
    orbit_info = list_orbit_info[0]

    if 'path' not in orbit_info: # orbit not found
        return None
    fil = os.path.join(p['edr_path'], orbit_info['path'], orbit_info['name'] + '_a.dat' )
    #logging.debug("aux(): opening {:s}".format(fil))
    out = np.fromfile(fil, dtype = aux_dtype, count = count)
    return out


# TODO GNG: Propose making this a member of SDSEnv
def alt(orbit, typ='deriv', p=None):
    """Output data processed and archived by the altimetry processor
    """
    if p is None:
        p=params()

    list_orbit_info = get_orbit_info(orbit, p)

    nitems = len( list_orbit_info )
    if nitems > 1:
       logging.warning("Orbit {:s} has {:d} files".format(orbit, nitems))
    orbit_info = list_orbit_info[0]

    if 'path' not in orbit_info: # orbit not found
        return None

    #orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
    #k = p['orbit_full'].index(orbit_full)
    path1 = os.path.join(p['alt_path'], orbit_info['path'], typ, '*')
    files = glob.glob( path1 )
    if not files:
        return None # no file found
    # TODO: assert glob only has one result
    return np.load(files[0])


# TODO GNG: Propose making this a member of SDSEnv
# TODO GNG: Propose renaming tec_data
# TODO GNG: Propose parallel function tec_filepaths?
def tec(orbit, typ='ion', p=None):
    """Output TEC data
    """
    if p is None:
        p=params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*TECU.txt']  )  )[0]
    foo = check(fil)
    out = np.loadtxt(fil)
    return out


# TODO GNG: Propose making this a member of SDSEnv
def cmp(orbit, typ='ion', p=None):
    """Output data processed and archived by the CMP processor
    """
    if p is None:
        p=params()
    orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['cmp_path'], p['orbit_path'][k], typ, '*.h5']   )   )[0]
    foo = check(fil)
    re = pd.read_hdf(fil,key='real').values
    im = pd.read_hdf(fil,key='imag').values
    return re + 1j*im
    

# TODO GNG: Propose making this a member of SDSEnv
def srf(orbit, typ='cmp', p=None):
    """Output data processed and archived by the altimetry processor
    """
    if p is None:
        p=params()

    orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
    k = p['orbit_full'].index(orbit_full)
    fil = glob.glob( '/'.join( [p['srf_path'], p['orbit_path'][k], typ, '*']  )  )[0]
    foo = check(fil)
    out = np.load(fil)
    return out


# TODO GNG: Propose making this a member of SDSEnv
def my(orbit, p=None):
    """Output martian year for a given orbit (gives the MY at the beginning of the orbit)
    """
    global p_timestamp

    if p is None:
        p=params()

    auxdata = aux(orbit, p, count=1)
    if auxdata is None:
        logging.debug("No data found for orbit '{:s}'".format(orbit))
        return None

    a = auxdata['GEOMETRY_EPOCH']

    # a=['2006-12-06T02:22:01.945' '2006-12-06T02:22:01.951'
    #logging.debug("a={!s}".format(a))
    #logging.debug("a[0]={!s}".format(a[0]))

    m_tim = p_timestamp.match(a[0].decode())
    if m_tim:
        yr, mnth, dy, hr, mnt, scnd = tuple([ int(s) for s in m_tim.group(1,2,3,4,5,6) ])
        MY, Ls = solar_longitude.Ls(yr, mnth, dy, hr, mnt, scnd)
    else:
        logging.error("Can't parse timestamp for orbit {:s}: '{:s}'".format(orbit, a[0]))
        MY = None
    return MY


def test_my(p):
    orbitnames1 = sorted(p['orbit_info'].keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for x in p['orbit_info'][orbit]:
            orbitnames2.append( x['name'] )
    orbitnames1.sort()



    # what happens when you run my on something that doesn't exist
    MYEAR = my('doesnt_exist',p)
    MYEAR = my('doesntexist',p)

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_my: Number of orbits: {:d}".format(len(orbitnames)))

        for i, orbit in enumerate(orbitnames):
            try:
                MYEAR = my(orbit, p)
            except ValueError as e:
                logging.info("orbit {:s}: error running my".format(orbit))
                raise # traceback.print_exc(file=sys.stdout)
                MYEAR = "ERROR"
            #logging.info("orbit {:s}: MY={!r}".format(orbit, MYEAR))

            if i % 2000 == 0:
                logging.info("test_my: {:d} of {:d}".format(i, len(orbitnames)) )

    logging.info("test_my: done")
    return 0

def test_alt(p):

    orbitnames1 = sorted(p['orbit_info'].keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for x in p['orbit_info'][orbit]:
            orbitnames2.append( x['name'] )
    orbitnames1.sort()



    # what happens when you run my on something that doesn't exist
    altdata = alt('doesnt_exist',p=p)
    altdata = alt('doesntexist',p=p)

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_alt: Test getting altimetry data. Number of orbits: {:d}".format(len(orbitnames)))

        nsucceeded=0
        nfailed=0
    
        for i, orbit in enumerate(orbitnames):
            altdata = alt(orbit, p=p)
            if altdata is None:
                nfailed += 1
            else:
                nsucceeded += 1
            if i % 2000 == 0:
                logging.info("test_alt: {:d} of {:d}".format(i, len(orbitnames)) )

        logging.info("test_alt: done.  {:d} succeeded, {:d} failed".format(nsucceeded, nfailed)) 
    return 0



def main(): 
    import time

    t0 = time.time()

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
        format='[%(relativeCreated)5d %(name)-6s %(levelname)-7s] %(message)s')
    logging.info("Starting main()")
    # Exercise functions
    p = params()


    # Test what happens when you look for an orbit that doesn't exist
    y = get_orbit_info('orbit_that_doesnt_exist', p)
    assert(len(y) == 1 and len(y[0]) == 0) # should be a list with a dict

    test_my(p)
    test_alt(p)


    logging.info("Done in {:0.1f} seconds".format(time.time() - t0))



if __name__ == "__main__":
    # execute only if run as a script
    main()




