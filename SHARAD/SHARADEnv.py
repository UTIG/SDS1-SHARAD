#!/usr/bin/env python3
# Tools to explore SHARAD data

__authors__ = ['Cyril Grima, cyril.grima@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'November 30 2018',
         'author': 'Cyril Grima, UTIG',
         'info': 'First release.'},
    '1.0':
        {'date': 'March 26 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'First release.'},


}

"""
SHARADEnv
SharadData
SHARADData


sharadenv = SHARADEnv()

sharadenv.orbit_data(orbitname)

sharadenv.alt_filename()
sharadenv.alt_data()

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
import h5py as h5

codepath = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(codepath, "../xlib")) )
#sys.path.append(os.path.normpath(os.path.join(codepath, "../xlib/rdr")))
from rdr import solar_longitude

#2006-12-06T02:22:01.951
p_timestamp = re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.(\d\d\d)')

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
        # full basic orbit name (excluding .lbl; this is orbit_full)
        'name': orbit_name1,
        # full filename and path (previously 'file')
        'fullpath': f,
        # relative path to the directory containing
        # the data files (previously 'path')
        'relpath': orbit_path,
    }

# TODO GNG: Improve globs to assert that there is only one file matching the pattern

class SHARADEnv:
    """ Class for interacting with data files in the SHARAD dataset """
    def __init__(self, data_path='/disk/kea/SDS/targ/xtra/SHARAD',
                     orig_path='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'):
        """Get various parameters defining the dataset
        """

        self.code_path = os.path.abspath( os.path.dirname(__file__))
        self.data_path = data_path
        self.orig_path = orig_path

        self.index_files()


    def index_files(self):
        """ Index files under the specified root directories """
        logging.debug("Indexing files in {:s}".format( self.get_edr_path() ) )

        self.out={}
        # GNG TODO: make this use SDS environment variable

        #self.out['data_product'] = os.listdir(self.data_path)
        for sname in os.listdir(self.data_path):
            self.out[sname + '_path'] = os.path.join(self.data_path, sname)

        for sname in os.listdir(self.orig_path):
            self.out[sname + '_path'] = os.path.join(self.orig_path, sname)
        # TODO: turn this into a dict, and make a search function called get_orbit?
        # Turn this into a normal iteration and generate lists


        globpat = os.path.join(self.get_edr_path(), '*/data/*/*/*.lbl')
        label_files =  glob.glob(globpat)

        logging.debug("Found {:d} label files".format(len(label_files)))

        ## For each label file, get the full name of the basename everything before the extension
        #out['orbit_full'] = [f.split('/')[-1].split('.')[0]
        #                 for f in label_files]
        ## For each full orbit name, extract just the orbit and put that into a list
        #out['orbit'] = [s.split('_')[1] for s in out['orbit_full']]
        ## Extract the orbit path also as part of that name
        #out['orbit_path'] = ['/'.join(f.split('/')[-5:-1])
        #                 for f in label_files]

        self.orbitinfo = {} # map orbit name prefix to full orbit name
        for f in label_files:
            orbit, orbitinfo = make_orbit_info(f)

            if orbit not in self.orbitinfo:
                self.orbitinfo[orbit] = []

            self.orbitinfo[orbit].append(orbitinfo)

        #out['dataset'] = os.path.basename(out['data_path'])
        #logging.debug("dataset: " + out['dataset'])
        # This isn't ever used.
        #self.mrosh = [i.split('/')[0] for i in out['orbit_path']]


    def get_edr_path(self):
        return os.path.join( self.orig_path, 'EDR' )

    # TODO GNG: Propose making a member of SDSEnv
    def orbit_to_full(self, orbit):
        """
        Output the full orbit name(s) available for a given orbit
        input "orbit" may either be the short orbit name (such as 0704902)
        or the full orbit name (such as e_0704902_001_ss05_700_a)
        If the short orbit name, return a list of all orbits matching that orbit

        """
        return self.get_orbit_info(orbit, True).get('name', None)


    def get_orbit_info(self, orbit, b_single=False):
        """
        Check if this is a short orbit name or a full orbit name
        If b_single is True, then return a single item, and throw a
        warning if there is more than one result.
        If b_single is False (default), then returns a list of all items
        matching the orbit search criterion
        """

        if '_' in orbit:
            # This is a full  name
            orbit_fullname = orbit
            orbit = orbit_fullname.split('_')[1]
        else:
            # This is a short name
            orbit_fullname = ''

        try:
            if orbit_fullname != '':
                # TODO: replace with filter()
                list_ret = []
                for x in self.orbitinfo[orbit]:
                    if x['name'] == orbit_fullname:
                        list_ret.append(x)
                if b_single:
                    if len(list_ret) > 1:
                        logging.warning("SHARADEnv found {:d} "
                                        "files for orbit {:s}".format(
                                        len(list_ret), orbit))
                    return list_ret[0]
                else:
                    return list_ret
            else:
                if b_single:
                    if len(self.orbitinfo[orbit]) > 1:
                        logging.warning("SHARADEnv found {:d} "
                                        "files for orbit {:s}".format(
                                        len(self.orbitinfo[orbit]), orbit))
                    return self.orbitinfo[orbit][0]
                else:
                    return self.orbitinfo[orbit]
                return self.orbitinfo[orbit]
        except KeyError as e:
            return [{}]


    def check(path, verbose=True):
        """Check if file/path exist
            TODO: this isn't really that useful.
        """
        path_exists = os.path.exists(path)
        if verbose and not path_exists:
            print('MISSING: ' + path)
        return path_exists


    def aux_data(self, orbit, count=-1):
        """Output content of the auxiliary file for a given orbit
        count is an optional variable that, if provided, limits the number
        of records read from the file.
        """
        global aux_dtype

        #logging.debug("getting info for orbit {:s}".format(orbit))
        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None
        fil = os.path.join(self.get_edr_path(), orbit_info['relpath'], orbit_info['name'] + '_a.dat')
        #logging.debug("aux(): opening {:s}".format(fil))
        out = np.fromfile(fil, dtype=aux_dtype, count=count)
        return out


    def alt_data(self, orbit, typ='deriv', ext='npy'):
        """Output data processed and archived by the altimetry processor
        """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None

        #orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
        #k = p['orbit_full'].index(orbit_full)
        path1 = os.path.join(self.out['alt_path'], orbit_info['relpath'], typ, '*.') + ext
        files = glob.glob(path1)
        if not files:
            return None # no file found
        # TODO: assert glob only has one result
        if ext is 'h5':
            d = h5.File(files[0], 'r')[typ][orbit]
            out = {'et':d['block0_values'][:,0]}
            for i, val in enumerate(d['block0_items'][:]):
                key = str(val).replace('b','').replace('\'','')
                vec = d['block0_values'][:,i]
                out[key] = vec
        elif ext is 'npy':
            d = np.load(files[0])
            out = d
        return out


    # TODO GNG: Propose parallel function tec_filepaths?
    def tec_data(self, orbit, typ='ion'):
        """Output total electron content data
        Total Electron Content
        """
        orbit_info = self.get_orbit_info(orbit, True)

        fil = glob.glob('/'.join(self.out['cmp_path'], self.orbit_info['relpath'], typ, '*TECU.txt'))[0]
        #foo = check(fil)
        out = np.loadtxt(fil)
        return out


    def cmp_data(self,orbit, typ='ion'):
        """Output data processed and archived by the CMP processor
        """
        orbit_info = self.get_orbit_info(orbit, True)

        globpat = os.path.join( self.out['cmp_path'],  orbit_info['relpath'], typ, '*.h5' )
        fil = sorted(glob.glob(globpat))[0]
        re = pd.read_hdf(fil, key='real').values
        im = pd.read_hdf(fil, key='imag').values
        return re + 1j*im


    def srf_data(self,orbit, typ='cmp'):
        """Output data processed and archived by the altimetry processor (surface)
        """

        orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
        k = senv.orbit_full.index(orbit_full)
        globpat = os.path.join(self.srf_path, self.orbit_path[k], typ, '*')
        fil = glob.glob(globpat)[0]
        # TODO: assert only one file found
        return np.load(fil)


    def my(self, orbit):
        """Output martian year for a given orbit (gives the MY at the beginning of the orbit)
        """
        global p_timestamp

        auxdata = self.aux_data(orbit, count=1)
        if auxdata is None:
            logging.debug("No data found for orbit '{:s}'".format(orbit))
            return None

        # a=['2006-12-06T02:22:01.945' '2006-12-06T02:22:01.951'
        try:
            ts = auxdata['GEOMETRY_EPOCH'][0].decode()
            jsec = solar_longitude.ISO8601_to_J2000(ts)
            MY, Ls = solar_longitude.Ls_J2000(jsec)
        except ValueError as e:
            logging.error("Can't parse timestamp for orbit {:s}: '{:s}'".format(orbit, ts))
            MY, Ls = None, None

        return MY


def test_my(senv):
    orbitnames1 = sorted(senv.orbitinfo.keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for x in senv.orbitinfo[orbit]:
            orbitnames2.append( x['name'] )
    orbitnames1.sort()


    # what happens when you run my on something that doesn't exist
    MYEAR = senv.my('doesnt_exist')
    MYEAR = senv.my('doesntexist')

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_my: Number of orbits: {:d}".format(len(orbitnames)))

        for i, orbit in enumerate(orbitnames):
            try:
                MYEAR = senv.my(orbit)
            except ValueError as e:
                logging.info("orbit {:s}: error running my".format(orbit))
                raise # traceback.print_exc(file=sys.stdout)
                MYEAR = "ERROR"
            #logging.info("orbit {:s}: MY={!r}".format(orbit, MYEAR))

            if i % 2000 == 0:
                logging.info("test_my: {:d} of {:d}".format(i, len(orbitnames)) )

    logging.info("test_my: done")
    return 0

def test_alt(senv):

    orbitnames1 = sorted(senv.orbitinfo.keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for x in senv.orbitinfo[orbit]:
            orbitnames2.append(x['name'])
    orbitnames1.sort()

    # what happens when you run my on something that doesn't exist
    altdata = senv.alt_data('doesnt_exist')
    altdata = senv.alt_data('doesntexist')

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_alt: Test getting altimetry data. Number of orbits: {:d}".format(len(orbitnames)))

        nsucceeded=0
        nfailed=0

        for i, orbit in enumerate(orbitnames):
            altdata = senv.alt_data(orbit)
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
    senv = SHARADEnv()

    # Test what happens when you look for an orbit that doesn't exist
    y = senv.get_orbit_info('orbit_that_doesnt_exist')
    assert(len(y) == 1 and len(y[0]) == 0) # should be a list with a dict

    test_my(senv)
    test_alt(senv)

    logging.info("Done in {:0.1f} seconds".format(time.time() - t0))



if __name__ == "__main__":
    # execute only if run as a script
    main()




