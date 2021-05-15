#!/usr/bin/env python3
# Tools to explore SHARAD data

__authors__ = ['Cyril Grima, cyril.grima@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'November 30 2018',
         'author': 'Cyril Grima, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'March 26 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'Changed to SHARADEnv'},


}

"""
sharadenv = SHARADEnv()

sharadenv.orbit_data(orbitname)

sharadenv.alt_filename()
sharadenv.alt_data()

"""



import os
import sys
import glob
import logging

import numpy as np
import pandas as pd
import h5py as h5

CODEPATH = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(CODEPATH, "../xlib")))
from rdr import solar_longitude


# READ AUX FILE
AUX_DTYPE = np.dtype([
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



def make_orbit_info(filename):
    """ Construct a dict of basic path information commonly
    used about an orbit """
    orbit_name1, _ = os.path.splitext(os.path.basename(filename))
    orbit = orbit_name1.split('_')[1]
    # orbit_path = os.path.dirname(filename)
    orbit_path = os.path.join(* filename.split('/')[-5:-1])

    return orbit, {
        # full basic orbit name (excluding .lbl; this is orbit_full)
        'name': orbit_name1,
        # full filename and path (previously 'file')
        'fullpath': filename,
        # relative path to the directory containing
        # the data files (previously 'path')
        'relpath': orbit_path,
    }

# TODO: Improve globs to assert that there is only one file matching the pattern

class SHARADEnv:
    """ Class for interacting with data files in the SHARAD dataset """
    def __init__(self, data_path='/disk/kea/SDS/targ/xtra/SHARAD',
                 orig_path='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'):
        """Get various parameters defining the dataset  """

        self.out = {}
        self.orbitinfo = {}
        self.code_path = os.path.abspath(os.path.dirname(__file__))
        self.data_path = data_path
        self.orig_path = orig_path

        self.index_files()


    def index_files(self):
        """ Index files under the specified root directories """
        logging.debug("Indexing files in {:s}".format(self.get_edr_path()))

        self.out = {}
        # GNG TODO: make this use SDS environment variable
        # TODO: glob, too?

        #self.out['data_product'] = os.listdir(self.data_path)
        for sname in os.listdir(self.data_path):
            self.out[sname + '_path'] = os.path.join(self.data_path, sname)

        for sname in os.listdir(self.orig_path):
            self.out[sname + '_path'] = os.path.join(self.orig_path, sname)

        globpat = os.path.join(self.get_edr_path(), '*/data/*/*/*.lbl')
        label_files = glob.glob(globpat)
        label_files.sort()

        logging.debug("Found {:d} label files".format(len(label_files)))

        ## For each label file, get the full name of the basename everything
        ## before the extension
        #out['orbit_full'] = [f.split('/')[-1].split('.')[0]
        #                 for f in label_files]
        ## For each full orbit name, extract just the orbit and put that into
        ## a list
        #out['orbit'] = [s.split('_')[1] for s in out['orbit_full']]
        ## Extract the orbit path also as part of that name
        #out['orbit_path'] = ['/'.join(f.split('/')[-5:-1])
        #                 for f in label_files]

        self.orbitinfo = {} # map orbit name prefix to full orbit name
        for filename in label_files:
            orbit, orbitinfo = make_orbit_info(filename)

            if orbit not in self.orbitinfo:
                self.orbitinfo[orbit] = []

            self.orbitinfo[orbit].append(orbitinfo)

        # List files of avaialble for all data products
        for orbit in self.orbitinfo:
            for subid, suborbit in enumerate(self.orbitinfo[orbit]):
                for typ in self.out:
                    if typ == 'EDR_path':
                        path = os.path.join(self.get_edr_path(),
                                            self.orbitinfo[orbit][subid]['relpath']
                                            ) + '/*'
                    elif typ == 'foc_path':
                        path = os.path.join(self.out[typ],
                                            self.orbitinfo[orbit][subid]['relpath']
                                            ) + '/*/*/*/*'
                    else:
                        path = os.path.join(self.out[typ], 
                                            self.orbitinfo[orbit][subid]['relpath']
                                            ) + '/**/' + suborbit['name'] + '*'
                    files = glob.glob(path)
                    self.orbitinfo[orbit][subid][typ.replace('_','')] = files
        
        #out['dataset'] = os.path.basename(out['data_path'])
        #logging.debug("dataset: " + out['dataset'])
        # This isn't ever used.
        #self.mrosh = [i.split('/')[0] for i in out['orbit_path']]


    def get_edr_path(self):
        """ Return the absolute absolute path of EDR data """
        return os.path.join(self.orig_path, 'EDR')


    def orbit_to_full(self, orbit):
        """
        Output the full orbit name(s) available for a given orbit
        input "orbit" may either be the short orbit name (such as 0704902)
        or the full orbit name (such as e_0704902_001_ss05_700_a)
        If the short orbit name, return a list of all full orbit names matching that orbit

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
                for rec in self.orbitinfo[orbit]:
                    if rec['name'] == orbit_fullname:
                        list_ret.append(rec)
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
        except KeyError:
            return {} if b_single else [{}]


    #def check(path, verbose=True):
    #    """Check if file/path exist
    #        TODO: this isn't really that useful.
    #    """
    #    path_exists = os.path.exists(path)
    #    if verbose and not path_exists:
    #        print('MISSING: ' + path)
    #    return path_exists


    def aux_data(self, orbit, count=-1):
        """Output content of the auxiliary file for a given orbit
        count is an optional variable that, if provided, limits the number
        of records read from the file.
        """
        #logging.debug("getting info for orbit {:s}".format(orbit))
        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None
        fil = os.path.join(self.get_edr_path(), orbit_info['relpath'],
                           orbit_info['name'] + '_a.dat')
        #logging.debug("aux(): opening {:s}".format(fil))
        out = np.fromfile(fil, dtype=AUX_DTYPE, count=count)
        return out


    def alt_data(self, orbit, typ='beta5', ext='h5'):
        """Output data processed and archived by the altimetry processor
        """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None

        #orbit_full = orbit if orbit.find('_') is 1 else orbit_to_full(orbit,p)
        #k = p['orbit_full'].index(orbit_full)
        path1 = os.path.join(self.out['alt_path'], orbit_info['relpath'],
                             typ, orbit_info['name'] + '*.' + ext)
        files = glob.glob(path1)
        if not files:
            return None # no file found
        # TODO: assert glob only has one result
        if ext == 'h5':
            try:
                data = h5.File(files[0], 'r')[typ]
                orbit_key = list(data.keys())[0]
                data = data[orbit_key]
            except (OSError, KeyError) as e:
                logging.error("Can't read {:s}: {:s}".format(files[0], str(e)))
                raise(e)

            out = {'et':data['block0_values'][:, 0]}
            for i, val in enumerate(data['block0_items'][:]):
                key = str(val).replace('b', '').replace('\'', '')
                vec = data['block0_values'][:, i]
                out[key] = vec
        elif ext == 'npy':
            out = np.load(files[0])
        return out


    # TODO GNG: Propose parallel function tec_filepaths?
    # TODO: tec_data isn't called here, it needs to be tested.
    def tec_data(self, orbit, typ='ion'):
        """Output total electron content data
        Total Electron Content
        """
        orbit_info = self.get_orbit_info(orbit, True)

        tecpat = os.path.join(self.out['cmp_path'],
                 orbit_info['relpath'], typ, orbit_info['name'] + '*TECU.txt')

        fil = glob.glob(tecpat)[0]
        out = np.loadtxt(fil)
        return out


    def cmp_data(self, orbit, typ='ion'):
        """Output data processed and archived by the CMP processor
        """
        orbit_info = self.get_orbit_info(orbit, True)

        globpat = os.path.join(self.out['cmp_path'],
                               orbit_info['relpath'], typ, orbit_info['name'] + '*.h5')
        fil = sorted(glob.glob(globpat))[0]
        redata = pd.read_hdf(fil, key='real').values
        imdata = pd.read_hdf(fil, key='imag').values
        return redata + 1j*imdata


    def srf_data(self, orbit, typ='cmp'):
        """Output data processed and archived by the altimetry processor
          (surface) """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None

        path1 = os.path.join(self.out['srf_path'], orbit_info['relpath'],
                             typ, orbit_info['name'] + '*.txt')
        files = glob.glob(path1)
        # TODO: assert glob only has one result
        if not files:
            return None # no file found

        # TODO: assert glob only has one result
        out = np.genfromtxt(files[0], delimiter=',', names=True)
        return out


    def processed(senv):
        """Output processed data products for each processing (e.g., cmp, alt)
        For now, each suborbit having at least one file in a certain processing
        folder is considered 'processed' for this specific processing category
        """
        output = {}
        for typ in senv.out:
            output[typ.split('_')[0]] = []

        for orbit in senv.orbitinfo:
            for suborbit in senv.orbitinfo[orbit]:
                for datatype in output.keys():
                     try:
                         #if any(suborbit[datatype + 'path']):
                         if any(s.endswith('.txt') for s in 
                                suborbit[datatype + 'path']):
                             output[datatype].append(suborbit['name'])
                         if any(s.endswith('.h5') for s in 
                                suborbit[datatype + 'path']):
                             output[datatype].append(suborbit['name'])
                         if any(s.endswith('.dat') for s in
                                suborbit[datatype + 'path']):
                             output[datatype].append(suborbit['name'])

                     except:
                         pass

        for typ in output.keys():
            output[typ] = sorted(output[typ])

        return output

    def my(self, orbit):
        """Output martian year for a given orbit
        (gives the MY at the beginning of the orbit)
        """
        auxdata = self.aux_data(orbit, count=1)
        if auxdata is None:
            logging.debug("No data found for orbit '{:s}'".format(orbit))
            return None

        # a=['2006-12-06T02:22:01.945' '2006-12-06T02:22:01.951'
        try:
            tstamp = auxdata['GEOMETRY_EPOCH'][0].decode()
            jsec = solar_longitude.ISO8601_to_J2000(tstamp)
            myear, _ = solar_longitude.Ls_J2000(jsec)
        except ValueError:
            logging.error("Can't parse timestamp for orbit "
                          "{:s}: '{:s}'".format(orbit, tstamp))
            myear, _ = None, None

        return myear


    def make_aux_file(self, filename='aux.h5', sampling=1000, verbose=True):
        """Gather aux data into a single text file to faciliate queries over 
        aux fields. Uncomment the columns you would want to see in the aux file.
        Fields that are not native from the aux dataset are in lower case.

        Input:
        ------

        filename: string
            Name (with full path) of the csv file to be created

        sampling: int
            Sampling of the aux data
        """

        # Aux fields to extract
        aux_columns=[
                     #"SCET_BLOCK_WHOLE",
                     #"SCET_BLOCK_FRAC",
                     "EPHEMERIS_TIME",
                     #"GEOMETRY_EPOCH",
                     "SOLAR_LONGITUDE",
                     #"ORBIT_NUMBER",
                     #"X_MARS_SC_POSITION_VECTOR",
                     #"Y_MARS_SC_POSITION_VECTOR",
                     #"Z_MARS_SC_POSITION_VECTOR",
                     "SPACECRAFT_ALTITUDE",
                     "SUB_SC_EAST_LONGITUDE",
                     "SUB_SC_PLANETOCENTRIC_LATITUDE",
                     #"SUB_SC_PLANETOGRAPHIC_LATITUDE",
                     #"X_MARS_SC_VELOCITY_VECTOR",
                     #"Y_MARS_SC_VELOCITY_VECTOR",
                     #"Z_MARS_SC_VELOCITY_VECTOR",
                     #"MARS_SC_RADIAL_VELOCITY",
                     #"MARS_SC_TANGENTIAL_VELOCITY",
                     "LOCAL_TRUE_SOLAR_TIME",
                     "SOLAR_ZENITH_ANGLE",
                     "SC_PITCH_ANGLE",
                     "SC_YAW_ANGLE",
                     "SC_ROLL_ANGLE",
                     #"MRO_SAMX_INNER_GIMBAL_ANGLE",
                     #"MRO_SAMX_OUTER_GIMBAL_ANGLE",
                     #"MRO_SAPX_INNER_GIMBAL_ANGLE",
                     #"MRO_SAPX_OUTER_GIMBAL_ANGLE",
                     #"MRO_HGA_INNER_GIMBAL_ANGLE",
                     #"MRO_HGA_OUTER_GIMBAL_ANGLE",
                     #"DES_TEMP",
                     #"DES_5V",
                     #"DES_12V",
                     #"DES_2V5",
                     #"RX_TEMP",
                     #"TX_TEMP",
                     #"TX_LEV",
                     #"TX_CURR",
                     "CORRUPTED_DATA_FLAG1",
                     "CORRUPTED_DATA_FLAG2",
                    ]

        all_columns = aux_columns.copy()
        all_columns.append('orbit')
        all_columns.append('martian_year')

        # Create hdf5 file
        df = pd.DataFrame(columns=all_columns)
        product = 'sampling_' + str(sampling)
        store = pd.HDFStore(filename)
        store.put(product, df, format='table', data_columns=True)

        # Extract and Store Data
        orbits = self.processed()['cmp']

        for i, orbit in enumerate(orbits):
            if verbose == True:
                print(str(i) + '/' + str(len(orbits)) + ' : ' +orbit)

            if self.my(orbit):
                df = pd.DataFrame(self.aux_data(orbit)[::sampling][aux_columns])
                df['orbit'] = np.full(len(df), orbit)
                df['martian_year'] = np.full(len(df), self.my(orbit))
                store.append(product, df)

        store.close()


    def aux_query(self, labels, conditions, aux_filename='aux.h5', 
                  product='sampling_1000', filename=None, **kwargs):
        """Provide a list of orbits matching conditions on their aux labels
        The data are searched within a csv file generated by tools.aux_file

        Input
        -----
        labels: list
            List of string corresponding to the auxilliary labels on which the
            query is done

        conditions: list
            list of number pairs (tuples) defining the lower and upper bonds 
            within which the data fall into. The list should be same length 
            as labels.

        aux_filename: string
            Name of the aux file (hdf5 format) produced by make_aux_file

        product: string
            Product to read in the aux file

        filename: string
            filename to write the output in. If None, do not write.

        Return
        ------
        List (numpy array) of orbit file names matching the requested conditions

        Example
        -------
        labels = ["SUB_SC_EAST_LONGITUDE", "SUB_SC_PLANETOCENTRIC_LATITUDE"]
        conditions = [[0, 90], [30, 60]] 
        a = aux_query(labels, conditions, filename='deuterolinus.csv')
        """
        df = pd.read_hdf(aux_filename, key=product, mode='a', columns=labels+['orbit'])
    
        # Test conditions on each label
        for n, label in enumerate(labels):
            check = (df[label] > conditions[n][0]) & \
	            (df[label] < conditions[n][1])
            if n == 0:
                ok = check
            else:
                ok = ok & check

        out = df['orbit'][ok].unique()

        if filename is not None:
            np.savetxt(filename, out, fmt="%s")
    
        return out


    def gather_datapoints(self, labels, conditions, product='srf', 
                          filename='gather_datapoints.h5', verbose=True):
        """Gather in a hd5 file the data points of orbits matching the
        requested conditions.
  
        Inputs
        ------
        labels: list
            List of string corresponding to the auxilliary labels on which the
            query is done

        conditions: list
            list of number pairs (tuples) defining the lower and upper bonds 
            within which the data fall into. The list should be same length 
            as labels

        product: string
            Product to read and store in the aux file

        filename: string
            filename to write the output in. If None, do not write.
        """

        # Get orbits matching the conditions
        orbits = self.aux_query(labels, conditions)
    
        # Create hdf5 file
        columns = self.srf_data(orbits[0]).dtype.names
        df = pd.DataFrame(columns=columns)
        store = pd.HDFStore(filename)
        store.put(product, df, format='table', data_columns=columns)

        for i, orbit in enumerate(orbits):
            if verbose == True:
                print(str(i) + '/' + str(len(orbits)) + ' : ' +orbit)
            aux = self.aux_data(orbit)
            data = getattr(self, product + '_data')(orbit)
            df = pd.DataFrame(data, columns=columns)

            # Select only points matching the conditions within the orbit
            for n, label in enumerate(labels):
                check = (df[label] > conditions[n][0]) & \
                        (df[label] < conditions[n][1])
                if n == 0:
                    ok = check
                else:
                    ok = ok & check

            if not (any(aux['CORRUPTED_DATA_FLAG1'] == 1) or 
                    any(aux['CORRUPTED_DATA_FLAG2'] == 1)):
                store.append(product, df[ok])

        store.close()


def test_my(senv):
    """ test martian year function """
    orbitnames1 = sorted(senv.orbitinfo.keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for rec in senv.orbitinfo[orbit]:
            orbitnames2.append(rec['name'])
    orbitnames2.sort()


    # what happens when you run my on something that doesn't exist
    myear = senv.my('doesnt_exist')
    assert myear is None
    myear = senv.my('doesntexist')
    assert myear is None

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_my: Number of orbits: {:d}".format(len(orbitnames)))

        for i, orbit in enumerate(orbitnames):
            try:
                myear = senv.my(orbit)
            except ValueError: # pragma: no cover
                logging.info("orbit {:s}: error running my".format(orbit))
                myear = "ERROR"
                raise # traceback.print_exc(file=sys.stdout)
            #logging.info("orbit {:s}: MY={!r}".format(orbit, myear))

            if i % 2000 == 0:
                logging.info("test_my: {:d} of {:d}".format(i, len(orbitnames)))

    logging.info("test_my: done")
    return 0

def test_alt(senv):

    orbitnames1 = sorted(senv.orbitinfo.keys())
    orbitnames2 = []
    for orbit in orbitnames1:
        for rec in senv.orbitinfo[orbit]:
            orbitnames2.append(rec['name'])
    orbitnames1.sort()

    # what happens when you run my on something that doesn't exist
    altdata = senv.alt_data('doesnt_exist')
    altdata = senv.alt_data('doesntexist')

    for orbitnames in (orbitnames1, orbitnames2):
        logging.info("test_alt: Test getting altimetry data. "
                     "Number of orbits: {:d}".format(len(orbitnames)))

        nsucceeded, nfailed = 0, 0

        for i, orbit in enumerate(orbitnames):
            try:
                altdata = senv.alt_data(orbit)
            except (OSError,KeyError) as e:
                #KeyError: "Unable to open object (object 'beta5' doesn't exist)"
                #OSError: Unable to open file (file signature not found)
                logging.error("Can't open h5 for {:s}: {:s}".format(orbit, str(e)))
                return None

            if altdata is None:
                nfailed += 1
            else:
                nsucceeded += 1
            if i % 2000 == 0:
                logging.info("test_alt: {:d}/{:d}".format(i, len(orbitnames)))

        logging.info("test_alt: done. "
                     "{:d} succeeded, {:d} failed".format(nsucceeded, nfailed))
    return 0


def test_orbit_info(senv):
    try:
        # Test what happens when you look for an orbit that doesn't exist
        oinfo = senv.get_orbit_info('orbit_that_doesnt_exist')
        assert len(oinfo) == 1
        assert len(oinfo[0]) == 0 # should be a list with a dict
        assert isinstance(oinfo[0],dict) # should be just a dict
        oinfo = senv.get_orbit_info('orbit_that_doesnt_exist', False)
        assert len(oinfo) == 1
        assert len(oinfo[0]) == 0 # should be a list with a dict
        assert isinstance(oinfo[0],dict) # should be just a dict
        oinfo = senv.get_orbit_info('orbit_that_doesnt_exist', True)
        assert isinstance(oinfo,dict) # should be just a dict
        assert len(oinfo) == 0
    except AssertionError as e:
        raise e

    # Show which orbits contain more than one value
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("Orbits with multiple items: ")
        orbitnames1 = sorted(senv.orbitinfo.keys())
        for orbit in orbitnames1:
            norbits = len(senv.orbitinfo[orbit])
            if norbits > 1:
                logging.debug("Orbit {:s} contains {:d} items:".format(orbit, norbits))
            for i, rec in enumerate(senv.orbitinfo[orbit]):
                logging.debug("{:2d}: {:s}".format(i, str(rec)))



def main():
    """ Test SHARADEnv """
    import time
    t0 = time.time()

    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stdout,
        format='[%(relativeCreated)5d %(name)-6s %(levelname)-7s] %(message)s')
    logging.info("Starting main()")
    # Exercise functions
    senv = SHARADEnv()

    test_orbit_info(senv)

    test_my(senv)
    test_alt(senv)

    logging.info("Done in {:0.1f} seconds".format(time.time() - t0))



if __name__ == "__main__":
    # execute only if run as a script
    main()




