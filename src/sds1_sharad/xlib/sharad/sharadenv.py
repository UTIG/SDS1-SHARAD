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



import csv
import os
import sys
import glob
import logging
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import h5py as h5
#import rsr

CODEPATH = os.path.dirname(__file__)
p1 = os.path.abspath(os.path.join(CODEPATH, ".."))
sys.path.insert(1, p1)
from rdr import solar_longitude
import misc.fileproc as fileproc


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
        # aka SHARAD product_id
        'name': orbit_name1,
        # full filename and path (previously 'file')
        'fullpath': filename,
        # relative path to the directory containing
        # the data files (previously 'path')
        'relpath': orbit_path,
    }

class DataMissingException(FileNotFoundError):
    pass

# TODO: Improve globs to assert that there is only one file matching the pattern

class SHARADEnv:
    """ Class for interacting with data files in the SHARAD dataset """
    def __init__(self, data_path:str=None, orig_path:str=None, b_index_files=True):
        """Get various parameters defining the dataset  """

        if data_path is None:
            data_path = '/disk/kea/SDS/targ/xtra/SHARAD'
            logging.warning("Creating SHARADEnv with default data_path parameters, which is deprecated")
        if orig_path is None:
            orig_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'
            logging.warning("Creating SHARADEnv with default orig_path parameters, which is deprecated")

        self.out = None
        self.orbitinfo = None
        self.code_path = os.path.abspath(os.path.dirname(__file__))
        self.data_path = data_path
        self.orig_path = orig_path
        self.sfiles = None

        if b_index_files:
            self.index_files()
        else:
            self.sfiles = SHARADFiles(data_path, orig_path, read_edr_index=True)

    def index_files(self, use_edr_index=False, index_intermediate_files=False):
        """ Index files under the specified root directories """
        logging.debug("Indexing files in %s", self.get_edr_path())

        self.out = {}
        # GNG TODO: make this use SDS environment variable
        # TODO: glob, too?

        #self.out['data_product'] = os.listdir(self.data_path)
        for sname in os.listdir(self.data_path):
            self.out[sname + '_path'] = os.path.join(self.data_path, sname)

        for sname in os.listdir(self.orig_path):
            self.out[sname + '_path'] = os.path.join(self.orig_path, sname)

        if not use_edr_index:
            # Look for files on disk
            # NOTE: this doesn't find *.LBL
            globpat = os.path.join(self.get_edr_path(), '*/data/*/*/*.lbl')
            label_files = glob.glob(globpat)
        else:
            # Look for files in the index
            label_files = self.list_edr_labels()
        label_files.sort()

        logging.debug("Found %d label files", len(label_files))

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
        if index_intermediate_files:
            self.index_intermediate_files()

    def index_intermediate_files(self):
        """ List files of available for all data products """
        for orbit, suborbits in self.orbitinfo.items():
            for suborbit in suborbits:
                for typ in self.out:
                    if typ == 'EDR_path':
                        path = os.path.join(self.get_edr_path(),
                                            suborbit['relpath']
                                            ) + '/*'
                    elif typ == 'foc_path':
                        path = os.path.join(self.out[typ],
                                            suborbit['relpath']
                                            ) + '/*/*/*/*'
                    else:
                        path = os.path.join(self.out[typ],
                                            suborbit['relpath']
                                            ) + '/**/' + suborbit['name'] + '*'
                    files = glob.glob(path)
                    suborbit[typ.replace('_','')] = files

        #out['dataset'] = os.path.basename(out['data_path'])
        #logging.debug("dataset: " + out['dataset'])
        # This isn't ever used.
        #self.mrosh = [i.split('/')[0] for i in out['orbit_path']]

    def list_edr_labels(self):
        labels = []
        basepath = os.path.join(self.sfiles.orig_path, 'EDR')
        for pinfo in self.sfiles.product_id_index.values():
            edr_lbl = os.path.join(basepath, pinfo['volume_id'], pinfo['file_specification_name'])
            labels.append(edr_lbl)
        return labels

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
            # This is a full name
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
                        # This shouldn't happen if fullname is full,
                        # but can if products show up in duplicate locations
                        logging.warning("SHARADEnv found %d files for orbit %s/%s",
                                        len(list_ret), orbit, orbit_fullname)
                    return list_ret[0]
                else:
                    return list_ret
            else:
                if b_single:
                    if len(self.orbitinfo[orbit]) > 1:
                        logging.warning("SHARADEnv found %d files for orbit %s",
                                        len(self.orbitinfo[orbit]), orbit)
                    return self.orbitinfo[orbit][0]
                else:
                    return self.orbitinfo[orbit]
                return self.orbitinfo[orbit]
        except KeyError:
            return {} if b_single else [{}]



    def aux_data(self, orbit, count=-1):
        """Output content of the auxiliary file for a given orbit
        count is an optional variable that, if provided, limits the number
        of records read from the file.
        """
        #logging.debug("getting info for orbit {:s}".format(orbit))
        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            raise DataMissingException('No aux data for orbit %s' % (orbit,))

        if self.sfiles is not None:
            assert orbit.startswith('e_'), "Only works with full orbit names"
            fil = self.sfiles.edr_product_paths(orbit)['edr_aux']
        else:
            fil = os.path.join(self.get_edr_path(), orbit_info['relpath'],
                               orbit_info['name'] + '_a.dat')
            #logging.debug("aux(): opening {:s}".format(fil))
        out = np.fromfile(fil, dtype=AUX_DTYPE, count=count)
        return out


    def alt_data(self, orbit, typ='beta5', ext='h5', quality_flag=False):
        """Output data processed and archived by the altimetry processor
        """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            raise DataMissingException('No alt data found for %s' % orbit)
        if self.sfiles is not None:
            # Prefer to use the index
            assert ext == 'h5', "Unsupported file extension"
            fil = self.sfiles.alt_product_paths(orbit_info['name'], typ=typ)['alt_h5']

        else:
            fil = os.path.join(self.out['alt_path'], orbit_info['relpath'],
                                 typ, orbit_info['name'] + '_a.' + ext)
        # Call global function
        return alt_data(fil, typ=typ, quality_flag=quality_flag)


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
        if self.sfiles is not None:
            # Prefer to use the newer index
            assert orbit.startswith('e_'), "Only works with full orbit names"
            fil = self.sfiles.cmp_product_paths(orbit)['cmp_rad']
        else:
            globpat = os.path.join(self.out['cmp_path'],
                                   orbit_info['relpath'], typ, orbit_info['name'] + '*.h5')
            fil = sorted(glob.glob(globpat))[0]
        if fil.endswith('.h5'):
            redata = pd.read_hdf(fil, key='real').values
            # Can be complex64 with no loss of precision, because these are 16-bit values,
            # which can be represented by a float32 that has a 24-bit significand
            out = np.empty(redata.shape, dtype=np.complex64)
            out.real = redata
            del redata
            out.imag = pd.read_hdf(fil, key='imag').values
        elif fil.endswith('.i'):
            json_path = Path(fil).with_suffix('.json')
            with json_path.open('rt') as fhjson:
                shape1 = tuple(json.load(fhjson)['shape'])
            assert len(shape1) == 3, "Expecting a 3D integer radargram (shape=%r)" % (shape1,)
            temp = np.memmap(fil, mode='r', dtype=np.int16, shape=shape1)
            out = np.empty(shape1[0:2], dtype=np.complex64)
            out.real = temp[:, :, 0]
            out.imag = temp[:, :, 1]
            return out
        else:
            raise RuntimeError("Don't know how to open %s" % (fil,))
        return out

    def srf_data(self, orbit, typ='cmp'):
        """Output data processed and archived by the surface processor
        """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            raise DataMissingException('No srf data found for %s' % orbit)

        path1 = os.path.join(self.out['srf_path'], orbit_info['relpath'],
                             typ, orbit_info['name'] + '*.txt')
        files = glob.glob(path1)
        # TODO: assert glob only has one result
        if not files:
            return None # no file found

        # TODO: assert glob only has one result
        out = np.genfromtxt(files[0], delimiter=',', names=True)
        return out


    def rsr_data(self, orbit, typ='cmp'):
        """Output data processed and archived by the rsr processor
        """

        orbit_info = self.get_orbit_info(orbit, True)

        if 'relpath' not in orbit_info: # orbit not found
            return None

        path1 = os.path.join(self.out['rsr_path'], orbit_info['relpath'],
                             typ, orbit_info['name'] + '*.txt')
        files = glob.glob(path1)
        # TODO: assert glob only has one result
        if not files:
            return None # no file found

        # TODO: assert glob only has one result
        out = np.genfromtxt(files[0], delimiter=',', names=True)
        return out



    def processed(self):
        """Output processed data products for each processing (e.g., cmp, alt)
        For now, each suborbit having at least one file in a certain processing
        folder is considered 'processed' for this specific processing category
        """
        # TODO: convert this to use sets
        output = {}
        for typ in self.out:
            output[typ.split('_')[0]] = []

        for suborbits in self.orbitinfo.values():
            for suborbit in suborbits:
                for datatype in output:
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
            output[typ] = list(np.unique(sorted(output[typ])))

        return output

    def processed1(self):
        """Output processed data products for each processing (e.g., cmp, alt)
        For now, each suborbit having at least one file in a certain processing
        folder is considered 'processed' for this specific processing category
        This is meant to be equivalent to processed, but faster
        """
        output = {}
        for typ in self.out:
            output[typ.split('_')[0]] = set()

        for orbit, suborbits in self.orbitinfo.items():
            for suborbit in suborbits:
                for datatype, productset in output.items():
                    k = datatype + 'path'
                    if k not in suborbit:
                        continue
                    added = False
                    for s in suborbit[k]:
                        # If any of the outputs appear, register as present
                        # (even though this isn't quite right)
                        for ext in ('.txt', '.h5', '.dat'):
                            if s.endswith(ext):
                                productset.add(suborbit['name'])
                                added = True
                                break
                        if added:
                            break

        for typ in output.keys():
            output[typ] = sorted(output[typ])

        return output

    def processed2(self, products_requested=None):
        """Output processed data products for each processing (e.g., cmp, alt)
        Each individual data product (suborbit) must have all files.

        This method, compared to processed, uses the SHARADFiles index and
        only checks against the files requested, rather than all files in
        the archive.
        """
        output = {}
        for typ in self.out:
            output[typ.split('_')[0]] = set()

        if products_requested is None:
            products_requested = self.orbitinfo.keys()

        for orbit in products_requested:
            for suborbit in self.orbitinfo[orbit]:
                for datatype in output.keys():
                    try:
                        assert isinstance(suborbit['name'], str), "suborbit=%r" % (suborbit)
                        paths = self.sfiles.product_paths(collection=datatype, product_id=suborbit['name'])
                    except AttributeError:
                        # datatype isn't a valid collection
                        continue
                    if fileproc.all_files_exist(paths.values()):
                        output[datatype].add(suborbit['name'])

        for typ in output.keys(): # convert sets to lists
            output[typ] = sorted(output[typ])

        return output


    def quadrangle(self, MCid, orbitlist=True, filename=None):
        """Gives a Martian quadrangle complete name, boundaries and orbits
        crossing it.

        Inputs
        ------
        MCid: string
            ID of the quadrangle (e.g., 'MC15')
        orbitlist: bool
            Whether to provide a list of orbits crossing the quandrangle
        filename: string
            file name to save the orbit list. Does not save if None.

        Output
        ------
            dictionary of parameters for the considered quadrangle
        """

        # Quadrangle dictionary
        quad = {'MC01': {'name':'Mare Boreum',
                         'lat':[65, 90], 'lon':[0, 360]},
                'MC02': {'name':'Diacria',
                         'lat':[30,65], 'lon':[180,240]},
                'MC03': {'name':'Arcadia',
                         'lat':[30,65], 'lon':[240,300]},
                'MC04': {'name':'Mare Acidalium',
                         'lat':[30,65], 'lon':[300,360]},
                'MC05': {'name':'Ismenius Lacus',
                         'lat':[30,65], 'lon':[0,60]},
                'MC06': {'name':'Casius',
                         'lat':[30,65], 'lon':[60,120]},
                'MC07': {'name':'Cebrenia',
                         'lat':[30,65], 'lon':[120,180]},
                'MC08': {'name':'Amazonis',
                         'lat':[0,30], 'lon':[180,225]},
                'MC09': {'name':'Tharsis',
                         'lat':[0,30], 'lon':[225,270]},
                'MC10': {'name':'Lunae Palus',
                         'lat':[0,30], 'lon':[270,315]},
                'MC11': {'name':'Oxia Palus',
                         'lat':[0,30], 'lon':[315,360]},
                'MC12': {'name':'Arabia',
                         'lat':[0,30], 'lon':[0,45]},
                'MC13': {'name':'Syrtis Major',
                         'lat':[0,30], 'lon':[45,90]},
                'MC14': {'name':'Amenthes',
                        'lat':[0,30], 'lon':[90,135]},
                'MC15': {'name':'Elysium',
                         'lat':[0,30], 'lon':[135,180]},
                'MC16': {'name':'Memnonia',
                         'lat':[-30,0], 'lon':[180,225]},
                'MC17': {'name':'Phoenicis Lacus',
                         'lat':[-30,0], 'lon':[225,270]},
                'MC18': {'name':'Coprates',
                         'lat':[-30,0], 'lon':[270,315]},
                'MC19': {'name':'Margaritifer Sinus',
                         'lat':[-30,0], 'lon':[315,360]},
                'MC20': {'name':'Sinus Sabaeus',
                         'lat':[-30,0], 'lon':[0,45]},
                'MC21': {'name':'Iapygia',
                         'lat':[-30,0], 'lon':[45,90]},
                'MC22': {'name':'Mare Tyrrhenum',
                         'lat':[-30,0], 'lon':[90,135]},
                'MC23': {'name':'Aeolis',
                         'lat':[-30,0], 'lon':[135,180]},
                'MC24': {'name':'Phaethontis',
                         'lat':[-65,-30], 'lon':[180,240]},
                'MC25': {'name':'Thaumasia',
                         'lat':[-65,-30], 'lon':[240,300]},
                'MC26': {'name':'Argyre',
                         'lat':[-65,-30], 'lon':[300,360]},
                'MC27': {'name':'Noachis',
                         'lat':[-65,-30], 'lon':[0,60]},
                'MC28': {'name':'Hellas',
                         'lat':[-65,-30], 'lon':[60,120]},
                'MC29': {'name':'Eridania',
                         'lat':[-65,-30], 'lon':[120,180]},
                'MC30': {'name':'Mare Australe',
                         'lat':[-90,-65], 'lon':[0,360]},
               }

        out = {'ID':MCid, 'name':quad[MCid]['name'],
               'lon':quad[MCid]['lon'], 'lat':quad[MCid]['lat']}

        if orbitlist:
            labels = ["SUB_SC_EAST_LONGITUDE", "SUB_SC_PLANETOCENTRIC_LATITUDE"]
            conditions = [out['lon'], out['lat']]
            orbits = self.aux_query(labels, conditions)
            out['orbits'] = orbits

        if filename is not None:
            np.savetxt(filename, out, fmt="%s")

        return out


    def my(self, orbit):
        """Output martian year for a given orbit
        (gives the MY at the beginning of the orbit)
        """
        try:
            auxdata = self.aux_data(orbit, count=1)
        except DataMissingException:
            logging.debug("No data found for orbit '%s'", orbit)
            return None

        # a=['2006-12-06T02:22:01.945' '2006-12-06T02:22:01.951'
        try:
            tstamp = auxdata['GEOMETRY_EPOCH'][0].decode()
            jsec = solar_longitude.ISO8601_to_J2000(tstamp)
            myear, _ = solar_longitude.Ls_J2000(jsec)
        except ValueError:
            logging.error("Can't parse timestamp for orbit %s: %r", orbit, tstamp)
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
        # TODO: remove verbose flag and just do logging
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
            if verbose:
                print(str(i) + '/' + str(len(orbits)) + ' : ' +orbit)

            if self.my(orbit):
                try:
                    df = pd.DataFrame(self.aux_data(orbit)[::sampling][aux_columns])
                except ValueError:
                    df = pd.DataFrame(self.aux_data(orbit)[::sampling][aux_columns].byteswap().newbyteorder())
                df['orbit'] = np.full(len(df), orbit)
                df['martian_year'] = np.full(len(df), self.my(orbit))
                store.append(product, df)

        store.close()


    def aux_query(self, labels, conditions, aux_filename='aux.h5',
                  product='sampling_1000', filename=None):
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
                          filename='gather_datapoints.h5', verbose=True,
                          bad_orbits_filename='bad_orbits.txt'):
        """Gather in a hdf5 file the data points of orbits matching the
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

        bad_orbits_filename: string
            a file where bad orbits that should not be considered are listed
        """

        # Get orbits matching the conditions and for which data exists
        orbits = self.aux_query(labels, conditions)
        processed = self.processed()['srf']
        orbits = list(set(orbits) & set(processed))
        if os.path.isfile(bad_orbits_filename):
            bad_orbits = np.loadtxt(bad_orbits_filename, dtype=str)
            orbits = list(set(orbits) - set(bad_orbits))
        orbits.sort()

        # Store data
        store = pd.HDFStore(filename)

        for i, orbit in enumerate(orbits):
            if verbose:
                print(str(i) + '/' + str(len(orbits)) + ' : ' +orbit)
            aux = self.aux_data(orbit)
            data = getattr(self, product + '_data')(orbit)
            df = pd.DataFrame(data)
            #df[orbit] = np.full(len(aux['ORBIT_NUMBER']), orbit)
            # Points without auxilliary flag
            ok = (aux['CORRUPTED_DATA_FLAG1'] == 0) &\
                 (aux['CORRUPTED_DATA_FLAG2'] == 0)
            # Points matching the conditions
            for n, label in enumerate(labels):
                ok = ok & (df[label] > conditions[n][0]) & \
                          (df[label] < conditions[n][1])
            if i == 0:
                store.put(product, df[ok], format='table', data_columns=True )
            else:
                store.append(product, df[ok], data_columns=True)

        store.close()


    def gather_datapoints_quadrangle(self, MCid, folder='./', **kwargs):
        """Applies self.gather_datapoints in a specific quadrangle

        Inputs
        ------
        ID: string
            ID of the quadrangle (e.g., 'MC15')
        product: string
            Product to read and store in the aux file
        folder: string
            Folder to store the file
        """
        q = self.quadrangle(MCid)
        labels = ["SUB_SC_EAST_LONGITUDE", "SUB_SC_PLANETOCENTRIC_LATITUDE"]
        conditions = [q['lon'], q['lat']]

        filename = (folder + q['ID'] + '_' + q['name'] +
                    '.h5').replace(' ', '_')

        self.gather_datapoints(labels, conditions, filename=filename, **kwargs)

######################################################
# Functions for reading slightly-complicated intermediate processing data products into
# useful numpy/pandas data structures
def alt_data(altfile: str, typ='beta5', quality_flag=False):
    """Output data processed and archived by the altimetry processor
    """
    ext = os.path.splitext(altfile)[1].lower()
    if not os.path.exists(altfile):
        raise FileNotFoundError(altfile + " does not exist")
    if ext == '.h5':
        with h5.File(altfile, 'r') as fh:
            data = fh[typ]
            orbit_key = list(data.keys())[0]
            data = data[orbit_key]

            out = {'et':data['block0_values'][:, 0]}
            for i, val in enumerate(data['block0_items'][:]):
                key = str(val).replace('b', '').replace('\'', '')
                vec = data['block0_values'][:, i]
                out[key] = vec

    elif ext == '.npy':
        out = np.load(altfile)
    else: # pragma: no cover
        raise ValueError("Unknown file extension for " + altfile)

    # Quality flag, assess if the orbit has picks that can be trusted
    if quality_flag:
        y = out['idx_fine']
        # Detect negative y coordinates
        flag = (y < 0)
        # Coherent content for the y position
        # (i.e., is the detected surface more or less continuous?)
        #cc = rsr.run.processor(y, fit_model='rice').power()['pc-pn']
        #if cc < 10:
        #    flag = flag + True
        flag = [int(i) for i in flag]
        out['flag'] = flag

    return out







class SHARADFiles:
    """ class for calculating SDS1-SHARAD file paths from product IDs
    and vice versa """
    def __init__(self, data_path:str, orig_path:str, read_edr_index=False):
        """Get various parameters defining the dataset  """
        # Typical values
        #data_path = '/disk/kea/SDS/targ/xtra/SHARAD'
        #orig_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD'

        #self.code_path = os.path.abspath(os.path.dirname(__file__))
        self.data_path = data_path
        self.orig_path = orig_path
        if read_edr_index:
            self.read_edr_index()

    def read_edr_index(self):
        """ Build index of product IDs to locations.
        We can read any one of cumindex.tab """
        self.product_id_index = {}
        self.orbit_index = defaultdict(list)
        fields = 'volume_id release_id file_specification_name product_id'.split()

        indexpat = os.path.join(self.orig_path, 'EDR/mrosh_000?/index/cumindex.tab')
        try:
            indexfile = glob.glob(indexpat)[0]
        except IndexError as ierr: # pragma: no cover
            raise FileNotFoundError("Could not find %s" % indexpat) from ierr

        logging.info("Read %s", indexfile)
        with open(indexfile, 'rt') as fhcsv:
            reader = csv.DictReader(fhcsv, fieldnames=fields, restkey='__rest__')
            for row in reader:
                del row['__rest__'] # keep only the first 4 columns
                # Remove extra spaces from fields
                for k in ('volume_id', 'file_specification_name', 'product_id'):
                    row[k] = row[k].rstrip().lower()
                row['relpath'] = os.path.dirname(row['file_specification_name'])
                self.product_id_index[row['product_id']] = row
                orbit = row['product_id'][2:9]
                self.orbit_index[orbit].append(row['product_id'])
        logging.debug("product_id_index has %d items", len(self.product_id_index))


    def data_product_paths(self, product_id: str, types=('edr', 'cmp', 'alt', 'srf', 'rsr', 'clu', 'rng', 'foc')):
        """ Populate a dictionary with all data file locations """
        products = {}
        for prod in types:
            f = getattr(self, prod + '_product_paths')
            products.update(f(product_id))
        return products

    def product_paths(self, collection: str, product_id: str, **kwargs):
        """ Return the paths of files with given product ID and collection types
        """
        f = getattr(self, collection + '_product_paths')
        return f(product_id, **kwargs)


    def edr_product_paths(self, product_id: str):
        """ Return paths to the three EDR files
        described in section 4.3.5 of the EDR SIS
        https://pds-geosciences.wustl.edu/mro/mro-m-sharad-3-edr-v1/mrosh_0001/document/edrsis.pdf
        """
        # /disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr01xxx/edr0187401/e_0187401_00
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.orig_path, 'EDR', pinfo['volume_id'], pinfo['relpath'])

        return {
            'edr_lbl': os.path.join(self.orig_path, 'EDR', pinfo['volume_id'], pinfo['file_specification_name']),
            'edr_aux': os.path.join(orbitdir, product_id + '_a.dat'),
            'edr_sci': os.path.join(orbitdir, product_id + '_s.dat'),
        }


    def cmp_product_paths(self, product_id: str, typ:str='ion'):
        """ Return paths to range compression outputs
        (produced by run_rng_cmp.py)
        TODO: better mnemonic than h5 
        Example: /disk/kea/SDS/targ/xtra/SHARAD/cmp/mrosh_0004/data/rm262/edr4839701/ion/e_4839701_001_ss4_700_a_s_TECU.txt
        """
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'cmp', pinfo['volume_id'], pinfo['relpath'], typ)

        return {
            'cmp_rad': os.path.join(orbitdir, product_id + '_s.h5'),
            #'cmp_rad': os.path.join(orbitdir, product_id + '_s.i'),
            'cmp_tecu': os.path.join(orbitdir, product_id + '_s_TECU.txt'),
        }

    def alt_product_paths(self, product_id: str, typ:str='beta5'):
        """ Calculate output paths for altimetry data products
        Example: /disk/kea/SDS/targ/xtra/SHARAD/alt/mrosh_0001/data/edr08xxx/edr0898101/beta5/e_0898101_001_ss19_700_a_a.h5
        """

        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'alt', pinfo['volume_id'], pinfo['relpath'], typ)
        return {'alt_h5': os.path.join(orbitdir, product_id + '_a.h5'),}

    def srf_product_paths(self, product_id: str, typ:str='cmp'):
        """ Example: 
        /disk/kea/SDS/targ/xtra/SHARAD/srf/mrosh_0001/data/edr20xxx/edr2007501/cmp/e_2007501_001_ss19_700_a.txt"""
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'srf', pinfo['volume_id'], pinfo['relpath'], typ)

        return {'srf_txt': os.path.join(orbitdir, product_id + '.txt'),}


    def rsr_product_paths(self, product_id: str, typ:str='cmp'):
        """ Example:
        /disk/kea/SDS/targ/xtra/SHARAD/rsr/mrosh_0002/data/edr24xxx/edr2484501/cmp/e_2484501_001_ss04_700_a.txt"""
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'rsr', pinfo['volume_id'], pinfo['relpath'], typ)
        return {'rsr_txt': os.path.join(orbitdir, product_id + '.txt'),}

    def clu_product_paths(self, product_id: str, typ='icd', start='0', end='end'):
        """ Cluttergram simulation outputs """
        pinfo = self.product_id_index[product_id]
        rangestr = str(start) + '-' + str(end)
        # New path
        # orbitdir = os.path.join(self.data_path, 'clu', pinfo['volume_id'], pinfo['relpath'], typ, rangestr)
        orbitdir = os.path.join(self.data_path, 'clu', pinfo['volume_id'], pinfo['relpath'], typ)
        return {
            'clu_rad': os.path.join(orbitdir, product_id + '.npz'),
            'clu_json': os.path.join(orbitdir, product_id + '.json'),
        }


    def rng_product_paths(self, product_id: str, typ:str='icd'):
        """ Ranging product produced by run_ranging.py. Example
        /disk/kea/SDS/targ/xtra/SHARAD/rng/mrosh_0004/data/rm270/edr4973503/icd/e_4973503_001_ss19_700_a_a.cluttergram.npy"""
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'rng', pinfo['volume_id'], pinfo['relpath'], typ)
        return {'rng_json': os.path.join(orbitdir, product_id + '.json'),}


    def foc_product_paths(self, product_id: str, typ:str=None):
        """ Focused radargrams produced by run_sar2.py Example:
        /disk/kea/SDS/targ/xtra/SHARAD/foc/mrosh_0001/data/edr04xxx/edr0490602/5m/5 range lines/6km/e_0490602_001_ss19_700_a_s.h5
        TODO: include all types, and remove spaces
        """
        pinfo = self.product_id_index[product_id]
        orbitdir = os.path.join(self.data_path, 'foc', pinfo['volume_id'], pinfo['relpath'])
        products = {}
        for ii, typ in enumerate(('5m/5 range lines/6km',)):
            k = 'foc_p%d' % (ii,)
            products[k] = os.path.join(orbitdir, typ, product_id + '_s.h5')
        return products

    def orbitid_to_productids(self, orbit_id: str):
        """ Return all product IDs associated with a given orbit (transaction) """
        assert len(orbit_id) == 7
        searchstr = 'e_' + orbit_id
        for k in self.product_id_index.keys():
            if k.startswith(searchstr):
                yield k


    def product_processing_status(self, product: str, product_id: str, **kwargs):
        """ Calculate whether the specified product is up to date
        Inputs:
        product is a processing step for the output.
        Choices are cmp, alt, srf, rsr, rng, foc.
        product_id is a SHARAD product id

        Return value:
        Returns the input and output status and list of
        files used.
        """
        prerequisites = {
            'cmp': ('edr',),
            'alt': ('cmp', 'edr'),
            'srf': ('cmp', 'alt'),
            'rsr': ('cmp', 'alt'),
            'foc': ('edr', 'cmp'),
        }
        if product not in prerequisites:
            raise KeyError('Unknown product %s' % product)

        input_files = []
        for type in prerequisites[product]:
            input_files.extend(self.product_paths(type, product_id).values(), **kwargs)

        output_files = list(self.product_paths(product, product_id, **kwargs).values())
        filestatus = fileproc.file_processing_status(input_files, output_files, **kwargs)
        return filestatus, input_files, output_files

