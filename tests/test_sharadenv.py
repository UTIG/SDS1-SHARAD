#!/usr/bin/env python3


import os
import sys
import logging
from pathlib import Path
import json
import time

import unittest


CODEPATH = Path(__file__).parent / '..' / 'src'# / 'sds1_sharad' / 'xlib' / 'sharad'
sys.path.insert(1, str(CODEPATH))

from sds1_sharad.xlib.sharad.sharadenv import SHARADEnv, DataMissingException

ORIG_PATH = None
TARG_PATH = None

#OUTDIR = Path('.')

SKIP_SLOW = False


def setupModule():
    sds = os.getenv('SDS', '/disk/kea/SDS')

    global ORIG_PATH, TARG_PATH
    ORIG_PATH = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD')
    TARG_PATH = os.path.join(sds, 'targ/xtra/SHARAD')


class TestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(orig_path=ORIG_PATH, data_path=TARG_PATH, b_index_files=False)
        cls.senv.index_files(index_intermediate_files=False)

    def test_instantiate_default(self):
        with self.assertLogs(level=logging.WARNING) as cm:
            _ = SHARADEnv(b_index_files=False)

    def test_instantiate_nondefault(self):
        self.senv.index_files(use_edr_index=True)


    def test_ppstatus(self):
        product_id = 'e_0955302_001_ss19_700_a'
        for type in ('cmp', 'alt', 'srf', 'rsr', 'foc'):
            filestatus, input_files, output_files = self.senv.sfiles.product_processing_status(type, product_id)
            assert isinstance(filestatus, tuple)
            assert isinstance(input_files, list)
            assert isinstance(output_files, list)


class TestMartianYear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(orig_path=ORIG_PATH, data_path=TARG_PATH, b_index_files=False)
        cls.senv.index_files(index_intermediate_files=False)

    def test_dne1(self):
        """ what happens when you run my on something that doesn't exist """
        myear = self.senv.my('doesnt_exist')
        assert myear is None
        myear = self.senv.my('doesntexist')
        assert myear is None

    def test_one(self):
        """ Test a few martian years """
        #product_id = 'e_0224401_007_ss05_700_a'
        product_id = 'e_0955302_001_ss19_700_a'
        _ = self.senv.my(product_id)

    #@unittest.skipIf(SKIP_SLOW, "Skipping slow tests")
    def test_z_all_my(self):
        """ test martian year function """
        nmax = 1000 if SKIP_SLOW else None

        product_ids = sorted(get_product_ids_sfiles(self.senv, nmax))

        for i, orbit in enumerate(product_ids):
            # Known empty timestamp for this orbit's label
            if orbit == 'e_0686801_001_ss05_700_a':
                continue
            myear = self.senv.my(orbit)
            #if i % 2000 == 0:
            #    logging.info("test_my: %d of %d", i, len(product_ids))


class TestDataReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(orig_path=ORIG_PATH, data_path=TARG_PATH, b_index_files=False)
        cls.senv.index_files(index_intermediate_files=False)

    def test1(self):
        """ Iterate through types of data readers for one product """
        #product_id = 'e_0224401_007_ss05_700_a'
        product_id = 'e_0955302_001_ss19_700_a'
        for type in ('aux', 'cmp', 'tec', 'alt', 'srf', 'rsr'):
            fname = type + '_data'
            with self.subTest(fname=fname):
                fget = getattr(self.senv, fname)
                try:
                    _ = fget(product_id)
                except FileNotFoundError as excp:
                    logging.warning("Product for %s/%s not found: %r", type,  product_id, excp)


class TestAltimetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(orig_path=ORIG_PATH, data_path=TARG_PATH, b_index_files=False)
        cls.senv.index_files(index_intermediate_files=False)

    def test_dne1(self):
        """ what happens when you run my on something that doesn't exist """
        with self.assertRaises(DataMissingException):
            altdata = self.senv.alt_data('doesnt_exist')

        with self.assertRaises(DataMissingException):
            altdata = self.senv.alt_data('doesntexist')

    def test_qualitytrue(self):
        """ Run with quality flag non-default (True) 
        TODO: have it automatically pick a product ID """
        product_id = 'e_0955302_001_ss19_700_a'
        _ = self.senv.alt_data(product_id, quality_flag=True)

    def test_z_alt_all(self):
        #nmax = 1000 if SKIP_SLOW else None
        nmax = 10

        product_ids = sorted(get_product_ids_sfiles(self.senv, nmax))

        logging.info("test_alt: Test getting altimetry data. "
                     "Number of orbits: %d", len(product_ids))

        nsucceeded, nfailed = 0, 0

        for i, orbit in enumerate(product_ids):
            try:
                altdata = self.senv.alt_data(orbit)
            except KeyError as e:
                #KeyError: "Unable to open object (object 'beta5' doesn't exist)"
                #OSError: Unable to open file (file signature not found)
                logging.warning("Malformatted h5 file for orbit %s: %s", orbit, str(e))
            except OSError as e:
                logging.debug("Can't open h5 for %s: %s", orbit, str(e))
                altdata = None

            if altdata is None:
                nfailed += 1
            else:
                nsucceeded += 1
            if i % 2000 == 0:
                logging.info("test_alt: %d/%d", i, len(product_ids))

        logging.info("test_alt: done. "
                     "%d succeeded, %d failed", nsucceeded, nfailed)

class TestOrbitInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(ORIG_PATH, TARG_PATH)

    def test_dne_default(self):
        """ Run with single=False """
        oinfo = TestOrbitInfo.senv.get_orbit_info('orbit_that_doesnt_exist')
        assert len(oinfo) == 1, "len(oinfo) = %d. oinfo=%r" % (len(oinfo), oinfo)
        assert len(oinfo[0]) == 0, "should be a list with a dict"
        assert isinstance(oinfo[0],dict), "should be just a dict"
    def test_dne_single_false(self):
        """ Run with single=False """
        oinfo = TestOrbitInfo.senv.get_orbit_info('orbit_that_doesnt_exist', False)
        assert len(oinfo) == 1
        assert len(oinfo[0]) == 0 # should be a list with a dict
        assert isinstance(oinfo[0],dict) # should be just a dict
    def test_dne_single_true(self):
        """ Run with single=True """
        oinfo = self.senv.get_orbit_info('orbit_that_doesnt_exist', True)
        assert isinstance(oinfo,dict) # should be just a dict
        assert len(oinfo) == 0

    @unittest.skipIf(SKIP_SLOW, "Skipping slow tests")
    def test_show_multiple_productids(self):
        """ Show which orbits contain more than one value 
        TODO: do this in SHARADFiles which can do it more efficiently
        """
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return

        logging.debug("Orbits with multiple items: ")
        orbitnames1 = sorted(TestOrbitInfo.senv.orbitinfo.keys())
        for orbit in orbitnames1:
            norbits = len(TestOrbitInfo.senv.orbitinfo[orbit])
            if norbits > 1:
                logging.debug("Orbit %s contains %d items:", orbit, norbits)
            for i, rec in enumerate(TestOrbitInfo.senv.orbitinfo[orbit]):
                logging.debug("%2d: %s", i, str(rec))


@unittest.skipIf(SKIP_SLOW, "Skipping slow tests")
class TestProcessed(unittest.TestCase):
    """ Test the functions that look for processed data.
    Do this last because it takes the longest to run """
    @classmethod
    def setUpClass(cls):
        cls.senv = SHARADEnv(orig_path=ORIG_PATH, data_path=TARG_PATH, b_index_files=False)
        cls.senv.index_files(index_intermediate_files=True)



    def test_processedfiles(self):
        """ Run processedfiles and compare it to processed2 """
        t0 = time.time()
        logging.info("index_files() %0.2f seconds", time.time() - t0)

        t0 = time.time()
        p1 = self.senv.processed()
        s1 = json.dumps(p1)
        t1 = time.time()
        dt1 = t1 - t0

        t0 = time.time()
        p2 = self.senv.processed1()
        s2 = json.dumps(p2)
        t1 = time.time()
        dt2 = t1 - t0

        logging.info("len(p1)=%d s1=%d", len(p1), len(s1))
        logging.info("processed() %0.2f seconds; processed1() %0.2f seconds", dt1, dt2)
        assert s1 == s2

    # This was an expected failure when the archive was incomplete
    #@unittest.expectedFailure
    def test_processed2(self):
        p2 = self.senv.processed2()
        #s1 = json.dumps(p2)


def get_product_ids(senv: SHARADEnv, nmax=None, step=None):
    """ Enumerate product IDs in the SHARADEnv index """
    #maxorbits = 1000
    orbitnames1 = sorted(senv.orbitinfo.keys()) # short names
    count = 0
    for orbit in orbitnames1: # full names
        for rec in senv.orbitinfo[orbit]:
            yield rec['name']
            count += 1
            if nmax is not None and count >= nmax:
                return

def get_product_ids_sfiles(senv: SHARADEnv, nmax=None):
    """ Enumerate product IDs in the SHARADEnv index 
    using keys in the SHARADFiles object
    """
    for count, k in enumerate(senv.sfiles.product_id_index, start=1):
        yield k
        if nmax is not None and count >= nmax:
            break


def main():
    """ Test SHARADEnv """
    logging.basicConfig(
        level=logging.WARNING, stream=sys.stdout,
        format='[%(relativeCreated)5d %(name)-6s %(levelname)-7s] %(message)s')

    setupModule()
    unittest.main()
    #senv = SHARADEnv()
    #test_alt(senv)




if __name__ == "__main__":
    main()
