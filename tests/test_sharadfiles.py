#!/usr/bin/env python3

import sys
import os
import logging

from pathlib import Path

import unittest

CODEPATH = Path(__file__).parent / '..' / 'src' / 'sds1_sharad' / 'xlib' / 'sharad'
sys.path.insert(1, str(CODEPATH))
from sharadenv import SHARADFiles



class TestSHARADFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sds = os.getenv('SDS', '/disk/kea/SDS')
        data_path = os.path.join(sds, 'targ/xtra/SHARAD')
        orig_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD')
        cls.sfiles = SHARADFiles(data_path=data_path, orig_path=orig_path)
        cls.sfiles.read_edr_index()


    def test_edr_path(self):
        """ Test the edr_product_path function """
        for product_id in self.sfiles.product_id_index.keys():
            with self.subTest(product_id=product_id):
                pp = self.sfiles.edr_product_paths(product_id)
                for file in pp.values():
                    p_file = Path(file)
                    if p_file.exists():
                        continue

                    # Check for capitalized label name
                    p_file2 = p_file.with_name(p_file.name.upper())
                    if p_file2.exists():
                        continue
                    self.assertTrue(False, file + " does not exist")

                allfiles = self.sfiles.data_product_paths(product_id)

    def test_find_products_by_orbit(self):
        """ Go from a orbit name to show all products """
        orbitids = set()
        for k in self.sfiles.product_id_index.keys():
            orbitid = k[2:9]
            orbitids.add(orbitid)

        for orbitid in sorted(orbitids)[0:20]:
            x = list(self.sfiles.orbitid_to_productids(orbitid))
            assert len(x) >= 1, "No products for orbit %s" % orbitid


    def test_find_issue28_products(self):
        """ Make sure we can find tracks from issue 28 """
        ISSUE28_PRODUCTIDS = """\
        e_1592701_001_ss19_700_a
        e_1590201_001_ss11_700_a
        e_1588801_001_ss11_700_a
        e_1593001_001_ss11_700_a
        e_1596001_001_ss19_700_a
        e_1590601_001_ss19_700_a
        e_1589701_001_ss19_700_a
        e_1592001_001_ss11_700_a
        e_1590301_001_ss11_700_a
        e_1599501_001_ss19_700_a
        e_1592201_001_ss19_700_a
        e_1596701_001_ss19_700_a
        e_1598901_001_ss19_700_a
        e_1591701_001_ss11_700_a
        e_1598301_001_ss19_700_a
        e_1594301_001_ss19_700_a
        e_1596301_001_ss11_700_a
        e_1588601_001_ss11_700_a
        e_1597601_001_ss11_700_a
        e_1597601_003_ss11_700_a
        e_1597601_002_ss19_700_a
        e_1592901_001_ss11_700_a
        e_1593101_001_ss19_700_a
        e_1588802_001_ss19_700_a
        e_1588701_001_ss19_700_a
        e_1596501_001_ss19_700_a
        e_1593002_001_ss19_700_a
        e_1591601_001_ss11_700_a
        e_1597101_001_ss19_700_a
        e_1590801_001_ss19_700_a
        e_1589101_001_ss19_700_a
        e_1598201_001_ss11_700_a""".split()


        logging.debug("find_issue28_products")
        for product_id in ISSUE28_PRODUCTIDS:
            pp = self.sfiles.edr_product_paths(product_id)
            for k, file in pp.items():
                logging.debug("%s: %s", k, file)
                p_file = Path(file)
                if p_file.exists():
                    continue

                # Check for capitalized label name
                p_file2 = p_file.with_name(p_file.name.upper())
                if p_file2.exists():
                    continue
                self.assertTrue(False, file + " does not exist")

            allfiles = self.sfiles.data_product_paths(product_id)
            f = self.sfiles.product_paths('rng', product_id)

if __name__ == "__main__":
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    unittest.main()
