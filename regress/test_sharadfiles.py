#!/usr/bin/env python3

import sys
import os
import logging

p1 = os.path.join(os.path.dirname(__file__), '../SHARAD')
sys.path.insert(1, os.path.abspath(p1))
import SHARADEnv


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sds = os.getenv('SDS', '/disk/kea/SDS')
    data_path = os.path.join(sds, 'targ/xtra/SHARAD')
    orig_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADEnv.SHARADFiles(data_path=data_path, orig_path=orig_path)
    sfiles.read_edr_index()
    test_edr_path(sfiles)

    find_issue28_products(sfiles)
    find_products_by_orbit(sfiles)

def test_edr_path(sfiles):
    for product_id in sfiles.product_id_index.keys():
        pp = sfiles.edr_product_paths(product_id)
        for file in pp.values():
            if os.path.exists(file):
                continue

            # Check for capitalized label name
            bname = os.path.basename(file).upper()
            file2 = os.path.join(os.path.dirname(file), bname)
            if os.path.exists(file2):
                continue
            logging.warning("%s does not exist", file)

        allfiles = sfiles.data_product_paths(product_id)

def find_products_by_orbit(sfiles):
    """ Go from a orbit name to show all products """
    orbitids = set()
    for k in sfiles.product_id_index.keys():
        orbitid = k[2:9]
        orbitids.add(orbitid)

    for orbitid in sorted(orbitids)[0:20]:
        x = list(sfiles.orbitid_to_productids(orbitid))
        assert len(x) >= 1, "No products for orbit %s" % orbitid


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


def find_issue28_products(sfiles):
    """ Make sure we can find tracks from issue 28 """
    logging.info("find_issue28_products")
    for product_id in ISSUE28_PRODUCTIDS:
        pp = sfiles.edr_product_paths(product_id)
        for k, file in pp.items():
            logging.debug("%s: %s", k, file)
            if os.path.exists(file):
                continue

            # Check for capitalized label name
            bn = os.path.basename(file).upper()
            file2 = os.path.join(os.path.dirname(file), bn)
            if os.path.exists(file2):
                continue
            logging.warning("%s does not exist", file)

        allfiles = sfiles.data_product_paths(product_id)
        f = sfiles.product_paths('rng', product_id)


if __name__ == "__main__":
    main()
