#!/usr/bin/env python3

import sys
import os

sys.path.insert(1, '../SHARAD')
import SHARADEnv
import logging


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sf = SHARADEnv.SHARADFiles(data_path='/disk/kea/SDS/targ/xtra/SHARAD',
                     orig_path='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD')

    sf.read_edr_index()

    test_edr_path(sf)


def test_edr_path(sf):
    for product_id in sf.product_id_index.keys():
        pp = sf.edr_data_product_paths(product_id)
        for k, file in pp.items():
            if os.path.exists(file):
                continue

            # Check for capitalized label name
            bn = os.path.basename(file).upper()
            file2 = os.path.join(os.path.dirname(file), bn)
            if os.path.exists(file2):
                continue
            logging.warning("%s does not exist", file)

        allfiles = sf.data_product_paths(product_id)



if __name__ == "__main__":
    main()
