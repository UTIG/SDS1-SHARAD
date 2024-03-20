#!/usr/bin/env python3
# Check whether the SHARAD PDS archive is synchronized or not



"""
Download a copy of the cumulative SHARAD index, then compare that
with the files in a local mirror on the hierarchy
"""

import os
import sys
import csv
import glob
from pathlib import Path
from dataclasses import dataclass
import logging
from collections import defaultdict

import pvl
import requests

# import pds3lbl


class SHARADIndex:
    """ EDR Index manager """
    def __init__(self):
        pass

    def read_edr_index(self, indexfile: str):
        """ Build index of product IDs to locations.
        We can read any one of cumindex.tab """
        self.product_id_index = {}
        fields = 'volume_id release_id file_specification_name product_id'.split()

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
        logging.debug("product_id_index has %d items", len(self.product_id_index))

    def load_file_index(self, product_id: str):
        """ Populate the file index with the list of all known files
        and expected file sizes/dates
        product ID including science file sizes and additional files"""
        self.file_index = {}

    def enumerate_label_status(self, local_root: str):
        """ Enumerate the status of whether the label exists locally """

    def enumerate_orphan_files(self, local_root: str):
        """ List all files that are not referred to by any label """

    def download_label_(self, product_id: str, local_root: str,
                        remote_root: str, overwrite:bool=False, session=None):
        """ Download the label for the given product ID into correct spot
        relative to the local root
        https://requests.readthedocs.io/en/latest/user/quickstart/#raw-response-content
        """
        info = self.product_id_index[product_id]
        relfile = info['file_specification_name']
        vid = info['volume_id']
        localpath = os.path.join(local_root, vid, relfile)
        remotepath = os.path.join(remote_root, vid, relfile)

        # TODO: check sizes and times against headers
        if (not overwrite) and os.path.exists(localpath):
            return False

        logging.info("GET %s", remotepath)
        if session is None:
            session = requests
        r = session.get(remotepath, allow_redirects=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(localpath), exist_ok=True)
        with open(localpath, 'wb') as fh:
            fh.write(r.content)
        return True

    def download_product_labels(self, local_root: str, remote_url: str, overwrite:bool=False):
        pass
        # Start a requests session
        s = requests.Session()
        nitems = len(self.product_id_index)
        for ii, product_id in enumerate(self.product_id_index.keys(), start=1):
            logging.info("[%7d of %d] %s", ii, nitems, product_id)
            self.download_label_(product_id, local_root, remote_url, overwrite, s)



@dataclass
class SHARADPDS3Mirror:
    """ Manage files in the SHARAD PDS3 mirror """
    local_dir: str
    remote_url: str

    def download_indexes(self):
        """ Download all of the cumulative indexes and indexes
        """

    def load_index(self):
        """ Load the data in the index """
        self.sidx = SHARADIndex()

    def download_labels(self):
        """ Download all labels listed in the index
        to their respective spots """

    def query_files_headers(self):
        """ Get HTTP headers to calculate expected sizes of
        all files in the system """



def read_edr_label(labelfile: str):
    """\
    read the label file and extract key information about the
    science telemetry table and the auxiliary table
    and return the expected size and location as a FileInfo object (size and path)

    TODO: move this to pds3lbl?
    """
    label = pvl.load(labelfile)
    dir1 = os.path.dirname(labelfile)

def build_filelist(rootpath: str):
    """ Build an hashed index of all filenames so we can find
    files that exist """
    logging.info("build_filelist(%s)", rootpath)
    fileindex = defaultdict(list)
    for root, dirs, files in os.walk(rootpath):
        dirs.sort()
        files.sort()
        for f in files:
            p = os.path.join(root, f)
            fileindex[f].append({
                'path': p,
                'size': os.path.getsize(p),
            })
    logging.info("build_filelist done")
    return fileindex
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    workdir = './work'
    remote_path = 'https://pds-geosciences.wustl.edu/mro/mro-m-sharad-3-edr-v1'

    local_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'

    idxfile = os.path.join(workdir, 'mrosh_0001/index/cumindex.tab')

    sidx = SHARADIndex()
    sidx.read_edr_index(idxfile)
    #sidx.download_product_labels(workdir, remote_path, overwrite=False)
    filelist = build_filelist(local_path)
    logging.info("Found %d files in %s", len(filelist), local_path)
    nmissing = 0
    nmisplaced = 0
    for k, row in sidx.product_id_index.items():
        f = row['file_specification_name']
        frel = os.path.join(row['volume_id'], row['file_specification_name'])
        #d = os.path.join(row['volume_id'], os.path.dirname(row['file_specification_name']))

        sci = f.replace('.lbl', '_s.dat')
        aux = f.replace('.lbl', '_a.dat')

        for f1 in (f, sci, aux):
            fname = os.path.basename(f1)
            if fname in filelist:
                # Check that it's in the right place
                relpaths = [os.path.relpath(info['path'], local_path) for info in filelist[fname]]
                assert len(relpaths) > 0, "relpaths=%r" % (relpaths,)
                f2 = os.path.join(row['volume_id'], f1)
                if f2 not in relpaths:
                    foundf = ', '.join(relpaths)
                    logging.info("%s should be at %s.\n  Found at %s", fname, f2, foundf)
                    nmisplaced += 1
            else:
                logging.info("not found: %s", f1)
                nmissing += 1

    logging.info("Missing files: %d of %d", nmissing, 3*len(sidx.product_id_index))
    logging.info("Misplaced files: %d of %d", nmisplaced, 3*len(sidx.product_id_index))

if __name__ == "__main__":
    main()
