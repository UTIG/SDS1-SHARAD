#!/usr/bin/env python3

"""
Create a report of what data products are up to date
"""

import os
import json
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from collections import Counter

from SHARADEnv import SHARADEnv, SHARADFiles
from run_rng_cmp import add_standard_args, process_product_args

def main():
    desc = "Create a report of what data products are up to date"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--products', default='cmp,alt,srf,rsr,foc',
                        help="comma-separated list of product types to check"
                             "(default=cmp,alt,srf,rsr,foc)")
    parser.add_argument('--jsonout', default=None,#'product_status.json',
                        help='Path for JSON output file')
    parser.add_argument('--textout', default=None,#'product_status.txt',
                        help='Path for text report output file')

    add_standard_args(parser)

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="show_product_status: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')

    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)

    productlist = process_product_args(args.product_ids, args.tracklist, sfiles)

    if not productlist: # Do all products
        productlist = sfiles.product_id_index.keys()

    types = args.products.split(',')

    status_index = {}
    for ii, product_id in enumerate(productlist, start=1):
        type_status = {}
        for type in types:
            statustup, _, _ = sfiles.product_processing_status(type, product_id)
            type_status[type] = 'OK' if statustup[1] == 'output_ok' else '--'

        status_index[product_id] = type_status
        if args.maxtracks > 0 and ii >= args.maxtracks:
            break

    if args.jsonout:
        Path(args.jsonout).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonout, 'wt') as fout:
            json.dump(status_index, fout, indent="\t")
        logging.info("Wrote to %s", args.jsonout)

    if args.textout:
        Path(args.textout).parent.mkdir(parents=True, exist_ok=True)
        with open(args.textout, 'wt') as fout:
            write_text(status_index, types, fout)
        logging.info("Wrote to %s", args.textout)
    else:
        write_text_table(status_index, types, sys.stdout)

def write_text_table(status_index: Dict[str, Dict[str,str]], types: List[str], file):
    """ Write detailed report of product status to specified file handle """
    delim = ' '
    width=7
    headerline = '#{:<24s}'.format(' product_id') + delim + fixed_headers(types, delimiter=delim)
    print(headerline, file=file)

    typecounts = {k: Counter() for k in types}

    for product_id, type_status in status_index.items():
        slist = [type_status[type] + ' ' for type in types]
        line = '{:<24s} '.format(product_id) + delim + fixed_headers(slist, width=width, delimiter=delim)
        print(line, file=file)

        # Tabulate completion by product type
        for type in types:
            typecounts[type][type_status[type]] += 1

    # At the bottom, print totals
    print(headerline, file=file)
    print("# Summary Totals")
    fmtstr = '{:%dd}' % (width,)
    for status_str in ('OK', '--'):
        field1 = '#{:>24s}'.format(status_str + ' ') + delim
        slist = [fmtstr.format(typecounts[type][status_str]) for type in types]
        line = field1 + delim.join(slist)
        print(line, file=file)


def fixed_headers(fields: List[str], width:int=4, fillchar=' ', delimiter=' '):
    fields_out = [s.center(7, fillchar) for s in fields]
    return delimiter.join(fields_out)


if __name__ == "__main__":
    main()
