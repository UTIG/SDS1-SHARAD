#!/usr/bin/env python3

__authors__ = ['Scott Kempf, scottk@ig.utexas.edu']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'October 9 2019',
         'author': 'Scott Kempf, UTIG',
         'info': 'First release.'}}

import sys
import os
import time
import logging
import argparse
import warnings
import multiprocessing
import traceback
import importlib.util
import numpy as np
#from scipy.optimize import curve_fit
import pandas as pd

# TODO: make this import more robust to allow this script
# TODO: to be run from outside the SHARAD directory
sys.path.insert(0, '../xlib')
#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



def cmp_processor(infile, outdir, idx_start=None, idx_end=None, taskname="TaskXXX", radargram=True,
                  chrp_filt=True, verbose=False, saving=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual tracks and returns pulse compressed data.

    Input:
    -----------
      infile    : Path to track file to be processed.
      outdir    : Path to directory to write output data
      idx_start : Start index for processing.
      idx_end   : End index for processing.
      chrp_filt : Apply a filter to the reference chirp
      verbose   : Gives feedback in the terminal.

    Output:
    -----------
      E          : Optimal E value
      cmp_pulses : Compressed pulses

    """

    try:
    #if chrp_filt:
        time_start = time.time()
        # Get science data structure
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        # Load data
        science_path = infile.replace('_a.dat', '_s.dat')
        data = pds3.read_science(science_path, label_path, science=True)
        aux  = pds3.read_science(infile      , aux_path,   science=False)

        stamp1 = time.time()-time_start
        logging.debug("{:s}: data loaded in {:0.1f} seconds".format(taskname, stamp1))

        # Array of indices to be processed
        if idx_start is None or idx_end is None:
            idx_start = 0
            idx_end = len(data)
        idx = np.arange(idx_start, idx_end)

        logging.debug('{:s}: Length of track: {:d}'.format(taskname, len(idx)))

        # Chop raw data
        raw_data = np.zeros((len(idx), 3600), dtype=np.complex)
        for j in range(0, 3600):
            raw_data[idx-idx_start, j] = data['sample'+str(j)][idx].values

        if data['COMPRESSION_SELECTION'][idx_start] == 0:
            compression = 'static'
        else:
            compression = 'dynamic'
        tps = data['TRACKING_PRE_SUMMING'][idx_start]
        presum_table = (1, 2, 3, 4, 8, 16, 32, 64)
        assert 0 <= tps <= 7
        presum = presum_table[tps]
        sdi = data['SDI_BIT_FIELD'][idx_start]
        bps = 8

        # Decompress the data
        decompressed = cmp.rng_cmp.decompress_sci_data(raw_data, compression, presum, bps, sdi)
        E_track = np.empty((idx_end-idx_start, 2))
        # Get groundtrack distance and define 30 km chunks
        tlp = np.array(data['TLP_INTERPOLATE'][idx_start:idx_end])
        tlp0 = tlp[0]
        chunks = []
        i0 = 0
        for i in range(len(tlp)):
            if tlp[i] > tlp0+30:
                chunks.append([i0, i])
                i0 = i
                tlp0 = tlp[i]
        if not chunks: # if chunks is empty
            chunks.append([0, idx_end-idx_start])
        if (tlp[-1]-tlp[i0]) >= 15:
            chunks.append([i0, idx_end-idx_start])
        else:
            chunks[-1][1] = idx_end-idx_start
        #chunks = np.array(chunks)

        logging.debug('{:s}: chunked into {:d} pieces'.format(taskname, len(chunks)))
        # Compress the data chunkwise and reconstruct
        for i, chunk in enumerate(chunks):
            start, end = chunk

            #check if ionospheric correction is needed
            iono_check = np.where(aux['SOLAR_ZENITH_ANGLE'][start:end]<100)[0]
            b_iono = len(iono_check) != 0
            minsza = min(aux['SOLAR_ZENITH_ANGLE'][start:end])
            logging.debug('{:s}: chunk {:03d}/{:03d} Minimum SZA: {:0.3f}  Ionospheric Correction: {!r}'.format(
                taskname, i, len(chunks), minsza, b_iono))

            # GNG: These concats seem relatively expensive.
            E, sigma, cmp_data = cmp.rng_cmp.us_rng_cmp(decompressed[start:end],\
                                 chirp_filter=chrp_filt, iono=b_iono, debug=verbose)
            if i == 0:
                cmp_track = cmp_data
            else:
                cmp_track = np.vstack((cmp_track, cmp_data))
            E_track[start:end, 0] = E
            E_track[start:end, 1] = sigma

        stamp3 = time.time() - time_start - stamp1
        logging.debug('{:s} Data compressed in {:0.2f} seconds'.format(taskname, stamp3))

        if saving:
            data_file   = os.path.basename(infile)
            outfilebase = data_file.replace('_a.dat', '_s.h5')
            outfile     = os.path.join(outdir, outfilebase)
            logging.debug('{:s}: Saving to folder: {:s}'.format(taskname,outdir))
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # restructure of data save
            real = np.array(np.round(cmp_track.real), dtype=np.int16)
            imag = np.array(np.round(cmp_track.imag), dtype=np.int16)
            dfreal = pd.DataFrame(real)
            dfimag = pd.DataFrame(imag)
            dfreal.to_hdf(outfile, key='real', complib='blosc:lz4', complevel=6)
            dfimag.to_hdf(outfile, key='imag', complib='blosc:lz4', complevel=6)
            #np.save(new_path+data_file.replace('.dat','.npy'),cmp_track)
            outfile_TECU = os.path.join(outdir, data_file.replace('_a.dat','_s_TECU.txt'))
            np.savetxt(outfile_TECU, E_track)

        if radargram:
            # Plot a radargram
            rx_window_start = data['RECEIVE_WINDOW_OPENING_TIME'][idx]
            tx0 = data['RECEIVE_WINDOW_OPENING_TIME'][0]
            tx = np.empty(len(data))
            # GNG: this seems likely to be wrong. Should be:
            #for rec in range(len(data['RECEIVE_WINDOW_OPENING_TIME'])):
            for rec in range(len(data)):
                tx[rec] = data['RECEIVE_WINDOW_OPENING_TIME'][rec]-tx0
            cmp.plotting.plot_radargram(cmp_track, tx, samples=3600)

    except Exception: # pylint: disable=W0703
        logging.error('{:s}: Error processing file {:s}'.format(taskname, infile))
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line))
        return 1
    logging.info('{:s}: Success processing file {:s}'.format(taskname, infile))
    return 0

Processors = [
    [("Name","Run Range Compression"),
     ("Input","_a.dat"),
     ("Input","_s.dat"),
     ("Processor", "run_rng_cmp.py"), ("Library", "xlib/cmp/pds3lbl.py"),
     ("Library", "xlib/cmp/plotting.py"), ("Library", "xlib/cmp/rng_cmp.py"),
     ("Outdir", "ion"),
     ("Output", "_s.h5"),
     ("Output", "_s_TECU.txt")
    ],
    [("Name","Run Altimetry"),
     ("Indir", "ion"),
     ("Input","_a.dat"),
     ("Input","_s.dat"),
     ("Processor", "run_rng_cmp.py"), ("Library", "xlib/cmp/pds3lbl.py"),
     ("Library", "xlib/cmp/plotting.py"), ("Library", "xlib/cmp/rng_cmp.py"),
     ("Outdir", "ion"),
     ("Output", "_s.h5"),
     ("Output", "_s_TECU.txt")
    ]
]

def getmtime(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = -1
    return (mtime,path)


def main():
    parser = argparse.ArgumentParser(description='SHARAD Pipeline')
    parser.add_argument('-o', '--output', default='/disk/kea/SDS/targ/xtra/SHARAD',
                        help="Output base directory")

    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="elysium.txt",
                        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0,
                        help="Maximum number of tracks to process")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="pipeline: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    with open(args.tracklist, 'r') as fin:
        lookup = [line.strip() for line in fin if line.strip() != '']
    logging.debug(lookup)

    # Build list of processes
    logging.info("Building task list")
    process_list = []
    path_outroot = args.output

    logging.debug("Base output directory: " + path_outroot)

    SHARADroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    for prod in Processors:
        for i, infile in enumerate(lookup):
            path_file = infile.replace(SHARADroot, '')
            data_file = os.path.basename(path_file).replace('_a.dat', '')
            path_file = os.path.dirname(path_file)
            root_file,ext_file = os.path.splitext(data_file)
            intimes = []
            outtimes = []
            for attr in prod:
                if (attr[0] == "Name"):
                    logging.info("Considering: " + attr[1])
                if (attr[0] == "Indir"):
                    indir = os.path.join(path_outroot, path_file, attr[1])
                if (attr[0] == "Input"):
                    # FIXME: This might be better if absolute paths are detected
                    if (attr[1] == '_a.dat' or attr[1] == '_s.dat'):
                         infile = os.path.join(SHARADroot,path_file,data_file+attr[1])
                    else:
                         infile = os.path.join(indir, data_file+attr[1])
                    intimes.append(getmtime(infile))
                if (attr[0] == "Processor"):
                    proc = attr[1]
                    intimes.append(getmtime(attr[1]))
                if (attr[0] == "Library"):
                    libfile = os.path.join('../',attr[1])
                    intimes.append(getmtime(libfile))
                if (attr[0] == "Outdir"):
                    outdir = os.path.join(path_outroot, path_file, attr[1])
                if (attr[0] == "Output"):
                    output = os.path.join(outdir, data_file+attr[1])
                    outtimes.append(getmtime((output)))
            intimes.sort(key = lambda x: x[0])
            outtimes.sort(key = lambda x: x[0])
            if (intimes[0][0] == -1):
                print('Input missing.')
            elif (intimes[-1][0] < outtimes[0][0]):
                print('Up to date.')
            else:
                print('Ready to process.')
                logging.debug("Adding " + infile)
                process_list.append(infile)
        else:
            logging.debug('File already processed. Skipping ' + infile)

    sys.exit(0)
    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)))

    start_time = time.time()

    named_params = {'saving':True, 'chrp_filt':True, 'verbose':args.verbose, 'radargram':False}

    if nb_cores <= 1:
        # Single processing (for profiling)
        for t in process_list:
            cmp_processor(*t, **named_params)
    else:
        # Multiprocessing
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(cmp_processor, t, \
                   named_params) for t in process_list]

        for i, result in enumerate(results):
            dummy = result.get()
            if dummy == 1:
                logging.error("{:s}: Problem running pulse compression".format(process_list[i][4]))
            else:
                logging.info("{:s}: Finished pulse compression".format(process_list[i][4]))

    logging.info("Done in {:0.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    # execute only if run as a script
    main()
