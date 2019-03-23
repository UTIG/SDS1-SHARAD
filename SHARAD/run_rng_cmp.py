#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu',
               'Kirk Scanlan, kirk.scanlan@gmail.com']
__version__ = '1.1'
__history__ = {
    '1.0':
        {'date': 'August 15 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '1.1':
        {'date': 'August 23 2018',
         'info': 'Added pulse decompression and 6bit data'},
    '1.2':
        {'date': 'October 16 2018',
         'info': 'Modified data saving method from .npy to .h5'}} 

import sys
import os
import time
import logging
import argparse
import warnings
import multiprocessing
import traceback

import numpy as np
from scipy.optimize import curve_fit
import importlib.util
import pandas as pd

# TODO: make this import more robust to allow this script
# TODO: to be run from outside the SHARAD directory
sys.path.insert(0, '../xlib')
#import misc.hdf
import cmp.pds3lbl as pds3
import cmp.plotting
import cmp.rng_cmp

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def cmp_processor(infile, outdir, idx_start=None, idx_end=None, taskname="TaskXXX", radargram=True,
                  chrp_filt=True, verbose=False, saving='hdf5'):
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
        science_path=infile.replace('_a.dat','_s.dat')
        data = pds3.read_science(science_path, label_path, science=True)
        aux  = pds3.read_science(infile      , aux_path,   science=False)

        stamp1=time.time()-time_start
        logging.debug("{:s}: data loaded in {:0.1f} seconds".format(taskname, stamp1))

        # Array of indices to be processed
        if idx_start is None or idx_end is None:
            idx_start = 0
            idx_end = len(data)
        idx = np.arange(idx_start,idx_end)

        logging.debug('{:s}: Length of track: {:d}'.format(taskname, len(idx)) )

        # Chop raw data
        raw_data = chop_raw_data(data, idx, idx_start)

        if data['COMPRESSION_SELECTION'][idx_start] == 0: compression = 'static'
        else: compression = 'dynamic'
        tps = data['TRACKING_PRE_SUMMING'][idx_start]
        assert(tps >= 0 and tps <= 7)
        if tps == 0: presum = 1
        elif tps == 1: presum = 2
        elif tps == 2: presum = 3
        elif tps == 3: presum = 4
        elif tps == 4: presum = 8
        elif tps == 5: presum = 16
        elif tps == 6: presum = 32
        elif tps == 7: presum = 64
        SDI = data['SDI_BIT_FIELD'][idx_start]
        bps = 8    

        # Decompress the data
        decompressed = cmp.rng_cmp.decompressSciData(raw_data, compression, presum, bps, SDI) 
        # TODO: E_track can just be a list of tuples
        E_track = np.empty((idx_end-idx_start,2))
        # Get groundtrack distance and define 30 km chunks
        tlp = np.array(data['TLP_INTERPOLATE'][idx_start:idx_end])
        tlp0 = tlp[0]
        chunks=[]
        i0=0
        for i in range(len(tlp)):
            if tlp[i]>tlp0+30: 
                 chunks.append([i0,i]) 
                 i0=i
                 tlp0=tlp[i]
        if len(chunks)==0: chunks.append([0,idx_end-idx_start])
        if (tlp[-1]-tlp[i0])>=15: chunks.append([i0, idx_end-idx_start])
        else: chunks[-1][1]=idx_end-idx_start

        logging.debug('{:s}: chunked into {:d} pieces'.format(taskname, len(chunks)) )
        # Compress the data chunkwise and reconstruct


        list_cmp_track=[]
        for i, chunk in enumerate(chunks):
            start,end = chunks[i]

            #check if ionospheric correction is needed
            iono_check = np.where(aux['SOLAR_ZENITH_ANGLE'][start:end]<100)[0]
            b_iono = len(iono_check) != 0
            minsza = min(aux['SOLAR_ZENITH_ANGLE'][start:end])
            logging.debug('{:s}: chunk {:03d}/{:03d} Minimum SZA: {:6.2f}  Ionospheric Correction: {!r}'.format(
                taskname, i, len(chunks), minsza, b_iono) )

            E, sigma, cmp_data = cmp.rng_cmp.us_rng_cmp(
                decompressed[start:end], chirp_filter=chrp_filt, iono=b_iono, debug=verbose)
            list_cmp_track.append(cmp_data)
            E_track[start:end,0] = E
            E_track[start:end,1] = sigma        

        cmp_track = np.vstack(list_cmp_track)
        list_cmp_track = None  # free memory


        stamp3=time.time()-time_start-stamp1
        logging.debug('{:s} Data compressed in {:0.2f} seconds'.format(taskname, stamp3))

        if saving and outdir != "":
            #path_outroot = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
            #path_file = science_path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
            #data_file = path_file.split('/')[-1]
            #path_file = path_file.replace(data_file,'')
            #new_path = path_outroot+path_file+'ion/'
            data_file   = os.path.basename(infile)
            outfilebase = data_file.replace('.dat','.h5')
            outfile     = os.path.join(outdir, outfilebase)

            logging.debug('{:s}: Saving to folder: {:s}'.format(taskname,outdir) )
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            # restructure of data save
            real = np.array(np.round(cmp_track.real), dtype=np.int16)
            imag = np.array(np.round(cmp_track.imag), dtype=np.int16)
            cmp_track = None # free memory
            if saving == 'hdf5':
                #dfreal = pd.DataFrame(real)
                #dfimag = pd.DataFrame(imag)
                pd.DataFrame(real).to_hdf(outfile, key='real', complib = 'blosc:lz4', complevel=6)
                pd.DataFrame(imag).to_hdf(outfile, key='imag', complib = 'blosc:lz4', complevel=6)
            elif saving == 'npy':
                # Round it just like in an hdf5 and save as side-by-side arrays
                cmp_track = np.vstack([real, imag])

                outfile = os.path.join(outdir, data_file.replace('.dat','.npy') )
                np.save(outfile,cmp_track)
            elif saving == 'none':
                pass
            else:
                logging.error("{:s}: Unrecognized output format '{:s}'".format(taskname, saving))

            outfile_TECU = os.path.join(outdir, data_file.replace('.dat','_TECU.txt') )
            np.savetxt(outfile_TECU,E_track)


        if radargram:
            # Plot a radargram
            rx_window_start = data['RECEIVE_WINDOW_OPENING_TIME'][idx]
            tx0=data['RECEIVE_WINDOW_OPENING_TIME'][0]
            tx=np.empty(len(data))
            # GNG: this seems likely to be wrong. Should be:
            #for rec in range(len(data['RECEIVE_WINDOW_OPENING_TIME'])):
            for rec in range(len(data)):
                tx[rec]=data['RECEIVE_WINDOW_OPENING_TIME'][rec]-tx0
            cmp.plotting.plot_radargram(cmp_track,tx,samples=3600)

    except Exception as e:

        logging.error('{:s}: Error processing file {:s}'.format(taskname, infile))
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line) )
        return 1
    logging.info('{:s}: Success processing file {:s}'.format(taskname, infile))
    return 0

def chop_raw_data(data, idx, idx_start):
    raw_data=np.zeros((len(idx),3600),dtype=np.complex)
    for j in range(3600):
        k = 'sample' + str(j)
        #logging.debug("raw_data[{:s},{:d}]=data[{:s}][{:s}].values".format(str(idx-idx_start), j, k,str(idx)) )
        #logging.debug("raw_data[{:s},{:d}]=data[{:s}][{:s}].values".format(":", j, k,str(idx)) )
        #raw_data[idx-idx_start,j]=data[k][idx].values
        raw_data[:,j]=data[k][idx].values
    return raw_data




def main():
    # TODO: improve description
    parser = argparse.ArgumentParser(description='Run SAR processing')
    parser.add_argument('-o','--output', default='/disk/kea/SDS/targ/xtra/SHARAD/cmp',
                        help="Output base directory")
    parser.add_argument('--ofmt', default='hdf5', choices=('hdf5','npy','none'),
                        help="Output file format")

    parser.add_argument('-j','--jobs', type=int, default=4, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="elysium.txt",
        help="List of tracks to process")
    parser.add_argument('--maxtracks', type=int, default=0, help="Maximum number of tracks to process")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_rng_cmp: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys() 
    #lookup = np.genfromtxt('lookup.txt',dtype='str')
    lookup = np.genfromtxt(args.tracklist, dtype = 'str')
    #lookup = np.genfromtxt('EDR_Cyril_SouthPole_Path.txt', dtype = 'str')

    # Build list of processes
    logging.info("Building task list")
    process_list=[]
    path_outroot = args.output

    logging.debug("Base output directory: " + path_outroot)
    for i,infile in enumerate(lookup):
    #for orbit in keys:
        #gob = int(orbit.replace('/orbit', ''))    
        #path = lookup[gob]
        #idx_start = h5file[orbit]['idx_start'][0]
        #idx_end = h5file[orbit]['idx_end'][0]

        # check if file has already been processed
        path_file = infile.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = os.path.basename(path_file)
        path_file = os.path.dirname(path_file)
        outdir    = os.path.join(path_outroot,path_file, 'ion')

        if not os.path.exists(outdir):
            logging.debug("Adding " + infile)
            process_list.append([infile, outdir, None, None, "Task{:03d}".format(i+1)])
        else:
            logging.debug('File already processed. Skipping ' + infile)

    #h5file.close()
    if args.maxtracks > 0:
        process_list = process_list[0:args.maxtracks]
    #process_list.append(['/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr01xxx/edr0188801/e_0188801_001_ss05_700_a_a.dat',None,None])
    if args.dryrun:
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)))

    start_time = time.time()

    named_params = {'saving':args.ofmt,'chrp_filt':True,'verbose':args.verbose,'radargram':False}

    if nb_cores <= 1:
        # Single processing (for profiling)
        for t in process_list:
            cmp_processor(*t, **named_params)
    else:
        # Multiprocessing
        pool = multiprocessing.Pool(nb_cores)
        results = [pool.apply_async(cmp_processor, t,
                named_params) for t in process_list]

        for i,result in enumerate(results):
            dummy = result.get()
            if dummy == 1: 
                logging.error("{:s}: Problem running pulse compression".format(process_list[i][4]))
            else:
                logging.info( "{:s}: Finished pulse compression".format(process_list[i][4] ))

    logging.info("Done in {:0.2f} seconds".format( time.time() - start_time ) )


if __name__ == "__main__":
    # execute only if run as a script
    main()

