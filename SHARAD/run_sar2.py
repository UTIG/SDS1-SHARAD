#!/usr/bin/env python3

__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 13 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '2.0':
        {'date': 'October 23 2018',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Second release.'},
     '3.0':
        {'date': 'April 30 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Third release.'}}


"""

Example usage:
./run_sar2.py 


You can tee to a log file using this command:
stdbuf -o 0 ./run_sar2.py -v | tee sar_crash.log


"""


import traceback
import sys
import os
import time
import logging
import argparse
import warnings
import importlib.util
import multiprocessing

import numpy as np
import pandas as pd

sys.path.insert(0, '../xlib/')
from sar import sar
import cmp.pds3lbl as pds3
import misc.hdf as hdf

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#import matplotlib
## Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


# Code mapping PRI codes to actual pulse repetition intervals
pri_table={
1: 1428E-6,
2: 1429E-6,
3: 1290E-6,
4: 2856E-6,
5: 2984E-6,
6: 2580E-6
}




def sar_processor(taskinfo, procparam, focuser='Delay Doppler v2',
                  saving="hdf5", debug=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual range compressed and ionosphere corrected tracks and
    performs SAR focusing using the desired algorithm.
    Note: delay Doppler is much faster than matched filter and is set to be the
          default option

    Input:
    -----------
      taskinfo  : dict of inputs required for SAR processing, with the following keys
                  - name: (required) name for this task, unique among all tasks being processed
                  - input: (required) path to EDR
                  - output: (required) path to output file (set to None to omit saving)
                  - idx_start: (optional)
                  - idx_end: (optional)
      focuser   : Flag for which SAR focuser to use.
     procparam  : dict of processing parameters required by the different SAR
                  processors, with the following keys
                  *** delay_doppler_v1 ***
                  - ddv1_posting_distance: distance [m] along track at which to place SAR columns
                  - ddv1_aperture_time   : length [s] of the synthetic aperture
                  - ddv1_bandwidth       : Doppler badnwidth [Hz] for multilooking
                  *** matched_filter ***
                  - mf_posting_distance  : distance [m] along track at which to place SAR columns
                  - mf_aperture_time     : length [s] of the synthetic aperture
                  - mf_bandwidth         : Doppler badnwidth [Hz] for multilooking
                  - mf_recalc_int        : interval [samples] at which to recalculate the matched filter
                  - mf_Er                : relative dielectric permittivity of the subsurface
                  *** delay_doppler_v2 ***
                  - ddv2_interpolate_dx  : groundtrack interval [m] at which to interpolate range lines
                  - ddv2_posting_interval: interpolated range line interval at which to place SAR columns
                  - ddv2_aperture_dist   : length [km] of the synthetic aperture
                  - ddv2_trim            : number of sample to keep in trimmed version of the processing
      saving    : Flag describing output save format
                  hdf5 - save in HDF5 format
                  npy  - save in numpy format
                  none - don't save output data
      debug     : Enter debug mode - show more info

    Output:
    -----------
      sar       : focused SAR data
      columns   : matrix of EDR range lines specifying the mid-aperture
                  position as well as the start and end of each aperture

    """

    taskname = taskinfo.get('name', "TaskXXX")

    try:
   
        idx_start = taskinfo.get('idx_start', None)
        idx_end   = taskinfo.get('idx_end', None)
        path      = taskinfo['input']
        outputfile= taskinfo['output']

        # print info in debug mode
        logging.debug("{:s}: SAR method: {:s}".format(taskname, focuser)) 
        if focuser == 'Delay Doppler v1':
            number_of_looks = no.floor(procparam['ddv1_aperture_time [s]'] * 2 * procparam['ddv1_bandwidth [Hz]'])
            logging.debug("{:s}: SAR column posting interval [m]: {:f}".format(taskname, procparam['ddv1_posting_distance [m]'])) 
            logging.debug("{:s}: SAR aperture length [s]: {:f}".format(taskname, procparam['ddv1_aperture_time [s]'])) 
            logging.debug('{:s}: SAR Doppler bandwidth [Hz]: {:f}'.format(taskname, procparam['ddv1_bandwidth [Hz]']))
            logging.debug('{:s}: SAR number of looks: {:d}'.format(taskname,  int(number_of_looks)))
            del number_of_looks
        elif focuser == 'Matched Filter':
            number_of_looks = no.floor(procparam['mf_aperture_time [s]'] * 2 * procparam['mf_bandwidth [Hz]'])
            logging.debug("{:s}: SAR column posting interval [m]: {:f}".format(taskname, procparam['mf_posting_distance [m]'])) 
            logging.debug("{:s}: SAR aperture length [s]: {:f}".format(taskname, procparam['mf_aperture_time [s]'])) 
            logging.debug('{:s}: SAR Doppler bandwidth [Hz]: {:f}'.format(taskname, procparam['mf_bandwidth [Hz]']))
            logging.debug('{:s}: SAR number of looks: {:d}'.format(taskname,  int(number_of_looks)))
            logging.debug('{:s}: SAR matched filter recalc interval: {:d}'.format(taskname,  procparam['mf_recalc_int [samples]']))
            logging.debug('{:s}: SAR subsurface permittivity: {:d}'.format(taskname,  procparam['mf_Er']))
            del number_of_looks
        elif focuser == 'Delay Doppler v2':
            logging.debug("{:s}: SAR range line interpolation interval [m]: {:f}".format(taskname, procparam['ddv2_interpolate_dx [m]'])) 
            logging.debug("{:s}: SAR column posting interval [range lines]: {:f}".format(taskname, procparam['ddv2_posting_interval [range lines]'])) 
            logging.debug('{:s}: SAR aperture distance [km]: {:f}'.format(taskname, procparam['ddv2_aperture_dist [km]']))
            if len(procparam['ddv2_trim [samples]']) != 0:
                logging.debug('{:s}: SAR fast-time trim [samples]: {:f}'.format(taskname, procparam['ddv2_trim [samples]']))

        # create cmp path
        path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        inputroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
        path_file = os.path.relpath(path, inputroot)
        data_file = os.path.basename(path_file)
        path_file = os.path.dirname(path_file)
        cmp_path = os.path.join(path_root, path_file, 'ion', data_file.replace('_a.dat','_s.h5'))
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path   = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        science_path=path.replace('_a.dat','_s.dat')

        logging.debug("{:s}: Loading cmp data from {:s}".format(taskname, cmp_path))
    
        # load the range compressed and ionosphere corrected data
        real = np.array(pd.read_hdf(cmp_path, 'real'))	
        imag = np.array(pd.read_hdf(cmp_path, 'imag'))
        cmp_track = real + 1j * imag

        idx_start = 0            if idx_start is None else max(0, idx_start)
        idx_end = len(cmp_track) if idx_end   is None else min(len(cmp_track), idx_end)
        cmp_track = cmp_track[idx_start:idx_end]

        logging.debug("{:s}: Processing track from start={:d} end={:d} (length={:d})".format(
            taskname, idx_start, idx_end, len(cmp_track)) )
        
        # load the relevant EDR files
        logging.debug("{:s}: Loading science data from EDR file: {:s}".format(taskname, science_path) )
        data = pds3.read_science(science_path, label_path, science=True, 
                                 bc=True)[idx_start:idx_end]

        auxfile=science_path.replace('_s.dat', '_a.dat')
        logging.debug("{:s}: Loading auxiliary data from EDR file: {:s}".format(taskname, auxfile))
        aux = pds3.read_science(auxfile, 
                                aux_path, science=False, bc=False)[idx_start:idx_end]

        logging.debug("{:s}: Length of selected EDR sci data: {:d}".format(taskname, len(data)) )
        logging.debug("{:s}: Length of selected EDR aux data: {:d}".format(taskname, len(aux)) )

        # load relevant spacecraft position information from EDR files
        pri_code = data['PULSE_REPETITION_INTERVAL'].values
        rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].values
        if focuser != 'Delay Doppler v2':
            for j,code  in enumerate(pri_code):
                pri = pri_table.get(code, 0.0)
                rxwot[j] *= 0.0375E-6 + pri - 11.98E-6
        et = aux['EPHEMERIS_TIME'].values
        tlp = data['TLP_INTERPOLATE'].values
        scrad = data['RADIUS_INTERPOLATE'].values
        if focuser == 'Delay Doppler v1':
            tpgpy = data['TOPOGRAPHY'].values
            v = np.hypot(data['TANGENTIAL_VELOCITY_INTERPOLATE'].values, 
                         data['RADIAL_VELOCITY_INTERPOLATE'].values)
        elif focuser == 'Delay Doppler v2':
            tpgpy = data['TOPOGRAPHY'].values
            lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE'].values
            long = aux['SUB_SC_EAST_LONGITUDE'].values
            band = np.zeros((len(data)), dtype=float)
        
        # correct the rx window opening times for along-track changes in spacecraft
        # radius
        rxwot2 = rxwot - (2 * (scrad - min(scrad)) * 1000 / 299792458)

        """
        sc = np.empty(len(ets))
        i = 0
        for et in ets:
            scpos, lt = spice.spkgeo(-74,et,'J2000',4)
            sc[i] = np.linalg.norm(scpos[0:3])
            i+=1
        """
        
        # execute sar processing
        logging.debug("{:s}: Start of SAR processing".format(taskname))
        if focuser == 'Delay Doppler v1':
            sardata, columns = sar.delay_doppler_v1(cmp_track, procparam['ddv1_posting_distance [m]'],
                                                    procparam['ddv1_aperture_time [s]'],
                                                    procparam['ddv1_bandwidth [Hz]'],
                                                    tlp, et, scrad, tpgpy, rxwot2 - min(rxwot2), v, debugtag=taskname)
        elif focuser == 'Matched Filter':
            sardata, columns = sar.matched_filter(cmp_track, procparam['mf_posting_distance [m]'],
                                                  procparam['mf_aperture_time [s]'],
                                                  procparam['mf_Er'],
                                                  procparam['mf_bandwidth [Hz]'],
                                                  procparam['mf_recalc_int [samples]'],
                                                  tlp, et, rxwot2)
        elif focuser == 'Delay Doppler v2':
            sardata, int_et, columns = sar.delay_doppler_v2(cmp_track, procparam['ddv2_interpolate_dx [m]'],
                                                            procparam['ddv2_posting_interval [range lines]'], 
                                                            procparam['ddv2_trim [samples]'],
                                                            procparam['ddv2_aperture_dist [km]'],
                                                            et, tpgpy, pri_code, scrad, rxwot, lat, long, band,
                                                            debugtag=taskname)
        
        # save the result
        if saving and outputfile is not None:
            new_path = os.path.dirname(outputfile)
            logging.debug("{:s}: Saving to file: {:s}".format(taskname, outputfile))

            if not os.path.exists(new_path):
                os.makedirs(new_path)
          
            if saving == "hdf5":
                if focuser != 'Delay Doppler v2':
                    # restructure and save data
                    dfsar = pd.DataFrame(sardata)
                    dfcol = pd.DataFrame(columns)
                    dfsar.to_hdf(outputfile, key='sar',
                                 complib = 'blosc:lz4', complevel=6)
                    dfcol.to_hdf(outputfile, key='columns',
                                 complib = 'blosc:lz4', complevel=6)
                elif focuser == 'Delay Doppler v2':
                    # restructure and save data
                    dfsar = pd.DataFrame(sardata)
                    dfcol = pd.DataFrame(columns)
                    dfet = pd.DataFrame(int_et)
                    dfsar.to_hdf(outputfile, key='sar',
                                 complib = 'blosc:lz4', complevel=6)
                    dfcol.to_hdf(outputfile, key='columns',
                                 complib = 'blosc:lz4', complevel=6)
                    dfet.to_hdf(outputfile, key='interpolated ephemeris',
                                 complib = 'blosc:lz4', complevel=6)
            elif saving == "npy":
                outputdir = os.path.dirname(outputfile)
                if focuser != 'Delay Doppler v2':
                    np.save(os.path.join(outputdir, "sar.npy"), sar)
                    np.save(os.path.join(outputdir, "columns.npy"), columns)
                elif focuser == 'Delay Doppler v2':
                    np.save(os.path.join(outputdir, "sar.npy"), sar)
                    np.save(os.path.join(outputdir, "columns.npy"), columns)
                    np.save(os.path.join(outputdir, "interpolated_ephemeris.npy"), int_et)
            else:
                logging.error("Can't save to format '{:s}'".format(saving))
                return 1

    except Exception as e:
        
        logging.error('{:s}: Error processing file: {:s}'.format(taskname,path))
        #logging.error("{:s}: {:s}".format(taskname, str(e)) )
        for line in traceback.format_exc().split("\n"):
            logging.error('{:s}: {:s}'.format(taskname, line) )
 

        return 1
    
    logging.debug('{:s}: Successfully processed file: {:s}'.format(taskname,path))
    
    return 0



def main():
    # TODO: improve description
    parser = argparse.ArgumentParser(description='Run SAR processing')
    parser.add_argument('-o','--output', default='/disk/kea/SDS/targ/xtra/SHARAD/foc/', help="Output base directory")
    parser.add_argument(     '--ofmt',   default='hdf5',choices=('hdf5','npy','none'), help="Output data format")
    parser.add_argument('-j','--jobs', type=int, default=3, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    parser.add_argument('--tracklist', default="elysium.txt",
        help="List of tracks to process")
    parser.add_argument('--maxtracks', default=None, type=int, help="Max number of tracks to process")

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout, 
        format="run_sar2: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs
    # Set output base directory
    outputroot = args.output

    # set SAR processing variables as a dictionary
    focuser = 'Delay Doppler v2'
    processing_parameters = {
            'ddv1_posting_distance [m]': 115,
            'ddv1_aperture_time [s]': 8.774,
            'ddv1_bandwidth [Hz]': 0.4,
            'mf_posting_distance [m]': 115,
            'mf_aperture_time [s]': 8.774,
            'mf_bandwidth [Hz]': 0.4,
            'mf_recalc_int [samples]': 20,
            'mf_Er': 1.00,
            'ddv2_interpolate_dx [m]': 5,
            'ddv2_posting_interval [range lines]': 5,
            'ddv2_aperture_dist [km]': 30,
            'ddv2_trim [samples]': []}

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    #lookup = np.genfromtxt('lookup.txt',dtype='str')
    lookup = np.genfromtxt(args.tracklist, dtype = 'str')
    # Build list of processes
    process_list=[]
    logging.info("Building task list from {:s}".format(args.tracklist))

    inputroot='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR'
    for i, path in enumerate(lookup):
    #for orbit in keys:
    #    gob = int(orbit.replace('/orbit', ''))
    #    path = lookup[gob]
    #    idx_start = h5file[orbit]['idx_start'][0]
    #    idx_end = h5file[orbit]['idx_end'][0]
        logging.debug("[{:03d} of {:03d}] Building task for {:s}".format(i+1, len(lookup), path))
    
        # check if file has already been processed
        path_file = os.path.relpath(path, inputroot)
        data_file = os.path.basename(path_file)
        orbit_name = data_file[2:7]
        path_file = os.path.dirname(path_file)
        if focuser == 'Delay Doppler v1':
            new_path = os.path.join(outputroot, path_file, 
                                    str(processing_parameters['ddv1_posting_distance [m]']) + 'm', 
                                    str(processing_parameters['ddv1_aperture_time [s]']) + 's',
                                    str(processing_parameters['ddv1_bandwidth [Hz]']) + 'Hz')
        elif focuser == 'Matched Filter':
            new_path = os.path.join(outputroot, path_file, 
                                    str(processing_parameters['mf_posting_distance [m]']) + 'm', 
                                    str(processing_parameters['mf_aperture_time [s]']) + 's',
                                    str(processing_parameters['mf_bandwidth [Hz]']) + 'Hz',
                                    str(processing_parameters['mf_Er']) + 'Er')
        elif focuser == 'Delay Doppler v2':
            new_path = os.path.join(outputroot, path_file, 
                                    str(processing_parameters['ddv2_interpolate_dx [m]']) + 'm', 
                                    str(processing_parameters['ddv2_posting_interval [range lines]']) + ' range lines',
                                    str(processing_parameters['ddv2_aperture_dist [km]']) + 'km')

        outputfile = os.path.join(new_path, data_file.replace('_a.dat','_s.h5') )

        logging.debug("Looking for {:s}".format(new_path))

        # For these orbits, process only the range described by these start/end indexes
        orbit_indexes = {
            # GNG: For performance, correctness testing only (shorten track)
            # TODO: allow this to be set with a command line argument
            '03366': [   0, 3000],
            '34340': [   0, 3000],
            '51166': [   0, 3000],
            '52729': [   0, 3000],
            #############################

            '05901': [78000, 141000],
            '10058': [30000, 62000],
            '16403': [38000, 80000],
            '17333': [39000, 71000],
            '17671': [43000, 75000],
            '23535': [43000, 75000],
            '26827': [43000, 75000],
            '27104': [43000, 75000],
            '32317': [ 3000, 32000],
            '50343': [13000, 45000],
            '50352': [13000, 45000],
            '50365': [13000, 45000],
            '50409': [13000, 45000],
            '22769': [55243, 61243],
            '49920': [55035, 61035],
        }

        if os.path.exists(new_path):
            logging.debug('File already processed. Skipping: '+path)
            continue

        # Get the orbit index from the dict, or just use None
        orbit_index = orbit_indexes.get(orbit_name, [None, None])
        taskinfo = {
        'name': 'Task{:03d}-{:s}'.format(i, orbit_name),
        'input': path,
        'output': outputfile,
        'idx_start': orbit_index[0],
        'idx_end':   orbit_index[1],        
        }
        process_list.append(taskinfo)
        logging.debug("{:s} input:  {:s}".format(taskinfo['name'], taskinfo['input']) )
        logging.debug("{:s} output: {:s}".format(taskinfo['name'], taskinfo['output']) )

    #h5file.close()

    if args.maxtracks:
        logging.info("Processing first {:d} tracks".format(args.maxtracks))
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        # If it's a dry run, print the list and stop
        print(process_list)
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)) )
    start_time = time.time()

#    params_pos = (posting,aperture,bandwidth,focuser,recalc,Er)
    params_named = {'saving':args.ofmt,'debug':args.verbose}
    if nb_cores <= 1:            
        for t in process_list:
            sar_processor(t, processing_parameters, focuser, args.ofmt, args.verbose)
            #params2 = (t,) + processing_parameters + focuser
            #sar_processor( *params2, **params_named )
    else:
        run_mp(nb_cores, processing_parameters, focuser, params_named, process_list)
    logging.info("Done in {:0.1f} seconds".format( time.time() - start_time ) )

def run_mp(nb_cores, processing_parameters, focuser, params_named, process_list):
    pool = multiprocessing.Pool(nb_cores)
    results = [pool.apply_async(sar_processor, (t, processing_parameters, focuser, params_named['saving'], params_named['debug']))
        for t in process_list]

    for i, result in enumerate(results):
        dummy = result.get()
        if dummy == 1:
            logging.error("Processing task {:d} of {:d} had a problem.".format(i+1, len(process_list) ))
        else:
            logging.info( "Processing task {:d} of {:d} successful.".format(i+1, len(process_list) ))

if __name__ == "__main__":
    # execute only if run as a script
    main()
