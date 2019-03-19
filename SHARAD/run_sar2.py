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
         'info': 'Second release.'}}


"""

Example usage:
./run_sar2.py 




"""


def sar_processor(inputlist, posting, aperture, bandwidth, focuser='Delay Doppler',
                  recalc=20, Er=1, taskid=None, saving=True, debug=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual range compressed and ionosphere corrected tracks and
    performs SAR focusing using the desired algorithm.
    Note: delay Doppler is much faster than matched filter and is set to be the
          default option

    Input:
    -----------
      inputlist : list of inputs required for SAR processing
                  - path to EDR
                  - idx_start
                  - idx_end
      posting   : SAR column posting interval [m]
      aperture  : length of the SAR aperture [s]
      bandwidth : Doppler bandwidth [Hz]
      focuser   : Flag for which SAR focuser to use.
      recalc    : Sample interval to recalculate the matched filter.
                  - only required for matched filter SAR processing
      Er        : relative dielectric permittivity of the subsurface
                  - only required for matched filter SAR processing and tests
                    show results are not strongly affected
      saving    : Flag to save the SAR output.
      debug     : Enter debug mode - more info.

    Output:
    -----------
      sar       : focused SAR data
      columns   : matrix of EDR range lines specifying the mid-aperture
                  position as well as the start and end of each aperture

    # TODO [gng]: change arguments such that if output path is specified, output,
    #  if it is none, then don't save
    # TODO: change the inputlist to a namedtuple or dict
    """
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, '../xlib/sar/')
    sys.path.insert(0, '../xlib/cmp/')
    import sar
    import pds3lbl as pds3

    taskname = "Task{:03d}".format(taskid)

    try:
    
        # split the path variable 
        idx_start = inputlist[1]
        idx_end = inputlist[2]
        path = inputlist[0]
        

        # print info in debug mode
        logging.debug("{:s}: SAR method: {:s}".format(taskname, focuser)) 
        logging.debug("{:s}: SAR column posting interval[m]: {:f}".format(taskname, posting)) 
        logging.debug("{:s}: SAR aperture length[s]: {:f}".format(taskname, aperture)) 
        logging.debug('{:s}: SAR Doppler bandwidth [Hz]: {:f}'.format(taskname, bandwidth) )
        logging.debug('{:s}: SAR number of looks: {:d}'.format(taskname,  int(np.floor(aperture * 2 * bandwidth))) )

        # create cmp path
        path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = path_file.split('/')[-1]
        path_file = path_file.replace(data_file,'')
        cmp_path = path_root+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        science_path=path.replace('_a.dat','_s.dat')

        logging.debug("{:s}: Loading cmp data from {:s}".format(taskname, cmp_path))
    
        # load the range compressed and ionosphere corrected data
        real = np.array(pd.read_hdf(cmp_path, 'real'))	
        imag = np.array(pd.read_hdf(cmp_path, 'imag'))
        cmp_track = real + 1j * imag
        if idx_start is not None:
            idx_start = max(0,idx_start)
        else:
            idx_start = 0
        if idx_end is not None:
            idx_end = min(len(cmp_track),idx_end)
        else:
            idx_end = len(cmp_track)
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
        pri_code = data['PULSE_REPETITION_INTERVAL'].as_matrix()
        rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].as_matrix()
        for j in range(len(pri_code)):                   
            if pri_code[j] == 1:   pri = 1428E-6
            elif pri_code[j] == 2: pri = 1429E-6
            elif pri_code[j] == 3: pri = 1290E-6
            elif pri_code[j] == 4: pri = 2856E-6
            elif pri_code[j] == 5: pri = 2984E-6
            elif pri_code[j] == 6: pri = 2580E-6
            else: pri = 0
            rxwot[j] = rxwot[j] * 0.0375E-6 + pri - 11.98E-6
        et = aux['EPHEMERIS_TIME'].as_matrix()
        tlp = data['TLP_INTERPOLATE'].as_matrix()
        scrad = data['RADIUS_INTERPOLATE'].as_matrix()
        if focuser == 'Delay Doppler':
            tpgpy = data['TOPOGRAPHY'].as_matrix()
            vt = data['TANGENTIAL_VELOCITY_INTERPOLATE'].as_matrix()
            vr = data['RADIAL_VELOCITY_INTERPOLATE'].as_matrix()
            v = np.zeros(len(vt), dtype=float)
            for j in range(len(vt)):
                v[j] = np.sqrt(vt[j]**2 + vr[j]**2)
        
        # correct the rx window opening times for along-track changes in spacecraft
        # radius
        rxwot = rxwot - (2 * (scrad - min(scrad)) * 1000 / 299792458)

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
        if focuser == 'Delay Doppler':
            sar, columns = sar.delay_doppler(cmp_track, posting, aperture, bandwidth,
                                            tlp, et, scrad, tpgpy, rxwot - min(rxwot), v)
        elif focuser == 'Matched Filter':
            sar, columns = sar.matched_filter(cmp_track, posting, aperture, Er, bandwidth,
                                             recalc, tlp, et, rxwot)
        
        # save the result
        if saving:
            save_root = '/disk/kea/SDS/targ/xtra/SHARAD/foc/'
            path_file = science_path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
            data_file = os.path.basename(path_file)
            path_file = os.path.dirname(path_file)
            new_path = save_root + path_file
            new_path = new_path + str(posting) + 'm/'
            new_path = new_path + str(aperture) + 's/'
            new_path = new_path + str(bandwidth) + 'Hz/'
            if focuser == 'Matched Filter':
                new_path = new_path + str(Er) + 'Er/'

            outputfile = os.path.join(new_path, datafile.replace(".dat", ".h5"))

            logging.debug("{:s}: Saving to file: {:s}".format(taskname, outputfile))

            if not os.path.exists(new_path):
                os.makedirs(new_path)
          
            # restructure and save data
            dfsar = pd.DataFrame(sar)
            dfcol = pd.DataFrame(columns)
            dfsar.to_hdf(outputfile, key='sar',
                         complib = 'blosc:lz4', complevel=6)
            dfcol.to_hdf(outputfile, key='columns',
                         complib = 'blosc:lz4', complevel=6)

    except Exception as e:
        
        logging.error('Task{:03d}: Error processing file: {:s}'.format(taskid,path))
        logging.error("Task{:03d}: {:s}".format(taskid, str(e)) )
        return 1
    
    logging.debug('Task{:03d}: Successfully processed file: {:s}'.format(taskid,path))
    
    return 0

# standard python libraries and systemwide libraries
import sys
import os
import numpy as np
import multiprocessing
import argparse
import time
import logging
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import importlib.util

#import matplotlib
## Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


# project libraries
sys.path.insert(0, '../xlib/misc')
import prog
import hdf


def main():
    # TODO: improve description
    parser = argparse.ArgumentParser(description='Run SAR processing')
    parser.add_argument('-j','--jobs', type=int, default=3, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    parser.add_argument('-m','--modifypst', help="Enable PST rewriting using specified script", required=False)
    parser.add_argument('--tracklist', default="elysium.txt",
        help="List of tracks to process")
    #parser.add_argument('-f','--format', help="When outputting to stdout, use this payload format (default=repr)", required=False, default='repr',choices= sorted(rawcat_fmts.keys()) )
    #parser.add_argument('--compact', action="store_true", help="When outputting to stdout, use compact output format")
    #parser.add_argument('--single','--full',  action="store_true", help="Output full data to a single transect directory (don't create per-transect directories)")
    #parser.add_argument('--stream',help="Comma separated list of stream names to include in output")
    # implies single core
    parser.add_argument('--profile',  action="store_true", help='Profile execution performance', required=False)
    #parser.add_argument('files', nargs='+', help='Input files to process')

    args = parser.parse_args()

    #logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)
    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout, 
        format="run_sar2: [%(levelname)-7s] %(message)s")

    # Set number of cores
    nb_cores = args.jobs

    # set SAR processing variables
    posting = 115
    aperture = 11.7
    bandwidth = 0.4
    focuser = 'Delay Doppler'
    recalc = 20
    Er = 1.00

    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    #lookup = np.genfromtxt('lookup.txt',dtype='str')
    lookup = np.genfromtxt(args.tracklist, dtype = 'str')
    # Build list of processes
    process_list=[]
    logging.info("Building task list from {:s}".format(args.tracklist))

    #p = prog.Prog(len(lookup), prog_symbol='*')
    path_root = '/disk/kea/SDS/targ/xtra/SHARAD/foc/'

    # Just run the first one
    #lookup = lookup[0:1]

    for i, path in enumerate(lookup):
    #for orbit in keys:
        #p.print_Prog(i)
    #    gob = int(orbit.replace('/orbit', ''))
    #    path = lookup[gob]
    #    idx_start = h5file[orbit]['idx_start'][0]
    #    idx_end = h5file[orbit]['idx_end'][0]
        logging.debug("[{:03d} of {:03d}] Building task for {:s}".format(i+1, len(lookup), path))
    
        taskid = "Task{:03d}".format(i+1)
        # check if file has already been processed
        path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = os.path.basename(path_file)
        orbit_name = data_file[2:7]
        path_file = os.path.dirname(path_file)
        new_path = path_root + path_file
        new_path = new_path + str(posting) + 'm/'
        new_path = new_path + str(aperture) + 's/'
        new_path = new_path + str(bandwidth) + 'Hz/'
        if focuser == 'Matched Filter':
            new_path = new_path + str(Er) + 'Er/'

        logging.debug("Looking for {:s}".format(new_path))

        if not os.path.exists(new_path):
            if orbit_name == '05901':
                process_list.append([path, 78000, 141000])
            elif orbit_name == '10058':
                process_list.append([path, 30000, 62000])
            elif orbit_name == '16403':
                process_list.append([path, 38000, 80000])
            elif orbit_name == '17333':
                process_list.append([path, 39000, 71000])
            elif orbit_name == '17671':
                process_list.append([path, 43000, 75000])
            elif orbit_name == '23535':
                process_list.append([path, 43000, 75000])
            elif orbit_name == '26827':
                process_list.append([path, 43000, 75000])
            elif orbit_name == '27104':
                process_list.append([path, 43000, 75000])
            elif orbit_name == '32317':
                process_list.append([path, 3000, 32000])
            elif orbit_name == '50343':
                process_list.append([path, 13000, 45000])
            elif orbit_name == '50352':
                process_list.append([path, 13000, 45000])
            elif orbit_name == '50365':
                process_list.append([path, 13000, 45000])
            elif orbit_name == '50409':
                process_list.append([path, 13000, 45000])
            else:
                process_list.append([path, None, None])
        else:
            logging.debug('File already processed. Skipping: '+path)

    #h5file.close()

    if args.dryrun:
        # If it's a dry run, print the list and stop
        print(process_list)
        sys.exit(0)

    logging.info("Start processing {:d} tracks".format(len(process_list)) )
    start_time = time.time()

    pool = multiprocessing.Pool(nb_cores)
    results = [pool.apply_async(sar_processor, (t,posting,aperture,bandwidth,focuser,recalc,Er),
                     {'saving':False,'debug':args.verbose, 'taskid':j+1}) for j,t in enumerate(process_list)]

    #p=prog.Prog(len(keys))
    #p = prog.Prog(len(lookup), prog_symbol='#')
    for i, result in enumerate(results):
        #p.print_Prog(i)
        dummy = result.get()
        if dummy == 1:
            logging.error("Processing task {:d} of {:d} had a problem.".format(i+1, len(process_list) ))
        else:
            logging.info( "Processing task {:d} of {:d} successful.".format(i+1, len(process_list) ))

    logging.info("Done in {:0.1f} seconds".format( time.time() - start_time ) )
    #p.close_Prog()


if __name__ == "__main__":
    # execute only if run as a script
    main()
