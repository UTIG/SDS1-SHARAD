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
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '../xlib/')
from sar import sar
import cmp.pds3lbl as pds3

from run_rng_cmp import run_jobs, process_product_args,\
                        should_process_products, add_standard_args, read_tracklistfile

from SHARADEnv import SHARADEnv, SHARADFiles

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#import matplotlib
## Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


# Code mapping PRI codes to actual pulse repetition intervals
PRI_TABLE = {
    1: 1428E-6,
    2: 1429E-6,
    3: 1290E-6,
    4: 2856E-6,
    5: 2984E-6,
    6: 2580E-6
}




def sar_processor(taskinfo, procparam, focuser,
                  senv: SHARADEnv, saving="hdf5", debug=False):
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
                  - input: (required) path to EDR data file
                  - output: (required) path to output file (or None to omit saving)
                  - sharadroot: path to root of SHARAD data
                  - SDS: Root directory (usually /disk/kea/SDS)
                  - idx_start: (optional)
                  - idx_end: (optional)
      focuser   : Flag for which SAR focuser to use.
     procparam  : dict of processing parameters required by the different SAR
                  processors, with the following keys
                  *** delay_doppler_v1 ***
                  - ddv1_posting_distance: distance [m] along track at which to place SAR columns
                  - ddv1_aperture_time   : length [s] of the synthetic aperture
                  - ddv1_bandwidth       : Doppler bandwidth [Hz] for multilooking
                  *** matched_filter ***
                  - mf_posting_distance  : distance [m] along track at which to place SAR columns
                  - mf_aperture_time     : length [s] of the synthetic aperture
                  - mf_bandwidth         : Doppler bandwidth [Hz] for multilooking
                  - mf_recalc_int        : interval [samples] at which to recalculate the matched filter
                  - mf_Er                : relative dielectric permittivity of the subsurface
                  *** delay_doppler_v2 ***
                  - ddv2_interpolate_dx  : groundtrack interval [m] at which to interpolate range lines
                  - ddv2_posting_interval: interpolated range line interval at which to place SAR columns
                  - ddv2_aperture_dist   : length [km] of the synthetic aperture
                  - ddv2_trim            : number of sample to keep in trimmed version of the processing
      senv: SHARADEnv object describing hierarchy organization

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
    sharad_root = taskinfo.get('sharad_root', None)

    try:

        idx_start = taskinfo.get('idx_start', None)
        idx_end = taskinfo.get('idx_end', None)
        outputfile = taskinfo['output']

        # print info in debug mode
        logging.debug("%s: SAR method: %s", taskname, focuser)
        if focuser == 'Delay Doppler v1':
            number_of_looks = np.floor(procparam['ddv1_aperture_time [s]'] * 2 * procparam['ddv1_bandwidth [Hz]'])
            logging.debug("%s: SAR column posting interval [m]: %f", taskname, procparam['ddv1_posting_distance [m]'])
            logging.debug("%s: SAR aperture length [s]: %f", taskname, procparam['ddv1_aperture_time [s]'])
            logging.debug('%s: SAR Doppler bandwidth [Hz]: %f', taskname, procparam['ddv1_bandwidth [Hz]'])
            logging.debug('%s: SAR number of looks: %d', taskname, int(number_of_looks))
            del number_of_looks
        elif focuser == 'Matched Filter':
            number_of_looks = np.floor(procparam['mf_aperture_time [s]'] * 2 * procparam['mf_bandwidth [Hz]'])
            logging.debug("%s: SAR column posting interval [m]: %f", taskname, procparam['mf_posting_distance [m]'])
            logging.debug("%s: SAR aperture length [s]: %f", taskname, procparam['mf_aperture_time [s]'])
            logging.debug('%s: SAR Doppler bandwidth [Hz]: %f', taskname, procparam['mf_bandwidth [Hz]'])
            logging.debug('%s: SAR number of looks: %d', taskname, int(number_of_looks))
            logging.debug('%s: SAR matched filter recalc interval: %d', taskname, procparam['mf_recalc_int [samples]'])
            logging.debug('%s: SAR subsurface permittivity: %f', taskname, procparam['mf_Er'])
            del number_of_looks
        elif focuser == 'Delay Doppler v2':
            logging.debug("%s: SAR range line interpolation interval [m]: %f", taskname, procparam['ddv2_interpolate_dx [m]'])
            logging.debug("%s: SAR column posting interval [range lines]: %f", taskname, procparam['ddv2_posting_interval [range lines]'])
            logging.debug('%s: SAR aperture distance [km]: %f', taskname, procparam['ddv2_aperture_dist [km]'])
            if len(procparam['ddv2_trim [samples]']) != 0:
                logging.debug('%f: SAR fast-time trim [samples]: %f', taskname, procparam['ddv2_trim [samples]'])

        product_id = Path(taskinfo['input']).name.replace('_a.dat', '')
        assert product_id.startswith('e_'), "Unexpected product_id"
        edr_paths = senv.sfiles.edr_product_paths(product_id)
        label_path = os.path.join(senv.sfiles.orig_path, 'EDR/mrosh_0004/label/science_ancillary.fmt')
        aux_path = os.path.join(senv.sfiles.orig_path, 'EDR/mrosh_0004/label/auxiliary.fmt')

        cmp_track = senv.cmp_data(product_id)

        idx_start = 0            if idx_start is None else max(0, idx_start)
        idx_end = len(cmp_track) if idx_end   is None else min(len(cmp_track), idx_end)
        cmp_track = cmp_track[idx_start:idx_end]

        logging.debug("%s: Processing track from start=%d end=%d (length=%d)",
            taskname, idx_start, idx_end, len(cmp_track))

        # load the relevant EDR files
        science_path = edr_paths['edr_sci']
        logging.debug("%s: Loading science data from EDR file: %s", taskname, science_path)
        scireader = pds3.SHARADDataReader(label_path, science_path)
        data = scireader.arr[idx_start:idx_end]
        bitdata = scireader.get_bitcolumns()[idx_start:idx_end]

        auxfile = science_path.replace('_s.dat', '_a.dat')
        logging.debug("%s: Loading auxiliary data from EDR file: %s", taskname, auxfile)
        aux = pds3.read_science(auxfile, aux_path)[idx_start:idx_end]

        logging.debug("%s: EDR sci data length: %d", taskname, len(data))
        logging.debug("%s: EDR aux data length: %d", taskname, len(aux))
        assert len(data) == len(aux), "Science and aux data lengths don't match"

        # load relevant spacecraft position information from EDR files
        pri_code = bitdata['PULSE_REPETITION_INTERVAL']
        rxwot = np.copy(data['RECEIVE_WINDOW_OPENING_TIME'])
        if focuser != 'Delay Doppler v2':
            for j, code in enumerate(pri_code):
                pri = PRI_TABLE.get(code, 0.0)
                rxwot[j] *= 0.0375E-6 + pri - 11.98E-6
        et = aux['EPHEMERIS_TIME']
        tlp = data['TLP_INTERPOLATE']
        scrad = data['RADIUS_INTERPOLATE']
        if focuser == 'Delay Doppler v1':
            tpgpy = data['TOPOGRAPHY']
            vel = np.hypot(data['TANGENTIAL_VELOCITY_INTERPOLATE'],
                           data['RADIAL_VELOCITY_INTERPOLATE'])
        elif focuser == 'Delay Doppler v2':
            tpgpy = data['TOPOGRAPHY']
            lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']
            lng = aux['SUB_SC_EAST_LONGITUDE']
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
        logging.debug("%s: Start of SAR processing", taskname)
        if focuser == 'Delay Doppler v1':
            sardata, columns = sar.delay_doppler_v1(cmp_track, procparam['ddv1_posting_distance [m]'],
                                                    procparam['ddv1_aperture_time [s]'],
                                                    procparam['ddv1_bandwidth [Hz]'],
                                                    tlp, et, scrad, tpgpy, rxwot2 - min(rxwot2), vel, debugtag=taskname)
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
                                                            et, tpgpy, pri_code, scrad, rxwot, lat, lng, band,
                                                            debugtag=taskname)

        # save the result
        if saving and outputfile is not None:
            new_path = os.path.dirname(outputfile)
            logging.debug("%s: Saving to file: %s", taskname, outputfile)
            os.makedirs(new_path, exist_ok=True)

            if saving == "hdf5":
                if focuser != 'Delay Doppler v2':
                    # restructure and save data
                    dfsar = pd.DataFrame(sardata)
                    dfcol = pd.DataFrame(columns)
                    dfsar.to_hdf(outputfile, key='sar',
                                 complib='blosc:lz4', complevel=6)
                    dfcol.to_hdf(outputfile, key='columns',
                                 complib='blosc:lz4', complevel=6)
                elif focuser == 'Delay Doppler v2':
                    # restructure and save data
                    dfsar = pd.DataFrame(sardata)
                    dfcol = pd.DataFrame(columns)
                    dfet = pd.DataFrame(int_et)
                    dfsar.to_hdf(outputfile, key='sar',
                                 complib='blosc:lz4', complevel=6)
                    dfcol.to_hdf(outputfile, key='columns',
                                 complib = 'blosc:lz4', complevel=6)
                    dfet.to_hdf(outputfile, key='interpolated_ephemeris',
                                 complib = 'blosc:lz4', complevel=6)
            elif saving == "npy":
                outputdir = os.path.dirname(outputfile)
                # TODO: change this from multiple npy files to one big npz file
                if focuser != 'Delay Doppler v2':
                    np.save(os.path.join(outputdir, "sar.npy"), sar)
                    np.save(os.path.join(outputdir, "columns.npy"), columns)
                elif focuser == 'Delay Doppler v2':
                    np.save(os.path.join(outputdir, "sar.npy"), sar)
                    np.save(os.path.join(outputdir, "columns.npy"), columns)
                    np.save(os.path.join(outputdir, "interpolated_ephemeris.npy"), int_et)
            elif saving == "none":
                pass
            else:
                logging.error("Can't save to format '%s'", saving)
                return 1

    except Exception: # pylint: disable=W0703
        logging.error('%s: Error processing %s', taskname, taskinfo.get('input', 'unknowninput'))
        for line in traceback.format_exc().split("\n"):
            logging.error('%s: %s', taskname, line)

        return 1

    logging.debug('%s: Successfully processed %s', taskname, product_id)

    return 0



def main():

    parser = argparse.ArgumentParser(description='Run SAR processing')
    parser.add_argument('--ofmt', default='hdf5',
                        choices=('hdf5', 'npy', 'none'),
                        help="Output data format")
    parser.add_argument('--focuser', default='ddv2',
                        choices=('ddv1', 'mf', 'ddv2'),
                        help="Focusing algorithm")
    parser.add_argument('--params', default=None,
                        help="Processing parameters configuration file (JSON format)")

    add_standard_args(parser)
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_sar2: [%(levelname)-7s] %(message)s")


    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')

    sharad_root = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    #sfiles = SHARADFiles(data_path=args.output, orig_path=sharad_root, read_edr_index=True)
    senv = SHARADEnv(data_path=args.output, orig_path=sharad_root, b_index_files=False)
    senv.index_files(use_edr_index=True, index_intermediate_files=False)
    sfiles = senv.sfiles


    # set SAR processing variables as a dictionary
    # TODO: make the calling function accept the codes below
    focuser_names = {
        'ddv1': 'Delay Doppler v1',
        'mf': 'Matched Filter',
        'ddv2': 'Delay Doppler v2',
    }
    focuser = focuser_names[args.focuser]
    processing_parameters = { # default SAR processing parameters
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
        'ddv2_aperture_dist [km]': 40,
        'ddv2_trim [samples]': []
    }
    if args.params is not None:
        # Load a json file if specified
        logging.debug("Loading processing parameters from %s", args.params)
        with open(args.params, 'r') as configfile:
            try:
                processing_parameters = json.load(configfile)
            except json.decoder.JSONDecodeError as e:
                logging.error("Problem parsing %s", args.params)
                raise e


    # Read lookup table associating gob's with tracks
    #h5file = pd.HDFStore('mc11e_spice.h5')
    #keys = h5file.keys()
    lookup = list(read_tracklistfile(args.tracklist))

    # Build list of processes
    process_list = []
    logging.info("Making task list from %s", args.tracklist)
    inputroot = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD/EDR')
    for i, path in enumerate(lookup):
    #for orbit in keys:
    #    gob = int(orbit.replace('/orbit', ''))
    #    path = lookup[gob]
    #    idx_start = h5file[orbit]['idx_start'][0]
    #    idx_end = h5file[orbit]['idx_end'][0]
        logging.debug("[%03d of %03d] Making task for %s", i+1, len(lookup), path)

        # check if file has already been processed
        path_file = os.path.relpath(path, inputroot)
        data_file = os.path.basename(path_file)
        orbit_name = data_file[2:7]
        path_file = os.path.dirname(path_file)

        if focuser == 'Delay Doppler v1':
            new_path = os.path.join(args.output, 'foc', path_file,
                                    str(processing_parameters['ddv1_posting_distance [m]']) + 'm',
                                    str(processing_parameters['ddv1_aperture_time [s]']) + 's',
                                    str(processing_parameters['ddv1_bandwidth [Hz]']) + 'Hz')
        elif focuser == 'Matched Filter':
            new_path = os.path.join(args.output, 'foc', path_file,
                                    str(processing_parameters['mf_posting_distance [m]']) + 'm',
                                    str(processing_parameters['mf_aperture_time [s]']) + 's',
                                    str(processing_parameters['mf_bandwidth [Hz]']) + 'Hz',
                                    str(processing_parameters['mf_Er']) + 'Er')
        elif focuser == 'Delay Doppler v2':
            new_path = os.path.join(args.output, 'foc', path_file,
                                    str(processing_parameters['ddv2_interpolate_dx [m]']) + 'm',
                                    str(processing_parameters['ddv2_posting_interval [range lines]']) + ' range lines',
                                    str(processing_parameters['ddv2_aperture_dist [km]']) + 'km')

        outputfile = os.path.join(new_path, data_file.replace('_a.dat', '_s.h5'))

        outputfile2 = foc_product_paths(senv.sfiles, focuser, processing_parameters, path)['foc_rad']
        assert outputfile == outputfile2, "%s != %s" % (outputfile, outputfile2)

        logging.debug("Looking for %s", new_path)

        # For these orbits, process only the range described by these start/end indexes
        orbit_indexes = {
            # GNG: For performance, correctness testing only (shorten track)
            # TODO: allow this to be set with a command line argument
            '03366': [    0,   3000],
            '34340': [    0,   3000],
            '51166': [    0,   3000],
            '52729': [    0,   3000],
            #############################

            '05901': [78000, 141000],
            '10058': [30000,  62000],
            '16403': [38000,  80000],
            '17333': [39000,  71000],
            '17671': [43000,  75000],
            '23535': [43000,  75000],
            '26827': [43000,  75000],
            '27104': [43000,  75000],
            '32317': [ 3000,  32000],
            '50343': [13000,  45000],
            '50352': [13000,  45000],
            '50365': [13000,  45000],
            '50409': [13000,  45000],
            '22769': [55243,  61243],
            '49920': [55035,  61035],

            # 1-BIT SAR TESTING ########
            '12945': [40000, 120000],
            '17481': [    0,  25000],

            ## WESTERN ALBA MONS #######
            '03512': [    0,  10688],
            '04774': [    0,   1844],
            '05552': [    0,   5693],
            '05968': [92890, 112903],
            '06475': [    0,   5681],
            '06620': [    0,   1468],
            '06910': [90088, 133182],
            '06976': [    0,   4374],
            '07121': [90132, 133225],
            '07141': [ 4381,   9802],
            '07543': [90199, 133348],
            '07754': [90172, 133276],
            '07787': [ 4336,   9802],
            '07965': [90127, 133245],
            '07978': [    0,   5654],
            '08044': [    0,  15565],
            '08110': [    0,  15616],
            '08123': [    0,  18608],
            '08143': [ 8944,  52041],
            '08387': [90329, 133491],
            '08433': [ 4378,   9627],
            '08769': [    0,   5618],
            '08809': [90218, 133370],
            '09125': [    0,  15543],
            '09191': [    0,  15464],
            '09270': [    0,   5592],
            '09402': [    0,   6193],
            '09435': [ 4377,   9627],
            '10569': [    0,  35073],
            '11347': [14467,  57560],
            '12633': [75572, 104153],
            '13035': [19977,  63060],
            '13767': [89900, 121346],
            '15995': [89959, 133048],
            '16417': [89811, 132953],
            '16628': [89786, 132889],
            '16839': [89802, 132893],
            '17050': [89915, 133049],
            '17261': [89763, 128850],
            '17683': [89978, 133085],
            '17894': [89869, 133012],
            '18105': [89865, 133008],
            '18316': [89933, 133043],
            '19371': [44976,  66542],
            '20215': [44968,  66540],
            '20637': [44928,  66503],
            '20848': [44937,  66503],
            '27441': [    0,   1466],
            '33955': [58861, 101988],
            '33968': [89735, 103246],
            '35788': [87061, 130208],
            '36645': [56635,  99770],
            '36658': [89772, 106969],
            '36671': [89838, 132949],
            '37944': [15763,  58861],
            '38432': [16036,  59136],
            '38564': [    0,   4303],
            '38590': [    0,   9834],
            '38597': [16194,  21007],
            '41208': [71678, 114837],
            '45599': [38235,  81375],
            '46245': [89751,  92819],
            '46291': [17466,  60589],
            '46733': [74553, 117662],
            '47076': [79989, 123149],
            '47089': [34766,  77909],
            '47432': [79149, 122281],
            '47445': [31957,  75072],
            '47498': [88737, 127313],
            '48078': [69323, 112471],
            '48658': [42009,  85145],
            '48790': [77397, 120540],
            '48856': [87889, 131032],
            '49014': [37757,  80883],
            '49027': [89765, 132879],
            '49146': [73036, 116179],
            '50214': [41105,  84238]
        }

        if not args.overwrite and os.path.exists(new_path):
            logging.debug('File already processed. Skipping: '+path)
            continue

        # Get the orbit index from the dict, or just use None
        orbit_index = orbit_indexes.get(orbit_name, [None, None])
        task = {
            'name': 'Task{:03d}-{:s}'.format(i, orbit_name),
            'input': path,
            'output': outputfile,
            'idx_start': orbit_index[0],
            'idx_end': orbit_index[1],
        }
        params = {
            'taskinfo': task,
            'focuser': focuser,
            'procparam': processing_parameters,
            'senv': senv,
            'saving': args.ofmt,
            'debug': args.verbose,
        }
        process_list.append(params)
        logging.debug("%s input:  %s", task['name'], task['input'])
        logging.debug("%s output: %s", task['name'], task['output'])


    if args.maxtracks > 0 and len(process_list) > args.maxtracks:
        # Limit to first args.maxtracks tracks
        process_list = process_list[0:args.maxtracks]

    if args.dryrun:
        # If it's a dry run, print the list and stop
        print(process_list)
        return

    start_time = time.time()

    run_jobs(sar_processor, process_list, args.jobs)
    logging.info("Done in %0.1f seconds", time.time() - start_time)

def foc_product_paths(sfiles: SHARADFiles, focuser, processing_parameters, edr_data_file):
    """ From the EDR data file name, construct the focused data file name and return it """
    p_edr_data_file = Path(edr_data_file)
    product_id = p_edr_data_file.name.replace('_a.dat','')
    relpath = p_edr_data_file.relative_to(Path(sfiles.orig_path) / 'EDR').parent
    assert str(relpath).startswith('mrosh_00'), "Unexpected path isn't in EDR archive '%s'" % (str(relpath),)

    if focuser == 'Delay Doppler v1':
        typ = "{:s}m/{:s}s/{:s}Hz".format(
                str(processing_parameters['ddv1_posting_distance [m]']),
                str(processing_parameters['ddv1_aperture_time [s]']),
                str(processing_parameters['ddv1_bandwidth [Hz]']))
    elif focuser == 'Matched Filter':
        typ = "{:s}m/{:s}s/{:s}Hz/{:s}Er".format(
                str(processing_parameters['mf_posting_distance [m]']),
                str(processing_parameters['mf_aperture_time [s]']),
                str(processing_parameters['mf_bandwidth [Hz]']),
                str(processing_parameters['mf_Er']),
                )
    elif focuser == 'Delay Doppler v2':
        typ = "{:s}m/{:s} range lines/{:s}km".format(
                str(processing_parameters['ddv2_interpolate_dx [m]']),
                str(processing_parameters['ddv2_posting_interval [range lines]']),
                str(processing_parameters['ddv2_aperture_dist [km]']))
    else:
        raise ValueError("Unknown value for focuser '%s'" % (focuser,))

    return {'foc_rad': os.path.join(sfiles.data_path, 'foc', relpath, typ, product_id + '_s.h5')}




if __name__ == "__main__":
    # execute only if run as a script
    main()
