#!/usr/bin/env python3

"""
run_surface.py -- compute surface echo powers
See README.md for general usage help among processing scripts

"""

import os
import sys
import argparse
#from datetime import datetime
import logging
import multiprocessing
import numpy as np
import pandas as pd
import subradar as sr

import SHARADEnv
from SHARADEnv import DataMissingException
from run_rng_cmp import add_standard_args, run_jobs, process_product_args, \
                        should_process_products


def surface_processor(orbit, typ='cmp', ywinwidth=100, archive=False,
                      gain=0, gain_altitude='grima2021', gain_sahga=True,
                      senv=None, method='grima2012', alt_data=True,
                      output_filename:str=None,
                      debug_dir=None):
    """
    Get the maximum of amplitude*(d amplitude/dt) within bounds defined by the
    altimetry processor

    Input:
    -----

    orbit: string
        the orbit number or the full name of the orbit file (w/o extension)
        if the orbit is truncated in several file
    typ: string
        The type of radar data used to get the amplitude from
    gain: float
        Any gain to be added to the signal (power in dB)
        For SHARAD, it includes the instrumental gain and theabsolute
        calibration value
    gain_altitude: boolean
        correct for altitude variation
    gain_sahga: boolean
        correct for SA and HGA orientation
    method: string
        surface detection method to use by subradar.surface.detector
    archive: boolean
        Whether to save the results in a txt file into the hierarchy
    alt_data: boolean
        Whether to use alt data or not

    Output:
    ------

    pandas dataframe with:
        y: surfac echo location in fast time
        amp: surface echo amplitude
        pdb: surface echo power in dB
    #    noise: background noise in dB

    """

    if senv is None: # pragma: no cover
        senv = SHARADEnv.SHARADEnv()

    #----------
    # Load data

    orbit_full = orbit if orbit.find('_') == 1 else senv.orbit_to_full(orbit)

    logging.debug('PROCESSING: Surface echo extraction for %s', orbit_full)

    assert typ == 'cmp', "Not getting the right type of radargram!"
    rdg = senv.cmp_data(orbit_full)

    if alt_data: # == True:
        alt = senv.alt_data(orbit_full, typ='beta5', ext='h5', quality_flag=True)

        flag = alt['flag']

        alty = alt['idx_fine']
        alty[alty < 0] = 0
        alty[alty > 3600] = 0
    else:
        alty = np.full(rdg.shape[0], np.ceil(ywinwidth/2))


    #------------------------
    # Get surface coordinates

    #np.nan_to_num(rdg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if method == 'mouginot2010':
        # works better with dB power
        rdg_for_detection = 20*np.log10(np.abs(rdg)) 
        #np.nan_to_num(rdg_for_detection, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    elif method == 'grima2012':
        # works better with linear power
        rdg_for_detection = np.abs(rdg)**2

    else:
        rdg_for_detection = rdg

    surf_y = sr.surface.detector(rdg=rdg_for_detection, y0=alty,
                                    winsize=ywinwidth, method=method, axis=0)

    #-----------------------
    # Get surface amplitudes

    surf_amp = surf_y*0
    for xindex, yindex in enumerate(surf_y):
        if np.isnan(yindex):
            surf_amp[xindex] = 0.0
        else:
            surf_amp[xindex] = np.abs(rdg[int(xindex), int(yindex)])

    #------------
    # Apply gains

    total_gain = gain
    aux = senv.aux_data(orbit_full)

    if gain_altitude:
        total_gain += relative_altitude_gain(aux)

    if gain_sahga:
        total_gain += relative_sahga_gain(aux)

    surf_amp = surf_amp * 10**(total_gain/20.)
    #noise = noise + total_gain

    #----------
    # Archiving

    if not alt_data:
        flag = np.full(len(surf_y), 0)
    out = {'y':surf_y, 'amp':surf_amp, 'flag':flag, } 
           #'noise':noise, 'pdb':20*np.log10(surf_amp)}

    if archive:
        archive_surface(output_filename, orbit_full, aux, out, typ)

    if debug_dir is not None:
        # Debugging output
        outfile = os.path.join(debug_dir, "srf_{:s}.npy".format(orbit_full))
        logging.debug("Saving to %s", outfile)
        os.makedirs(debug_dir, exist_ok=True)
        np.save(outfile, out)


    return out


def relative_altitude_gain(aux, method='grima2012'):
    """Provide relative altitude gain for an orbit following
    Grima et al. 2012 or Campbell et al. (2021, eq.1)
    """

    if method  == 'grima2012':
        alt_ref = 250 # Reference altitude in km
        alt = aux['SPACECRAFT_ALTITUDE']
        gain = 20*np.log10(alt/alt_ref)

    if method == 'campbell2021':
        lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']
        lat = lat/90 # Normalization between -1 and 1 (Bruce's 2021/04/23 email )
        gain = -.41 -1.62*lat -1.10*lat**2 +.65*lat**3 +.66*lat**4  #dB
        gain = -gain

    return gain


def relative_sahga_gain(aux):
    """Provide relative SA and HGA gain for an orbit following
    Campbell et al. (2021, eq.4)
    """
    samxin = aux['MRO_SAMX_INNER_GIMBAL_ANGLE']
    sapxin = aux['MRO_SAPX_INNER_GIMBAL_ANGLE']
    hgaout = aux['MRO_HGA_OUTER_GIMBAL_ANGLE']

    gain = 0.0423*np.abs(samxin) + 0.0274*np.abs(sapxin) - 0.0056*np.abs(hgaout)
    return -gain


def archive_surface(output_filename, orbit_full: str, aux, srf_data, typ: str):
    """
    Archive in the hierarchy results obtained from srf_processor

    Input:
    -----

    output_filename: str - path where to save data

    orbit_full: string
        the orbit number or the full name of the orbit file (w/o extension)
        if the orbit is truncated in several file

    srf_data: list
        output from srf_processor

    typ: string
        The type of radar data used to get the amplitude from

    """

    # Gather columns from auxilliary information
    columns = ['EPHEMERIS_TIME',
               'SUB_SC_PLANETOCENTRIC_LATITUDE',
               'SUB_SC_EAST_LONGITUDE',
               'SOLAR_ZENITH_ANGLE',
               'SOLAR_LONGITUDE',
               'SPACECRAFT_ALTITUDE',
               'SC_ROLL_ANGLE',
               'surf_y',
               'surf_amp',
               'surf_pow',
               'flag']

    values = [aux['EPHEMERIS_TIME'],
              aux['SUB_SC_PLANETOCENTRIC_LATITUDE'],
              aux['SUB_SC_EAST_LONGITUDE'],
              aux['SOLAR_ZENITH_ANGLE'],
              aux['SOLAR_LONGITUDE'],
              aux['SPACECRAFT_ALTITUDE'],
              aux['SC_ROLL_ANGLE'],
              srf_data['y'],
              srf_data['amp'],
              20*np.log10(srf_data['amp']),
              srf_data['flag']
             ]

    out = pd.DataFrame(values).transpose()
    out.columns = columns

    # Archive

    #k = p['orbit_full'].index(orbit_full)
    #list_orbit_info = senv.get_orbit_info(orbit_full)
    #orbit_info = list_orbit_info[0]

    #assert typ == 'cmp', "This probably works fine, but hasn't been tested with any value other than 'cmp'"

    #archive_path = os.path.join(senv.out['srf_path'],
    #                            orbit_info['relpath'], typ)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    #fil = os.path.join(archive_path, orbit_full + '.txt')
    out.to_csv(output_filename, index=None, sep=',')
    logging.info('CREATED: %s', output_filename)


def main():
    """Executed if run as script
    """
    desc = 'Produce standard data products for surface echo power'
    parser = argparse.ArgumentParser(description=desc)

    #--------------------
    # Job control options
    add_standard_args(parser, script='srf')

    #------------------
    # Algorithm options

    parser.add_argument('-y', '--ywinwidth', nargs='+', type=int,
                        default=100,
                        help='Number of samples defining the fast-time relative \
                        boundaries around the altimetry surface return where \
                        the surface will be looked for')
    parser.add_argument('-t', '--type', type=str, default='cmp',
                        help='Type of radar data used to get amplitude')


    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_srf: [%(levelname)-7s] %(message)s")

    if args.output is None:
        args.output = os.path.join(args.SDS, 'targ/xtra/SHARAD')

    orig_path = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')
    senv = SHARADEnv.SHARADEnv(data_path=args.output, orig_path=orig_path, b_index_files=False)
    #--------------------------
    # Requested Orbits handling
    if args.orbits == ['all']:
        # Get all known product IDs from the index
        productlist = senv.sfiles.product_id_index.keys()
    else:
        #senv.index_files(index_intermediate_files=False)
        productlist = process_product_args(args.orbits, args.tracklist, senv.sfiles)

    assert productlist, "No files to process"


    # Keyword arguments for processing
    kwargs = {
        'typ':args.type,
        'ywinwidth':args.ywinwidth,
        'archive':True,
        'gain':0,
        'gain_altitude':'grima2021',
        'gain_sahga':True,
        'method':'grima2012',
        'alt_data':True,
        'senv': senv,
    }

    process_list = []
    ii = 0
    for ii, product_id in enumerate(productlist, start=1):
        try:
            infiles = senv.sfiles.product_paths(args.type, product_id) # nominally args.type=='cmp'
            infiles.update(senv.sfiles.product_paths('alt', product_id))
            outfiles = senv.sfiles.product_paths('srf', product_id, typ=args.type)
        except KeyError: # pragma: no cover
            logging.debug("Can't find product ID %s in index for %s, alt, srf", product_id, args.type)
            continue

        if not should_process_products(product_id, infiles, outfiles, args.overwrite, loglevel=logging.DEBUG):
            continue

        params = {'orbit': product_id, 'output_filename': outfiles['srf_txt']}
        params.update(kwargs)
        process_list.append(params)

    logging.info("Processing %d orbits of %d requested", len(process_list), ii)

    if args.dryrun:
        logging.debug("orbits: %s", ' '.join([p['orbit'] for p in process_list]))
        return

    run_jobs(surface_processor, process_list, args.jobs)



if __name__ == "__main__":
    # execute only if run as a script
    main()
