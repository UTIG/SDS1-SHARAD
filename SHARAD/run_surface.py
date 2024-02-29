#!/usr/bin/env python3

"""
Example command:

To process a single orbit (0887601) and archive to the default location:

./run_surface.py 0887601

To process a single orbit (0887601), not archive to the default location (--ofmt
none),
and output a numpy file into a debug directory:

./run_surface.py 0887601 --ofmt none --output ./surface_data

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


class DataMissingException(Exception):
    pass


class Async:
    """Class to support multi processing jobs. For calling example, see:
    http://masnun.com/2014/01/01/async-execution-in-python-using-multiprocessing-pool.html
    """
    def __init__(self, func, cb_func, nbcores=1):
        self.func = func
        self.cb_func = cb_func
        self.pool = multiprocessing.Pool(nbcores)

    def call(self,*args, **kwargs):
        return self.pool.apply_async(self.func, args, kwargs, self.cb_func)

    def wait(self):
        self.pool.close()
        self.pool.join()


def surface_processor(orbit, typ='cmp', ywinwidth=100, archive=False,
                      gain=0, gain_altitude='grima2021', gain_sahga=True,
                      senv=None, method='grima2012', alt_data=True):
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

    if senv is None:
        senv = SHARADEnv.SHARADEnv()

    #----------
    # Load data

    orbit_full = orbit if orbit.find('_') == 1 else senv.orbit_to_full(orbit)

    logging.debug('PROCESSING: Surface echo extraction for %s', orbit_full)

    rdg = senv.cmp_data(orbit)
    if rdg is None: # pragma: no cover
        raise DataMissingException("No CMP data for orbit %s", orbit)

    if alt_data: # == True:
        alt = senv.alt_data(orbit, typ='beta5', ext='h5', quality_flag=True)
        if alt is None: # pragma: no cover
            logging.info("No Altimetry Data for orbit %s", orbit)
            return None

        flag = alt['flag']

        alty = alt['idx_fine']
        alty[alty < 0] = 0
        alty[alty > 3600] = 0
    else:
        alty = np.full(rdg.shape[0], np.ceil(ywinwidth/2))

    #rdg = senv.cmp_data(orbit)
    #if rdg is None: # pragma: no cover
    #    raise DataMissingException("No CMP data for orbit %s", orbit)

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

    if gain_altitude:
        _gain = relative_altitude_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    if gain_sahga:
        _gain = relative_sahga_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    surf_amp = surf_amp * 10**(total_gain/20.)
    #noise = noise + total_gain

    #----------
    # Archiving

    if not alt_data:
        flag = np.full(len(surf_y), 0)
    out = {'y':surf_y, 'amp':surf_amp, 'flag':flag, } 
           #'noise':noise, 'pdb':20*np.log10(surf_amp)}

    if archive:
        archive_surface(senv, orbit_full, out, typ)

    return out


def relative_altitude_gain(senv, orbit_full, method='grima2012'):
    """Provide relative altitude gain for an orbit following
    Grima et al. 2012 or Campbell et al. (2021, eq.1)
    """
    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise DataMissingException("No Auxiliary Data for orbit %s", orbit_full)

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


def relative_sahga_gain(senv, orbit_full):
    """Provide relative SA and HGA gain for an orbit following
    Campbell et al. (2021, eq.4)
    """
    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise DataMissingException("No Auxiliary Data for orbit %s", orbit_full)

    samxin = aux['MRO_SAMX_INNER_GIMBAL_ANGLE']
    sapxin = aux['MRO_SAPX_INNER_GIMBAL_ANGLE']
    hgaout = aux['MRO_HGA_OUTER_GIMBAL_ANGLE']

    gain = 0.0423*np.abs(samxin) + 0.0274*np.abs(sapxin) - 0.0056*np.abs(hgaout)
    gain = -gain

    return gain


def archive_surface(senv, orbit_full, srf_data, typ):
    """
    Archive in the hierarchy results obtained from srf_processor

    Input:
    -----

    orbit_full: string
        the orbit number or the full name of the orbit file (w/o extension)
        if the orbit is truncated in several file

    srf_data: list
        output from srf_processor

    typ: string
        The type of radar data used to get the amplitude from

    """

    # Gather auxilliary information

    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise DataMissingException("No Auxiliary Data for orbit %s", orbit_full)

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
    list_orbit_info = senv.get_orbit_info(orbit_full)
    orbit_info = list_orbit_info[0]

    assert typ == 'cmp'

    archive_path = os.path.join(senv.out['srf_path'],
                                orbit_info['relpath'], typ)

    os.makedirs(archive_path, exist_ok=True)
    fil = os.path.join(archive_path, orbit_full + '.txt')
    out.to_csv(fil, index=None, sep=',')
    logging.info('CREATED: %s', fil)

    return out


def main():
    """Executed if run as script
    """
    parser = argparse.ArgumentParser(description='Processing routines \
                                     for surface echo power extraction')

    #--------------------
    # Job control options

    #outpath = os.path.join(os.getenv('SDS'), 'targ/xtra/SHARAD')

    parser.add_argument('-o', '--output', default=None,
                        help="Debugging output data directory")
    #parser.add_argument('--ofmt',   default='hdf5',choices=('hdf5','none'),
    #        help="Output data format")
    parser.add_argument('orbits', metavar='orbit', nargs='*',
                        help='Orbit IDs to process (including leading zeroes).'
                        'If "all", processes all orbits')
    parser.add_argument('--orbitlist', help='Text file containing list of orbits to process')
    parser.add_argument('-j', '--jobs', type=int, default=8,
            help="Number of jobs (cores) to use for multi-core processing.")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")
    parser.add_argument('-n', '--dryrun', action="store_true",
                        help="Dry run. Build task list but do not run")
    #parser.add_argument('--tracklist', default="elysium.txt",
    #    help="List of tracks to process")
    #parser.add_argument('--maxtracks', default=None, type=int,
    #    help="Max number of tracks to process")

    #------------------
    # Algorithm options

    parser.add_argument('-y', '--ywinwidth', nargs='+', type=int,
                        default=100,
                        help='2 numbers defining the fast-time relative \
                        boundaries around the altimetry surface return where \
                        the surface will be looked for')
    parser.add_argument('-t', '--type', type=str, default='cmp',
                        help='Type of radar data used to get the amplitude \
                        from')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='Delete and reprocess files already processed, \
                        only if [orbit] is [all]')

    parser.add_argument('--SDS', default=os.getenv('SDS', '/disk/kea/SDS'),
                            help="Root directory (default: environment variable SDS)")

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_srf: [%(levelname)-7s] %(message)s")

    # debug output only if not multiprocessing
    assert not(args.output and args.jobs > 1)

    # Construct directory names
    data_path = os.path.join(args.SDS, 'targ/xtra/SHARAD')
    orig_path = os.path.join(args.SDS, 'orig/supl/xtra-pds/SHARAD')

    senv = SHARADEnv.SHARADEnv(data_path=data_path, orig_path=orig_path)

    #--------------------------
    # Requested Orbits handling

    logging.debug("Checking orbit processing status")
    processed = senv.processed()
    set_available = set(processed['cmp']) # To convert to EDR orbit list
    set_processable = set_available & set(processed['alt'])
    if 'srf' in processed:
        set_processable_unprocessed = set_processable - set(processed['srf'])
    else:
        set_processable_unprocessed = set_processable

    logging.debug("Done checking orbit processing status")

    if args.orbits == ['all']:
        set_requested = set_available
    else:
        set_requested = set(args.orbits)

    if args.orbitlist:
        # Load list of files from orbit list
        set_requested.add(list(np.genfromtxt(args.orbitlist, dtype='str')))

    if args.delete:
        args.orbits = list(set_processable & set_requested)
    else:
        args.orbits = list(set_processable_unprocessed & set_requested)

    args.orbits.sort()

    #-----------
    # Processing

    logging.info("TOTAL: %d orbits to process", len(args.orbits))

    if args.dryrun:
        logging.info("Dry run only -- Orbits:\n" + '\n'.join(args.orbits))
        #logging.info(f"TOTAL: {len(args.orbits)} to process")
        sys.exit(0)

    # Keyword arguments for processing
    kwargs = {'typ':args.type,
              'ywinwidth':args.ywinwidth,
              'archive':True,
              'gain':0,
              'gain_altitude':'grima2021',
              'gain_sahga':True,
              'method':'grima2012',
              'alt_data':True,
              'senv':senv
             }

    # Create Async class for multiprocessing
    if args.jobs > 0:
        async_surface = Async(surface_processor, None, nbcores=args.jobs)

    # Processing
    for orbit in args.orbits:
        #logging.debug('({}) {:>5}/{:>5}: {}'.format(datetime.now().strftime(
        #              '%Y-%m-%d %H:%M:%S'), i+1, len(args.orbits), orbit, ))

        if args.jobs < 2:
            # Do NOT use the multiprocessing package
            srf = surface_processor(orbit, **kwargs)
            if args.output is not None:
                # Debugging output
                outfile = os.path.join(args.output, "srf_{:s}.npy".format(orbit))
                logging.debug("Saving to %s", outfile)
                os.makedirs(args.output, exist_ok=True)
                np.save(outfile, srf)
        else:
            # Do use the multiprocessing package
            async_surface.call(orbit, **kwargs)


    if args.jobs >= 2:
        async_surface.wait()

    #if args.jobs == -1:
    #    for i, orbit in enumerate(args.orbits):
    #        print('({}) {:>5}/{:>5}: {}'.format(datetime.now().strftime(
    #            '%Y-%m-%d %H:%M:%S'), i+1, len(args.orbits), orbit, ))
    #        b = surface_processor(orbit, **kwargs)
    #        #b = surface_processor(orbit, typ=args.type,
    #        #                      ywinwidth=args.ywinwidth,
    #        #                      archive=True, gain=0, gain_altitude=True,
    #        #                      gain_sahga=True, senv=senv)

    #        if args.output is not None:
    #            # Debugging output
    #            outfile = os.path.join(args.output, "srf_{:s}.npy".format(orbit))
    #            logging.debug("Saving to " + outfile)
    #            if not os.path.exists(args.output):
    #                os.makedirs(args.output)
    #            np.save(outfile, b)

    #if args.jobs > 0:
    #    async_surface = Async(surface_processor, None, nbcores=jobs)
    #    for orbit in args.orbits:
    #        async_surface.call(orbit, **kwargs)


if __name__ == "__main__":
    # execute only if run as a script
    main()
