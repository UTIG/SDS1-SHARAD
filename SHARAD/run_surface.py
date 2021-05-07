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
from datetime import datetime
import logging
import multiprocessing
import numpy as np
import pandas as pd

import SHARADEnv


class DataMissingException(Exception):
    pass


class Async:
    """
    Class to support multi procesing jobs. For calling example, see:
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


def surface_processor(orbit, typ='cmp', ywinwidth=(-100, 100), archive=False,
                      gain=0, gain_altitude='grima2021', gain_sahga=True,
                      senv=None, **kwargs):
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
    save: boolean
        Whether to save the results in a txt file into the hierarchy

    Output:
    ------

    pandas dataframe with auxilliary data and surface amplitudes

    """

    if senv is None:
        senv = SHARADEnv.SHARADEnv()

    # Load data

    orbit_full = orbit if orbit.find('_') == 1 else senv.orbit_to_full(orbit)

    logging.debug('PROCESSING: Surface echo extraction for ' + orbit_full)

    alt = senv.alt_data(orbit, typ='beta5', ext='h5')
    if alt is None: # pragma: no cover
        raise DataMissingException("No Altimetry Data for orbit " + orbit)
    rdg = senv.cmp_data(orbit)
    if rdg is None: # pragma: no cover
        raise DataMissingException("No CMP data for orbit " + orbit)

   # Get surface amplitude

    alty = alt['idx_fine']
    surf_y = alty * 0
    surf_amp = alty * 0
    for i, val in enumerate(alty):
        if (not np.isfinite(val)) or (val <= 0):
            surf_y[i] = np.nan
            surf_amp[i] = np.nan
        else:
            # Pulse amplitude
            pls = np.abs(rdg[i, :])
            # Product of the pulse with its derivative
            prd = np.abs(np.roll(np.gradient(pls), 2) * pls)
            # interval within which to retrieve the surface
            val = int(val)
            itv = prd[val+ywinwidth[0]:val+ywinwidth[1]]
            if len(itv):
                maxprd = np.max(itv)
                maxind = val + ywinwidth[0] + np.argmax(itv) # The surface echo
                maxvec = pls[maxind] # The y coordinate of the surface echo
            else:
                maxprd = 0
                maxind = 0
                maxvec = 0
            surf_y[i] = maxind
            surf_amp[i] = maxvec

    # Apply gains

    total_gain = gain

    if gain_altitude:
        _gain = relative_altitude_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    if gain_sahga:
        _gain = relative_sahga_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    surf_amp = surf_amp * 10**(total_gain/20.)

    # Archiving

    out = {'y':surf_y, 'amp':surf_amp}

    if archive:
        archive_surface(senv, orbit_full, out, typ)

    return out


def relative_altitude_gain(senv, orbit_full, method='grima2012'):
    """Provide relative altitude gain for an orbit following 
    Grima et al. 2012 or Campbell et al. (2021, eq.1)
    """
    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise("No Auxiliary Data for orbit " + orbit_full)

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
        raise("No Auxiliary Data for orbit " + orbit_full)

    samxin = aux['MRO_SAMX_INNER_GIMBAL_ANGLE']
    sapxin = aux['MRO_SAPX_INNER_GIMBAL_ANGLE']
    hgaout = aux['MRO_HGA_OUTER_GIMBAL_ANGLE']

    gain = 0.0423*np.abs(samxin) + 0.0274*np.abs(sapxin) - 0.0056*np.abs(hgaout)
    gain = -gain

    return gain


def archive_surface(senv, orbit_full, srf_data, typ, **kwargs):
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
        raise("No Auxiliary Data for orbit " + orbit_full)

    columns = ['EPHEMERIS_TIME',
               'SUB_SC_PLANETOCENTRIC_LATITUDE',
               'SUB_SC_EAST_LONGITUDE',
               'SOLAR_ZENITH_ANGLE',
               'SOLAR_LONGITUDE',
               'SPACECRAFT_ALTITUDE',
               'SC_ROLL_ANGLE',
               'surf_y',
               'surf_amp',
               'surf_pow']

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
             ]

    out = pd.DataFrame(values).transpose()
    out.columns = columns

    # Archive

    #k = p['orbit_full'].index(orbit_full)
    # TODO: what is the correct response if there is more than one result?  GNG
    list_orbit_info = senv.get_orbit_info(orbit_full)
    orbit_info = list_orbit_info[0]

    if typ == 'cmp':
        archive_path = os.path.join(senv.out['srf_path'],
                                    orbit_info['relpath'], typ)
    else: # pragma: no cover
        assert False
    if not os.path.exists(archive_path):
        os.makedirs(archive_path)
    fil = os.path.join(archive_path, orbit_full + '.txt')
    out.to_csv(fil, index=None, sep=',')
    logging.info('CREATED: ' + fil)

    return out


def cb_surface_processor():
    """Callback function for surface processor
    """
    pass


def main():
    parser = argparse.ArgumentParser(description='Processing routines \
                                     for surface echo power extraction')

    #--------------------
    # Job control options

    #outpath = os.path.join(os.getenv('SDS'), 'targ/xtra/SHARAD')

    parser.add_argument('-o', '--output', default=None,
                        help="Debugging output data directory")
    #parser.add_argument('--ofmt',   default='hdf5',choices=('hdf5','none'),
    #        help="Output data format")
    parser.add_argument('orbits', metavar='orbit', nargs='+',
                        help='Orbit IDs to process (including leading zeroes). \
                        If "all", processes all orbits')
    parser.add_argument('-j','--jobs', type=int, default=8,
            help="Number of jobs (cores) to use for processing. -1 to disable\
            multiprocessing")
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
                        default=(-100, 100),
                        help='2 numbers defining the fast-time relative \
                        boundaries around the altimetry surface return where \
                        the surface will be looked for')
    parser.add_argument('-t', '--type', type=str, default='cmp',
                        help='Type of radar data used to get the amplitude \
                        from')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='Delete and reprocess files already processed, \
                        only if [orbit] is [all]')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="run_srf: [%(levelname)-7s] %(message)s")

    senv = SHARADEnv.SHARADEnv()

    #--------------------------
    # Requested Orbits handling

    available = senv.processed()['cmp'] # To convert to EDR orbit list
    processed = senv.processed()
    processable = list(set(processed['cmp']) & set(processed['alt']))
    processable_unprocessed = list(set(processable) - set(processed['srf']))

    if 'all' in args.orbits:
        requested = available
    elif '.' in args.orbits[0]: # Input is a filename
        requested = list(np.genfromtxt(args.orbits[0], dtype='str'))    
    else:
        requested = args.orbits

    if args.delete:
        args.orbits = list(set(processable) & set(requested))
    else:
        args.orbits = list(set(processable_unprocessed) & set(requested))

    args.orbits.sort()
    
    #-----------
    # Processing

    if args.dryrun: # pragma: no cover
        logging.info(f"{' '.join(args.orbits)}")
        #logging.info(f"TOTAL: {len(args.orbits)} to process")
        sys.exit(0)

    logging.info(f"TOTAL: {len(args.orbits)} to process")

    # Keyword arguments for processing
    kwargs = {'typ':args.type,
              'ywinwidth':args.ywinwidth,
              'archive':True,
              'gain':0,
              'gain_altitude':True,
              'gain_sahga':True,
              'senv':senv
             }

    # Create Async class for multiprocessing
    if args.jobs > 0:
        async_surface = Async(surface_processor, None, nbcores=args.jobs)

    # Processing
    for i, orbit in enumerate(args.orbits):
        #logging.debug('({}) {:>5}/{:>5}: {}'.format(datetime.now().strftime(
        #              '%Y-%m-%d %H:%M:%S'), i+1, len(args.orbits), orbit, ))

        # Do NOT use the multiprocessing package
        if args.jobs == -1:
            b = surface_processor(orbit, **kwargs)

        # Do use the multiprocessing package
        if args.jobs > 0:
            async_surface.call(orbit, **kwargs)

        if args.output is not None:
                # Debugging output
                outfile = os.path.join(args.output, "srf_{:s}.npy".format(orbit))
                logging.debug("Saving to " + outfile)
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                np.save(outfile, b)

    if args.jobs > 0:
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
