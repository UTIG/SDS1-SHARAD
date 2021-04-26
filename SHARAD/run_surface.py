import argparse
from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
import sys

import SHARADEnv

class DataMissingException(Exception):
    pass


def surface_processor(orbit, typ='cmp', ywinwidth=[-100,100], archive=False,
        gain=0, gain_altitude=True, gain_sahga=True,
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

    if gain_altitude == True:
        _gain = relative_altitude_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    if gain_sahga == True:
        _gain = relative_sahga_gain(senv, orbit_full)
        total_gain = total_gain + _gain

    surf_amp = surf_amp * 10**(total_gain/20.)

    # Archiving

    out = {'y':surf_y, 'amp':surf_amp}

    if archive == True:
        archive_surface(senv, orbit_full, out, typ)

    return out


def relative_altitude_gain(senv, orbit_full):
    """Provide relative altitude gain for an orbit following Campbell et al. (2021, eq.1)
    """
    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise("No Auxiliary Data for orbit " + orbit_full)

    lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']

    lat = lat/90 # Normalization between -1 and 1 (Bruce's 2021/04/23 email )

    gain = -.41 -1.62*lat -1.10*lat**2 +.65*lat**3 +.66*lat**4  #dB

    return gain


def relative_sahga_gain(senv, orbit_full):
    """ Provide relative SA and HGA gain for an orbit following Campbell et al. (2021, eq.4)
    """
    aux = senv.aux_data(orbit_full)
    if aux is None: # pragma: no cover
        raise("No Auxiliary Data for orbit " + orbit_full)
    
    SAMXin = aux['MRO_SAMX_INNER_GIMBAL_ANGLE']
    SAPXin = aux['MRO_SAPX_INNER_GIMBAL_ANGLE']
    HGAout = aux['MRO_HGA_OUTER_GIMBAL_ANGLE']

    gain = 0.0423*np.abs(SAMXin) + 0.0274*np.abs(SAPXin) - 0.0056*np.abs(HGAout)

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
    fil = os.path.join(archive_path,  orbit_full + '.txt')
    out.to_csv(fil, index=None, sep=',')
    #print("CREATED: " + fil )

    return out


def todo(delete=False, senv=None, filename=None, verbose=False):
    """List the orbits that are not already srf-processed to be
    processed but for which a cmp file exists

    Inputs
    ------

    delete: boolean
        If True, delete existing SRF products and re-process.
        If False, only generate SRF products that do not exist.

    filename: string
        specific list of orbits to process (optional)

    Output
    ------

    array of orbits
    """

    if senv is None:
        senv = SHARADEnv.SHARADEnv()

    # Numpy errors
    errlevel = 'warn' if verbose else 'ignore'
    np.seterr(all=errlevel)

    # Existing orbits
    alt_orbits = []
    srf_orbits = []
    for orbit in senv.orbitinfo:
        for suborbit in senv.orbitinfo[orbit]:
            try:
                if any(s.endswith('.h5') for s in suborbit['altpath']):
                    alt_orbits.append(suborbit['name'])
                if any(s.endswith('.txt') for s in suborbit['srfpath']):
                    srf_orbits.append(suborbit['name'])
            except KeyError:
                pass

   # Available orbits
    if filename is not None:
        fil_orbits = list(np.genfromtxt(filename, dtype='str'))
        available_orbits = [i for i in fil_orbits if i in alt_orbits]
    else:
        available_orbits = alt_orbits

    # Unprocessed orbits
    unprocessed_orbits = [i for i in available_orbits if i not in srf_orbits]

    if delete is True:
        out = available_orbits
    else:
        out = unprocessed_orbits
    out.sort()

    # Remove bad altimetry orbits
    if os.path.isfile('bad_alt.txt'):
        bad_orbits = list(np.genfromtxt('bad_alt.txt', dtype='str'))
        out = [i for i in out if i not in bad_orbits]

    #print(str(len(out)) + ' orbits to processed')

    return out


def main():
    parser = argparse.ArgumentParser(description='RSR processing routines')

    # Job control options

    #outpath = os.path.join(os.getenv('SDS'), 'targ/xtra/SHARAD')

    parser.add_argument('-o','--output', default=None,
            help="Debugging output data directory")
    #parser.add_argument(     '--ofmt',   default='hdf5',choices=('hdf5','none'),
    #        help="Output data format")
    parser.add_argument('orbits', metavar='orbit', nargs='+',
            help='Orbit IDs to process (including leading zeroes). If "all",\
            processes all orbits')
    parser.add_argument('-v','--verbose', action="store_true",
            help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true",
            help="Dry run. Build task list but do not run")
    #parser.add_argument('--tracklist', default="elysium.txt",
    #    help="List of tracks to process")
    #parser.add_argument('--maxtracks', default=None, type=int,
    #    help="Max number of tracks to process")

    # Algorithm options

    parser.add_argument('-y', '--ywinwidth', nargs='+', type=int, default=[-100,100],
            help='2 numbers defining the fast-time relative boundaries around\
            the altimetry surface return where the surface will be looked for')
    parser.add_argument('-t', '--type', type=str, default='cmp',
            help='Type of radar data used to get the amplitude from')
    parser.add_argument('-d', '--delete', action='store_true',
            help='Delete and reprocess files already processed, only if [orbit] is [all]')

    args = parser.parse_args()

    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_srf: [%(levelname)-7s] %(message)s")

    senv = SHARADEnv.SHARADEnv()


    if 'all' in args.orbits:
        # Uses todo to define orbits to process
        assert len(args.orbits) == 1
        args.orbits = todo(delete=args.delete, senv=senv)

    if '.' in args.orbits[0]:
        # Use a file to define orbits to process
        args.orbits = todo(delete=args.delete, senv=senv, filename=args.orbits[0])

    if args.dryrun: # pragma: no cover
        logging.info("Process orbits: " + ' '.join(args.orbits))
        sys.exit(0)

    # TODO: implement multiprocessing
    for i, orbit in enumerate(args.orbits):
        print('({}) {:>5}/{:>5}: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i+1, len(args.orbits), orbit, ))
        #print(str(str(i)+'/'+len(orbit))+ ': ' + orbit)
        b = surface_processor(orbit, typ=args.type, ywinwidth=args.ywinwidth, archive=True,
        gain=0, gain_altitude=False, gain_sahga=False,
        senv=senv)

        if args.output is not None:
            # Debugging output
            outfile = os.path.join(args.output, "srf_{:s}.npy".format(orbit))
            logging.debug("Saving to " + outfile)
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            np.save(outfile, b)


if __name__ == "__main__":
    # execute only if run as a script
    main()



