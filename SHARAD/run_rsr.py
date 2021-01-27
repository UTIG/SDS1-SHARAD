#!/usr/bin/env python3

"""
Example command:

To process a single orbit (0887601) and archive to the default location:

./run_rsr.py 0887601

To process a single orbit (0887601), not archive to the default location (--ofmt
none),
and output a numpy file into a debug directory:

./run_rsr.py 0887601 --ofmt none --output ./rsr_data

"""

__authors__ = ['Cyril Grima, cyril.grima@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'March 13 2019',
         'author': 'Cyril Grima, UTIG'}}

import logging
import argparse
import multiprocessing
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import SHARADEnv

sys.path.append('../xlib/')
import rsr
import subradar as sr

class DataMissingException(Exception):
    pass

def surface_amp(senv, orbit, typ='cmp', ywinwidth=[-100,100], gain=0, sav=True,
        verbose=True, **kwargs):
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
    save: boolean
        Whether to save the results in a txt file into the hierarchy

    Output:
    ------

    pandas dataframe with auxilliary data and surface amplitudes

    """

    # Load data

    orbit_full = orbit if orbit.find('_') == 1 else senv.orbit_to_full(orbit)

    logging.debug('PROCESSING: Surface echo extraction for ' + orbit_full)

    alt = senv.alt_data(orbit, typ='beta5', ext='h5')
    if alt is None: # pragma: no cover
        raise DataMissingException("No Altimetry Data for orbit " + orbit)
    rdg = senv.cmp_data(orbit)
    if rdg is None: # pragma: no cover
        raise DataMissingException("No CMP data for orbit " + orbit)
    aux = senv.aux_data(orbit)
    if aux is None: # pragma: no cover
        raise("No Auxiliary Data for orbit " + orbit)

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

    # Archive
    # TODO: def archive_surface_amp(senv, orbit, srf_data)

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
              surf_y,
              surf_amp,
              20*np.log10(surf_amp),
             ]
    
    out = pd.DataFrame(values).transpose()
    out.columns = columns

    if sav is True:
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


def rsr_processor(orbit, typ='cmp', gain=0, sav=True,
    senv=None, **kwargs):
    """
    Output the results from the Radar Statistical Reconnaissance Technique
    applied along a SHARAD orbit

    Inputs
    -----

    orbit: string
        the orbit number or the full name of the orbit file (w/o extension)
        if te orbit is trunl=ked in several file
    typ: string
        the type of radar data used to get the amplitude from
    gain: float
        Any gain to be added to the signal (power in dB)
        For SHARAD, it includes the instrumental gain and the absolute
        calibration value
    save: boolean
        Whether to save the results in a txt file into the hierarchy
    senv: SHARADEnv
        A SHARADEnv environment object
    Any keywords from rsr.utils.inline_estim, especially:

    fit_model : string
        pdf to use to estimate the histogram statistics (inherited from rsr.fit)
    bins : string
        method to compute the bin width (inherited from astroML.plotting.hist)
    inv : string
        inversion method (inherited from rsr.invert)
    winsize : int
        number of amplitude values within a window
    sampling : int
        window repeat rate
    verbose : bool
        Display fit results informations

    Output
    ------
    Results are gathered in a pandas Dataframe that inherits from some of the
    auxilliary data plus the following columns:
    xa: First x-coordinate of the window gathering the considered amplitudes
    xb: Last x-coordinate of the window gathering the considered amplitudes
    xo: Middle x-coordinate of the window gathering the considered amplitudes
    pt: Total power received at the antenna
    pc: Coherent power received at the antenna
    pn: Incoherent power received at the antenna
    crl: Coefficient correlation of the RSR fit
    chisqr: Chi-square of the RSR fit
    mu: HK structure parameter
    ok:  Whether the RSR fit converged correctly (1) or not (0)
    """

    if senv is None:
        senv = SHARADEnv.SHARADEnv()

    # This should be done in aux_data?
    orbit_full = orbit if orbit.find('_') == 1 else senv.orbit_to_full(orbit)

    # Surface amplitudes
    surf = surface_amp(senv, orbit, gain=gain, **kwargs)

    # Surface coefficients (RSR)
    logging.debug('PROCESSING: Surface Statistical Reconnaissance for '
            + orbit_full)
    b = rsr.run.along(surf['surf_amp'].values, **kwargs)

    # Reformat results
    xo = b['xo'].values.astype(int)# along-track frame number
    for key in surf.keys():
        b[key] = surf[key].to_numpy()[xo]

    b = b.rename(index=str, columns={"flag":"ok"})

    # Work-around for pandas' bug "ValueError: Big-endian buffer not supported
    # on little-endian compiler"
    b = pd.DataFrame(np.array(b).byteswap().newbyteorder(), columns=b.keys())

    # Archive
    # TODO: make this a separate function (should it be a member of SHARADEnv?)
    # archive_rsr(senv, orbit, rsr_data)
    if sav is True:
        # TODO: change this to use b_single
        # orbit_info = senv.get_orbit_info(orbit_full, True)
        list_orbit_info = senv.get_orbit_info(orbit_full)
        orbit_info = list_orbit_info[0]
        if typ == 'cmp':
            archive_path = os.path.join(senv.out['rsr_path'],
                    orbit_info['relpath'], typ)
        else: # pragma: no cover
            assert False
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
        fil = os.path.join(archive_path,  orbit_full + '.txt')
        b.to_csv(fil, index=None, sep=',')
        #print("CREATED: " + fil)
        logging.debug("CREATED: " + fil )

    return b


def todo(delete=False, senv=None, filename=None, verbose=False):
    """List the orbits that are not already RSR-processed to be
    processed but for which an altimetry file exists

    Inputs
    ------

    delete: boolean
        If True, delete existing RSR products and re-process.
        If False, only generate RSR products that do not exist.

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
    rsr_orbits = []
    for orbit in senv.orbitinfo:
        for suborbit in senv.orbitinfo[orbit]:
            try:
                if any(s.endswith('.h5') for s in suborbit['altpath']):
                    alt_orbits.append(suborbit['name'])
                if any(s.endswith('.txt') for s in suborbit['rsrpath']):
                    rsr_orbits.append(suborbit['name'])
            except KeyError:
                pass

    # Available orbits
    if filename is not None:
        fil_orbits = list(np.genfromtxt(filename, dtype='str'))
        available_orbits = [i for i in fil_orbits if i in alt_orbits]
    else:
        available_orbits = alt_orbits

    # Unprocessed orbits
    unprocessed_orbits = [i for i in available_orbits if i not in rsr_orbits]

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
    parser.add_argument(     '--ofmt',   default='hdf5',choices=('hdf5','none'),
            help="Output data format")
    parser.add_argument('orbits', metavar='orbit', nargs='+',
            help='Orbit IDs to process (including leading zeroes). If "all",\
            processes all orbits')
    parser.add_argument('-j','--jobs', type=int, default=8,
            help="Number of jobs (cores) to use for processing. -1 to disable\
            multiprocessing")
    parser.add_argument('-v','--verbose', action="store_true",
            help="Display verbose output")
    parser.add_argument('-n','--dryrun', action="store_true",
            help="Dry run. Build task list but do not run")
    #parser.add_argument('--tracklist', default="elysium.txt",
    #    help="List of tracks to process")
    #parser.add_argument('--maxtracks', default=None, type=int,
    #    help="Max number of tracks to process")

    # Algorithm options

    parser.add_argument('-w', '--winsize', type=int, default=1000,
            help='Number of consecutive echoes within a window where statistics\
            are determined')
    parser.add_argument('-s', '--sampling', type=int, default=250,
            help='Step at which a window is repeated')
    parser.add_argument('-y', '--ywinwidth', nargs='+', type=int, default=[-100,100],
            help='2 numbers defining the fast-time relative boundaries around\
            the altimetry surface return where the surface will be looked for')
    parser.add_argument('-b', '--bins', type=str, default='fd',
            help='Method to compute the bin width (inherited from numpy.histogram)')
    parser.add_argument('-f', '--fit_model', type=str, default='hk',
            help='Name of the function (in pdf module) to use for the fit')
    parser.add_argument('-d', '--delete', action='store_true',
            help='Delete and reprocess files already processed, only if [orbit] is [all]')

    args = parser.parse_args()

    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_rsr: [%(levelname)-7s] %(message)s")

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

    for i, orbit in enumerate(args.orbits):
        print('({}) {:>5}/{:>5}: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i+1, len(args.orbits), orbit, ))
        #print(str(str(i)+'/'+len(orbit))+ ': ' + orbit)
        b = rsr_processor(orbit, winsize=args.winsize, sampling=args.sampling,
                nbcores=args.jobs, verbose=args.verbose, winwidht=args.ywinwidth,
                bins=args.bins, fit_model=args.fit_model, sav=(args.ofmt == 'hdf5'),
                senv=senv)

        if args.output is not None:
            # Debugging output
            outfile = os.path.join(args.output, "rsr_{:s}.npy".format(orbit))
            logging.debug("Saving to " + outfile)
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            np.save(outfile, b)



if __name__ == "__main__":
    # execute only if run as a script
    main()




