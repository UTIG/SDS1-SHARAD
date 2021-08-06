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


def rsr_processor(orbit, typ='cmp', gain=0, sav=True,
    output_root=None,
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
        Display fit results information

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
    surf = pd.DataFrame(senv.srf_data(orbit_full))

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
        assert typ == 'cmp'
        if output_root is None:
            output_root = senv.out['rsr_path']
        archive_path = os.path.join(output_root,
                orbit_info['relpath'], typ)
        fil = os.path.join(archive_path,  orbit_full + '.txt')
        save_text(b, fil)

    return b


def save_text(b, output_filename):
    """ Given a pandas dataframe, save it to the specified location """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    b.to_csv(output_filename, index=None, sep=',')
    logging.debug("Wrote text output to %s", output_filename)


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
    srf_orbits = []
    rsr_orbits = []
    for orbit in senv.orbitinfo:
        for suborbit in senv.orbitinfo[orbit]:
            try:
                if any(s.endswith('.txt') for s in suborbit['srfpath']):
                    srf_orbits.append(suborbit['name'])
                if any(s.endswith('.txt') for s in suborbit['rsrpath']):
                    rsr_orbits.append(suborbit['name'])
            except KeyError:
                pass

    # Available orbits
    if filename is not None:
        fil_orbits = list(np.genfromtxt(filename, dtype='str'))
        available_orbits = [i for i in fil_orbits if i in srf_orbits]
    else:
        available_orbits = srf_orbits

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

    #print(str(len(out)) + ' orbits to process')

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
                output_root=args.output,
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




