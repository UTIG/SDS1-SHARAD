#!/usr/bin/env python3

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

import numpy as np
import pandas as pd

import SHARADEnv

sys.path.append('../xlib/')
import rsr
import subradar as sr


def surface_amp(senv, orbit, typ='cmp', winwidth=[-6,7], gain=0, sav=True, verbose=True, **kwargs):
    """
    Get the maximum of amplitude*(d amplitude/dt) within bounds defined by the altimetry processor

    Input:
    -----

    orbit: string
        the orbit number or the full name of the orbit file (w/o extension)
        if the orbit is truncated in several file
    typ: string
        The type of radar data used to get the amplitude from
    gain: float
        Any gain to be added to the signal (power in dB)
        For SHARAD, it includes the instrumental gain and theabsolute calibration value
    save: boolean
        Whether to save the results in a txt file into the hierarchy

    Output:
    ------

    amp: Surface amplitudes

    """

    #----------
    # Load data
    #----------

    orbit_full = orbit if orbit.find('_') is 1 else senv.orbit_to_full(orbit)

    if verbose is True:
        print('PROCESSING: Surface echo extraction for ' + orbit_full)

    alt = senv.alt_data(orbit, typ='beta5', ext='h5')
    rdg = senv.cmp_data(orbit)
    aux = senv.aux_data(orbit)

    et = aux['EPHEMERIS_TIME']
    lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']
    lon = aux['SUB_SC_EAST_LONGITUDE']
    rng = aux['SPACECRAFT_ALTITUDE']
    roll = aux['SC_ROLL_ANGLE']

    #----------------------
    # Get surface amplitude
    #----------------------

    alty = alt['idx_fine']
    y = alty * 0
    amp = alty * 0
    for i, val in enumerate(alty):
        if np.isfinite(val) == False:
            y[i] = np.nan
            amp[i] = np.nan
        else:
            # Pulse amplitude
            pls = np.abs(rdg[i, :])
            # Product of the pulse with its derivative
            prd = np.abs(np.roll(np.gradient(pls), 2) * pls)
            # interval within which to retrieve the surface
            val = int(val)
            itv = prd[val+winwidth[0]:val+winwidth[1]]
            if len(itv):
                maxprd = np.max(itv)
                maxind = val - 2 + np.argmax(itv) # The value of the surface echo
                maxvec = pls[maxind] # The y coordinate of the surface echo
            else:
                maxprd = 0
                maxind = 0
                maxvec = 0
            y[i] = maxind
            amp[i] = maxvec

    #--------
    # Archive
    #--------

    out = {'et':et, 'lat':lat, 'lon':lon, 'rng':rng, 'roll':roll, 'y':y, 'amp':amp}
    out = pd.DataFrame(data=out)
    #out = out.reindex(columns=['utc', 'lat', 'lon', 'rng', 'roll', 'y', 'amp'])
    #out = out[['utc', 'lat', 'lon', 'rng', 'roll', 'y', 'amp']]

    if sav is True:
        #k = p['orbit_full'].index(orbit_full)
        list_orbit_info = senv.get_orbit_info(orbit_full)
        orbit_info = list_orbit_info[0]


        if typ is 'cmp':
            archive_path = os.path.join(senv.out['srf_path'], orbit_info['relpath'], typ)
        else:
            assert(False)
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
        fil = os.path.join(archive_path,  orbit_full + '.txt')
        out.to_csv(fil, index=None, sep=',')
        #print("CREATED: " + fil )

    return out


def rsr_processor(orbit, typ='cmp', gain=-210.57, sav=True, verbose=True, **kwargs):
    """
    Output the results from the Radar Statistical Reconnaissance Technique applied along
    a SHARAD orbit

    Inputs
    -----

    orbit: string
        the orbit number or the full name of the orbit file (w/o extension)
        if te orbit is trunl=ked in several file
    typ: string
        the type of radar data used to get the amplitude from
    gain: float
        Any gain to be added to the signal (power in dB)
        For SHARAD, it includes the instrumental gain and theabsolute calibration value
    save: boolean
        Whether to save the results in a txt file into the hierarchy

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
    Results are gathered in a pandas Dataframe that includes the following columns:
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
    rng: Range to surface [m]
    lc: Coherent geometric losses [dB power]
    ln: Incoherent geometric losses [dB power]
    rc: Surface reflection coefficient [dB power]
    rn: Surface scattering coefficient [dB power]
    """

    #----------
    # Load data
    #----------

    senv = SHARADEnv.SHARADEnv()
    orbit_full = orbit if orbit.find('_') is 1 else senv.orbit_to_full(orbit)

    aux = senv.aux_data(orbit_full)
    utc = aux['EPHEMERIS_TIME']
    lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']
    lon = aux['SUB_SC_EAST_LONGITUDE']
    rng = aux['SPACECRAFT_ALTITUDE']
    roll = aux['SC_ROLL_ANGLE']

    #----------------------
    # Get surface amplitude
    #----------------------

    surf = surface_amp(senv, orbit, **kwargs)
    amp = surf['amp'].values

    #-------------------------------
    # Get surface coefficients (RSR)
    #-------------------------------

    if verbose is True:
        print('PROCESSING: Surface Statistical Reconnaissance for ' + orbit_full)

    # Amplitude with gain and 2-way coherent geometric losses
    # If Geo losses are not applied, the amplitudes would be << 1 and
    # the RSR fitting would fail. Geo losses are removed after processing.
    Lc = 10*np.log10( sr.utils.geo_loss(2*rng*1e3)   )
    pdb = 20*np.log10(amp) + gain - Lc
    amp2 = 10**(pdb/20)
    # RSR process
    #b = rsr.utils.inline_estim(amp2, frq=20e6, **kwargs)
    b = rsr.run.along(amp2, **kwargs)
    # Geometric losses
    b['rng'] = rng[ b['xo'].values.astype(int)  ]*1e3
    b['lc'] = 10*np.log10(sr.utils.geo_loss(2*b['rng'].values))
    b['ln'] = 10*np.log10(sr.utils.geo_loss(b['rng'].values)**2)
    # Remove pre-added coherent geometric losses on received powers
    b['pt'] = b['pt'].values + b['lc'].values
    b['pc'] = b['pc'].values + b['lc'].values
    b['pn'] = b['pn'].values + b['lc'].values
    # Pulse-limited footprint surface area
    b['as'] = 10*np.log10( np.pi * sr.utils.footprint_rad_pulse(b['rng'].values, 10e6)**2 )
    # Surface coefficients
    b['rc'] = b['pc'].values - b['lc'].values
    b['rn'] = b['pn'].values - b['as'].values - b['ln'].values
    # reformat/clean results
    b['utc'] = utc[ b['xo'].values.astype(int) ]
    b['lon'] = lon[ b['xo'].values.astype(int) ]
    b['lat'] = lat[ b['xo'].values.astype(int) ]
    b['roll'] = roll[ b['xo'].values.astype(int) ]
    b = b.rename(index=str, columns={"flag":"ok"})
    #b = b.drop(columns=['as', 'eps', 'sh'])
    #b = b[['utc', 'lat', 'lon', 'rng', 'roll', 'xa', 'xo', 'xb', 'pt', 'pc', 'pn', 'mu', 'lc', 'ln', 'rc', 'rn', 'crl', 'chsqr', 'ok']]

    #--------
    # Archive
    #--------

    if sav is True:
        #k = p['orbit_full'].index(orbit_full)
        list_orbit_info = senv.get_orbit_info(orbit_full)
        orbit_info = list_orbit_info[0]
        if typ is 'cmp':
            archive_path = os.path.join(senv.out['rsr_path'], orbit_info['relpath'], typ)
        else:
            assert(False)
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
        fil = os.path.join(archive_path,  orbit_full + '.txt')
        b.to_csv(fil, index=None, sep=',')
        if verbose is True:
            print("CREATED: " + fil )

    return b



def main():
    parser = argparse.ArgumentParser(description='RSR processing routines')
    #parser.add_argument('-o','--output', default='', help="Output directory")
    #parser.add_argument(     '--ofmt',   default='npy',choices=('hdf5','npy','none'), help="Output data format")
    parser.add_argument('orbit', help='Orbit number (including leading zeroes)')
    parser.add_argument('-j','--jobs', type=int, default=8, help="Number of jobs (cores) to use for processing")
    parser.add_argument('-v','--verbose', action="store_true", help="Display verbose output")
    #parser.add_argument('-n','--dryrun', action="store_true", help="Dry run. Build task list but do not run")
    #parser.add_argument('--tracklist', default="elysium.txt",
    #    help="List of tracks to process")
    #parser.add_argument('--maxtracks', default=None, type=int, help="Max number of tracks to process")
    parser.add_argument('-w', '--winsize', type=int, default=1000, help='Number of consecutive echoes within a window where statistics are determined')
    parser.add_argument('-s', '--sampling', type=int, default=250, help='Step at which a window is repeated')
    parser.add_argument('-y', '--ywinwidth', nargs='+', type=int, default=[-6,7], help='2 numbers defining the fast-time relative boundaries around the altimetry surface return where the surface will be looked for')
    parser.add_argument('-b', '--bins', type=str, default='fd', help='Method to compute the bin width (inherited from numpy.histogram)')
    parser.add_argument('-f', '--fit_model', type=str, default='hk', help='Name of the function (in pdf module) to use for the fit')

    args = parser.parse_args()

    loglevel=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout,
        format="run_rsr: [%(levelname)-7s] %(message)s")

    b = rsr_processor(args.orbit, winsize=args.winsize, sampling=args.sampling, nbcores=args.jobs, verbose=args.verbose, winwidht=args.ywinwidth, bins=args.bins, fit_model=args.fit_model, sav=True)

    #if args.output != "":
        # TODO: improve naming
    #    outfile = os.path.join(args.output, "rsr.npy")
    #    logging.debug("Saving to " + outfile)
    #    np.save(outfile, b)

if __name__ == "__main__":
    # execute only if run as a script
    main()




