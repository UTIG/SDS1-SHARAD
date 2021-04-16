import logging
import numpy as np
import os
import pandas as pd

import SHARADEnv

class DataMissingException(Exception):
    pass


def srf_processor(orbit, typ='cmp', ywinwidth=[-100,100], gain=0, archive=False,
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

    out = {'y':surf_y, 'amp':surf_amp}

    if archive == True:
        archive_srf(senv, orbit_full, out, typ)

    return out


def archive_srf(senv, orbit_full, srf_data, typ):
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

