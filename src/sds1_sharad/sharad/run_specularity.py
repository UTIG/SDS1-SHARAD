#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm to run specularity assessment on already processed SHARAD orbits.
Which orbits are to be analyzed are passed using a text file.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '../xlib')
import cmp.pds3lbl as pds3

SDS = os.getenv('SDS', '/disk/kea/SDS')

def snr(data, noise_window=250):
    '''
    converting radargram voltages to SNR powers

    Inputs:
    ------------
         data: radargram

    Output:
    ------------
        radargram in SNR
    '''

    out = np.zeros((len(data), np.size(data, axis=1)), dtype=float)
    for jj in range(len(data)):
        noise = np.sqrt(np.mean(np.abs(data[jj, 0:noise_window])**2))
        out[jj, :] = np.divide(np.abs(data[jj, :]), noise)

    out = 20 * np.log10(out)

    return out

# ***************************************************************************
# INPUTS
file = 'Western_Alba_Mons_Path_Subset.txt'
short = {'interp_dx': 5,
         'column_interval': 5,
         'aperture': 6}     # dictionary of short aperture SAR parameters
long = {'interp_dx': 5,
        'column_interval': 5,
        'aperture': 40}     # dictionary of long aperture SAR parameters
max_surf_diff = 5 # maximum allowable difference between short sar and long sar surface picks
debug = False
# ***************************************************************************


# define files to be read in
with open(file) as fil:
    a = fil.readlines()

# extract specularity for each line
ii = 0
if ii == 0:

    # set the path to the data
    rawfn = a[ii].replace('\n', '')
    temp = rawfn.split('/')
    print('Working: ' + temp[-1])
    base = (
            Path(SDS) / 'targ/xtra/SHARAD/foc' /
            temp[-5] / temp[-4] / temp[-3] / temp[-2]
    )
    short_fn = (
        base / (str(short['interp_dx']) + 'm') /
        (str(short['column_interval']) + ' range lines') /
        (str(short['aperture']) + 'km') /
        (temp[-1].replace('a.dat', 's.h5')
    )
    long_fn = (base /
        (str(long['interp_dx']) + 'm') /
        (str(long['column_interval']) + ' range lines)' /
        (str(long['aperture']) + 'km') /
        (temp[-1].replace('a.dat', 's.h5')
    )

    # load the radar data and interpolated ephemeris times
    short_sar = np.array(pd.read_hdf(short_fn, 'sar'))
    short_et = np.array(pd.read_hdf(short_fn, 'interpolated_ephemeris'))
    short_col = np.array(pd.read_hdf(short_fn, 'columns'))
    long_sar = np.array(pd.read_hdf(long_fn, 'sar'))
    long_et = np.array(pd.read_hdf(long_fn, 'interpolated_ephemeris'))
    long_col = np.array(pd.read_hdf(long_fn, 'columns'))

    # convert amplitudes to SNR
    short_sar = snr(short_sar)
    short_et = short_et[short_col[:, 1]]
    long_sar = snr(long_sar)
    long_et = long_et[long_col[:, 1]]
    del short_col, long_col

    # trim long aperture to match short aperture
    ind = np.zeros(len(long_et), dtype=int)
    for ii in range(len(ind)):
        ind[ii] = np.argwhere(long_et[ii] == short_et)[0][0]
    et = long_et
    short_sar = short_sar[ind, :]

    if debug:
        plt.figure()
        plt.subplot(311)
        plt.plot(short_et, '.r', label=str(short['aperture']) + 'km aperture')
        plt.plot(long_et, '.b', label=str(long['aperture'])+ 'km aperture')
        plt.plot(et, '.g', label='common et')
        plt.legend()
        plt.title('Interpolated Ephemeris Times')
        plt.subplot(312)
        plt.imshow(np.transpose(short_sar[:, 0:1500]), aspect='auto', cmap='seismic')
        plt.clim([0, 30])
        plt.title(str(short['aperture']) + 'km Aperture')
        plt.subplot(313)
        plt.imshow(np.transpose(long_sar[:, 0:1500]), aspect='auto', cmap='seismic')
        plt.clim([0, 30])
        plt.title(str(long['aperture']) + 'km Aperture')
        plt.show()
    
    # load unedited ephemeris, latitude, and longitude information
    aux_lbl = str(Path(SDS) / 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')
    aux = pds3.read_science(rawfn, aux_lbl)
    raw_et = aux['EPHEMERIS_TIME']
    raw_lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE']
    raw_lon = aux['SUB_SC_EAST_LONGITUDE']

    # interpolate raw latitude and longitude onto already interpolated ephemeris
    latitude = np.interp(et, raw_et, raw_lat)
    longitude = np.interp(et, raw_et, raw_lon)

    if debug:
        plt.figure()
        plt.subplot(211)
        plt.plot(raw_et, raw_lat, '-r', label='raw')
        plt.plot(et, latitude, '-b', label='focused')
        plt.legend()
        plt.xlabel('Ephemeris Time [s]')
        plt.ylabel('Latitude [deg]')
        plt.subplot(212)
        plt.plot(raw_et, raw_lon, '-r', label='raw')
        plt.plot(et, longitude, '-b', label='focused')
        plt.legend()
        plt.xlabel('Ephemeris Time [s]')
        plt.ylabel('Longitude [deg]')
        plt.show()
    del raw_et, raw_lat, raw_lon, aux

    # extract surface echo power from short and long aperture results
    long_surf = np.zeros(len(et))
    short_surf = np.zeros(len(et))
    for jj in range(len(et)):
        long_surf_ind = np.argmax(long_sar[jj, 0:1500])
        short_surf_ind = np.argmax(short_sar[jj, 0:1500])
        if np.abs(long_surf_ind - short_surf_ind) <= max_surf_diff:
            long_surf[jj] = long_surf_ind
            short_surf[jj] = short_surf_ind
        else:
            long_surf[jj] = np.nan
            short_surf[jj] = np.nan

    if debug:
        print('-- ' + str(len(et)) + ' range lines')
        print('-- ' + str(len(et) - np.sum(np.isnan(long_surf))) + ' range lines with picked surface')
        plt.figure()
        plt.subplot(311)
        plt.plot(et, long_surf, '.r', label=str(short['aperture']) + 'km aperture')
        plt.plot(et, short_surf, '.b', label=str(long['aperture'])+ 'km aperture')
        plt.legend()
        #plt.hist(long_surf - short_surf, bins=400)
        plt.title('Max SNR Index')
        plt.subplot(312)
        plt.imshow(np.transpose(short_sar[:, 0:1500]), aspect='auto', cmap='seismic')
        plt.plot(short_surf, '.g')
        plt.clim([0, 30])
        plt.title(str(short['aperture']) + 'km Aperture')
        plt.subplot(313)
        plt.imshow(np.transpose(long_sar[:, 0:1500]), aspect='auto', cmap='seismic')
        plt.plot(long_surf, '.g')
        plt.clim([0, 30])
        plt.title(str(long['aperture']) + 'km Aperture')
        plt.show()
        
    # extract surface amplitudes at positions with acceptable surface
    # picks and calculate specularity
    short_sar_snr = np.zeros(len(et))
    long_sar_snr = np.zeros(len(et))
    for jj in range(len(et)):
        if not np.isnan(short_surf[jj]):
            short_sar_snr[jj] = short_sar[jj, int(short_surf[jj])]
        else:
            short_sar_snr[jj] = np.nan
        if not np.isnan(long_surf[jj]):
            long_sar_snr[jj] = long_sar[jj, int(short_surf[jj])]
        else:
            long_sar_snr[jj] = np.nan
    specularity = np.divide(short_sar_snr, long_sar_snr)

    if True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Ephemeris Time [s]')
        ax1.set_ylabel('SNR [dB]')
        ax1.plot(et, short_sar_snr, color='r', label='short')
        ax1.plot(et, long_sar_snr, color='b', label='long')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Specularity')
        ax2.plot(et, specularity, color='k', linewidth=2)
        fig.tight_layout()
        plt.show()




