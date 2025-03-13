#!/usr/bin/env python3
__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']

__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'December 15, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'
                 'WARNING: This algorithm is deprecated. It is strongly'
                 'recommended to use the beta5 algorithm instead'
                 'The threshold altimetry algorithm uses pulse compressed data'
                 'for surface picking. It calculates the derivative of the '
                 'waveform and picks the first peak exceeding a threshold'}}

import sys
import os
import logging
import numpy as np
import spiceypy as spice
import pandas as pd
from scipy.constants import c
from scipy.ndimage import shift
#import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cmp.pds3lbl as pds3
from cmp.plotting import plot_radargram
from rot import mars
from misc import coord



def alt_profile(label_path, aux_path, science_path,
                cmp_track_path, sar_window,
                kernel_path='/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm',
                save_path=None, idx_start=None, idx_end=None, fix_pri=None):

    rot_model = mars.mars_rot_model('IAU2000')
    re = pd.read_hdf(cmp_track_path, key='real')
    im = pd.read_hdf(cmp_track_path, key='imag')
    #cmp_track = abs(re.values + 1j*im.values)
    cmp_track = np.hypot(re.values, im.values)

    # Set start and end point of evaluation
    # If not specified in input take the full track

    if idx_start is None:
        idx_start = 0
    idx_start = max(0, idx_start)

    if idx_end is None:
        idx_end = len(cmp_track)
    idx_end = min(len(cmp_track), idx_end)

    # Read pulse-compressed and complementing EDR data
    cmp_track = cmp_track[idx_start:idx_end]
    data = pds3.read_science(science_path, label_path)[idx_start:idx_end]
    auxdata = science_path.replace('_s.dat', '_a.dat')
    aux = pds3.read_science(auxdata, aux_path)[idx_start:idx_end]

    # Get the RWOT parameters
    range_window_start = data['RECEIVE_WINDOW_OPENING_TIME']
    r_tx0 = int(min(range_window_start))
    r_offset = int(max(range_window_start)) - r_tx0

    # S/C position
    ets = np.array(aux['EPHEMERIS_TIME'])
    sc = np.zeros(len(ets))
    scpos = np.zeros((len(ets), 6))

    # Use specified rotational model to transform into body-fixed
    i = 0
    for j in range(len(ets)):
        scpos[j], lt = spice.spkgeo(-74, ets[j], 'J2000', 4)
    sc  = np.linalg.norm(scpos[:, 0:3], axis=1)
    bfx = rot_model.r_bf(scpos[:, 0:3], t=ets)
    sph = coord.cart2sph(bfx)

    # Get parameters to align range lines
    sc_cor = np.array(2000*sc/c/0.0375E-6).astype(int)
    phase = -sc_cor+range_window_start
    tx0 = int(min(phase))
    offset = int(max(phase) - tx0)


    # Code mapping PRI codes to actual pulse repetition intervals
    pri_table = {
        1: 1428E-6, 2: 1429E-6,
        3: 1290E-6, 4: 2856E-6,
        5: 2984E-6, 6: 2580E-6
    }

    # Get shot frequency (assumed stable over full track)

    if fix_pri is None:
        pri_code = data['PULSE_REPETITION_INTERVAL'][0]
        #pri_code = data['PULSE_REPETITION_INTERVAL'].values[idx_start:idx_end]
    else:
        pri_code = fix_pri
        #np.full_like(sc, fix_pri)
    pri = pri_table.get(pri_code, 0.0)

    # Construct radargram
    radargram = np.zeros((len(data), 3600 + offset), dtype=np.double)
    for rec in range(len(data)):
        # adjust for range window start
        dat = np.pad(cmp_track[rec], (0, offset), 'constant', constant_values=0)
        radargram[rec] = shift(dat, phase[rec] - tx0, cval=0)

    # Average data
    avg = np.empty((len(data), 3600 + offset))
    for i in range(3600 + offset):
        avg[:, i] = running_mean(radargram[:, i], sar_window)

    # Build the derivative, calculate the treshold based on the
    # first 128 noise samples, and make surface pick
    delta = np.empty(len(avg))
    delta[:] = np.nan
    shift_param = phase-tx0
    for i in range(len(avg)):
        dif = np.diff(avg[i])
        # TODO: why np.sqrt(np.var()) and not np.std()?
        trsh = np.sqrt(np.var(dif[2
                                  + int(shift_param[i]):130
                                  + int(shift_param[i])])) * 4.5
        idx = np.where(dif > trsh)[0]
        if len(idx) > 0:
            idx2 = idx[np.where(idx > 128 + int(shift_param[i]))]
            if len(idx2) > 0:
                delta[i] = idx2[0]
        else:
            delta[i] = np.argmax(avg[i])

    # Compute the time-of-flight
    tx_sample = range_window_start + delta - phase + tx0
    tx = tx_sample * 0.0375E-6 + pri - 11.98E-6
    # One-way range in km
    d = tx * c / 2000
    # Elevation from Mars reference sphere / CoM
    r = sc - d
    lon = sph[:, 1]
    lat = sph[:, 0]

    if save_path is not None:
        np.save(save_path, r)

    return lat, lon, r, ets, delta - shift_param, d


def test_alt_profile():
    sds = os.getenv('SDS', '/disk/kea/SDS')
    inpath = os.path.join(sds, "orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr01xxx/edr0168901/e_0168901_001_ss19_700_a_a.dat")


    # input to process_alt()

    # TODO: function for calculating paths
    # create cmp path
    path_root_alt = '/disk/kea/SDS/targ/xtra/SHARAD/alt/'
    path_root_cmp = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
    path_root_edr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/'
    # Relative path to this file
    fname = os.path.basename(inpath)
    obn = fname[2:9] # orbit name
    # Relative directory of this file
    reldir = os.path.dirname(os.path.relpath(inpath, path_root_edr))
    logging.debug("inpath: " + inpath)
    logging.debug("reldir: " + reldir)
    logging.debug("path_root_edr: " + path_root_edr)
    cmp_path = os.path.join(path_root_cmp, reldir, 'ion',
                                fname.replace('_a.dat', '_s.h5'))
    #cmp_path = path_root_cmp+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')


    label_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
    aux_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')


    science_path = inpath.replace('_a.dat', '_s.dat')
    if not os.path.exists(cmp_path):
        cmp_path = cmp_path.replace('_s.h5', '_a.h5')
        if not os.path.exists(cmp_path):
            logging.warning(cmp_path + " does not exist")
            return

    kernel_path = '/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm'
    spice.furnsh(kernel_path)

    # GNG: Not sure what reasonable values for sar_window is.
    sar_window = 10

    #def alt_profile(label_path, aux_path, science_path,
    #            cmp_track_path, sar_window,
    #            kernel_path='/disk/kea/SDS/orig/supl/kernels/mro/mro_v01.tm',
    #            save_path=None, idx_start=None, idx_end=None):
    for sar_window in (1, 5, 10):
        save_path = None
        # save_path = None if sar_window == 1 else 'treshold_test.npy'
        alt_profile(label_path, aux_path, science_path, cmp_path, sar_window, fix_pri=1, save_path=save_path)


def running_mean(x, N):
    # Algorithm to compute a running mean
    xp = np.pad(x, (N//2, N-1-N//2), mode='edge')
    res = np.convolve(xp, np.ones((N,))/N, mode='valid')
    return res


def running_mean2(x, N):
    """ https://stackoverflow.com/questions/13728392/moving-average-or-running-mean """
    res = np.empty_like(x)
    cumsum = [0]
    for i, x in enumerate(x, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            res[i - N] = moving_ave
            #moving_aves.append(moving_ave)
    return res

def test_running_mean():

    for arraylen in (500, 501, 1000, 1001):
        for windowlen in (1, 3, 5, 7):
            # Check for centeredness
            x1 = np.arange(1, arraylen, 0.2)
            y1 = running_mean(x1, windowlen)
            x2 = np.flip(x1)
            y2 = running_mean(x2, windowlen)

            try:
                assert np.max(np.abs(np.flip(y2) - y1)) < 1e-5
            except AssertionError:
                print("Failed with arraylen={:d}, windowlen={:d}".format(arraylen, windowlen))
                raise

    arraylen = 11
    windowlen = 53
    x1 = np.arange(1, arraylen, 0.2)
    y1 = running_mean(x1, windowlen)
    y2 = running_mean2(x1, windowlen)

    #print('y1 ', y1)
    #print('y2 ', y2)
    # They're not equivalent. GNG
    #assert (y1 == y2).all()


def main():
    test_running_mean()
    test_alt_profile()

    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()

