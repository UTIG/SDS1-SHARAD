#!/usr/bin/env python3

import time
import os
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.ndimage import shift
import spiceypy as spice

import cmp.pds3lbl as pds3
import cmp.plotting as plotting
from misc import prog

def running_mean(x, N):
    xp = np.pad(x, (N//2, N-1-N//2), mode='edge')
    res = np.convolve(xp, np.ones((N,))/N, mode='valid')
    return res

def sar_processor(rec):
    # TODO: are global variables really the right thing to do?  -- GNG
    global cmp_track
    global sc_pos
    global range_window_start
    global pri
    global topo

    rec = int(rec)
    r_0 = sc_pos[rec]
    r_alt = r_0 - r_0/spice.vnorm(r_0)*topo[rec]

    tx0=range_window_start[rec]

    et = np.zeros(corr_window + 1)
    phase = np.zeros(corr_window + 1)
    signal = np.zeros((corr_window + 1, 3600), dtype=complex)
    alongtrack = np.zeros(corr_window + 1)
    record = np.zeros((corr_window + 1, 3600))
    delta = np.arange(0, 3600)
    j = 0
    for n in range(int(rec-corr_window/2), int(rec+corr_window/2+1)):
        if n >= 0 and n < len(cmp_track):
            tx = range_window_start[n]
            #Compute phase shift
            r_sc = r_0 - sc_pos[n]
            r_sl = r_alt + r_sc
            dn = spice.vnorm(r_sl) - spice.vnorm(r_alt)
            d = abs(dn)*1000/c/0.0375E-6
            phase[j] = d - tx + tx0
            for delta in range(0, 3600):
                sample = int(delta + phase[j])
                if sample >= 0 and sample < 3600:
                    signal[n - int(rec - corr_window/2), delta] = cmp_track[n, sample]
            j += 1
    #plt.plot(phase)
    #plt.show()
    doppler = np.zeros((11, 3600))
    for delta in range(0, 3600):
        doppler[0:6, delta] = abs(np.fft.fft(signal[:, delta])[0:6])
        doppler[6:11, delta] = abs(np.fft.fft(signal[:, delta])[-5:])

    return rec,doppler,tx0

def main():

    corr_window = 1024
    sds = os.getenv('SDS', '/disk/kea/SDS')

    aux_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')
    label_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
    science_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0003/data/edr28xxx/edr2821503/e_2821503_001_ss4_700_a_s.dat')
    cmp_track = np.load(os.path.join(sds, 'targ/xtra/SHARAD/cmp/mrosh_0003/data/edr28xxx/edr2821503/ion/e_2821503_001_ss4_700_a_s.npy'))
    data = pds3.read_science(science_path, label_path)
    aux = pds3.read_science(science_path.replace('_s.dat', '_a.dat'), aux_path)
    spice.furnsh(os.path.join(sds, 'orig/supl/kernels/mro/mro_v01.tm'))

    mini = 1000
    maxi = 1501

    # map
    range_window_start = data['RECEIVE_WINDOW_OPENING_TIME']
    topo = data['TOPOGRAPHY']
    et = aux['EPHEMERIS_TIME']

    sc = np.empty(len(et))
    sc_pos = np.empty((len(et), 3))
    for i in range(len(et)):
        state, lt = spice.spkgeo(-74, et[i], 'J2000', 4)
        sc_pos[i] = state[0:3]
        sc[i] = np.linalg.norm(sc_pos[i])

    pri_table = {
        1: 1428E-6, 2: 1429E-6, 3: 1290E-6,
        4: 2856E-6, 5: 2984E-6, 6: 2580E-6
    }

    pri_code = data['PULSE_REPETITION_INTERVAL'][mini]
    pri = pri_table.get(pri_code, 0.0)

    out = []
    process_list = []
    pool = multiprocessing.Pool(10)
    for i in range(mini, maxi):
        process_list.append([i])

    results = [pool.apply_async(sar_processor, t) for t in process_list]

    p = prog.Prog(len(process_list))
    alt = []

    sar_track = np.zeros((len(cmp_track), 3600))
    tx0s = np.zeros(len(cmp_track))
    i = 0
    for result in results:
        p.print_Prog(i)
        rec, doppler, tx0 = result.get()
        for j in range(0, 6):
            if rec + j < len(sar_track): sar_track[rec+j] += doppler[j]
        for j in range(6, 11):
            k = 11 - j
            if rec-k >= 0: sar_track[rec - k] += doppler[j]
        tx0s[rec] = tx0
        i += 1
    for i in range(0, len(sar_track)):
        dist=((np.argmax(sar_track[i]) + tx0s[rec])*0.0375E-6 + pri-11.98E-6)*c/2000
        alt.append(dist)

    np.save('sar_test', sar_track)
    np.save('alt_test', np.array(alt))

    #sar_track = np.load('sar_test.npy')
    #alt = np.load('alt_test.npy')

    d = np.array(alt)
    record = np.array(sar_track)
    plt.style.use('dark_background')
    out = record.transpose()
    out = out[~np.all(out == 0, axis=1)]
    plt.imshow(out, cmap='binary_r', aspect='auto')
    plt.show()

    plt.plot(sc - 3389)

    # Elevation from Mars reference sphere
    fig, ax = plt.subplots()
    ax.set_ylim(-50,50)
    r = sc - d - 3389
    plt.plot(r)
    plt.show()
    plt.scatter(np.arange(len(r)), r, s=0.1)
    plt.scatter(np.arange(len(r)), topo - 3389,s=0.1)
    plt.show()



if __name__ == "__main__":
    # execute only if run as a script
    main()
