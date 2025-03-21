#!/usr/bin/env python3

import time
import os
import numpy as np
import spiceypy as spice
from scipy.constants import c
from scipy.ndimage import shift
import matplotlib.pyplot as plt

import cmp.pds3lbl as pds3
import cmp.plotting as plotting

def running_mean(x, N):
    xp = np.pad(x, (N//2, N-1-N//2), mode='edge')
    res = np.convolve(xp, np.ones((N,))/N, mode='valid')
    return res

t0 = time.time()

sar_length = 200
sds = os.getenv('SDS', '/disk/kea/SDS')
label_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt')
aux_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt')

science_path = os.path.join(sds, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0003/data/edr28xxx/edr2821503/e_2821503_001_ss4_700_a_s.dat')
npypath = os.path.join(sds, 'targ/xtra/SHARAD/cmp/mrosh_0003/data/edr28xxx/edr2821503/ion/e_2821503_001_ss4_700_a_s.npy')
cmp_track = np.load(npypath)
data = pds3.read_science(science_path, label_path)
aux = pds3.read_science(science_path.replace('_s.dat', '_a.dat'), aux_path)
spicepath = os.path.join(sds, 'orig/supl/kernels/mro/mro_v01.tm')
spice.furnsh(spicepath)

range_window_start = data['RECEIVE_WINDOW_OPENING_TIME']
topo = data['TOPOGRAPHY']
r_tx0 = int(min(range_window_start))
r_offset = int(max(range_window_start))-r_tx0

#plotting.plot_radargram(cmp_track,range_window_start-range_window_start[0])

# S/C position
et = aux['EPHEMERIS_TIME']
sc = np.empty(len(et))
for i in range(len(et)):
    scpos, lt = spice.spkgeo(-74, et[i], 'J2000', 4)
    sc[i] = np.linalg.norm(scpos[0:3])

sc_cor = np.array(2000*sc/c/0.0375E-6).astype(int)
phase = -sc_cor + range_window_start
tx0 = int(min(phase))
offset = int(max(phase) - tx0)

PRI_TABLE = {
    1: 1428E-6,
    2: 1429E-6,
    3: 1290E-6,
    4: 2856E-6,
    5: 2984E-6,
    6: 2580E-6
}
pri_code = data['PULSE_REPETITION_INTERVAL'][0]
pri = PRI_TABLE.get(pri_code, 0.0)

radargram = np.zeros((len(data), 3600+offset))
for rec in range(len(cmp_track)):
    # adjust for range window start
    dat = np.pad(cmp_track[rec], (0, offset), 'constant', constant_values=0)
    radargram[rec] = shift(abs(dat), phase[rec] - tx0, cval=0)

avg = np.empty((len(data), 3600+offset))

for i in range(3600+offset):
    avg[:, i] = running_mean(radargram[:, i], sar_length)

t1 = time.time()-t0
print('done in', t1, 'seconds')

plt.style.use('dark_background')
out = avg.transpose()
out = out[~np.all(out == 0, axis=1)]
plt.imshow(out, cmap='binary_r', aspect='auto')
plt.show()

delta = np.argmax(avg, axis=1)
# ToF
tx = (range_window_start + delta - phase + tx0) * 0.0375E-6 + pri - 11.98E-6
# One-way range in km
d = tx * c / 2000

# Elevation from Mars reference sphere
r = sc - d - 3389

plt.scatter(np.arange(len(r)), r, s=0.1)
plt.scatter(np.arange(len(r)), topo - 3389, s=0.1)
plt.show()
