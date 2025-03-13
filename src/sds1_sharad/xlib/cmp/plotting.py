#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift

def plot_radargram(data, tx, save_path=None, samples=3600):
    radargram = np.zeros((len(data), 10000))
    for rec in range(len(data)):
        # adjust for range window start
        dat = np.pad(data[rec], (0, 10000 - samples),
                     'constant', constant_values=0)
        
        sig = np.abs(dat) #np.power(np.abs(dat), 2)
        #db = 10 * np.log10(pow_out)
        #maxdb = np.amax(db)
        #sig = db/maxdb*255
        #sig[np.where(sig < 30)] = 0
        #sig[np.where(sig > 255)] = 255
        radargram[rec] = shift(sig, tx[rec]+5000, cval=0)

    plt.style.use('dark_background')
    out = radargram.transpose()
    out = out[~np.all(out == 0, axis=1)]
    plt.imshow(out, cmap='binary_r', aspect='auto')
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
