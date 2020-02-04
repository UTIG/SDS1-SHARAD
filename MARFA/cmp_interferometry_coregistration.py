#!/usr/bin/env python3
__authors__ = ['Gregory Ng', 'ngg@ig.utexas.edu']
__version__ = '0.1'
__history__ = {
    '0.1':
        {'date': 'February 1, 2020',
         'author': 'Gregory Ng, UTIG',
         'info': 'Compare interferometry outputs'}}

import sys
import numpy as np
bplot = True
# Compare coregistration

post_coreg1 = 'GOG3_NAQLK_JKB2j_ZY1b_0to12000_AfterCoregistration_ric_0.npz'
post_coreg2 = 'GOG3_NAQLK_JKB2j_ZY1b_0to12000_AfterCoregistration_ric_1.npz'

print(post_coreg1)
print(post_coreg2)
with np.load(post_coreg1) as data1, np.load(post_coreg2) as data2:
    print('data1: ', ','.join(data1.keys()))
    print('data2: ', ','.join(data2.keys()))

    cmpA1 = data1['cmpA3']
    cmpA2 = data2['cmpA3']
    rmsa = np.sqrt(np.square(abs(cmpA1 - cmpA2)).mean(axis=1))

    cmpB1 = data1['cmpB3']
    cmpB2 = data2['cmpB3']
    rmsb = np.sqrt(np.square(abs(cmpB1 - cmpB2)).mean(axis=1))
    print('RMS cmpA = {:0.3f}'.format(rmsa.mean()))
    print('RMS cmpB = {:0.3f}'.format(rmsb.mean()))
    print('size cmpA = {:s}'.format(str(cmpA1.shape)))
    print('size cmpB = {:s}'.format(str(cmpB1.shape)))

    shift2 = data2['shift_array']

if bplot:
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    print("Plotting")
    plt.subplot(211)
    plt.plot(rmsa)
    plt.plot(rmsb)
    plt.legend(['RMSE cmpA','RMSE cmpB'])
    
    plt.subplot(212)
    plt.plot(shift2)
    plt.grid()
    plt.legend(('shift2',))
    plt.show()
