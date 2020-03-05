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

post_coreg1 = 'GOG3_NAQLK_JKB2j_ZY1b_0to12000_AfterCoregistration_ric_1.npz'
post_coreg2 = 'GOG3_NAQLK_JKB2j_ZY1b_0to12000_AfterCoregistration_ric.npz'

print(post_coreg1)
print(post_coreg2)
with np.load(post_coreg1) as data1, np.load(post_coreg2) as data2:
    print('data1: ', ','.join(data1.keys()))
    print('data2: ', ','.join(data2.keys()))

    cmpA1 = data1['cmpA3']
    cmpA2 = data2['cmpA3']
    rmsa = np.sqrt(np.square(abs(cmpA1 - cmpA2)).mean(axis=0))

    cmpB1 = data1['cmpB3']
    cmpB2 = data2['cmpB3']
    rmsb = np.sqrt(np.square(abs(cmpB1 - cmpB2)).mean(axis=0))
    print('RMS cmpA = {:0.3f}'.format(rmsa.mean()))
    print('RMS cmpB = {:0.3f}'.format(rmsb.mean()))
    print('size cmpA = {:s}'.format(str(cmpA1.shape)))
    print('size cmpB = {:s}'.format(str(cmpB1.shape)))

    shift1 = data1['shift_array']
    shift2 = data2['shift_array']

    shiftdiff= abs(shift1 - shift2).mean()
    print('Average shift error = {:0.1f}'.format(shiftdiff))

if bplot:
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    print("Plotting")

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(rmsa)
    axes[0].plot(rmsb)
    axes[0].grid()
    axes[1].set_ylabel('Amplitude')
    axes[0].set_title('RMS error between old and new co-registered outputs')
    axes[0].legend(['RMSE cmpA','RMSE cmpB'])
    
    axes[1].plot(shift1, linewidth=1)
    axes[1].plot(shift2, linewidth=1)
    axes[1].grid()
    axes[1].set_ylabel('Fast time samples')
    axes[1].set_title('Old and new shift values')

    axes[1].legend(('Old shift', 'New shift',))
    
    axes[2].plot(shift2 - shift1)
    axes[2].set_title('Difference between Old Shift and New Shift')
    axes[2].set_ylabel('Fast time samples')
    axes[2].set_xlabel('Slow time samples')
    axes[2].grid()
    #plt.legend(('dshift',))
    plt.show()
