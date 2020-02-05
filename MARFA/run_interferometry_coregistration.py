__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '0.1'
__history__ = {
    '0.1':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'First build on interferometry processor'}}
"""
attempt to implement an interferometry approach to clutter discrimination based
on what has been presented in Haynes et al. (2018). As we can't actually test
with REASON measurments, the goal is to use MARFA.
# GNG: there seems to be some trouble with the Quit Picking dialog getting frozen.
# Maybe it's not getting closed?

# Just run up to coregistration

"""

import sys
sys.path.insert(0, '../xlib/clutter/')
import os.path
from os import path as pth

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from tkinter import *

import interferometry_funclib as fl
import interface_picker as ip



def select_foi_and_srf():
    """ FOI, SRF = select_foi() """

    # FEATURE OF INTEREST SELECTION ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print(' ')
    print('FEATURE OF INTEREST SELECTION')

    # Load the combined and focused 1m MARFA data product and stack to the 
    # desired trace spacing in preparation for feature detection and selection
    print('-- load combined and focused radar product')
    if gain == 'low':
        pwr_image, lim = fl.load_power_image(line, '1', trim, fresnel_stack, 'averaged', pth=path + 'S4_FOC/')
    elif gain == 'high':
        pwr_image, lim = fl.load_power_image(line, '2', trim, fresnel_stack, 'averaged', pth=path + 'S4_FOC/')
    if trim[3] == 0:
        trim[3] = lim
    if debug and bplot:
        plt.figure()
        plt.imshow(pwr_image, aspect='auto', cmap='gray'); plt.title('power image at Fresnel trace spacing')
        plt.colorbar()
        plt.clim([0, 20])
        plt.show()

    # Feature selection from the stacked power image
    print('-- select feature of interest')
    Nf = 0
    ind = 0
    while Nf < 5:
        if ind != 0:
            print('feature of interest not long enough - only', str(Nf),'samples re-select')
        #FOI = np.transpose(np.load('ZY1b_testpicksALL_15Stack.npy'))
        #FOI = np.transpose(np.load('ZY1b_testpicksSTART_15Stack.npy'))
        #FOI = np.transpose(np.load('ZY1b_testpicksEND_15Stack.npy'))
        #FOI = np.transpose(np.load('BWN01b_testpicks_15Stack.npy'))
        #FOI = np.transpose(np.load('SRH_Y81a_testpicks_15Stack.npy'))
        #FOI = np.transpose(np.load('SRH_Y81a_lakepicks_15Stack.npy'))
        FOI = ip.picker(np.transpose(pwr_image), snap_to=FOI_selection_method)
        #np.save('SRH_Y81a_lakepicks_15Stack.npy', FOI)
        FOI = np.transpose(FOI)
        if debug:
            plt.figure()
            plt.imshow(pwr_image, aspect='auto', cmap='gray')
            plt.title('power image with picked FOI')
            plt.imshow(FOI, aspect='auto')
            plt.colorbar()
            plt.show()
        # check the length of the picked FOI
        temp = 0
        for ii in range(np.size(FOI, axis=1)):
            if len(np.argwhere(FOI[:, ii] == 1)) >= 1:
                temp += 1
        Nf = temp; del temp, ii
        ind += 1

    # output: FOI

    # Feature surface above the picked FOI from the stacked power image
    print('-- select surface above feature of interest')
    SRF = fl.surface_pick(pwr_image, FOI)
    if bplot and debug:
        plt.figure()
        plt.imshow(pwr_image, aspect='auto', cmap='gray')
        plt.title('power image with picked FOI and associated SURFACE')
        plt.imshow(FOI, aspect='auto')
        plt.imshow(SRF, aspect='auto')
        plt.colorbar()
        plt.show()

    print('FEATURE OF INTEREST SELECTION -- complete')
    print(' ')
    # output: SRF 

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    return FOI, SRF



#project = 'ICP10'
#line = 'AMY/JKB2u/Y226b/'


project = 'GOG3'
line = 'NAQLK/JKB2j/ZY1b/'

#project = 'SRH1'
#line = 'DEV2/JKB2t/Y81a/'
bplot = False
bsave = True
debug = False
gain = 'low'
fresnel_stack = 15
fc = 60E6
B = 19
fs = 50E6
roll_shift = 0
FOI_selection_method = 'maximum'
interferogram_correction_mode = 'Roll'
roll_correction = True

path = '/disk/kea/WAIS/targ/xtra/' + project + '/FOC/Best_Versions/'
if project == 'SRH1' or project == 'ICP10':
    rawpath = '/disk/kea/WAIS/orig/xlob/' + line + 'RADnh5/'
    chirp_bp = True
else:
    rawpath = '/disk/kea/WAIS/orig/xlob/' + line + 'RADnh3/'
    chirp_bp = False
tregpath = '/disk/kea/WAIS/targ/treg/' + line + 'TRJ_JKB0/'
chirppath = path + 'S4_FOC/'
print('chirppath = ' + chirppath)
print('tregpath = ' + tregpath)
print('rawpath = ' + rawpath)

if line == 'NAQLK/JKB2j/ZY1b/':
    trim = [0, 1000, 0, 12000]
    chirpwin = [120, 150]
elif line == 'GOG3/JKB2j/BWN01b/':
    trim = [0, 1000, 0, 15000]
    chirpwin = [120, 150]
elif line == 'GOG3/JKB2j/BWN01a/':
    trim = [0, 1000, 15000, 27294]
    chirpwin = [120, 150]
else:
    trim = [0, 1000, 0, 0]
    chirpwin = [0, 200]

if project == 'GOG3':
    mb_offset = 155.6
elif project == 'SRH1':
    mb_offset = 127.5
elif project == 'ICP10':
    mb_offset = '127.5'


FOI_cache_file = 'run_interferometry__FOI.npz'
SRF_cache_file = 'run_interferometry__SRF.npz'

if os.path.exists(FOI_cache_file):
    print('Loading picks from cache ' + FOI_cache_file)
    with np.load(FOI_cache_file) as data:
        FOI = data['FOI']
        SRF = data['SRF']
        trim = data['trim']
else:
    FOI, SRF = select_foi_and_srf()
    np.savez_compressed(FOI_cache_file, FOI=FOI, SRF=SRF, trim=trim)
    print('Saved picks to ' + FOI_cache_file)




# CHIRP STABILITY ASSESSMENT AND COREGISTRATION ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print('CHIRP STABILITY ASSESSMENT AND COREGISTRATION')

# Test to see if co-registered data product already exists for this PST
A = line.split('/')
post_coreg = project + '_' + A[0] + '_' + A[1] + '_' + A[2] + '_' + str(trim[2]) + 'to' + str(trim[3]) 
post_coreg = post_coreg + '_AfterCoregistration_ric.npz'
coreg_test = pth.exists(post_coreg)
print('post_coreg = ' + post_coreg)

if not coreg_test:
    # Load the focused SLC 1m port and starboard radargrams
    print('-- load port and starboard single-look products')
    if gain == 'low':
        chan_a, chan_b = ('5', '7')
    elif gain == 'high':
        chan_a, chan_b = ('6', '8')
    else:
        assert False

    magA, phsA = fl.load_marfa(line, chan_a, pth=path + 'S4_FOC/')
    magA = magA[trim[0]:trim[1], trim[2]:trim[3]]
    phsA = phsA[trim[0]:trim[1], trim[2]:trim[3]]
    cmpA = fl.convert_to_complex(magA, phsA)
    del magA, phsA

    magB, phsB = fl.load_marfa(line, chan_b, pth=path + 'S4_FOC/')
    magB = magB[trim[0]:trim[1], trim[2]:trim[3]]
    phsB = phsB[trim[0]:trim[1], trim[2]:trim[3]]
    cmpB = fl.convert_to_complex(magB, phsB)
    del magB, phsB

    # Sub-pixel co-registration of the port and starboard range lines
    print('-- co-registration of port and starboard radargrams')
    cmpA3, cmpB3, shift_array = fl.coregistration(cmpA, cmpB, (1 / 50E6), 10)
    print('Saving to ' + post_coreg)
    if bsave:
        np.savez(post_coreg, cmpA3=cmpA3, cmpB3=cmpB3, shift_array=shift_array)
