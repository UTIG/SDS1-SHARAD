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
"""

import sys
sys.path.insert(0, '../xlib/clutter/')
import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal
import interferometry_funclib as fl
import interface_picker as ip

line = 'NAQLK/JKB2j/ZY1b/'
debug = False
N = 15
fc = 60E6
B = 19
FOI_selection_method = 'maximum'

if line == 'NAQLK/JKB2j/ZY1b/':
    inpath = '/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S4_FOC/'
    pik1path = '/disk/kea/WAIS/targ/xtra/GOG3/CMP/pik1/'
    normpath = '/disk/kea/WAIS/targ/norm/'
    S1path = '/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S1_POS/'
    gain = 'low'
    trim = [0, 1000, 0, 12000]
    #trim = [0, 1000, 0, 1000]
    sigwin = [0, 1500]
elif line == 'GOG3/JKB2j/BWN01b/':
    inpath = '/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S4_FOC/'
    pik1path = '/disk/kea/WAIS/targ/xtra/GOG3/CMP/pik1/'
    gain = 'low'
    trim = [0, 1000, 0, 12000]
    sigwin = [0, 1000]
elif line == 'DEV2/JKB2t/Y81a/':
    inpath = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC/'
    gain= 'low'
    trim = [0, 1000, 0, 26907]
#    trim = [0, 1000, 0, 5000]
    sigwin = [0, 300]
elif line == 'DEV/JKB2t/X78a/':
    inpath = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC/'
    gain= 'low'
    #trim = [0, 1200, 0, 50017]
    trim = [0, 1200, 0, 5000]
    sigwin = [0, 200]

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# STEP 1a/2a -- load the combined and focused 1m MARFA data product
#               and stack to the desired trace spacing in preparation
#               for feature detection and selection
if gain == 'low':
    pwr_image = fl.load_power_image(line, '1', trim, N, 'averaged', pth=inpath)
elif gain == 'high':
    pwr_image = fl.load_power_image(line, '2', trim, N, 'averaged', pth=inpath)
if debug:
    plt.figure()
    plt.imshow(pwr_image, aspect='auto', cmap='gray'); plt.title('power image at Fresnel trace spacing')
    plt.colorbar()
    plt.clim([0, 20])
    plt.show()
# ----------------------------------------------------------------------------

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# STEP 3a -- Feature selection from the stacked power image
# ****************************************************************************
# REQUIREMENT: According to Haynes et al. (2018) this feature should be tracked
#              across at least 67 traces in the power image
# ****************************************************************************
Nf = 0
ind = 0
while Nf < 67:
    if ind != 0:
        print('feature of interest not long enough - re-select')
    if N == 15:
        FOI = np.load('ZY1b_testpicks_15Stack.npy')
    elif N == 1:
        FOI = np.load('ZY1b_testpicks_NoStack.npy')
    #FOI = np.load('ZY1b_surfpicks_15Stack.npy')
    FOI = np.transpose(FOI)
    #FOI = ip.picker(np.transpose(pwr_image), color='gray', snap_to=FOI_selection_method)
    #FOI = np.transpose(FOI)
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
# -----------------------------------------------------------------------------

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# STEP 1b -- load the focused SLC 1m port and starboard radargrams
if gain == 'low':
    magA, phsA = fl.load_marfa(line, '5', pth=inpath)
    magB, phsB = fl.load_marfa(line, '7', pth=inpath)
elif gain == 'high':
    magA, phsA = fl.load_marfa(line, '6', pth=inpath)
    magB, phsB = fl.load_marfa(line, '8', pth=inpath)
noiseA = np.sqrt(np.mean(np.mean(magA[-400::, :])))
noiseB = np.sqrt(np.mean(np.mean(magB[-400::, :])))
magA = magA[trim[0]:trim[1], trim[2]:trim[3]]
phsA = phsA[trim[0]:trim[1], trim[2]:trim[3]]
magB = magB[trim[0]:trim[1], trim[2]:trim[3]]
phsB = phsB[trim[0]:trim[1], trim[2]:trim[3]]
if debug:
    plt.figure()
    plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('antenna A magntiude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='nipy_spectral'); plt.title('antenna A phase'); plt.clim([-180, 180]); plt.colorbar()
    plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='nipy_spectral'); plt.title('antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
    plt.show()
# ------------------------------------------------------------------------------

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# STEP 2b -- Interferometric processing of port and starboard 1m radargrams.
#            Interferometric processing is based on the approach outlined in
#            Castelletti et al. (2018)

## STEP 2bi - cross-correlate port and starboard records using the loop-back chirp
#if gain == 'low':
#    pik1_magA = fl.load_pik1(line, '5', pth=pik1path)
#    pik1_magB = fl.load_pik1(line, '7', pth=pik1path)
#elif gain == 'high':
#    pik1_magA = fl.load_pik1(line, '6', pth=pik1path)
#    pik1_magB = fl.load_pik1(line, '8', pth=pik1path)
#pik1_magA = pik1_magA[sigwin[0]:sigwin[1], :]
#pik1_magB = pik1_magB[sigwin[0]:sigwin[1], :]
#if True:       
#    plt.figure()
#    plt.subplot(211); plt.imshow(pik1_magA, aspect='auto', cmap='gray'); plt.title('antenna A pik1 magnitude')
#    plt.subplot(212); plt.imshow(pik1_magB, aspect='auto', cmap='gray'); plt.title('antenna B pik1 magnitude')
#    plt.show()
#phs_shft = np.zeros((np.size(pik1_magA, axis=1), ), dtype=int)
##ii = 0
##if ii == 0:
#for ii in range(np.size(pik1_magA, axis=1)):
#    test = scipy.signal.correlate(pik1_magA[:, ii], pik1_magB[:, ii])
#    phs_shft[ii] = np.argwhere(test == np.max(test))
#del test

# STEP 2bii - sub-pixel co-registration of the port and starboard range lines
if line == 'NAQLK/JKB2j/ZY1b/' and trim[0] == 0 and trim[1] == 1000 and trim[2] == 0 and trim[3] == 12000:
    temp = np.load('ZY1b_0to12000_post2bii.npz')
    magA2 = temp['magA2']
    phsA2 = temp['phsA2']
    magB2 = temp['magB2']
    phsB2 = temp['phsB2']
    del temp
else:
    magA2, phsA2, magB2, phsB2 = fl.coregistration(magA, phsA, magB, phsB, (1 / 50E6), 10)
    #np.savez_compressed('ZY1b_0to12000_post2bii', magA2=magA2, phsA2=phsA2, magB2=magB2, phsB2=phsB2)
if debug:
    plt.figure()
    plt.subplot(411); plt.imshow(magA2, aspect='auto', cmap='gray'); plt.title('co-registered antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(412); plt.imshow(np.rad2deg(phsA2), aspect='auto', cmap='nipy_spectral'); plt.title('co-registered antenna A phase'); plt.clim([-180, 180]); plt.colorbar() 
    plt.subplot(413); plt.imshow(magB2, aspect='auto', cmap='gray'); plt.title('co-registered antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(414); plt.imshow(np.rad2deg(phsB2), aspect='auto', cmap='nipy_spectral'); plt.title('co-registered antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
    plt.show()

# STEP 2biii - interferogram formation
int_image = fl.stacked_interferogram(magA2, phsA2, magB2, phsB2, N, method='Smoothed')
if debug:
    plt.figure()
    plt.imshow(int_image, aspect='auto', cmap='nipy_spectral')
    plt.title('interferogram at Fresnel trace spacing')
    plt.colorbar()
    plt.clim([-180, 180])
    plt.imshow(FOI, aspect='auto')
    plt.show()

# STEP 2biv - roll correction
roll_dist, roll_ang = fl.load_roll(normpath + line + 'AVN_JKBa/', S1path + line)
roll_dist = roll_dist[trim[2]:trim[3]]
roll_ang = roll_ang[trim[2]:trim[3]]
if True:
    plt.figure()
    plt.subplot(211)
    plt.plot(roll_dist, roll_ang)
    plt.title('Aircraft roll angle [deg]')
    plt.xlim(trim[2], trim[3])
    plt.subplot(212)
    plt.imshow(int_image, aspect='auto', cmap='seismic')
    plt.title('interferogram')
    #plt.colorbar()
    plt.clim([-180, 180])
    plt.imshow(FOI, aspect='auto')
    plt.show()
# ------------------------------------------------------------------------------------

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# STEP 3b - estimate feature phase
FOI_phs = fl.FOI_interferometric_phase(int_image, FOI)
if debug:
    plt.figure()
    plt.hist(FOI_phs, bins=50)
    plt.title('distribution of interferometric phase angles')
    plt.xlim([-180, 180])
    plt.show()
mean_phi = np.mean(FOI_phs)
# ------------------------------------------------------------------------------------

## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## calculate a nadir empirical interferometric phase PDF for the selected FOI
#iphi, emp_pdf, gamma = fl.empirical_pdf(fc, B, N, 0, FOI, magA, magB, noiseA, noiseB, 'averaged')
#print('gamma:', gamma)
#if debug:
#    plt.figure()
#    plt.plot(iphi, emp_pdf)
#    plt.title('empirical interferometric phase pdf centered at zero')
#    plt.xlabel('interferometric phase angle [deg]')
#    plt.ylabel('pdf [$\phi$]')
#    plt.xlim([-180, 180])
#    plt.show()
#del magA, magB
## --------------------------------------------------------------------------------------    

## determine the empirical sample mean from the empirical distribution
##sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, mean_phi)
#sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, 0)
#print('sigma_m:', sigma_m, 'mean_phi:', mean_phi)
#
## compare the empirical sample mean with the mean of the FOI interferometric
## phase angles to determine whether the FOI is at nadir or not
#if sigma_m < np.abs(0.5 * mean_phi):
#    print('FOI of interest appears to be off-nadir')
#    print('-- mean interferometric phase angle:', np.round(mean_phi, 3), 'degrees')
#else:
#    print('FOI cannot be distinguished from a nadir reflection')   
