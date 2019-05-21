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
import matplotlib.colors as col
from tkinter import *
import interferometry_funclib as fl
import interface_picker as ip

#project = 'GOG3'
#line = 'NAQLK/JKB2j/ZY1b/'
#line = 'GOG3/JKB2j/BWN01a/'
project = 'SRH1'
line = 'DEV2/JKB2t/Y81a/'
debug = False
gain = 'low'
fresnel_stack = 15
fc = 60E6
B = 19
roll_shift = 0
FOI_selection_method = 'maximum'
roll_correction = True

path = '/disk/kea/WAIS/targ/xtra/' + project + '/FOC/Best_Versions/'
if project == 'SRH1':
    rawpath = '/disk/kea/WAIS/orig/xlob/' + line + 'RADnh5/'
else:
    rawpath = '/disk/kea/WAIS/orig/xlob/' + line + 'RADnh3/'
normpath = '/disk/kea/WAIS/targ/norm/'
chirppath = path + 'S4_FOC/'


if line == 'NAQLK/JKB2j/ZY1b/':
    trim = [0, 1000, 0, 12000]
    chirpwin = [120, 150]
elif line == 'GOG3/JKB2j/BWN01b/':
    trim = [0, 1000, 0, 15000]
    chirpwin = [120, 150]
elif line == 'GOG3/JKB2j/BWN01a/':
    trim = [0, 1000, 15000, 27294]
    chirpwin = [120, 150]
elif line == 'DEV2/JKB2t/Y81a/':
    trim = [0, 1000, 0, 24000]
    chirpwin = [0, 200]


# FEATURE OF INTEREST SELECTION ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print(' ')
print('FEATURE OF INTEREST SELECTION')

# Step 1 - load the combined and focused 1m MARFA data product and stack to the 
#          desired trace spacing in preparation for feature detection and selection
print('1) load combined and focused radar product')
if gain == 'low':
    pwr_image = fl.load_power_image(line, '1', trim, fresnel_stack, 'averaged', pth=path + 'S4_FOC/')
elif gain == 'high':
    pwr_image = fl.load_power_image(line, '2', trim, fresnel_stack, 'averaged', pth=path + 'S4_FOC/')
if debug:
    plt.figure()
    plt.imshow(pwr_image, aspect='auto', cmap='gray'); plt.title('power image at Fresnel trace spacing')
    plt.colorbar()
    plt.clim([0, 20])
    plt.show()

# Step 2 - Feature selection from the stacked power image
print('2) select feature of interest')
Nf = 0
ind = 0
while Nf < 5:
    if ind != 0:
        print('feature of interest not long enough - only', str(Nf),'samples re-select')
    #FOI = np.transpose(np.load('ZY1b_testpicksALL_15Stack.npy'))
    #FOI = np.transpose(np.load('ZY1b_testpicksSTART_15Stack.npy'))
    #FOI = np.transpose(np.load('ZY1b_testpicksEND_15Stack.npy'))
    #FOI = np.transpose(np.load('BWN01b_testpicks_15Stack.npy'))
    #FOI = np.transpose(np.load('Y81a_testpicks_15Stack.npy'))
    FOI = ip.picker(np.transpose(pwr_image), snap_to=FOI_selection_method)
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

print('FEATURE OF INTEREST SELECTION -- complete')
print(' ')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------





# PHASE STABILITY ASSESSMENT ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print('CHIRP STABILITY ASSESSMENT')

# Step 1 - load and range compress the interpolated 1m data products
print('1) load and range compress raw radar data')
#dechirpA, dechirpB = fl.denoise_and_dechirp(gain, trim, rawpath, path + 'S2_FIL/' + line, chirppath + line, do_cinterp=False)
if debug:
    plt.figure()
    plt.subplot(211); plt.imshow(20 * np.log10(np.abs(dechirpA[0:1000, :])), aspect='auto', cmap='gray'); plt.title('Antenna A')
    plt.subplot(212); plt.imshow(20 * np.log10(np.abs(dechirpB[0:1000, :])), aspect='auto', cmap='gray'); plt.title('Antenna B')
    plt.show()

# Step 2 - extract the loop-back chirp
print('2) extract the loop-back chirp')
#loopbackA = dechirpA[chirpwin[0]:chirpwin[1], :]
#loopbackB = dechirpB[chirpwin[0]:chirpwin[1], :]
if debug:
    plt.figure()
    plt.subplot(411); plt.imshow(20 * np.log10(np.abs(loopbackA)), aspect='auto', cmap='gray'); plt.title('Loopback A Magntiude [dB]')
    plt.subplot(412); plt.imshow(np.angle(loopbackA), aspect='auto', cmap='jet'); plt.title('Loopback A Phase')
    plt.subplot(413); plt.imshow(20 * np.log10(np.abs(loopbackB)), aspect='auto', cmap='gray'); plt.title('Loopback B Magnitude [dB]')
    plt.subplot(414); plt.imshow(np.angle(loopbackB), aspect='auto', cmap='jet'); plt.title('Loopback B Phase')
    plt.figure()
    plt.plot(20 * np.log10(np.abs(loopbackA[:, 0000])), label='loopbackA - 0')
    plt.plot(20 * np.log10(np.abs(loopbackA[:, 1000])), label='loopbackA - 1000')
    plt.plot(20 * np.log10(np.abs(loopbackA[:, 2000])), label='loopbackA - 2000')
    plt.plot(20 * np.log10(np.abs(loopbackB[:, 0000])), label='loopbackB - 0')
    plt.plot(20 * np.log10(np.abs(loopbackB[:, 1000])), label='loopbackB - 1000')
    plt.plot(20 * np.log10(np.abs(loopbackB[:, 2000])), label='loopbackB - 2000')
    plt.legend()
    plt.title('range compressed loopback chirp')
    plt.xlabel('fast-time sample')
    plt.ylabel('magnitude [dB]')
    plt.show()

# Step 3 - assess the phase stability of the loop-back chirp for each antenna
print('3) characterize chirp stability')
#stabilityA = fl.chirp_phase_stability(loopbackA[:, 0], loopbackA, method='xcorr', rollval=20)
#stabilityB = fl.chirp_phase_stability(loopbackB[:, 0], loopbackB, method='xcorr', rollval=20)
stabilityA = np.zeros((trim[3]), dtype=int)
stabilityB = np.zeros((trim[3]), dtype=int)
if debug:
    plt.figure()
    plt.plot(np.arange(0, np.size(loopbackA, axis=1)), stabilityA, label='Antenna A')
    plt.plot(np.arange(0, np.size(loopbackB, axis=1)), stabilityB, label='Antenna B')
    plt.xlabel('Range Line #')
    plt.ylabel('Shift for Optimal Chirp Stability')
    plt.legend()
    plt.show()

#del dechirpA, dechirpB, loopbackA, loopbackB
print('CHIRP STABILITY ASSESSMENT -- complete')
print(' ')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



  
# INTERFEROMETRY ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print('INTERFEROMETRY')

# Step 1 - load the focused SLC 1m port and starboard radargrams
print('1) load port and starboard single-look products')
if gain == 'low':
    magA, phsA = fl.load_marfa(line, '5', pth=path + 'S4_FOC/')
    magB, phsB = fl.load_marfa(line, '7', pth=path + 'S4_FOC/')
elif gain == 'high':
    magA, phsA = fl.load_marfa(line, '6', pth=path + 'S4_FOC/')
    magB, phsB = fl.load_marfa(line, '8', pth=path + 'S4_FOC/')
if trim[3] != 0:
    magA = magA[trim[0]:trim[1], trim[2]:trim[3]]
    phsA = phsA[trim[0]:trim[1], trim[2]:trim[3]]
    magB = magB[trim[0]:trim[1], trim[2]:trim[3]]
    phsB = phsB[trim[0]:trim[1], trim[2]:trim[3]]
cmpA = fl.convert_to_complex(magA, phsA)
cmpB = fl.convert_to_complex(magB, phsB)
del magA, phsA, magB, phsB
if debug:
    magA, phsA = fl.convert_to_magphs(cmpA)
    magB, phsB = fl.convert_to_magphs(cmpB)
    plt.figure()
    plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('antenna A magntiude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='seismic'); plt.title('antenna A phase'); plt.clim([-180, 180]); plt.colorbar()
    plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='seismic'); plt.title('antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
    RGB = np.zeros((len(magA), np.size(magA, axis=1), 3), dtype=float)
    RGB[:, :, 0] = np.divide(magA, np.max(np.max(magA)))
    RGB[:, :, 2] = np.divide(magB, np.max(np.max(magA)))
    plt.figure()
    plt.subplot(211); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('antenna A magnitude'); plt.clim([0, 20])
    plt.subplot(212); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('antenna B magnitude'); plt.clim([0, 20])
    #plt.subplot(313); plt.imshow(RGB, aspect='auto')
    #plt.title('RGB image with normalized magA in R and normalized magB in B\n normalization to max amplitude in magA')
    plt.show()
    del magA, phsA, magB, phsB

# Step 2 - apply shifts calculated from chirp stability analysis to align 
#          range lines
print('2) chirp stability adjustment')
cmpA2 = fl.phase_stability_adjustment(cmpA, stabilityA)
cmpB2 = fl.phase_stability_adjustment(cmpB, stabilityB)
if debug:
    magA, phsA = fl.convert_to_magphs(cmpA2)
    magB, phsB = fl.convert_to_magphs(cmpB2)
    plt.figure()
    plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('A radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='seismic'); plt.title('A phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
    plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('B radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='seismic'); plt.title('B phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
    plt.show()
    del magA, phsA, magB, phsB

# Step 3 - sub-pixel co-registration of the port and starboard range lines
print('3) co-registration of port and starboard radargrams')
temp = np.load('Y81a_0to24000_AfterCoregistration.npz')
cmpA3 = temp['cmpA3']
cmpB3 = temp['cmpB3']
#cmpA3, cmpB3 = fl.coregistration(cmpA2, cmpB2, (1 / 50E6), 10)
#np.savez_compressed('ZY1b_0to24000_AfterCoregistration', cmpA3=cmpA3, cmpB3=cmpB3)
#cmpA3 = cmpA2
#cmpB3 = cmpB2
if debug:
    magA, phsA = fl.convert_to_magphs(cmpA3)
    magB, phsB = fl.convert_to_magphs(cmpB3)
    plt.figure()
    plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('co-registered antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='seismic'); plt.title('co-registered antenna A phase'); plt.clim([-180, 180]); plt.colorbar() 
    plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('co-registered antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
    plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='seismic'); plt.title('co-registered antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
    plt.show()

# Step 4 - determine the number of azimuth samples between independent looks
print('4) detemine azimuth samples between independent range lines')
az_step = fl.independent_azimuth_samples(cmpA3, cmpB3, FOI)
az_step = int(az_step)
print('   > azimuth samples between independent range lines:', str(az_step))

# Step 5 - roll correction 
print('5) derive roll correction')
roll_phase, roll_ang = fl.roll_correction(np.divide(299792458, fc), B, trim, normpath + line + 'AVN_JKBa/', path + 'S1_POS/' + line, roll_shift=roll_shift)
if debug:
    plt.figure()
    plt.plot(np.rad2deg(roll_phase))
    plt.title('roll correction interferometric phase angle [deg]')
    plt.show()

# Step 6 - interferogram
print('6) producing interferogram')
int_image = fl.stacked_interferogram(cmpA3, cmpB3, fresnel_stack, roll_phase, roll_correction, az_step=az_step)
if debug:
    int_image_noroll = fl.stacked_interferogram(cmpA3, cmpB3, fresnel_stack, roll_phase, False, az_step=az_step)
    plt.figure()
    plt.subplot(313); plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='seismic')
    plt.title('interferogram with defined roll correction [deg]'); plt.clim([-180, 180]); plt.imshow(FOI, aspect='auto')
    plt.subplot(311); plt.imshow(np.rad2deg(int_image_noroll), aspect='auto', cmap='seismic')
    plt.title('interferogram no roll [deg]'); plt.clim([-180, 180]); plt.imshow(FOI, aspect='auto')
    plt.subplot(312)
    plt.plot(np.linspace(0, 1, len(roll_ang)), roll_ang - roll_shift, label='actual')
    plt.plot(np.linspace(0, 1, len(roll_ang)), roll_ang, label='applied')
    plt.title('roll angle [deg]'); plt.xlim([0, 1]); plt.legend()
    plt.show()

# Step 7 - correlation map
print('7) producing correlation map')
corrmap = fl.stacked_correlation_map(cmpA3, cmpB3, fresnel_stack, az_step=az_step)
if debug:
    plt.figure()
    plt.subplot(212)
    plt.imshow(corrmap, aspect='auto', cmap='nipy_spectral')
    plt.title('magnitude of correlation map at Fresnel trace spacing')
    plt.colorbar()
    plt.clim([0, 1])
    plt.imshow(FOI, aspect='auto')
    plt.subplot(211)
    plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='seismic')
    plt.title('interferogram [deg]'); plt.clim([-180, 180]); plt.colorbar(); plt.imshow(FOI, aspect='auto')
    #RGB = np.zeros((len(corrmap), np.size(corrmap, axis=1), 3), dtype=float)
    #RGB[:, :, 0] = np.divide(np.abs(corrmap), np.max(np.max(np.abs(corrmap))))
    #RGB[:, :, 1] = RGB[:, :, 1] + 0.5
    #RGB[:, :, 2] = np.divide(int_image + 180, 360)
    #plt.figure()
    #plt.subplot(311); plt.imshow(int_image, aspect='auto', cmap='seismic'); plt.title('interferogram')
    #plt.subplot(312); plt.imshow(np.abs(corrmap), aspect='auto', cmap='nipy_spectral'); plt.title('correlation map')
    #plt.subplot(313); plt.imshow(col.hsv_to_rgb(RGB), aspect='auto')
    #plt.title('HSV image with correlation map in Hue and interferogram in Values\n Saturation set to 0.5')
    plt.show()

# Step 8 - extract feature of interest phase and correlation
print('8) extract information along FOI')
FOI_phs = fl.FOI_extraction(int_image, FOI)
FOI_cor = fl.FOI_extraction(np.abs(corrmap), FOI)
if debug:
    plt.figure()
    plt.subplot(211); plt.hist(np.rad2deg(FOI_phs), bins=20)
    plt.title('distribution of interferometric phase angles')
    plt.xlim([-180, 180])
    plt.subplot(212); plt.hist(FOI_cor, bins=50)
    plt.title('distribution of interferometric correlation')
    plt.show()
mean_phi = np.rad2deg(np.mean(FOI_phs))
gamma = np.mean(FOI_cor)
#mean_phi = np.rad2deg(np.median(FOI_phs))
#gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N = int(np.floor(np.divide(fresnel_stack, az_step)) + 1)
Nf = int(np.round(np.divide(Nf, az_step)) + 1)
print('   > mean interferometric phase:', str(mean_phi))
print('   > mean interferometric correlation:', str(gamma))
print('   > number of independent looks:', str(N))
print('   > number of independent multi-looked pixels used to define the FOI:', str(Nf))

# Step 9 - calculate nadir empirical interferometric phase PDF and empirical PDF
#          corresponding to the observed mean phase
print('9) calculate observed and nadir empirical interferometric phase PDFs')
_, nadir_emp_pdf = fl.empirical_pdf(fc, B, N, gamma)
iphi, obs_emp_pdf = fl.empirical_pdf(fc, B, N, gamma, phi_m=mean_phi)
if True:
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('interferometric phase angle [deg]')
    ax1.set_ylabel('histogram counts', color='b')
    ax1.hist(np.rad2deg(FOI_phs), bins=50, color='b', label='observed data')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('pds [$\phi$]', color='r')
    ax2.plot(np.rad2deg(iphi), nadir_emp_pdf, '--r', label='nadir empirical pdf')
    ax2.plot(np.rad2deg(iphi), obs_emp_pdf, '-r', label='observed empirical pdf')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.ylim([0, np.max([1.1 * np.max(nadir_emp_pdf), 1.1 * np.max(obs_emp_pdf)])])
    fig.tight_layout()
    plt.xlim([-180, 180])
    plt.legend()
    plt.show()
  
# Step 10 - determine the uncertainty in the empirical sample mean from the
#          empirical distributions
print('10) determine uncertainties in empirical sample mean of each PDF')
nadir_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma)
obs_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, phi_m=mean_phi)
print('   > uncertainty in nadir empirical sample mean:', str(nadir_sigma_m))
print('   > uncertainty in observed empirical sample mean:', str(obs_sigma_m))
    
print('INTERFEROMETRY -- complete')
print(' ')

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# compare the empirical sample mean with the mean of the FOI interferometric
# phase angles to determine whether the FOI is at nadir or not
if obs_sigma_m < np.abs(0.5 * mean_phi):
    print('>> FOI can be distinguished from an identical feature at nadir')
    print('   > mean interferometric phase angle:', np.round(mean_phi, 3), 'degrees')
else:
    print('>> FOI cannot be distinguished from a nadir reflection')  
print(' ')


#stack_val = 15
#
## load
#pwr_image = fl.load_power_image(line, '1', trim, 1, 'averaged', pth=path + 'S4_FOC/')
#temp = np.load('ZY1b_0to12000_AfterCoregistration.npz')
#cmpA = temp['cmpA3']
#cmpB = temp['cmpB3']
#magA, _ = fl.convert_to_magphs(cmpA)
#magB, _ = fl.convert_to_magphs(cmpB)
#
## stack
#pwr_image_stack = np.divide(fl.stack(pwr_image, stack_val), stack_val)
#magA_stack = np.divide(fl.stack(magA, stack_val), stack_val)
#magB_stack = np.divide(fl.stack(magB, stack_val), stack_val)
#
## RGB 
#RGB = np.zeros((len(pwr_image), np.size(pwr_image, axis=1), 3), dtype=float)
#RGB[:, :, 0] = np.divide(pwr_image, np.max(np.max(pwr_image)))
#RGB[:, :, 1] = np.divide(magA, np.max(np.max(pwr_image)))
#RGB[:, :, 2] = np.divide(magB, np.max(np.max(pwr_image)))
#
## RBG stack
#RGB_stack = np.zeros((len(pwr_image_stack), np.size(pwr_image_stack, axis=1), 3), dtype=float)
#RGB_stack[:, :, 0] = np.divide(pwr_image_stack, np.max(np.max(pwr_image_stack)))
#RGB_stack[:, :, 1] = np.divide(magA_stack, np.max(np.max(pwr_image_stack)))
#RGB_stack[:, :, 2] = np.divide(magB_stack, np.max(np.max(pwr_image_stack)))
#
## pre-stacked radargrams
#plt.figure()
#plt.subplot(311); plt.imshow(pwr_image, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
#plt.subplot(312); plt.imshow(magA, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
#plt.subplot(313); plt.imshow(magB, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
## stacked radargrams
#plt.figure()
#plt.subplot(311); plt.imshow(pwr_image_stack, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
#plt.subplot(312); plt.imshow(magA_stack, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
#plt.subplot(313); plt.imshow(magB_stack, aspect='auto', cmap='gray'); plt.colorbar(); plt.clim([0, 12])
## RBG
#plt.figure()
#plt.imshow(RGB, aspect='auto'); plt.colorbar()
#plt.title('RGB: combined in R, A in G, B in B - normalized to maximum of combined')
## stacked RBG
#plt.figure()
#plt.imshow(RGB_stack, aspect='auto'); plt.colorbar()
#plt.title('stacked RGB: combined in R, A in G, B in B - normalized to maximum of combined')
#plt.show()
 











# combination debug plot showing the picked interface on all power images
if debug:
    plt.figure()
    # power image
    plt.subplot(321)
    plt.imshow(pwr_image, aspect='auto', cmap='gray')
    plt.imshow(FOI, aspect='auto', cmap='jet')
    plt.title('power image')
    if fresnel_stack == 1:
        plt.xlim([3200, 4400])
        plt.ylim([580, 520])
    elif fresnel_stack == 15:
        plt.xlim([75, 300])
        plt.ylim([600, 480])
    # antenna A
    plt.subplot(323)
    plt.title('antenna A image')
    if fresnel_stack == 1:
        plt.imshow(magA, aspect='auto', cmap='gray')
        plt.xlim([3200, 4400])
        plt.ylim([580, 520])
    else:
        plt.imshow(np.divide(fl.stack(magA, fresnel_stack), fresnel_stack), aspect='auto', cmap='gray')
        if fresnel_stack == 15:
            plt.xlim([75, 300])
            plt.ylim([600, 480])
    plt.imshow(FOI, aspect='auto', cmap='jet')
    # antenna B
    plt.subplot(325)
    plt.title('antenna B image')
    if fresnel_stack == 1:
        plt.imshow(magB, aspect='auto', cmap='gray')
        plt.xlim([3200, 4400])
        plt.ylim([580, 520])
    else:
        plt.imshow(np.divide(fl.stack(magB, fresnel_stack), fresnel_stack), aspect='auto', cmap='gray')
        if frensel_stack == 15:
            plt.xlim([75, 300])
            plt.ylim([600, 480])
    plt.imshow(FOI, aspect='auto', cmap='gray')
    # interferogram without roll
    plt.subplot(322)
    plt.title('interferogram without roll')
    plt.imshow(np.rad2deg(int_image_noroll), aspect='auto', cmap='seismic'); plt.clim([-180, 180])
    plt.imshow(FOI, aspect='auto', cmap='jet')
    if fresnel_stack == 1:
        plt.xlim([3200, 4400])
        plt.ylim([580, 520])
    elif fresnel_stack == 15:
        plt.xlim([75, 300])
        plt.ylim([600, 480])
    # roll angles
    plt.subplot(324)
    plt.plot(roll_ang - roll_shift, label='actual')
    plt.plot(roll_ang, label='applied')
    plt.legend()
    plt.title('roll angle [deg]')
    plt.xlim([3200, 4400])
    # interferogram with roll
    plt.subplot(326)
    plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='seismic'); plt.clim([-180, 180])
    plt.imshow(FOI, aspect='auto', cmap='jet')
    plt.title('interferogram with roll')
    if fresnel_stack == 1:
        plt.xlim([3200, 4400])
        plt.ylim([580, 520])
    elif fresnel_stack == 15:
        plt.xlim([75, 300])
        plt.ylim([600, 480])
    plt.show()
