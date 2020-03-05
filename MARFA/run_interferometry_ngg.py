__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu', 'Gregory Ng, ngg@ig.utexas.edu']
__version__ = '0.2'
__history__ = {
    '0.1':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'First build on interferometry processor'},
    '0.2':
        {'date': 'March 1, 2020',
         'author': 'Gregory Ng, UTIG',
         'info': 'Reorganizing'}}
"""
attempt to implement an interferometry approach to clutter discrimination based
on what has been presented in Haynes et al. (2018). As we can't actually test
with REASON measurments, the goal is to use MARFA.
# GNG: there seems to be some trouble with the Quit Picking dialog getting frozen.
# Maybe it's not getting closed?
"""

import sys
sys.path.insert(0, '../xlib/clutter/')
import os
import logging
import argparse

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


def main():
    #project = 'ICP10'
    #line = 'AMY/JKB2u/Y226b/'

    """
    #project = 'ICP10'
    #line = 'ICP10/JKB2u/F01T01a/'


    FEATURE OF INTEREST SELECTION
    -- load combined and focused radar product
    Traceback (most recent call last):
      File "run_interferometry_ngg.py", line 91, in <module>
        pwr_image, lim = fl.load_power_image(line, '1', trim, fresnel_stack, 'averaged', pth=path + 'S4_FOC/')
      File "../xlib/clutter/interferometry_funclib.py", line 166, in load_power_image
        mag, phs = load_marfa(line, channel, pth)
      File "../xlib/clutter/interferometry_funclib.py", line 51, in load_marfa
        mag = np.transpose(np.reshape(mag, (ncol, nsamp))) / 10000
      File "/disk/kea/SDS/syst/ext/lib/python3.7/site-packages/numpy/core/fromnumeric.py", line 292, in reshape
        return _wrapfunc(a, 'reshape', newshape, order=order)
      File "/disk/kea/SDS/syst/ext/lib/python3.7/site-packages/numpy/core/fromnumeric.py", line 56, in _wrapfunc
        return getattr(obj, method)(*args, **kwds)
    ValueError: cannot reshape array of size 1981809664 into shape (619315,3200)
    """

    parser = argparse.ArgumentParser(description='Interferometry')
    parser.add_argument('--fresnelstack', type=int, default=15, help='fresnel_stack')
    parser.add_argument('--plot', action='store_true', help='Plot debugging graphs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose script output')
    parser.add_argument('--targ', default='/disk/kea/SDS/targ/xtra/SHARAD', help='targ data base directory')
    parser.add_argument('--mode', default='Roll', choices=('Roll','Reference', 'none'), help='Interferogram Correction Mode')
    parser.add_argument('-p', '--project', default='GOG3',
                        help='Project name')
    parser.add_argument('--line', default='NAQLK/JKB2j/ZY1b',
                        help='Line name: project/set/transect')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    project = 'GOG3'
    line = 'NAQLK/JKB2j/ZY1b/'
    #line = 'GOG3/JKB2j/BWN01a/'
    #project = 'SRH1'
    #line = 'DEV2/JKB2t/Y81a/'
    bplot = False
    bsave = True
    debug = True
    gain = 'low'
    fresnel_stack = args.fresnelstack #15
    fc = 60E6
    B = 19
    fs = 50E6
    roll_shift = 0
    FOI_selection_method = 'maximum'
    interferogram_correction_mode = 'Roll'
    roll_correction = True

    path = '/disk/kea/WAIS/targ/xtra/' + project + '/FOC/Best_Versions/'
    if project in ('SRH1', 'ICP9', 'ICP10'):
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
        #trim = [0, 1000, 0, 12000]
        trim = [0, 1000, 0, 6000]
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
    print('FOI_cache_file = ' + FOI_cache_file)
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
    post_coreg = "{0:s}_{1[0]:s}_{1[1]:s}_{1[2]:s}_{2[2]:d}to{2[3]:d}_AfterCoregistration" \
                 ".npz".format(project, A, trim)

    if os.path.exists(post_coreg):

        # Load and range compress the interpolated 1m data products
        print('-- load and range compress raw radar data')
        print('chirpwin: ' + str(chirpwin))
        trimchirp = list(trim)
        trimchirp[0] = None # no fast-time trimming
        trimchirp[1] = None
        #trimchirp[0] = 0 #trimchirp[0] = 4
        #trimchirp[1] = max(1000, chirpwin[1]) # limit to 1000 to allow plots to plot.
        dechirpA, dechirpB = fl.denoise_and_dechirp(gain, trimchirp, rawpath, path + 'S2_FIL/' + line, chirppath + line, do_cinterp=False, bp=chirp_bp)
        if bplot and debug:
            plt.figure()
            xmax = min(1000, dechirpA.shape[0]) # plot at most first 1000 samples
            plt.subplot(211)
            plt.imshow(20 * np.log10(np.abs(dechirpA[0:xmax, :])), aspect='auto', cmap='gray')
            plt.title('Antenna A')
            plt.subplot(212)
            plt.imshow(20 * np.log10(np.abs(dechirpB[0:xmax, :])), aspect='auto', cmap='gray')
            plt.title('Antenna B')
            plt.show()

        # Extract the loop-back chirp
        print('-- extract the loop-back chirp')
        loopbackA = dechirpA[chirpwin[0]:chirpwin[1], :]
        loopbackB = dechirpB[chirpwin[0]:chirpwin[1], :]
        del dechirpA, dechirpB
        if bplot:
            tempA = [0, int(np.round(np.size(loopbackA, axis=1) / 2)), np.size(loopbackA, axis=1) - 1]
            tempB = [0, int(np.round(np.size(loopbackB, axis=1) / 2)), np.size(loopbackB, axis=1) - 1]
            plt.figure()
            plt.subplot(411); plt.imshow(20 * np.log10(np.abs(loopbackA)), aspect='auto', cmap='gray'); plt.title('Loopback A Magnitude [dB]')
            plt.subplot(412); plt.imshow(np.angle(loopbackA), aspect='auto', cmap='jet'); plt.title('Loopback A Phase')
            plt.subplot(413); plt.imshow(20 * np.log10(np.abs(loopbackB)), aspect='auto', cmap='gray'); plt.title('Loopback B Magnitude [dB]')
            plt.subplot(414); plt.imshow(np.angle(loopbackB), aspect='auto', cmap='jet'); plt.title('Loopback B Phase')
            plt.figure()
            plt.plot(20 * np.log10(np.abs(loopbackA[:, tempA[0]])), label='loopbackA - ' + str(tempA[0]))
            plt.plot(20 * np.log10(np.abs(loopbackA[:, tempA[1]])), label='loopbackA - ' + str(tempA[1]))
            plt.plot(20 * np.log10(np.abs(loopbackA[:, tempA[2]])), label='loopbackA - ' + str(tempA[2]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, tempB[0]])), label='loopbackB - ' + str(tempB[0]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, tempB[1]])), label='loopbackB - ' + str(tempB[1]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, tempB[2]])), label='loopbackB - ' + str(tempB[2]))
            plt.legend()
            plt.title('range compressed loopback chirp')
            plt.xlabel('fast-time sample')
            plt.ylabel('magnitude [dB]')
            del tempA, tempB
            plt.show()

        # Assess the phase stability of the loop-back chirp for each antenna
        method = 'xcorr2' 
        print('-- characterize chirp stability ({:s})'.format(method))
        stabilityA = fl.chirp_phase_stability(loopbackA[:, 0], loopbackA, method=method, rollval=20)
        stabilityB = fl.chirp_phase_stability(loopbackB[:, 0], loopbackB, method=method, rollval=20)
        if bplot:
            plt.figure()
            plt.plot(np.arange(0, np.size(loopbackA, axis=1)), stabilityA, label='Antenna A')
            plt.plot(np.arange(0, np.size(loopbackB, axis=1)), stabilityB, label='Antenna B')
            plt.xlabel('Range Line #')
            plt.ylabel('Shift for Optimal Chirp Stability')
            plt.legend()
            plt.show()

        print('A chirp shift: mean={:0.3f}, std dev={:0.3g}'.format(np.mean(stabilityA), np.std(stabilityA)))
        print('B chirp shift: mean={:0.3f}, std dev={:0.3g}'.format(np.mean(stabilityB), np.std(stabilityB)))

        stability_save_file = 'chirp_phase_stability.npz'
        np.savez(stability_save_file, stabilityA=stabilityA, stabilityB=stabilityB)
        print("Saved data to " + stability_save_file)



        # Load the focused SLC 1m port and starboard radargrams
        print('-- load port and starboard single-look products')
        if gain == 'low':
            magA, phsA = fl.load_marfa(line, '5', pth=path + 'S4_FOC/')
            magB, phsB = fl.load_marfa(line, '7', pth=path + 'S4_FOC/')
        elif gain == 'high':
            magA, phsA = fl.load_marfa(line, '6', pth=path + 'S4_FOC/')
            magB, phsB = fl.load_marfa(line, '8', pth=path + 'S4_FOC/')
        magA = magA[trim[0]:trim[1], trim[2]:trim[3]]
        phsA = phsA[trim[0]:trim[1], trim[2]:trim[3]]
        magB = magB[trim[0]:trim[1], trim[2]:trim[3]]
        phsB = phsB[trim[0]:trim[1], trim[2]:trim[3]]
        cmpA = fl.convert_to_complex(magA, phsA)
        cmpB = fl.convert_to_complex(magB, phsB)
        del magA, phsA, magB, phsB
        if bplot and debug:
            magA, phsA = fl.convert_to_magphs(cmpA)
            magB, phsB = fl.convert_to_magphs(cmpB)
            plt.figure()
            plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
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

        # Apply shifts calculated from chirp stability analysis to align 
        # range lines
        print('-- chirp stability adjustment')
        cmpA2 = fl.phase_stability_adjustment(cmpA, stabilityA)
        cmpB2 = fl.phase_stability_adjustment(cmpB, stabilityB)
        if bplot and debug:
            magA, phsA = fl.convert_to_magphs(cmpA2)
            magB, phsB = fl.convert_to_magphs(cmpB2)
            plt.figure()
            plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('A radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='seismic'); plt.title('A phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
            plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('B radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='seismic'); plt.title('B phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
            plt.show()
            del magA, phsA, magB, phsB

        # Sub-pixel co-registration of the port and starboard range lines
        print('-- co-registration of port and starboard radargrams')
        cmpA3, cmpB3, shift_array = fl.coregistration(cmpA2, cmpB2, (1 / 50E6), 10)
        if bsave:
            print('Saving to ' + post_coreg)
            np.savez(post_coreg, cmpA3=cmpA3, cmpB3=cmpB3, shift_array=shift_array)


    else:

        # Load previously coregistered datasets
        print('-- loading previously co-registered MARFA datasets')
        with np.load(post_coreg) as temp:
            cmpA3 = temp['cmpA3']
            cmpB3 = temp['cmpB3']

    del post_coreg
    if bplot and debug:
        magA, phsA = fl.convert_to_magphs(cmpA3)
        magB, phsB = fl.convert_to_magphs(cmpB3)
        plt.figure()
        plt.subplot(411); plt.imshow(magA, aspect='auto', cmap='gray'); plt.title('co-registered antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
        plt.subplot(412); plt.imshow(np.rad2deg(phsA), aspect='auto', cmap='seismic'); plt.title('co-registered antenna A phase'); plt.clim([-180, 180]); plt.colorbar() 
        plt.subplot(413); plt.imshow(magB, aspect='auto', cmap='gray'); plt.title('co-registered antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
        plt.subplot(414); plt.imshow(np.rad2deg(phsB), aspect='auto', cmap='seismic'); plt.title('co-registered antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
        plt.show()

    print('CHIRP STABILITY ASSESSMENT AND COREGISTRATION -- complete')
    print(' ')

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------




    # INTERFEROMETRY ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print('INTERFEROMETRY')

    # Determine the number of azimuth samples between independent looks
    print('-- determine azimuth samples between independent range lines')
    #az_step = int(fl.independent_azimuth_samples(cmpA3, cmpB3, FOI))
    az_step = 8
    print('   > azimuth samples between independent range lines:', str(az_step))

    if interferogram_correction_mode == 'Roll':
       
        # Roll correction 
        print('-- derive roll correction')
        roll_phase, roll_ang = fl.roll_correction(np.divide(299792458, fc), B, trim, tregpath, path + 'S1_POS/' + line, roll_shift=roll_shift)
        if debug:
            plt.figure()
            plt.subplot(211)
            plt.plot(np.linspace(0, 1, len(roll_ang)), np.rad2deg(roll_ang))
            plt.title('roll angle [deg]'); plt.xlim([0, 1])
            plt.subplot(212)
            plt.plot(np.linspace(0, 1, len(roll_ang)), np.rad2deg(roll_phase))
            plt.title('roll correction interferometric phase angle [deg]'); plt.xlim([0, 1])
            plt.show()

        # Interferogram
        print('-- producing interferogram')
        int_image = fl.stacked_interferogram(cmpA3, cmpB3, fresnel_stack, roll_phase, roll_correction, az_step=az_step)
        if debug:
            int_image_noroll = fl.stacked_interferogram(cmpA3, cmpB3, fresnel_stack, roll_phase, False, az_step=az_step)
            plt.figure()
            plt.subplot(313); plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='hsv'); plt.colorbar()
            plt.title('interferogram with defined roll correction [deg]'); plt.clim([-180, 180]); plt.imshow(FOI, aspect='auto')
            plt.subplot(311); plt.imshow(np.rad2deg(int_image_noroll), aspect='auto', cmap='hsv'); plt.colorbar()
            plt.title('interferogram no roll [deg]'); plt.clim([-180, 180]); plt.imshow(FOI, aspect='auto')
            plt.subplot(312)
            plt.plot(np.linspace(0, 1, len(roll_ang)), roll_ang - roll_shift, label='actual')
            plt.plot(np.linspace(0, 1, len(roll_ang)), roll_ang, label='applied')
            plt.title('roll angle [deg]'); plt.xlim([0, 1]); plt.legend()
            plt.show()

    elif interferogram_correction_mode == 'Reference':

        roll_ang = np.zeros((np.size(cmpA3, axis=1)), dtype=float)

        # Pick reference surface from the combined power image
        print('-- pick reference surface')
        #reference = np.transpose(np.load('SRH_Y81a_referencepicks_15Stack.npy'))
        reference = ip.picker(np.transpose(pwr_image), snap_to=FOI_selection_method)
        #np.save('SRH_Y81a_referencepicks_15Stack.npy', reference)
        reference = np.transpose(reference)
        if debug:
            plt.figure()
            plt.imshow(pwr_image, aspect='auto', cmap='gray')
            plt.title('power image with picked reference surface')
            plt.imshow(reference, aspect='auto')
            plt.colorbar()
            plt.show()

        # Create uncorrected interferogram
        print('-- producing interferogram')
        uncorr_interferogram = fl.stacked_interferogram(cmpA3, cmpB3, fresnel_stack, np.zeros((np.size(cmpA3, axis=1)), dtype=float), False, az_step=az_step)
        if debug:
            plt.figure()
            plt.imshow(np.rad2deg(uncorr_interferogram), aspect='auto', cmap='hsv')
            plt.title('uncorrected interferogram [deg]'); plt.clim([-180, 180]); plt.colorbar()
            plt.imshow(FOI, aspect='auto')
            plt.imshow(reference, aspect='auto')
            plt.show()

        # Normalize uncorrected interferogram
        print('-- normalizing interferogram')
        int_image, reference_phase, reference = fl.interferogram_normalization(uncorr_interferogram, reference)
        if debug:
            plt.figure()
            plt.subplot(311); plt.imshow(np.rad2deg(uncorr_interferogram), aspect='auto', cmap='hsv')
            plt.title('uncorrected interferogram [deg]'); plt.clim([-180, 180]); plt.colorbar()
            plt.imshow(FOI, aspect='auto'); plt.imshow(reference, aspect='auto')
            plt.subplot(312)
            plt.plot(np.arange(0, len(reference_phase)), np.rad2deg(reference_phase))
            plt.title('reference interferometric phase [deg]')
            plt.xlim([0, len(reference_phase)])
            plt.subplot(313); plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='hsv')
            plt.title('corrected interferogram [deg]'); plt.clim([-180, 180]); plt.colorbar()
            plt.imshow(FOI, aspect='auto'); plt.imshow(reference, aspect='auto')
            plt.show()

    # save interferogram
    if bsave:
        plt.savefig('ri_ngg__int_image.png', dpi=600)
        out_filename = 'ri_ngg__int_image.npz'
        np.savez(out_filename, int_image=int_image)
        print("Saved " + out_filename)



    # Correlation map
    print('-- producing correlation map')
    corrmap = fl.stacked_correlation_map(cmpA3, cmpB3, fresnel_stack, az_step=az_step)
    if True:
        RGB = np.zeros((len(corrmap), np.size(corrmap, axis=1), 3), dtype=float)
        RGB[:, :, 2] = np.divide(np.abs(corrmap), np.max(np.max(np.abs(corrmap))))
        RGB[:, :, 1] = 1
        RGB[:, :, 0] = np.divide(np.rad2deg(int_image) + 180, 360)
        plt.figure()
        plt.subplot(311)
        plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='hsv'); plt.title('interferogram'); plt.colorbar()
        plt.imshow(FOI, aspect='auto'); plt.imshow(SRF, aspect='auto')
        plt.subplot(312)
        plt.imshow(np.abs(corrmap), aspect='auto', cmap='nipy_spectral'); plt.title('correlation map'); plt.colorbar()
        plt.imshow(FOI, aspect='auto'); plt.imshow(SRF, aspect='auto')
        plt.subplot(313)
        plt.imshow(col.hsv_to_rgb(RGB), aspect='auto', cmap='hsv'); plt.colorbar()
        plt.imshow(FOI, aspect='auto'); plt.imshow(SRF, aspect='auto')
        plt.title('HSV image with interferogram in Hue and correlation in Values\n Saturation set to 1')
        #plt.show()

    # save correlation map
    if bsave:
        plt.savefig('ri_ngg_corrmap.png', dpi=600)
        out_filename = 'ri_ngg__corrmap.npz'
        np.savez(out_filename, corrmap=corrmap)
        print("Saved " + out_filename)


    # Extract feature-of-interest interferometric phase and correlation 
    # as well as the mean interferometric phase of the feature-of-interest
    # as if it were off-nadir clutter
    print('-- extract information')
    FOI_phs = fl.FOI_extraction(int_image, FOI)
    FOI_cor = fl.FOI_extraction(np.abs(corrmap), FOI)
    SRF_phs = fl.offnadir_clutter(FOI, SRF, roll_ang, fresnel_stack, B, mb_offset, np.divide(299792458, fc), np.divide(1, fs))
    if debug:
        plt.figure()
        plt.subplot(211); plt.hist(np.rad2deg(FOI_phs), bins=20)
        plt.title('distribution of interferometric phase angles')
        plt.xlim([-180, 180])
        plt.subplot(212); plt.hist(FOI_cor, bins=50)
        plt.title('distribution of interferometric correlation')
        plt.show()
    mean_phi = np.rad2deg(np.mean(FOI_phs))
    mean_srf = np.rad2deg(np.mean(SRF_phs))
    if mean_srf < 0:
        mean_srf = np.multiply(360, (mean_srf / 360) + np.floor(np.abs(mean_srf / 360)))
    else:
        mean_srf = np.multiply(360, (mean_srf / 360) - np.floor(np.abs(mean_srf / 360)))
    gamma = np.mean(FOI_cor)
    N = int(np.floor(np.divide(fresnel_stack, az_step)))
    if N != 1:
        N = N + 1
    Nf = int(np.round(np.divide(N, az_step)) + 1)
    print('   > unwrapped cross-track surface clutter mean interferometric phase:', str(np.round(mean_srf, 3)))
    print('   > FOI mean interferometric phase:', str(np.round(mean_phi, 3)))
    print('   > FOI mean interferometric correlation:', str(np.round(gamma, 3)))
    print('   > number of independent looks:', str(N))
    print('   > number of independent multi-looked pixels used to define the FOI:', str(Nf))

    # Calculate nadir, FOI and off-nadir surface clutter empirical interferometric
    # phase PDFs
    print('-- calculate empirical interferometric phase PDFs')
    _, nadir_emp_pdf = fl.empirical_pdf(fc, B, N, gamma)
    _, srf_emp_pdf = fl.empirical_pdf(fc, B, N, gamma, phi_m=mean_srf)
    iphi, obs_emp_pdf = fl.empirical_pdf(fc, B, N, gamma, phi_m=mean_phi)
    if True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('interferometric phase angle [deg]')
        ax1.set_ylabel('histogram counts', color='b')
        ax1.hist(np.rad2deg(FOI_phs), bins=50, color='b', label='observed data')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('pds [$\phi$]', color='r')
        ax2.plot(np.rad2deg(iphi), nadir_emp_pdf, '--r', label='nadir empirical pdf', linewidth=10)
        ax2.plot(np.rad2deg(iphi), srf_emp_pdf, ':r', label='off-nadir clutter empirical pdf', linewidth=10)
        ax2.plot(np.rad2deg(iphi), obs_emp_pdf, '-g', label='observed empirical pdf', linewidth=10)
        ax2.tick_params(axis='y', labelcolor='r')
        plt.ylim([0, np.max([1.1 * np.max(nadir_emp_pdf), 1.1 * np.max(obs_emp_pdf)])])
        fig.tight_layout()
        plt.xlim([-180, 180])
        plt.legend(loc=1)
        #plt.show()
      
    # Determine the uncertainty in the nadir empirical sample mean
    print('-- determine uncertainty in the nadir empirical sample mean')
    nadir_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma)
    #obs_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, phi_m=mean_phi)
    #srf_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, phi_m=mean_srf)
    print('   > uncertainty in nadir empirical sample mean:', str(np.round(nadir_sigma_m, 3)))
    #print('   > uncertainty in cross-track surface clutter empirical sample mean:', str(np.round(srf_sigma_m, 3)))
    #print('   > uncertainty in observed empirical sample mean:', str(np.round(obs_sigma_m, 3)))
        
    print('INTERFEROMETRY -- complete')
    print(' ')

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # Compare the uncertainty in the nadir distirbution to the mean FOI interferometric
    # phase angle to determine whether the FOI is at nadir or not
    if nadir_sigma_m < np.abs(0.5 * mean_phi):
        print('>> FOI can be distinguished from an identical feature at nadir')
        print('   > mean interferometric phase angle:', np.round(mean_phi, 3), 'degrees')
    else:
        print('>> FOI cannot be distinguished from a nadir feature')  
    print(' ')

    ## Compare the uncertainty in the nadir distribution to the mean surface interferometric
    ## phase angle to determine whether the FOI is at off-nadir or not
    #if mean_srf <= -180:
    #    test_phi = np.abs(-360 - mean_srf)
    #elif mean_srf >= 180:
    #    test_phi = -1 * (360 - mean_srf)
    #else:
    #    test_phi = mean_srf
    #if nadir_sigma_m + test_phi < np.abs(0.5 * mean_phi):
    #    print('>> FOI can be distinguished from an identical off-nadir clutter feature')
    #    print('   > mean interferometric phase angle:', np.round(mean_phi, 3), 'degrees')
    #else:
    #    print('>> FOI cannot be distinguished from an off-nadir clutter feature')
    #print(' ')
    if bplot:
        plt.show()
    
if __name__ == "__main__":
    main()
