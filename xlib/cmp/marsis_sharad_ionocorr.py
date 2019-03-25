#!/usr/bin/env python3

__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'dual-frequency ionospheric correction'},
    '1.1':
        {'date': 'March 20 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'Reorganized for noninteractive work'}
}

b_plot=False

import os
import numpy as np
import importlib.util

if b_plot:
    # GNG: on melt:
    # ModuleNotFoundError: No module named 'matplotlib'
    import matplotlib.pyplot as plt

# I think we can just use a string and os.path.exists
from pathlib import Path
import pandas as pd

''' 
Algorithm attempting to produce an ionospheric correction using the delay
offset between MARSIS and SHARAD data collected at the same SZA in the same
geogrpahic region.

data files for the two instruments have their own unique structure;
MARSIS
     -- solar zenith angle
     -- spacecraft altitude [km]
     -- spacecraft latitude [deg]
     -- spacecraft longitude [deg]
     -- band1 centre frequency identifier
     -- band2 centre frequency identifier
     -- rx window in current frame (item 1) - expressed in terms of samples
     -- rx window in current frame (item 2) - expressed in terms of samples
     -- band1 data (expressed in the frequency domain)
     -- band2 data (expressed in the frequency domain)
SHARAD
     -- solar zenith angle
     -- spacecraft altitude [km]
     -- spacecraft latitude [deg]
     -- spacecraft longitude [deg]
     -- rx window opening times - expressed in terms of time [s]
     -- data (expressed in the frequency domain)

I want to stick as close as possible to the L2 REASON flowchart for
accomplishing a similar goal. Obviously some steps will have to be altered but
the general procedure should be followed
'''

spec = importlib.util.spec_from_file_location('marsis_sharad_ionocorr_funclib', 'marsis_sharad_ionocorr_funclib.py')
L2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L2)

area = 'OlympiaUndae'

sha_thrsh = 4.5
mar_thrsh = 4
#sha_thrsh = 4
#mar_thrsh = 4
mar_bands = '4.0 5.0'
minSZA = 60
maxSZA = 100
dSZA = 0.5
chunk_sharad = True
chunk_sharad_size = 10000
trim_sharad = False
trim_sharad_number = 100
tec_plot = False
debug = False
verbose = True

# define the sza bins and identify range lines by sza bin
sza_bin = np.zeros((7, int((maxSZA - minSZA) / dSZA)))
sza_bin[0, :] = np.arange(60, 100, dSZA, dtype=float)

# define altimetry error output
altimetry_error = np.zeros((10, int((maxSZA - minSZA) / dSZA)))
altimetry_error[0, :] = np.arange(60, 100, dSZA, dtype=float)



# import MOLA dataset for the NE quarter of MARS
if area == 'NP AREA1':
    molaS = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr44n090hb.img')
    molaN = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr88n090hb.img')
    mola = np.concatenate((molaN, molaS), axis=0); del molaS, molaN
    mola_limits = [0, 88, 90, 180]
#    mola = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr90n000gb.img')
#    mola_limits = [0, 90, 0, 180]
elif area == 'OlympiaUndae':
    molaW = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr88n090hb.img')
    molaE = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr88n180hb.img')
    mola = np.concatenate((molaW, molaE), axis=1); del molaW, molaE
    mola_limits = [44, 88, 90, 270]
#    molaW = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr90n000gb.img')
#    molaE = L2.load_mola('/disk/kea/SDS/orig/supl/xtra-pds/MOLA/megr90n180gb.img')
#    mola = np.concatenate((molaW, molaE), axis=1); del molaW, molaE
#    mola_limits = [0, 90, 0, 360]

# normalize MOLA to IAU2000
mola = L2.normalize_mola(mola, mola_limits, ddeg=128)

SHARAD_root="/disk/kea/SDS/targ/xtra/SHARAD"

# look into chunking the data in order to be able to load the data within each
# SZA bin
chunks = np.zeros((1, int((maxSZA - minSZA) / dSZA)))
if chunk_sharad:
#    ii = 0
#    if ii == 0:
    for ii in range(np.size(sza_bin, axis=1)):
        # GNG: did you mean for there to be a space in here, because this causes problems.
        # GNG:_ perhaps
        hdf_path1 = Path(os.path.join(SHARAD_root, 'sza', area, str(dSZA) + ' dSZA', 
            str(sza_bin[0, ii]) + '-' + str(sza_bin[0, ii] + dSZA, 'data.h5') )
        if hdf_path1.exists():
            temp = np.array(pd.read_hdf(hdf_path1, 'sza'))
            chunks[0, ii] = np.ceil(len(temp) / chunk_sharad_size)
            del temp
else:
    for ii in range(np.size(sza_bin, axis=1)):
        chunks[0, ii] = 1
chunks = chunks.astype(int)

# loop through the various SZA bins and calculate a TEC
#ii = 39
#if ii == 39:
for ii in range(np.size(sza_bin, axis=1)):
#for ii in range(0, 50):
#for ii in [24, 70, 79]:
    
    path1 = Path('/disk/kea/SDS/targ/xtra/MARSIS/sza/' + area + '/' + str(dSZA) + ' dSZA/' + 
            str(sza_bin[0, ii]) + '-' + str(sza_bin[0, ii] + dSZA) + '/data.npz')

    if path1.exists():
        path2 = Path('/disk/kea/SDS/targ/xtra/SHARAD/sza/' + area + '/' + str(dSZA) + ' dSZA/' + 
            str(sza_bin[0, ii]) + '-' + str(sza_bin[0, ii] + dSZA) + '/data.h5')
        if path2.exists():

#            jj = 0
#            if jj == 0:
            for jj in range(chunks[0, ii]):

                if verbose and chunks[0, ii] == 1:
                    print('Working SZA bin -', sza_bin[0, ii], 'to', sza_bin[0, ii] + dSZA)
                elif verbose and chunks[0, ii] != 1:
                    print('Working SZA bin -', sza_bin[0, ii], 'to', sza_bin[0, ii] + dSZA, '- chunk', jj + 1, 'of', chunks[0, ii])
                iteration = 0
                new_TECU = 0
                dTECU = 1

                while np.abs(dTECU) > 0.05:

                    TECU = new_TECU

                    # load the data correpsonding to the particular SZA bin
                    # under analysis
                    mar_data = np.load('/disk/kea/SDS/targ/xtra/MARSIS/sza/' + area + '/' + str(dSZA) + ' dSZA/' + str(sza_bin[0, ii]) + '-' + str(sza_bin[0, ii] + dSZA) + '/data.npz')
                    sha_fn = '/disk/kea/SDS/targ/xtra/SHARAD/sza/' + area + '/' + str(dSZA) + ' dSZA/' + str(sza_bin[0, ii]) + '-' + str(sza_bin[0, ii] + dSZA) + '/data.h5'

                    # chunk the raw sharad data
                    start_ind = jj * chunk_sharad_size
                    sha_data_sza = np.array(pd.read_hdf(sha_fn, 'sza'))
                    if chunks[0, ii] != jj:
                        end_ind = (jj + 1) * chunk_sharad_size
                    else:
                        end_ind = len(sha_data_sza)
                    sha_data_sza = sha_data_sza[start_ind:end_ind]
                    sha_data_scalt = np.array(pd.read_hdf(sha_fn, 'altitude'))[start_ind:end_ind]
                    sha_data_lat = np.array(pd.read_hdf(sha_fn, 'latitude'))[start_ind:end_ind]
                    sha_data_lon = np.array(pd.read_hdf(sha_fn, 'longitude'))[start_ind:end_ind]
                    sha_data_rxwt = np.array(pd.read_hdf(sha_fn, 'rxwot'))[start_ind:end_ind]
                    sha_data_real = np.array(pd.read_hdf(sha_fn, 'real'))[start_ind:end_ind, :]
                    sha_data_imag = np.array(pd.read_hdf(sha_fn, 'imag'))[start_ind:end_ind, :]
                    sha_data_data = sha_data_real + 1j * sha_data_imag
                    del sha_data_real, sha_data_imag

                    # range compress the radar data
                    mar_b1_rngcmp = L2.rngcmp(mar_data['arr_8'], [0.7E6, 1.4E6, 250E-6, 1E6], 'MARSIS')
                    mar_b2_rngcmp = L2.rngcmp(mar_data['arr_9'], [0.7E6, 1.4E6, 250E-6, 1E6], 'MARSIS')
                    sha_rngcmp = L2.rngcmp(sha_data_data, [20E6, (1 / 0.0375E-6), 85.05E-6, 10E6], 'SHARAD')
                    del sha_data_data
                    if debug:
                        plt.figure()
                        plt.subplot(1, 3, 1); plt.imshow(np.transpose(np.abs(mar_b1_rngcmp)), aspect='auto')
                        plt.subplot(1, 3, 2); plt.imshow(np.transpose(np.abs(mar_b2_rngcmp)), aspect='auto')
                        plt.subplot(1, 3, 3); plt.imshow(np.transpose(np.abs(sha_rngcmp)), aspect='auto')

                    # ionosphere correct the radar data
                    # (TECU of 0 for iteration 0)
                    mar_b1_ioncorr = L2.ioncorr(mar_b1_rngcmp, TECU * 1E16, [0.7E6, 1.4E6, 250E-6, 1E6], 'MARSIS')
                    mar_b2_ioncorr = L2.ioncorr(mar_b2_rngcmp, TECU * 1E16, [0.7E6, 1.4E6, 250E-6, 1E6], 'MARSIS')
                    sha_ioncorr = L2.ioncorr(sha_rngcmp, TECU * 1E16, [20E6, (1 / 0.0375E-6), 85.05E-6, 10E6], 'SHARAD')
                    del mar_b1_rngcmp, mar_b2_rngcmp, sha_rngcmp
                    if debug:
                        plt.figure()
                        plt.subplot(1, 3, 1); plt.imshow(np.transpose(np.abs(mar_b1_ioncorr)), aspect='auto')
                        plt.subplot(1, 3, 2); plt.imshow(np.transpose(np.abs(mar_b2_ioncorr)), aspect='auto')
                        plt.subplot(1, 3, 3); plt.imshow(np.transpose(np.abs(sha_ioncorr)), aspect='auto')
#                        plt.figure();
#                        plt.subplot(1, 3, 1); plt.imshow(np.transpose(10 * np.log10(np.abs(mar_b1_ioncorr))), aspect='auto')
#                        plt.subplot(1, 3, 2); plt.imshow(np.transpose(10 * np.log10(np.abs(mar_b2_ioncorr))), aspect='auto')
#                        plt.subplot(1, 3, 3); plt.imshow(np.transpose(10 * np.log10(np.abs(sha_ioncorr))), aspect='auto')

                    # pick the surface echo
                    mar_b1_surfpick = L2.surfpick(mar_b1_ioncorr, mar_thrsh, 0, 50)
                    mar_b2_surfpick = L2.surfpick(mar_b2_ioncorr, mar_thrsh, 0, 50)
                    sha_surfpick = L2.surfpick(sha_ioncorr, sha_thrsh, 0, 200)
                    if debug:
                        plt.figure()
                        plt.subplot(1, 3, 1); plt.imshow(np.transpose(np.abs(mar_b1_ioncorr)), aspect='auto'); plt.plot(mar_b1_surfpick, 'r')
                        plt.subplot(1, 3, 2); plt.imshow(np.transpose(np.abs(mar_b2_ioncorr)), aspect='auto'); plt.plot(mar_b2_surfpick, 'r')
                        plt.subplot(1, 3, 3);  plt.imshow(np.transpose(np.abs(sha_ioncorr)), aspect='auto'); plt.plot(sha_surfpick, 'r')

#                    # -------------------------------------------------------------------------------
#                    # pick the surface echo from dB radargram
#                    mar_b1_surfpickdB = L2.surfpick(10 * np.log10(np.abs(mar_b1_ioncorr)), 1.5, 0, 50)
#                    mar_b2_surfpickdB = L2.surfpick(10 * np.log10(np.abs(mar_b2_ioncorr)), 1.5, 0, 50)
#                    sha_surfpickdB = L2.surfpick(10 * np.log10(np.abs(sha_ioncorr)), 1.5, 0, 200)
#                    if True:
#                        plt.figure()
#                        plt.subplot(1, 3, 1); plt.imshow(10 * np.log10(np.transpose(np.abs(mar_b1_ioncorr))), aspect='auto'); plt.plot(mar_b1_surfpickdB, 'r')
#                        plt.subplot(1, 3, 2); plt.imshow(10 * np.log10(np.transpose(np.abs(mar_b2_ioncorr))), aspect='auto'); plt.plot(mar_b2_surfpickdB, 'r')
#                        plt.subplot(1, 3, 3);  plt.imshow(10 * np.log10(np.transpose(np.abs(sha_ioncorr))), aspect='auto'); plt.plot(sha_surfpickdB, 'r')
#                    # -------------------------------------------------------------------------------

                    # NaN surface picks for MARSIS bands that won't be used in
                    # the analysis if so desired
                    if mar_bands != '1.8 3.0 4.0 5.0':
                        mar_b1_surfpick = L2.marsis_band_mute(mar_b1_surfpick, mar_data['arr_4'], mar_bands, '1')
                        mar_b2_surfpick = L2.marsis_band_mute(mar_b2_surfpick, mar_data['arr_5'], mar_bands, '2')
                        if debug:
                            plt.figure()
                            plt.subplot(1, 3, 1); plt.imshow(np.transpose(np.abs(mar_b1_ioncorr)), aspect='auto'); plt.plot(mar_b1_surfpick, 'r')
                            plt.subplot(1, 3, 2); plt.imshow(np.transpose(np.abs(mar_b2_ioncorr)), aspect='auto'); plt.plot(mar_b2_surfpick, 'r')
                            plt.subplot(1, 3, 3); plt.imshow(np.transpose(np.abs(sha_ioncorr)), aspect='auto'); plt.plot(sha_surfpick, 'r')

                    # trim to a number of SHARAD range lines that is equal to
                    # the number of MARSIS range lines that have picked
                    # surfaces to try and speed things up if so desired
                    if trim_sharad:
                        if len(mar_b1_surfpick) >= len(sha_surfpick):
                            if verbose:
                                print('Trimming the SHARAD data is not recommended - More MARSIS range lines than SHARAD - skipping')
                        else:
                            sha_surfpick = L2.sharad_trim(sha_surfpick, mar_b1_surfpick, mar_b2_surfpick, 'B', trim_sharad_number)
                            if debug:
                                plt.imshow(np.transpose(np.abs(sha_ioncorr)), aspect='auto'); plt.plot(sha_surfpick)

                    # extract MOLA topography for each range line having a
                    # surface pick
                    mar_b1_topo = L2.topography(mar_b1_surfpick, mar_data['arr_2'], mar_data['arr_3'], mola, mola_limits)
                    mar_b2_topo = L2.topography(mar_b2_surfpick, mar_data['arr_2'], mar_data['arr_3'], mola, mola_limits)
                    sha_topo = L2.topography(sha_surfpick, sha_data_lat, sha_data_lon, mola, mola_limits)
                    del mar_b1_ioncorr, mar_b2_ioncorr, sha_ioncorr
                    if debug:
                        plt.figure(); plt.plot(mar_b1_topo); plt.plot(mar_b2_topo); plt.plot(sha_topo)

                    # calculate the TEC for each combination of MARSIS and
                    # SHARAD range lines
                    sha_mar_b1_TEC = L2.tec_calc(TECU * 1E16, sha_surfpick, sha_topo, sha_data_scalt, sha_data_rxwt, mar_b1_surfpick, mar_b1_topo, mar_data['arr_1'], mar_data['arr_6'], mar_data['arr_4'], '1')
                    sha_mar_b2_TEC = L2.tec_calc(TECU * 1E16, sha_surfpick, sha_topo, sha_data_scalt, sha_data_rxwt, mar_b2_surfpick, mar_b2_topo, mar_data['arr_1'], mar_data['arr_7'], mar_data['arr_5'], '2')
                    sha_mar_TEC = np.concatenate((sha_mar_b1_TEC, sha_mar_b2_TEC), axis=0)
                    if tec_plot:
                        plt.figure()
                        plt.subplot(2, 1, 1); plt.hist(sha_mar_b1_TEC, bins=100); plt.hist(sha_mar_b2_TEC, bins=100)
                        plt.subplot(2, 1, 2); plt.hist(sha_mar_TEC, bins=100)
#
                    # extract an estimate estimate for optimal TEC from the bin
                    # centre for the histogram bin exhibiting the greatest
                    # number of counts
                    TECU_est, low_std, high_std = L2.optimal_tecu(sha_mar_TEC, 100)

                    # update TECU and re-enter the loop if necessary
                    new_TECU = TECU + TECU_est
                    if verbose:
                        print('--- after iteration', iteration, ', TEC estimate is', new_TECU, 'TECU')
                    if iteration == 0 and new_TECU == 0:
                        dTECU = TECU_est
                    elif iteration == 0 and new_TECU != 0:
                        dTECU = new_TECU - TECU
                    else:
                        dTECU = new_TECU - TECU
    #                    dTECU = 0.01
                    iteration += 1
#                    dTECU = 0.01

                if jj == 0:
                    sza_bin[1, ii] = len(mar_b1_topo)      # number of marsis range lines within the SZA window
                    sza_bin[2, ii] = len(sha_topo)         # number of sharad range lines within the SZA window
                    sza_bin[3, ii] = len(sha_mar_TEC)      # number of marsis/sharad combinations used in TEC estimation
                    sza_bin[4, ii] = new_TECU              # optimal TECU estimate
                    sza_bin[5, ii] = low_std               # TECU one standard deviation below the optimal value
                    sza_bin[6, ii] = high_std              # TECU one standard deviation above the optimal value
                    if verbose:
                        print('--- best TEC estimate is', new_TECU, 'TECU')
                else:
                    sza_bin[2, ii] = sza_bin[2, ii] + len(sha_topo)         # number of sharad range lines within the SZA window
                    sza_bin[3, ii] = sza_bin[3, ii] + len(sha_mar_TEC)      # number of marsis/sharad combinations used in TEC estimation
                    sza_bin[4, ii] = np.mean([sza_bin[4, ii], new_TECU])    # optimal TECU estimate
                    sza_bin[5, ii] = np.mean([sza_bin[5, ii], low_std])     # TECU one standard deviation below the optimal value
                    sza_bin[6, ii] = np.mean([sza_bin[6, ii], high_std])    # TECU one standard deviation above the optimal value
                    if verbose:
                        print('--- best TEC estimate is', sza_bin[4, ii], 'TECU')

                # In an effort to understand the error bars associated with the
                # TECU estimates, we want to characterize the accuracy of the
                # surface pick for each range line used in TECU calculation. To
                # do this, we want to compare the elevation of the picked
                # surface with that determined from MOLA.
                As = L2.altimetry_error(sha_data_scalt, sha_topo, sha_surfpick, sha_data_rxwt, sza_bin[4, ii], 'SHARAD')
                m1_diff = L2.altimetry_error(mar_data['arr_1'], mar_b1_topo, mar_b1_surfpick, mar_data['arr_6'], sza_bin[4, ii], 'MARSIS', '1', mar_data['arr_4'])
                m2_diff = L2.altimetry_error(mar_data['arr_1'], mar_b2_topo, mar_b2_surfpick, mar_data['arr_7'], sza_bin[4, ii], 'MARSIS', '2', mar_data['arr_5'])
                del sha_data_scalt, sha_topo, sha_surfpick, sha_data_rxwt
                del mar_data, mar_b1_topo, mar_b1_surfpick
                del mar_b2_topo, mar_b2_surfpick
                del sha_data_lat, sha_data_lon, sha_data_sza
                if jj == 0:
                    sha_diff = As
                else:
                    sha_diff = np.concatenate((sha_diff, As), axis=0)

    # output the mean and bounds on the altimetry errors for all rangelines
    # used in TECU calculation
    # altimetry error --> the difference between the expected one-way
    #                     MRO-surface distance (from the reported MRO altitude
    #                     and MOLA) and the picked distance from the radar data
    #                     after optimal ionospheric correction.
    # the altimetry_error output array is defined as follows:
    # row 0 - lower bound of SZA bin
    # row 1 - mean SHARAD altimetry error [km]
    # row 2 - SHARAD altimetry error one Gaussian std. dev. below the mean [km]
    # row 3 - SHARAD altimetry error one Gaussian std. dev. above the mean [km]
    # row 4 - mean MARSIS B1 altimetry error [km]
    # row 5 - MARSIS B1 altimetry error one Gaussian std. dev. below the mean [km]
    # row 6 - MARSIS B1 altimetry error one Gaussian std. dev. above the mean [km]
    # row 7 - mean MARSIS B2 altimetry error [km]
    # row 8 - MARSIS B2 altimetry error one Gaussian std. dev. below the mean [km]
    # row 9 - MARSIS B2 altimetry error one Gaussian std. dev. above the mean [km]
    inds = np.argwhere(np.isnan(sha_diff) == False)
    indm1 = np.argwhere(np.isnan(m1_diff) == False)
    indm2 = np.argwhere(np.isnan(m2_diff) == False)
    temp0, temp1, temp2 = L2.optimal_tecu(sha_diff[inds[:, 0]], 100)
    altimetry_error[1, ii] = temp0 * 1E16
    altimetry_error[2, ii] = temp1 * 1E16
    altimetry_error[3, ii] = temp2 * 1E16
    del temp0, temp1, temp2
    temp0, temp1, temp2 = L2.optimal_tecu(m1_diff[indm1[:, 0]], 100)
    altimetry_error[4, ii] = temp0 * 1E16
    altimetry_error[5, ii] = temp1 * 1E16
    altimetry_error[6, ii] = temp2 * 1E16
    del temp0, temp1, temp2
    temp0, temp1, temp2 = L2.optimal_tecu(m2_diff[indm2[:, 0]], 100)
    altimetry_error[7, ii] = temp0 * 1E16
    altimetry_error[8, ii] = temp1 * 1E16
    altimetry_error[9, ii] = temp2 * 1E16
    del temp0, temp1, temp2
    del inds, indm1, indm2, As, sha_diff, m1_diff, m2_diff


if b_plot:
    # plot the result
    fig, ax1 = plt.subplots()
    plt.title('Combined Geometric SHARAD/MARSIS TEC Estimate in ' + area)
    ax1.bar(sza_bin[0, :], sza_bin[2, :], edgecolor='k', width=dSZA, color='lightsteelblue')
    ax1.bar(sza_bin[0, :], sza_bin[1, :], bottom=sza_bin[2, :], edgecolor='k', width=dSZA, color='rosybrown')
    ax1.set_xlabel('SZA')
    ax1.set_ylabel('Number of Available Range Lines')
    plt.legend(('SHARAD', 'MARSIS'))
    ax2 = ax1.twinx()
    ax2.errorbar(sza_bin[0,:], sza_bin[4,:], yerr=np.abs(sza_bin[5:7,:]), ecolor='k', fmt='o', c='r', elinewidth=1)
    ax2.set_ylabel('TECU [TEC/1.0E16]')

# compare with SHARAD-only TEC results
if area == 'NP AREA1':
    SHARAD = np.load('SHARAD_NP_AREA1_rdrtec.npz')
elif area == 'OlympiaUndae':
    SHARAD = np.load('SHARAD_OlympiaUndae_rdrtec.npz')
SHARAD = SHARAD['arr_0']

if b_plot:
    plt.figure()
    plt.scatter(sza_bin[4, :], SHARAD[2, :])
    plt.plot([-1, 1], [-1, 1])
    plt.xlabel('MARSIS/SHARAD Time of Flight TECU [TEC/1E16]'); plt.xlim([-0.2, 0.7])
    plt.ylabel('SHARAD Autofocus TECU [TEC/1E16]'); plt.ylim([-0.2, 0.7])

    # plot altimetry errors for the three datasets
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # sharad
    ax1.fill_between(altimetry_error[0, :], altimetry_error[3, :], altimetry_error[2, :], facecolor=[0.9, 0.9, 0.9])
    ax1.plot(altimetry_error[0, :], altimetry_error[1, :], 'k', linewidth=2)
    ax1.plot(altimetry_error[0, :], altimetry_error[3, :], c='k', linestyle=':', linewidth=1)
    ax1.plot(altimetry_error[0, :], altimetry_error[2, :], c='k', linestyle=':', linewidth=1)
    ax1.set_xlim([np.min(altimetry_error[0, :]), np.max(altimetry_error[0, :])])
    ax1.set_ylim([-4, 2])
    ax1.grid()
    ax1.set_title('SHARAD')
    ax1.set_ylabel('Altimetry Error [km]')
    ax1.legend(('Mean', '+/- 1 $\sigma$'), loc=1)
    # marsis b1
    ax2.fill_between(altimetry_error[0, :], altimetry_error[6, :], altimetry_error[5, :], facecolor=[0.9, 0.9, 0.9])
    ax2.plot(altimetry_error[0, :], altimetry_error[4, :], 'k', linewidth=2)
    ax2.plot(altimetry_error[0, :], altimetry_error[6, :], c='k', linestyle=':', linewidth=1)
    ax2.plot(altimetry_error[0, :], altimetry_error[5, :], c='k', linestyle=':', linewidth=1)
    ax2.set_xlim([np.min(altimetry_error[0, :]), np.max(altimetry_error[0, :])])
    ax2.set_ylim([-4, 2])
    ax2.grid()
    ax2.set_title('MARSIS B1')
    ax2.set_ylabel('Altimetry Error [km]')
    ax2.legend(('Mean', '+/- 1 $\sigma$'), loc=1)
    # marsis b2
    ax3.fill_between(altimetry_error[0, :], altimetry_error[9, :], altimetry_error[8, :], facecolor=[0.9, 0.9, 0.9])
    ax3.plot(altimetry_error[0, :], altimetry_error[7, :], 'k', linewidth=2)
    ax3.plot(altimetry_error[0, :], altimetry_error[9, :], c='k', linestyle=':', linewidth=1)
    ax3.plot(altimetry_error[0, :], altimetry_error[8, :], c='k', linestyle=':', linewidth=1)
    ax3.set_xlim([np.min(altimetry_error[0, :]), np.max(altimetry_error[0, :])])
    ax3.set_ylim([-4, 2])
    ax3.set_xlabel('SZA')
    ax3.set_title('MARSIS B2')
    ax3.set_ylabel('Altimetry Error [km]')
    ax3.grid()
    ax3.legend(('Mean', '+/- 1 $\sigma$'), loc=1)


""" The expected results of the marsis_sharad_ionocorr.py analysis are two numpy 
arrays (sza_bin and altimetry_error) that are used to produce final plots or can be 
manually saved for later. Columns in these two arrays refer to specific SZA bins 
(between minSZA and maxSZA with a step size of dSZA) and rows contain the relevant 
information;

A) sza_bin

-- row 0: number of MARSIS range lines within the SZA bin
-- row 1: number of SHARAD range lines within the SZA bin
-- row 2: number of MARSIS/SHARAD combinations used in TEC estimation
-- row 3: optimal TECU estimate
-- row 4: TECU one Gaussian standard deviation above the optimal value
-- row 5: TECU one Gaussian standard deviation below the optimal value

B) altimetry_error

-- row 0: lower bound of the SZA bin
-- row 1: mean SHARAD altimetry error [km]
-- row 2: SHARAD altimetry error one Gaussian standard deviation below the mean [km]
-- row 3: SHARAD altimetry error one Gaussian standard deviation above the mean [km] 
-- row 4: mean  MARSIS Band 1 altimetry error [km]
-- row 5:  MARSIS Band 1 altimetry error one Gaussian standard deviation below the mean [km]
-- row 6: MARSIS Band 1 altimetry error one Gaussian standard deviation above the mean [km] 
-- row 7: mean  MARSIS Band 2 altimetry error [km]
-- row 8:  MARSIS Band 2 altimetry error one Gaussian standard deviation below the mean [km]
-- row 9: MARSIS Band 2 altimetry error one Gaussian standard deviation above the mean [km] 

"""
