#!/usr/bin/env python3
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

""" Summary

Attempt to implement an interferometry approach to clutter discrimination based
on what has been presented in Haynes et al. (2018). As we can't actually test
with REASON measurements, the goal is to use MARFA.

Usage example

run_interferometry takes as input a transect specification.  With this command,
run_interferometry will prompt you to pick a feature of interest and the surface in a GUI.
Generated intermediate files, such as plots and your picks, will be saved into the folder
out_ZY1b_a

./run_interferometry.py --project GOG3 --line NAQLK/JKB2j/ZY1b/ --plot --mode Reference --save out_ZY1b_a


To reuse these intermediate picks in later runs, use the --pickfile and --refpickfile arguments.
If you don't specify these arguments, you'll be prompted to pick them using the GUI as needed.

This command line reuses the pickfile generated above, but allows you to pick a new reference
pick, which is saved into folder out_ZY1b_b.

./run_interferometry.py --project GOG3 --line NAQLK/JKB2j/ZY1b/ --plot --mode Reference \
                        --pickfile out_ZY1b_a/run_interferometry_FOI_save0.npz --save out_ZY1b_b



# GNG: there seems to be some trouble with the Quit Picking dialog getting frozen.
# Maybe it's not getting closed?

# TODO: support reference picks for reference mode
# TODO: save and load picks in a standard format
# Save intermediate products to named intermediate files

    #'NAQLK/JKB2j/ZY1b/'
    #line = 'GOG3/JKB2j/BWN01a/'
    #project = 'SRH1'
    #line = 'DEV2/JKB2t/Y81a/'
    #project = 'ICP10'
    #line = 'AMY/JKB2u/Y226b/'


"""

import os
import sys
import logging
import argparse
#from tkinter import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

xlib_clutter = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xlib/clutter'))
sys.path.insert(1, xlib_clutter)


import interferometry_funclib as fl
import interface_picker as ip
import pick_interferometry as pickint

class Radargram:
    """ A radargram is an array of radar records and a collection of fast time shifts
    along with fast time shifts due to a variety of effects. Passing these together allows us to
    encapsulate all information together"""
    def __init__(self, ft_shift_chirp=None, ft_shift_coreg=None):
        # Radar records: an n x m samples, where n is fast time axis and m is slow time axis
        self.records = None
        # Fast time shifts
        # Correction due to chirp stability ( 1xm numpy array)
        self.ft_shift_chirp = ft_shift_chirp
        # Correction due to coregistration ( 1xm numpy array)
        self.ft_shift_coreg = ft_shift_coreg

    def apply_ft_shifts(self):
        """ Apply fast time shifts, and return shifts that were applied.
        once a set of shifts is applied, it is removed and returned. """
        ft_shift_total = self.ft_shift_chirp + self.ft_shift_coreg

        # Shift data


        # Clear shifts
        self.ft_shift_chirp = None
        self.ft_shift_coreg = None
        return ft_shift_total

def main():

    """
    #project = 'ICP10'
    #line = 'ICP10/JKB2u/F01T01a/'

    Note: The bxds for this file seems to be the wrong shape
    """

    parser = argparse.ArgumentParser(description='Interferometry')
    parser.add_argument('-p', '--project', default='GOG3',
                        help='Project name')
    parser.add_argument('--line', default='NAQLK/JKB2j/ZY1b/',
                        help='Line name: project/set/transect')
    parser.add_argument('--fresnelstack', type=int, default=15, help='fresnel_stack')
    parser.add_argument('--plot', action='store_true', help='Plot debugging graphs')
    parser.add_argument('--save', default=None, help='Location to save intermediate files and plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose script output')
    parser.add_argument('--targ', default=None, #default='$SDS/targ/xtra/SHARAD',
                        help='targ data base directory')
    parser.add_argument('--mode', default='Roll', choices=('Roll', 'Reference', 'none'),
                        help='Interferogram Correction Mode')
    # TODO: make coregistration algorithm flag a more friendly string name
    # TODO: update coregistration to be the accepted default
    parser.add_argument('--coregmethod', default=0, type=int, choices=range(8),
                        help="Coregistration method option")
    parser.add_argument('--coregifactor', default=10, type=int,
                        help="Coregistration interpolation upsampling factor")
    parser.add_argument('--pickfile', help="Use pick file generated by pick_interferometry.py")
    parser.add_argument('--refpickfile', help="Use reference pick file")

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO

    fmtstr = 'ri: [%(levelname)-5s] %(message)s'
    logging.basicConfig(level=loglevel, format=fmtstr, stream=sys.stdout)
    # Set default number of decimal places to display for numpy arrays
    np.set_printoptions(precision=3)

    project = args.project #'GOG3'
    line = args.line
    bplot = args.plot
    debug = True
    gain = 'low' # TODO: try with gain = 'high'
    fresnel_stack = args.fresnelstack #15
    fc = 60E6
    B = 19
    fs = 50E6
    roll_shift = 0
    FOI_selection_method = 'maximum'
    interferogram_correction_mode = args.mode
    roll_correction = True

    sds = os.getenv('SDS', '/disk/kea/SDS')
    wais = os.getenv('WAIS', '/disk/kea/WAIS')

    if args.targ is None:
        args.targ = os.path.join(sds, 'targ/xtra/SHARAD')

    focpath = os.path.join(wais, 'targ/xtra', project, 'FOC/Best_Versions')
    if project in ('SRH1', 'ICP9', 'ICP10'):
        snm, chirp_bp = 'RADnh5', True
    else:
        snm, chirp_bp = 'RADnh3', False
    rawpath = os.path.join(wais, 'orig/xlob', line.rstrip('/'), snm) + '/'
    tregpath = os.path.join(wais, 'targ/treg', line,  'TRJ_JKB0/')
    chirppath = os.path.join(focpath, 'S4_FOC')
    print('chirppath = ' + chirppath)
    print('tregpath = ' + tregpath)
    print('rawpath = ' + rawpath)

    LINE_PARAMS = {
        'NAQLK/JKB2j/ZY1b': {
            'trim': [0, 1000, 0, 12000],
            'chirpwin': [120, 150],
        },
        'GOG3/JKB2j/BWN01b': {
            'trim': [0, 1000, 0, 15000],
            'chirpwin': [120, 150],
        },
        'GOG3/JKB2j/BWN01a': {
            'trim': [0, 1000, 15000, 27294],
            'chirpwin': [120, 150],
        },
        '__DEFAULT__': {
            'trim': [0, 1000, 0, 0],
            'chirpwin': [0, 200],
        },
    }

    lineparm = LINE_PARAMS.get(line.rstrip('/'), LINE_PARAMS['__DEFAULT__'])
    trim = lineparm['trim']
    chirpwin = lineparm['chirpwin']

    mb_offsets = {'GOG3': 155.6, 'SRH1': 127.5, 'ICP10': 127.5}
    mb_offset = mb_offsets.get(project, mb_offsets['SRH1'])

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)

    if args.pickfile:
        # TODO: validate pick parameters against interferometry parameters from
        # argparse and make sure they match.
        print('Loading picks from ' + args.pickfile)
        with np.load(args.pickfile) as data:
            FOI = data['FOI']
            SRF = data['SRF']
            trim = data['trim']
        # check the length of the picked FOI
        Nf = pickint.FOI_picklen(FOI)
    else:
        # Pick an FOI, and save it to a cache file for reuse
        if args.save is not None:
            savefile = os.path.join(args.save, 'run_interferometry_FOI_save0.npz')
            logging.info("Caching picks to %s", savefile)
        else:
            savefile = None
        FOI, SRF, Nf = pickint.select_foi_and_srf(line=line, path=focpath,
                       chan=pickint.CHANNELS[gain], fresnel_stack=args.fresnelstack,
                       FOI_selection_method=FOI_selection_method,
                       trim=trim, bplot=args.plot, debug=debug, savefile=savefile)


    # CHIRP STABILITY ASSESSMENT AND COREGISTRATION ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('CHIRP STABILITY ASSESSMENT AND COREGISTRATION')

    # Test to see if co-registered data product already exists for this PST
    pst = line.split('/') # TODO: make 'A' more dsecriptive (pst)
    logging.info("project=%s pst=%s trim=%s", project, str(pst), str(trim))
    savedir = "." if args.save is None else args.save
    post_coreg = os.path.join(savedir, "{0:s}_{1[0]:s}_{1[1]:s}_{1[2]:s}_{2[2]:d}to{2[3]:d}_AfterCoregistration" \
                 ".npz".format(project, pst, trim))

    if not os.path.exists(post_coreg):

        # Load and range compress the interpolated 1m data products
        print('-- load and range compress raw radar data')
        trimchirp = list(trim)
        trimchirp[0] = 0 #trimchirp[0] = 4
        # trimchirp[1] = max(1000, chirpwin[1]) # limit to 1000 to allow plots to plot.
        trimchirp[1] = chirpwin[1] + (chirpwin[1] - chirpwin[0])
        geo_path = os.path.join(focpath, 'S2_FIL', line)
        dechirpA, dechirpB = fl.denoise_and_dechirp(gain, trimchirp, rawpath, geo_path=geo_path,
                                                    chirp_path=os.path.join(chirppath, line), do_cinterp=False, bp=chirp_bp)
        if bplot and debug:
            plt.figure()
            xmax = min(1000, dechirpA.shape[0]) # plot at most first 1000 samples
            plt.subplot(211)
            plt.imshow(20 * np.log10(np.abs(dechirpA[0:xmax, :])), aspect='auto', cmap='gray')
            plt.title('Antenna A')
            plt.subplot(212)
            plt.imshow(20 * np.log10(np.abs(dechirpB[0:xmax, :])), aspect='auto', cmap='gray')
            plt.title('Antenna B')
            if args.save is not None:
                outfile = os.path.join(args.save, 'chirp_stability1.png')
                plt.savefig(outfile, dpi=300)
                print("Saving plot to " + outfile)

            plt.show()



        # Extract the loop-back chirp
        print('-- extract the loop-back chirp')
        loopbackA = dechirpA[chirpwin[0]:chirpwin[1], :]
        loopbackB = dechirpB[chirpwin[0]:chirpwin[1], :]
        del dechirpA, dechirpB
        if bplot:
            # Get slow time indices of first, middle, and last traces
            refja = [0, loopbackA.shape[1] // 2, loopbackA.shape[1] - 1]
            refjb = [0, loopbackB.shape[1] // 2, loopbackB.shape[1] - 1]
            plt.figure()
            plt.subplot(411); plt.imshow(20 * np.log10(np.abs(loopbackA)), aspect='auto', cmap='gray'); plt.title('Loopback A Magnitude [dB]')
            plt.subplot(412); plt.imshow(np.angle(loopbackA), aspect='auto', cmap='jet'); plt.title('Loopback A Phase')
            plt.subplot(413); plt.imshow(20 * np.log10(np.abs(loopbackB)), aspect='auto', cmap='gray'); plt.title('Loopback B Magnitude [dB]')
            plt.subplot(414); plt.imshow(np.angle(loopbackB), aspect='auto', cmap='jet'); plt.title('Loopback B Phase')
            plt.figure()
            plt.plot(20 * np.log10(np.abs(loopbackA[:, refja[0]])), label='loopbackA - ' + str(refja[0]))
            plt.plot(20 * np.log10(np.abs(loopbackA[:, refja[1]])), label='loopbackA - ' + str(refja[1]))
            plt.plot(20 * np.log10(np.abs(loopbackA[:, refja[2]])), label='loopbackA - ' + str(refja[2]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, refjb[0]])), label='loopbackB - ' + str(refjb[0]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, refjb[1]])), label='loopbackB - ' + str(refjb[1]))
            plt.plot(20 * np.log10(np.abs(loopbackB[:, refjb[2]])), label='loopbackB - ' + str(refjb[2]))
            plt.legend()
            plt.title('range compressed loopback chirp')
            plt.xlabel('fast-time sample')
            plt.ylabel('magnitude [dB]')
            del refja, refjb

            if args.save is not None:
                outfile = os.path.join(args.save, 'chirp_loopback1.png')
                plt.savefig(outfile, dpi=300)
                print("Saving plot to " + outfile)

            plt.show()

        # Assess the phase stability of the loop-back chirp for each antenna
        method = 'xcorr2'
        print('-- characterize chirp stability ({:s})'.format(method))
        stabilityA = fl.chirp_phase_stability(loopbackA[:, 0], loopbackA, method=method, rollval=20)
        stabilityB = fl.chirp_phase_stability(loopbackB[:, 0], loopbackB, method=method, rollval=20)
        if bplot:
            plt.figure()
            plt.plot(np.arange(0, loopbackA.shape[1]), stabilityA, label='Antenna A')
            plt.plot(np.arange(0, loopbackB.shape[1]), stabilityB, label='Antenna B')
            plt.xlabel('Range Line #')
            plt.ylabel('Shift for Optimal Chirp Stability')
            plt.legend()
            if args.save is not None:
                outfile = os.path.join(args.save, 'chirp_stability2.png')
                plt.savefig(outfile, dpi=300)
                logging.info("Saving chirp stability plot to %s", outfile)
            plt.show()
        del loopbackA, loopbackB

        print('A chirp shift: mean={:0.3f}, std dev={:0.3g}'.format( \
              np.mean(stabilityA), np.std(stabilityA)))
        print('B chirp shift: mean={:0.3f}, std dev={:0.3g}'.format( \
              np.mean(stabilityB), np.std(stabilityB)))


        if args.save is not None:
            savefile = os.path.join(args.save, 'run_interferometry_FOI_save0.npz')
            stability_save_file = os.path.join(args.save, 'chirp_phase_stability.npz')
            np.savez(stability_save_file, stabilityA=stabilityA, stabilityB=stabilityB)
            logging.info("Saved chirp phase stability data to %s", stability_save_file)



        # Load the focused SLC 1m port and starboard radargrams
        print('-- load port and starboard single-look products')
        if gain == 'low':
            chan1, chan2 = '5', '7'
        elif gain == 'high':
            chan1, chan2 = '6', '8'
        else: #pragma: no cover
            assert False
        marfa_pth = os.path.join(focpath, 'S4_FOC')
        cmp_a = fl.convert_to_complex(*fl.load_marfa(line, chan1, pth=marfa_pth, trim=trim))
        cmp_b = fl.convert_to_complex(*fl.load_marfa(line, chan2, pth=marfa_pth, trim=trim))

        # TODO: complete reorganization to use Radargram class. GNG
        #rdr_a = Radargram(cmp_a, ft_shift_chirp=stabilityA)
        #rdr_b = Radargram(cmp_b, ft_shift_chirp=stabilityB)


        if bplot and debug:
            mag_a, phs_a = fl.convert_to_magphs(cmp_a)
            mag_b, phs_b = fl.convert_to_magphs(cmp_b)
            plt.figure()
            plt.subplot(411); plt.imshow(mag_a, aspect='auto', cmap='gray')
            plt.title('antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(412); plt.imshow(np.rad2deg(phs_a), aspect='auto', cmap='seismic')
            plt.title('antenna A phase'); plt.clim([-180, 180]); plt.colorbar()
            plt.subplot(413); plt.imshow(mag_b, aspect='auto', cmap='gray')
            plt.title('antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(414); plt.imshow(np.rad2deg(phs_b), aspect='auto', cmap='seismic')
            plt.title('antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
            RGB = np.zeros((len(mag_a), np.size(mag_a, axis=1), 3), dtype=float)
            RGB[:, :, 0] = np.divide(mag_a, np.max(np.max(mag_a)))
            RGB[:, :, 2] = np.divide(mag_b, np.max(np.max(mag_a)))
            plt.figure()
            plt.subplot(211); plt.imshow(mag_a, aspect='auto', cmap='gray')
            plt.title('antenna A magnitude'); plt.clim([0, 20])
            plt.subplot(212); plt.imshow(mag_b, aspect='auto', cmap='gray')
            plt.title('antenna B magnitude'); plt.clim([0, 20])
            #plt.subplot(313); plt.imshow(RGB, aspect='auto')
            #plt.title('RGB image with normalized mag_a in R and normalized mag_b in B\n'
            # 'normalization to max amplitude in mag_a')

            if args.save is not None:
                outfile = os.path.join(args.save, 'magphs1.png')
                plt.savefig(outfile, dpi=300)
                logging.info("Saving magphs plot to %s", outfile)

            plt.show()
            del mag_a, phs_a, mag_b, phs_b

        # Apply shifts calculated from chirp stability analysis to align range lines
        # TODO GNG: allow us to omit this and pass cmp_a and stabilityA onward
        print('-- chirp stability adjustment')
        cmp_a2 = fl.phase_stability_adjustment(cmp_a, stabilityA)
        cmp_b2 = fl.phase_stability_adjustment(cmp_b, stabilityB)
        del cmp_a, cmp_b, stabilityA, stabilityB
        if bplot and debug:
            mag_a, phs_a = fl.convert_to_magphs(cmp_a2)
            mag_b, phs_b = fl.convert_to_magphs(cmp_b2)
            plt.figure()
            plt.subplot(411); plt.imshow(mag_a, aspect='auto', cmap='gray')
            plt.title('A radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(412); plt.imshow(np.rad2deg(phs_a), aspect='auto', cmap='seismic')
            plt.title('A phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
            plt.subplot(413); plt.imshow(mag_b, aspect='auto', cmap='gray')
            plt.title('B radargram after chirp stability adjustment'); plt.colorbar(); plt.clim([0, 20])
            plt.subplot(414); plt.imshow(np.rad2deg(phs_b), aspect='auto', cmap='seismic')
            plt.title('B phase after chirp stability adjustment'); plt.colorbar(); plt.clim([-180, 180])
            plt.show()
            if args.save is not None:
                outfile = os.path.join(args.save, 'chirp_stability3.png')
                plt.savefig(outfile, dpi=300)
                logging.info("Saving plot to %s", outfile)
            del mag_a, phs_a, mag_b, phs_b

        # Sub-pixel co-registration of the port and starboard range lines
        # TODO GNG: make fl.coregistration take cmp_a, cmp_b, and stabilityA and stabilityB as inputs,
        # and output shift_array and qual_array as outputs
        print('-- co-registration of port and starboard radargrams')
        myshift = 300*args.coregifactor
        cmp_a3, cmp_b3, shift_array, qual_array, qual_array2 = fl.coregistration(cmp_a2, cmp_b2, (1 / 50E6),
              args.coregifactor, shift=myshift, method=args.coregmethod)
        del cmp_a2, cmp_b2
        if args.save is not None:
            logging.info("Saving coregistration result to %s", post_coreg)
            np.savez(post_coreg, cmp_a3=cmp_a3, cmp_b3=cmp_b3, shift_array=shift_array, qual=qual_array, qual2=qual_array2)
        if bplot:
            plt.figure()
            plt.subplot(211); plt.plot(shift_array)
            plt.subplot(212); plt.plot(qual_array)
            if args.save is not None:
                outfile = os.path.join(args.save, 'coregistration1.png')
                plt.savefig(outfile, dpi=300)
                print("Saving plot to " + outfile)

            plt.show()
    else:

        # Load previously coregistered datasets
        print('-- loading previously co-registered MARFA datasets')
        logging.debug("Reading " + post_coreg)
        with np.load(post_coreg) as temp:
            cmp_a3 = temp['cmp_a3']
            cmp_b3 = temp['cmp_b3']

    del post_coreg
    if bplot and debug:
        mag_a, phs_a = fl.convert_to_magphs(cmp_a3)
        mag_b, phs_b = fl.convert_to_magphs(cmp_b3)
        plt.figure()
        plt.subplot(411); plt.imshow(mag_a, aspect='auto', cmap='gray')
        plt.title('co-registered antenna A magnitude'); plt.colorbar(); plt.clim([0, 20])
        plt.subplot(412); plt.imshow(np.rad2deg(phs_a), aspect='auto', cmap='seismic')
        plt.title('co-registered antenna A phase'); plt.clim([-180, 180]); plt.colorbar()
        plt.subplot(413); plt.imshow(mag_b, aspect='auto', cmap='gray')
        plt.title('co-registered antenna B magnitude'); plt.colorbar(); plt.clim([0, 20])
        plt.subplot(414); plt.imshow(np.rad2deg(phs_b), aspect='auto', cmap='seismic')
        plt.title('co-registered antenna B phase'); plt.clim([-180, 180]); plt.colorbar()
        del mag_a, phs_a, mag_b, phs_b
        if args.save is not None:
            outfile = os.path.join(args.save, 'coregistration2.png')
            plt.savefig(outfile, dpi=300)
            print("Saving plot to " + outfile)
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
    #az_step = int(fl.independent_azimuth_samples(cmp_a3, cmp_b3, FOI))
    #az_step = int(az_step)
    az_step = 8
    print('   > azimuth samples between independent range lines:', str(az_step))

    if interferogram_correction_mode == 'Roll':
        # Roll correction
        print('-- derive roll correction')
        norm_path = os.path.join(focpath, 'S1_POS', line)
        roll_phase, roll_ang = fl.roll_correction(np.divide(299792458, fc), B, trim, tregpath, \
                               norm_path, roll_shift=roll_shift)
        if bplot and debug:
            plt.figure()
            plt.subplot(211)
            plt.plot(np.linspace(0, 1, len(roll_ang)), np.rad2deg(roll_ang))
            plt.title('roll angle [deg]'); plt.xlim([0, 1])
            plt.subplot(212)
            plt.plot(np.linspace(0, 1, len(roll_ang)), np.rad2deg(roll_phase))
            plt.title('roll correction interferometric phase angle [deg]')
            plt.xlim([0, 1])
            plt.show()

        # TODO: make a function that rolls data if taking a shift.
        #  make this take cmp_a and a shift, cmp_b and a shift,
        # and compute a total shifted product cmp_b3


        # Interferogram

        print('-- producing interferogram')
        int_image = fl.stacked_interferogram(cmp_a3, cmp_b3, fresnel_stack, roll_phase,
                                             roll_correction, az_step=az_step)
        if bplot and debug:
            int_image_noroll = fl.stacked_interferogram(cmp_a3, cmp_b3, fresnel_stack, roll_phase, False, az_step=az_step)
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

        roll_ang = np.zeros((np.size(cmp_a3, axis=1)), dtype=float)

        if not args.refpickfile:
            # Pick reference surface from the combined power image and save if requested
            savefile = os.path.join(args.save, 'run_interferometry_referencepicks.npz') if args.save else None
            pwr_image, reference = pickint.select_ref(line, path=focpath, chan=pickint.CHANNELS[gain], fresnel_stack=args.fresnelstack,
                                           trim=trim, FOI_selection_method='maximum', bplot=bplot, debug=debug,
                                           savefile=savefile)
        else:
            pth = os.path.join(focpath, 'S4_FOC/')
            pwr_image, _ = fl.load_power_image(line, channel=pickint.CHANNELS[gain],
                                               trim=trim, fresnel=args.fresnelstack, mode='averaged', pth=pth)
            logging.info("Load reference surface pick from %s", args.refpickfile)
            with np.load(args.refpickfile) as refpicks:
                # Old ones used to be called arr_0
                try:
                    reference = refpicks['reference']
                except KeyError: # pragma: no cover
                    reference = refpicks['arr_0']

        reference = np.transpose(reference)

        if bplot and debug:
            plt.figure()
            plt.imshow(pwr_image, aspect='auto', cmap='gray')
            plt.title('power image with picked reference surface')
            plt.imshow(reference, aspect='auto')
            plt.colorbar()
            plt.show()

        # Create uncorrected interferogram
        print('-- producing interferogram')
        uncorr_interferogram = fl.stacked_interferogram(cmp_a3, cmp_b3, fresnel_stack, np.zeros((np.size(cmp_a3, axis=1)), dtype=float), False, az_step=az_step)
        if bplot and debug:
            plt.figure()
            plt.imshow(np.rad2deg(uncorr_interferogram), aspect='auto', cmap='hsv')
            plt.title('uncorrected interferogram [deg]'); plt.clim([-180, 180]); plt.colorbar()
            plt.imshow(FOI, aspect='auto')
            plt.imshow(reference, aspect='auto')
            plt.show()

        # Normalize uncorrected interferogram
        print('-- normalizing interferogram')
        int_image, reference_phase, reference = fl.interferogram_normalization(uncorr_interferogram, reference)
        if bplot and debug:
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
    if args.save is not None:
        plt.savefig(os.path.join(args.save, 'ri_int_image.png'), dpi=600)
        out_filename = os.path.join(args.save, 'ri_int_image.npz')
        np.savez(out_filename, int_image=int_image)
        logging.info("Saved %s", out_filename)
        try: # save noroll if it exists.
            out_filename = os.path.join(args.save, 'ri_int_image_noroll.npz')
            np.savez(out_filename, int_image_noroll=int_image_noroll)
        except UnboundLocalError: # pragma: no cover
            logging.info("int_image_noroll was not produced as part of this run, so not saving intermediate output.")



    # Correlation map
    print('-- producing correlation map')
    corrmap = fl.stacked_correlation_map(cmp_a3, cmp_b3, fresnel_stack, az_step=az_step)
    del cmp_a3, cmp_b3
    if bplot:
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
        plt.show()

    ## save correlation map
    if args.save is not None:
        plt.savefig(os.path.join(args.save, 'ri_corrmap.png'), dpi=600)
        out_filename = os.path.join(args.save, 'ri_corrmap.npz')
        np.savez(out_filename, corrmap=corrmap)
        logging.info("Saved %s", out_filename)


    # Extract feature-of-interest interferometric phase and correlation
    # as well as the mean interferometric phase of the feature-of-interest
    # as if it were off-nadir clutter
    print('-- extract information')
    FOI_phs = fl.FOI_extraction(int_image, FOI)
    FOI_cor = fl.FOI_extraction(np.abs(corrmap), FOI)
    SRF_phs = fl.offnadir_clutter(FOI, SRF, roll_ang, fresnel_stack, B, mb_offset, np.divide(299792458, fc), np.divide(1, fs))
    if bplot and debug:
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
    Nf = int(np.round(np.divide(Nf, az_step)) + 1)
    print('   > unwrapped cross-track surface clutter mean interferometric phase:',
          str(np.round(mean_srf, 3)))
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
    if bplot:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('interferometric phase angle [deg]')
        ax1.set_ylabel('histogram counts', color='b')
        ax1.hist(np.rad2deg(FOI_phs), bins=50, color='b', label='observed data')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('pds [$\phi$]', color='r')
        iphid = np.rad2deg(iphi)
        ax2.plot(iphid, nadir_emp_pdf, '--r', label='nadir empirical pdf', linewidth=10)
        ax2.plot(iphid, srf_emp_pdf, ':r', label='off-nadir clutter empirical pdf', linewidth=10)
        ax2.plot(iphid, obs_emp_pdf, '-g', label='observed empirical pdf', linewidth=10)
        ax2.tick_params(axis='y', labelcolor='r')
        plt.ylim([0, np.max([1.1 * np.max(nadir_emp_pdf), 1.1 * np.max(obs_emp_pdf)])])
        fig.tight_layout()
        plt.xlim([-180, 180])
        plt.legend(loc=1)
        plt.show()

    # Determine the uncertainty in the nadir empirical sample mean
    print('-- determine uncertainty in the nadir empirical sample mean')
    nadir_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma)
    #obs_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, phi_m=mean_phi)
    #srf_sigma_m = fl.empirical_sample_mean(N, Nf, iphi, gamma, phi_m=mean_srf)
    print('   > uncertainty in nadir empirical sample mean:', str(np.round(nadir_sigma_m, 3)))
    #print('   > uncertainty in cross-track surface clutter empirical sample mean:',
     # str(np.round(srf_sigma_m, 3)))
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

if __name__ == "__main__":
    main()
