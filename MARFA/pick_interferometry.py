#!/usr/bin/env python3
__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu', 'Gregory Ng, ngg@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'March 1, 2020',
         'author': 'Gregory Ng, UTIG',
         'info': 'Separate interferometry picking into its own script'}}
"""
attempt to implement an interferometry approach to clutter discrimination based
on what has been presented in Haynes et al. (2018). As we can't actually test
with REASON measurments, the goal is to use MARFA.
# GNG: there seems to be some trouble with the Quit Picking dialog getting frozen.
# Maybe it's not getting closed?

Usage example

To pick feature of interest and the surface above:

./pick_interferometry.py --project GOG3 --line NAQLK/JKB2j/ZY1b -o pick_NAQLK_JKB2j_ZY1b.npz

To pick a reference feature (usually the surface):

./pick_interferometry.py --project GOG3 --line NAQLK/JKB2j/ZY1b -t ref -o pick_ref_NAQLK_JKB2j_ZY1b.npz


"""

import os
import logging
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as col
#from tkinter import *

clutter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xlib.clutter')))
sys.path.insert(1, clutter_path)
import interferometry_funclib as fl
import interface_picker as ip



def select_foi_and_srf(line, path, chan, fresnel_stack, trim,
                       FOI_selection_method='maximum', bplot=True, debug=False,
                       savefile=None):
    """ FOI, SRF, Nf = select_foi() """

    # FEATURE OF INTEREST SELECTION ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print(' ')
    print('FEATURE OF INTEREST SELECTION')

    # Load the combined and focused 1m MARFA data product and stack to the
    # desired trace spacing in preparation for feature detection and selection
    print('-- load combined and focused radar product')
    pth = os.path.join(path, 'S4_FOC/')
    logging.debug("Reading from %s", pth)
    pwr_image, lim = fl.load_power_image(line, chan, trim, fresnel_stack, 'averaged', pth=pth)
    if trim[3] == 0:
        trim[3] = lim

    # load_power_image only trims the second axis (slow time)
    # so trim fast time here.
    if trim is not None:
        assert trim[0] < trim[1] <= pwr_image.shape[0]
        pwr_image = pwr_image[trim[0]:trim[1], :]
        logging.info("Trimming power image fast time records to %d:%d", *trim[0:2])

    if debug and bplot:
        plt.figure()
        plt.imshow(pwr_image, aspect='auto', cmap='gray')
        plt.title('power image at Fresnel trace spacing')
        plt.colorbar()
        plt.clim([0, 20])
        plt.show()

    # Feature selection from the stacked power image. Must be at least 5 samples long
    print('-- select feature of interest')
    Nf = 0
    min_samples = 5
    while Nf < min_samples:
        #FOI = np.transpose(np.load('SRH_Y81a_lakepicks_15Stack.npy'))
        FOI = ip.picker(np.transpose(pwr_image), snap_to=FOI_selection_method)
        #np.save('SRH_Y81a_lakepicks_15Stack.npy', FOI)
        FOI = np.transpose(FOI)
        assert FOI.shape == pwr_image.shape # assert that what got loaded is the right shape
        if debug:
            plt.figure()
            plt.imshow(pwr_image, aspect='auto', cmap='gray')
            plt.title('power image with picked FOI')
            plt.imshow(FOI, aspect='auto')
            plt.colorbar()
            plt.show()
        # check the length of the picked FOI
        Nf = FOI_picklen(FOI)
        if Nf < min_samples:
            msg = 'Feature of interest selected was only {:d} samples. ' \
                  'FOI must be at at least {:d} samples. Please re-select'.format(Nf, min_samples)
            logging.error(msg)
            # continue around again

    logging.info('Picked FOI with length %d samples', Nf)

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

    assert SRF.shape == FOI.shape
    print('FEATURE OF INTEREST SELECTION -- complete')
    print(' ')

    # -----------------------------------------------------------------------------
    # Save if requested
    if savefile:
        np.savez_compressed(savefile, \
            # metadata for pick \
            line=line, chan=chan, fresnel_stack=fresnel_stack, trim=trim, \
            # actual pick data \
            FOI=FOI, SRF=SRF, Nf=Nf) # Nf is maybe not necessary
        print('Saved picks to ' + savefile)

    # -----------------------------------------------------------------------------

    return FOI, SRF, Nf

def select_ref((line, path, chan, fresnel_stack, trim,
               FOI_selection_method='maximum', bplot=True, debug=False,
               savefile=None):
    """ Select the feature for a reference pick """
    # Load the combined and focused 1m MARFA data product and stack to the
    # desired trace spacing in preparation for feature detection and selection
    pth = os.path.join(path, 'S4_FOC/')
    logging.info("Loading radar products from %s", pth)
    pwr_image, lim = fl.load_power_image(line, chan, trim, fresnel_stack, 'averaged', pth=pth)

    if trim[3] == 0:
        trim[3] = lim
    # load_power_image only trims the second axis (slow time)
    # so trim fast time here.
    if trim is not None:
        assert trim[0] < trim[1] <= pwr_image.shape[0]
        pwr_image = pwr_image[trim[0]:trim[1], :]
        logging.info("Trimming power image fast time records to %d:%d", *trim[0:2])

    if debug and bplot:
        plt.figure()
        plt.imshow(pwr_image, aspect='auto', cmap='gray')
        plt.title('power image at Fresnel trace spacing')
        plt.colorbar()
        plt.clim([0, 20])
        plt.show()

    # Feature selection from the stacked power image. Must be at least 5 samples long
    print('-- select feature of interest')
    Nf = 0
    min_samples = 5
    while Nf < min_samples:
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
        Nf = FOI_picklen(FOI)
        if Nf < min_samples:
            msg = 'Feature of interest selected was only {:d} samples. ' \
                  'FOI must be at at least {:d} samples. Please re-select'.format(Nf, min_samples)
            logging.error(msg)
            # continue around again

    logging.info('Picked FOI with length {:d} samples'.format(Nf))

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

    # -----------------------------------------------------------------------------
    # Save if requested
    if savefile:
        np.savez_compressed(savefile, \
            # metadata for pick \
            line=line, chan=chan, fresnel_stack=fresnel_stack, trim=trim, \
            # actual pick data \
            FOI=FOI, SRF=SRF, Nf=Nf) # Nf is maybe not necessary
        print('Saved picks to ' + savefile)




def FOI_picklen(foi):
    """ Count the number of features in a pick """
    nfeat = 0
    for ii in range(foi.shape[1]):
        if len(np.argwhere(foi[:, ii] == 1)) >= 1:
            nfeat += 1
    return nfeat


# Some pre-defined trim parameters for commonly-used lines
TRIMS = {
    'NAQLK/JKB2j/ZY1b': [0, 1000, 0, 12000],
    #'GOG3/JKB2j/BWN01b': [0, 1000, 0, 15000],
    'GOG3/JKB2j/BWN01a': [0, 1000, 15000, 27294],
    '__DEFAULT__': [0, 1000, 0, 0],
}
CHANNELS = {'low': '1', 'high': '2'} # channel names to numbers



def main():

    """
    pick interferometry features for use with run_interferometry.py

    #project = 'ICP10'
    #line = 'AMY/JKB2u/Y226b/'
    #project = 'ICP10'
    #line = 'ICP10/JKB2u/F01T01a/'


    """

    parser = argparse.ArgumentParser(description='Pick interferometry features')

    parser.add_argument('-p', '--project', default='GOG3',
                        help='Project name')
    parser.add_argument('--line', default='NAQLK/JKB2j/ZY1b/',
                        help='Line name: project/set/transect')

    parser.add_argument('--fresnelstack', type=int, default=15, help='fresnel_stack')
    parser.add_argument('--plot', action='store_true', help='Plot debugging graphs')
    #parser.add_argument('--save', action='store_true', help='Save intermediate files')
    parser.add_argument('-t', '--type', choices=('foi', 'ref'), default='foi', help="Output file type")
    parser.add_argument('-o', '--output', help='Output pick filename (e.g., pick1.npz', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose script output')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    line = args.line
    debug = True # TODO: make this an option.
    gain = 'low'

    path = os.path.join('/disk/kea/WAIS/targ/xtra', args.project, 'FOC/Best_Versions')

    trim = TRIMS.get(line.rstrip('/'), TRIMS['__DEFAULT__'])

    if args.type == 'foi':
        #FOI, SRF, Nf =
        select_foi_and_srf(line=line, path=path, chan=CHANNELS[gain], fresnel_stack=args.fresnelstack,
                           trim=trim, bplot=args.plot, debug=debug, savefile=args.output)

        # Show command line to use this pick including other params
        cmd = ' '.join(['run_interferometry.py', '--project', args.project,
                   '--line', args.line, '--pick', args.output])
        print("CMD:",  cmd)
    else:
        assert args.type == 'ref'
        # Do a reference pick
        raise RuntimeError("Do reference pick")

if __name__ == "__main__":
    main()
