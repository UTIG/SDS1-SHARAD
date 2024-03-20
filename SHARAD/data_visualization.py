#!/usr/bin/env python3

"""
Example command lines

Focused
./data_visualization.py --input '/disk/kea/SDS/targ/xtra/SHARAD/foc/mrosh_0001/data/edr10xxx/edr1058901/5m/3 range lines/30km/e_1058901_001_ss19_700_a_s.h5'

cmp:
./data_visualization.py --product cmp

self-test:
./data_visualization.py --selftest

"""

__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'data visualization GUI'},
    '1.1':
        {'date': 'April 15, 2020',
         'author': 'Gregory Ng, UTIG',
         'info': 'refactor separating GUI inputs from plotting'},
}

import sys
import os
import argparse
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

sys.path.insert(0, '../xlib/')
import cmp.pds3lbl as pds3

def snr(data, noise_window=250):
    '''
    convert radar data amplitudes to SNR power. RMS noise values defined in the upper portion
    of the data down to the noise_window input.

    Inputs:
    ------------
       data: radar data (expects range lines to be along the rows)

    Outputs:
    ------------
       radargram in SNR
    '''

    out = np.zeros((len(data), np.size(data, axis=1)), dtype=float)
    for jj in range(len(data)):
        noise = np.sqrt(np.mean(np.abs(data[jj, 0:noise_window])**2))
        out[jj, :] = np.divide(np.abs(data[jj, :]), noise)
    out = 20 * np.log10(out)

    return out


class MsgBoxChooseFocusType:
    """ Instantiate a class of the focus type window """
    choices = ('Delay Doppler v1', 'Matched Filter', 'Delay Doppler v2')

    def __init__(self, default=choices[0]):
        self.master = tk.Tk()
        self.master.title('Select Data Type')
        self.variable = tk.StringVar(self.master)
        self.variable.set(default)
        self.sel_foctype = tk.OptionMenu(self.master, self.variable, *MsgBoxChooseFocusType.choices)
        self.sel_foctype.pack()

        self.btn_select = tk.Button(self.master, text='Select', command=self.master.quit)
        self.btn_select.pack()

    def get_foc_type(self):
        self.master.mainloop()
        return self.variable.get()


class MsgBoxSelectParams:
    def __init__(self, names, prompt='Select Parameters'):
        self.master = tk.Tk()
        self.master.title('Select Parameters')
        self.entries = []
        tk.Label(self.master, text=prompt).grid(row=0, column=0, columnspan=2)
        for i, name in enumerate(names):
            lbl = tk.Label(self.master, text=name).grid(row=i+1, column=0)
            txt = tk.Entry(self.master)
            txt.grid(row=i+1, column=1)
            self.entries.append((lbl, txt))
        # TODO: validate parameters?
        tk.Button(self.master, text='OK', command=self.master.quit).grid(row=len(names)+1, column=1)

    def get_params(self):
        self.master.mainloop()
        return [x[1].get() for x in self.entries]


def test():
    """ Test instantiating message boxes as much as possible """
    _ = MsgBoxChooseFocusType()
    _ = MsgBoxSelectParams(names=['choice1', 'choice2', 'choice3'])
    _ = snr(np.random.normal(size=(1000,1000)))

def main():
    parser = argparse.ArgumentParser(description='Data Visualization')
    parser.add_argument('--input', help='Path to HDF5 file to visualize')
    parser.add_argument('--maxsamp', type=int, default=3600, help='maximum fast-time sample to include in radargram plots')
    parser.add_argument('--plotsnr', action='store_true', help='plot final radargrams in SNR?')
    parser.add_argument('--targ', default='/disk/kea/SDS/targ/xtra/SHARAD', help='targ data base directory')
    parser.add_argument('--product', default='foc', choices=('foc','cmp'), help='Type of data product to be visualized')
    parser.add_argument('--orbit', default='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr10xxx/edr1058901/e_1058901_001_ss19_700_a_a.dat',
                        help='Path to auxiliary file for orbit of interest')

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging output')
    parser.add_argument('--selftest', action='store_true', help='Internal script self tset')
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel)

    inputroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR'
    pathfile = os.path.relpath(args.orbit, inputroot)
    datapath = os.path.join(args.targ, args.product, pathfile)
    datafile = os.path.basename(datapath).replace('_a.dat', '_s.h5')
    datapath = os.path.dirname(datapath)
    datapath1 = datapath
    auxpath = args.orbit
    scipath = args.orbit.replace('a.dat', 's.dat')
    scifmtpath = os.path.join(inputroot, 'mrosh_0004/label/science_ancillary.fmt')
    auxfmtpath = os.path.join(inputroot, 'mrosh_0004/label/auxiliary.fmt')

    plotfunc = snr if args.plotsnr else np.abs

    if args.selftest:
        return test()

    if args.product == 'foc':
        if args.input is None:
            proc_info, datapath = get_foc_datapath(datapath, datafile)
        else:
            proc_info, datapath = os.path.basename(args.input), args.input

        # load the relevant radargram
        data = plotfunc(np.array(pd.read_hdf(datapath, 'sar')))

        # plot the radargram
        plt.figure()
        plt.imshow(np.transpose(data[:, 0:args.maxsamp]), aspect='auto', cmap='gray')
        plt.colorbar()
        plt.ylabel('Fast-Time Sample #')
        plt.xlabel('Slow-Time Sample #')
        plt.title(proc_info)
        plt.clim([0, 30])
        plt.show()

    elif args.product == 'cmp':

        # specify the datapath
        datapath = os.path.join(datapath, 'ion', datafile)
        tecupath = datapath.replace('.h5', '_TECU.txt')

        # load relevant metadata
        scireader = pds3.SHARADDataReader(scifmtpath, scipath)
        scimd = scireader.arr
        scibits = scireader.get_bitcolumns()

        auxmd = pds3.read_science(auxpath, auxfmtpath)
        rxwot = scimd['RECEIVE_WINDOW_OPENING_TIME']
        pricode = scibits['PULSE_REPETITION_INTERVAL']
        scrad = scimd['RADIUS_INTERPOLATE']

        # determine range to start of each receive window
        dt = 0.0375E-6
        t = np.zeros(len(pricode), dtype=float)
        pri_steps = [float('nan'), 1428E-6, 1429E-6, 1290E-6, 2856E-6, 2984E-6, 2580E-6]
        for ii in range(len(pricode)):
            pri_step = pri_steps[pricode[ii]]
            t[ii] = rxwot[ii] * dt + pri_step - 11.98E-6
        t = t - (2 * (scrad - min(scrad)) * 1000 / 299792458)
        t = t - min(t)

        # load the relevant complex-valued radargram
        real = np.array(pd.read_hdf(datapath, 'real'))
        imag = np.array(pd.read_hdf(datapath, 'imag'))
        data = real + 1j * imag
        del real, imag

        # load the relevant TECU data
        ## tecu estimate in column 0
        ## 1-sigma of tecu estimate in column 1
        tecu = np.genfromtxt(tecupath)

        # align radar data according to receive window opening times
        n = np.size(data, axis=1)
        fs = np.fft.fftfreq(n, dt)
        adata = np.zeros((len(data), n), dtype=complex)
        for ii in range(len(data)):
            tempA = np.multiply(np.fft.fft(data[ii], norm='ortho'),
                                np.exp(-1j * 2 * np.pi * t[ii] * fs))
            adata[ii] = np.fft.ifft(tempA, norm='ortho')
        del tempA

        # plot the results
        plt.figure()
        grid = plt.GridSpec(4, 1)
        plt.subplot(grid[0, 0])
        plt.plot(tecu[:, 0], 'b')
        plt.title(datafile + ': TECU Estimate')
        plt.ylabel('TECU [1e16/m^3]')
        plt.xlim([0, len(tecu)])
        plt.xticks([])
        plt.subplot(grid[1:4, 0])
        plt.imshow(np.transpose(plotfunc(data[:, 0:args.maxsamp])), aspect='auto', cmap='gray')
        plt.title(datafile + ': Range Compressed and Ionosphere-Corrected Radargram')
        plt.ylabel('Fast-Time Sample #')
        plt.xlabel('Slow-Time Sample #')
        plt.show()



def get_foc_datapath(datapath, datafile):
    # define specific SAR focuser type
    foc_type = MsgBoxChooseFocusType().get_foc_type()

     # define path to specific processing output of interest
    if foc_type == 'Delay Doppler v1':
        prompt = "Select Delay Doppler V1 parameters"
        params = ['Posting interval [m]', 'Aperture length [s]', 'Doppler bandwidth [Hz]']

        ddv1_dist, ddv1_apt, ddv1_bw = MsgBoxSelectParams(names=params, prompt=prompt).get_params()

        proc_info = datafile + ': ' + str(ddv1_dist) + 'm ' + str(ddv1_apt) + 's ' + str(ddv1_bw) + 'Hz'
        datapath = os.path.join(datapath, str(ddv1_dist) + 'm', str(ddv1_apt) + 's', str(ddv1_bw) + 'Hz', datafile)
    elif foc_type == 'Matched Filter':
        prompt = 'Select Matched Filter parameters'
        params = ['posting interval [m]', 'aperture length [s]', 'Doppler bandwidth [Hz]', 'permittivity']
        mf_dist, mf_apt, mf_bw, mf_er = MsgBoxSelectParams(names=params, prompt=prompt).msgbox_params.get_params()

        proc_info = datafile + ': ' + str(mf_dist) + 'm ' + str(mf_apt) + 's ' + str(mf_bw) + 'Hz ' + str(mf_er) + 'Er'
        datapath = os.path.join(datapath, str(mf_dist) + 'm', str(mf_apt) + 's', str(mf_bw) + 'Hz', str(mf_er) + 'Er', datafile)
    elif foc_type == 'Delay Doppler v2':
        prompt = 'Select Delay Doppler v2 parameters'
        params = ['interpolation distance [m]', 'posting interval [range lines]', 'aperture length [km]']
        ddv2_int, ddv2_pst, ddv2_apt = MsgBoxSelectParams(names=params, prompt=prompt).get_params()

        proc_info = datafile + ': ' + str(ddv2_int) + 'm ' + str(ddv2_pst) + ' range lines ' + str(ddv2_apt) + 'km'
        datapath = os.path.join(datapath, str(ddv2_int) + 'm', str(ddv2_pst) + ' range lines', str(ddv2_apt) + 'km', datafile)
    else: # pragma: no-cover
        assert False

    return proc_info, datapath

if __name__ == '__main__':
    sys.exit(main())
