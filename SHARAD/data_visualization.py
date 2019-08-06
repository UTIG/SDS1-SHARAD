#!/usr/bin/env python3

__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'data visualization function'}}

import os
import argparse
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


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


def main():

    parser = argparse.ArgumentParser(description='Data Visualization')
    parser.add_argument('--maxsamp', type=int, default=3600, help='maximum fast-time sample to include in radargram plots')
    parser.add_argument('-snr', '--snr', default='Yes', help='plot final radargrams in SNR?')
    parser.add_argument('-o','--output', default='/disk/kea/SDS/targ/xtra/SHARAD', help='Output base directory')
    parser.add_argument('--product', default='foc', help='Type of data product to be visualized')
    parser.add_argument('--orbit', default='/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr17xxx/edr1748102/e_1748102_001_ss19_700_a_a.dat',
                        help='Path to auxiliary file for orbit of interest')
    
    args = parser.parse_args()

    if args.snr == 'Yes':
        SNR = True
    else:
        SNR = False
    inputroot = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR'
    pathfile = os.path.relpath(args.orbit, inputroot)
    datapath = os.path.join(args.output, args.product, pathfile)
    datafile = os.path.basename(datapath).replace('_a.dat', '_s.h5')
    datapath = os.path.dirname(datapath)

    if args.product == 'foc':
        
        # define specific SAR focuser type
        master = Tk()  
        choices = ['Delay Doppler v1', 'Matched Filter', 'Delay Doppler v2']
        variable = StringVar(master)
        variable.set(choices[0])
        w = OptionMenu(master, variable, *choices)
        w.pack()
        def foc_choice():
            global foc_type
            foc_type = variable.get()
            master.quit()
        button = Button(master, text='Ok', command=foc_choice)
        button.pack()
        mainloop()
       
        # define path to specific processing output of interest
        if foc_type == 'Delay Doppler v1':
            def DDV1():
                global ddv1_dist, ddv1_apt, ddv1_bw
                ddv1_dist = e1.get()
                ddv1_apt = e2.get() 
                ddv1_bw = e3.get()
            master = Tk()
            Label(master, text='Delay Doppler v1 posting interval [m]').grid(row=0) 
            Label(master, text='Delay Doppler v1 aperture length [s]').grid(row=1)
            Label(master, text='Delay Doppler v1 doppler bandwidth [Hz]').grid(row=2)
            e1 = Entry(master); e1.grid(row=0, column=1)
            e2 = Entry(master); e2.grid(row=1, column=1)
            e3 = Entry(master); e3.grid(row=2, column=1)
            Button(master, text='Enter', command=DDV1).grid(row=3, column=0)
            Button(master, text='Done', command=master.quit).grid(row=3, column=1)
            mainloop()
            proc_info = datafile + ': ' + str(ddv1_dist) + 'm ' + str(ddv1_apt) + 's ' + str(ddv1_bw) + 'Hz'
            datapath = os.path.join(datapath, str(ddv1_dist) + 'm', str(ddv1_apt) + 's', str(ddv1_bw) + 'Hz', datafile)
        elif foc_type == 'Matched Filter':
            def MF():
                global mf_dist, mf_apt, mf_bw, mf_er
                mf_dist = e1.get()
                mf_apt = e2.get() 
                mf_bw = e3.get()
                mf_er = e4.get()
            master = Tk()
            Label(master, text='Matched Filter posting interval [m]').grid(row=0) 
            Label(master, text='Matched Filter aperture length [s]').grid(row=1)
            Label(master, text='Matched Filter doppler bandwidth [Hz]').grid(row=2)
            Label(master, text='Matched Filter permittivity').grid(row=3)
            e1 = Entry(master); e1.grid(row=0, column=1)
            e2 = Entry(master); e2.grid(row=1, column=1)
            e3 = Entry(master); e3.grid(row=2, column=1)
            e4 = Entry(master); e4.grid(row=3, column=1)
            Button(master, text='Enter', command=MF).grid(row=4, column=0)
            Button(master, text='Done', command=master.quit).grid(row=4, column=1)
            mainloop()
            proc_info = datafile + ': ' + str(mf_dist) + 'm ' + str(mf_apt) + 's ' + str(mf_bw) + 'Hz ' + str(mf_er) + 'Er'
            datapath = os.path.join(datapath, str(mf_dist) + 'm', str(mf_apt) + 's', str(mf_bw) + 'Hz', str(mf_er) + 'Er', datafile)
        elif foc_type == 'Delay Doppler v2':
            def DDV2():
                global ddv2_int, ddv2_pst, ddv2_apt
                ddv2_int = e1.get()
                ddv2_pst = e2.get() 
                ddv2_apt = e3.get()
            master = Tk()
            Label(master, text='Delay Doppler v2 interpoaltion distance [m]').grid(row=0) 
            Label(master, text='Delay Doppler v2 posting interval [range lines]').grid(row=1)
            Label(master, text='Delay Doppler v2 aperture length [km]').grid(row=2)
            e1 = Entry(master); e1.grid(row=0, column=1)
            e2 = Entry(master); e2.grid(row=1, column=1)
            e3 = Entry(master); e3.grid(row=2, column=1)
            Button(master, text='Enter', command=DDV2).grid(row=3, column=0)
            Button(master, text='Done', command=master.quit).grid(row=3, column=1)
            mainloop()
            proc_info = datafile + ': ' + str(ddv2_int) + 'm ' + str(ddv2_pst) + ' range lines ' + str(ddv2_apt) + 'km'
            datapath = os.path.join(datapath, str(ddv2_int) + 'm', str(ddv2_pst) + ' range lines', str(ddv2_apt) + 'km', datafile)
       
        # load the relevant radargram
        if SNR:
            data = snr(np.array(pd.read_hdf(datapath, 'sar')))
        else:
            data = np.abs(np.array(pd.read_hdf(datapath, 'sar')))

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
        
        # load the relevant complex-valued radargram
        real = np.array(pd.read_hdf(datapath, 'real'))
        imag = np.array(pd.read_hdf(datapath, 'imag'))
        data = real + 1j * imag
        if SNR:
            data = snr(data)
        else:
            data = np.abs(data)

        # load the relevant TECU data
        ## tecu estimate in column 0
        ## 1-sigma of tecu estimate in column 1
        tecu = np.genfromtxt(tecupath)

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
        plt.imshow(np.transpose(data[:, 0:args.maxsamp]), aspect='auto', cmap='gray')
        plt.title(datafile + ': Range Compressed and Ionosphere-Corrected Radargram')
        plt.ylabel('Fast-Time Sample #')
        plt.xlabel('Slow-Time Sample #')
        plt.show()

if __name__ == '__main__':
    main()
