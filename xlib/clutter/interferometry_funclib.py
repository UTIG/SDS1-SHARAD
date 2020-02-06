#!/usr/bin/env python3
__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu; Gregory Ng, ngg@ig.utexas.edu']
__version__ = '0.2'
__history__ = {
    '0.1':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'library of functions required for interferometry'},
    '0.2':
        {'date': 'February 01, 20202',
         'author': 'Gregory Ng, UTIG',
         'info': 'Optimize algorithms, esp coregistration'},
}

import sys
import math
import pytest
from tkinter import *
import logging
import os

import pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from scipy import integrate
from scipy import signal
from scipy.signal import detrend, correlate
import scipy.special

import interface_picker as ip
import unfoc_KMS2 as unfoc
from parse_channels import parse_channels

def load_marfa(line, channel, pth='./Test Data/MARFA/', nsamp=3200):
    '''
    function for loading MARFA data (magnitude and phase) for a specific line

    Inputs:
    ------------
          line: string specifying the line of interest
       channel: string specifying MARFA channel
           pth: path to line of interest
         nsamp: number of fast-time samples

    Outputs:
    ------------
        outputs are arrays of the magnitude and phase
    '''

    # set path to the magnitude and phase files
    mag_pth = os.path.join(pth, line, 'M1D' + channel)
    phs_pth = os.path.join(pth, line, 'P1D' + channel)

    # load and organize the magnitude file
    logging.debug("Loading " + mag_pth)
    mag = np.fromfile(mag_pth, dtype='>i4')
    ncol = int(len(mag) / nsamp)
    mag = np.transpose(np.reshape(mag, (ncol, nsamp))) / 10000

    # load and organize the phase file
    logging.debug("Loading " + phs_pth)
    phs = np.fromfile(phs_pth, dtype='>i4')
    ncol = int(len(phs) / nsamp)
    phs = np.transpose(np.reshape(phs, (ncol, nsamp))) / 2**16

    return mag, phs

def test_load_marfa():
    line = 'DEV2/JKB2t/Y81a'
    path = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC'
    mag, phs = load_marfa(line, '1', pth=path)

    assert mag.shape == phs.shape


def load_S2_bxds(pth, channel, nsamp=3200):
    '''
    function for loading data from S2_FIL. these data are interpolated to
    a 1m trace spacing with some azimuthal filtering but are not range
    compressed or SAR focused.

    Inputs:
    ------------
           pth: string specifying the path to the line of interest
       channel: string specifying the MARFA channel
         nsamp: number of fast-time samples

    Outputs:
    ------------
        output is an array of bxds integers
    '''

    fn = os.path.join(pth, 'bxds' + channel + '.i')
    arr = np.fromfile(fn, dtype='<i2')
    ncol = int(len(arr) / nsamp)
    bxds = np.transpose(np.reshape(arr, (ncol, nsamp)))

    return bxds

def find_S2_bxds(basepath, maxcount=None):
    # basepath of the  form 
    # /disk/kea/WAIS/targ/xtra
    # /disk/kea/WAIS/targ/xtra/GOG2/FOC/Best_Versions/S2_FIL/GDS/JKB2f/X2a/bxds2.i
    globpat = os.path.join(basepath, '*/FOC/Best_Versions/S2_FIL/*/*/*/bxds?.i')
    outfiles = glob.glob(globpat)
    outfiles.sort()
    return outfiles

def test_load_S2_bxds():
    # TODO: randomize for random file testing
    basepath = '/disk/kea/WAIS/targ/xtra'
    files = list(find_S2_bxds(basepath))
    logging.debug("Found {:d} files".format(len(files)))
    for i, filepath in enumerate(files):
        logging.debug("[{:2d}] Loading {:s}".format(i+1, filepath))
        dirpath = os.path.dirname(filepath)
        channel = filepath[-3]
        bxds = load_S2_bxds(dirpath, channel)
        assert len(bxds)
        if i >= 9: # only process the first 10 files
            break
        




def load_pik1(line, channel, pth='./Test Data/MARFA/', nsamp=3200, IQ='mag'):
    '''
    function for loading MARFA data (magnitude and phase) for a specific line

    Inputs:
    ------------
          line: string specifying the line of interest
       channel: string specifying MARFA channel
           pth: path to line of interest
         nsamp: number of fast-time samples per trace
            IQ: 'mag' to return magnitude, 'phs' to return phase data

    Outputs:
    ------------
        outputs are arrays of the magnitude and phase
    '''

    assert pth.endswith('/') # must have trailing slash.
    assert line.endswith('/') # expecting a trailing slash

    # set path to the magnitude and phase files
    if IQ == 'mag':
        pth = pth + line + 'MagLoResInco' + channel
    elif IQ == 'phs':
        pth = pth + line + 'PhsLoResInco' + channel
    else:
        raise ValueError('Invalid option specified for parameter IQ')

    # load and organize the data
    data = np.fromfile(pth, dtype='>i4')
    ncol = int(len(data) / nsamp)
    if IQ == 'mag':
        data = np.transpose(np.reshape(data, (ncol, nsamp))) / 10000
    elif IQ == 'phs':
        data = np.transpose(np.reshape(data, (ncol, nsamp))) / 2**24
    else:
        raise ValueError("Invalid value for IQ: '{:s}'".format(IQ))

    return data


def test_load_pik1():
    # TODO: test some RADnh5
    # /disk/kea/WAIS/targ/xped/ICP6/quality/xlob/pyk1.RADnh3/ICP6/JKB2k/F01T04a/MagLoResInco1
    # /disk/kea/WAIS/targ/xped/ICP6/quality/xlob/pyk1.RADnh3/ICP6/JKB2l/F02T01a/MagLoResInco1

    pth = os.path.join(os.getenv('WAIS', '/disk/kea/WAIS'), 'targ/xped/ICP6/quality/xlob/pyk1.RADnh3') + '/'
    line = 'DVG/MKB2l/Y06a/'

    logging.debug("pth=" + pth)
    logging.debug("line=" + line)

    list_channels = ['1', '2', '5', '6', '7', '8']

    for channel in list_channels:
        for IQ in ('mag','phs'):
            load_pik1(line, channel, pth=pth, IQ='mag')

        with pytest.raises(ValueError):
            load_pik1(line, channel, pth=pth, IQ='invalidtype')


def load_power_image(line, channel, trim, fresnel, mode, pth='./Test Data/MARFA/'):
    '''
    function for loading combined and focused MARFA data products and
    then stacking that to a desired trace spacing interval

    Inputs:
    ------------
          line: string specifying the line of interest
       channel: string specifying the MARFA channel
          trim: limits used to trim the full profile
       fresnel: Fresnel zone length in units of trace spacings
          mode: mode to use when stacking range lines to Fresenl zone interval
           pth: path the relevant MARFA files

    Outputs:
    ------------
        output: power image for the rangeline of interest
           lim: number of range lines in the untrimmed radargram
    '''

    # load the MARFA dataset
    mag, phs = load_marfa(line, channel, pth)
    # GNG Downstream functions aren't expecting this to be trimmed yet
    #mag = mag[trim[0]:trim[1], :]
    if trim is not None and trim[3] != 0:
        mag = mag[:, trim[2]:trim[3]]
    lim = np.size(mag, axis=1)

    # incoherently stack to desired trace spacing
    if fresnel != 1:
        if mode == 'summed':
            output = stack(mag, fresnel)
        elif mode == 'averaged':
            output = np.divide(stack(mag, fresnel), fresnel)
    else:
        output = mag

    return output, lim

def convert_to_complex(magnitude, phase, mag_dB=True, pwr_flag=True):
    '''
    converting MARFA datasets back to complex valued voltages

    Inputs:
    ------------
      magnitude: array of magnitudes
          phase: array of phases
         mag_dB: flag for indicating whether the input magnitudes as in dB
       pwr_flag: flag for indicating whether the input magnitudes are in power

    Outputs:
    ------------
        complex array of voltages
    '''

    # get magnitudes out of dB if they are
    if mag_dB:
        if pwr_flag:
            magnitude = 10 ** (magnitude / 20)
        else:
            magnitude = 10 ** (magnitude / 10)

    # calculate the complex values
    cmp = magnitude * np.exp(1j * phase)

    return cmp

def convert_to_magphs(cmp, mag_dB=True, pwr_flag=True):
    '''
    converting MARFA datasets back to magnitude and phase

    Inputs:
    ------------
      magnitude: array of complex values
         mag_dB: flag for indicating whether the input magnitudes as in dB
       pwr_flag: flag for indicating whether the input magnitudes are in power

    Outputs:
    ------------
        arrays of magnitude and phase [rad]
    '''

    # calculate magnitudes
    if mag_dB:
        if pwr_flag:
            magnitude = 20 * np.log10(np.abs(cmp))
        else:
            magnitude = 10 * np.log10(np.abs(cmp))

    # calculate the phases
    phase = np.angle(cmp)

    return magnitude, phase

def stack(data, fresnel, datatype='float'):
    '''
    stacking data to a new trace posting interval

    Inputs:
    ------------
          data: data to be stacked
       fresnel: trace spacing according to Fresnel zone
                -- must be given in integers of the trace spacing for the input
                   radargrams
      datatype: type of output (float, complex, int ...)

    Output:
    ------------
        stacked array
    '''

    # incoherently stack to desired trace spacing
    indices = np.arange(np.floor(fresnel / 2) + 1, np.size(data, axis=1), fresnel) - 1
    if np.size(data, axis=1) - indices[-1] < np.floor(fresnel / 2):
        col = len(indices) - 1
    else:
        col = len(indices)

    # pre-define the output
    if datatype == 'complex':
        output = np.zeros((np.size(data, axis=0), col), dtype=complex)
    elif datatype == 'float':
        output = np.zeros((np.size(data, axis=0), col), dtype=float)

    # perform stacking
    for ii in range(col):
        start_ind = int(indices[ii] - np.floor(fresnel / 2))
        end_ind = int(indices[ii] + np.floor(fresnel / 2))
        output[:, ii] = np.sum(data[:, start_ind:end_ind], axis=1)

    return output

def stacked_power_image(magA, phsA, magB, phsB, fresnel, mode):
    '''
    producing a combined power image at the desired trace posting interval
    from the two antenna datasets

    Inputs:
    ------------
         magA: array of magnitudes [typically dB] for antenna A
         phsA: array of phases [radians] for antenna A
         magB: array of magnitudes [typically dB] for antenna B
         phsB: array of phases [radians] for antenna B
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
         mode: how amplitudes in the power image are represented
               -- 'averaged': average amplitude of the stack
               -- 'summed': sum of the stack

    Output:
    ------------
        real-valued power image at the Fresnel zone trace spacing from the two
        antennae
    '''

    # make sure the new trace spacing passed is odd
    if fresnel % 2 != 1:
        fresnel = fresnel - 1

    # convert back to complex values
    cmpA = convert_to_complex(magA, phsA)
    cmpB = convert_to_complex(magB, phsB)

    # coherently combine the two antennas
    comb_cmp = np.multiply(cmpA, np.conj(cmpB))

    # produce a combined power image
    comb_mag, comb_phs = convert_to_magphs(comb_cmp)

    # incoherently stack to desired trace spacing
    if mode == 'summed':
        output = stack(comb_mag, fresnel)
    elif mode == 'averaged':
        output = np.divide(stack(comb_mag, fresnel), fresnel)

    return output

def stacked_correlation_map(cmpA, cmpB, fresnel, n=2, az_step=1):
    '''
    producing a correlation map at the desired trace posting interval from the
    two antenna datasets. Taken from Rosen et al. (2000) - Equation 57

    Inputs:
    ------------
         cmpA: antenna A complex-valued radargram
         cmpB: antenna B complex-valued radargram
      frensel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
            n: fast-time smoother required during implementation
               of the Castelletti et al. (2018) approach
      az_step: azimuthal sample interval during multi-look

    Outputs:
    ------------
        real-valued map of correlation
    '''

    # make sure the new trace spacing passed is odd
    if fresnel % 2 != 1:
        frensel = fresnel - 1

    # calculate correlation map and average (multi-look) if desired
    if fresnel != 1:
        
        #top = np.divide(stack(np.multiply(cmpA, np.conj(cmpB)), fresnel, datatype='complex'), fresnel)
        #bottomA = np.divide(stack(np.square(np.abs(cmpA)), fresnel), fresnel)
        #bottomB = np.divide(stack(np.square(np.abs(cmpB)), fresnel), fresnel)
        
        # setup bounds for each multilook window (output range line in middle)
        indices = np.arange(np.floor(fresnel / 2) + 1, np.size(cmpA, axis=1), fresnel) - 1
        if np.size(cmpA, axis=1) - indices[-1] < np.floor(fresnel / 2):
            col = len(indices) - 1
        else:
            col = len(indices)
        # predefine the correlation map
        corrmap = np.zeros((np.size(cmpA, axis=0), col), dtype=float)
        # calculate correlation map
        for ii in range(col):
            num = np.floor((fresnel / 2) / az_step)
            val = np.multiply(az_step, np.arange(-num, num + 0.1))
            val = indices[ii] + val
            #start_ind = int(indices[ii] - np.floor(fresnel / 2))
            #end_ind = int(indices[ii] + np.floor(fresnel / 2))
            for jj in range(len(cmpA)):
                ywindow = np.arange(jj, jj + n, 1)
                viable = np.argwhere(ywindow <= len(cmpA) - 1)
                ywindow = ywindow[viable]
                #S1 = cmpA[ywindow.astype(int), np.arange(start_ind, end_ind, az_step)]
                #S2 = cmpB[ywindow.astype(int), np.arange(start_ind, end_ind, az_step)]
                S1 = cmpA[ywindow.astype(int), val.astype(int)]
                S2 = cmpB[ywindow.astype(int), val.astype(int)]
                top = np.mean(np.multiply(S1, np.conj(S2)))
                bottomA = np.mean(np.square(np.abs(S1)))
                bottomB = np.mean(np.square(np.abs(S2)))
                bottom = np.sqrt(np.multiply(bottomA, bottomB))
                corrmap[jj, ii] = np.abs(np.divide(top, bottom))
    else:
        top = np.multiply(cmpA, np.conj(cmpB))
        bottomA = np.square(np.abs(cmpA))
        bottomB = np.square(np.abs(cmpB))
        bottom = np.sqrt(np.multiply(bottomA, bottomB))
        corrmap = np.abs(np.divide(top, bottom))

    return corrmap

def stacked_interferogram(cmpA, cmpB, fresnel, rollphase, roll=True, n=2, az_step=1):
    '''
    producing a phase interferogram at the desired trace posting interval from
    the two antenna datasets

    Inputs:
    ------------
         cmpA: complex-valued antenna A radargram
         cmpB: complex-valued antenna B radargram
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
               -- if set to something less than 1, the interferogram is smoothed
                  and stacked
    rollphase: interferometric phase related to roll
         roll: True/False flag to apply roll correction
            n: fast-time smoother required during implementation
               of the Castelletti et al. (2018) approach
      az_step: azimuthal sample interval during multi-look

    Output:
    ------------
        real-valued interferogram at the Fresnel zone trace spacing from the
        two antennae
    '''

    # make sure the new trace spacing passed is odd
    if fresnel % 2 != 1:
        fresnel = fresnel - 1

    # calculate interferogram
    if fresnel <= 1:
        output = np.angle(np.multiply(cmpA, np.conj(cmpB)))
        if roll:
            for ii in range(np.size(output, axis=1)):
                output[:, ii] = output[:, ii] + rollphase[ii]
    else:
        # setup bounds for each multilook window (output range line in middle)
        indices = np.arange(np.floor(fresnel / 2) + 1, np.size(cmpA, axis=1), fresnel) - 1
        if np.size(cmpA, axis=1) - indices[-1] < np.floor(fresnel / 2):
            col = len(indices) - 1
        else:
            col = len(indices)
        # predefine the interferogram
        output = np.zeros((np.size(cmpA, axis=0), col), dtype=float)
        # calculate interferogram
        for ii in range(col):
            num = np.floor((fresnel / 2) / az_step)
            val = np.multiply(az_step, np.arange(-num, num + 0.1))
            val = indices[ii] + val
            #start_ind = int(indices[ii] - np.floor(fresnel / 2))
            #end_ind = int(indices[ii] + np.floor(fresnel / 2))
            for jj in range(len(cmpA)):
                ywindow = np.arange(jj, jj + n, 1)
                viable = np.argwhere(ywindow <= len(cmpA) - 1)
                ywindow = ywindow[viable]
                #S1 = cmpA[ywindow.astype(int), np.arange(start_ind, end_ind, az_step)]
                #S2 = cmpB[ywindow.astype(int), np.arange(start_ind, end_ind, az_step)]
                S1 = cmpA[ywindow.astype(int), val.astype(int)]
                S2 = cmpB[ywindow.astype(int), val.astype(int)]
                temp = np.mean(np.multiply(S1, np.conj(S2)))
                output[jj, ii] = np.angle(temp)
            if roll:
                #output[:, ii] = output[:, ii] - np.mean(rollphase[start_ind:end_ind])
                output[:, ii] = output[:, ii] + np.mean(rollphase[val.astype(int)])
        #inter = np.zeros((len(cmpA), np.size(cmpA, axis=1)), dtype=float)
        #for ii in range(np.size(cmpA, axis=1)):
        #    #if ii % 500 == 0:
        #    #    print(str(ii), 'of', str(np.size(cmpA, axis=1)))
        #    xwindow = np.arange(ii - (fresnel - 1) / 2, ii + 1 +(fresnel - 1) / 2, 1)
        #    if ii < fresnel:
        #        viable = np.argwhere(xwindow >= 0)
        #        xwindow = np.transpose(xwindow[viable])
        #    if np.size(cmpA, axis=1) - ii < fresnel:
        #        viable = np.argwhere(xwindow < np.size(cmpA, axis=1) - 1)
        #        xwindow = np.transpose(xwindow[viable])
        #    for jj in range(len(cmpA)):
        #        ywindow = np.arange(jj, jj + n, 1)
        #        viable = np.argwhere(ywindow <= len(cmpA) - 1)
        #        ywindow = ywindow[viable]
        #        temp = np.mean(np.multiply(cmpA[ywindow.astype(int), xwindow.astype(int)], np.conj(cmpB[ywindow.astype(int), xwindow.astype(int)])))
        #        inter[jj, ii] = np.angle(temp)
        #    if roll:
        #        inter[:, ii] = inter[:, ii] - np.mean(rollphase[xwindow.astype(int)])

    #if fresnel != 1:
    #    output = np.divide(stack(inter, fresnel), fresnel)
    #else:
    #    output = inter

    return output

def FOI_extraction(image, FOI):
    '''
    extract the interferometric from a 2D array at the location of a specific FOI

    Inputs:
    ------------
          image: 2D array
            FOI: array with pick FOI
                 - FOI array should have the same dimensions as image with
                   samples related to the FOI marked with ones

    Output:
    ------------
        extracted information
    '''

    # extract indices related to the FOI
    indices = np.argwhere(FOI == 1)

    # extract information related to these indices
    output = image[indices[:, 0], indices[:, 1]]

    return output

def ipdf(N, gamma, iphi, phi0):
    '''
    Calculate an empirical interferometric probability density function for a
    nadir reflector with defined parameters

    Inputs:
    ------------
         N: number of interferometric looks
     gamma: interferometric correlation
       iphi: interferometric phase angles [rad]
       phi0: center of interferometric phase pdf [rad]

    Output:
    ------------
        interferometric phase pdf
    '''

    beta = gamma * np.cos(iphi - phi0)
    ghf = scipy.special.hyp2f1(1, N, 1/2, beta**2)
    G1 = scipy.special.gamma(N + 1/2)
    G2 = scipy.special.gamma(1/2)
    G3 = scipy.special.gamma(N)
    f = (((1 - gamma**2)**N) / (2 * np.pi)) * ghf
    f = f + ((G1 * ((1 - gamma**2)**N) * beta) / \
       (2 * G2 * G3 * (1 - beta**2)**(N + 1/2)))

    return f

def empirical_pdf(fc, B, fresnel, gamma, phi_m=0):
    '''
    Calculate an empirical interferometric probability density function for the
    picked feature of interest as if it were at nadir

    Inputs:
    ------------
           fc: radar center frequency [Hz]
            B: interferometric baseline [m]
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
        gamma: interferometric correlation
        phi_m: mean interferometric phase angle of the FOI

    Output:
    ------------
        phi: interferometric phase angles
          f: interferometric phase pdf
    '''

    # calculate the nadir emprirical interferometric phase pdf
    phi = np.linspace(-np.pi, np.pi, 10000)
    #phi = (2 * np.pi * B * np.sin(phi)) / (299792458 / fc)
    f = ipdf(fresnel, gamma, phi, np.deg2rad(phi_m))

    return phi, f

def empirical_sample_mean(N, Nf, iphi, gamma, phi_m=0):
    '''
    Calculate the variance of the interferometric phase pdf as well as the
    sample error of the mean

    Inputs:
    ------------
          N: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
         Nf: number of multi-looked pixels used to define the FOI
       iphi: empirical interferometric phase angles
      gamma: interferometric correlation
      phi_m: mean interferometric phase of empirical pdf

    Output:
    ------------
      sigma_phi: variance of the interferometric phase pdf
        sigma_m: sample error of the mean
    '''

    # calculate the standard deviation of the emprirical phase distribution
    func = lambda x: np.multiply(x**2, ipdf(N, gamma, x, np.deg2rad(phi_m)))
    sigma_phi = np.rad2deg(np.sqrt(integrate.quad(func, -np.pi, np.pi)[0]))

    # calculate the empirical sample mean based on procedures presented in
    # Haynes et al. (2018)
    if sigma_phi < 30:
        # characterize sample mean for Gaussian distribution
        sigma_m = np.divide(sigma_phi, np.sqrt(Nf))
    else:
        # perform Monte Carlo simulations to characterize the sample mean if
        # SNR or number of looks is too small
        simulations = int(5E5)
        M = np.zeros((simulations, 1), dtype=float)
        #phi = np.linspace(-np.pi, np.pi, 10000)
        f = ipdf(N, gamma, iphi, np.deg2rad(phi_m))
        for ii in range(simulations):
            # draw Nf samples from the emprirical interferometric phase pdf
            phin = np.random.choice(iphi, Nf, p=np.divide(f, sum(f)))
            # calculate the sample mean of the selected Nf samples
            #M[ii] = np.angle(np.mean(np.exp(np.multiply(1j, phin))), deg=True)
            M[ii] = np.rad2deg(np.mean(phin))
        sigma_m = np.sqrt(np.var(M))

    return sigma_m



def sinc_interpolate(data, orig_sample_interval, subsample_factor):
    '''
    function for interpolating a vector using a sinc interpolation kernel

    Inputs:
    ------------
                      data: input data vector
      orig_sample_interval: sampling interval of the input data
          subsample_factor: factor used to subsample the input data at

    Outputs:
    ------------
        output is the interpolated data vector
    '''

    # GN -- this is a lot slower than frequency interpolation because of generating the
    # gigantic sinc matrix

    # define sample vectors
    new_sample_interval = orig_sample_interval / subsample_factor
    orig_t = np.linspace(0, (len(data) - 1) * orig_sample_interval, len(data))
    nsamples_new = len(data) * subsample_factor
    new_t = np.linspace(0, (nsamples_new - 1) * new_sample_interval, nsamples_new)

    # perform the interpolation
    # Generate shifted x values to input to sinc
    sincM = np.tile(new_t, (len(orig_t), 1)) - np.tile(orig_t[:, np.newaxis], (1, len(new_t)))
    output = np.dot(data, np.sinc(sincM / orig_sample_interval))

    return output


def frequency_interpolate(data, subsample_factor):
    '''
    function for interpolating a vector by padding the data in the frequency
    domain.

    Inputs:
    ------------
                  data: complex-valued range line
      subsample_factor: factor by which the user wants to subsample the data by

    Outputs:
    ------------
       interpolated data vector
    '''
    fft = np.fft.fft(data, norm='ortho')
    fft_shift = np.fft.fftshift(fft)
    x = int((len(data) * subsample_factor - len(data)) / 2)
    fft_int = np.pad(fft_shift, (x, x), 'constant', constant_values=(0, 0))
    fft_int_shift = np.fft.fftshift(fft_int)
    # Without the np.sqrt(subsample_factor), this is an energy-preserving function.
    # But we want to preserve value, not total signal energy
    return np.sqrt(subsample_factor)*np.fft.ifft(fft_int_shift, norm='ortho')

def test_interpolate():
    """ Test equivalence of interpolation algorithms, and run with
    a variety of input sizes """
    bplot = True
    
    for repeatsize in (128, 255, 77):
        logging.debug("repeatsize = {:d}".format(repeatsize))
        # Series of step functions
        sig = np.repeat([0.5, 0., 1., 1., 0., 1., 0., 0., 1., 0.5], repeatsize)
        sig -= sig.mean()
        x = np.linspace(0, 100.0, len(sig))
        # noisy signal
        sig_noise = sig + 1e-3 * np.random.randn(len(sig))
        osi = 1 / 50e6 # original signal interval
        for ifactor in (2,5, 10, 13, 21):
            x2 = np.linspace(0, max(x), len(sig_noise)*ifactor) # get interpolated indices
            sig_interp0 = np.interp(x2, x, sig_noise) # linear interpolation
            sig_interp1 = np.real(sinc_interpolate(sig_noise, osi, ifactor))
            sig_interp2 = np.real(frequency_interpolate(sig_noise, ifactor))
    
            rms1 = np.sqrt(np.square(abs(sig_interp0 - sig_interp1)).mean())
            rms2 = np.sqrt(np.square(abs(sig_interp0 - sig_interp2)).mean())
            rms3 = np.sqrt(np.square(abs(sig_interp1 - sig_interp2)).mean())
            logging.info("interpolate: ifactor={:0.1f} RMS(lin-sinc)={:0.4g}"
                         " RMS(lin-freq)={:0.4g} RMS(sinc-freq)={:0.4g}".format(ifactor, rms1, rms2, rms3))
            try:
                assert rms3 < 5e-4
            except AssertionError:
                bplot = True
                
            if bplot: # pragma: no-cover
                plt.subplot(211)
                plt.plot(x2, sig_interp0, x2, sig_interp1, x2, sig_interp2)
                plt.legend(['linear', 'sinc', 'frequency'])
                plt.subplot(212)
                plt.plot(abs(sig_interp1-sig_interp2))
                plt.title("Sinc vs Frequency Interpolation error")
                plt.show()



def coregistration(cmpA, cmpB, orig_sample_interval, subsample_factor, shift=30):
    '''
    function for sub-sampling and coregistering complex-valued range lines
        tempB = np.mean(np.square(np.abs(subsampA))) # GNG: move this out?
    from two radargrams as required to perform interferometry. Follows the
    steps outlines in Castelletti et al. (2018)

    Inputs:
    -------------
                      cmpA: complex-valued input A radargram
                      cmpB: complex-valued input B radargram
      orig_sample_interval: sampling interval of the input data
          subsample_factor: factor used modify the original fast-time sampling
                            interval

    Outputs:
    -------------
       coregA: coregistered complex-valued A radargram
       coregB: coregistered complex-valued B radargram
    '''

    # define the output
    coregB = np.empty_like(cmpB, dtype=complex)

    shift_array = np.zeros(cmpA.shape[1])

    for ii in range(cmpA.shape[1]): #range(np.size(cmpA, axis=1)):
        # subsample
        # Older sinc interpolation method, determined equivalent nto frequency_interpolate
        #subsampA = sinc_interpolate(cmpA[:, ii], orig_sample_interval, subsample_factor)
        #subsampB = sinc_interpolate(cmpB[:, ii], orig_sample_interval, subsample_factor)
        subsampA = frequency_interpolate(cmpA[:, ii], subsample_factor)
        subsampB = frequency_interpolate(cmpB[:, ii], subsample_factor)
        #logging.debug("subsampA.shape = " + str(subsampA.shape))


        #if ii == 0:
        #    plt.figure()
        #    plt.plot(np.linspace(0, 1, len(cmpA)), np.abs(cmpA[:, ii]), label='original A')
        #    plt.plot(np.linspace(0, 1, len(subsampA)), np.abs(subsampA), label='interpolated A')
        #    plt.plot(np.linspace(0, 1, len(cmpB)), np.abs(cmpB[:, ii]), label='original B')
        #    plt.plot(np.linspace(0, 1, len(subsampB)), np.abs(subsampB), label='interpolated B')
        #    plt.legend()
        #    plt.show()

        # co-register and shift
        shifts = np.arange(-1 * shift, shift, 1)
        rho = np.zeros((len(shifts), ), dtype=float)
        b_do_scipy = True
        if b_do_scipy:
            # calculate using a fourier cross-correlation (scipy.signal.correlate)
            rho = np.abs(correlate(subsampA, subsampB[shift:-(shift-1)], mode='valid'))
        else:
            tempB = np.mean(np.square(np.abs(subsampA))) 
            for jj in range(len(shifts)):
                tempA = np.abs(np.mean(np.multiply(subsampA, np.conj(np.roll(subsampB, shifts[jj])))))
                tempC = np.mean(np.square(np.abs(np.roll(subsampB, shifts[jj]))))
                #tempD = np.sqrt(np.multiply(tempB, tempC)) # GNG: sqrt is un-necessary for this relative compare
                tempD = np.multiply(tempB, tempC)
                rho[jj] = np.divide(tempA, tempD)
        to_shift = shifts[np.argwhere(rho == np.max(rho))][0][0]
        subsampB = np.roll(subsampB, to_shift)
        shift_array[ii] = to_shift
        # remove subsampling
        coregB[:, ii] = subsampB[np.arange(0, len(subsampB), subsample_factor)]

    return cmpA, coregB, shift_array


def test_coregistration():
    pass

def read_ztim(filename, field_names=None):
    '''
    Function to read a binary ztim file.

    Inputs:
    -------------
          filename: path to the ztim file
       field_names: names of the data columns in the ztim file

    Output:
    ------------
        data frame of the ztim 
    '''

    ztim_format = 'S1, i2, i2, i4'
    ztim_names = ['PLUS', 'YEAR', 'DAY', 'itim']
    field_format = ', f8'

    with open(filename, 'rb') as zfile:
        zformat = zfile.readline()
        num_fields = len(zformat.lstrip(b'zfil1z').rstrip())
        zformat = ztim_format + num_fields * field_format

        if field_names is None:
            field_names = ['d' + str(elem) for elem in range(num_fields)]

        data = np.core.records.fromfile(zfile, formats=zformat, names=ztim_names + field_names,
                            aligned=True, byteorder='>')

    return pd.DataFrame(data) 

def load_roll(treg_path, s1_path):
    '''
    Function to extract and interpolate aircraft roll information such that it can
    be compared to the MARFA 1m focused data product.

    Inputs:
    -------------
       treg_path: path to the 'treg' folder containing the relavant ztim binary
                  file from which the roll data will be loaded
         s1_path: path to the 'S1_POS' folder containing the relevant 'ztim_xyhd'
                  ascii file for the specific line being investigated.

    Output:
    ------------
       interpolated roll vector at 1m trace intervals
    '''

    ## load the roll data
    #norm_roll = np.genfromtxt(norm_path + 'roll_ang')
    #
    ## extract the ztim vector from norm
    #temp = pd.read_csv(norm_path + 'syn_ztim', header=None)
    #norm_ztim = np.zeros((len(temp), ), dtype=float)
    #for ii in range(len(norm_ztim)):
    #    norm_ztim[ii] = float(temp[2][ii].replace(')', ''))
    #del temp

    # load base roll and timing data
    fn = treg_path + 'ztim_llzrphsaaa.bin'
    fieldnames = ['lat', 'long', 'z', 'roll', 'pitch', 'heading', 'sigma', 'EW_acc', 'NS_acc', 'z_acc']
    test = read_ztim(fn, fieldnames)
    norm_ztim = test['itim']
    norm_roll = test['roll']

    # load the timing and alongtrack position associated with each range line
    # from the S1_POS folder
    temp = pd.read_csv(s1_path + 'ztim_xyhd', header=None, delimiter=' ')
    S1_ztim = np.zeros((len(temp), ), dtype=float)
    S1_dist = np.zeros((len(temp), ), dtype=float)
    for ii in range(len(S1_ztim)):
        S1_ztim[ii] = float(temp[2][ii].replace(')', ''))
        S1_dist[ii] = float(temp[6][ii])
    del temp

    # interpolate norm_roll to S1_ztim
    S1_roll = np.interp(S1_ztim, norm_ztim, norm_roll)

    # interpolate S1_roll to a 1m product
    one_m_dist = np.arange(np.floor(min(S1_dist)), np.ceil(max(S1_dist)), 1)
    one_m_roll = np.interp(one_m_dist, S1_dist, S1_roll)

    return one_m_dist, one_m_roll

def roll_correction(l, B, trim, norm_path, s1_path, roll_shift=0):
    '''
    Function to correct phase of each radargram for aircraft roll.

    Inputs:
    -------------
               l: radar wavelength [m]
               B: interferometric baseline [m]
            trim: bounds on the portion of the radargram under investigation
       norm_path: path to the 'norm' folder containing the relevant 'roll_ang'
                  and 'syn_ztim' ascii files for the specific line being
                  investigated
         s1_path: path to the 'S1_POS' folder containing the relevant 'ztim_xyhd'
                  ascii file for the specific line being investigated
      roll_shift: DC shift we want to apply to the roll channel that changes
                  the zero baseline

    Outputs:
    -------------
        roll_phase: phase relating to roll [rad]
          roll_ang: recorded roll angle [rad]
    '''

    # load the roll data
    roll_dist, roll_ang = load_roll(norm_path, s1_path)
    roll_dist = roll_dist[trim[2]:trim[3]]
    roll_ang = roll_ang[trim[2]:trim[3]] + roll_shift
    roll_ang = np.deg2rad(roll_ang)

    # convert roll angle to a phase shift as if the roll angle represents
    # a change in the radar look angle
    roll_phase = np.multiply(np.divide(2 * np.pi * B, l), np.sin(-1 * roll_ang))

    return roll_phase, roll_ang

def cinterp(sweep_fft, index):
    '''
    Function called during the denoise and dechirp of HiCARS/MARFA airborne data
    interpolation in the complex domain to remove some coherent noise

    Inputs:
    -------------
       sweep_fft: fft of the sweep
           index: bin affected by the L0 noise

    Outputs:
    ------------
       output is corrected fft of the sweep
    '''

    r = (np.abs(sweep_fft[index - 1]) + np.abs(sweep_fft[index + 1])) / 2
    t1 = np.angle(sweep_fft[index - 1])
    t2 = np.angle(sweep_fft[index + 1])
    if (np.abs(t1 - t2) > np.pi):
        t1 = t1 + 2 * np.pi
    theta = (t1 + t2) / 2
    sweep_fft[index] = r * (np.cos(theta) + 1j * np.sin(theta))

    return sweep_fft

def get_ref_chirp(path, bandpass=True, nsamp=3200):
    '''
    Load the HiCARS/MARFA reference chirp

    Inputs:
    ------------
                 I_path: path to array of the integer component of the chirp
                 Q_path: path to array of the quadrature component of the chirp
               bandpass: bandpass sampling, False for legacy HiCARS.
                         disables cinterp and flips the chirp
     trunc_sweep_length: number of samples

    Outputs:
    -------------
       frequency-domain representation of the HiCARS reference chirp
    '''

    I = np.fromfile(os.path.join(path, 'I.bin'), '>i4')
    Q = np.fromfile(os.path.join(path, 'Q.bin'), '>i4')

    if len(I) < nsamp:
        I = I[0:nsamp]
        Q = Q[0:nsamp]

    if not bandpass:
        rchirp = np.flipud(I + np.multiply(1j, Q))
    else:
        rchirp = I + np.multiply(1j, Q)

    return np.fft.fft(rchirp, n=nsamp)

def hamming(trunc_sweep_length):
    '''
    Compute a hamming window

    Inputs:
    --------------
      trunc_sweep_length: number of samples

    Outputs:
    --------------
       hamming window
    '''

    filt = np.zeros((trunc_sweep_length, ))
    a = np.round(2.5 * trunc_sweep_length / 50)
    b = np.round(17.5 * trunc_sweep_length / 50)
    diff = b - a
    hamming = np.sin(np.arange(0, 1, 1/diff) * np.pi)
    filt[int(a):int(b)] = np.transpose(hamming)
    filt[int(trunc_sweep_length - b + 2):int(trunc_sweep_length - a + 2)] = hamming

    return filt

def denoise_and_dechirp(gain, sigwin, raw_path, geo_path, chirp_path,
                        output_samples=3200, do_cinterp=True, bp=True):
    '''
    Denoise and dechirp HiCARS/MARFA data

    Inputs:
    ------------
               gain: sets which MARFA interferometric datasets are to be analyzed
             sigwin: section of the full range line being analyzed [samples]
           raw_path: path to the raw radar bxds files living under ORIG
           geo_path: path to the geometry files that live under S2_FIL
         chirp_path: path to the files containing the integer and quadrature components
                     of the reference chirp
     output_samples: number of output fast-time samples (3200 for MARFA)
         do_cinterp: (does something for HiCARS)
                 bp: flag to set bandpass filter for chirp (False for HiCARS legacy)

    Output:
    -----------
      denoised and dechirped HiCARS/MARFA data
    '''

    logging.debug("raw_path = " + raw_path)
    logging.debug("geo_path = " + geo_path)
    logging.debug("sigwin = " + str(sigwin))
    #sys.exit(0)

    # load the bxds datasets
    if gain == 'low':
        bxdsA = raw_bxds_load(raw_path, geo_path, '5', sigwin)
        bxdsB = raw_bxds_load(raw_path, geo_path, '7', sigwin)
    elif gain == 'high':
        bxdsA = raw_bxds_load(raw_path, geo_path, '6', sigwin)
        bxdsB = raw_bxds_load(raw_path, geo_path, '8', sigwin)

    assert bxdsA.shape == bxdsB.shape

    # trim of the range lines if desired. 
    # TODO: some of the downstream functions aren't aware of this.
    #logging.debug("bxds original shape="  + str(bxdsA.shape))
    #if sigwin[0] >= 0 and sigwin[1] > 0:
    #    bxdsA = bxdsA[sigwin[0]:sigwin[1], :]
    #    bxdsB = bxdsB[sigwin[0]:sigwin[1], :]

    # Output shape needs to be the same as the input
    output_samples = bxdsA.shape[0]

    if sigwin[3] != 0:
        bxdsA = bxdsA[:, sigwin[2]:sigwin[3]]
        bxdsB = bxdsB[:, sigwin[2]:sigwin[3]]
    logging.debug("bxds trimmed  shape="  + str(bxdsA.shape))

    # prepare the reference chirp
    hamm = hamming(output_samples)
    refchirp = get_ref_chirp(chirp_path, bandpass=bp, nsamp=output_samples)

    #plt.figure()
    #plt.subplot(311); plt.imshow(np.abs(bxdsA[sigwin[0]:sigwin[1], :]), aspect='auto'); plt.title('bxdsA')
    #plt.subplot(312); plt.imshow(np.abs(bxdsB[sigwin[0]:sigwin[1], :]), aspect='auto'); plt.title('bxdsB')
    #plt.subplot(313)
    #plt.plot(20 * np.log10(np.abs(bxdsA[0:200, 0000])), label='bxdsA - 0')
    #plt.plot(20 * np.log10(np.abs(bxdsA[0:200, 1000])), label='bxdsA - 1000')
    #plt.plot(20 * np.log10(np.abs(bxdsA[0:200, 2000])), label='bxdsA - 2000')
    #plt.plot(20 * np.log10(np.abs(bxdsB[0:200, 0000])), label='bxdsB - 0')
    #plt.plot(20 * np.log10(np.abs(bxdsB[0:200, 1000])), label='bxdsB - 1000')
    #plt.plot(20 * np.log10(np.abs(bxdsB[0:200, 2000])), label='bxdsB - 2000')
    #plt.legend()
    #plt.show()

    # prepare the outputs
    #dechirpA = np.zeros((len(bxdsA), np.size(bxdsA, axis=1)), dtype=complex)
    #dechirpB = np.zeros((len(bxdsB), np.size(bxdsB, axis=1)), dtype=complex)
    dechirpA = np.empty_like(bxdsA, dtype=complex)
    dechirpB = np.empty_like(bxdsB, dtype=complex)

    # dechirp
    for ii in range(np.size(bxdsA, axis=1)):
        dechirpA[:, ii] = dechirp(bxdsA[:, ii], refchirp, do_cinterp)
    for ii in range(np.size(bxdsB, axis=1)):
        dechirpB[:, ii] = dechirp(bxdsB[:, ii], refchirp, do_cinterp)

    return dechirpA, dechirpB

def test_denoise_and_dechirp():
    gain = 'low'
    trim_default = [0, 1000, 0, 0]

    # Test with a RADnh3 line
    snms = {'SRH1': 'RADnh5'}

    inputs = [
        {'prj': 'SRH1', 'line': 'DEV2/JKB2t/Y81a', 'trim': trim_default },
        {'prj': 'GOG3', 'line': 'NAQLK/JKB2j/ZY1b', 'trim': [0, 1000, 0, 12000] },
        {'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01a/', 'trim': [0, 1000, 15000, 27294] },
        {'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01b/', 'trim': [0, 1000, 0, 15000] },
    ]

    for rec in inputs:
        logging.debug("Processing line " + rec['line'])
        snm = snms.get(rec['prj'], 'RADnh3') # get stream name
        path = os.path.join('/disk/kea/WAIS/targ/xtra', rec['prj'], 'FOC/Best_Versions/')
        rawpath = os.path.join('/disk/kea/WAIS/orig/xlob', rec['line'], snm)
        geopath = os.path.join(path, 'S2_FIL', rec['line'])
        chirppath = os.path.join(path, 'S4_FOC', rec['line'])
        chirp_bp = snm == 'RADnh5' # not strictly true, but pretty close
        #(gain, sigwin, raw_path, geo_path, chirp_path,
        #                    output_samples=3200, do_cinterp=True, bp=True):
        da, db = denoise_and_dechirp(gain, rec['trim'], rawpath, geopath, chirppath, do_cinterp=False, bp=chirp_bp)


def test_load_power_image():
    logging.info("test_load_power_image()")
    path = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC/'

    # default value for trim
    trim = [0, 1000, 0, 0]
    #chirpwin = [0, 200]
    fresnel_stack = 15

    for line in ('DEV2/JKB2t/Y81a/', 'DEV2/JKB2t/Y81a'): # allow either
        for method in ('averaged','summed'):
            img = load_power_image(line, '1', trim, fresnel_stack, method, pth=path)

def dechirp(trace, refchirp, do_cinterp, output_samples=3200):
    '''
    Range line dechirp processor

    Inputs:
    -----------
            trace: radar range line
         refchirp: reference chirp
       do_cinterp: do frequency domain interpolation (coherent noise
                   removal) for HiCARS/MARFA data

    Outputs:
    -----------
      dechirped range line
    '''

    # find peak energy below blanking samples
    shifter = int(np.median(np.argmax(trace)))
    trace = np.roll(trace, -shifter)

    #DFT = np.fft.fft(trace)
    DFT = np.fft.fft(signal.detrend(trace))

    if do_cinterp:
        # Remove five samples per cycle problem
        DFT = cinterp(DFT, int(output_samples * (1 / 5) + 1))
        DFT = cinterp(DFT, int(output_samples * (1 - 1 / 5) + 1))
        # Remove the first harmonic for five samples
        DFT = cinterp(DFT, int(output_samples * (2 / 5) + 1))
        DFT = cinterp(DFT, int(output_samples * (1 - 2 / 5) + 1))

    # do the dechirp
    Product = np.multiply(refchirp, DFT)
    Dechirped = np.fft.ifft(Product)
    Dechirped = np.roll(Dechirped, shifter)

    return Dechirped

def chirp_phase_stability(reference, data, method='coherence', fs=50E6, rollval=10):
    '''
    Assessing the phase stability of the loop-back chirp. Will analyze
    the stability in the loopback chirps for alongtrack variations. I
    think the varibility between antennas should be handled correctly
    by the co-regesitration step later on.

    Inputs:
    ----------------
       reference: complex-valued range-compressed loopback chirp used
                  as a reference
                  -- typically set to be the first one in the array
            data: complex-valued loopback chirp data we want to compare
                  against the reference
          method: method to be used when assessing chirp stability
                  -- coherence: use coherence to compare signals
                     (doesn't work)
                  -- xcorr: cross-correlate reference with data across
                     some roll window to find the point of maximum
                     correlation.
              fs: HiCARS/MARFA fast-time sampling frequency [Hz]
         rollval: number of fast-time samples to roll through when
                  implementing xcorr method [-rollval, rollval)

    Outputs:
    ----------------
      output is an assessment of phase stability
    '''

    reference = 20 * np.log10(np.abs(reference))
    data = 20 * np.log10(np.abs(data))

    if method == 'coherence':
        ii = 1
        if ii == 1:
        #for ii in range(np.size(data, axis=1)):
            Cxy, f = signal.coherence(np.angle(reference),
                                      np.angle(data[:, ii]), fs,
                                      nperseg=len(reference))
    elif method == 'xcorr':

        C = np.zeros((np.size(data, axis=1))) + -99999
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        # for definitions of k, N
        N = max(data.shape[0], reference.shape[0])
        for ii in range(np.size(data, axis=1)): # for ii in range(data.shape[1]):
            # Use standard scipy correlation, which can be faster
            # R = correlate(data, reference)
            # k = int(np.argwhere(R == np.max(R)))
            # C[ii] = k - N + 1

            R = np.zeros((2 * rollval))
            rolls = np.arange(-rollval, rollval)
            for jj in range(len(rolls)):
                # is np.corrcoef supposed to be the same as complex_correlation_coefficient?
                CC = np.corrcoef(reference, np.roll(data[:, ii], rolls[jj]))
                R[jj] = np.abs(CC[0, 1])
            C[ii] = rolls[int(np.argwhere(R == np.max(R)))]

    else:
        raise ValueError('Unrecognized method {:s}'.format(method))

    return C

def phase_stability_adjustment(data, stability):
    '''
    application of chirp stability assessment results to the actual data.
    simply a roll of the data by some number of samples.

    Inputs:
    ---------------
          data: complex-valued radar data
     stability: shifts required to achieve chirp stability

    Outputs:
    ---------------
       chirp stability corrected complex-valued radar data
    '''

    out = np.zeros((len(data), np.size(data, axis=1)), dtype=complex)
    for ii in range(np.size(data, axis=1)):
        out[:, ii] = np.roll(data[:, ii], int(stability[ii]))

    return out

def quad3(X, X1, X2, X3, P1, P2, P3):
    '''
    function required in MARFA raw data load algorithm
    '''

    XX1 = X-X1
    XX2 = X-X2
    XX3 = X-X3
    X1X2 = X1-X2
    X2X3 = X2-X3
    X3X1 = X3-X1
    A = - (XX2*XX3)/(X1X2*X3X1)
    B = - (XX1*XX3)/(X1X2*X2X3)
    C = - (XX1*XX2)/(X3X1*X2X3)

    return A*P1 + B*P2 + C*P3

def raw_bxds_load(RadPath, GeoPath, channel, trim, DX=1, MS=3200, NR=1000, NRr=100):
    '''
    function to load raw MARFA bxds with the loopback chirp preserved.

    Inputs:
    ----------------
        InPath: Path to the raw radar files
       GeoPath: Path to the raw geometry files required to perform interpolation
       channel: desired MARFA channel to load
          trim: vector of values with trim[2]:trim[3] is range lines of interest.
                None to use entire range
            DX: alongtrack range line spacing after interpolation
            MS: number of fast-time samples in the output
            NR: block size to load data
           NRr: overlap between blocks(?)

    Outputs:
    ----------------
       output is an array of raw MARFA data for the line and channel in question
    '''

    RadName = os.path.join(RadPath, 'bxds')
    out = 0
    HiCARS = 2
    undersamp = True
    combined = True
    channel = int(channel)

    # load metadata
    Nc = np.fromfile(os.path.join(GeoPath, "Nc"), dtype=int, sep=" ")
    Xo = np.fromfile(os.path.join(GeoPath, "Xo"), sep=" ")
    NRt = np.fromfile(os.path.join(GeoPath, "NRt"), dtype=int, sep=" ")

    # define number of tears
    NumTears = len(NRt)

    # Make certain that NRr is an even number.
    if NRr % 2 == 1:
        NRr = NRr + 1

    # Check for single block case and force variables accordingly.
    if (NumTears == 1) and (NR > NRt[1-1]):
        NR = NRt[1-1]
        NRr = 0

    # NRb = Number of records included in each along-track filtering block
    NRb = NR + NRr

    ifd = None
#    OFD = open(OutName, "wb")

    # Define Range Filtering.
    # Tr = Range sampling time (0.02 microseconds; 50 MHz sampling)
    Tr = 0.02
    FilterR = np.zeros([MS, 1], complex)
    Freq1 = 02.5
    Freq2 = 17.5
    M1 = int(math.floor((Freq1 * Tr * MS) + 0.5)) + 1
    M2 = int(math.floor((Freq2 * Tr * MS) + 0.5)) + 1
    BW = M2 - M1
    #pcheck.pcheck(np.linspace(0.0,1.0,BW+1) * np.pi, 'Range')
    Hanning = np.reshape(np.sin(np.linspace(0.0, 1.0, BW+1) * np.pi), (-1, 1))
    FilterR[M1 - 1:M2] = Hanning
    FilterR[MS + 2 - M2 - 1:MS + 2 - M1] = Hanning

    # Define Along-Track Filtering.
    # Ta = Along-track sampling time (0.0025 s; 400 Hz sampling)
    Ta = 0.0025
    FilterA = np.zeros([NRb, 1], complex)
    Freq1 = 35.0
    Freq2 = 40.0
    N1 = int(math.floor((Freq1 * Ta * NRb) + 0.5)) + 1
    N2 = int(math.floor((Freq2 * Ta * NRb) + 0.5)) + 1
    BW = N2 - N1
    #pcheck.pcheck(np.linspace(0.0,1.0,BW+1) * np.pi, 'Range')
    Hanning = np.reshape(0.5 + 0.5 * np.cos(np.linspace(0.0, 1.0, BW + 1) * np.pi), (-1, 1))
    FilterA[N1-1:N2] = Hanning
    FilterA[NRb + 2 - N2 - 1:NRb + 2 - N1] = 1.0 - Hanning
    FilterA[0:N1 - 1] = 1.0
    FilterA[NRb + 3 - N1 - 1:NRb] = 1.0

    # Combine into 2D Filter
    Filter = FilterR * FilterA.conj().transpose()

    if channel in (5, 6, 7, 8):
        channel_specs = parse_channels('[1,%d,1,0,0]' % (int(channel)-4))
    else:
        raise Exception('filterRA: illegal channel number requested')
    tracegen = unfoc.read_RADnhx_gen(RadName, channel_specs)
    stackgen = unfoc.stacks_gen(tracegen, channel_specs, 1)

    NumRead = []

    # start processing

    for NT in range(1, NumTears + 1):

        # NRs = Number of records to process up to the next data tear
        if NT == 1:
            NRs = NRt[1 - 1]
        else:
            NRs = NRt[NT - 1] - NRt[NT - 1 - 1]

        # NumNBlocks = Number of along-track blocks
        NumNBlocks = max(1, int(math.floor((NRs + NR - 1 - NRr / 2) / NR)))

    #    NB = 1
    #    if NB == 1:
    #    for NB in range(1, 3):
        for NB in range(1, NumNBlocks + 1):

            if NB > 1:
                NRp = NumRead

            # NumRead = Number of new records to read
            NumRead = NR
            if NB == 1:
                NumRead = int(NR + (NRr/2))
            if NB == NumNBlocks:
                NumRead = int(NRs - ((NB-1)*NR) - (NRr/2))
            if (NB == 1) and (NB == NumNBlocks):
                NumRead = int(NRs)

            # NGPri = Number of initial (start) record being processed this block
            # NGPrf = Number of  final  (stop)  record being processed this block
            # NOTE: NGPri and NGPrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are not used anywhere else in this code,
            # but they are output here for progress reporting.
            if NT == 1:
                NGPri = ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = 1
                if NB == NumNBlocks:
                    NGPrf = NRs
            else:
                NGPri = NRt[NT - 1 - 1] + ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = NRt[NT - 1 - 1] + (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = NRt[NT - 1 - 1] + 1
                if NB == NumNBlocks:
                    NGPrf = NRt[NT - 1 - 1] + NRs

            # NGWri = Number of initial (start) record for controlling output on this processed block
            # NGWrf = Number of  final  (stop)  record for controlling output on this processed block
            # NOTE: NGWri and NGWrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are output for progress reporting.
            if NT == 1:
                NGWri = ((NB - 1) * NR) + 1
                NGWrf = (NB * NR)
                if NB == NumNBlocks:
                    NGWrf = NRs
            else:
                NGWri = NRt[NT - 1 - 1] + ((NB - 1) * NR) + 1
                NGWrf = NRt[NT - 1 - 1] + (NB * NR)
                if (NB == NumNBlocks):
                    NGWrf = NRt[NT - 1 - 1] + NRs

            # Read Data and define signal.
            # Pad (NRr/2) overlap region with first/last records on first/last blocks.
            if NB == 1:
                S = np.empty((MS, NumRead))
                if ifd is None:
                    for i in range(1, NumRead + 1):
                        try:
                            trace = next(stackgen)
                        except StopIteration:
                            sys.exit("Short read (stackgen failed at NB {} {})\n".format(NB, i))
                        S[:, i-1] = trace.data[0:MS]
                else:
                    for i in range(1, NumRead + 1):
                        data = np.fromfile(ifd, "<i2", MS)
                        if S.size < MS:
                            sys.exit("Short read (%d of %d)\n" % (S.size, MS))
                        S[:, i-1] = data

                signal = np.empty((MS, int(NRr / 2 + NumRead)))
                for N in range(1, int(NRr / 2 + 1)):
                    signal[:, N - 1] = S[:, 1 - 1]
                signal[:, int((NRr / 2) + 1 - 1):int((NRr / 2) + NumRead)] = S
            else:
                signal = np.empty((MS, int(NRr + NumRead)))
                signal[:, 1 - 1:NRr] = S[:, int(NRp - NRr + 1 - 1):NRp]
                if ifd is None:
                    S = np.empty((MS, NumRead))
                    for i in range(1, NumRead+1):
                        #print('Working:', NB, i)
                        try:
                            trace = next(stackgen)
                        except StopIteration:
                            #sys.exit("Short read (\n")
                            test = 1
                        S[:, i - 1] = trace.data[0:MS]
                else:
                    S = np.empty((MS, NumRead))
                    for i in range(1, NumRead + 1):
                        data = np.fromfile(ifd, "<i2", MS)
                        if S.size < MS:
                            sys.exit("Short read (%d of %d)\n" % (S.size, MS))
                        S[:, i - 1] = data
                signal[:, NRr + 1 - 1:NRr + NumRead] = S

            if (NB > 1) and (NB == NumNBlocks):
                signal = np.resize(signal, (MS, NRb))
                # This gives a warning when running profile or coverage
                #ValueError: cannot resize an array that references or is referenced
                #by another array in this way.
                #Use the np.resize function or refcheck=False

                #signal.resize((MS, NRb))
                for N in range(NRr + NumRead + 1, NRb + 1):
                    signal[:, N - 1] = S[:, NumRead - 1]

            F = pyfftw.interfaces.numpy_fft.fft2(detrend(signal, 0), [MS, NRb])

            ### Clear top samples
            #if (ifd == -1):
            #    # HiCARS2
            #    signal[0:250,:] = 0

            if undersamp == 0:
                Fs = pyfftw.interfaces.numpy_fft.fft2(signal[MS - 800 - 1:MS - 1], [800 - 1, NRb])
                F[2561-1, :] = F[2561-1, :] - 4 * Fs[641 - 1, :]

            F = Filter * F
            signal = pyfftw.interfaces.numpy_fft.ifft2(F, [MS, NRb])

            if NB == 1:
                Nii = int(math.floor((Xo[NGWri - 1] / DX) + 0.99999)) + 1
            else:
                Nii = Nif + 1
            Nif = int(math.floor(Xo[NGWrf - 1] / DX)) + 1

            # Interpolate filtered signal to resampling points
            signali = np.empty((MS, Nif - Nii + 1), complex)
            for Ni in range(Nii, Nif + 1):
                Nci = Nc[Ni - 1]
                Nci = max(Nci, 2)
                Nci = min(Nci, len(Xo) - 1)
                X = (Ni - 1) * DX
                X1 = Xo[Nci - 1 - 1]
                X2 = Xo[Nci - 1]
                X3 = Xo[Nci + 1 - 1]
                P1 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) - 1 - 1)]
                P2 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) - 1)]
                P3 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) + 1 - 1)]
                signali[:, Ni - Nii + 1 - 1] = quad3(X, X1, X2, X3, P1, P2, P3)

            # Part 1: Generate missing data at start of data tear.
            if (NT > 1) and (NB == 1):
                D1 = Xo[NRt[NT - 1 - 1] - 1]
                D2 = Xo[NRt[NT - 1 - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                for Ni in range(N1 + int(math.floor((N2 - N1 + 2) / 2)),
                                N1 + int(math.floor((N2 - N1 + 2) / 2))
                                   + max(0, int(math.floor((N2 - N1 - 19) / 2))) - 1 + 1):
                    signalim[1 - 1:MS - 1] = 0.0
                    out = out + 1
                for Ni in range(N1 + int(math.floor((N2 - N1 + 2) / 2))
                                   + max(0, int(math.floor((N2 - N1 - 19) / 2))), N2 + 1):
                    Wt = 0.5 - 0.5 * math.cos((math.pi / 10.0) * (Ni - (N2 - 9)))
                    signalim = Wt * signali[:, 1 - 1]
                    out = out + 1
                signalout = signalim

            # Part 2: Output good resampled points.
            if (NT <= 1) and (NB == 1):
                signalout = signali
            else:
                signalout = np.concatenate((signalout, signali), axis=1)
            out = out + signali.shape[1]

            # Part 3: Generate missing data at end of data tear.
            if (NT < NumTears) and (NB == NumNBlocks):
                D1 = Xo[NRt[NT - 1] - 1]
                D2 = Xo[NRt[NT - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                for Ni in range(N1 + 1, N1 + 1 + min(9, int(math.floor((N2 - N1) / 2))) - 1 + 1):
                    Wt = 0.5 + 0.5 * math.cos((math.pi / 10.0) * (Ni - N1))
                    signalim = Wt * signali[:, Nif - Nii + 1 - 1]
                    out = out + 1
                for Ni in range(N1 + min(10, int(math.floor((N2 - N1 + 2) / 2))),
                                N1 + min(10, int(math.floor((N2 - N1 + 2) / 2)))
                                   + max(0, int(math.floor((N2 - N1 - 18) / 2))) - 1 + 1):
                    signalim[1 - 1:MS - 1] = 0.0
                    out = out + 1
                signalout = np.concatenate((signalout, signalim), axis=1)

    if ifd is not None:
        ifd.close()

    if trim is not None and trim[3] != 0:
        signalout = signalout[:, trim[2]:trim[3]]

    return signalout



def raw_bxds_readplan(Nc, Xo, NRt, DX=1, MS=3200, NR=1000, NRr=100):
    '''
    function to calculate the number of records to read, and sizes of output blocks, but
    don't actually do any reading.

    Inputs:
    ----------------
            DX: alongtrack range line spacing after interpolation
            MS: number of fast-time samples in the output
            NR: block size to load data (input bxds records per output record)
           NRr: overlap between blocks(?)

    Outputs:
    ----------------
       output is an array of raw MARFA data for the line and channel in question
    '''
    logging.debug("raw_bxds_readplan() start")
    out = 0 # number of output records
    #HiCARS = 2
    #undersamp = True
    #combined = True

    # define number of tears
    NumTears = len(NRt)

    # Make certain that NRr is an even number.
    if NRr % 2 == 1:
        NRr += 1

    # Check for single block case and force variables accordingly.
    if (NumTears == 1) and (NR > NRt[1-1]):
        NR = NRt[1-1]
        NRr = 0

    # NRb = Number of records included in each along-track filtering block
    NRb = NR + NRr

    NumRead = None

    # start processing

    for NT in range(1, NumTears + 1):

        # NRs = Number of records to process up to the next data tear
        if NT == 1:
            NRs = NRt[0] #NRt[1 - 1]
        else:
            NRs = NRt[NT - 1] - NRt[NT - 2] #NRt[NT - 1] - NRt[NT - 1 - 1]

        # NumNBlocks = Number of along-track blocks
        NumNBlocks = max(1, int(math.floor((NRs + NR - 1 - NRr / 2) / NR)))

        for NB in range(1, NumNBlocks + 1):

            if NB > 1:
                NRp = NumRead

            # TODO: GNG -- this structure is peculiar.
            # NumRead = Number of new records to read
            NumRead = NR
            if NB == 1:
                NumRead = int(NR + (NRr/2))
            if NB == NumNBlocks:
                NumRead = int(NRs - ((NB-1)*NR) - (NRr/2))
            if (NB == 1) and (NB == NumNBlocks):
                NumRead = int(NRs)

            # NGPri = Number of initial (start) record being processed this block
            # NGPrf = Number of  final  (stop)  record being processed this block
            # NOTE: NGPri and NGPrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are not used anywhere else in this code,
            # but they are output here for progress reporting.
            if NT == 1:
                NGPri = ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = 1
                if NB == NumNBlocks:
                    NGPrf = NRs
            else:
                NGPri = NRt[NT - 1 - 1] + ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = NRt[NT - 1 - 1] + (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = NRt[NT - 1 - 1] + 1
                if NB == NumNBlocks:
                    NGPrf = NRt[NT - 1 - 1] + NRs

            # NGWri = Number of initial (start) record for controlling output on this processed block
            # NGWrf = Number of  final  (stop)  record for controlling output on this processed block
            # NOTE: NGWri and NGWrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are output for progress reporting.
            if NT == 1:
                NGWri = ((NB - 1) * NR) + 1
                NGWrf = (NB * NR)
                if NB == NumNBlocks:
                    NGWrf = NRs
            else:
                NGWri = NRt[NT - 1 - 1] + ((NB - 1) * NR) + 1
                NGWrf = NRt[NT - 1 - 1] + (NB * NR)
                if (NB == NumNBlocks):
                    NGWrf = NRt[NT - 1 - 1] + NRs

            # Read Data and define signal.
            # Pad (NRr/2) overlap region with first/last records on first/last blocks.
            if NB == 1:
                S = np.empty((MS, NumRead))
                signal = np.empty((MS, int(NRr / 2 + NumRead)))
                signal[:, int((NRr / 2) + 1 - 1):int((NRr / 2) + NumRead)] = S
            else:
                signal = np.empty((MS, int(NRr + NumRead)))
                S = np.empty((MS, NumRead))

            if (NB > 1) and (NB == NumNBlocks):
                signal = np.resize(signal, (MS, NRb))

            if NB == 1:
                Nii = int(math.floor((Xo[NGWri - 1] / DX) + 0.99999)) + 1
            else:
                Nii = Nif + 1
            Nif = int(math.floor(Xo[NGWrf - 1] / DX)) + 1

            # Interpolate filtered signal to resampling points
            signali = np.empty((MS, Nif - Nii + 1), complex)

            # GNG: Part 1 and Part 2 are basically mutually exclusive conditions. Refactor?
            # Part 1: Generate missing data at start of data tear segment.
            if (NT > 1) and (NB == 1):
                D1 = Xo[NRt[NT - 1 - 1] - 1]
                D2 = Xo[NRt[NT - 1 - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                range_start = N1 + int(math.floor((N2 - N1 + 2) / 2))
                range_len = max(0, int(math.floor((N2 - N1 - 19) / 2))) - 1 + 1
                for Ni in range(range_start, range_start + range_len):
                    signalim[1 - 1:MS - 1] = 0.0
                    out += 1
                range_start = N1 + int(math.floor((N2 - N1 + 2) / 2)) \
                            + max(0, int(math.floor((N2 - N1 - 19) / 2))), N2 + 1
                range_end = N2 + 1
                for Ni in range(range_start, range_end):
                    Wt = 0  #0.5 - 0.5 * math.cos((math.pi / 10.0) * (Ni - (N2 - 9)))
                    signalim = Wt * signali[:, 1 - 1]
                    out += 1

                assert signali.shape[1] == (range_len + (range_end - range_start))
                signalout = signalim

            # Part 2: Output good resampled points.
            out += signali.shape[1]

            # Part 3: Generate missing data at end of data tear.
            if (NT < NumTears) and (NB == NumNBlocks):
                D1 = Xo[NRt[NT - 1] - 1]
                D2 = Xo[NRt[NT - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                range_start = N1 + 1
                range_len0 = min(9, int(math.floor((N2 - N1) / 2))) - 1 + 1
                for Ni in range(range_start, range_start + range_len0):
                    Wt = 0 # 0.5 + 0.5 * math.cos((math.pi / 10.0) * (Ni - N1))
                    signalim = Wt * signali[:, Nif - Nii + 1 - 1]
                    out += 1
                range_start = N1 + min(10, int(math.floor((N2 - N1 + 2) / 2)))
                range_len1 = max(0, int(math.floor((N2 - N1 - 18) / 2))) - 1 + 1
                for Ni in range(range_start, range_start + range_len1):
                    signalim[1 - 1:MS - 1] = 0.0
                    out += 1
                assert range_len0 + range_len1 == signalim.shape[1]
                #signalout = np.concatenate((signalout, signalim), axis=1)

    logging.debug("bxds_load_readplan() = {:d}".format(out))
    return out


def raw_bxds_load__read_stackgen_block(stackgen, MS, num_read, NB):
    """ 
    Read a block of NumRead traces with trace length == MS from a stack generator
    Inputs:
    ---------------
    stackgen: stack generator
    MS: number of samples per trace
    num_read: Number of traces to read
    NB: block number
    
    Outputs:
    -------
    s: array of traces
    """

    s = np.empty((MS, num_read))
    for i in range(num_read):
        try:
            trace = next(stackgen)
        except StopIteration:
            raise Exception("Short read (stackgen failed at NB={:d} i={:d})\n".format(int(NB), i))
        s[:, i] = trace.data[0:MS]

    return s

def raw_bxds_load__read_file_block(ifd, MS, num_read, NB):
    """ 
    Read a block of NumRead traces with trace length == MS from a file
    Inputs:
    ---------------
         ifd: input file descriptor
          MS: number of samples per trace
    num_read: Number of traces to read
          NB: block number
    
    Outputs:
    -------
    s: array of traces
    """

    s = np.empty((MS, num_read))
    for i in range(num_read):   
        data = np.fromfile(ifd, "<i2", MS)
        if data.size < MS:
            raise Exception("Short read at NB={:d} i={:d} ({:d} of {:d})\n".format(NB, i, S.size, MS))
        s[:, i] = data
    return s

def raw_bxds_load2(RadPath, GeoPath, channel, trim, DX=1, MS=3200, NR=1000, NRr=100):
    '''
    function to load raw MARFA bxds with the loopback chirp preserved.
    being optimized for speed, and to remove idiosyncracies from the matlab conversion. ngg

    Inputs:
    ----------------
        InPath: Path to the raw radar files
       GeoPath: Path to the raw geometry files required to perform interpolation
       channel: desired MARFA channel to load
          trim: vector of values with trim[2]:trim[3] is range lines of interest.
                None to use entire range
            DX: alongtrack range line spacing after interpolation
            MS: number of fast-time samples in the output
            NR: block size to load data (input bxds records per output record)
           NRr: overlap between blocks(?)

    Outputs:
    ----------------
       output is an array of raw MARFA data for the line and channel in question
    '''

    RadName = os.path.join(RadPath, 'bxds')
    out = 0 # number of output records
    HiCARS = 2
    undersamp = True
    combined = True
    channel = int(channel)

    # load metadata
    Nc = np.fromfile(os.path.join(GeoPath, "Nc"), dtype=int, sep=" ")
    Xo = np.fromfile(os.path.join(GeoPath, "Xo"), sep=" ")
    NRt = np.fromfile(os.path.join(GeoPath, "NRt"), dtype=int, sep=" ")

    out_predicted = raw_bxds_readplan(Nc, Xo, NRt, DX, MS, NR, NRr)
    signalout = np.empty((MS, out_predicted), dtype=complex)
    #sys.exit(0)

    # define number of tears
    NumTears = len(NRt)

    # Make certain that NRr is an even number.
    if NRr % 2 == 1:
        NRr += 1

    # Check for single block case and force variables accordingly.
    if (NumTears == 1) and (NR > NRt[1-1]):
        NR = NRt[1-1]
        NRr = 0

    # NRb = Number of records included in each along-track filtering block
    NRb = NR + NRr

    # File descriptor for debugging
    ifd = None
#    OFD = open(OutName, "wb")

    # Define Range Filtering.
    # Tr = Range sampling time (0.02 microseconds; 50 MHz sampling)
    Tr = 0.02
    FilterR = np.zeros([MS, 1], complex)
    Freq1 = 02.5
    Freq2 = 17.5
    M1 = int(math.floor((Freq1 * Tr * MS) + 0.5)) + 1
    M2 = int(math.floor((Freq2 * Tr * MS) + 0.5)) + 1
    BW = M2 - M1
    #pcheck.pcheck(np.linspace(0.0,1.0,BW+1) * np.pi, 'Range')
    Hanning = np.reshape(np.sin(np.linspace(0.0, 1.0, BW+1) * np.pi), (-1, 1))
    FilterR[M1 - 1:M2] = Hanning
    FilterR[MS + 2 - M2 - 1:MS + 2 - M1] = Hanning

    # Define Along-Track Filtering.
    # Ta = Along-track sampling time (0.0025 s; 400 Hz sampling)
    Ta = 0.0025
    FilterA = np.zeros([NRb, 1], complex)
    Freq1 = 35.0
    Freq2 = 40.0
    N1 = int(math.floor((Freq1 * Ta * NRb) + 0.5)) + 1
    N2 = int(math.floor((Freq2 * Ta * NRb) + 0.5)) + 1
    BW = N2 - N1
    #pcheck.pcheck(np.linspace(0.0,1.0,BW+1) * np.pi, 'Range')
    Hanning = np.reshape(0.5 + 0.5 * np.cos(np.linspace(0.0, 1.0, BW + 1) * np.pi), (-1, 1))
    FilterA[N1-1:N2] = Hanning
    FilterA[NRb + 2 - N2 - 1:NRb + 2 - N1] = 1.0 - Hanning
    FilterA[0:N1 - 1] = 1.0
    FilterA[NRb + 3 - N1 - 1:NRb] = 1.0

    # Combine into 2D Filter
    Filter = FilterR * FilterA.conj().transpose()

    if channel in (5, 6, 7, 8):
        channel_specs = parse_channels('[1,%d,1,0,0]' % (int(channel)-4))
    else:
        raise ValueError('filterRA: illegal channel number requested')
    tracegen = unfoc.read_RADnhx_gen(RadName, channel_specs)
    stackgen = unfoc.stacks_gen(tracegen, channel_specs, 1)

    NumRead = None

    # start processing each segment
    for NT in range(1, NumTears + 1):

        # NRs = Number of records to process up to the next data tear
        if NT == 1:
            NRs = NRt[0] #NRt[1 - 1]
        else:
            NRs = NRt[NT - 1] - NRt[NT - 2] #NRt[NT - 1] - NRt[NT - 1 - 1]

        # NumNBlocks = Number of along-track blocks
        NumNBlocks = max(1, int(math.floor((NRs + NR - 1 - NRr / 2) / NR)))

        # Process each block in this segment
        for NB in range(1, NumNBlocks + 1):

            if NB > 1:
                NRp = NumRead

            # TODO: GNG -- this structure is peculiar.
            # NumRead = Number of new records to read
            NumRead = NR
            if NB == 1:
                NumRead = int(NR + (NRr/2))
            if NB == NumNBlocks:
                NumRead = int(NRs - ((NB-1)*NR) - (NRr/2))
            if (NB == 1) and (NB == NumNBlocks):
                NumRead = int(NRs)

            # NGPri = Number of initial (start) record being processed this block
            # NGPrf = Number of  final  (stop)  record being processed this block
            # NOTE: NGPri and NGPrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are not used anywhere else in this code,
            # but they are output here for progress reporting.
            if NT == 1:
                NGPri = ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = 1
                if NB == NumNBlocks:
                    NGPrf = NRs
            else:
                NGPri = NRt[NT - 1 - 1] + ((NB - 1) * NR) - (NRr / 2) + 1
                NGPrf = NRt[NT - 1 - 1] + (NB * NR) + (NRr / 2)
                if NB == 1:
                    NGPri = NRt[NT - 1 - 1] + 1
                if NB == NumNBlocks:
                    NGPrf = NRt[NT - 1 - 1] + NRs

            # NGWri = Number of initial (start) record for controlling output on this processed block
            # NGWrf = Number of  final  (stop)  record for controlling output on this processed block
            # NOTE: NGWri and NGWrf are in the global index system, where
            #       "global" refers to the full set of records.
            # These variables are output for progress reporting.
            if NT == 1:
                NGWri = ((NB - 1) * NR) + 1
                NGWrf = (NB * NR)
                if NB == NumNBlocks:
                    NGWrf = NRs
            else:
                NGWri = NRt[NT - 1 - 1] + ((NB - 1) * NR) + 1
                NGWrf = NRt[NT - 1 - 1] + (NB * NR)
                if (NB == NumNBlocks):
                    NGWrf = NRt[NT - 1 - 1] + NRs

            # Read Data and define signal.
            # Pad (NRr/2) overlap region with first/last records on first/last blocks.
            if NB == 1:
                #S = np.empty((MS, NumRead))
                if ifd is None:
                    S = raw_bxds_load__read_stackgen_block(stackgen, MS=MS, num_read=NumRead, NB=NB)
                else:
                    S = raw_bxds_load__read_file_block(ifd, MS=MS, num_read=NumRead, NB=NB)

                signal = np.empty((MS, int(NRr / 2 + NumRead)))
                for N in range(1, int(NRr / 2 + 1)):
                    signal[:, N - 1] = S[:, 1 - 1]
                signal[:, int((NRr / 2) + 1 - 1):int((NRr / 2) + NumRead)] = S
            else:
                signal = np.empty((MS, int(NRr + NumRead)))
                signal[:, 1 - 1:NRr] = S[:, int(NRp - NRr + 1 - 1):NRp]
                if ifd is None:
                    S = raw_bxds_load__read_stackgen_block(stackgen, MS=MS, num_read=NumRead, NB=NB)
                else:
                    S = raw_bxds_load__read_file_block(ifd, MS=MS, num_read=NumRead, NB=NB)

                signal[:, NRr + 1 - 1:NRr + NumRead] = S

            if (NB > 1) and (NB == NumNBlocks):
                signal = np.resize(signal, (MS, NRb))
                # This gives a warning when running profile or coverage
                #ValueError: cannot resize an array that references or is referenced
                #by another array in this way.
                #Use the np.resize function or refcheck=False

                #signal.resize((MS, NRb))
                # pad with the last value
                for N in range(NRr + NumRead + 1, NRb + 1):
                    signal[:, N - 1] = S[:, NumRead - 1]

            F = pyfftw.interfaces.numpy_fft.fft2(detrend(signal, 0), [MS, NRb])

            ### Clear top samples
            #if (ifd == -1):
            #    # HiCARS2
            #    signal[0:250,:] = 0

            if undersamp == 0:
                Fs = pyfftw.interfaces.numpy_fft.fft2(signal[MS - 800 - 1:MS - 1], [800 - 1, NRb])
                F[2561-1, :] = F[2561-1, :] - 4 * Fs[641 - 1, :]

            F = Filter * F
            signal = pyfftw.interfaces.numpy_fft.ifft2(F, [MS, NRb])

            if NB == 1:
                Nii = int(math.floor((Xo[NGWri - 1] / DX) + 0.99999)) + 1
            else:
                Nii = Nif + 1
            Nif = int(math.floor(Xo[NGWrf - 1] / DX)) + 1

            # Interpolate filtered signal to resampling points
            signali = np.empty((MS, Nif - Nii + 1), complex)
            for Ni in range(Nii, Nif + 1):
                Nci = Nc[Ni - 1]
                Nci = max(Nci, 2)
                Nci = min(Nci, len(Xo) - 1)
                X = (Ni - 1) * DX
                X1 = Xo[Nci - 1 - 1]
                X2 = Xo[Nci - 1]
                X3 = Xo[Nci + 1 - 1]
                P1 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) - 1 - 1)]
                P2 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) - 1)]
                P3 = signal[:, int((Nci - NGWri + (NRr / 2) + 1) + 1 - 1)]
                signali[:, Ni - Nii + 1 - 1] = quad3(X, X1, X2, X3, P1, P2, P3)

            # GNG: Part 1 and Part 2 are basically mutually exclusive conditions. Refactor?
            # Part 1: Generate missing data at start of data tear.
            if (NT > 1) and (NB == 1):
                D1 = Xo[NRt[NT - 1 - 1] - 1]
                D2 = Xo[NRt[NT - 1 - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                range_start = N1 + int(math.floor((N2 - N1 + 2) / 2))
                range_len = max(0, int(math.floor((N2 - N1 - 19) / 2))) - 1 + 1
                for Ni in range(range_start, range_start + range_len):
                    signalim[1 - 1:MS - 1] = 0.0
                    out += 1
                for Ni in range(range_start + range_len, N2 + 1):
                    Wt = 0.5 - 0.5 * math.cos((math.pi / 10.0) * (Ni - (N2 - 9)))
                    signalim = Wt * signali[:, 1 - 1]
                    out += 1
                #signalout = signalim
                signalout[:, 0:out] = signalim

            # Part 2: Output good resampled points.
            if (NT <= 1) and (NB == 1):
                #signalout = signali
                assert out == 0
                signalout[:, out:(out + signali.shape[1])] = signali
            else:
                #signalout = np.concatenate((signalout, signali), axis=1)
                signalout[:, out:(out + signali.shape[1])] = signali
            out += signali.shape[1]

            # Part 3: Generate missing data at end of data tear.
            if (NT < NumTears) and (NB == NumNBlocks):
                D1 = Xo[NRt[NT - 1] - 1]
                D2 = Xo[NRt[NT - 1] + 1 - 1]
                N1 = int(math.floor(D1 / DX)) + 1
                N2 = int(math.floor(D2 / DX)) + 1
                range_start = N1 + 1
                range_len = min(9, int(math.floor((N2 - N1) / 2))) - 1 + 1
                out0 = out
                for Ni in range(range_start, range_start + range_len):
                    Wt = 0.5 + 0.5 * math.cos((math.pi / 10.0) * (Ni - N1))
                    signalim = Wt * signali[:, Nif - Nii + 1 - 1]
                    out += 1
                range_start = N1 + min(10, int(math.floor((N2 - N1 + 2) / 2)))
                range_len = max(0, int(math.floor((N2 - N1 - 18) / 2))) - 1 + 1
                for Ni in range(range_start, range_start + range_len):
                    signalim[1 - 1:MS - 1] = 0.0
                    out += 1
                #signalout = np.concatenate((signalout, signalim), axis=1)
                signalout[:, out0:out] = signalim

    assert out_predicted == out

    if ifd is not None:
        ifd.close()

    if trim is not None and trim[3] != 0:
        signalout = signalout[:, trim[2]:trim[3]]

    return signalout

def test_raw_bxds_load():
    #raw_path = "/disk/kea/WAIS/orig/xlob/DEV2/JKB2t/Y81a/RADnh5"
    #geo_path = "/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV2/JKB2t/Y81a"
    #sigwin = [0, 1000, 0, 0]

    raw_path = "/disk/kea/WAIS/orig/xlob/NAQLK/JKB2j/ZY1b/RADnh3/"
    geo_path = "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1b/"
    sigwin = [0, 1000, 0, 12000]



    chan = '5'
    logging.debug("raw_bxds_load2()")
    bxds2 = raw_bxds_load2(raw_path, geo_path, chan, sigwin)

    logging.debug("raw_bxds_load()")
    bxds1 = raw_bxds_load(raw_path, geo_path, chan, sigwin)

    rmse = np.sqrt(np.square(abs(bxds2 - bxds1)).mean())
    logging.debug("RMSE(raw_bxds_load - raw_bxds_load2) = {:0.3g}".format(rmse))
    assert rmse < 1e-9



def complex_correlation_coefficient(cmp1, cmp2):
    '''
    Calculate the complex correlation coefficient between two complex
    valued datasets.

    Inputs
    ----------------
        cmp1: complex-valued dataset 1
        cmp2: complex-valued dataset 2

    Outputs:
    ----------------
      complex valued correlation coefficient
    '''

    tempA = np.mean(np.multiply(cmp1, np.conj(cmp2)))
    tempB = np.sqrt(np.multiply(np.mean(np.square(np.abs(cmp1))), np.mean(np.square(np.abs(cmp2)))))
    out = np.divide(tempA, tempB)

    return out

def azimuth_pixel2pixel_coherence(data, FOI, roll_range=100, ft_step=1):
    '''
    Assessment of the coherence between azimuth pixels as a function
    of the distance between them. Non-independent azimuth pixels will
    exhibit greater pixel-to-pixel coherence than truly independent
    ones. Based on discussion presented in Lee et al. (1994).

    The algorithm will only evaluate pixel-to-pixel coherence in the fast-
    time sample range covering the defined the feature-of-interest.

    Inputs:
    ----------------
           data: complex-valued data relating to one of the two
                 interferometric channels
                 -- fast-time samples as rows
                 -- range lines as columns
            FOI: array of ones and zeros where ones correspond to the
                 picked feature-of-interest being evaluated for possible
                 off-nadir clutter
                 -- fast-time samples as rows
                 -- range lines as columns
     roll_range: range of azimuth sample to roll the data over when
                 evaluating pixel-to-pixel coherency
        ft_step: fast-time sample interval at which to evaluate the azimuthal
                 pixel-to-pixel coherence

    Outputs:
    ----------------
            rho: array of the pixel-to-pixel coherency
                 -- rows contain evaluations of the pixel-to-pixel coherency
                    for individual fast-time samples in the input radar data
                 -- columns represent the distances over which the coherency
                    was evaluated
          rolls: list of the distances between pixels over which the
                 pixel-to-pixel coherency was calculated
     ft_samples: list of fast-time samples along with coherencies were evaluated
    '''

    # from FOI, identify fast-time samples covering the FOI
    test = np.zeros((len(FOI)), dtype=float)
    for ii in range(len(FOI)):
        if np.nansum(FOI[ii, :]) != 0:
            test[ii] = 1
    start_sample = np.min(np.argwhere(test == 1))
    end_sample = np.max(np.argwhere(test == 1))
    ft_samples = np.arange(start_sample, end_sample, ft_step)

    # define the pixel-to-pixel distances to test
    rolls = np.arange(0, roll_range + 1)

    # calculate the pixel-to-pixel coherence
    rho = np.zeros((len(ft_samples), len(rolls)), dtype=float)
    for ii in range(len(ft_samples)):
        cmp1 = data[ft_samples[ii], :]
        for jj in range(len(rolls)):
            cmp2 = np.roll(cmp1, rolls[jj])
            rho[ii, jj] = np.abs(complex_correlation_coefficient(cmp1, cmp2))

    return rho, rolls, ft_samples

def independent_azimuth_samples(cmpA, cmpB, FOI, roll_range=100, ft_step=1):
    '''
    Function to determine the azimuth sample interval between independent
    range lines.
    
    Inputs:
    ----------------
           cmpA: complex-valued antenna A radargram
                 -- fast-time samples as rows
                 -- range lines as columns
           cmpB: complex-valued antenna B radargram
                 -- fast-time samples as rows
                 -- range lines as columns
            FOI: array of ones and zeros where ones correspond to the
                 picked feature-of-interest being evaluated for possible
                 off-nadir clutter
                 -- fast-time samples as rows
                 -- range lines as columns
     roll_range: range of azimuth sample to roll the data over when
                 evaluating pixel-to-pixel coherency
        ft_step: fast-time sample interval at which to evaluate the azimuthal
                 pixel-to-pixel coherence

    Outputs:
    ----------------
      output is the azimuth distances [samples] between independent range lines
    '''

    # evaluate pixel-to-pixel stability
    rhoA, rolls, ft_samples = azimuth_pixel2pixel_coherence(cmpA, FOI, roll_range=roll_range, ft_step=ft_step)
    rhoB, _, _ = azimuth_pixel2pixel_coherence(cmpB, FOI, roll_range=roll_range, ft_step=ft_step)

    # plot the pixel-to-pixel correlation
    plt.figure()
    plt.subplot(211)
    num = 0
    for ii in range(len(rhoA)):
        num += 1
        plt.plot(rhoA[ii, :], label=str(ft_samples[ii]))
    plt.title('Antenna A pixel-to-pixel coherence')
    plt.ylim([0, 1]); plt.xlim([0, len(rolls)])
    plt.xlabel('pixel-to-pixel distance [samples]')
    plt.xticks(np.arange(0, len(rolls)), rolls)
    plt.ylabel('coherence')
    if num <= 15: plt.legend()
    plt.subplot(212)
    num = 0
    for ii in range(len(rhoB)):
        num += 1
        plt.plot(rhoB[ii, :], label=str(ft_samples[ii]))
    plt.title('Antenna B pixel-to-pixel coherence')
    plt.ylim([0, 1]); plt.xlim([0, len(rolls)])
    plt.xlabel('pixel-to-pixel distance [samples]')
    plt.xticks(np.arange(0, len(rolls)), rolls)
    plt.ylabel('coherence')
    if num <= 15: plt.legend()

    # define the number of azimuth samples between independent looks
    def callback():
        global az_step
        az_step = e1.get()
    master = Tk()
    Label(master, text='Interval between independent looks (Enter to assign)').grid(row=0)
    e1 = Entry(master)
    e1.grid(row=0, column=1)
    Button(master, text='Enter', command=callback).grid(row=1, column=0, sticky=W, pady=4)
    Button(master, text='Done', command=master.quit).grid(row=1, column=1, sticky=W, pady=4)
    plt.show()

    return az_step

def interferogram_normalization(interferogram, surface):
    '''
    Function to normalize interferogram using the interferometric phases defined along
    a defined surface. The function has an internal step to make sure the 'surface'
    input covers the entire breadth of the interferogram. The function interpolates the
    picked surface into spaces where the surface isn't defined.
    
    Inputs:
    ----------------
      interferogram: stacked interferogram
            surface: picked surface for interferogram normalization

    Outputs:
    ----------------
      output is the normalized interferogram
    '''

    # ensure the picked surface covers the breadth of the interferogram
    test = np.nansum(surface, axis=0)
    x = np.argwhere(test == 1)[:, 0]
    y = np.zeros((len(x)), dtype = int)
    for ii in range(len(x)):
        y[ii] = np.argwhere(surface[:, x[ii]] == 1)
    xall = np.arange(0, np.size(surface, axis=1))
    yall = np.interp(xall, x, y)
    surface2 = np.zeros((len(surface), np.size(surface, axis=1)), dtype=float)
    surface2[:, :] = np.nan
    for ii in range(len(xall)):
        surface2[int(yall[ii]), ii] = 1

    # extract interferometric phase along the surface
    #surface_phase = FOI_extraction(interferogram, surface2)
    surface_phase = np.zeros((len(xall)), dtype=float)
    for ii in range(len(xall)):
        surface_phase[ii] = interferogram[int(yall[ii]), int(xall[ii])]

    # normalize interferogram
    #output = np.zeros((len(interferogram), len(xall)), dtype=float)
    #for ii in range(len(surface_phase)):
    #    output[:, ii] = interferogram[:, ii] - surface_phase[ii]
    corr = np.ones((len(interferogram), len(xall)), dtype=float)
    for ii in range(len(xall)):
        corr[:, ii] = np.multiply(corr[:, ii], surface_phase[ii])
    output = interferogram - corr

    return output, surface_phase, surface2

def surface_pick(image, FOI, pick_method='maximum'):
    '''
    Function to pick the surface echo overlying an already defined subsurface
    feature-of-interest
    
    Inputs:
    ----------------
           image: power image
             FOI: array of the same size as the power image with ones where
                  a feature-of-interest has been defined
     pick_method: string indentifying how individual pixels corresponding
                  to the surface are picked during surface definition

    Outputs:
    ----------------
      out is an array of the same size as the poower image with ones
          where the surface overlying the feature-of-interst has been
          defined
    '''

    # extract the along-track sample bounds in the picked FOI
    inds = np.argwhere(FOI == 1)[:, 1]
    indA = min(inds)
    indB = max(inds)
    image2 = image[:, indA:indB]

    # pick the surface within the area covered by the FOI
    SRF = ip.picker(np.transpose(image2), snap_to=pick_method)
    SRF = np.transpose(SRF)
    
    # ensure the picked surface covers the breadth of the trimmed power image
    test = np.nansum(SRF, axis=0)
    x = np.argwhere(test == 1)[:, 0]
    y = np.zeros((len(x)), dtype = int)
    for ii in range(len(x)):
        y[ii] = np.argwhere(SRF[:, x[ii]] == 1)
    xall = np.arange(0, np.size(SRF, axis=1))
    yall = np.interp(xall, x, y)
    SRF2 = np.zeros((len(SRF), np.size(SRF, axis=1)), dtype=float)
    SRF2[:, :] = np.nan
    for ii in range(len(xall)):
        SRF2[int(yall[ii]), ii] = 1

    # position the picked surface within the full breadth of the power image
    tempA = np.nan * np.ones((len(FOI), indA))
    tempB = np.nan * np.ones((len(FOI), np.size(FOI, axis=1) - indB))
    out = np.concatenate((tempA, SRF, tempB), axis=1)

    return out

def offnadir_clutter(FOI, SRF, rollang, N, B, mb_offset, l, dt):
    '''
    Function to estimate the mean phase as if the user defined FOI were
    off-nadir surface clutter. This is done by first estimating the
    cross-track look angle assuming all propagation is at the speed
    of light and look angle can be estimated using the time delay to the
    surface as well as the time delay to the picked FOI. 

    Inputs:
    ----------------
           FOI: array of the same size as the radargram with ones where
                a feature-of-interest has been defined
           SRF: array of the same size as the radargram with ones where
                the surface above the feaure-of-interest has been defined
       rollang: vector of roll angles [rad]
             N: along-track stacking interval [samples]
             B: interferometric baseline [m]
     mb_offset: delay (in fast-time samples) between the start of the
                radargram and signal transmission [samples]
             l: radar wavelength in free space [m]
            dt: fast-time sample interval [s]

    Outputs:
    ----------------
      output is an estimate of the mean interferometric phase of the FOI as
             if it were off-nadir surface clutter
    '''
    out = []
    # find indices where surface and feature-of-interest are defined
    indFOI = np.argwhere(FOI == 1)
    indSRF = np.argwhere(SRF == 1)
    indx = np.intersect1d(indFOI[:, 1], indSRF[:, 1])

    # extract the sample to the surface and FOI for each common range line
    # and subtract the main-band offset
    sampl = np.zeros((2, len(indx)), dtype=float)
    for ii in range(len(indx)):
        sampl[0, ii] = np.argwhere(SRF[:, indx[ii]] == 1) - mb_offset
        sampl[1, ii] = np.argwhere(FOI[:, indx[ii]] == 1) - mb_offset

    # convert sample numbers to one-way times
    times = np.divide(np.multiply(sampl, dt), 2)
    
    # convert one-way times to look angles
    rollang = rollang[(np.arange(np.floor(N / 2) + 1, len(rollang), N) - 1).astype(int)]
    rollang = rollang[indx]
    thetal = np.arccos(np.divide(times[0, :], times[1, :]))

    # calculate interferometric phase
    iphase = -1 * np.multiply(np.divide((2 * np.pi * B), l), np.sin(thetal - rollang))
    #iphase = -1 * np.multiply(np.divide((4 * np.pi * B), l), np.sin(thetal))

    #plt.figure()
    #plt.subplot(211); plt.plot(np.rad2deg(thetal)); plt.title('look angle')
    #plt.subplot(212); plt.plot(np.rad2deg(iphase)); plt.title('interferometric phase')
    #plt.show()

    return iphase


def main():
    logging.basicConfig(level=logging.DEBUG)

    test_raw_bxds_load(); sys.exit(0)

    test_interpolate()
    test_load_pik1()
    test_load_S2_bxds()
    test_load_marfa()
    test_load_power_image()
    test_denoise_and_dechirp()


if __name__ == "__main__":
    import glob
    main()
