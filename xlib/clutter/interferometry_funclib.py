#!/usr/bin/env python3
__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu; Gregory Ng, ngg@ig.utexas.edu']
__version__ = '0.2'
__history__ = {
    '0.1':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'library of functions required for interferometry'},
    '0.2':
        {'date': 'February 01, 2020',
         'author': 'Gregory Ng, UTIG',
         'info': 'Optimize algorithms, esp coregistration'},
}

import sys
import math
import pytest
import logging
import os
import csv
import argparse
import glob
#from tkinter import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy import ndimage
from scipy import integrate
from scipy import signal
from scipy.signal import detrend, correlate
import scipy.special

import interface_picker as ip
from parse_channels import parse_channels
import filter_ra
import peakint

def load_marfa(line, channel, pth='./Test Data/MARFA/', nsamp=3200, trim=None):
    '''
    function for loading MARFA data (magnitude and phase) for a specific line

    Inputs:
    ------------
          line: string specifying the line of interest
       channel: string specifying MARFA channel
           pth: path to line of interest
         nsamp: number of fast-time samples
          trim: trim[0]:trim[1] is fast time trimming, trim[2]:trim[3] is slow time trimming

    Outputs:
    ------------
        outputs are arrays of the magnitude and phase
    '''

    # set path to the magnitude and phase files
    mag_pth = os.path.join(pth, line, 'M1D' + channel)
    phs_pth = os.path.join(pth, line, 'P1D' + channel)

    # load and organize the magnitude file
    mag = load_and_trim(mag_pth, trim, nsamp)

    # load and organize the phase file
    phs = load_and_trim(phs_pth, trim, nsamp)

    return mag / 10000, phs / 2**16

def load_and_trim(infile, trim, nsamp=3200):
    logging.debug("Loading " + infile)
    fbytes = os.path.getsize(infile)
    fsamples = fbytes // 4
    ncols = fsamples // nsamp
    fullbytes = ncols * nsamp * 4
    if fullbytes != fbytes: # pragma: no cover
        msg = "{:s} has extra bytes -- {:d} traces" \
              " + {:d} extra bytes, {:d} bytes total".format(infile, ncols, fbytes - fullbytes, fbytes)
        logging.warning(msg)

    data = np.fromfile(infile, dtype='>i4', count=ncols*nsamp)
    data = np.reshape(data, (ncols, nsamp))

    if trim is not None: # trim before transpose
        data = data[trim[2]:trim[3], trim[0]:trim[1]]
    return data.T

def test_load_marfa():
    line = 'DEV2/JKB2t/Y81a'
    path = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC'
    mag0, phs0 = load_marfa(line, '1', pth=path)

    assert mag0.shape == phs0.shape

    # Test a file that has a bad length
    line = 'ICP10/JKB2u/F01T01a'
    path = '/disk/kea/WAIS/targ/xtra/ICP10/FOC/Best_Versions/S4_FOC'
    mag, phs = load_marfa(line, '1', pth=path)
    return mag0, phs0

def test_stacked_power_image(mag, phs):
    logging.info("test_stacked_power_image()")
    assert mag.shape == phs.shape
    cmpA = convert_to_complex(mag, phs)
    cmpB = convert_to_complex(mag, -phs)
    rollphase = np.random.normal(loc=0, scale=0.1, size=(cmpA.shape[1],))
    for fresnel_stack in (1, 4, 15):
        for method in ('averaged', 'summed'):
            _ = stacked_power_image(mag, phs, mag, phs, fresnel_stack, method)

            _ = stacked_correlation_map(cmpA, cmpB, fresnel_stack)
            _ = stacked_interferogram(cmpA, cmpB, fresnel_stack, rollphase)



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
    else: # pragma: no cover
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
            load_pik1(line, channel, pth=pth, IQ=IQ)

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
          trim: limits used to trim the full profile. Note that only slow time axis is trimmed in this function.
       fresnel: Fresnel zone length in units of trace spacings
          mode: mode to use when stacking range lines to Fresenl zone interval
           pth: path the relevant MARFA files

    Outputs:
    ------------
        output: power image for the rangeline of interest. First axis is fast time; second axis is slow time.
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
        scalef = 20 if pwr_flag else 10
        magnitude = 10 ** (magnitude / scalef)



    # calculate the complex values
    cmp = magnitude * np.exp(1j * phase)

    return cmp

# TODO: show that convert_to_magphs is invers of convert_to_complex
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
        scale = 20 if pwr_flag else 10
        magnitude = scale * np.log10(np.abs(cmp))

    # calculate the phases
    phase = np.angle(cmp)

    return magnitude, phase

def stack(data, fresnel, datatype=float):
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
    col = len(indices)
    if data.shape[1] - indices[-1] < np.floor(fresnel / 2):
        col -= 1

    # pre-define the output
    output = np.zeros((np.size(data, axis=0), col), dtype=datatype)


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
        fresnel -= 1

    # calculate correlation map and average (multi-look) if desired
    if fresnel != 1:

        #top = np.divide(stack(np.multiply(cmpA, np.conj(cmpB)), fresnel, datatype='complex'), fresnel)
        #bottomA = np.divide(stack(np.square(np.abs(cmpA)), fresnel), fresnel)
        #bottomB = np.divide(stack(np.square(np.abs(cmpB)), fresnel), fresnel)

        # setup bounds for each multilook window (output range line in middle)
        indices = np.arange(np.floor(fresnel / 2) + 1, cmpA.shape[1], fresnel) - 1
        if cmpA.shape[1] - indices[-1] < np.floor(fresnel / 2):
            col = len(indices) - 1
        else:
            col = len(indices)
        # predefine the correlation map
        corrmap = np.zeros((np.size(cmpA, axis=0), col), dtype=float)
        # calculate correlation map

        # Correlation with windowed moving average
        # mean(A*conj(B)) / mean(A)*mean(B)
        for ii in range(col):
            num = np.floor((fresnel / 2) / az_step)
            val = np.multiply(az_step, np.arange(-num, num + 0.1))
            val = (indices[ii] + val).astype(int)
            ttop = np.multiply(cmpA[:, val], np.conj(cmpB[:, val]))
            tbot_a = np.square(np.abs(cmpA[:, val]))
            tbot_b = np.square(np.abs(cmpB[:, val]))
            if False:
                for jj in range(len(cmpA)):
                    ymax = min(jj + n, len(cmpA))
                    #S1 = cmpA[jj:ymax, val]
                    #S2 = cmpB[jj:ymax, val]
                    #top = np.mean(np.multiply(S1, np.conj(S2)))
                    #bottomA = np.mean(np.square(np.abs(S1)))
                    #bottomB = np.mean(np.square(np.abs(S2)))
                    top = np.mean(ttop[jj:ymax])
                    bottomA = np.mean(tbot_a[jj:ymax])
                    bottomB = np.mean(tbot_b[jj:ymax])
                    bottom = np.sqrt(np.multiply(bottomA, bottomB))
                    corrmap[jj, ii] = np.abs(np.divide(top, bottom))
            else:
                # running mean method of windowed average
                ttopm = np.empty((ttop.shape[0] - 1, len(val)), dtype=complex)
                tbot_am = np.empty((tbot_a.shape[0] - 1, len(val)))
                tbot_bm = np.empty((tbot_b.shape[0] - 1, len(val)))
                for jj in range(len(val)):
                    ttopm[:, jj] = running_mean(ttop[:, jj], n)
                    tbot_am[:, jj] = running_mean(tbot_a[:, jj], n)
                    tbot_bm[:, jj] = running_mean(tbot_b[:, jj], n)

                bottom = np.sqrt(np.multiply(tbot_am, tbot_bm))
                corrmap[0:-1, ii] = np.mean(np.abs(np.divide(ttopm, bottom)), axis=1)
                del ttopm, tbot_am, tbot_bm, bottom

    else: # fresnel == 1
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
        fresnel -= 1

    # calculate interferogram
    if fresnel <= 1:
        output = np.angle(np.multiply(cmpA, np.conj(cmpB)))
        if roll:
            for ii in range(np.size(output, axis=1)):
                output[:, ii] = output[:, ii] + rollphase[ii]
    else:
        # setup bounds for each multilook window (output range line in middle)
        indices = np.arange(np.floor(fresnel / 2) + 1, np.size(cmpA, axis=1), fresnel) - 1
        if cmpA.shape[1] - indices[-1] < np.floor(fresnel / 2):
            col = len(indices) - 1
        else:
            col = len(indices)
        # predefine the interferogram
        output = np.zeros((np.size(cmpA, axis=0), col), dtype=float)
        # calculate interferogram
        corr = np.multiply(cmpA, np.conj(cmpB))
        # Calculate windowed average
        for ii in range(col):
            num = np.floor((fresnel / 2) / az_step)
            val = np.multiply(az_step, np.arange(-num, num + 0.1))
            val = (indices[ii] + val).astype(int)

            # running mean method
            temp = np.empty((corr.shape[0] - 1, len(val)), dtype=complex)
            for jj in range(len(val)):
                temp[:, jj] = running_mean(corr[:, val[jj]], n)
            output[0:-1, ii] = np.angle(np.mean(temp, axis=1))

            if roll:
                output[:, ii] += np.mean(rollphase[val])

    return output

def running_mean(x, N):
    """ This one is not centered """
    cumsum = np.cumsum(np.insert(x, 0, 0), dtype=x.dtype)
    return (cumsum[N:] - cumsum[:-N]) / float(N)



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



def sinc_interpolate(data, orig_sample_interval, upsample_factor):
    '''
    function for interpolating a vector using a sinc interpolation kernel

    Inputs:
    ------------
                      data: input data vector
      orig_sample_interval: sampling interval of the input data
          upsample_factor: factor used to subsample the input data at

    Outputs:
    ------------
        output is the interpolated data vector
    '''

    # GN -- this is a lot slower than frequency interpolation because of generating the
    # gigantic sinc matrix

    # define sample vectors
    new_sample_interval = orig_sample_interval / upsample_factor
    orig_t = np.linspace(0, (len(data) - 1) * orig_sample_interval, len(data))
    nsamples_new = len(data) * upsample_factor
    new_t = np.linspace(0, (nsamples_new - 1) * new_sample_interval, nsamples_new)

    # perform the interpolation
    # Generate shifted x values to input to sinc
    sincM = np.tile(new_t, (len(orig_t), 1)) - np.tile(orig_t[:, np.newaxis], (1, len(new_t)))
    output = np.dot(data, np.sinc(sincM / orig_sample_interval))

    return output


def frequency_shift(data, upsample_factor, offset):
    """
    Shift a vector by a sub-sample amount, by upsampling by upsample_factor and taking the offset
    function for interpolating a vector by padding the data in the frequency
    domain.

    Inputs:
    ------------
                 data: complex-valued range line
      upsample_factor: factor by which the user wants to subsample the data by
               offset: Shift offset in units of samples

    Outputs:
    ------------
       shifted data vector
    """
    subsamp_b = frequency_interpolate(data, upsample_factor)
    subsamp_b = np.roll(subsamp_b, int(offset)) # TODO: can we do this without rolling?
    return subsamp_b[np.arange(0, len(subsamp_b), upsample_factor)] # subsamp_b[offset, len(subsamp_b), upsample_factor]

def frequency_shift2(data, foffset):
    """
    Shift a vector by a sub-sample amount, by upsampling by upsample_factor and taking the offset
    function for interpolating a vector by padding the data in the frequency
    domain.

    Inputs:
    ------------
                 data: complex-valued range line
      upsample_factor: factor by which the user wants to subsample the data by
               offset: Shift offset (delay) in units of samples

    Outputs:
    ------------
       shifted data vector
    """
    # Calculate the kernel for shifted interpolation (a shifted sinc), or rect in frequency domain
    n1 = len(data)
    m1 = n1 // 2
    wn = 2*np.pi*(-foffset/n1)
    t = np.arange(m1 - n1, m1)
    fh = np.fft.fftshift(np.exp(wn*1j * t))
    fh[m1] = 0 # lowpass filter

    return np.fft.ifft(np.fft.fft(data) * fh)


def fshiftfunc(x, offset=0.0):
    # Function to use for testing frequency shift
    return 10*np.sin(2*np.pi*x + offset)

def test_frequency_shift(plot=False):
    for dx in (0.2, 0.1, 0.05):
        x1 = np.arange(0, 2.0, dx)
        y1 = fshiftfunc(x1)

        for upsamp in np.arange(2, 10):
            for offset in np.arange(0, upsamp):
                # Compute the actual functional value with a shift
                y2 = fshiftfunc(x1 - dx*offset/upsamp)  # calculate upsampled sine directly
                y3 = frequency_shift(y1, upsamp, offset) # calculate by old method
                y4 = frequency_shift2(y1, offset/upsamp) # calculate by new method

                try:
                    assert (np.abs(y2 - y3) < 1e-6).all()
                    assert (np.abs(y2 - y4) < 1e-6).all()
                except AssertionError as e: #pragma: no cover
                    logging.warning("Assert error with upsamp={:d} offset={:f}".format(upsamp, offset))
                    print(e)
                    plot = True

                if plot or offset == 0 and upsamp == 2: # calculate at least once, for coverage
                    plt.clf()
                    plt.subplot(3,1,1)
                    plt.plot(x1, y1, label='orig', marker='o', linewidth=0)
                    plt.plot(x1, np.real(y2), label='y2real')
                    plt.plot(x1, np.real(y3), label='y3real', marker='x', linewidth=0)
                    plt.plot(x1, np.real(y4), label='y4real', marker='v', linewidth=0)
                    plt.legend()
                    plt.grid(True)
                    plt.title('{:0.0f} / {:0.0f}'.format(offset, upsamp))
                    plt.subplot(3,1,2);
                    plt.plot(x1, np.zeros_like(x1), label='orig', marker='.', linewidth=0)
                    plt.plot(x1, np.imag(y2), label='y2imag')
                    plt.plot(x1, np.imag(y3), label='y3imag', marker='x', linewidth=0)
                    plt.plot(x1, np.imag(y4), label='y4imag', marker='v', linewidth=0)
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(3, 1, 3)
                    plt.plot(x1, np.real(y3 - y2), label='y3 - y2', marker='x', linewidth=1, color='g')
                    plt.plot(x1, np.real(y4 - y2), label='y4 - y2', marker='v', linewidth=1, color='r')
                    plt.title('{:0.0f} / {:0.0f} - Real Residual'.format(offset, upsamp))
                    plt.legend()
                    plt.grid(True)

                if plot: # pragma: no cover
                    plt.show()



def frequency_interpolate(data, upsample_factor):
    '''
    function for interpolating a vector by padding the data in the frequency
    domain.

    Inputs:
    ------------
                  data: complex-valued range line
      upsample_factor: factor by which the user wants to subsample the data by

    Outputs:
    ------------
       interpolated data vector
    '''
    fft = np.fft.fft(data, norm='ortho')
    fft_shift = np.fft.fftshift(fft)
    x = int((len(data) * upsample_factor - len(data)) / 2)
    fft_int = np.pad(fft_shift, (x, x), 'constant', constant_values=(0, 0))
    fft_int_shift = np.fft.ifftshift(fft_int)
    # Without the np.sqrt(upsample_factor), this is an energy-preserving function.
    # But we want to preserve value, not total signal energy
    return np.sqrt(upsample_factor)*np.fft.ifft(fft_int_shift, norm='ortho')

def test_interpolate(bplot=False):
    """ Test equivalence of interpolation algorithms, and run with
    a variety of input sizes
    Test that average value and amplitude of signals mathes
    """

    meanval = 0.0

    for repeatsize in (128, 255, 77):
        logging.debug("repeatsize = {:d}".format(repeatsize))
        # Series of step functions
        sig = np.repeat([0.5, 0., 1., 1., 0., 1., 0., 0., 1., 0.5], repeatsize)
        sig -= sig.mean() + meanval
        x = np.linspace(0, 100.0, len(sig))
        # noisy signal
        sig_noise = sig + 1e-3 * np.random.randn(len(sig))
        osi = 1 / 50e6 # original signal interval
        for ifactor in (2, 5, 10, 13, 21):
            x2 = np.linspace(0, max(x), len(sig_noise)*ifactor) # get interpolated indices
            sig_interp0 = np.interp(x2, x, sig_noise) # linear interpolation
            sig_interp1 = np.real(sinc_interpolate(sig_noise, osi, ifactor))
            sig_interp2 = np.real(frequency_interpolate(sig_noise, ifactor))

            rms1 = np.sqrt(np.square(abs(sig_interp0 - sig_interp1)).mean())
            rms2 = np.sqrt(np.square(abs(sig_interp0 - sig_interp2)).mean())
            rms3 = np.sqrt(np.square(abs(sig_interp1 - sig_interp2)).mean())
            logging.info("interpolate: ifactor={:0.1f} RMS(lin-sinc)={:0.4g}"
                         " RMS(lin-freq)={:0.4g} RMS(sinc-freq)={:0.4g}".format(ifactor, rms1, rms2, rms3))


            statso = np.array([np.mean(sig_noise), np.std(sig_noise)])
            stats0 = np.array([np.mean(sig_interp0), np.std(sig_interp0)])
            stats1 = np.array([np.mean(sig_interp1), np.std(sig_interp1)])
            stats2 = np.array([np.mean(sig_interp2), np.std(sig_interp2)])

            try:
                #assert (np.abs(stats0 - stats1) < 1e-3).all()
                assert (np.abs(stats1 - stats2) < 1e-2).all()
                #assert (np.abs(statso - stats2) < 1e-3).all()
            except AssertionError: # pragma: no cover
                logging.error("statso = " + str(statso))
                logging.error("stats0 = " + str(stats0))
                logging.error("stats1 = " + str(stats1))
                logging.error("stats2 = " + str(stats2))
                raise

            try:
                assert rms3 < 5e-4
            except AssertionError: # pragma: no cover
                logging.error("repeatsize={:d} ifactor={:d} RMS interpolation "
                              "difference: {:f} (limit {:f})".format(repeatsize, ifactor, rms3, 5e-4))
                bplot = True

            if bplot: #pragma: no cover
                plt.subplot(211)
                plt.plot(x2, sig_interp0, x2, sig_interp1, x2, sig_interp2)
                plt.legend(['linear', 'sinc', 'frequency'])
                plt.subplot(212)
                plt.plot(abs(sig_interp1-sig_interp2))
                plt.title("Sinc vs Frequency Interpolation error")
                plt.show()

def norm_radargrams(a, b):
    """ subtract mean and divide standard deviation for two radargrams """
    ua, ub = np.mean(a), np.mean(b)
    sa, sb = np.std(a), np.std(b)
    assert sa >= 0 and sb >= 0
    a1 = a - ua
    b1 = b - ub

    if sa >= 0:
        a1 /= sa
    if sb >= 0:
        b1 /= sb

    return a1, b1

def coregister(cmp_a, cmp_b, orig_sample_interval, upsample_factor, shift, method=0x0):
    '''
    function for co-registering complex-valued fast time records.
    This function finds the offset, but does not shift any input records.
    from two radargrams as required to perform interferometry.

    Meets requirements outlined in Castelletti et al. (2018), of having resolution at
    least 1/10 of sample

    Inputs:
    -------------
                     cmp_a: complex-valued input A radargram
                     cmp_b: complex-valued input B radargram
      orig_sample_interval: sampling interval of the input data (in seconds)
           upsample_factor: Upsample radargrams by this factor, using sinc interpolation,
                            before performing correlation.
                     shift: Max shift to consider
                    method: feature flags for coregistration algorithm
                            default=0x0

                            bit 0 - zero-normalized cross correlation (ZNCC)
                            bit 1 - subsample peak interpolation
                            bit 2 - constant quality adaptive windowing (CQAW)

                            ZNCC:
                            subtract out the mean and divide by std deviation of traces
                            before performing cross correlation

                            subsample peak interpolation:
                            Use qint to interpolate peak to a sub-sample location

                            CQAW:
                             evalutes the quality of the coregistration, determined
                            by the ratio of the peak correlation to the average correlation.
                            If this ratio is below a threshold of 4.0, it incorporates the two
                            neighboring traces until this value is above the threshold,
                            up to a limit of 16 iterations.

    Outputs:
    -------------
      shift_array: Array (same shape as cmp_a.shape[1]) containing mumber of samples
                   (possibly non-integer) to offset each fast time record in cmp_a, to those in cmp_b, to
                       have data best align between cmp_a and cmp_b.
                   Note that this array expresses the shift in units of the upsampled samples.

       qual_array: Array of quality factor of match. Higher is better match.
    '''

    shift_array = np.zeros(cmp_a.shape[1])
    qual_array = np.zeros(cmp_a.shape[1])
    qual_array2 = None #np.zeros(cmp_a.shape[1])
    logging.debug("coregister: upsample_factor={:f} shift={:d}".format(upsample_factor, shift))
    #---------------------------------
    # adaptive windowing parameters
    awin_qual_threshold = 4.0    # adaptive windowing quality threshold
    awin_maxwin = 16             # maximum window size


    # if bit 2 is set, perform adaptive windowing
    maxwin = awin_maxwin if (method & 0x4) else 1

    for ii in range(cmp_a.shape[1]):
        data_a, data_b = np.copy(cmp_a[:, ii]), np.copy(cmp_b[:, ii])

        stackcount = 1 # number of stacks done for adaptive windowing
        for jj in range(maxwin):
            if jj > 0:
                # If not the first round, add neighboring traces
                for kk in (ii - jj, ii + jj):
                    if 0 <= kk < cmp_a.shape[1]:
                        data_a += cmp_a[:, kk]
                        data_b += cmp_b[:, kk]
                        stackcount += 1

            subsampA = frequency_interpolate(data_a, upsample_factor)
            subsampB = frequency_interpolate(data_b, upsample_factor)

            if (method & 0x1) or jj > 0:
                # perform normalization for adaptive windowing,
                # or if requested with feature bit flags.
                subsampA, subsampB = norm_radargrams(subsampA, subsampB)

            # co-register and shift
            rho = np.abs(scipy.signal.correlate(subsampA, subsampB[shift:-(shift-1)], mode='valid'))

            if method & 0x2:
                # if bit 1 is set, perform subsample interpolation
                # interpolate maximum between subsample points
                x = np.argmax(rho)
                p, y, _ = peakint.qint(rho, x)
                to_shift = x + p - shift
            else:
                to_shift = np.argmax(rho) - shift
                y = np.max(rho)

            # if quality is good enough, quit adaptive window loop
            q = y / np.mean(rho)
            if q >= awin_qual_threshold:
                break

        # end adaptive window loop
        qual_array[ii] = q

        if qual_array2 is None:
            qual_array2 = np.empty((cmp_a.shape[1], len(rho)))
        qual_array2[ii] = rho
        shift_array[ii] = to_shift

    return shift_array, qual_array, qual_array2


def coregistration(cmpA, cmpB, orig_sample_interval, upsample_factor, shift=300, method=0):
    '''
    function for sub-sampling and coregistering complex-valued range lines
    from two radargrams as required to perform interferometry. Follows the
    steps outlines in Castelletti et al. (2018)

    TODO: coregistration should probably occur using either just the first half of the image,
    or it should be done in a log-scaled domain (is it already log-scaled?).  Otherwise the only
    effective contribution to the signal will be the echoes near the surface.

    Inputs:
    -------------
                      cmpA: complex-valued input A radargram
                      cmpB: complex-valued input B radargram (seconds)
      orig_sample_interval: sampling interval of the input data
          upsample_factor: factor used modify the original fast-time sampling
                            interval
                     shift:
                    method: coregistration algorithm method described in coregister() function

    Outputs:
    -------------
       coregA: coregistered complex-valued A radargram
       coregB: coregistered complex-valued B radargram
    '''

    shift_array, qual_array, qual_array2 = coregister(cmpA, cmpB, orig_sample_interval, upsample_factor, shift, method=method)

    # define the output and shift data.
    coregB = np.empty_like(cmpB, dtype=complex)
    for ii in range(cmpB.shape[1]):
        coregB[:, ii] = frequency_shift2(cmpB[:, ii], shift_array[ii] / upsample_factor)

    # Convert from upsampled units to original shift units
    shift_array /= upsample_factor

    logging.info("shift_array: mean={:0.3f}, median={:0.1f} std={:0.3f} min={:0.1f} max={:0.1f}".format(
                np.mean(shift_array), np.median(shift_array), np.std(shift_array),
                np.min(shift_array), np.max(shift_array)))
    logging.info(" qual_array: mean={:0.3f}, median={:0.1f} std={:0.3f} min={:0.1f} max={:0.1f}".format(
                np.mean(qual_array), np.median(qual_array), np.std(qual_array),
                np.min(qual_array), np.max(qual_array)))
    return cmpA, coregB, shift_array, qual_array, qual_array2


def test_coregistration():
    chanlist = {'low': ('5','7')} # not all high gain data exists, 'high': ('6','8')}
    # Load the focused SLC 1m port and starboard radargrams

    #"/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1a/"
    line = "NAQLK/JKB2j/ZY1a/"
    path = "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S4_FOC"
    trim = [0, 1000, 0, 12000]
    for name, chans in chanlist.items():
        logging.info("Coregistration " + name)
        cmpa = convert_to_complex(*load_marfa(line, chans[0], pth=path, trim=trim))
        cmpb = convert_to_complex(*load_marfa(line, chans[1], pth=path, trim=trim))
        logging.info("Coregistration done loading data")
        # orig_sample_interval is unused; TODO: remove
        for ifactor in range(1, 2, 4):
            for method in (0, 7):
                _ = coregistration(cmpa, cmpb, orig_sample_interval=None, upsample_factor=ifactor, method=method)


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
    fn = os.path.join(s1_path, 'ztim_xyhd')
    logging.debug("Reading " + fn)
    S1_ztim = []
    S1_dist = []
    with open(fn, 'r') as fin:
        for rec in csv.reader(fin, delimiter=' '):
            S1_ztim.append(float(rec[2].replace(')', ''))) # seconds
            S1_dist.append(float(rec[6])) # distance (meters)

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
    assert I.shape == Q.shape
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

    # load the bxds datasets
    chanlist = {'low': ['5', '7'], 'high': ['6', '8']}
    chans = chanlist[gain]

    # TODO: reduce max mem usage by streaming

    bxdsA = raw_bxds_load(raw_path, geo_path, chans[0], sigwin)
    bxdsB = raw_bxds_load(raw_path, geo_path, chans[1], sigwin)

    assert bxdsA.shape == bxdsB.shape

    # trim of the range lines if desired.
    # TODO: some of the downstream functions aren't aware of this.
    #logging.debug("bxds original shape="  + str(bxdsA.shape))
    #if sigwin[0] >= 0 and sigwin[1] > 0:
    #    bxdsA = bxdsA[sigwin[0]:sigwin[1], :]
    #    bxdsB = bxdsB[sigwin[0]:sigwin[1], :]

    # Output shape needs to be the same as the input
    output_samples = bxdsA.shape[0]

    # trimming now occurs in raw_bxds_load
    #if sigwin[3] != 0:
    #    bxdsA = bxdsA[:, sigwin[2]:sigwin[3]]
    #    bxdsB = bxdsB[:, sigwin[2]:sigwin[3]]
    #logging.debug("bxds trimmed  shape="  + str(bxdsA.shape))

    # prepare the reference chirp
    hamm = hamming(output_samples)
    refchirp = get_ref_chirp(chirp_path, bandpass=bp, nsamp=output_samples)
    # TODO: do we need to make the chirp complex?
    #refchirp *= hamming
    # Since we did an FFT with a detrend in it in filter_ra, we can skip detrending.
    detrend = False #'constant'
    # prepare the outputs
    dechirpA = np.empty_like(bxdsA, dtype=complex)
    # dechirp
    for ii in range(np.size(bxdsA, axis=1)):
        dechirpA[:, ii] = dechirp(bxdsA[:, ii], refchirp, do_cinterp, detrend=detrend)
    del bxdsA

    dechirpB = np.empty_like(bxdsB, dtype=complex)
    for ii in range(np.size(bxdsB, axis=1)):
        dechirpB[:, ii] = dechirp(bxdsB[:, ii], refchirp, do_cinterp, detrend=detrend)
    del bxdsB

    return dechirpA, dechirpB

def test_denoise_and_dechirp():
    gain = 'low'
    trim_default = [0, 1000, 0, 0]

    # Test with a RADnh3 line
    snms = {'SRH1': 'RADnh5'}

    inputs = [
        {'prj': 'SRH1', 'line': 'DEV2/JKB2t/Y81a', 'trim': trim_default },
        {'prj': 'GOG3', 'line': 'NAQLK/JKB2j/ZY1b', 'trim': [0, 1000, 0, 12000] },
        # Trimming at trim[2] == 15000 isn't supported yet
        #{'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01a/', 'trim': [0, 1000, 15000, 27294] },
        {'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01a/', 'trim': [0, 1000, 0, 27294] },
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
    method = 'summed'

    for line in ('DEV2/JKB2t/Y81a/', 'DEV2/JKB2t/Y81a'): # allow either
        img = load_power_image(line, '1', trim, fresnel_stack, method, pth=path)

    line = 'DEV2/JKB2t/Y81a'
    for fresnel_stack in (1, 2, 15):
        for method in ('averaged','summed'):
            img = load_power_image(line, '1', trim, fresnel_stack, method, pth=path)

def dechirp(trace, refchirp, do_cinterp, output_samples=3200, detrend='linear'):
    '''
    Range line dechirp processor

    Inputs:
    -----------
            trace: radar range line
         refchirp: reference chirp
       do_cinterp: do frequency domain interpolation (coherent noise
                   removal) for HiCARS/MARFA data
          detrend: parameter to pass to detrend. If False, don't detrend

    Outputs:
    -----------
      dechirped range line
    '''

    # find peak energy below blanking samples
    shifter = int(np.median(np.argmax(trace)))
    trace = np.roll(trace, -shifter)

    if detrend:
        DFT = np.fft.fft(signal.detrend(trace, type=detrend))
    else:
        DFT = np.fft.fft(trace)

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
            Cxy, f = signal.coherence(np.angle(reference), np.angle(data[:, ii]), fs,
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

            C[ii] = rolls[np.argmax(R)]
            # C[ii] = rolls[int(np.argwhere(R == np.max(R)))]
    elif method == 'xcorr2': # faster cross correlation
        C = np.empty((data.shape[1],) )
        N = data.shape[0] // 2
        for ii in range(data.shape[1]):
            R = scipy.correlate(data[:, ii], reference, mode='same')
            # TODO: interpolate for peak
            C[ii] = np.argmax(R) - N


    else: # pragma: no cover
        raise ValueError('Unrecognized method {:s}'.format(method))

    return C

def phase_stability_adjustment(data, stability):
    '''
    Apply chirp stability adjustment results to actual data.
    simply a roll of the data by some number of integer samples.

    Inputs:
    ---------------
          data: complex-valued radar data
     stability: shifts required to achieve chirp stability

    Outputs:
    ---------------
       chirp stability corrected complex-valued radar data
    '''

    out = np.zeros(data.shape, dtype=complex)
    for ii in range(data.shape[1]):
        out[:, ii] = np.roll(data[:, ii], int(stability[ii]))

    return out



def raw_bxds_load(rad_path, geo_path, channel, trim, DX=1, MS=3200, NR=1000, NRr=100):
    '''
    function to load raw MARFA bxds with the loopback chirp preserved.

    Inputs:
    ----------------
      rad_path: Path to the raw radar files
      geo_path: Path to the raw geometry files required to perform interpolation
       channel: desired MARFA channel to load
          trim: vector of values with trim[2]:trim[3] is slow time range lines of interest.
                None to use entire range. trim[0]:trim[1] is fast time samples of interest
            DX: alongtrack range line spacing after interpolation
            MS: number of fast-time samples in the output
            NR: block size to load data
           NRr: overlap between blocks(?)

    Outputs:
    ----------------
       signalout is an array of raw MARFA data for the line and channel in question
    '''


    rad_name = os.path.join(rad_path, 'bxds')
    undersamp = True

    combined = True
    channel = int(channel)
    snm = None #'RADnh5' # either RADnh3 or RADnh5

    signalout = filter_ra.filter_ra(rad_name, geo_path, DX, MS, NR, NRr, channel, snm=snm,
              undersamp=undersamp, combined=combined, blank=False, trim=trim)

    return signalout





def test_raw_bxds_load():

    # /disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/AGAE/JKB2i/X5Aa/Xo
    testcases = [
        {
        'raw_path': "/disk/kea/WAIS/orig/xlob/DEV2/JKB2t/Y81a/RADnh5",
        'geo_path': "/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV2/JKB2t/Y81a",
        'sigwin': [0, 1000, 0, 0],
        #}, { # 1 tear
        #'raw_path': "/disk/kea/WAIS/orig/xlob/NAQLK/JKB2j/ZY1a/RADnh3/",
        #'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1a/",
        #'sigwin': [0, 1000, 0, 0]
        }, { # 1 tear
        'raw_path': "/disk/kea/WAIS/orig/xlob/NAQLK/JKB2j/ZY1b/RADnh3/",
        'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1b/",
        'sigwin': [0, 1000, 0, 12000]
        #}, { # 0 tears -- no metadata.
        #'raw_path': "/disk/kea/WAIS/orig/xlob/CLEM/JKB2j/COL01a",
        #'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/CLEM/JKB2j/COL01a",
        #'sigwin': [0, 1000, 0, 0]
        #}, { # 49 tears and a short read at the end
        ##/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV/JKB2t/Y49a/NRt
        #'raw_path': "/disk/kea/WAIS/orig/xlob/DEV/JKB2t/Y49a/RADnh5",
        #'geo_path': "/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV/JKB2t/Y49a",
        #'sigwin': [0, 1000, 0, 0]

        },
    ]
    chan = '5'
    #testcases = (testcases[0],)
    for i, rec in enumerate(testcases):
        logging.info("[{:d} of {:d}] Testing raw_bxds_load with {:s}".format(i+1, len(testcases), rec['raw_path']))

        logging.debug("raw_bxds_load()")
        bxds1 = raw_bxds_load(rec['raw_path'], rec['geo_path'], chan, rec['sigwin'])

        """
        logging.debug("raw_bxds_load2()")
        bxds2 = raw_bxds_load2(rec['raw_path'], rec['geo_path'], chan, rec['sigwin'])

        assert bxds2.shape == bxds1.shape
        rmse = np.sqrt(np.square(abs(bxds2 - bxds1)).mean())
        logging.debug("RMSE(raw_bxds_load - raw_bxds_load2) = {:0.3g}".format(rmse))
        assert rmse < 1e-9
        """


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
    return np.divide(tempA, tempB)

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
    # TODO: these two loops (i and j) should probably be reversed to reduce calls to np.roll
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

    # TODO: rename as normalize_interferogram

    Inputs:
    ----------------
      interferogram: stacked interferogram
            surface: picked surface for interferogram normalization

    Outputs:
    ----------------
      output is the normalized interferogram
    '''

    # make sure we aren't going to have an error.
    assert interferogram.shape == surface.shape

    # ensure the picked surface covers the breadth of the interferogram
    test = np.nansum(surface, axis=0)
    x = np.argwhere(test == 1)[:, 0]
    y = np.zeros((len(x)), dtype=int)
    for ii in range(len(x)):
        y[ii] = np.argwhere(surface[:, x[ii]] == 1)
    xall = np.arange(0, surface.shape[1])
    yall = np.interp(xall, x, y)
    surface2 = np.full(surface.shape, np.nan, dtype=float)
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
           image: power image.  First axis is fast time, second axis is slow time.
             FOI: array of the same size as the power image with ones where
                  a feature-of-interest has been defined
     pick_method: string indentifying how individual pixels corresponding
                  to the surface are picked during surface definition

    Outputs:
    ----------------
      out is an array of the same size as the power image with ones
          where the surface overlying the feature-of-interest has been
          defined
    '''

    # extract the along-track sample bounds in the picked FOI
    # indA is the lower record index where the FOI was picked.
    inds = np.argwhere(FOI == 1)[:, 1]
    indA = min(inds)
    indB = max(inds)
    image2 = image[:, indA:indB]
    logging.info("Picking surface from slow time record (:, %d:%d)", indA, indB)

    # pick the surface within the area covered by the FOI
    SRF = ip.picker(np.transpose(image2), snap_to=pick_method)
    SRF = np.transpose(SRF)
    logging.info("surface_pick(): SRF shape = %s", SRF.shape)

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
    tempA = np.full((FOI.shape[0], indA), np.nan)
    tempB = np.full((FOI.shape[0], FOI.shape[1] - indB), np.nan)
    try:
        out = np.concatenate((tempA, SRF, tempB), axis=1)
        assert out.shape == image.shape
    except (ValueError, AssertionError) as _: #usually mismatched dimensions
        logging.warning("tempA.shape=%s", tempA.shape)
        logging.warning("SRF.shape=%s", SRF.shape)
        logging.warning("tempB.shape=%s", tempB.shape)
        raise

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
    parser = argparse.ArgumentParser(description='Interferometry Function Library Test')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose script output')
    parser.add_argument('--plot', action='store_true', help='Show debugging plots')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    magphs = test_load_marfa()
    test_stacked_power_image(*magphs)
    del magphs

    test_load_pik1()
    test_raw_bxds_load()
    test_load_S2_bxds()
    test_load_power_image()
    test_denoise_and_dechirp()
    test_frequency_shift(plot=args.plot)
    test_interpolate(bplot=args.plot)
    test_coregistration()


if __name__ == "__main__":
    main()

