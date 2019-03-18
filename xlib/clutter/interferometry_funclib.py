__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '0.1'
__history__ = {
    '0.1':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'library of functions required for interferometry'}}

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
    import numpy as np

    # set path to the magnitude and phase files
    mag_pth = pth + line + 'M1D' + channel
    phs_pth = pth + line + 'P1D' + channel

    # load and organize the magnitude file
    mag = np.fromfile(mag_pth, dtype='>i4')
    ncol = int(len(mag) / nsamp)
    mag = np.transpose(np.reshape(mag, (ncol, nsamp))) / 10000

    # load and organize the phase file
    phs = np.fromfile(phs_pth, dtype='>i4')
    ncol = int(len(phs) / nsamp)
    phs = np.transpose(np.reshape(phs, (ncol, nsamp))) / 2**16

    return mag, phs

def load_pik1(line, channel, pth='./Test Data/MARFA/', nsamp=3200, IQ='mag'):
    '''
    function for loading MARFA data (magnitude and phase) for a specific line

    Inputs:
    ------------
          line: string specifying the line of interest
       channel: string specifying MARFA channel
           pth: path to line of interest
         nsamp: number of fast-time samples
            IQ: in-phase or quadrature 

    Outputs:
    ------------
        outputs are arrays of the magnitude and phase
    '''
    import numpy as np

    # set path to the magnitude and phase files
    if IQ == 'mag':
        pth = pth + line + 'MagLoResInco' + channel
    elif IQ == 'phs':
        pth = pth + line + 'PhsLoResInco' + channel

    # load and organize the data
    data = np.fromfile(pth, dtype='>i4')
    ncol = int(len(data) / nsamp)
    if IQ == 'mag':
        data = np.transpose(np.reshape(data, (ncol, nsamp))) / 10000
    elif IQ == 'phs':
        data = np.transpose(np.reshape(data, (ncol, nsamp))) / 2**24

    return data

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
       power image for the rangeline of interst
    '''
    import numpy as np

    # load the MARFA dataset
    mag, phs = load_marfa(line, channel, pth)
    mag = mag[trim[0]:trim[1], trim[2]:trim[3]]

    # incoherently stack to desired trace spacing
    if fresnel != 1:
        if mode == 'summed':
            output = stack(mag, fresnel)
        elif mode == 'averaged':
            output = np.divide(stack(mag, fresnel), fresnel)
    else:
        output = mag

    return output
    
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
    import numpy as np
#    import math

    # get magnitudes out of dB if they are
    if mag_dB:
        if pwr_flag:
#            magnitude = math.pow(10, (magnitude / 20))
            magnitude = 10 ** (magnitude / 20)
        else:
#            magnitude = math.pow(10, (magnitude / 10))
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
        arrays of magntiude and phase [rad]
    '''
    import numpy as np

    # calculate magnitudes
    if mag_dB:
        if pwr_flag:
            magnitude = 20 * np.log10(np.abs(cmp))
        else:
            magnitude = 10 * np.log10(np.abs(cmp))

    # calculate the phases
    phase = np.angle(cmp)

    return magnitude, phase

def stack(data, fresnel):
    '''
    stacking data to a new trace posting interval

    Inputs:
    ------------
         data: data to be stacked
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams

    Output:
    ------------
        stacked array
    '''
    import numpy as np

    # incoherently stack to desired trace spacing
    indices = np.arange(np.floor(fresnel / 2) + 1, np.size(data, axis=1), fresnel) - 1
    if np.size(data, axis=1) - indices[-1] < np.floor(fresnel / 2):
        col = len(indices) - 1
    else:
        col = len(indices)
    output = np.zeros((np.size(data, axis=0), col), dtype=float)
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
    import numpy as np

    # make sure the new trace spacing passed is odd
    if fresnel % 2 != 1:
        fresnel = fresnel - 1

    # convert back to complex values
    cmpA = convert_to_complex(magA, phsA)
    cmpB = convert_to_complex(magB, phsB)

    # coherently combine the two antennas
    #comb_cmp = np.add(cmpA, cmpB)
    # -----------------------------------------------------
    comb_cmp = np.multiply(cmpA, np.conj(cmpB))
    # -----------------------------------------------------

    # produce a combined power image
    comb_mag, comb_phs = convert_to_magphs(comb_cmp)

    # incoherently stack to desired trace spacing
    if mode == 'summed':
        output = stack(comb_mag, fresnel)
    elif mode == 'averaged':
        output = np.divide(stack(comb_mag, fresnel), fresnel)

    return output

def stacked_interferogram(magA, phsA, magB, phsB, fresnel, method='Unsmoothed', n=2):
    '''
    producing a phase interferogram at the desired trace posting interval from
    the two antenna datasets

    Inputs:
    ------------
         magA: array of magnitudes [typically dB] for antenna A
         phsA: array of phases [radians] for antenna A
         magB: array of magnitudes [typically dB] for antenna B
         phsB: array of phases [radians] for antenna B
      fresnel: trace spacing according to Fresnel zone
               -- must be given in integers of the trace spacing for the input
                  radargrams
       method: choice to either take the interferogram directly without 2D smoothing
               [as is done in Castelletti et al., (2018)]
            n: fast-time smoother required during implementation of the
               Castelletti et al. (2018) approach

    Output:
    ------------
        real-valued interferogram at the Fresnel zone trace spacing from the
        two antennae
    '''
    import numpy as np
    from scipy import ndimage

    # make sure the new trace spacing passed is odd
    if fresnel % 2 != 1:
        fresnel = fresnel - 1

    cmpA = convert_to_complex(magA, phsA)
    cmpB = convert_to_complex(magB, phsB)
   
    if method == 'Unsmoothed':
        inter = np.angle(np.multiply(cmpA, np.conj(cmpB)), deg=True)
    elif method == 'Smoothed':
        inter = np.zeros((len(cmpA), np.size(cmpA, axis=1)), dtype=float)
        for ii in range(np.size(cmpA, axis=1)):
            xwindow = np.arange(ii - (fresnel - 1) / 2, ii + 1 +(fresnel - 1) / 2, 1)
            if ii < fresnel:
                viable = np.argwhere(xwindow >= 0)
                xwindow = np.transpose(xwindow[viable])
            if np.size(cmpA, axis=1) - ii < fresnel:
                viable = np.argwhere(xwindow < np.size(cmpA, axis=1) - 1)
                xwindow = np.transpose(xwindow[viable])
            for jj in range(len(cmpA)):   
                ywindow = np.arange(jj, jj + n, 1)
                viable = np.argwhere(ywindow <= len(cmpA) - 1)
                ywindow = ywindow[viable] 
                temp = np.mean(np.multiply(cmpA[ywindow.astype(int), xwindow.astype(int)], np.conj(cmpB[ywindow.astype(int), xwindow.astype(int)])))
                inter[jj, ii] = np.angle(temp, deg=True)

    if fresnel != 1:
        output = np.divide(stack(inter, fresnel), fresnel)
    else:
        output = inter
    
    return output

def FOI_interferometric_phase(int_image, FOI):
    '''
    extract the interferometric phase angles along the picked FOI

    Inputs:
    ------------
      int_image: array of interferometric angles
            FOI: array with pick FOI
                 - FOI array should have the same dimensions as int_image with
                   samples related to the FOI marked with ones

    Output:
    ------------
        array of interferometric angles along the FOI
    '''
    import numpy as np

    # extract indices related to the FOI
    indices = np.argwhere(FOI == 1)

    # define the interferometric phase angles related to these indices
    output = int_image[indices[:, 0], indices[:, 1]]
    
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
    import numpy as np
    import scipy.special

    beta = gamma * np.cos(iphi - phi0)
    ghf = scipy.special.hyp2f1(1, N, 1/2, beta**2)
    G1 = scipy.special.gamma(N + 1/2)
    G2 = scipy.special.gamma(1/2)
    G3 = scipy.special.gamma(N)
    f = (((1 - gamma**2)**N) / (2 * np.pi)) * ghf 
    f = f + ((G1 * ((1 - gamma**2)**N) * beta) / (2 * G2 * G3 * (1 - beta**2)**(N + 1/2)))

    return f

def empirical_pdf(fc, B, fresnel, phi_m, FOI, magA, magB, noiseA, noiseB, mode):
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
        phi_m: mean interferometric phase angle of the FOI         
          FOI: picked FOI
         magA: array of magnitudes [typically dB] for antenna A
         magB: array of magnitudes [typically dB] for antenna B
       noiseA: noise floor for antenna A
       noiseB: noise floor for antenna B
         mode: how amplitudes in the power image are represented
               -- 'averaged': average amplitude of the stack         
               -- 'summed': sum of the stack 

    Output:
    ------------
        phi: interferometric phase angles
          f: interferometric phase pdf
      gamma: interferometric correlation
    '''
    import numpy as np

    # incoherently stack the A and B magnitude arrays to the desired spacing
    if fresnel != 1:
        if mode == 'summed':
            magA = stack(magA, fresnel)
            magB = stack(magB, fresnel)
        elif mode == 'averaged':
            magA = np.divide(stack(magA, fresnel), fresnel)
            magB = np.divide(stack(magB, fresnel), fresnel)   

    # calculate the interferometric correlation between the two antennas for
    # the feature of interest
    sA = np.multiply(FOI, magA)
    sB = np.multiply(FOI, magB)
    snrA = np.mean(sA[np.isnan(sA) == False]) - noiseA
    snrB = np.mean(sB[np.isnan(sB) == False]) - noiseB
    gamma = np.sqrt(np.multiply(snrA / (snrA + 1), snrB / (snrB + 1)))
    #print('snrA:', snrA, 'snrB:', snrB, 'gamma:', gamma)

    # calculate the nadir emprirical interferometric phase pdf
    phi = np.linspace(-np.pi, np.pi, 10000)
    phi = (2 * np.pi * B * np.sin(phi)) / (299792458 / fc)
    

    #f = ipdf(1, gamma, phi, (phi_m / 180) * np.pi)
    f = ipdf(fresnel, gamma, phi, (phi_m / 180) * np.pi)


    phi = phi * 180 / np.pi

    return phi, f, gamma

def empirical_sample_mean(N, Nf, iphi, gamma, phi_m):
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
      phi_m: mean interferometric phase angle of the FOI

    Output:
    ------------
      sigma_phi: variance of the interferometric phase pdf
        sigma_m: sample error of the mean
    '''
    import numpy as np
    from scipy import integrate

    # calculate the standard deviation of the emprirical phase distribution
    func = lambda x: np.multiply(x**2, ipdf(N, gamma, x, 0))
    sigma_phi = np.sqrt(integrate.quad(func, -np.pi, np.pi)[0]) * 180 / np.pi
    print('sigma_phi:', sigma_phi)

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
        phi = np.linspace(-np.pi, np.pi, 1000)
        f = ipdf(N, gamma, phi, (phi_m / 180) * np.pi)
        for ii in range(simulations):
            # draw Nf samples from the emprirical interferometric phase pdf
            phin = np.random.choice(f, Nf)
            # calculate the sample mean of the selected Nf samples
            M[ii] = np.angle(np.mean(np.exp(np.multiply(1j, phin))), deg=True)
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

    import numpy as np
    import matplotlib.pyplot as plt

    # define sample vectors
    new_sample_interval = orig_sample_interval / subsample_factor
    orig_t = np.arange(0, (len(data) - 1) * orig_sample_interval, orig_sample_interval)
    new_t = np.arange(0, len(data) * orig_sample_interval, new_sample_interval)

    # perform the interpolation
    sincM = np.tile(new_t, (len(orig_t), 1)) - np.tile(orig_t[:, np.newaxis], (1, len(new_t)))
    output = np.dot(data, np.sinc(sincM / orig_sample_interval))
    
    return output

def coregistration(magA, phsA, magB, phsB, orig_sample_interval, subsample_factor, shift=50):
    '''
    function for sub-sampling and coregistering complex-valued range lines
    from two radargrams as requried to perform interferometry. Follows the
    steps outlines in Castelletti et al. (2018)

    Inputs:
    -------------
                      magA: magnitude of input A radargram
                      phsA: phase of input A radargram
                      magB: magnitude of input B radargram
                      phsB: phase of input B radargram
      orig_sample_interval: sampling interval of the input data
          subsample_factor: factor used modify the original fast-time sampling
                            interval

    Outputs:
    -------------
       coregA: coregistered complex-valued A radargram
       coregB: coregistered complex-valued B radargram
    '''

    import numpy as np
    import matplotlib.pyplot as plt

    # convert magnitude and phase to complex values
    cmplxA = convert_to_complex(magA, phsA)
    cmplxB = convert_to_complex(magB, phsB)

    # define the output
    coregA = np.zeros((len(magA), np.size(magA, axis=1)), dtype=complex)
    coregB = np.zeros((len(magB), np.size(magB, axis=1)), dtype=complex)

    for ii in range(np.size(magA, axis=1)):
        # subsample
        subsampA = sinc_interpolate(cmplxA[:, ii], orig_sample_interval, subsample_factor)
        subsampB = sinc_interpolate(cmplxB[:, ii], orig_sample_interval, subsample_factor)
        # co-register and shift
        shifts = np.arange(-1 * shift, shift, 1)
        rho = np.zeros((len(shifts), ), dtype=float)
        for jj in range(len(shifts)):
            tempA = np.abs(np.mean(np.multiply(subsampA, np.conj(np.roll(subsampB, shifts[jj])))))
            tempB = np.mean(np.square(np.abs(subsampA)))
            tempC = np.mean(np.square(np.abs(np.roll(subsampB, shifts[jj]))))
            tempD = np.sqrt(np.multiply(tempB, tempC))
            rho[jj] = np.divide(tempA, tempD)
        to_shift = shifts[np.argwhere(rho == np.max(rho))][0][0]
        subsampB = np.roll(subsampB, to_shift)
        # remove subsampling
        coregA[:, ii] = subsampA[np.arange(0, len(subsampA), subsample_factor)] 
        coregB[:, ii] = subsampB[np.arange(0, len(subsampB), subsample_factor)]

    # convert coregistered radargrams to magnitude and phase    
    magA_out, phsA_out = convert_to_magphs(coregA)
    magB_out, phsB_out = convert_to_magphs(coregB)

    return magA_out, phsA_out, magB_out, phsB_out

def load_roll(norm_path, s1_path):
    '''
    Function to extract and interpolate aircraft roll information such that it can
    be compared to the MARFA 1m focused data product.

    Inputs:
    -------------
       norm_path: path to the 'norm' folder containing the relavant 'roll_ang'
                  and 'syn_ztim' ascii files for the specific line being
                  investigated
         s1_path: path to the 'S1_POS' folder containing the relevant 'ztim_xyhd'
                  ascii file for the specific line being investigated.

    Output:
    ------------
       interpolated roll vector at 1m trace intervals
    '''

    import numpy as np
    import pandas as pd

    # load the roll data
    norm_roll = np.genfromtxt(norm_path + 'roll_ang')

    # extract the ztim vector from norm
    temp = pd.read_csv(norm_path + 'syn_ztim', header=None)
    norm_ztim = np.zeros((len(temp), ), dtype=float)
    for ii in range(len(norm_ztim)):
        norm_ztim[ii] = float(temp[2][ii].replace(')', ''))
    del temp

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
