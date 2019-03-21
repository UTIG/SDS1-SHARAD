
authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu',
             'Kirk Scanlan, kirk.scanlan@gmail.com']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'August 15, 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'
                 'CMP Library is a collection of functions used'
                 'for the pulse compression of SHARAD data and'
                 'to correct for the ionospheric distortion'}}

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def us_refchirp(iono=True, custom=None, maxTECU=1, resolution=50):
    """
    This subroutine creates SHARAD reference chirps used for
    pulse compression as a function of different TECU values.
    TECU is TEC x 10^-16. TEC is correlated with the E value
    by TEC = E/0.29. For reference see
    Campbell et al. 2011 doi: 10.1109/LGRS.2011.2143692
    Campbell and Watters 2016 doi: 10.1002/2015JE004917
    If no Ionosphere is present TECU is set to 0.

    Input:
    -----------
        maxTECU (optional): Maximum TECU value to be expected.
                            Default value is 1.

    Output:
    -----------
        fs: Set of filter functions (reference chirps)
            suitable for pulse compression
    """

    # Parameters
    fl = 15E+6   # Sharad lower frequency 15 MHz
    a = (10E+6/85.05E-6) # Frequency rate 10 MHz/85 mus
    t = np.arange(0, 85.05E-6, 0.0375E-6) # Times in rng window

    # Chirp can beexpressed as an instantaneous angular frequency
    phi = 2*np.pi*(t*fl+a/2*t**2)

    if iono:
        if custom is None:
            # Initialize filter array
            fs = np.empty((maxTECU*resolution, 3600), dtype=np.complex_)
            for i in range(0, maxTECU*resolution):
                # Calculate empiric E value
                E = i*1E+16/(0.29*resolution)
                # Compute phase shift due to ionospheric distortion
                phase = E*(fl+a*t)**(-1.93)
                # Compute distorted chirp
                C = -np.sin(phi-phase)+1j*np.cos(phi-phase)
                # Pad to 3600 samples
                ref_chirp = np.pad(C, (3600-len(C), 0),
                                   'constant', constant_values=0)
                # Chirp needs to be flipped due to SHARAD aliasing
                ref_chirp = np.fft.fft(np.flipud(ref_chirp))
                fs[i] = ref_chirp
        else:
            E = custom*1E+16/(0.29*resolution)
            # Compute phase shift due to ionospheric distortion
            phase = E*(fl+a*t)**(-1.93)
            # Compute distorted chirp
            C = -np.sin(phi-phase)+1j*np.cos(phi-phase)
            # Pad to 3600 samples
            ref_chirp = np.pad(C, (3600-len(C), 0),
                               'constant', constant_values=0)
            # Chirp needs to be flipped due to SHARAD aliasing
            fs = np.fft.fft(np.flipud(ref_chirp))
    else:
        # Without ionosphere - no phase
        C = -np.sin(phi)+1j*np.cos(phi)
        # Pad to 3600 samples
        ref_chirp = np.pad(C, (3600-len(C), 0), 
                           'constant', constant_values=0)
        # Flip and fft
        fs = np.fft.fft(np.flipud(ref_chirp))

    return fs


def us_rng_cmp(data, chirp_filter=True, iono=True, maxTECU=1, resolution=50,
               debug=True):
    """
    Performs the range compression according to the Bruce Campbell
    method. In case of ionosphere it tries to find the optimal
    TEC expressed by the empiric factor E to get the best SNR return

    Input:
    -----------
        raw_data:       Track to be compressed [len(track) x 3600 samples]
        chirp_filter: If a filter is applied to the reference chirps
        iono (optional): If ionospheric correction is needed
        maxTECU:    Maximum TECU = TEC x 10E-16 to be expected
        resolution: In how many steps the TECU shall be tested
    Output:
    -----------
        E:         Optimal E value found
        dechirped: Pulse compressed with optimal E value
    """
    # TODO: make plotting optional with an arg
    # Compute list of reference chirps
    fs = us_refchirp(iono, resolution=resolution,
                     maxTECU=maxTECU)
    if iono:
        csnr = np.empty((len(fs), len(data)))
        # Perform range compression per filter and record SNR
        # GNG: for i, chirp in enumerate(fs)
        for i in range(0, len(fs)):
            # GNG: TODO: move this fft call out of the loop
            product = np.fft.fft(data)*np.conj(fs[i])
            # apply frequency domain filter if desired
            if chirp_filter:
                # GNG: move hamming filter out of the loop
                product = np.multiply(product, Hamming(15E6, 25E6))
            dechirped = np.fft.ifft(product)
            # Noise is recorded within first 266 samples
            var = np.var(dechirped[:, 0:266], axis=1)
            # Signal is the maximum amplitude
            maxi = np.max(abs(dechirped), axis=1)
            csnr[i] = maxi**2/var

        Emax = np.argmax(csnr, axis=0)
        # Create a histogram of SNR maximizing E's
        hist, edges = np.histogram(Emax, bins=resolution*maxTECU)
        if debug:
            plt.bar(np.arange(len(hist)), hist)
            plt.show()

        # Fit histogram by a Gauss function
        try:
            opt, cov = curve_fit(Gaussian, np.arange(len(hist)),
                                 hist, 
                                 p0=[len(data)/2, resolution*maxTECU/2, 20])
        except:
            opt = [-1, 0, -1]
            cov = [-1, -1, -1]

        if debug: 
            print('Gauss fit opt/cov:', opt, cov)

        x0 = min(49, max(0, opt[1]))
        E = x0/maxTECU/resolution
        sigma = opt[2]/maxTECU/resolution

        # Pulse compress whole track with optimal E
        fs = us_refchirp(iono, resolution=resolution,
                         custom=x0)
        product = np.fft.fft(data)*np.conj(fs)
        # apply frequency domain filter if desired
        if chirp_filter:
            product = np.multiply(product, Hamming(15E6, 25E6))
        dechirped = np.fft.ifft(product)

    else:
        E = 0
        sigma = 0
        product = np.fft.fft(data)*np.conj(fs)
        # apply frequency domain filter if desired
        if chirp_filter:
            product = np.multiply(product, Hamming(15E6, 25E6))
        dechirped = np.fft.ifft(product)
        if debug:
            plt.show()
    return E, sigma, dechirped

def Gaussian(x, a, x0, sigma):
    """
    Simple Gaussian distribution
    This function is used internally for curve fitting.

    Input:
    -----------
        x: Input array of x-values for Gaussian to be evaluated
        a: Amplitude at x0
       x0: Center of Gaussian
    sigma: Standard deviation of Guassian

    Output:
    -----------
        Gaussian at x values

    """
    return a*np.exp(-(x - x0)**2/(2*sigma**2))

def Hamming(Fl, Fh):
    """
    Create a frequency domain Hamming filter

    Input:
    -----------
       Fl: lower cutoff frequency for the Hamming filter
       Fh: upper cutoff frequency for the Hamming filter

    Output:
    -----------
        Frequency domain Hamming filter
    """

    min_freq = int(round((Fl) * 3600 / (1/0.0375E-6)))
    max_freq = int(round((Fh) * 3600 / (1/0.0375E-6)))
    dfreq = max_freq - min_freq + 1
    hamming = np.sin(np.linspace(0, 1, num=dfreq) * np.pi)
    hfilter = np.flipud(np.hstack((np.zeros(min_freq), hamming,
                        np.zeros(3600 - min_freq - hamming.size))))
    return hfilter

def decompressSciData(data, compression, presum, bps, SDI):
    """
    Decompress the science data according to the
    SHARAD interface specification.

    Input:
    -----------
        data: data to be decompressed
        compression: type of compression.
                     use 'static' or 'dynamic'
        presum: Onboard presumming parameter
        bps: Compression parameter
        SDI: compression parameter
    Output:
    -----------
        Decompressed data
    """

    #TODO: only static decompression is currently implemented
    #      implement dynamic decompression

    if compression == 'static' or compression == 'dynamic':
        if compression == 'static': # Static scaling
            L = np.ceil(np.log2(int(presum)))
            R = bps
            S = L - R + 8
            N = presum
            decomp = np.power(2, S) / N
            decompressed_data = data * decomp
        # dynamic currently disabled!
        elif compression == True:#dynamic scaling
            N = presum
            if SDI <= 5:
                S = SDI
            elif 5 < SDI <= 16:
                S = SDI - 6
            elif SDI > 16:
                S = SDI - 16
            decompressed_data = data * (np.power(2, S) / N)
        return decompressed_data
    else:
        # TODO: logging, should this be an exception?
        print('Decompression Error: Compression Type {} not understood'.format(compression))
    return

