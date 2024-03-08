import math
import glob
import cmath as cm
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt


def itus_iono_optimize(data, tx, rx, f0, B, E=None):

    chirp = ita_iono_refchirp(rx, tx)
    if E is None:
        # define possible E values
        E = np.linspace(1E15, 5E17, 500)
        csnr = np.empty((len(data), len(E)), dtype=float)
        for jj in range(len(E)):
            # apply the selected E value
            corr_data = itus_iono_rng_comp(data, chirp, E[jj], f0, B)
            # extract SNR for each range line
            var = np.var(corr_data[:,0:266], 1)
            maxi = np.max(abs(corr_data), 1)
            csnr[:, jj] = maxi**2/var
        # define the E value yeilding the greatest SNR
        maxE = np.argmax(csnr, axis=1)
        histE = E[maxE]
        # extract the mean E value yielding the greatest SNR
        meanE = E[int(round(np.mean(maxE)))]
    else:
        histE = E
        meanE = E

    # apply the mean E value yielding the greatest SNR to the data
    dechirped = itus_iono_rng_comp(data, chirp, meanE, f0, B)

    return dechirped, meanE, histE

def itus_iono_rng_comp(data, chirp, E, f0, B, baseband=False):

    # separate the italian chirp function into its
    # magnitude and phase components in the time domain
    if baseband:
        chirp = np.fft.ifft(np.fft.ifftshift(chirp))
    else:
        chirp = np.fft.ifft(chirp, 4096)
    phi = np.zeros(len(chirp), dtype=float)
    mag = np.zeros(len(chirp), dtype=float)
    for ii in range(len(chirp)):
        phi[ii] = cm.phase(chirp[ii])
        mag[ii] = np.abs(chirp[ii])

    # prepare the science data
    tm = ((3/80)*10**-6)*np.arange(0, 4096, dtype=np.float)
    n = 4096-3600;
    data = np.pad(data, ((0,0),(0,n)), 'constant', constant_values=0)
    if baseband:
        comb_exp = np.exp(2*math.pi*(((80/3)-20)*10**6)*tm*1j)
        temp1 = np.multiply(data, comb_exp)
    else:
        temp1 = data
    temp2 = np.fft.fft(temp1)
    temp3 = temp2[:, 0:2048]

    # modify the phase of the chirp according to US methodology
    if baseband:
        tm = tm[0:2048]
        a = B/np.max(tm)
    else:
        a = B/85.05E-6
    dphi = E*((f0+B/2)-a*tm)**-1.93
    corr_phi = phi-dphi
    corr_chirp = np.zeros(len(dphi), dtype=complex)
    for ii in range(len(corr_chirp)):
        corr_chirp[ii] = cm.rect(mag[ii], corr_phi[ii])
    if baseband:
        corr_chirp = np.fft.fft(corr_chirp)
        corr_chirp = corr_chirp[0:2048]
    else:
        corr_exp = np.exp(-2*math.pi*(((80/3)-20)*10**6)*tm*1j)
        corr_chirp = np.flipud(np.multiply(corr_chirp, corr_exp))
        corr_chirp = np.fft.fft(corr_chirp)[1025:3073]

    # apply to the radar data
    dechirped = np.fft.ifft(np.multiply(temp3, np.conj(corr_chirp)))
    dechirped = np.pad(dechirped, ((0,0),(0,1552)), 'constant', constant_values=0)

    return dechirped

def read_refchirp(path):
    """
    This routine reads binary files with reference chirps for SHARAD as
    given on PDS by the italian team. Returns a numpy array with 2048
    complex value samples.

    Input:
    -----------
        path: Path to the according file.
    Output:
        numpy array with 2048 samples. Complex values.
    """

    fil = glob.glob(path)[0]
    arr = np.fromfile(fil, dtype='f4')
    arr = arr[0:2048] + 1j*arr[2048:4096]
    return arr

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def ita_comp(tx, rx, uncompressed, SDS=None):
    """ Select calibrated reference chirp from tx and rx temperature """
    txtemp = [-20, -15, -10, -5, 0, 20, 40, 60]
    rxtemp = [-20, 0, 20, 40, 60]
    txval = find_nearest(txtemp, tx)
    rxval = find_nearest(rxtemp, rx)

    prefix_tx = 'm' if txval < 0 else 'p'
    prefix_rx = 'm' if rxval < 0 else 'p'

    if SDS is None:
        SDS = os.getenv('SDS', '/disk/kea/SDS')
    chirpfile = 'reference_chirp_' + prefix_tx+str(txval).zfill(2)+'tx_'+prefix_rx+str(rxval).zfill(2)+'rx.dat'
    path = os.path.join(SDS, 'orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/calib', chirpfile)

    cplx_ref_chirp = read_refchirp(path)

    # Adjust sample size from 3600 to 4096 by zero padding
    padded = np.pad(uncompressed, (0, 496), 'constant', constant_values=0)
    # Sample frequency
    fc = (3/80)*1E-6
    t = np.arange(4096)*fc
    adj_raw = np.exp(2*np.pi*1j*(80/3 - 20)*1E+6*t)*padded
    # Pick center of frequency space
    rchirp = np.fft.fft(adj_raw)[0:2048]
    product = rchirp*np.conj(cplx_ref_chirp)
    dechirped = np.fft.ifft(product)

    return dechirped

def iono_rng_compression(i, *args):
    fs, data = args
    i = min(999, max(0, i))

    padded=np.pad(data,(0,496), 'constant', constant_values=0)
    # Sample frequency
    fc = (3/80)*1E-6
    t = np.arange(4096)*fc
    adj_raw = np.exp(2*np.pi*1j*(80/3 - 20)*1E+6*t)*padded
    # Pick center of frequency space
    rchirp = np.fft.fft(adj_raw)[0:2048]
    product = rchirp*np.conj(fs[i])
    dechirped = np.fft.ifft(product)

    var = np.var(dechirped[0:266])
    maxi = np.max(np.abs(dechirped))
    return 1/(maxi**2/var)

class MyBounds(object):
    def __init__(self, xmax=[999,999], xmin=[0,0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def iono_optimize(data, fs):
    mybounds = MyBounds()
    res = scipy.optimize.basinhopping(iono_rng_compression, 100,
               minimizer_kwargs = {'args':(fs,data)},
               niter=500, T=1.0, accept_test=mybounds,
               stepsize=10, interval=50, niter_success=None)

    dT =  int(res.x)
    dphi = 2*np.pi*dT*37.5E-3
    dt = dphi/(2*np.pi*20E+6)

    product = np.fft.fft(data)*np.conj(np.fft.fft(fs[dT]))
    dechirped = np.flipud(np.fft.ifft(product))

    return dt, dT, dphi, dechirped

def iono_rng_comp(data,fs):

    padded=np.pad(data,(0,496), 'constant', constant_values=0)
    fc=(3/80)*1E-6
    t=np.arange(4096)*fc
    adj_raw=np.exp(2*np.pi*1j*(80/3 - 20)*1E+6*t)*padded
    rchirp=np.fft.fft(adj_raw)[0:2048]
    csnr = np.empty(len(fs))

    for l in range(0, len(fs)):
        product = rchirp*np.conj(fs[l])
        dechirped = np.fft.ifft(product)

        var = np.var(dechirped[0:266])
        maxi = np.max(abs(dechirped))
        csnr[l] = maxi**2/var

    dT = np.argmax(csnr)
    dphi = 2*np.pi*dT*37.5E-3
    dt = dphi/(2*np.pi*20E+6)

    product = rchirp*np.conj(fs[int(dT)])
    dechirped = np.fft.ifft(product)
    #plt.plot(dechirped)
    #plt.show()
    return dt, dT, dphi, dechirped

def ita_iono_refchirp(rx,tx,hfilter=False,cutoff=None):

    nb_samples = 2048
    T = 85.05E-6
    cf = 20E+6
    bw = 10E+6
    f_sample = 1/0.0375E-6

    # Select calibrated reference chirp from tx and rx temperature
    txtemp = [-20, -15, -10, -5, 0, 20, 40, 60]
    rxtemp = [-20, 0, 20, 40, 60]
    txval = find_nearest(txtemp, tx)
    rxval = find_nearest(rxtemp, rx)

    prefix_tx = 'm' if txval < 0 else 'p'
    prefix_rx = 'm' if rxval < 0 else 'p'

    path='/disk/daedalus/sharaddownload/mrosh_0001/calib/reference_chirp_' \
          + prefix_tx+str(txval).zfill(2)+'tx_'+prefix_rx+str(rxval).zfill(2)+'rx.dat'

    refchirp = read_refchirp(path)

    if hfilter is True:
        refchirp = np.fft.fft(refchirp)
        # Compute Hamming Filter
        if cutoff is None:
            min_freq = int(round((cf-bw/2) * nb_samples / f_sample))
            max_freq = int(round((cf+bw/2) * nb_samples / f_sample))
        else:
            min_freq = int(round(cutoff[0] * nb_samples / f_sample))
            max_freq = int(round(cutoff[1] * nb_samples / f_sample))
        dfreq = max_freq - min_freq + 1
        hamming = np.sin(np.linspace(0, 1, num=dfreq) * np.pi)
        hfilter = np.hstack((np.zeros(min_freq),
                             hamming,
                             np.zeros(f_sample - min_freq - hamming.size)))

        refchirp = np.fft.ifft(hfilter*refchirp)

    return refchirp

def ref_chirp(cf, bw, pw, r_window_size, f_sample, hfilter=True, cutoff=None):

    window = (np.arange(r_window_size)*1/f_sample)[0:int(pw*f_sample)]
    chirp = signal.chirp(window, cf-bw/2, pw, cf+bw/2) \
           +1j*signal.chirp(window, cf-bw/2, pw, cf+bw/2, phi=90)


    chirp = np.pad(chirp, (0,int((r_window_size-int(pw*f_sample)))),
                   'constant', constant_values=0)

    # Make a reference chirp
    rchirp = np.flipud(chirp)
    ref_chirp = np.fft.fft(rchirp, r_window_size)

    if hfilter is True:
        # Compute Hamming Filter
        if cutoff is None:
            min_freq = int(round((cf-bw/2) * r_window_size / f_sample))
            max_freq = int(round((cf+bw/2) * r_window_size / f_sample))
        else:
            min_freq = int(round(cutoff[0] * r_window_size / f_sample))
            max_freq = int(round(cutoff[1] * r_window_size / f_sample))
        dfreq = max_freq - min_freq + 1
        hamming = np.sin(np.linspace(0, 1, num=dfreq) * np.pi)
        hfilter = np.hstack((np.zeros(min_freq),
                             hamming,
                             np.zeros(r_window_size - min_freq - hamming.size)))

        ## Filter Chirp
        ref_chirp = np.multiply(ref_chirp, hfilter)

    return ref_chirp

def ref_chirp(cf, bw, pw, r_window_size, f_sample, hfilter=True, cutoff=None):

    window = (np.arange(r_window_size)*1/f_sample)[0:int(pw*f_sample)]
    chirp = signal.chirp(window, cf-bw/2, pw, cf+bw/2) \
           +1j*signal.chirp(window, cf-bw/2, pw, cf+bw/2, phi=90)


    chirp = np.pad(chirp, (0,int((r_window_size-int(pw*f_sample)))),
                   'constant', constant_values=0)

    # Make a reference chirp
    rchirp = np.flipud(chirp)
    ref_chirp = np.fft.fft(rchirp, r_window_size)

    if hfilter is True:
        # Compute Hamming Filter
        if cutoff is None:
            min_freq = int(round((cf-bw/2) * r_window_size / f_sample))
            max_freq = int(round((cf+bw/2) * r_window_size / f_sample))
        else:
            min_freq = int(round(cutoff[0] * r_window_size / f_sample))
            max_freq = int(round(cutoff[1] * r_window_size / f_sample))
        dfreq = max_freq - min_freq + 1
        hamming = np.sin(np.linspace(0, 1, num=dfreq) * np.pi)
        hfilter = np.hstack((np.zeros(min_freq),
                             hamming,
                             np.zeros(r_window_size - min_freq - hamming.size)))

        ## Filter Chirp
        ref_chirp = np.multiply(ref_chirp, hfilter)

    return ref_chirp

def iono_bruce_refchirp(maxE, resolution, A=None, txval=20, rxval=20):
    FL = 15E+6
    FH = 25E+6

    a = (10E+6/85.05E-6)
    t = np.arange(0,85.05E-6,0.0375E-6)

    if A is None:
        prefix_tx = 'm' if txval < 0 else 'p'
        prefix_rx = 'm' if rxval < 0 else 'p'

        path='/disk/daedalus/sharaddownload/mrosh_0001/calib/reference_chirp_' \
              + prefix_tx+str(txval).zfill(2)+'tx_'+prefix_rx+str(rxval).zfill(2)+'rx.dat'
        A = abs(np.fft.ifft(read_refchirp(path)))
        A = np.pad(A,(0,2268-2048),'constant',constant_values=0)

    phi = 2*np.pi*(t*FL+a/2*t**2)
    fs = np.zeros((maxE*resolution,3600),dtype=complex)
    for i in range(0,maxE*resolution):
        E = i/resolution
        phi2 = E*1E+16/0.29*(FL+a*t)**(-1.93)
        C = -A*np.sin(phi-phi2)+1j*A*np.cos(phi-phi2)
        ref_chirp = np.pad(C,(3600-len(C),0),'constant',constant_values=0)
        ref_chirp = np.fft.fft(np.flipud(ref_chirp))
        fs[i] = ref_chirp

    return fs
