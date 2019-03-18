authors__ = ['Gregor Steinbruegge (UTIG), gregor@ig.utexas.edu']

__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'March 05, 2019',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'Initial Release.'}}

import numpy as np
from scipy.constants import c
from scipy.ndimage.interpolation import shift
import spiceypy as spice
import pandas as pd
from misc import coord as crd
from scipy.optimize import least_squares
from math import tan, pi, erf, sqrt
import cmp.pds3lbl as pds3

def beta5_altimetry(cmp_path, science_path, label_science, label_aux,
                    idx_start=0, idx_end=None, use_spice=False, ft_avg=10,
                    max_slope=25, noise_scale=20, fix_pri=None, fine=True):

    """
    Computes altimetry profile based on Mark Haynes Memo.
    Method is the Beta5 model based on ocean altimetry.

    Input
    -----
    cmp_path: string
        Path to pulse compressed data. h5 file expected
    science_path: string
        Path to EDR science data
    label_path: string
        Path to science label of EDR data
    aux_label: string
        Path to auxillary label file
    idx_start (optional): integer
        Start index for processing track. Default is 0.
    idx_end (optional): integer
        End index for processing track. Default is None.
    use_spice (optional): boolean
        Specifies is spacecraft position is taken from the EDR
        data or if it is re-computed using a spice kernel.
        IMPORTANT NOTE: If spice is set to True then a valid
                        kernel must be furnished!
    ft_avg(optional): integer
        Window for fast time averaging. Set to None to turn off.
        Default is 10.
    max_slope (optional): double
        Maximum slope to be considered in coherent averaging.
        Default is 25 degrees.
    noise_scale (optional): double
        Scaling factor for the rms noise used within the threshold
        detection. Default is set to 20.
    fix_pri (optional): integer
        Pulse repetition interval code. If None it will be taken from
        the input data, otherwise it can be set to a fixed code.
        Fixing it avoids reading the bitcolumns in the input making
        the reading faster.
    fine (optional): boolean
        Apply fine detection with beta-5 model fit. Default is True.

    Output
    ------
    et: double
        Ephemeris time
    lat: double
        Latitude
    lon: double
        Longitude
    r: double
        Radius from CoM
 
    Output if generated as data frame.
    """

    #============================
    # Read and prepare input data
    #============================

    # Read input data
    if fix_pri is None:
        dbc = True
    else:
        dbc = False
    data = pds3.read_science(science_path, label_science, science=True, bc=dbc)
    aux = pds3.read_science(science_path.replace('_s.dat', '_a.dat'),
                            label_aux, science=False, bc=False)
    re = pd.read_hdf(cmp_path, key='real').values[idx_start:idx_end]
    im = pd.read_hdf(cmp_path, key='imag').values[idx_start:idx_end]
    cmp_track = re+1j*im

    # Get Range window start
    range_window_start = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]

    # Compute or read S/C position
    ets = aux['EPHEMERIS_TIME'].values[idx_start:idx_end]
    if use_spice is True:
        sc = np.empty(len(ets))
        lat = np.empty(len(ets))
        lon = np.empty(len(ets))
        for i in range(len(ets)):
            scpos, lt = spice.spkgeo(-74, ets[i], 'J2000', 4)
            llr = crd.cart2sph(scpos[0:3])
            lat[i] = llr[0]
            lon[i] = llr[1]
            sc[i] = np.linalg.norm(scpos[0:3])
    else:
        sc_x = aux['X_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc_y = aux['Y_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc_z = aux['Z_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        sc = np.sqrt(sc_x**2+sc_y**2+sc_z**2)
        lon = aux['SUB_SC_EAST_LONGITUDE'].values[idx_start:idx_end]
        lat = aux['SUB_SC_PLANETOCENTRIC_LATITUDE'].values[idx_start:idx_end]

    # Calculate offsets for radargram
    sc_cor = np.array(2000*sc/c/0.0375E-6).astype(int)
    phase = -sc_cor+range_window_start
    tx0 = int(min(phase))
    offset = int(max(phase) - tx0)

    # Get the shot frequency.
    if fix_pri is None:
        pri_code = data['PULSE_REPETITION_INTERVAL'].values[idx_start:idx_end]
    else:
        pri_code = np.full_like(sc, fix_pri)

    pri = np.zeros(len(pri_code))
    pri[np.where(pri_code == 1)] = 1428E-6
    pri[np.where(pri_code == 2)] = 1429E-6
    pri[np.where(pri_code == 3)] = 1290E-6
    pri[np.where(pri_code == 4)] = 2856E-6
    pri[np.where(pri_code == 5)] = 2984E-6
    pri[np.where(pri_code == 6)] = 2580E-6

    # Compute SAR apertures for coherent and incoherent stacking
    sc_alt = aux['SPACECRAFT_ALTITUDE'].values*1000
    vel_t = aux['MARS_SC_TANGENTIAL_VELOCITY'].values*1000
    fresnel = np.sqrt(sc_alt*c/10E6+(c/(20E6))**2)
    sar_window = int(np.mean(2*fresnel/vel_t/pri)/2)
    coh_window = int(np.mean((c/10E6/tan(max_slope*pi/180)/vel_t/pri)))

    #========================
    # Actual pulse processing
    #========================

    # Perform smoothing of the waveform aka averaging in fast time
    if ft_avg is not None:
        wvfrm = np.empty((len(cmp_track),3600),dtype=np.complex)
        for i in range(len(cmp_track)):
            rmean = running_mean(abs(cmp_track[i]),10)
            wvfrm[i] = rmean-np.mean(rmean)
            wvfrm[i,:10] = wvfrm[i,20:30]
            wvfrm[i,-10:] = wvfrm[i,-30:-20] 
    else:
        wvfrm = cmp_track

    # Construct radargram
    radargram=np.zeros((len(cmp_track),3600),dtype=np.complex)
    for rec in range(len(cmp_track)):
        radargram[rec] = np.roll(wvfrm[rec],int(phase[rec+idx_start]-tx0))

    # Perform averaging in slow time. Pulse is averaged over the 1/4 the
    # distance of the first pulse-limited footprint and according to max
    # slope specification.
    avg_c = np.empty((len(data[idx_start:idx_end]), 3600), dtype=np.complex)
    for i in range(3600):
        avg_c[:, i] = running_mean(radargram[:, i], coh_window)

    avg = np.empty((len(data[idx_start:idx_end]), 3600))
    for i in range(3600):
        avg[:, i] = abs(running_mean(abs(avg_c[:, i]), sar_window))

    # Coarse detection
    coarse = np.zeros(len(avg), dtype=int)
    deriv = np.zeros((len(avg), 3599))

    for i in range(len(avg)):
        deriv[i] = abs(np.diff(avg[i]))

    noise = np.sqrt(np.var(deriv[:, -1000:], axis=1))*noise_scale
    for i in range(0, len(deriv)):
        found = False
        for lvl in np.arange(1, 0, -0.1):
            for j in range(int(phase[i]-tx0)+20,
                           min(int(phase[i]-tx0)+1020, len(deriv[i]))):
                if deriv[i, j] > noise[i]*lvl:
                    coarse[i] = j
                    found = True
                    break
            if found == True: break

    # Perform least-squares fit of waveform according to beta-5 re-tracking
    # model
    delta = np.zeros(len(avg))
    if fine is True:
        b3 = 100
        b4 = 2
        b5 = 3E-2
        for i in range(0, len(avg[idx_start:idx_end])):
            wdw = avg[i, max(coarse[i]-100, 0):min(coarse[i]+100, len(avg[i]))]
            b1 = np.mean(wdw[0:128])
            b2 = max(wdw)-b1
            res = least_squares(model5, [b1, b2, b3, b4, b5], args=wdw,
                                bounds=([-500, 0, 0, 0, 0],
                                        [np.inf, np.inf, np.inf, np.inf, 1]))
            delta[i] = res.x[2]+max(coarse[i]-100, 0)
    else:
        delta = coarse

    # Time-of-Flight (ToF)
    tx = (range_window_start+delta-phase+tx0)*0.0375E-6+pri-11.98E-6
    # One-way range in km
    d = tx*c/2000

    # Elevation from Mars CoM
    r = sc-d

    spots = np.empty((len(r), 4))
    columns = ['et', 'spot_lat', 'spot_lon', 'spot_radius']
    for i in range(len(r)):
        spots[i, :] = [ets[i], lat[i], lon[i], r[i]]
    df = pd.DataFrame(spots, columns=columns)

    return df

def running_mean(x, N):
    res = np.zeros(len(x), dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0), dtype=x.dtype)
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

def model5(beta, *wvfrm):
    t = np.arange(len(wvfrm))
    erf_v = np.vectorize(erf)
    Q = t-(beta[2]+0.5*beta[3])
    Q[np.where(t < (beta[2]-2*beta[3]))] = 0
    y = beta[0]+beta[1]*np.exp(-beta[4]*Q)*0.5*(1+erf_v((t-beta[2])/(sqrt(2)*beta[3])))
    return y-wvfrm

