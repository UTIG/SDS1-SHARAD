authors__ = ['Gregor Steinbruegge (UTIG), gregor@ig.utexas.edu']

__version__ = '0.1'
__history__ = {
    '0.1':
        {'date': 'April 15, 2019',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'Initial Release. Working version. Not fully documented!'},
    '1.0':
	{'data': 'August 28, 2019',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'Integrated the clutter simulation code and updated documentation'}}

import numpy as np
import sys
import logging
sys.path.append('../../xlib')
import cmp.pds3lbl as pds3
import scipy.signal
import scipy.constants
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import rng.icsim as icsim
import clutter.peakint as peakint

def gen_or_load_cluttergram(cmp_path, dtm_path, science_path, label_science,
                aux_path, label_aux, idx_start, idx_end, 
                debug = False, ipl = False, #window = 50,
                #average = 30,
                cluttergram_path=None, save_clutter_path=None,
                do_progress=False, maxechoes=0):


    """ TODO: put this into icsim.py
    Returns a cluttergram simulation result

    """
    # SHARAD sampling frequency
    #f = 1/0.0375E-6

    # Number of range lines
    Necho = idx_end - idx_start

    #============================
    # Assertions about input dimensions
    assert idx_start >= 0
    assert idx_end > 0
    assert idx_end - idx_start >= 800


    #============================
    # Read and prepare input data
    #============================

    # GNG: This data gets reread in run_ranging. Is it worth caching?

    # Data for RXWOTs
    data = pds3.read_science(science_path, label_science, science=True,
                              bc=False)
    # Range window starts
    rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]


    # Perform clutter simulation or load existing cluttergram
    if cluttergram_path is None:
        logging.debug("Performing clutter simulation")
        aux = pds3.read_science(aux_path, label_aux, science=False,
                                bc=False)
        pri_code = np.ones(Necho)
        p_scx = aux['X_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        p_scy = aux['Y_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        p_scz = aux['Z_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        v_scx = aux['X_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        v_scy = aux['Y_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        v_scz = aux['Z_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        state = np.vstack((p_scx, p_scy, p_scz, v_scx, v_scy, v_scz))
        # GNG 2020-01-27 transpose seems to give matching dimensions to pulse compressed radargrams
        sim = icsim.incoherent_sim(state, rxwot, pri_code, dtm_path, idx_start, idx_end,
                                   do_progress=do_progress, maxechoes=maxechoes).transpose()
        if save_clutter_path is not None: 
            logging.debug("Saving cluttergram to " + save_clutter_path)
            np.save(save_clutter_path, sim)
    else:
        logging.debug("Loading cluttergram from " + cluttergram_path)
        sim = np.load(cluttergram_path)
    return sim


def icd_ranging_2(cmp_data_sliced, sim_data_sliced, idx_start, idx_end, 
                debug=False, ipl=False, window=50, 
                average=30, co_sim=10, co_data=30,
                cluttergram_path=None, save_clutter_path=None,
                do_progress=False):
    """ Alternative icd_ranging assuming some of the data files have been read or calculated externally
    cmp_data_sliced -- pulse-compressed data, already sliced to only idx_start to idx_end
    sim_data_sliced -- cluttergram data, already sliced to only idx_start to idx_end
    idx_start -- starting index of data being considered, mapping to  cmp_data_sliced[0] and sim_data_sliced[-1]
    idx_end -- ending index of data being considered, mapping to cmp_data_sliced[-1]

    """
    assert cmp_data_sliced.size == sim_data_sliced.size


def icd_ranging(cmp_path, dtm_path, science_path, label_science,
                aux_path, label_aux, idx_start, idx_end, 
                debug = False, ipl = False, window = 50, 
                average = 30, co_sim = 10, co_data = 30,
                cluttergram_path=None, save_clutter_path=None,
                do_progress=False, maxechoes=0):

    """
    Computes the differential range between two tracks
    by matching incoherent cluttergrams with pulse compressed data.

    # TODO: document what the minimum window size is (min value of idx_end - idx_start)
    # TODO: rework naming and make more consistent cluttergram_path and save_clutter_path naming and mechanics
    # TODO: do template matching using sub-sample resolution

    # Question: ensure that moving average is centered?

    Input
    -----
    cmp_path: string
        Path to pulse compressed data. h5 file expected
    sim_path: string
        Clutter simulation for track
    science_path: string
        Path to EDR science data
    label_path: string
        Path to science label of EDR data
    idx_start: integer
        start index for xover
    idx_end: integer
        end index for xover  
    do_progress: boolean
        Show progress bar to console if true
        
    Output
    ------
    delta
    dz
    min(md)


    """


    sim = gen_or_load_cluttergram(cmp_path, dtm_path, science_path, label_science,
                aux_path, label_aux, idx_start, idx_end, 
                debug = debug, ipl=ipl, #window=window, 
                #average=average, co_sim=co_sim, co_data=co_data,
                cluttergram_path=cluttergram_path, save_clutter_path=save_clutter_path,
                do_progress=do_progress, maxechoes=maxechoes)

    return icd_ranging_cg(cmp_path, dtm_path, science_path, label_science,
                aux_path, label_aux, idx_start, idx_end, sim,
                debug=debug, ipl=ipl, window=window,
                average=average, co_sim=co_sim, co_data=co_data,
                do_progress=do_progress, maxechoes=maxechoes)




def icd_ranging_cg3(cmp_path, dtm_path, science_path, label_science,
                aux_path, label_aux, idx_start, idx_end, sim,
                debug=False, ipl=False, window=50,
                average=30, co_sim=10, co_data = 30,
                do_progress=False, maxechoes=0):

    """
    Computes the differential range between two tracks
    by matching incoherent cluttergrams with pulse compressed data.
    Based on icd_ranging_cg, but modified to use a shifted MAD rather than rolled,
    and adding peakint.  Consider removing interpolation if results are good.



    # TODO: document what the minimum window size is (min value of idx_end - idx_start)
    # TODO: rework naming and make more consistent cluttergram_path and save_clutter_path naming and mechanics
    # TODO: do template matching using sub-sample resolution

    # Question: ensure that moving average is centered?

    Input
    -----
    cmp_path: string
        Path to pulse compressed data. h5 file expected
    sim_path: string
        Clutter simulation for track
    science_path: string
        Path to EDR science data
    label_path: string
        Path to science label of EDR data
    idx_start: integer
        start index for xover
    idx_end: integer
        end index for xover  
    do_progress: boolean
        Show progress bar to console if true

    sim: ndarray
        cluttergram

    Output
    ------
    delta
    dz
    min(md)


    """

    if debug:
        plt.imshow(sim)
        plt.title('Clutter simulation')
        plt.show()


    # SHARAD sampling frequency
    fs = 1/0.0375E-6

    # Number of range lines
    Necho = idx_end - idx_start

    #============================
    # Assertions about input dimensions
    assert idx_start > 0
    assert idx_end > 0
    assert idx_end - idx_start >= 800


    #============================
    # Read and prepare input data
    #============================

    # GNG: This data gets reread in run_ranging. Is it worth caching?

    # Data for RXWOTs
    data = pds3.read_science(science_path, label_science, science=True, bc=False)
    # Range window starts
    rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]
    del data


    # Pulse compressed radargrams
    re = pd.read_hdf(cmp_path, key='real').values[idx_start:idx_end].astype(float)
    im = pd.read_hdf(cmp_path, key='imag').values[idx_start:idx_end].astype(float)
    #power_db = 20*np.log10(np.abs(re+1j*im)+1E-3)
    #power_db = 20*np.log10(np.hypot(re, im)+1E-3)
    power_db = 10*np.log10(re*re + im*im +1E-3)

    # can we assert len(sim) == idx_end - idx_start now?

    # Free memory - science data not needed anymore
    
    try:
        #assert 0 < power_db.shape[0] <= len(rxwot)
        #assert 0 < sim.shape[0] <= len(rxwot)
        # Cut to 800 samples.
        data_new = shift_ft_arr(power_db[:, 0:800], rxwot)
        sim_new = shift_ft_arr(sim[:, 0:800], rxwot)
    except (IndexError, AssertionError):
        logging.error("size of power_db: " + str(power_db.shape))
        logging.error("size of sim: " + str(sim.shape))
        logging.error("rxwot shape: " + str(rxwot.shape))
        logging.error("rxwot: " + str(rxwot))
        raise
    # Free memory
    del re, im, power_db

    # TODO: document which axis of data is slow time and which is fast time
    # In data and avg, axis 0 is slow time, axis 1 is fast time
 
    # Perform correlation

    logging.debug("icd_ranging: Cut noise floor")
    # Cut noise floor
    # TODO: GNG -- I think this can be optimized
    data_new[np.where(data_new < (np.max(data_new) - co_data))] = (np.max(data_new) - co_data)
    sim_new[np.where(sim_new < (np.max(sim_new) - co_sim))] = (np.max(sim_new) - co_sim)

    logging.debug("icd_ranging: Normalize")
    # Normalize
    # TODO: GNG -- I think this can be optimized
    data_norm = (data_new-np.min(data_new))/(np.max(data_new)-np.min(data_new))
    sim_norm = (sim_new-np.min(sim_new))/(np.max(sim_new)-np.min(sim_new))
    #sim_norm[np.where(sim_norm>0.1)] = 1
    #data_norm[np.where(data_norm>0.1)] = 1

    logging.debug("icd_ranging: Averaging")
    # Average: Perform averaging samples of the radargram in slow time (axis=1)
    # GNG: we do a running mean here, which could be a pretty big low pass filter,
    # GNG: but should we be doing an academically robust lowpass filter of some sort?
    # GNG: It might also be algorithmically quicker. But cumsum/running mean is pretty quick
    # GNG: versus the correlation below right now. 2020-01-29

    data_avg = np.empty_like(data_norm)
    for i in range(data_norm.shape[1]):
        data_avg[:, i] = running_mean(data_norm[:, i], average)

    sim_avg = np.empty_like(sim_norm)
    for i in range(sim_norm.shape[1]):
        sim_avg[:, i] = running_mean(sim_norm[:, i], average)

    del data_norm, sim_norm


    # Correlate non-interpolated radargram rolling through the window

    # Correlate averaged radargram, rolling in fast time.  Will there be local minima?
    logging.debug("icd_ranging: Correlating arrays of length window={:d} and shape {:s}".format(\
                  window, str(sim_avg[average:-average].shape)))

    p = icsim.prg.Prog(2*window) if do_progress else None
    # TODO: correlate using scipy correlation
    md = np.empty(2*window)
    for j, i in enumerate(range(-window, window, 1)):
        if p:
            p.print_Prog(j)
        #md[j] = mean_abs_diff_rolled(i, sim_avg[average:-average], data_avg[average:-average])
        md[j] = mean_abs_diff_shifted(i, sim_avg[average:-average], data_avg[average:-average])
    if p:
        p.close_Prog()



    # Interpolate radar records in fast time
    if ipl:

        # Use peakint to calculate sub-sample minimum
        xmin = np.argmin(md)
        p, y, _ = peakint.qint(-md, xmin)
        xmin += p
        logging.debug("icd_ranging: Interpolating from x={:0.3f}".format(xmin))

        ifactor = 10 # Interpolation factor
        w_range = ifactor*window
        p_step = 0.1

        # Calculate the index of best match from the correlation, once we upsample
        i_interp = (-window + xmin)*ifactor


        data_ipl = np.empty((data_avg.shape[0], data_avg.shape[1]*ifactor))
        for j in range(data_avg.shape[0]):
            data_ipl[j] = sinc_interpolate(data_avg[j], ifactor)

        sim_ipl = np.empty((sim_avg.shape[0], sim_avg.shape[1]*ifactor))
        for j in range(sim_avg.shape[0]):
            sim_ipl[j] = sinc_interpolate(sim_avg[j], ifactor)

        if debug:
            # Show interpolated data
            fig, axs = plt.subplots(2, 1)
            im1 = axs[0].imshow(np.transpose(data_ipl), aspect='auto')
            axs[0].set_title('Interpolated Data')
            fig.colorbar(im1, ax=axs[0])
            im2 = axs[1].imshow(np.transpose(sim_ipl), aspect='auto')
            axs[1].set_title('Interpolated Simulation')
            fig.colorbar(im2, ax=axs[1])

            savefile = "data_sim_radargram_cosim_{:0.2f}.png".format(co_sim)
            logging.info("Saving " + savefile)
            fig.savefig(savefile)
            #plt.show()

        logging.debug("icd_ranging: Correlating interpolated arrays of length w_range={:d} and shape {:s}".format(\
                      w_range, str(sim_ipl[average:-average].shape)))
        # TODO: use scipy correlation function

        logging.debug("MAD from i={:0.3f} to i={:0.3f}".format(i_interp - 10, i_interp + 10))
        # Correlate interpolated array only around desired window.
        #p = icsim.prg.Prog(2*10) if do_progress else None
        mask = np.ones(2*w_range) # mask of items that have not been checked
        md1 = np.zeros(2*w_range)
        for i in range(-w_range, w_range, 1):
            if np.abs(i - i_interp) > ifactor: # 1 sample radius
                continue # Don't calculate for outside the window around the minimum
            md1[i + w_range] = mean_abs_diff_shifted(i, sim_ipl[average:-average], data_ipl[average:-average])
            mask[i + w_range] = 0
        md1 += mask*max(md) # set unchecked to max
        del data_ipl, sim_ipl
    else:
        #data_ipl = np.copy(data_avg)
        #sim_ipl = np.copy(sim_avg)
        md1 = md
        w_range = window
        p_step = 1


    logging.debug("icd_ranging: Finished correlating")

    if debug:
        fig, axs = plt.subplots(2, 1, figsize=(6,6))
        x = np.arange(-window, window, 1)
        axs[0].plot(x, md, lw=3)
        axs[0].set_xlabel('Offset (samples)')
        axs[0].set_ylabel('Residual (dB)')
        axs[0].set_title('Residual between data and sim')
        x = np.arange(-window, window, p_step)
        axs[1].plot(x, md1, lw=3)
        axs[1].set_xlabel('Offset (samples)')
        axs[1].set_ylabel('Residual (dB)')
        axs[1].set_title('Interpolated residual between data and sim')
        plt.show()

    # Use peakint to calculate sub-sample minimum
    xmin = np.argmin(md)
    p, y, _ = peakint.qint(-md, xmin)
    xmin += p

    # Convert range as number of samples (delta) to physical units (dz)
    delta = (-w_range + xmin)*p_step
    dz = delta/fs * scipy.constants.c /2
    logging.debug("Finish icd_ranging")
    return (delta, dz, y)


def shift_ft_arr(a, n):
    b = np.empty_like(a)
    #logging.debug("len(a)={:d}; len(n)={:d}".format(len(a), len(n)))
    assert a.shape[0] == len(n)
    for j in range(len(a)):
        b[j] = np.roll(a[j], int(n[j]))
    return b

def test_shift_ft_arr():
    x = np.random.rand(10, 11)
    a1 = np.full((10,), 5)
    a2 = np.full((10,), 6)
    y = shift_ft_arr(x, a1)
    z = shift_ft_arr(y, a2)
    assert (x == z).all()



def running_mean(x, N):
    # assert N >= 3
    res = np.zeros(len(x), dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0), dtype=x.dtype)
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

def test_running_mean():
    x1 = np.arange(10)
    x2 = np.flip(x1)
    for y in range(3, 8):
        z1 = running_mean(x1, y)
        z2 = running_mean(x2, y)
        assert np.abs(np.mean(z1) - np.mean(z2)) < 1e-6


def sinc_interpolate(data, subsample_factor):
    """ sinc interpolate a 1d vector. Assumes data is a real-valued vector.
    """
    #fft = np.fft.fft(data)
    fft_shift = np.fft.fftshift(np.fft.fft(data))
    x = int((len(data)*subsample_factor-len(data))/2)
    fft_int  = np.pad(fft_shift, (x, x), 'constant', constant_values=(0, 0))
    fft_int_shift = np.fft.fftshift(fft_int)
    return subsample_factor*np.abs(np.fft.ifft(fft_int_shift))

def mean_abs_diff_rolled(x, a, b):
    """ Compute the mean absolute difference (MAD) of two arrays
    while rolling the second array by x
    arrays a and b are complex arrays.

    effectively return np.mean(abs(a - np.roll(b, x, axis=1))?

    This is meant as an optimized, drop-in replacement for the above subtract(x, a, b) function
    """
    assert a.shape == b.shape
    roll_x = int(x)
    # simple case
    if roll_x == 0:
        return np.mean(abs(a - b))

    #----------------------------------------
    #diff_ab = a - shift_ft(b, x)
    sumdiff = 0.0
    #diff_ab = np.zeros(len(a))
    #len_a2 = a.shape[1]
    for j in range(a.shape[0]):
        # collapse rows to prevent reallocation
        #diff_ab[j] += np.mean(np.abs(a[j] - np.roll(b[j], roll_x)))
        sumdiff += np.sum(np.abs(a[j] - np.roll(b[j], roll_x)))
    #----------------------------------------
    #return np.mean(diff_ab)
    return sumdiff / float(a.shape[0]*a.shape[1])
 
def mean_abs_diff_shifted(x, a, b):
    """ Compute the mean absolute difference (MAD) of two arrays
    while shifting the second array by x in the 2nd dimension.
    arrays a and b are complex arrays.

    computes a value similar to, but not exactly, np.mean(abs(a - np.roll(b, x, axis=1))?

    This is meant as an optimized, slightly-modified replacement for the above subtract(x, a, b) function
    """
    assert a.shape == b.shape
    roll_x = int(x)
    # simple case
    if roll_x == 0:
        return np.mean(abs(a - b))

    i1 = max(0, roll_x)
    i2 = min(roll_x + b.shape[1], b.shape[1])
    blen = i2 - i1 # length of section of b to be used
    return np.mean(np.abs(a[:][0:blen] -  b[:][i1:i2]))


def main():
    test_running_mean()
    test_shift_ft_arr()
    #gen_or_load_cluttergram()

if __name__ == "__main__":
    main()

