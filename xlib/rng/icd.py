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
import scipy.constants
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import rng.icsim as icsim


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
                do_progress=False):

    """
    Computes the differential range between two tracks
    by matching incoherent cluttergrams with pulse compressed data.

    # TODO: document what the minimum window size is (min value of idx_end - idx_start)
    # TODO: rework naming and make more consistent cluttergram_path and save_clutter_path naming and mechanics

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

    """
    # SHARAD sampling frequency
    f = 1/0.0375E-6    

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
        sim = icsim.incoherent_sim(state, rxwot, pri_code, dtm_path, idx_start, idx_end, do_progress=do_progress).transpose()
        if save_clutter_path is not None: 
            logging.debug("Saving cluttergram to " + save_clutter_path)
            np.save(save_clutter_path, sim)
    else:
        logging.debug("Loading cluttergram from " + cluttergram_path)
        sim = np.load(cluttergram_path)

    plt.imshow(sim)
    plt.show()

    # Pulse compressed radargrams
    re = pd.read_hdf(cmp_path, key='real').values[idx_start:idx_end]
    im = pd.read_hdf(cmp_path, key='imag').values[idx_start:idx_end]
    power_db = 20*np.log10(np.abs(re+1j*im)+1E-3)
    """
    # Adjust indexes 
    #TODO: Check if that is necessary
    if len(sim) != idx_end-idx_start:
        # Increase the window by increasing the end , if we have enough data available,
        # otherwise increase it by moving the beginning toward 0
        # This seems to assume that if len(sim) is not equal to idx_end - idx_start,
        # it will be less than idx_end
        if len(data)>(idx_end+1):
            logging.debug("Adjusting idx_end += 1")
            idx_end += 1
        else:
            logging.debug("Adjusting idx_start -= 1")
            idx_start -= 1
    """
    # can we assert len(sim) == idx_end - idx_start now?

    # Free memory - science data not needed anymore
    del data
    
    try:
        #assert 0 < power_db.shape[0] <= len(rxwot)
        #assert 0 < sim.shape[0] <= len(rxwot)
        # Cut to 800 samples.
        data_new = shift_ft_arr(power_db[:, 0:800], rxwot)
        sim_new = shift_ft_arr(sim[:, 0:800], rxwot)
    except IndexError as e:
        logging.error("size of power_db: " + str(power_db.shape))
        logging.error("size of sim: " + str(sim.shape))
        logging.error("rxwot shape: " + str(rxwot.shape))
        logging.error("rxwot: " + str(rxwot))
        raise(e)
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
    md = np.empty(2*window)
    for j, i in enumerate(range(-window, window, 1)):
        if p:
            p.print_Prog(j)
        #md[j] = subtract(i, sim_ipl[average:-average], data_ipl[average:-average])
        md[j] = mean_abs_diff_rolled(i, sim_avg[average:-average], data_avg[average:-average])
    if p:
        print(" ")





    # Interpolate radar records in fast time
    if ipl:

        # TODO: only interpolate in the region around the minimum.
        logging.debug("icd_ranging: Interpolating")
        data_ipl = np.empty((data_avg.shape[0], data_avg.shape[1]*10))
        for j in range(data_avg.shape[0]):
            data_ipl[j] = sinc_interpolate(data_avg[j], 10)*10
            
        sim_ipl = np.empty((sim_avg.shape[0], sim_avg.shape[1]*10))
        for j in range(sim_avg.shape[0]):
            sim_ipl[j] = sinc_interpolate(sim_avg[j], 10)*10
        w_range = 10*window
        p_step = 0.1
        if debug:
            # Show interpolated data
            plt.imshow(np.transpose(data_ipl), aspect='auto')
            plt.colorbar()
            plt.show()
            plt.imshow(np.transpose(sim_ipl), aspect='auto')
            plt.colorbar()
            plt.show()

        logging.debug("icd_ranging: Correlating interpolated arrays of length w_range={:d} and shape {:s}".format(\
                      w_range, str(sim_ipl[average:-average].shape)))

        # Calculate the index of best match from the correlation, once we upsample
        i_interp = (-window + np.argmin(md))*10

        logging.debug("MAD from i={:d} to i={:d}".format(i_interp - 10, i_interp + 10))
        # Correlate interpolated array only around desired window.
        #p = icsim.prg.Prog(2*10) if do_progress else None
        md = np.ones(2*w_range)*1e99 # Set correlations to un-calculated offsets to a big number
        for i in range(-w_range, w_range, 1):
            if np.abs(i - i_interp) > 10: # 1 sample radius
                continue # Don't calculate for outside the window around the minimum
            #md[i+w_range] = subtract(i, sim_ipl[average:-average], data_ipl[average:-average])
            md[i + w_range] = mean_abs_diff_rolled(i, sim_ipl[average:-average], data_ipl[average:-average])
        del data_ipl, sim_ipl
    else:
        #data_ipl = np.copy(data_avg)
        #sim_ipl = np.copy(sim_avg)
        w_range = window
        p_step = 1


    logging.debug("icd_ranging: Finished correlating")

    if debug:
        fig, ax = plt.subplots(figsize=(6,6))  
        ax.set_xlabel('Sample')
        ax.set_ylabel('Residual')
        plt.plot(np.arange(-w_range, w_range, p_step), md, lw=3)
        plt.show()

    # Convert range as number of samples to physical units (dz)
    delta = (-w_range + np.argmin(md))*p_step
    dz = delta/f * scipy.constants.c /2
    logging.debug("Finish icd_ranging")
    return [delta, dz, min(md)]



def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    ir = np.abs(np.fft.ifft2((f0 * f1.conjugate()) / (np.abs(f0) * np.abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]

def shift_ft_arr(a, n):
    b = np.empty_like(a)
    #logging.debug("len(a)={:d}; len(n)={:d}".format(len(a), len(n)))
    assert a.shape[0] == len(n)
    for j in range(len(a)):
        b[j] = np.roll(a[j], int(n[j]))
    return b

def test_shift_ft_arr():
    logging.warning("TODO") # TODO


def shift_ft(a, n):
    b = np.empty_like(a)
    n1 = int(n)
    for j in range(len(a)):
        b[j] = np.roll(a[j], n1)
    return b

def test_shift_ft():
    logging.warning("TODO") # TODO


def running_mean(x, N):
    res = np.zeros(len(x), dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0),dtype=x.dtype) 
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

def test_running_mean():
    logging.warning("TODO") # TODO



def sinc_interpolate(data, subsample_factor):
    """ sinc interpolate a 1d vector. Assumes data is a real-valued vector.
    """
    fft = np.fft.fft(data)
    fft_shift = np.fft.fftshift(fft)
    x = int((len(data)*subsample_factor-len(data))/2)
    fft_int  = np.pad(fft_shift, (x, x), 'constant', constant_values=(0, 0))
    fft_int_shift = np.fft.fftshift(fft_int)
    return abs(np.fft.ifft(fft_int_shift))

def subtract(x, a, b):
    roll_x = int(x)    
    #diff = np.zeros_like(a)
    if roll_x == 0:
        diff = a - b
    else:
        diff = a - shift_ft(b, x)

    res = np.mean(abs(diff)) 
    return res
 


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
 
