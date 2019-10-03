uthors__ = ['Gregor Steinbruegge (UTIG), gregor@ig.utexas.edu']

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
sys.path.append('../../xlib')
import cmp.pds3lbl as pds3
from scipy.constants import c
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import rng.icsim as icsim

def icd_ranging(cmp_path, dtm_path, science_path, label_science, 
                idx_start, idx_end, debug = False, ipl = False,
                window = 50, average = 30, co_sim = 10, co_data = 30,
                cluttergram_path = None, save_clutter_path = None):

    """
    Computes the differential range between two tracks
    by matching inchorent cluttergrams with pulse compressed data.

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
        
    Output
    ------

    """
    # SHARAD sampling frequency
    f = 1/0.0375E-6    

    # Number of range lines
    Necho = idx_end-idx_start

    #============================
    # Read and prepare input data
    #============================
    
    # Data for RXWOTs
    data = pds3.read_science(science_path, label_science, science=True, 
                              bc=False)
    # Range window starts
    rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].values[idx_start:idx_end]

    # Perform clutter simulation or load existing cluttergram
    if cluttergram is None:
        pri_code = np.ones(Necho)
        p_scx = aux['X_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        p_scy = aux['Y_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        p_scz = aux['Z_MARS_SC_POSITION_VECTOR'].values[idx_start:idx_end]
        v_scx = aux['X_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        v_scy = aux['Y_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        v_scz = aux['Z_MARS_SC_VELOCITY_VECTOR'].values[idx_start:idx_end]
        state = np.vstack((p_scx, p_scy, p_scz, v_scx, v_scy, v_scz))
        sim = icsim.incoherent_sim(state, rxwot, pri_code, dtm_path, idx_start, idx_end)
        if save_clutter_path is not None: 
            np.save(save_clutter_path,sim)
    else:
        sim = np.load(cluttergram_path)

    plt.imshow(rgram)
    plt.show()

    #TODO: Check if that is necessary
    lsim = len(sim)
    if lsim != idx_end-idx_start:
        if len(data)>(idx_end+1): idx_end+=1
        else: idx_start-=1
    
    # Free memory - science data not needed anymore
    del data
    
    # Pulse compressed radargrams
    re = pd.read_hdf(cmp_path, key='real').values[idx_start:idx_end]
    im = pd.read_hdf(cmp_path, key='imag').values[idx_start:idx_end]
    power_db = 20*np.log10(np.abs(re+1j*im)+1E-3)
       
    # Cut to 800 samples.
    data_new = shift_ft_arr(power_db[:,0:800], rxwot)
    sim_new = shift_ft_arr(sim[:,0:800], rxwot)   
 
    # Perform correlation

    # Cut noise floor
    data_new[np.where(data_new < (np.max(data_new) - co_data))] = (np.max(data_new) - co_data)
    sim_new[np.where(sim_new < (np.max(sim_new) - co_sim))] = (np.max(sim_new) - co_sim)

    # Normalize
    data_norm = (data_new-np.min(data_new))/(np.max(data_new)-np.min(data_new))
    sim_norm = (sim_new-np.min(sim_new))/(np.max(sim_new)-np.min(sim_new))
    #sim_norm[np.where(sim_norm>0.1)] = 1
    #data_norm[np.where(data_norm>0.1)] = 1
    
    # Average
    data_avg = np.zeros_like(data_norm)
    for i in range(len(data_norm[0])):
        data_avg[:,i] = running_mean(data_norm[:,i],average)
        
    sim_avg = np.zeros_like(sim_norm)
    for i in range(len(sim_norm[0])):
        sim_avg[:,i] = running_mean(sim_norm[:,i],average)
    
    # Interpolate
    if ipl is True:
        data_ipl = np.zeros((len(data_avg),len(data_avg[0])*10))
        for j in range(0,len(data_norm)):
            data_ipl[j] = sinc_interpolate(data_avg[j], 10)*10
            
        sim_ipl = np.zeros((len(sim_avg),len(sim_avg[0])*10))
        for j in range(0,len(sim_avg)):
            sim_ipl[j] = sinc_interpolate(sim_avg[j], 10)*10
        w_range = 10*window
        p_step = 0.1
    else:
        data_ipl = np.copy(data_avg)
        sim_ipl = np.copy(sim_avg)
        w_range = window
        p_step = 1

    
    if debug:
        plt.imshow(np.transpose(data_ipl), aspect = 'auto')
        plt.colorbar()
        plt.show()
        plt.imshow(np.transpose(sim_ipl), aspect = 'auto')
        plt.colorbar()
        plt.show()
        
    # Correlate
    md = np.zeros(2*w_range)
    for i in range(-w_range,w_range,1):
        md[i+w_range] = subtract(i, sim_ipl[average:-average], data_ipl[average:-average])              

    if debug:
        fig, ax = plt.subplots(figsize=(6,6))  
        ax.set_xlabel('Sample')
        ax.set_ylabel('Residual')
        plt.plot(np.arange(-w_range,w_range,p_step),md, lw=3)
        plt.show()

    delta = (-w_range+np.argmin(md))*p_step
    dz = delta/f*c/2

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

def shift_ft_arr(a,n):
    b = np.empty_like(a)
    for j in range(len(a)):
        b[j]= np.roll(a[j],int(n[j]))
    return b

def shift_ft(a,n):
    b = np.empty_like(a)
    for j in range(len(a)):
        b[j]= np.roll(a[j],int(n))
    return b

def running_mean(x, N):
    res = np.zeros(len(x),dtype=x.dtype)
    cumsum = np.cumsum(np.insert(x, 0, 0),dtype=x.dtype) 
    res[N//2:-N//2+1] = (cumsum[N:] - cumsum[:-N]) / N
    return res

def sinc_interpolate(data,subsample_factor):
    fft = np.fft.fft(data)
    fft_shift = np.fft.fftshift(fft)
    x = int((len(data)*subsample_factor-len(data))/2)
    fft_int  = np.pad(fft_shift, (x,x), 'constant', constant_values=(0, 0))
    fft_int_shift = np.fft.fftshift(fft_int)
    return abs(np.fft.ifft(fft_int_shift))
 
def subtract(x,a,b):
    roll_x = int(x)    
    diff = np.zeros_like(a)
    if roll_x == 0:
        diff = a - b
    else:
        diff = a - shift_ft(b,x)

    res = np.mean(abs(diff)) 
    return res
 
