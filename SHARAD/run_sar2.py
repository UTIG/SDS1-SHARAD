__authors__ = ['Gregor Steinbruegge, gregor@ig.utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'September 13 2018',
         'author': 'Gregor Steinbruegge, UTIG',
         'info': 'First release.'},
    '2.0':
        {'date': 'October 23 2018',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Second release.'}}

def sar_processor(inputlist, posting, aperture, bandwidth, focuser='Delay Doppler',
                  recalc=20, Er=1, saving=True, debug=False):
    """
    Processor for individual SHARAD tracks. Intended for multi-core processing
    Takes individual range compressed and ionosphere corrected tracks and
    performs SAR focusing using the desired algorithm.
    Note: delay Doppler is much faster than matched filter and is set to be the
          default option

    Input:
    -----------
      inputlist : list of inputs required for SAR processing
                  - path to EDR
                  - idx_start
                  - idx_end
      posting   : SAR column posting interval [m]
      aperture  : length of the SAR aperture [s]
      bandwidth : Doppler bandwidth [Hz]
      focuser   : Flag for which SAR focuser to use.
      recalc    : Sample interval to recalculate the matched filter.
                  - only required for matched filter SAR processing
      Er        : relative dielectric permittivity of the subsurface
                  - only required for matched filter SAR processing and tests
                    show results are not strongly affected
      saving    : Flag to save the SAR output.
      debug     : Enter debug mode - more info.

    Output:
    -----------
      sar       : focused SAR data
      columns   : matrix of EDR range lines specifying the mid-aperture
                  position as well as the start and end of each aperture
    """
    import sys
    sys.path.insert(0, '../xlib/sar/')
    sys.path.insert(0, '../xlib/cmp/')
    import sar
    import numpy as np
    import pandas as pd
    import pds3lbl as pds3

    try:
    
        # split the path variable 
        idx_start = inputlist[1]
        idx_end = inputlist[2]
        path = inputlist[0]
        
        # print info in debug mode
        if debug:
            print('SAR method:', focuser)
            print('SAR column posting interval [m]:', posting)
            print('SAR aperture length [s]:', aperture)
            print('SAR Doppler bandwidth [Hz]:', bandwidth)
            print('SAR number of looks:', int(np.floor(aperture * 2 * bandwidth)))

        # create cmp path
        path_root = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'
        path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        data_file = path_file.split('/')[-1]
        path_file = path_file.replace(data_file,'')
        cmp_path = path_root+path_file+'ion/'+data_file.replace('_a.dat','_s.h5')
        label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
        aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
        science_path=path.replace('_a.dat','_s.dat')
        if debug:
            print('Loading cmp data from:', cmp_path)
    
        # load the range compressed and ionosphere corrected data
        real = np.array(pd.read_hdf(cmp_path, 'real'))	
        imag = np.array(pd.read_hdf(cmp_path, 'imag'))
        cmp_track = real + 1j * imag
        if idx_start is not None:
            idx_start = max(0,idx_start)
        else:
            idx_start = 0
        if idx_end is not None:
            idx_end = min(len(cmp_track),idx_end)
        else:
            idx_end = len(cmp_track)
        cmp_track = cmp_track[idx_start:idx_end]
        if debug:
            print('Start index of track to be processed:', idx_start)
            print('End index of track to be processed:', idx_end)
            print('Length of cmp track to be processed:', len(cmp_track))
        
        # load the relevant EDR files
        if debug:
            print('Loading science data from EDR file:', science_path)
        data = pds3.read_science(science_path, label_path, science=True, 
                                 bc=True)[idx_start:idx_end]
        if debug:
            print('Loading auxiliary data from EDR file:', 
                                science_path.replace('_s.dat', '_a.dat'))
        aux = pds3.read_science(science_path.replace('_s.dat','_a.dat'), 
                                aux_path, science=False, bc=False)[idx_start:idx_end]
        if debug:
            print('Length of selected EDR sci data:',len(data))
            print('Length of selected EDR aux data:', len(aux))
    
        # load relevant spacecraft position information from EDR files
        pri_code = data['PULSE_REPETITION_INTERVAL'].as_matrix()
        rxwot = data['RECEIVE_WINDOW_OPENING_TIME'].as_matrix()
        for j in range(len(pri_code)):                   
            if pri_code[j] == 1:   pri = 1428E-6
            elif pri_code[j] == 2: pri = 1429E-6
            elif pri_code[j] == 3: pri = 1290E-6
            elif pri_code[j] == 4: pri = 2856E-6
            elif pri_code[j] == 5: pri = 2984E-6
            elif pri_code[j] == 6: pri = 2580E-6
            else: pri = 0
            rxwot[j] = rxwot[j] * 0.0375E-6 + pri - 11.98E-6
        et = aux['EPHEMERIS_TIME'].as_matrix()
        tlp = data['TLP_INTERPOLATE'].as_matrix()
        scrad = data['RADIUS_INTERPOLATE'].as_matrix()
        if focuser == 'Delay Doppler':
            tpgpy = data['TOPOGRAPHY'].as_matrix()
            vt = data['TANGENTIAL_VELOCITY_INTERPOLATE'].as_matrix()
            vr = data['RADIAL_VELOCITY_INTERPOLATE'].as_matrix()
            v = np.zeros(len(vt), dtype=float)
            for j in range(len(vt)):
                v[j] = np.sqrt(vt[j]**2 + vr[j]**2)
        
        # correct the rx window opening times for along-track changes in spacecraft
        # radius
        rxwot = rxwot - (2 * (scrad - min(scrad)) * 1000 / 299792458)

        """
        sc = np.empty(len(ets))
        i = 0
        for et in ets:
            scpos, lt = spice.spkgeo(-74,et,'J2000',4)
            sc[i] = np.linalg.norm(scpos[0:3])
            i+=1
        """
        
        # execute sar processing
        if debug:
            print('Start of SAR processing') 
        if focuser == 'Delay Doppler':
            sar, columns = sar.delay_doppler(cmp_track, posting, aperture, bandwidth,
                                            tlp, et, scrad, tpgpy, rxwot - min(rxwot), v)
        elif focuser == 'Matched Filter':
            sar, columns = sar.matched_filter(cmp_track, posting, aperture, Er, bandwidth,
                                             recalc, tlp, et, rxwot)
        
        # save the result
        if saving:
            save_root = '/disk/kea/SDS/targ/xtra/SHARAD/foc/'
            path_file = science_path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
            data_file = path_file.split('/')[-1]
            path_file = path_file.replace(data_file,'')
            new_path = save_root + path_file
            new_path = new_path + str(posting) + 'm/'
            new_path = new_path + str(aperture) + 's/'
            new_path = new_path + str(bandwidth) + 'Hz/'
            if focuser == 'Matched Filter':
                new_path = new_path + str(Er) + 'Er/'

            if debug: 
                print('Saving to file:', new_path + data_file.replace('.dat','.h5'))

            if not os.path.exists(new_path):
                os.makedirs(new_path)
          
            # restructure and save data
            dfsar = pd.DataFrame(sar)
            dfcol = pd.DataFrame(columns)
            dfsar.to_hdf(new_path+data_file.replace('.dat','.h5'), key='sar',
                         complib = 'blosc:lz4', complevel=6)
            dfcol.to_hdf(new_path+data_file.replace('.dat','.h5'), key='columns',
                         complib = 'blosc:lz4', complevel=6)

    except Exception as e:
        
        logging.debug('Error in file:'+path+'\n'+str(e)+'\n')
        return 1
    
    logging.debug('Successfully processd file:'+path)
    
    return 0

import sys
sys.path.insert(0, '../xlib/misc')
import os
import numpy as np
import importlib.util
import pandas as pd
import multiprocessing
import time
import logging
import prog
import hdf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Set number of cores
nb_cores = 3

# set SAR processing variables
posting = 115
aperture = 11.7
bandwidth = 0.4
focuser = 'Delay Doppler'
recalc = 20
Er = 1.00

# Read lookup table associating gob's with tracks
#h5file = pd.HDFStore('mc11e_spice.h5')
#keys = h5file.keys()
#lookup = np.genfromtxt('lookup.txt',dtype='str')
lookup = np.genfromtxt('elysium.txt', dtype = 'str')
# Build list of processes
print('build task list')
process_list=[]
logging.basicConfig(filename='sar_crash.log',level=logging.DEBUG)

p = prog.Prog(len(lookup))
i=0
path_root = '/disk/kea/SDS/targ/xtra/SHARAD/foc/'

i = 0
if i == 0:
#for i in range(len(lookup)): 
#for orbit in keys:
    p.print_Prog(i)
#    gob = int(orbit.replace('/orbit', ''))
#    path = lookup[gob]
#    idx_start = h5file[orbit]['idx_start'][0]
#    idx_end = h5file[orbit]['idx_end'][0]
    path = lookup[i]
    
    # check if file has already been processed
    path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
    data_file = path_file.split('/')[-1]
    orbit_name = data_file[2:7]
    path_file = path_file.replace(data_file,'')
    new_path = path_root + path_file
    new_path = new_path + str(posting) + 'm/'
    new_path = new_path + str(aperture) + 's/'
    new_path = new_path + str(bandwidth) + 'Hz/'
    if focuser == 'Matched Filter':
        new_path = new_path + str(Er) + 'Er/'

    if not os.path.exists(new_path):
        if orbit_name == '05901':
            process_list.append([path, 78000, 141000])
        elif orbit_name == '10058':
            process_list.append([path, 30000, 62000])
        elif orbit_name == '16403':
            process_list.append([path, 38000, 80000])
        elif orbit_name == '17333':
            process_list.append([path, 39000, 71000])
        elif orbit_name == '17671':
            process_list.append([path, 43000, 75000])
        elif orbit_name == '23535':
            process_list.append([path, 43000, 75000])
        elif orbit_name == '26827':
            process_list.append([path, 43000, 75000])
        elif orbit_name == '27104':
            process_list.append([path, 43000, 75000])
        elif orbit_name == '32317':
            process_list.append([path, 3000, 32000])
        elif orbit_name == '50343':
            process_list.append([path, 13000, 45000])
        elif orbit_name == '50352':
            process_list.append([path, 13000, 45000])
        elif orbit_name == '50365':
            process_list.append([path, 13000, 45000])
        elif orbit_name == '50409':
            process_list.append([path, 13000, 45000])
        else:
            process_list.append([path, None, None])
        i+=1
    else:
        logging.debug('File already processed. Skipping:'+path)

p.close_Prog()
#h5file.close()
print('start processing',len(process_list),'tracks')

start_time = time.time()

pool = multiprocessing.Pool(nb_cores)
results = [pool.apply_async(sar_processor, (t,posting,aperture,bandwidth,focuser,recalc,Er),
                 {'saving':True,'debug':False}) for t in process_list]

#p=prog.Prog(len(keys))
p = prog.Prog(len(lookup))
i=0
for result in results:
    p.print_Prog(i)
    dummy = result.get()
    if dummy == 1: print('WARNING: Error in SAR processing - see logfile')
    i+=1

print('done in ', (time.time()-start_time), 'seconds')
p.close_Prog()

