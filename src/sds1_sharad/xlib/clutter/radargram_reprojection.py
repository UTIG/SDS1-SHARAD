__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'radargram reprojection algorithm'}}

'''
Attempt to make a off-track clutter discrimination algorithm. Based on
projecting radargram signals onto a DTM. DTM is derived from MOLA radius
measurements.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import radargram_reprojection_funclib as fl

sar_pth = '../../Other Investigations/SHARAD Specularity/Python/Arsia Mons/sar/'
sar_fn = '115m_11.7s_1.77s_5looks_e_1395201_001_ss19_700_a_a.h5'
raw_pth = '../../Other Investigations/SHARAD Specularity/Python/Arsia Mons/raw/'
raw_fn = 'e_1395201_001_ss19_700_a_a.h5'
DTM_pth = './Test Data/MOLA/'
DTM_fn = 'megr00n180hb.img'
echo_mode = 'manual'         # how to select echoes of interest
db_threshold = -1.0           # unused if echo_mode not 'threshold'
debug = False

# set paths
sar_fn = sar_pth + sar_fn; del sar_pth
raw_fn = raw_pth + raw_fn; del raw_pth
DTM_fn = DTM_pth + DTM_fn; del DTM_pth

# load the data
scradius = np.array(pd.read_hdf(raw_fn, 'scrad')) * 1000
latitude = np.array(pd.read_hdf(raw_fn, 'latitude'))
longitude = np.array(pd.read_hdf(raw_fn, 'longitude'))
pri = np.array(pd.read_hdf(raw_fn, 'pri'))
rx = np.array(pd.read_hdf(raw_fn, 'rxwot'))
sar_data = np.array(pd.read_hdf(sar_fn, 'long'))
sar_column = np.array(pd.read_hdf(sar_fn, 'trace'))
dtm = fl.load_mola(DTM_fn) + 3396E3

# restrict the raw datasets to the sar columns
latitude = latitude[sar_column[:, 0]]
longitude = longitude[sar_column[:, 0]]

# calculate distance to the start of the radargram
rx = fl.distance_to_radargram(pri, rx, scradius)
min_scradius = np.min(scradius)
scradius = scradius[sar_column[:, 0]]
rx = rx[sar_column[:, 0]]

# extract the DTM for the area covered by the range line
min_lat = np.floor(np.min(latitude)) - 1
max_lat = np.ceil(np.max(latitude)) + 1
min_lon = np.floor(np.min(longitude)) - 1
max_lon = np.ceil(np.max(longitude)) + 1
area = [min_lat, max_lat, min_lon, max_lon]; del min_lat, max_lat, min_lon, max_lon
dtm = fl.mola_area(dtm, [0, 180], 'h', area)

# extaract mola indices for each latitude and longitude position
nadir_idx = np.zeros((len(latitude), 2), dtype=int)
for ii in range(len(latitude)):
    nadir_idx[ii, :] = fl.extract_molaidx(dtm, area, [latitude[ii], longitude[ii]])
if debug:
    plt.figure()
    plt.imshow(dtm, cmap='gray')
    plt.colorbar()
    plt.scatter(nadir_idx[:, 1], nadir_idx[:, 0], s=0.5, c='k', marker='.')
    plt.title('sharad groundtrack overlain on mola surface')

# convert the radargram amplitudes to dB
dB_data = 20 * np.log10(sar_data)
if debug:
    plt.figure()
    plt.subplot(2, 1, 1); plt.imshow(np.transpose(sar_data), aspect='auto')
    plt.title('radargram in amplitude')
    plt.colorbar()
    plt.subplot(2, 1, 2); plt.imshow(np.transpose(dB_data), aspect='auto')
    plt.title('radargram in dB')
    plt.colorbar()

# define the echoes of interest
picks = fl.echo_select(dB_data, echo_mode, db_threshold)
if True:
    plt.figure()
    plt.imshow(np.transpose(dB_data))
    if echo_mode == 'maximum':
        plt.plot(picks, '.r')
    elif echo_mode != 'maximum':
        plt.imshow(np.transpose(picks))
    plt.title('radargram with picked echoes of interest')

clutter_latitude_idx = []
clutter_longitude_idx = []
pick_test = np.zeros((len(rx), 1), dtype=float)

# compare echo ranges with one-way distances between the spacecraft and the
# cross-track surface
#ii = 2500
#if ii == 2500:
#for ii in range(590, 600):
for ii in range(len(latitude)):

    # select out data related to an individual range line
    rl_scrad = scradius[ii, :]
    rl_lat = latitude[ii, :]
    rl_lon = longitude[ii, :]
    rl_idx = nadir_idx[ii, :]
    nadir_mola = fl.extract_mola(dtm, rl_lat, rl_lon, area)

    # check to see if column is empty (i.e. no radar data)
    testA = np.max(dB_data[ii, :] - np.max(dB_data[ii, :]))
    # check to make sure there is a picked surface
    if len(np.argwhere(picks[ii, :] == 1)) == 0:
        testB = False
    else:
        testB = True

    # proceed if the range line passes the test
    if np.isnan(testA) == False and testB:

        # convert the selected range samples with echoes to distance from the
        # spacecraft
        # need to account for the wrapping that occurs during sar focusing. The
        # the start of the range window in the focused data product does not
        # match the timing of the start of the range window in the raw data
        # product due to shifting required to correctly align traces
        wrap = np.min(rx) * 2 / 299792458 / 0.0375E-6 / 3600
        if echo_mode == 'maximum':
            pick_frac = picks[ii] / 3600
#            pick_delay = wrap + pick_frac
#            if pick_frac > (np.ceil(wrap) - wrap):
#                pick_delay = np.round((np.ceil(wrap) + pick_frac) * 3600)
#            else:
#                pick_delay = np.round((np.floor(wrap) + pick_frac) * 3600)
#            pick_delay = pick_delay * 3600 - 25
#            pick_test[ii] = pick_delay
#            echo = pick_delay * 0.0375E-6 * 299792458 / 2
        elif echo_mode != 'maximum':
            pick_frac = np.argwhere(picks[ii, :] == 1) / 3600
        pick_delay = wrap + pick_frac
        pick_delay = pick_delay * 3600 - 25
        echo = pick_delay * 0.0375E-6 * 299792458 / 2

        # define a perpendicular vector
        perp_idx, perp_latitude, perp_longitude = fl.xtrack_vector(nadir_idx, rl_idx, len(dtm), np.size(dtm, axis=1), area)
        if debug:
            plt.figure()
            plt.imshow(dtm, cmap='gray')
            plt.colorbar()
            plt.scatter(nadir_idx[:, 1], nadir_idx[:, 0], s=0.5, c='k', marker='.')
            plt.scatter(perp_idx[:, 1], perp_idx[:, 0], s=0.5, c='r', marker='.')
            plt.title('groundtrack and cross-track vector on mola surface radius')

        # extract topography along the perpendicular groundtrack
        perp_mola = np.zeros(len(perp_idx), dtype=float)
        for jj in range(len(perp_mola)):
            perp_mola[jj] = dtm[perp_idx[jj, 0], perp_idx[jj, 1]]
        if debug:
            plt.figure()
            plt.plot(perp_mola)
            plt.title('cross-track mola radius [m]')

        # determine distance from the spacecraft to the perpendicular vector
        R0 = fl.sc2xtrack_distance(perp_mola, perp_latitude, perp_longitude, 
                                   rl_scrad, rl_lat, rl_lon)
        R0 = R0 - (rl_scrad - min_scradius)
        if debug:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(R0 / 1000)
            plt.title('distance from spacecraft to points on cross-track vector')
            plt.ylabel('km')
            plt.subplot(1, 2, 2)
            plt.imshow(dtm, cmap='gray')
            plt.scatter(nadir_idx[:, 1], nadir_idx[:, 0], s=0.5, c='k', marker='.')
            plt.scatter(perp_idx[:, 1], perp_idx[:, 0], s=0.5, c=R0/1000, marker='.')
            plt.colorbar()
            plt.title('groundtrack and distance from spacecraft to cross-track vector [km]')

        # identify the cross-track position whose distance to the spacecraft is
        # the most similar to the calculated echo distance
#        pick_test[ii] = np.min(np.abs(R0 - echo))
        for jj in range(len(echo)):
            ind = np.argwhere(np.abs(R0 - echo[jj]) == np.min(np.abs(R0 - echo[jj])))[:, 0]
            for kk in range(len(ind)):
                clutter_latitude_idx.append(perp_idx[ind[kk], 0])
                clutter_longitude_idx.append(perp_idx[ind[kk], 1])
            del ind
        if debug:
            plt.figure()
            plt.imshow(dtm, cmap='gray')
            plt.scatter(clutter_longitude_idx, clutter_latitude_idx, s=0.2, c='r', marker='*')

# overlay effective echo points that may end up being clutter onto MOC surface
# imagery
moc_nadir_idx = np.zeros((np.size(nadir_idx, axis=0), 2), dtype=int)
moc = fl.moc_area('Arsia Mons', area)
moc_clutter_longitude = np.multiply(clutter_longitude_idx, 2)
moc_clutter_latitude = np.abs(len(moc) - np.multiply(clutter_latitude_idx, 2))
moc_nadir_idx[:, 1] = np.multiply(nadir_idx[:, 1], 2)
moc_nadir_idx[:, 0] = np.abs(len(moc) - np.multiply(nadir_idx[:, 0], 2))
plt.figure()
plt.imshow(np.flipud(moc), origin='lower', cmap='gray')
plt.scatter(moc_nadir_idx[:, 1], moc_nadir_idx[:, 0], s=0.5, c='k', marker='.')
plt.scatter(moc_clutter_longitude, moc_clutter_latitude, s=0.2, c='y', marker='*')
plt.title('select echo ranges overlain on CTX imagery')
plt.figure()
plt.imshow(dtm)
plt.scatter(nadir_idx[:, 1], nadir_idx[:, 0], s=0.5, c='k', marker='.')
plt.scatter(clutter_longitude_idx, clutter_latitude_idx, s=0.2, c='m', marker='*')
plt.title('select echo ranges overlain on MOLA')
