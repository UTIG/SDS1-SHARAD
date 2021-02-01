__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'RDR TECU extraction'}}
"""
Algorithm to walk through a set of SHARAD US RDR geom files and determine the
variability in TECU as a function of SZA for particular areas on Mars. Areas
are defined by latitutde and longitude bounds with some level of overlap if so
desired.
"""

import numpy as np
import csv
import os
import pandas as pd

# set path to the RDR geom files (.csv)
path_to_rdr = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/RDR/mrosh_2001/data/geom/'

# analysis parameters
latitude_width = 10           # latitude width of areas in degrees
longitude_width = 10          # longitude width of areas in degrees
latitude_overlap = 50         # % overlap in latitude between areas
longitude_overlap = 50        # % overlap in longitude between adjacent areas
full_area = [0, 70, 310, 360]  # latitude and longitude bounds of the full area of interest
sza_range = [60, 110]         # sza range of interest
dsza = 1.0                    # bin width to discretize sza range of interst

# set the centers of each area
dlat = latitude_width * (latitude_overlap / 100)
dlon = longitude_width * (longitude_overlap / 100)
latitude_centers = np.flipud(np.arange(full_area[0] + dlat, full_area[1], dlat))
longitude_centers = np.arange(full_area[2] + dlon, full_area[3], dlon)

# set the sza bin centers
sza_centers = np.arange(sza_range[0] + (dsza / 2), sza_range[1], dsza)

# define the output
number_TECU = np.zeros((len(sza_centers), len(longitude_centers), len(latitude_centers)), dtype=float)
mean_TECU = np.zeros((len(sza_centers), len(longitude_centers), len(latitude_centers)), dtype=float)
minimum_TECU = np.zeros((len(sza_centers), len(longitude_centers), len(latitude_centers)), dtype=float)
maximum_TECU = np.zeros((len(sza_centers), len(longitude_centers), len(latitude_centers)), dtype=float)
orbit_TECU = np.zeros((len(sza_centers), len(longitude_centers), len(latitude_centers)), dtype=float)

# walk through the rdr files extracting TEC files within the latitude,
# longitude and sza range being worked on

#ii = 11
#if ii == 11:
for ii in range(len(sza_centers)):
    minSZA = sza_centers[ii] - (dsza / 2)
    maxSZA = minSZA + dsza
#    jj = 19
#    if jj == 19:
    for jj in range(len(longitude_centers)):
        minlon = longitude_centers[jj] - (longitude_width / 2)
        maxlon = minlon + longitude_width
#        kk = 10
#        if kk == 10:
        for kk in range(len(latitude_centers)):
            minlat = latitude_centers[kk] - (latitude_width / 2)
            maxlat = minlat + latitude_width

            print('SZA:', minSZA, maxSZA, '- LONG:', minlon, maxlon, '- LAT:', minlat, maxlat)

            TECU = []
            unique_orbits = 0

            for root, dirs, files in os.walk(path_to_rdr):
                for file in files:
                    if file.endswith('.tab'):
                        filename = os.path.join(root, file)
                        old_TECU_length = len(TECU)
                        with open(filename) as csvfile:
                            OFile = csv.reader(csvfile, delimiter=',')
                            for row in OFile:
                                if float(row[8]) >= minSZA and float(row[8]) < maxSZA:
                                    if float(row[2]) >= minlat and float(row[2]) < maxlat:
                                        if float(row[3]) >= minlon and float(row[3]) < maxlon:
                                            TECU.append(float(row[9]) * 0.29)
                        if len(TECU) > old_TECU_length:
                            unique_orbits += 1

            if len(TECU) > 0:
                number_TECU[ii, jj, kk] = len(TECU)
                mean_TECU[ii, jj, kk] = np.mean(TECU)
                minimum_TECU[ii, jj, kk] = np.min(TECU)
                maximum_TECU[ii, jj, kk] = np.max(TECU)
                orbit_TECU[ii, jj, kk] = unique_orbits
            else:
                number_TECU[ii, jj, kk] = np.nan
                mean_TECU[ii, jj, kk] = np.nan
                minimum_TECU[ii, jj, kk] = np.nan
                maximum_TECU[ii, jj, kk] = np.nan
                orbit_TECU[ii, jj, kk] = unique_orbits

#print('#:', number_TECU[ii, jj, kk])
#print('Mean:', mean_TECU[ii, jj, kk])
#print('Minimum:', minimum_TECU[ii, jj, kk])
#print('Maximum:', maximum_TECU[ii, jj, kk])
#print('Orbits:', orbit_TECU[ii, jj, kk])

# save the results
if full_area[0] < 0:
    A = 'M' + str(np.abs(full_area[0]));
else:
    A = str(full_area[0])
if full_area[1] < 0:
    B = 'M' + str(np.abs(full_area[1]))
else:
    B = str(full_area[1])
basename = 'GlobalTECU_Lat' + A + '-' + B + '_Long' + str(full_area[2]) + '-' + str(full_area[3])
np.save(basename + '_Number.npy', number_TECU)
np.save(basename + '_Mean.npy', mean_TECU)
np.save(basename + '_Minimum.npy', minimum_TECU)
np.save(basename + '_Maximum.npy', maximum_TECU)
np.save(basename + '_Groundtracks.npy', orbit_TECU)







