__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Extractor of SHARAD USGEOM RDR datasets from a defined'
                 'list of orbits'}}
"""
Investigation into the local variability of SHARAD radiometric autofocus TECU
estimates as a function of SZA and solar longitude.

Loads RDR geometry files listed in the input text file and outputs desired
values within the given latitude range
--> tecu, sza, longitude, latitude, Martian year, solar longitude
"""

import numpy as np
import csv
import solar_longitude
import pandas as pd

in_file = '60to70_hier.txt'
minlat = 60
maxlat = 70

with open(in_file, 'r') as content_file:
    content = content_file.read()
content_file.closed

tecu = []         # row 9 of USGEOM SHARAD files
latitude = []     # row 2 of USGEOM SHARAD files
longitude = []    # row 3 of USGEOM SHARAD files
sza = []          # row 8 of USGEOM SHARAD files
date = []         # row 1 of USGEOM SHARAD files

# extract
filename = ''
for line in content:
    if line != '\n':
        filename += line
    else:
        with open(filename) as csvfile:
            OFile = csv.reader(csvfile, delimiter=',')
            for row in OFile:
                if float(row[2]) >= minlat and float(row[2]) < maxlat:
                    tecu.append(float(row[9]) * 0.29)
                    sza.append(float(row[8]))
                    #if float(row[8]) > 60 and float(row[8]) < 120:
                    #    print(str(row[1]))
                    latitude.append(float(row[2]))
                    longitude.append(float(row[3]))
                    date.append(str(row[1]))
        filename = ''
tecu = np.array(tecu)
sza = np.array(sza)
latitude = np.array(latitude)
longitude = np.array(longitude)

# time conversion
MY = np.zeros((len(tecu), ), dtype=int)
Ls = np.zeros((len(tecu), ), dtype=float)
#ii = 0
#if ii == 0:
for ii in range(len(tecu)):
    year = int(date[ii][0:4])
    month = int(date[ii][5:7])
    day = int(date[ii][8:10])
    hour = int(date[ii][11:13])
    minute = int(date[ii][14:16])
    try:
        second = int(np.floor(float(date[ii][17:23])))
        if second == 60:
            minute = minute + 1
            second = 0
    except:    
        second = 30
    MY[ii], Ls[ii] = solar_longitude.Ls(year, month, day, hour, minute, second)
    del year, month, day, hour, minute, second

#print('tecu:', len(tecu))
#print('sza:', len(sza))
#print('MY:', len(MY))
#print('Ls:', len(Ls))
#print('Latitude:', len(latitude))
#print('Longitude:', len(longitude))

# save
#df_tec = pd.DataFrame(tecu)
#df_sza = pd.DataFrame(sza)
#df_lat = pd.DataFrame(latitude)
#df_lon = pd.DataFrame(longitude)
#df_MY = pd.DataFrame(MY)
#df_Ls = pd.DataFrame(Ls)
#df_tec.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='tecu')#, complib='blosc:lz4', complevel=6)
#df_sza.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='sza')#, complib='blosc:lz4', complevel=6)
#df_lat.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='latitude')#, complib='blosc:lz4', complevel=6)
#df_lon.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='longitude')#, complib='blosc:lz4', complevel=6)
#df_MY.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='MY')#, complib='blosc:lz4', complevel=6)
#df_Ls.to_hdf(in_file.replace('_hier.txt', '_new.h5'), key='Ls')#, complib='blosc:lz4', complevel=6)
np.savez_compressed(in_file.replace('_hier.txt', ''), a=tecu, b=sza, c=latitude, d=longitude, e=MY, f=Ls)
#np.savez(in_file.replace('_hier.txt', '_new2'), a=tecu, b=sza, c=latitude, d=longitude, e=MY, f=Ls)

