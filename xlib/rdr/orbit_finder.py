__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Tool to identify orbits with specific parameters
                  based on datasets available in the USGEOM RDR
                  files'}}

import numpy as np
import csv
import solar_longitude

in_file = 'Cyril_RDRReferenceArea_hier.txt'
minlat = -84.5
maxlat = -81.5
minlon = 180
maxlon = 200
minls = 270
maxls = 360
minsza = 100

with open(in_file, 'r') as content_file:
    content = content_file.read()
content_file.closed

# extract
filename = ''
for line in content:
    if line != '\n':
        filename += line
    else:
        with open(filename) as csvfile:
            OFile = csv.reader(csvfile, delimiter=',')
            ind = 0
            for row in OFile:
                if float(row[8]) >= minsza:
                    if float(row[2]) >= minlat and float(row[2]) < maxlat:
                        if float(row[3]) >= minlon and float(row[3]) < maxlon:
                            date = str(row[1])
                            year = int(date[0:4])
                            month = int(date[5:7])
                            day = int(date[8:10])
                            hour = int(date[11:13])
                            minute = int(date[14:16])
                            try:
                                second = int(np.floor(float(date[17:23])))
                                if second == 60:
                                    minute = minute + 1
                                    second = 0
                            except:    
                                second = 30
                            MY, Ls = solar_longitude.Ls(year, month, day, hour, minute, second)
                            if Ls >= minls and Ls < maxls and ind == 0:
                                 ind += 1
                                 print(filename)
        filename = ''
    
