#!/usr/bin/env python3
__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': '''Tool to identify orbits with specific parameters
                  based on datasets available in the USGEOM RDR
                  files'''}}

import csv
import numpy as np
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
    for line in content_file:
        filename = line.strip()
        with open(filename, 'r') as csvfile:
            OFile = csv.reader(csvfile, delimiter=',')
            for row in OFile:
                lat = float(row[2])
                lon = float(row[3])
                sza = float(row[8])
                if sza < minsza:
                    continue
                if not minlat <= lat < maxlat:
                    continue
                if not minlon <= lon < maxlon:
                    continue

                # Parse a date.
                # We can call solar_longitude.ISO8601_to_J2000 instead
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
                except (ValueError, IndexError):
                    second = 30
                MY, Ls = solar_longitude.Ls(year, month, day, hour, minute, second)
                if not minls <= Ls < maxls:
                    continue
                # This record matches. Print it and move to the next.
                print(filename)
                break
