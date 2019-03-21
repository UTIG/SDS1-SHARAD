__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'function library for radargram reprojection'}}

import glob
import numpy as np
import pandas as pd


import sys
sys.path.append('/usr/local/anaconda3/lib/python3.5/site-packages/')
import pvl


def load_mola(pth):
    '''
    Python script for plotting MOLA .img data
    
    Inputs:
    -------------
    pth: path to image file

    Outputs:
    -------------
    pst_trc: matrix of MOLA topographies
    '''
    #import matplotlib.pyplot as plt
    
    label = pvl.load(pth.replace('.img','.lbl'))
    lon_samp = label['IMAGE']['LINE_SAMPLES']
    lat_samp = label['IMAGE']['LINES']
    dtype=[]
    for ii in range(lon_samp):
        dtype.append(('Line_'+str(ii), '>i2'))
    dtype = np.dtype(dtype)
    fil = glob.glob(pth)[0]
    arr = np.fromfile(fil, dtype=dtype)
    out = np.reshape(arr, [1, lat_samp])[0]
    dfr = pd.DataFrame(arr)      
    
    topo = np.zeros((lat_samp, lon_samp))
    for jj in range(lon_samp):
        topo[:,jj] = dfr[0:lat_samp]['Line_'+str(jj)].as_matrix()
    del dtype, arr, dfr, fil, out    
        
    return topo

def mola_area(data, top_left, resolution, lim):
    '''
    process for extracting mola data for a specific area within the specified
    data array

    Inputs:
    -------------
           data: mola data
       top_left: latitude and longitude of the top left corner of the mola data
     resolution: resolution of the input mola DTM
            lim: latitude and longitude bounds for the area of interest

    Outputs:
    -------------
         output is the mola array for the specified area
    '''
    import numpy as np

    # set MOLA resolution
    if resolution == 'c': ddeg = 4
    elif resolution == 'e': ddeg = 16
    elif resolution == 'f': ddeg = 32
    elif resolution == 'g': ddeg = 64
    elif resolution == 'h': ddeg = 128

    # set latitude and longitude bounds
    maxlat = top_left[0]
    minlat = maxlat - len(data) / ddeg
    minlon = top_left[1]
    maxlon = minlon + np.size(data, axis=1) / ddeg

    # set latitude and longitude vectors
    longitude = np.arange(minlon, maxlon, 1 / ddeg)
    latitude = np.flipud(np.arange(minlat, maxlat, 1 / ddeg))

    # find indices equal to desired corners
    a = np.argwhere(latitude == lim[0])[0, 0].astype(int)
    b = np.argwhere(latitude == lim[1])[0, 0].astype(int)
    c = np.argwhere(longitude == lim[2])[0, 0].astype(int)
    d = np.argwhere(longitude == lim[3])[0, 0].astype(int)

    # output the result
    out = data[b:a, c:d]

    return out

def extract_mola(data, lat, lon, bounds):
    '''
    extracting mola surface elevation at a defined latitude and longitude

    Inputs:
    -------------
        data: matrix of mola topography
         lat: latitude at which we want to extract surface heights
         lon: longitude at which we want to extract surface heights
      bounds: latitude and longitude bounds on the data matrix
              [min_lat, max_lat, min_lon, max_lon]

    Outputs:
    -------------
        topo: interpolated surface at the specified latitude and longitude
    '''

    import numpy as np

    minlat = bounds[0]
    maxlat = bounds[1]
    minlon = bounds[2]
    maxlon = bounds[3]

    # define latitude vector
    y = np.flipud(np.linspace(minlat, maxlat, np.size(data, axis=0)))
    x = np.linspace(minlon, maxlon, np.size(data, axis=1))

    # find lat/log positions surrounding the point of interest
    A = np.argmin(np.abs(y - lat))
    if y[A] - lat > 0: B = A - 1
    elif y[A] - lat < 0: B = A + 1
    else: B = A
    C = np.argmin(np.abs(x - lon))
    if x[C] - lon > 0: D = C - 1
    elif x[C] - lon < 0: D = C + 1
    else: D = C

    # weight surrounding points for topography at the point of interest
    if A != B and D != C:
        crnrAC = data[A, C]
        crnrAD = data[A, D]
        crnrBC = data[B, C]
        crnrBD = data[B, D]
        dA = np.abs(y[A] - lat) / y[1]
        dB = np.abs(y[B] - lat) / y[1]
        dC = np.abs(x[C] - lon) / x[1]
        dD = np.abs(x[D] - lon) / x[1]
        dcrnrAC = (1 - (np.sqrt(dA**2 + dC**2) / np.sqrt(2)))
        dcrnrAD = (1 - (np.sqrt(dA**2 + dD**2) / np.sqrt(2)))
        dcrnrBC = (1 - (np.sqrt(dB**2 + dC**2) / np.sqrt(2)))
        dcrnrBD = (1 - (np.sqrt(dB**2 + dD**2) / np.sqrt(2)))
        wAC = dcrnrAC / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wAD = dcrnrAD / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wBC = dcrnrBC / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        wBD = dcrnrBD / (dcrnrAC + dcrnrAD + dcrnrBC + dcrnrBD)
        topo = wAC * crnrAC + wAD * crnrAD + wBC * crnrBC + wBD * crnrBD
    elif A != B and D == C:
        crnrAC = data[A, C]
        crnrBC = data[B, C]
        dA = np.abs(y[A] - lat) / y[1]
        dB = np.abs(y[B] - lat) / y[1]
        dcrnrAC = (1 - dA)
        dcrnrBC = (1 - dB)
        wAC = dcrnrAC / (dcrnrAC + dcrnrBC)
        wBC = dcrnrBC / (dcrnrAC + dcrnrBC)
        topo = wAC * crnrAC + wBC * crnrBC
    elif A == B and D != C:
        crnrAC = data[A, C]
        crnrAD = data[A, D]
        dC = np.abs(x[C] - lon) / x[1]
        dD = np.abs(x[D] - lon) / x[1]
        dcrnrAC = (1 - dC)
        dcrnrAD = (1 - dD)
        wAC = dcrnrAC / (dcrnrAC + dcrnrAD)
        wAD = dcrnrAD / (dcrnrAC + dcrnrAD)
        topo = wAC * crnrAC + wAD * crnrAD
    else:
        topo = data[A, C]

    return topo

def extract_molaidx(dtm, area, point):
    '''
    extract the indices on the mola grid for a particular latitude and
    longitude location defined in the 'point' input

    Inputs:
    -------------
       dtm: matrix of mola topography
      area: latitude and longitude bounds for the area of interest
     point: latitude and longitude of the point for which we want the
            corresponding indices

    Outputs:
    -------------
      idx: index of the latitude and longitude corrdinates within the mola grid
    '''

    import numpy as np

    # define the latitude and longitude vector
    y = np.flipud(np.linspace(area[0], area[1], np.size(dtm, axis=0)))
    x = np.linspace(area[2], area[3], np.size(dtm, axis=1))

    # find the gridpoint closest to the input latitude and longitude
    # coordinates
    idx = np.zeros(2, dtype=int)
    idx[0] = np.min(np.argwhere(np.abs(y - point[0]) == np.min(np.abs(y - point[0]))))
    idx[1] = np.min(np.argwhere(np.abs(x - point[1]) == np.min(np.abs(x - point[1]))))

    return idx

def latlong_distance(point1, point2, radius):
    '''
    algorithm for calculating the distance between two lat/long positions.

    Inputs:
    -------------
      point1: latitude and longitude positions for point 1 [degrees]
      point2: latitude and longitude positions for point 2 [degrees]
      radius: radius of the body [m]

    Outputs:
    -------------
      output is the distance between the two points in meters
    '''

    import numpy as np
    import math

    # convert coordinates from degrees to radians
    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    dlat = math.radians(point2[0] - point1[0])
    dlon = math.radians(point2[1] - point1[1])
    
    # calculate the haversine
    a = math.pow(math.sin(dlat / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(math.sin(dlon / 2), 2)
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    
    return d

def moc_area(region, lim):
    '''
    process for extracting moc imagery for a particular area

    Inputs:
    -------------
        region: string defining the area of interest we want to extract imagery
                for
           lim: latitude and longitude coordinates for the area of interest

    Outputs:
    -------------
         output is the moc image array for the specified area
    '''
    import numpy as np
    from PIL import Image

    # details of the area
    if 'Arsia Mons' in region:
        img1 = Image.open('./Test Data/MOC/moc_256_N-30_210.png')
        img2 = Image.open('./Test Data/MOC/moc_256_N-30_240.png')
#        img1 = Image.open('moc_256_N-30_210.png')
#        img2 = Image.open('moc_256_N-30_240.png')
        img_bounds = [-30, 0, 210, 270]
        img = np.concatenate((img1, img2), axis=1)
        del img1, img2

    # define the latitude and longitude vectors for the image
    latitude = np.flipud(np.linspace(img_bounds[0], img_bounds[1], len(img)))
    longitude = np.linspace(img_bounds[2], img_bounds[3], np.size(img, axis=1))

    # find indices equal to desired corners
    a = np.argmin(np.abs(latitude - lim[0]))
    b = np.argmin(np.abs(latitude - lim[1]))
    c = np.argmin(np.abs(longitude - lim[2]))
    d = np.argmin(np.abs(longitude - lim[3]))

    # define the output
    moc = img[b:a, c:d]

    return moc

def xtrack_vector(nadir_idx, rl_idx, m, n, area):
    '''
    algorithm for defining a perpendicular cross-track vector at some point
    along a ground track

    Inputs:
    -------------
       nadir_idx: indices of the nadir points for the ground track within the
                  context of the 2D DTM grid used to derive topography
          rl_idx: indices of the narid point for the particular range line of
                  interest
               m: vertical (axis=0) dimension of the 2D DTM grid
               n: horizontal (axis=1) dimension of the 2D DTM grid
            area: latitude and longitude bounds for the 2D DTM grid

    Outputs:
    -------------
       perp_idx: indices for the perpendicular line within the dimensions of
                 the DTM grid
       perp_latitude: latitude coordinates for points on the perpendicular line
       perp_longitude: longitude coordinates for points on the perpendicular
                       line
    '''
    import numpy as np
    
    
    # define a vector perpendicular to the groundtrack at that location
    r = np.array(((0, -1),(1, 0)))
    temp_idx = np.zeros((len(nadir_idx), 2), dtype=int)
    temp_idx[:, 0] = nadir_idx[:, 0] - rl_idx[0]
    temp_idx[:, 1] = nadir_idx[:, 1] - rl_idx[1]
    perp_idx = np.matmul(temp_idx, r)
    perp_idx[:, 0] = perp_idx[:, 0] + rl_idx[0]
    perp_idx[:, 1] = perp_idx[:, 1] + rl_idx[1]
    
    # shift the perpendicular vector such that it's midpoint is located at
    # the range line of interest
    perp_idx_mid = perp_idx[int(np.round(len(perp_idx) / 2)), :]
    didx0 = perp_idx_mid[0] - rl_idx[0]
    didx1 = perp_idx_mid[1] - rl_idx[1]
    perp_idx[:, 0] = perp_idx[:, 0] - didx0
    perp_idx[:, 1] = perp_idx[:, 1] - didx1
    
    # restrict to the area covered by the DTM
    if nadir_idx[len(nadir_idx) - 1, 1] < nadir_idx[0, 1]:
        indA = np.min(np.argwhere(perp_idx[:, 1] <= 0))
        if np.max(perp_idx[:, 1]) > n:
            indB = np.max(np.argwhere(perp_idx[:, 1] >= n))
        else:
            indB = 0
    else:
        indA = np.max(np.argwhere(perp_idx[:, 1] <= 0))
        if np.max(perp_idx[:, 1]) > n:
            indB = np.min(np.argwhere(perp_idx[:, 1] >= n))
        else:
            indB = n
    indC = np.min([indA, indB])
    indD = np.max([indA, indB])
    perp_idx = perp_idx[indC + 1:indD, :]
    
    # for each point on the perpendicular groundtrack, extract a latitude
    # and longitude coordinate
    y = np.flipud(np.linspace(area[0], area[1], m))
    x = np.linspace(area[2], area[3], n)
    perp_latitude = np.zeros(len(perp_idx), dtype=float)
    perp_longitude = np.zeros(len(perp_idx), dtype=float)
    for jj in range(len(perp_latitude)):
        perp_latitude[jj] = y[perp_idx[jj, 0]]
        perp_longitude[jj] = x[perp_idx[jj, 1]]
        
    return perp_idx, perp_latitude, perp_longitude

def echo_select(dB_data, echo_mode, dB_threshold):
    '''
    algorithm for selecting which echoes we want to back map onto the DTM

    Inputs:
    -------------
            dB_data: sar-focused radargram array
          echo_mode: flag for how echoes of interest will be selected
                     -- maximum: select the maximum echo for each range line
                     -- threshold: select all indices where the range-line
                                   specific echo amplitudes [in dB] are above a
                                   defined threshold
                     -- manual: manually select echoes of interest by selecting
                                an upper and lower bound
       dB_threshold: minimum dB limit to use when selecting echoes of interest
                     under the 'threshold' mode

    Outputs:
    -------------
       picks: array of similar dimensions to the dB_data input with ones in
              indices where echoes of interest have been selected and nan
              everywhere else
    '''
    import numpy as np
    import interface_picker as ip

    if echo_mode == 'maximum':
        picks = np.zeros((len(dB_data), 1), dtype=float)
        # extract index with the maximum amplitude along each range line
        for ii in range(len(dB_data)):
            test = np.max(dB_data[ii, :] - np.max(dB_data[ii, :]))
            if np.isnan(test) == False:
                rl_data = dB_data[ii, :] - np.max(dB_data[ii, :])
                picks[ii] = np.argwhere(rl_data == 0)
            else:
                picks[ii] = np.nan
    elif echo_mode == 'threshold':
        picks = np.zeros((len(dB_data), np.size(dB_data, axis=1)), dtype=float)
        # extract index with amplitude above defined threshold along each line
        for ii in range(len(dB_data)):
            test = np.max(dB_data[ii, :] - np.max(dB_data[ii, :]))
            if np.isnan(test) == False:
                rl_data = dB_data[ii, :] - np.max(dB_data[ii, :])
                for jj in range(len(rl_data)):
                    if rl_data[jj] >= dB_threshold:
                        picks[ii, jj] = 1
                    else:
                        picks[ii, jj] = np.nan
    elif echo_mode == 'manual':
        picks = ip.picker(dB_data)
    
    return picks

def iau2000_ellipsoid_radius(lat):
    '''
    algorithm for determining martian IAU2000 ellipsoid radius at a particular
    latitude angle

    Inputs:
    -------------
       latitude: latitude at which we want to extract the martian ellipsoid
                 radius

    Outputs:
    -------------
      output is the radius of the martian ellipsoid [m]
    '''
    import numpy as np

    # set the radii for Mars
    a = 3396.19E3        # equatorial martian radius [m]
    if lat >= 0:
        b = 3373.19E3    # polar radius in the northern hemisphere [m]
    elif lat < 0:
        b = 3379.21E3    # polar radius in the southern hemisphere [m]
        lat = np.abs(lat)

    # define latitude in both degrees and radians
    lat_rad = np.flipud(np.linspace(0, np.pi / 2, 90 * 1000))
    lat_deg = np.flipud(np.linspace(0, 90, 90 * 1000))

    # define the radius as a function of latitude (absolute value)
    tempA = np.square(b) * np.square(np.cos(lat_rad))
    tempB = np.square(a) * np.square(np.sin(lat_rad))
    rad = (a * b) / np.sqrt(tempA + tempB)

    # find the index closest to the desired latitude
    ind = np.argwhere(np.abs(lat_deg - lat) == np.min(np.abs(lat_deg - lat)))
    rad = rad[ind]

    return rad

def mola2iautopo(dtm, lat_bounds):
    '''
    algorithm for creating a topographic map relative to the IAU2000 martian
    reference ellipsoid from a MOLA martian surface radii dtm

    Inputs:
    -------------
               dtm: MOLA martian surface radius dtm
        lat_bounds: lower (most southern) and upper (most northen) latitudes of
                    the MOLA dtm

    Outputs:
    -------------
      output is a topographic dtm normalized to the IAU2000 martian ellipsoid
    '''
    import numpy as np

    # define the latitude coordinates of the DTM grid
    lat = np.linspace(lat_bounds[0], lat_bounds[1], np.size(dtm, axis=0))

    # extract an IAU2000 ellipsoid radii for each latitude
    iau_rad = np.zeros((len(lat), 1), dtype=float)
    for ii in range(len(iau_rad)):
        iau_rad[ii, 0] = iau2000_ellipsoid_radius(lat[ii])

    # produce an ellipsoid map
    iau_map = np.tile(np.flipud(iau_rad), (1, np.size(dtm, axis=1)))
    
    # produce a difference map between the reference spheroid and the ellipsoid
    diff_map = 3396E3 - iau_map

    # create topographic map
    dtm = dtm + diff_map

    return dtm

def sc2xtrack_distance(perp_mola, perp_lat, perp_lon, rl_scrad, rl_lat, rl_lon):
    '''
    algorithm for determining the distance between the spacecraft and
    individual points on the cross-track vector

    Inputs:
    -------------
        perp_mola: radius from center-of-mass to each point on the cross-track
                   vector [m]
         perp_lat: latitude of each point on the cross-track vector [deg]
         perp_lon: longitude of each point on the cross-track vector [deg]
         rl_scrad: radius of the spacecraft from center-of-mass [m]
           rl_lat: sub-spacecraft latitude [deg]
           rl_lon: sub-spacecraft longitude [deg]

    Outputs:
    -------------
      output is a vector of distances between the spacecraft and points on the
      cross-track vector
    '''
    import numpy as np

    # define the surface along the perpendicular track in spherical
    # coordinates
    perp_sph = np.zeros((len(perp_mola), 3), dtype=float)
    perp_sph[:, 0] = perp_mola                        # radius [m]
    perp_sph[:, 1] = np.pi * ((90 - perp_lat) / 180)  # polar inclination [rad]
    perp_sph[:, 2] = 2 * np.pi * (perp_lon / 360)     # azimuth [rad]

    # define the spacecraft position in spherical coordinates
    sc_sph = np.zeros((1, 3), dtype=float)
    sc_sph[:, 0] = rl_scrad                       # radius [m]
    sc_sph[:, 1] = np.pi * ((90 - rl_lat) / 180)  # polar inclination [rad]
    sc_sph[:, 2] = 2 * np.pi * (rl_lon / 360)     # azimuth [rad]

    # for each point on the perpendicular, calculate the distance to the
    # spacecraft
    R0 = np.zeros(len(perp_sph), dtype=float)
    for jj in range(len(R0)):
        tempA = np.square(perp_sph[jj, 0])
        tempB = np.square(sc_sph[0, 0])
        tempC = 2 * perp_sph[jj, 0] * sc_sph[0, 0]
        tempD = np.sin(perp_sph[jj, 1]) * np.sin(sc_sph[0, 1]) * np.cos(perp_sph[jj, 2] - sc_sph[0, 2])
        tempE = np.cos(perp_sph[jj, 1]) * np.cos(sc_sph[0, 1])
        R0[jj] = np.sqrt(tempA + tempB - tempC * (tempD + tempE))

    return R0

def distance_to_radargram(pri, rx, scradius):
    '''
    algorithm for calculating the distance between platform and the start of
    the radargram. A correction is made for changes in spacecraft radius

    Inputs:
    -------------
           pri: array of pulse-repetition interval flags
            rx: array of receive window opening times [samples]
      scradius: array of MRO radii measured from the center of Mars [m]

    Outputs:
    -------------
      output is a vector of one-way distances between the spacecraft and the
      start of the radargram
    '''
    import numpy as np

    # calculate distances
    for ii in range(len(rx)):
        if pri[ii] == 1: val = 1428E-6
        elif pri[ii] == 2: val = 1429E-6
        elif pri[ii] == 3: val = 1290E-6
        elif pri[ii] == 4: val = 2856E-6
        elif pri[ii] == 5: val = 2984E-6
        elif pri[ii] == 6: val = 2580E-6
        else: val = 0
        rx[ii] = rx[ii] * 0.0375E-6 + val - 11.98E-6

    # convert to one-way distances
    out = (rx * 299792458 / 2) - (scradius - np.min(scradius))

    return out
