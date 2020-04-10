#!/usr/bin/env python3
"""
Convert coordinates in a pandas dataframe
"""
import numpy as np
import pandas as pd
from scipy.constants import degree as deg
from math import pi as pi

__version__ = '1.0'
__all__ = ['ellipsoid', 'sph2cart', 'cart2sph', 'latlon_profiles']

norm = np.linalg.norm
nsin = np.sin
ncos = np.cos
nasin = np.arcsin
nacos = np.arccos
natan2 = np.arctan2
nsqrt = np.sqrt


def _coord_check(coord, cols):
    """ Check column names for correctness and convert to numpy array if needed. """
    if isinstance(coord, (list, tuple, set)):
        ncoord = np.array(coord)
    elif isinstance(coord, pd.core.frame.DataFrame):
        if set(cols).issubset(set(coord.keys())):
            ncoord = coord[cols].values
        else:
            print('ERROR: Wrong column names ' +
                  str(cols) + ' for pandas DataFrame object with keys ' +
                  str(list(coord.keys()))+'.')
            return None
    else:
        ncoord = np.array(coord)
    if len(ncoord.shape) == 1:
        ncoord = ncoord[np.newaxis]
    return ncoord


def ellipsoid(sph, axes=(0, 0, 0), radius=0, indeg=True, unit=1, **kwargs):
    """
    Computes ellipsoid radii at specified spherical coordinates.

    Parameters
    ----------
    sph: ndarray
         Array of latitude and longitude.
    axes: list, optional
        Axes of the ellipsoid expressed in km and given as [a, b, c].
    radius: double, optional
        Radius of the sphere expressed in km.
    indeg: boolean, optional
        Angles in sph expressed in degrees.
    unit: double, optional
        Unit for output array (1 = m, 1000 = km)

    Output
    ------
    Array with ellipsoid radii at specified latitudes and longitudes.
    In case of a sphere axes=[0, 0, 0] and radius != 0 an array with the radius
    is returned. If axes and radius are zero an array with zeros is returned.

    Examples
    --------
    >>> ellipsoid(sph, axes, indeg=True)

    """
    a, b, c = axes
    abc = 1000/unit * a*b*c
    if abc == 0:
        if radius == 0:
            return np.zeros(len(sph))
        else:
            return np.full(len(sph), 1000/unit * radius)
    else:
        scl = deg if indeg else 1.0
        sph = np.reshape(sph, (-1, 2))
        clat = np.cos(sph[:, 0]*scl)
        slat = np.sin(sph[:, 0]*scl)
        clon = np.cos(sph[:, 1]*scl)
        slon = np.sin(sph[:, 1]*scl)
        return abc/(np.sqrt((a*b*slat)**2 +
                            (b*c*clat*clon)**2 +
                            (a*c*clat*slon)**2))
    # endif some axis is zero


def sph2cart(coord, indeg=True,
             cols=('spot_lat', 'spot_lon', 'spot_radius'), **kwargs):
    """
    Coverts spherical to cartesian coordinates.

    Parameters
    ----------
    coord : array-like
        Array with spherical coordinates to be transformed.
    radius : float, optional
        Reference radius to be used for the third component (i.e. height) of
        the input array.
    axes : tuple of floats, optional
        Axes of ellipsoid. The third component (i.e. height) of the input array
        is considered to be expressed with respect to the ellipsoid defined by
        these axes.
    indeg : boolean, optional
        Angles to be expressed in degrees.
    cols : Names of columns containing the spherical coordinates. The sequence
           for cartesian coordinates is Latitude, Longitude, Radius/Height.
    **kwargs: keywords, optional
        Keywords for :func:`~coord.ellipsoid`
    Returns
    -------
    cart : ndarray
        Array with cartesian coordinates (x, y, z).
    """

    sph = _coord_check(coord, cols)
    cart = np.empty_like(sph, dtype=float)
    if indeg: # if in degrees, convert to radians
        sph1 = np.array(sph)
        sph1[:, 0] *= deg
        sph1[:, 1] *= deg
    else:
        sph1 = sph

    r = sph[:, 2] + ellipsoid(sph1[:, 0:2], indeg=False, **kwargs)
    clat = ncos(sph1[:, 0])
    #cart[:, 0] = r * clat
    #cart[:, 1] = cart[:, 0]
    #cart[:, 0] *= ncos(sph1[:, 1])
    #cart[:, 1] *= nsin(sph1[:, 1])
    cart[:, 0] = r * ncos(sph1[:, 1]) * clat
    cart[:, 1] = r * nsin(sph1[:, 1]) * clat
    cart[:, 2] = r * nsin(sph1[:, 0])

    return cart


def cart2sph(coord, indeg=True,
             cols=('spot_x', 'spot_y', 'spot_z'), **kwargs):
    """
    Converts cartesian to spherical coordinates.

    Parameters
    ----------
    coord: array-like
        Array with cartesian coordinates to be transformed.
    radius: float, optional
        Reference radius to be used for the third component (i.e. height) of
        the output array.
    axes: tuple of floats, optional
        Axes of ellipsoid. The third component (i.e. height) of the output
        array is expressed with respect to the ellipsoid defined by these axes.
    indeg: boolean, optional
        Angles to be expressed in degrees.
    cols: Names of columns containing the cartesian coordinates. The sequence
          for cartesian coordinates is x, y, z.
    **kwargs: keywords, optional
        Keywords for :func:`~pydlr.misc.coord.ellipsoid`
        The unit keyword is not applied to the radii, it is only used by the
        ellipsoid routine, i.e. when spherical coordinates over a reference
        surface are requested the user needs to match the unit of the reference
        surface (radius, axes) with the unit of the cartesian coordinates.

    Returns
    -------
    sph: ndarray
        Array with spherical coordinates (Latitude, Longitude, Radius/Height).

    Examples
    --------
    >>> cart = np.array([1, 1, 0], [0, 1, 1])
    >>> sph = cart2sph(cart, radius=0.5)
    """
    cart = _coord_check(coord, cols)
    sph = np.empty_like(cart, dtype=float)

    scale = deg if indeg else 1.0
    sph[:, 2] = nsqrt(cart[:, 0]**2 + cart[:, 1]**2 + cart[:, 2]**2)
    sph[:, 0] = nasin(cart[:, 2]/sph[:, 2])/scale
    sph[:, 1] = ((natan2(cart[:, 1], cart[:, 0]) + 2*pi) % (2*pi))/scale
    sph[:, 2] = sph[:, 2] - ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    return sph


def sph2lsh(coord, Dtm, indeg=True,
            cols=('spot_lat', 'spot_lon', 'spot_radius'), **kwargs):
    sph = _coord_check(coord, cols)
    lsh = np.empty_like(sph, dtype=float)

    scale = deg if indeg else 1.0
    # carto routine wants degrees for the coordinates
    lsh[:, 0:2] = Dtm.carto(sph[:, 0:2]/scale, -1)
    # the 3rd component is kept as it is
    lsh[:, 2] = sph[:, 2]  # + ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    return lsh


def lsh2sph(coord, Dtm, indeg=True,
            cols=('line', 'sample', 'height'), **kwargs):
    lsh = _coord_check(coord, cols)
    scale = deg if indeg else 1.0
    sph = np.empty_like(lsh, dtype=float)
    # carto routine gives degrees for coordinates
    sph[:, 0:2] = Dtm.carto(lsh[:, 0:2], 1)*scale
    # the 3rd component is kept as it is
    sph[:, 2] = lsh[:, 2]  # + ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    return sph


def lsh2cart(coord, Dtm, **kwargs):
    sph = lsh2sph(coord, Dtm, **kwargs)
    sph[:, 2] = sph[:, 2] + ellipsoid(sph[:, 0:2], **kwargs)
    return sph2cart(sph, **kwargs)


def cart2lsh(coord, Dtm, **kwargs):
    # from cartesian coordinates
    kwargs.update({'indeg': True})
    sph = cart2sph(coord, **kwargs)
    lsh = np.empty_like(sph, dtype=float)
    # carto routine wants degrees for the coordinates
    lsh[:, 0:2] = Dtm.carto(sph[:, 0:2], -1)
    # the 3rd component is a radii
    lsh[:, 2] = sph[:, 2] + ellipsoid(sph[:, 0:2], **kwargs)
    return lsh


def latlon_profiles(prof, lat=(-90, 90), lon=(0, 360), min_len=0,
                    verbose=False):
    min_lat = min(lat)
    max_lat = max(lat)
    min_lon = min(lon)
    max_lon = max(lon)
    cut_prof = []
    if verbose:
        import pydlr.misc.prog as pr
        progress = pr.Prog(len(prof))
    for i in range(len(prof)):
        ind = np.where((prof[i][:, 0] <= max_lat) &
                       (prof[i][:, 0] >= min_lat) &
                       (prof[i][:, 1] <= max_lon) &
                       (prof[i][:, 1] >= min_lon))[0]
        if len(ind) > min_len:
            cut_prof.append(prof[i][ind])
            if verbose:
                progress.print_Prog(i)
    if verbose:
        progress.close_Prog()
    return np.asarray(cut_prof)


def sph_dist(sph1, sph2, radius, indeg=True):
    if indeg:
        sph1 = sph1*deg
        sph2 = sph2*deg
    ds = np.arccos(np.sin(sph1[:, 0]) * np.sin(sph2[:, 0]) +
                   np.cos(sph1[:, 0]) * np.cos(sph2[:, 0]) *
                   np.cos(sph1[:, 1] - sph2[:, 1]))
    return radius*ds


def test():
    # Test that functions are inverses of each other

    # Run with a variety of dimensions
    coords1 = []
    coords2 = []
    inc = 0.5
    for lat1 in np.arange(-90.0, 90.0, inc):
        for lon1 in np.arange(-180.0, 180.0, inc):
            coords1.append((lat1, lon1, 1000))
            coords2.append((lat1, lon1))
    print("calculating")

    coords1.reverse()
    coords3 = np.array(coords1)
    coords1.reverse()
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    radius = 12345.0

    edata = ellipsoid(coords2, radius=radius, unit=1.0)
    edata = ellipsoid(coords2, radius=radius, unit=1000.0)


    y = latlon_profiles(np.array([coords1,coords3]))
    epsa = 1e-5 # angle delta
    epsh = 1e-3 # height delta
    c1 = cart2sph(sph2cart(coords1))
    #c2 = cart2sph(sph2cart(coords2))
    assert (np.abs(c1[:, 0] - coords1[:, 0]) < epsa).all()
    da = c1[:, 1] - coords1[:, 1]
    da = (da + 180) % 360 - 180 # wrap deltas
    assert (np.abs(da) < epsa).all()
    assert (np.abs(c1[:, 2] - coords1[:, 2]) < epsh).all()

    #c1 = lsh2sph(sph2lsh(coords1))
    #c2 = cart2sph(sph2lsh(coords2))

    coords1_orig = coords1.copy()
    assert np.array_equal(coords1_orig, coords1)
    d1 = sph_dist(coords1, coords3, radius=radius, indeg=True)
    d1 = sph_dist(coords1, coords3, radius=radius, indeg=False)
    d1 = sph_dist(coords1, coords3, radius=radius, indeg=True)
    d1 = sph_dist(coords1, coords3, radius=radius, indeg=False)
    d1 = sph_dist(coords1, coords3, radius=radius, indeg=True)
    assert np.array_equal(coords1_orig, coords1)
    
def time_sph2cart(niter=100):
    import timeit
    # Run with a variety of dimensions
    coords1 = []
    coords2 = []
    inc = 0.2
    for lat1 in np.arange(-90.0, 90.0, inc):
        for lon1 in np.arange(-180.0, 180.0, inc):
            coords1.append((lat1, lon1, 1000))
            coords2.append((lat1*deg, lon1*deg, 1000))
    setupcode = """
from __main__ import sph2cart
import scipy.constants
import numpy as np
# Run with a variety of dimensions
coords1 = []
coords2 = []
inc = 0.2
for lat1 in np.arange(-90.0, 90.0, inc):
    for lon1 in np.arange(-180.0, 180.0, inc):
        coords1.append((lat1, lon1, 1000))
        coords2.append((lat1*deg, lon1*deg, 1000))

    """
    #et1 = timeit.repeat('sph2cart(coords1, indeg=True)', setup=setupcode, number=4)
    #et2 = timeit.repeat('sph2cart(coords2, indeg=False)', setup=setupcode, number=4)
    #print('degrees: avg={:0.3f} sec, min={:0.3f} sec'.format(np.mean(et1), np.min(et1)))
    #print('radians: avg={:0.3f} sec, min={:0.3f} sec'.format(np.mean(et2), np.min(et2)))

    print("Starting timing (degrees)")
    t0 = time.time()

    coords1 = np.array(coords1)
    for x in range(niter):
        c1 = sph2cart(coords1, indeg=True)

    print("Elapsed time {:0.2f}".format(time.time() - t0))
    print("Starting timing (radians)")
    t0 = time.time()

    coords2 = np.array(coords2)
    for x in range(niter):
        c2 = sph2cart(coords2, indeg=False)

    print("Elapsed time {:0.2f}".format(time.time() - t0))

    epsa = 1e-5 # angle delta
    epsh = 1e-3 # height delta
    assert (np.abs(c1[:, 0] - c2[:, 0]) < epsa).all()
    da = c1[:, 1] - c2[:, 1]
    da = (da + 180) % 360 - 180 # wrap deltas
    assert (np.abs(da) < epsa).all()
    assert (np.abs(c1[:, 2] - c2[:, 2]) < epsh).all()



def main():
    test()
    time_sph2cart(niter=30)


if __name__ == "__main__":
    # execute only if run as a script
    import os
    import time
    main()







