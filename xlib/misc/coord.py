"""

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
    if (isinstance(coord, (list, tuple, set))):
        ncoord = np.array(coord)
    elif (isinstance(coord, pd.core.frame.DataFrame)):
        if (set(cols).issubset(set(coord.keys()))):
            ncoord = coord[cols].values
        else:
            print('ERROR: Wrong column names ' +
                  str(cols) + ' for pandas DataFrame object with keys ' +
                  str(list(coord.keys()))+'.')
            return None
    else:
        ncoord = np.array(coord)
    if (len(ncoord.shape) == 1):
        ncoord = ncoord[np.newaxis]
    return ncoord


def ellipsoid(sph, axes=[0, 0, 0], radius=0, indeg=True, unit=1, **kwargs):
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
    [a, b, c] = axes
    abc = 1000/unit * a*b*c
    if (abc == 0):
        if (radius == 0):
            return np.zeros(len(sph))
        else:
            return np.full(len(sph), 1000/unit * radius)
    else:
        scl = 1.0
        if (indeg):
            scl = deg
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
             cols=['spot_lat', 'spot_lon', 'spot_radius'], **kwargs):
    """
    Coverts spherical to catresian coordinates.

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
    scl = 1.0
    if (indeg):
        scl = deg
    clat = ncos(sph[:, 0]*scl)
    slat = nsin(sph[:, 0]*scl)
    clon = ncos(sph[:, 1]*scl)
    slon = nsin(sph[:, 1]*scl)
    r = sph[:, 2] + ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    cart[:, 0] = r * clon * clat
    cart[:, 1] = r * slon * clat
    cart[:, 2] = r * slat
    return cart


def cart2sph(coord, indeg=True,
             cols=['spot_x', 'spot_y', 'spot_z'], **kwargs):
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
    scl = 1.0
    if (indeg):
        scl = deg
    sph[:, 2] = nsqrt(cart[:, 0]**2 + cart[:, 1]**2 + cart[:, 2]**2)
    sph[:, 0] = nasin(cart[:, 2]/sph[:, 2])/scl
    sph[:, 1] = ((natan2(cart[:, 1], cart[:, 0]) + 2*pi) % (2*pi))/scl
    sph[:, 2] = sph[:, 2] - ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    return sph


def sph2lsh(coord, Dtm, indeg=True,
            cols=['spot_lat', 'spot_lon', 'spot_radius'], **kwargs):
    sph = _coord_check(coord, cols)
    lsh = np.empty_like(sph, dtype=float)
    scl = 1.0
    if not(indeg):
        scl = deg
    # carto routine wants degrees for the coordinates
    lsh[:, 0:2] = Dtm.carto(sph[:, 0:2]/scl, -1)
    # the 3rd component is kept as it is
    lsh[:, 2] = sph[:, 2]  # + ellipsoid(sph[:, 0:2], indeg=indeg, **kwargs)
    return lsh


def lsh2sph(coord, Dtm, indeg=True,
            cols=['line', 'sample', 'height'], **kwargs):
    lsh = _coord_check(coord, cols)
    scl = 1.0
    if not(indeg):
        scl = deg
    sph = np.empty_like(lsh, dtype=float)
    # carto routine gives degrees for coordinates
    sph[:, 0:2] = Dtm.carto(lsh[:, 0:2], 1)*scl
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


def latlon_profiles(prof, lat=[-90, 90], lon=[0, 360], min_len=0,
                    verbose=False):
    import numpy as np
    min_lat = min(lat)
    max_lat = max(lat)
    min_lon = min(lon)
    max_lon = max(lon)
    cut_prof = []
    if (verbose):
        import pydlr.misc.prog as pr
        p = pr.Prog(len(prof))
    for i in range(len(prof)):
        ind = np.where((prof[i][:, 0] <= max_lat) &
                       (prof[i][:, 0] >= min_lat) &
                       (prof[i][:, 1] <= max_lon) &
                       (prof[i][:, 1] >= min_lon))[0]
        if (len(ind) > min_len):
            cut_prof.append(prof[i][ind])
            if (verbose):
                p.print_Prog(i)
    if (verbose):
        p.close_Prog()
    return np.asarray(cut_prof)


def sph_dist(sph1, sph2, radius, indeg=True):
    import numpy as np
    if (indeg):
        sph1 = sph1*deg
        sph2 = sph2*deg
    ds = np.arccos(np.sin(sph1[:, 0]) * np.sin(sph2[:, 0]) +
                   np.cos(sph1[:, 0]) * np.cos(sph2[:, 0]) *
                   np.cos(sph1[:, 1] - sph2[:, 1]))
    return radius*ds
