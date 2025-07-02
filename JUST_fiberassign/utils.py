import logging as log
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.spatial import KDTree


def get_fiberpos():
    ifile_fiberpos = "/home/zjding/fiberassignment/JUST/modify_focalplane/just-fiberpos.txt"
    _,_,_, fp_x, fp_y, fp_z = np.loadtxt(ifile_fiberpos, skiprows=1, unpack=True)
    fiber_centers = np.vstack([fp_x, fp_y]).T
    return fiber_centers

_platescale = None
def load_platescale():
    """Loads platescale.txt.

    Returns
    -------
    :class:`~numpy.recarray`
        The data table read from the file.

    Notes
    -----
    The returned object has these columns:

    radius
        Radius from center of focal plane [mm].

    theta
        Radial angle that has a centroid at this radius [deg].

    radial_platescale
        Meridional (radial) plate scale [um/arcsec].

    az_platescale:
        Sagittal (azimuthal) plate scale [um/arcsec].

    arclength:
        Unknown description.
    """
    global _platescale
    if _platescale is not None:
        return _platescale

    #infile = findfile("focalplane/platescale.txt")
    infile = "/home/zjding/.conda/envs/desi_fiberassign/data/focalplane/platescale.txt"
    columns = [
        ("radius", "f8"),
        ("theta", "f8"),
        ("radial_platescale", "f8"),
        ("az_platescale", "f8"),
        ("arclength", "f8"),
    ]
    try:
        _platescale = np.loadtxt(infile, usecols=[0, 1, 6, 7, 8], dtype=columns)
        log.debug("Loaded the platescale unexpectedly!!!!!!")  # ZD, 04-01-2025
    except (IndexError,ValueError):
        # - no "arclength" column in this version of desimodel/data
        # - Get info from separate rzs file instead

        _platescale = np.loadtxt(infile, usecols=[0, 1, 6, 7, 7], dtype=columns)
        #rzs = Table.read(findfile("focalplane/rzsn.txt"), format="ascii")
        rzs = Table.read("/home/zjding/.conda/envs/desi_fiberassign/data/focalplane/rzsn.txt", format="ascii")

        from scipy.interpolate import interp1d
        from numpy.lib.recfunctions import append_fields

        arclength = interp1d(rzs["R"], rzs["S"], kind="quadratic")
        _platescale["arclength"] = arclength(_platescale["radius"])
        log.debug("Have loaded the platescale.")  # ZD, 04-01-2025

    return _platescale


def get_tile_radius_deg():
    '''Returns maximum radius in degrees covered by the outermost positioner.
    '''
    _tile_radius_deg = None
    if _tile_radius_deg is None:
        ##rmax = get_tile_radius_mm()
        rmax = 280.0   # ZD, maximum of fibercenter+radius, 279.2+0.6
        platescale = load_platescale()
        fn = interp1d(platescale['radius'], platescale['theta'], kind='quadratic')
        _tile_radius_deg = float(fn(rmax))
    return _tile_radius_deg

def _embed_sphere(ra, dec):
    """Embed `ra`, `dec` to a uniform sphere in three dimensions.
    """
    phi = np.radians(np.asarray(ra))
    theta = np.radians(90.0 - np.asarray(dec))
    r = np.sin(theta)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.cos(theta)
    return np.array((x, y, z)).T

def is_point_in_just(tiles_ra, tiles_dec, ra, dec, radius=None, return_tile_index=False):
    """If a point (`ra`, `dec`) is within `radius` distance from center of any
    tile, it is in DESI.

    Args:
        tiles (Table-like): The output of :func:`desimodel.io.load_tiles`, or
            a similar Table.
        ra (scalar or array-like): Right Ascension in degrees.
        dec (scalar or array-like): Declination in degrees.  The size of `dec`
            must match the size of `ra`.
        radius (float, optional): Tile radius in degrees;
            if `None` use :func:`desimodel.focalplane.get_tile_radius_deg`.
        return_tile_index (bool, optional): If ``True``, return the index of
            the nearest tile in tiles array.

    Returns:
        Return ``True`` if points given by `ra`, `dec` lie in the set of `tiles`.

    Notes:
        This function is optimized to query a lot of points.
    """
    #from scipy.spatial import cKDTree as KDTree
    #from .focalplane import get_tile_radius_deg

    if radius is None:
        radius = get_tile_radius_deg()

    tilecenters = _embed_sphere(tiles_ra, tiles_dec)
    tree = KDTree(tilecenters)
    # radius to 3d distance
    threshold = 2.0 * np.sin(np.radians(radius) * 0.5)
    xyz = _embed_sphere(ra, dec)
    if not xyz.flags['C_CONTIGUOUS']:
        xyz = xyz.copy()

    d, i = tree.query(xyz, k=1)

    ##injust = d < threshold    # ZD
    # ---- added by ZD to accommodate the special shape of JUST focalplane
    threshold_small = 2.0 * np.sin(np.radians(0.0867) * 0.5)  # Shark-Hartmann circle
    injust = (d < threshold)&(d > threshold_small)
    #tree = KDTree(tilecenters)
    
    # ---------------
    
    if return_tile_index:
        return injust, i
    else:
        return injust
    
def get_radius_mm(theta):
    """Returns an array of radii in mm given an array of radii in degrees using the platescale data
    relative to the center of the focal plane as (0,0). Supports scalar and vector inputs.

    Parameters
    ----------
    theta : :class:`float` or array-like
        An array that represents the angle from the center of the focal plane.

    Returns
    -------
    :class:`float` or array-like
        Radii in mm.
    """
    platescale = load_platescale()
    #platescale = np.loadtxt("/home/zjding/fiberassignment/JUST/modify_focalplane/platescale.txt", unpack=True)
    # Uses a quadratic one-dimensional interpolation to approximate the radius in degrees versus radius in mm
    fn = interp1d(platescale['theta'], platescale['radius'], kind = 'quadratic')
    radius = fn(theta)
    if(np.isscalar(theta)):
        return float(radius)
    else:
        return radius
    

def get_radius_deg(x, y):
    """Returns the radius in degrees given `x`, `y` coordinates using the
    platescale data.

    Parameters
    ----------
    x : :class:`float`
        The x coordinate in mm of a location on the focal plane
    y : :class:`float`
        The y coordinate in mm of a location on the focal plane

    Returns
    -------
    :class:`float`
        Radius corresponding to `x`, `y`.
    """
    #- support scalars, lists, and arrays
    if not np.isscalar(x):
        x = np.asarray(x)
    if not np.isscalar(y):
        y = np.asarray(y)

    radius = np.sqrt(x**2 + y**2)
    platescale = load_platescale()
    fn = interp1d(platescale['radius'], platescale['theta'],
                                    kind='quadratic')
    degree = fn(radius).astype(float)
    return degree

def xy2radec(telra, teldec, x, y):
    """Returns the new RA and Dec of an `x`, `y` position on the focal plane
    in the sky given an arbitrary telescope pointing in RA and Dec.

    Parameters
    ----------
    telra : :class:`float`
        The telescope's RA pointing in degrees.
    teldec : :class:`float`
        The telescope's Dec pointing in degrees.
    x : :class:`float`
        The x coordinate in mm of a location on the focal plane
    y : :class:`float`
        The y coordinate in mm of a location on the focal plane

    Returns
    -------
    tuple
        The RA, Dec corresponding to `x`, `y`.
    """
    # radial distance on the focal plane in radians
    r_rad = np.radians(get_radius_deg(x, y))

    # q signifies the angle the position makes with the +x-axis of focal plane
    q = np.degrees(np.arctan2(y, x))
    q_rad = np.radians(q)

    # Consider a unit sphere (x,y,z)
    # starting at (RA,dec) = (0,0) -> v0 = (1,0,0)
    # v0 = np.array([1.0, 0.0, 0.0])

    # The focal plane is oriented with +yfocal = +dec but +xfocal = -RA
    # Rotate clockwise around z by r_rad
    # zrotate = np.zeros(shape=(3,3))
    # zrotate[0] = [np.cos(r_rad), np.sin(r_rad), 0]
    # zrotate[1] = [-np.sin(r_rad), np.cos(r_rad), 0]
    # zrotate[2] = [0, 0, 1]
    # v1 = zrotate.dot(v0)

    x1 = np.cos(r_rad)      # y0=0 so drop sin(r_rad) term
    y1 = -np.sin(r_rad)     # y0=0 so drop cos(r_rad) term
    z1 = np.zeros_like(x1)

    # clockwise rotation around the x-axis
    # xrotate = np.zeros(shape=(3,3))
    # q_rad = np.radians(q)
    # xrotate[0] = [1, 0, 0]
    # xrotate[1] = [0, np.cos(q_rad), np.sin(q_rad)]
    # xrotate[2] = [0, -np.sin(q_rad), np.cos(q_rad)]

    x2 = x1
    y2 = y1*np.cos(q_rad)           # z1=0 so drop sin(q_rad) term
    z2 = -y1*np.sin(q_rad)          # z1=0 so drop cos(q_rad) term
    v2 = np.stack([x2, y2, z2])

    # Clockwise rotation around y axis by declination of the tile center
    decrotate = np.zeros(shape=(3,3))
    teldec_rad = np.radians(teldec)
    decrotate[0] = [np.cos(teldec_rad), 0, -np.sin(teldec_rad)]
    decrotate[1] = [0, 1, 0]
    decrotate[2] = [np.sin(teldec_rad), 0, np.cos(teldec_rad)]

    # Counter-clockwise rotation around the z-axis by the right ascension of the tile center
    rarotate = np.zeros(shape=(3,3))
    telra_rad = np.radians(telra)
    rarotate[0] = [np.cos(telra_rad), -np.sin(telra_rad), 0]
    rarotate[1] = [np.sin(telra_rad), np.cos(telra_rad), 0]
    rarotate[2] = [0, 0, 1]

    x3, y3, z3 = v3 = rarotate.dot(decrotate.dot(v2))

    ra_deg = np.degrees(np.arctan2(y3, x3)) % 360
    dec_deg = np.degrees((np.pi/2) - np.arccos(z3))

    return ra_deg, dec_deg

    
def radec2xy(telra, teldec, ra, dec):
    """Returns arrays of the x, y positions of given celestial objects
    on the focal plane given an arbitrary telescope pointing in RA and Dec and
    arrays of the `ra` and `dec` of celestial objects in the sky.

    Parameters
    ----------
    telra : :class:`float`
        The telescope's RA pointing in degrees.
    teldec : :class:`float`
        The telescope's Dec pointing in degrees.
    ra : array-like
        An array of RA values for locations in the sky.
    dec : array-like
        An array of Dec values for locations in the sky.

    Returns
    -------
    tuple
        The x, y positions corrsponding to `ra`, `dec`.

    Notes
    -----
    Implements the Haversine formula.
    """
    # Inclination is 90 degrees minus the declination in degrees
    dec = np.asarray(dec)
    inc = 90 - dec
    ra = np.asarray(ra)
    x0 = np.sin(np.radians(inc)) * np.cos(np.radians(ra))
    y0 = np.sin(np.radians(inc)) * np.sin(np.radians(ra))
    z0 = np.cos(np.radians(inc))
    coord = [x0, y0, z0]

    # Clockwise rotation around the z-axis by the right ascension of the tile center
    rarotate = np.zeros(shape=(3,3))
    telra_rad = np.radians(telra)
    rarotate[0] = [np.cos(telra_rad), np.sin(telra_rad), 0]
    rarotate[1] = [-np.sin(telra_rad), np.cos(telra_rad), 0]
    rarotate[2] = [0, 0, 1]

    # Counter-Clockwise rotation around y axis by declination of the tile center
    decrotate = np.zeros(shape=(3,3))
    teldec_rad = np.radians(teldec)
    decrotate[0] = [np.cos(teldec_rad), 0, np.sin(teldec_rad)]
    decrotate[1] = [0, 1, 0]
    decrotate[2] = [-np.sin(teldec_rad), 0, np.cos(teldec_rad)]

    coord1 = np.matmul(rarotate, coord)
    coord2 = np.matmul(decrotate, coord1)
    x = coord2[0]
    y = coord2[1]
    z = coord2[2]

    newteldec = 0
    newtelra = 0
    ra_rad = np.arctan2(y, x)
    dec_rad = (np.pi / 2) - np.arccos(z / np.sqrt((x**2) + (y**2) + (z**2)))
    radius_rad = 2 * np.arcsin(np.sqrt((np.sin((dec_rad - newteldec) / 2)**2) + ((np.cos(newteldec)) * np.cos(dec_rad) * (np.sin((ra_rad - newtelra) / 2)**2))))
    radius_deg = np.degrees(radius_rad)

    q_rad = np.arctan2(z, -y)

    radius_mm = get_radius_mm(radius_deg)

    x_focalplane = radius_mm * np.cos(q_rad)
    y_focalplane = radius_mm * np.sin(q_rad)

    return x_focalplane, y_focalplane

def plot_patrol_circle(center_x, center_y, ax):
    radius = 6.0
    num_points = 100

    # 生成圆周参数方程坐标
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)

    # 绘制
    ax.plot(x, y, 'r-', linewidth=0.5)
    

def get_spherearea(ramin, ramax, decmin, decmax):
    '''
    Input ra, dec range, output the inclosed area on a sphere.
    '''
    dphi = np.radians(ramax) - np.radians(ramin)   # dphsi
    dtheta = np.sin(np.radians(decmax)) - np.sin(np.radians(decmin))   # sintheta*dtheta
    area_deg = dphi*dtheta * (180./np.pi)**2
    return area_deg