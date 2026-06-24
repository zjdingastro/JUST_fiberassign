import numpy as np
import pandas as pd
import logging as log
from pathlib import Path
from astropy.table import Table
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.io import fits as astropy_fits
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


def get_fiberpos():
    ##ifile_fiberpos = "/home/zjding/fiberassignment/JUST/modify_focalplane/just-fiberpos.txt"
    ##_,_,_, fp_x, fp_y, fp_z = np.loadtxt(ifile_fiberpos, skiprows=1, unpack=True)
    _pkg_root = Path(__file__).resolve().parent.parent
    ifile_fiberpos = _pkg_root / "parameters" / "fiberpos.csv"
    df = pd.read_csv(ifile_fiberpos, names=['radius', 'x', 'y'])
  
    fiber_centers = np.vstack([df['x'], df['y']]).T
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

    _pkg_root = Path(__file__).resolve().parent.parent
    infile = _pkg_root / "parameters" / "platescale.txt"
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
        rzs = Table.read(_pkg_root / "parameters" / "rzsn.txt", format="ascii")

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

    if radius is None:
        radius = get_tile_radius_deg()

    tilecenters = _embed_sphere(tiles_ra, tiles_dec)
    tree = cKDTree(tilecenters)
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

    theta = np.linspace(0, 2*np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)

    ax.plot(x, y, 'r-', linewidth=0.5)
    

def get_spherearea(ramin, ramax, decmin, decmax):
    '''
    Input ra, dec range, output the inclosed area on a sphere.
    '''
    dphi = np.radians(ramax) - np.radians(ramin)   # dphsi
    dtheta = np.sin(np.radians(decmax)) - np.sin(np.radians(decmin))   # sintheta*dtheta
    area_deg = dphi*dtheta * (180./np.pi)**2
    return area_deg


def mask_targets_in_tile(tile_ra, tile_dec, gal_ra, gal_dec, r_s, r_l):
    """ Output the mask of targets that are covered by a JUST tile.
    tile_ra: RA of one tile
    tile_dec: DEC of one tile
    gal_ra: RA of galaxy catalog
    gal_dec: DEC of galaxy catalog
    r_s: inner radius of the focalplane (unit: deg)
    r_l: outer radius of the focalplane (unit: deg)
    """

    tile_coord = SkyCoord(tile_ra, tile_dec, frame='icrs', unit='deg')
    gal_coord = SkyCoord(gal_ra, gal_dec, frame='icrs', unit='deg')
    sep = tile_coord.separation(gal_coord).to(units.deg)
    r_s_deg = r_s * units.deg
    r_l_deg = r_l * units.deg
    mask = (sep > r_s_deg)&(sep < r_l_deg)
    return mask

def find_targets_in_patrol_radius(fiber_ra, fiber_dec, targets_ra, targets_dec, r_patrol_deg):
    fiber_coord = SkyCoord(fiber_ra, fiber_dec, frame='icrs', unit='deg')
    targets_coord = SkyCoord(targets_ra, targets_dec, frame='icrs', unit='deg')
    sep = fiber_coord.separation(targets_coord).to(units.deg)
    mask = (sep<r_patrol_deg*units.deg)
    return mask

def find_neighboring_fibers(fiberpos_xy, patrol_center_separation=12.0):
    """Identify physically neighboring fiber pairs based on XY distance"""
    N_fibers = fiberpos_xy.shape[0]
    neighboring_pairs = []
    
    for i in range(N_fibers):
        for j in range(i+1, N_fibers):
            dx = fiberpos_xy[i, 0] - fiberpos_xy[j, 0]
            dy = fiberpos_xy[i, 1] - fiberpos_xy[j, 1]
            distance_mm = np.sqrt(dx*dx + dy*dy)
            
            if distance_mm < patrol_center_separation:
                neighboring_pairs.append((i, j))
    
    return neighboring_pairs


def _radec_deg_to_unit_xyz(ra_deg, dec_deg):
    """ICRS (RA, Dec) in degrees -> unit vectors on the celestial sphere."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def find_targets_in_one_tile(args):
    tile_id, tile_ra, tile_dec, gal_cat, fiberpos_xy, TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, r_patrol_deg, TARGETID = args

    targets_id_list_onetile = {}
    fibers_ra, fibers_dec = xy2radec(tile_ra, tile_dec, fiberpos_xy[:, 0], fiberpos_xy[:, 1])

    # Targets in the tile annulus (unchanged from the reference implementation).
    mask = mask_targets_in_tile(
        tile_ra, tile_dec, gal_cat["RA"], gal_cat["DEC"],
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG,
    )
    gal_in_tile = gal_cat[mask]
    if len(gal_in_tile) == 0:
        return f"tile_{tile_id}", targets_id_list_onetile

    targets_ra = np.asarray(gal_in_tile["RA"], dtype=np.float64)
    targets_dec = np.asarray(gal_in_tile["DEC"], dtype=np.float64)
    target_ids = np.asarray(gal_in_tile[TARGETID])

    # Vectorized patrol-radius test for all fibers at once (same sep < r_patrol rule).
    fibers_xyz = _radec_deg_to_unit_xyz(fibers_ra, fibers_dec)
    targets_xyz = _radec_deg_to_unit_xyz(targets_ra, targets_dec)
    cos_sep = np.clip(fibers_xyz @ targets_xyz.T, -1.0, 1.0)
    sep_deg = np.degrees(np.arccos(cos_sep))
    in_patrol = sep_deg < float(r_patrol_deg)

    for fiber_id in np.flatnonzero(in_patrol.any(axis=1)):
        idx = np.flatnonzero(in_patrol[fiber_id])
        targets_id_list = target_ids[idx].tolist()
        if len(targets_id_list) > 0:
            targets_id_list_onetile[f"fiber_{fiber_id}"] = targets_id_list

    return f"tile_{tile_id}", targets_id_list_onetile


## Compute collision constraints
## For each neighboring fiber pair, check which target pairs would collide
## If angular separation between two targets < 15.625 arcsec, they cannot both be assigned to neighboring fibers
def _angular_sep_arcsec_matrix(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Pairwise angular separation in arcsec; inputs shape (n,) and (m,) -> (n, m)."""
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=np.float64)).reshape(-1, 1)
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=np.float64)).reshape(-1, 1)
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=np.float64)).reshape(1, -1)
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=np.float64)).reshape(1, -1)
    sin_ddec = np.sin(0.5 * (dec2 - dec1))
    sin_dra = np.sin(0.5 * (ra2 - ra1))
    a = sin_ddec * sin_ddec + np.cos(dec1) * np.cos(dec2) * sin_dra * sin_dra
    sep_rad = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(sep_rad) * 3600.0


def _build_target_position_lookup(target_positions):
    """Sorted TARGETID -> (ra, dec) arrays for fast vectorized lookup."""
    n_pos = len(target_positions)
    target_ids = np.empty(n_pos, dtype=np.int64)
    ra_lookup = np.empty(n_pos, dtype=np.float64)
    dec_lookup = np.empty(n_pos, dtype=np.float64)
    for k, (tid, (ra, dec)) in enumerate(target_positions.items()):
        target_ids[k] = int(tid)
        ra_lookup[k] = ra
        dec_lookup[k] = dec
    order = np.argsort(target_ids)
    return target_ids[order], ra_lookup[order], dec_lookup[order]


def _positions_for_target_ids(target_ids, sorted_ids, ra_sorted, dec_sorted):
    ids = np.asarray(target_ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, ids)
    valid = (pos < len(sorted_ids)) & (sorted_ids[pos] == ids)
    pos = pos[valid]
    return ids[valid], ra_sorted[pos], dec_sorted[pos]


def find_collided_pairs_in_one_tile(args):
    """Compute collision constraints for a single tile, for parallelization."""
    (tile_id, tile_idx, tiles_ra, tiles_dec, targets_id_list_alltiles,
     neighboring_fiber_pairs, target_positions, fiberpos_xy,
     COLLISION_SEPARATION_ARCSEC) = args

    tile_targets = targets_id_list_alltiles[tile_id]
    if not tile_targets:
        return {}

    sorted_ids, ra_sorted, dec_sorted = _build_target_position_lookup(
        target_positions
    )
    sep_max = float(COLLISION_SEPARATION_ARCSEC)
    tile_collision_constraints = {}

    for fiber_i, fiber_j in neighboring_fiber_pairs:
        fiber_i_key = f"fiber_{fiber_i}"
        fiber_j_key = f"fiber_{fiber_j}"
        if fiber_i_key not in tile_targets or fiber_j_key not in tile_targets:
            continue

        ids_i, ra_i, dec_i = _positions_for_target_ids(
            tile_targets[fiber_i_key], sorted_ids, ra_sorted, dec_sorted
        )
        ids_j, ra_j, dec_j = _positions_for_target_ids(
            tile_targets[fiber_j_key], sorted_ids, ra_sorted, dec_sorted
        )
        if len(ids_i) == 0 or len(ids_j) == 0:
            continue

        sep = _angular_sep_arcsec_matrix(ra_i, dec_i, ra_j, dec_j)
        hit = sep < sep_max
        hit &= ids_i[:, None] != ids_j[None, :]

        ri, cj = np.nonzero(hit)
        if ri.size == 0:
            continue
        tile_collision_constraints[(tile_id, fiber_i, fiber_j)] = list(
            zip(ids_i[ri], ids_j[cj])
        )

    return tile_collision_constraints



def find_overlapped_tile_groups(tiles_ra, tiles_dec, radius_threshold_deg):
    """
    Find groups of overlapped tiles using KD-tree.
    
    Args:
        tiles_ra: array of tile RA coordinates
        tiles_dec: array of tile DEC coordinates
        radius_threshold_deg: distance threshold for considering tiles as overlapped
    
    Returns:
        list of lists: each inner list contains tile indices that form a group
    """
    n_tiles = len(tiles_ra)
    if n_tiles == 0:
        return []
    
    # Convert RA/DEC to 3D Cartesian coordinates for accurate spherical distance
    # Using astropy for proper celestial coordinate handling
    
    coords = SkyCoord(tiles_ra * units.deg, tiles_dec * units.deg, frame='icrs')
    xyz = np.column_stack([coords.cartesian.x, coords.cartesian.y, coords.cartesian.z])
    
    # Build KD-tree
    tree = cKDTree(xyz)
    
    # Calculate chord length threshold from angular separation
    # chord_length = 2 * sin(angle/2) for unit sphere
    max_sep_rad = np.deg2rad(radius_threshold_deg)
    chord_threshold = 2 * np.sin(max_sep_rad / 2)
    
    # Find all pairs within threshold
    pairs = tree.query_pairs(chord_threshold, p=2)
    
    # Build adjacency list for connected components
    adjacency = {i: set() for i in range(n_tiles)}
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Also add self-loops so isolated tiles form their own group
    for i in range(n_tiles):
        if i not in adjacency:
            adjacency[i] = set()
    
    # Find connected components using BFS
    visited = set()
    groups = []
    
    for start in range(n_tiles):
        if start in visited:
            continue
        # BFS to find all connected tiles
        group = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            group.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    
    return groups
    

def write_fba_onetile(
    tile_id,
    assigned_targets_id,
    assigned_fiber_id,
    targets_id_list_alltiles,
    out_fits_path=None,
    overwrite=True,
):
    """
    Write one tile's fiber-assignment result into a FITS file with two table HDUs.

    Parameters
    ----------
    tile_id : str
        Tile key, e.g. "tile_0".
    assigned_targets_id : array-like
        Assigned TARGETID values for this tile.
    assigned_fiber_id : array-like
        Fiber IDs matching `assigned_targets_id` one-to-one.
    targets_id_list_alltiles : dict
        Reachability dictionary, e.g. targets_id_list_alltiles[tile_id]["fiber_<id>"] -> [target ids].
    out_fits_path : str or None
        Output FITS path. If None, use "fba_<tile_id>.fits" in current directory.
    overwrite : bool
        Whether to overwrite existing file.
    """
    if len(assigned_targets_id) != len(assigned_fiber_id):
        raise ValueError("assigned_targets_id and assigned_fiber_id must have the same length")

    tile_id = str(tile_id)
    assigned_targets_id = np.asarray(assigned_targets_id, dtype=np.int64)
    assigned_fiber_id = np.asarray(assigned_fiber_id, dtype=np.int32)

    # HDU1: assigned targets on fibers for this tile.
    order_assigned = np.lexsort((assigned_targets_id, assigned_fiber_id))
    tbl_assign = Table(
        [assigned_targets_id[order_assigned], assigned_fiber_id[order_assigned]],
        names=["TARGETID", "FIBERID"],
    )

    # HDU2: all reachable TARGETID-FIBERID pairs for this tile.
    per_fiber = targets_id_list_alltiles.get(tile_id, {})
    reachable_pairs = []
    for fiber_key, tlist in per_fiber.items():
        if not str(fiber_key).startswith("fiber_"):
            continue
        try:
            fid = int(str(fiber_key).split("_", 1)[1])
        except Exception:
            continue
        for tid in tlist:
            reachable_pairs.append((int(tid), fid))

    reachable_pairs.sort(key=lambda x: (x[1], x[0]))
    if reachable_pairs:
        reach_tids = np.asarray([x[0] for x in reachable_pairs], dtype=np.int64)
        reach_fids = np.asarray([x[1] for x in reachable_pairs], dtype=np.int32)
    else:
        reach_tids = np.asarray([], dtype=np.int64)
        reach_fids = np.asarray([], dtype=np.int32)

    tbl_reach = Table([reach_tids, reach_fids], names=["TARGETID", "FIBERID"])

    if out_fits_path is None:
        safe_tile = tile_id.replace("/", "_")
        out_fits_path = f"fba_{safe_tile}.fits"

    primary_hdu = astropy_fits.PrimaryHDU()
    hdu_assign = astropy_fits.table_to_hdu(tbl_assign)
    hdu_assign.name = "ASSIGNED"
    hdu_reach = astropy_fits.table_to_hdu(tbl_reach)
    hdu_reach.name = "REACHABLE"

    astropy_fits.HDUList([primary_hdu, hdu_assign, hdu_reach]).writeto(
        out_fits_path, overwrite=overwrite
    )

    return out_fits_path
