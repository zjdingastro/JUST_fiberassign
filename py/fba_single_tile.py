import os
import healpy as hp
import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree
import networkx as nx
from parameters import (
    COLLISION_SEPARATION_ARCSEC,
    TILE_INNER_RADIUS_DEG,
    TILE_OUTER_RADIUS_DEG,
    R_PATROL_DEG)
from utils import (
    _log,
    mask_targets_in_tile,
    find_targets_in_one_tile,
    find_collided_pairs_in_one_tile
    )
from network_flow import (
    solve_tile_group,
    aggregate_group_assignments_with_pairwise_repair
    )
import network_flow as _network_flow_module
import contextlib


COST_OVERFLOW = 10000.0   # self defined, can be dependent on target priority
N_FIBERS = 2184

max_iterations=1   # does not matter for the debelended set of targets

def _fits_path_for_tile(out_dir, tile_id):
    return os.path.join(out_dir, f"fba_tile_{int(tile_id)}.fits")

def _healpix_ids_for_disc(ra_deg, dec_deg, radius_deg, nside, nest=False):
    """HEALPix pixel IDs that can contain sources within radius_deg of (ra, dec)."""
    vec = hp.ang2vec(ra_deg, dec_deg, lonlat=True)
    return np.unique(
        hp.query_disc(nside, vec, np.radians(radius_deg), nest=nest, inclusive=True)
    ).astype(np.int64)



def load_galaxies_from_mtlpix(
    ra_deg,
    dec_deg,
    odir_mtl,
    radius_deg,
    nside=32,
    nest=False,
):
    """
    Load galaxies from mtl_nside*/mtl_healpix_*.fits for HEALPix pixels
    overlapping a disc of radius_deg around (ra_deg, dec_deg).
    """
    pix_ids = _healpix_ids_for_disc(ra_deg, dec_deg, radius_deg, nside, nest=nest)
    cats = []
    for pix_id in pix_ids:
        fpath = os.path.join(odir_mtl, f"mtl_healpix_{int(pix_id):05d}.fits")
        if os.path.isfile(fpath):
            cats.append(Table.read(fpath))
    if not cats:
        return Table(
            names=["TARGETID", "RA", "DEC", "PRIORITY", "SUBPRIORITY", "HEALPIXID"],
            dtype=[np.int64, np.float64, np.float64, np.float64, np.float64, np.int64],
        )

    return vstack(cats, metadata_conflicts="silent")


def _degrade_mtl_priorities_on_disk(mtl_dir, assigned_target_ids, priority_degraded, healpix_ids):
    """Set PRIORITY=priority_degraded for assigned targets in HEALPix MTL FITS files."""
    if not assigned_target_ids:
        return 0

    n_updated = 0
    for pix_id in np.unique(healpix_ids):
        fpath = os.path.join(mtl_dir, f"mtl_healpix_{int(pix_id):05d}.fits")
        if not os.path.isfile(fpath):
            continue
        cat_pix = Table.read(fpath)
        mask = np.isin(cat_pix["TARGETID"], assigned_target_ids)
        if not np.any(mask):
            continue
        cat_pix["PRIORITY"][mask] = priority_degraded
        cat_pix.write(fpath, overwrite=True)
        n_updated += int(mask.sum())
    return n_updated

def _radec_to_unit_xyz(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def _weighted_independent_set_on_component(conflict_subgraph, node_weights):
    """
    Independent set on one conflict component.

    Lexicographic objectives (after separation constraint):
      1) maximize sum(priority + subpriority)
      2) tie-break by keeping more targets
    """
    nodes = sorted(conflict_subgraph.nodes)
    k = len(nodes)
    if k <= 1:
        return set(nodes)

    weights = np.asarray([float(node_weights[n]) for n in nodes], dtype=np.float64)

    if k <= 24:
        node_to_i = {node: i for i, node in enumerate(nodes)}
        adj_mask = [0] * k
        for u, v in conflict_subgraph.edges:
            i, j = node_to_i[u], node_to_i[v]
            adj_mask[i] |= 1 << j
            adj_mask[j] |= 1 << i

        best_score = (-1.0, -1)
        best_mask = 0
        for mask in range(1, 1 << k):
            bits = [i for i in range(k) if (mask >> i) & 1]
            independent = True
            for a, i in enumerate(bits):
                for j in bits[a + 1 :]:
                    if (adj_mask[i] >> j) & 1:
                        independent = False
                        break
                if not independent:
                    break
            if not independent:
                continue
            score = (float(weights[bits].sum()), len(bits))
            if score > best_score:
                best_score = score
                best_mask = mask
        return {nodes[i] for i in range(k) if (best_mask >> i) & 1}

    selected = set()
    for node in sorted(nodes, key=lambda n: node_weights[n], reverse=True):
        if not any(conflict_subgraph.has_edge(node, other) for other in selected):
            selected.add(node)
    return selected


def _priority_weighted_collision_free_target_ids(
    ra_deg, dec_deg, target_ids, priority_total, sep_arcsec_min
):
    """
    Select TARGETIDs with pairwise separation > sep_arcsec_min, favoring high
    priority+subpriority and then keeping as many targets as possible.
    """
    target_ids = np.asarray(target_ids, dtype=np.int64)
    priority_total = np.asarray(priority_total, dtype=np.float64)
    n = len(target_ids)
    if n == 0:
        return set()
    if n == 1:
        return {int(target_ids[0])}

    node_weights = {i: float(priority_total[i]) for i in range(n)}
    xyz = _radec_to_unit_xyz(ra_deg, dec_deg)
    tree = cKDTree(xyz)
    max_sep_rad = np.deg2rad(float(sep_arcsec_min) / 3600.0)
    chord_limit = 2.0 * np.sin(max_sep_rad / 2.0)
    conflict_pairs = list(tree.query_pairs(chord_limit, p=2))

    if not conflict_pairs:
        return {int(t) for t in target_ids}

    conflict_graph = nx.Graph()
    conflict_graph.add_nodes_from(range(n))
    conflict_graph.add_edges_from(conflict_pairs)

    selected_idx = set()
    for component in nx.connected_components(conflict_graph):
        selected_idx.update(
            _weighted_independent_set_on_component(
                conflict_graph.subgraph(component), node_weights
            )
        )
    return {int(target_ids[i]) for i in selected_idx}


def _select_priority_collision_free_targets(gal_in_tile, sep_arcsec_min):
    """Filter to a collision-free subset ranked by priority then target count."""
    if len(gal_in_tile) == 0:
        return gal_in_tile

    _, first_idx = np.unique(gal_in_tile["TARGETID"], return_index=True)
    first_rows = gal_in_tile[np.sort(first_idx)]
    keep_ids = _priority_weighted_collision_free_target_ids(
        first_rows["RA"],
        first_rows["DEC"],
        first_rows["TARGETID"],
        first_rows["PRIORITY"] + first_rows["SUBPRIORITY"],
        sep_arcsec_min,
    )
    return gal_in_tile[np.isin(gal_in_tile["TARGETID"], list(keep_ids))]


class _SerialEvalPool:
    def __init__(self, processes=None, **kwargs):
        self.processes = processes or 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]

    def map(self, func, iterable):
        return [func(item) for item in iterable]


@contextlib.contextmanager
def _avoid_nested_pool_in_solve():
    original_pool = _network_flow_module.multiprocessing.Pool
    _network_flow_module.multiprocessing.Pool = _SerialEvalPool
    try:
        yield
    finally:
        _network_flow_module.multiprocessing.Pool = original_pool

def fba_onetile_decollided(tile_ra, tile_dec, tile_id, gal_mtl, neighboring_fiber_pairs, fiberpos_xy, eval_workers):
    """
    tile_ra: float
    tile_dec: float
    tile_id: int
    gal_mtl: galaxy catalog select from healpixID fits files
    neighboring_fiber_pairs: list of fiber pairs
    fiberpos_xy: fiber positions in xy plane
    eval_workers: number of workers for evaluation
    Returns:
    fba_result: dictionary containing the fiber assignment results
    targets_id_list_alltiles_reachable: dictionary containing the target IDs for each tile that are reachable
    """
    in_tile = mask_targets_in_tile(
        tile_ra, tile_dec, gal_mtl["RA"], gal_mtl["DEC"],
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG,
    )
    gal_in_tile = gal_mtl[in_tile]

    n_before = len(np.unique(gal_in_tile["TARGETID"])) if len(gal_in_tile) else 0
    gal_in_tile = _select_priority_collision_free_targets(
        gal_in_tile, COLLISION_SEPARATION_ARCSEC
    )
    n_after = len(np.unique(gal_in_tile["TARGETID"])) if len(gal_in_tile) else 0
    if n_before:
        kept_priority = float(
            np.sum(gal_in_tile["PRIORITY"] + gal_in_tile["SUBPRIORITY"])
        )
        _log(
            f"  Tile {tile_id}: priority-weighted collision-free subset "
            f"{n_after}/{n_before} targets "
            f"(sep > {COLLISION_SEPARATION_ARCSEC} arcsec, "
            f"sum priority+subpriority={kept_priority:.1f})"
        )

    if len(gal_in_tile) == 0:
        raise RuntimeError(
            f"Tile {tile_id} ({tile_ra:.4f}, {tile_dec:.4f}): "
            f"no targets in tile annulus"
        )

    _, first_idx = np.unique(gal_in_tile["TARGETID"], return_index=True)
    first_rows = gal_in_tile[np.sort(first_idx)]
    target_id_array_unique = first_rows["TARGETID"]
    priority = first_rows["PRIORITY"]
    subpriority = first_rows["SUBPRIORITY"]

    tile_key, targets_id_list = find_targets_in_one_tile((
        tile_id, tile_ra, tile_dec, gal_in_tile, fiberpos_xy,
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, R_PATROL_DEG, "TARGETID",
    ))
    targets_id_list_alltiles = {tile_key: targets_id_list}   # just one tile

    _, targets_id_list_reachable = find_targets_in_one_tile((
        tile_id, tile_ra, tile_dec, gal_in_tile, fiberpos_xy,
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, R_PATROL_DEG, "TARGETID",
    ))
    targets_id_list_alltiles_reachable = {tile_key: targets_id_list_reachable}
    
    target_positions = {
        int(tid): (float(ra), float(dec))
        for tid, ra, dec in zip(first_rows["TARGETID"], first_rows["RA"], first_rows["DEC"])
    }

    tiles_ra = np.asarray([tile_ra], dtype=np.float64)  # to make it as a list in order to use solve_tile_group function
    tiles_dec = np.asarray([tile_dec], dtype=np.float64)
    tiles_id_arr = np.asarray([tile_id], dtype=np.int64)

    collision_constraints = find_collided_pairs_in_one_tile((
        tile_key, 0, tiles_ra, tiles_dec, targets_id_list_alltiles,
        neighboring_fiber_pairs, target_positions, fiberpos_xy,
        COLLISION_SEPARATION_ARCSEC,
    ))
    n_pairs = sum(len(p) for p in collision_constraints.values())
    _log(f"  Tile {tile_id}: {len(collision_constraints)} fiber-pair constraints, {n_pairs} target pairs")
    
    if eval_workers <= 1:
        with _avoid_nested_pool_in_solve():
            flow_dict, cost, forbidden, n_assigned, n_used_fibers = solve_tile_group(
                target_positions, [0], targets_id_list_alltiles,
                target_id_array_unique, priority, subpriority, collision_constraints,
                COST_OVERFLOW, N_FIBERS,
                max_iterations=max_iterations,
                n_workers=eval_workers,
                tiles_id=tiles_id_arr,
            )
    else:
        flow_dict, cost, forbidden, n_assigned, n_used_fibers = solve_tile_group(
            target_positions, [0], targets_id_list_alltiles,
            target_id_array_unique, priority, subpriority, collision_constraints,
            COST_OVERFLOW, N_FIBERS,
            max_iterations=max_iterations,
            n_workers=eval_workers,
            tiles_id=tiles_id_arr,
        )
    _log(
        f"  solve_tile_group finished: "
        f"cost={cost:.2f}, assigned={n_assigned}, used_fibers={n_used_fibers}/{N_FIBERS}"
    )
    
    fba_result = aggregate_group_assignments_with_pairwise_repair(
        flow_dict=flow_dict,
        target_ids=target_id_array_unique,
        all_forbidden=set(),  # parameter can be removed
        collision_constraints=collision_constraints,
        gal_cat=gal_in_tile,
        apply_repair=True,
        verbose=True,
    )
    
    return fba_result, targets_id_list_alltiles_reachable


def fba_onetile(tile_ra, tile_dec, tile_id, gal_mtl, neighboring_fiber_pairs, fiberpos_xy, eval_workers):
    """
    tile_ra: float
    tile_dec: float
    tile_id: int
    gal_mtl: galaxy catalog select from healpixID fits files
    neighboring_fiber_pairs: list of fiber pairs
    fiberpos_xy: fiber positions in xy plane
    eval_workers: number of workers for evaluation
    """
    in_tile = mask_targets_in_tile(
        tile_ra, tile_dec, gal_mtl["RA"], gal_mtl["DEC"],
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG,
    )
    gal_in_tile = gal_mtl[in_tile]

    if len(gal_in_tile) == 0:
        raise RuntimeError(
            f"Tile {tile_id} ({tile_ra:.4f}, {tile_dec:.4f}): "
            f"no targets in tile annulus"
        )

    _, first_idx = np.unique(gal_in_tile["TARGETID"], return_index=True)
    first_rows = gal_in_tile[np.sort(first_idx)]
    target_id_array_unique = first_rows["TARGETID"]
    priority = first_rows["PRIORITY"]
    subpriority = first_rows["SUBPRIORITY"]

    tile_key, targets_id_list = find_targets_in_one_tile((
        tile_id, tile_ra, tile_dec, gal_in_tile, fiberpos_xy,
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, R_PATROL_DEG, "TARGETID",
    ))
    targets_id_list_alltiles = {tile_key: targets_id_list}   # just one tile

    _, targets_id_list_reachable = find_targets_in_one_tile((
        tile_id, tile_ra, tile_dec, gal_in_tile, fiberpos_xy,
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, R_PATROL_DEG, "TARGETID",
    ))
    targets_id_list_alltiles_reachable = {tile_key: targets_id_list_reachable}
    
    target_positions = {
        int(tid): (float(ra), float(dec))
        for tid, ra, dec in zip(first_rows["TARGETID"], first_rows["RA"], first_rows["DEC"])
    }

    tiles_ra = np.asarray([tile_ra], dtype=np.float64)  # to make it as a list in order to use solve_tile_group function
    tiles_dec = np.asarray([tile_dec], dtype=np.float64)
    tiles_id_arr = np.asarray([tile_id], dtype=np.int64)

    collision_constraints = find_collided_pairs_in_one_tile((
        tile_key, 0, tiles_ra, tiles_dec, targets_id_list_alltiles,
        neighboring_fiber_pairs, target_positions, fiberpos_xy,
        COLLISION_SEPARATION_ARCSEC,
    ))
    n_pairs = sum(len(p) for p in collision_constraints.values())
    _log(f"  Tile {tile_id}: {len(collision_constraints)} fiber-pair constraints, {n_pairs} target pairs")
    
    if eval_workers <= 1:
        with _avoid_nested_pool_in_solve():
            flow_dict, cost, forbidden, n_assigned, n_used_fibers = solve_tile_group(
                target_positions, [0], targets_id_list_alltiles,
                target_id_array_unique, priority, subpriority, collision_constraints,
                COST_OVERFLOW, N_FIBERS,
                max_iterations=max_iterations,
                n_workers=eval_workers,
                tiles_id=tiles_id_arr,
            )
    else:
        flow_dict, cost, forbidden, n_assigned, n_used_fibers = solve_tile_group(
            target_positions, [0], targets_id_list_alltiles,
            target_id_array_unique, priority, subpriority, collision_constraints,
            COST_OVERFLOW, N_FIBERS,
            max_iterations=max_iterations,
            n_workers=eval_workers,
            tiles_id=tiles_id_arr,
        )
    _log(
        f"  solve_tile_group finished: "
        f"cost={cost:.2f}, assigned={n_assigned}, used_fibers={n_used_fibers}/{N_FIBERS}"
    )
    
    fba_result = aggregate_group_assignments_with_pairwise_repair(
        flow_dict=flow_dict,
        target_ids=target_id_array_unique,
        all_forbidden=set(),  # parameter can be removed
        collision_constraints=collision_constraints,
        gal_cat=gal_in_tile,
        apply_repair=True,
        verbose=True,
    )
    
    return fba_result, targets_id_list_alltiles_reachable
