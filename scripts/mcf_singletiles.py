#!/usr/bin/env python
# coding: utf-8
# We run fiber assignment for one tile at a time, which is slow but may not meet OOM issue. --06-17-2026
# python mcf_singletiles.py \
#   --rmagcut 20.5 --Npasses 3 \
#   --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
#   --n_workers 8 --eval_workers 1 \
#   --max_iterations 3 --rand_seed 3 --nside 32

import os, sys, json
import glob
import signal
import numpy as np
import time, gc
import argparse
from astropy.table import Table, vstack
import multiprocessing
import contextlib
from numpy.random import Generator, PCG64
from collections import defaultdict
sys.path.append("/home/zjding/installed_packages/JUST_fiberassign/py/")
from utils import (
    get_fiberpos, write_fba_onetile,
    mask_targets_in_tile,
    find_targets_in_one_tile,
    find_neighboring_fibers,
    find_collided_pairs_in_one_tile,
    find_overlapped_tile_groups
    )
from parameters import (
    COLLISION_SEPARATION_ARCSEC,
    COLLISION_SEPARATION_DEG,
    COLLISION_SEPARATION_MM,
    TILE_INNER_RADIUS_DEG,
    TILE_OUTER_RADIUS_DEG,
    R_PATROL_DEG)
from network_flow import (
    solve_tile_group,
    aggregate_group_assignments_with_pairwise_repair
    )
import network_flow as _network_flow_module

#import fitsio   # can not be loaded in the Gravity environment of spectrum_workshop
import healpy as hp


def _log(msg):
    print(msg, flush=True)


class MCFSolveTimeout(TimeoutError):
    """Raised when solve_tile_group exceeds the per-tile time limit."""


def _call_solve_tile_group(
    use_serial_pool,
    target_positions,
    targets_id_list_alltiles,
    target_id_array_unique,
    priority_total,
    collision_constraints,
    cost_overflow,
    n_fibers,
    max_iterations,
    eval_workers,
    tiles_id_arr,
):
    kwargs = dict(
        target_positions=target_positions,
        group_tile_indices=[0],
        targets_id_list_alltiles=targets_id_list_alltiles,
        target_id_array_unique=target_id_array_unique,
        priority_array=priority_total,
        collision_constraints=collision_constraints,
        COST_OVERFLOW=cost_overflow,
        N_fibers=n_fibers,
        max_iterations=max_iterations,
        n_workers=eval_workers,
        tiles_id=tiles_id_arr,
    )
    if use_serial_pool:
        with _avoid_nested_pool_in_solve():
            return solve_tile_group(**kwargs)
    return solve_tile_group(**kwargs)


def _solve_tile_group_with_timeout(timeout_sec, use_serial_pool, **solve_kwargs):
    if timeout_sec is None or timeout_sec <= 0:
        return _call_solve_tile_group(use_serial_pool=use_serial_pool, **solve_kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise MCFSolveTimeout(
            f"solve_tile_group exceeded {timeout_sec:.0f}s"
        )

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return _call_solve_tile_group(use_serial_pool=use_serial_pool, **solve_kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _solve_singletile_mcf(
    tile_id,
    target_positions,
    targets_id_list_alltiles,
    target_id_array_unique,
    priority_total,
    collision_constraints,
    cost_overflow,
    n_fibers,
    max_iterations,
    eval_workers,
    tiles_id_arr,
    mcf_timeout_sec,
):
    """
    Run MCF for one tile. Try in-graph collision gadgets first; on timeout,
    retry without gadgets and rely on aggregate_group_assignments_with_pairwise_repair.
    """
    use_serial_pool = eval_workers <= 1
    common = dict(
        target_positions=target_positions,
        targets_id_list_alltiles=targets_id_list_alltiles,
        target_id_array_unique=target_id_array_unique,
        priority_total=priority_total,
        cost_overflow=cost_overflow,
        n_fibers=n_fibers,
        eval_workers=eval_workers,
        tiles_id_arr=tiles_id_arr,
    )

    t0 = time.time()
    used_gadgets = True
    try:
        result = _solve_tile_group_with_timeout(
            mcf_timeout_sec,
            use_serial_pool,
            collision_constraints=collision_constraints,
            max_iterations=max_iterations,
            **common,
        )
    except MCFSolveTimeout as exc:
        elapsed = time.time() - t0
        _log(
            f"  Tile {tile_id}: {exc} with collision gadgets "
            f"(elapsed {elapsed:.1f}s); retrying without gadgets "
            f"(pairwise repair will enforce collisions)"
        )
        used_gadgets = False
        result = _call_solve_tile_group(
            use_serial_pool=use_serial_pool,
            collision_constraints={},
            max_iterations=1,
            **common,
        )

    flow_dict, cost, forbidden, n_assigned, n_used_fibers = result
    mode = "collision gadgets" if used_gadgets else "no gadgets (pairwise repair)"
    _log(
        f"  solve_tile_group finished ({mode}): "
        f"cost={cost:.2f}, assigned={n_assigned}, "
        f"used_fibers={n_used_fibers}/{n_fibers}, elapsed={time.time() - t0:.1f}s"
    )
    return flow_dict, cost, forbidden, n_assigned, n_used_fibers


def _fits_path_for_tile(out_dir, tile_id):
    return os.path.join(out_dir, f"fba_tile_{int(tile_id)}.fits")


def _checkpoint_path(out_dir):
    return os.path.join(out_dir, "fba_checkpoint.json")


def _list_completed_tiles(out_dir):
    completed = set()
    for path in glob.glob(os.path.join(out_dir, "fba_tile_*.fits")):
        base = os.path.basename(path)
        try:
            completed.add(int(base[len("fba_tile_") : -len(".fits")]))
        except ValueError:
            continue
    return completed


def _save_checkpoint(path, state):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


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


def fba_onetile(
    tile_ra,
    tile_dec,
    tile_id,
    gal_mtl,
    neighboring_fiber_pairs,
    fiberpos_xy,
    n_fibers,
    cost_overflow,
    max_iterations,
    eval_workers,
    mcf_timeout_sec=240.0,
):
    """Run network-flow fiber assignment for one tile."""
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

    target_id_array_unique = np.unique(gal_in_tile["TARGETID"])
    _log(f"  Tile {tile_id}: {len(target_id_array_unique)} unique targets in annulus")


    priority_total = gal_in_tile["PRIORITY"] + gal_in_tile["SUBPRIORITY"]

    tile_key, targets_id_list = find_targets_in_one_tile((
        tile_id, tile_ra, tile_dec, gal_in_tile, fiberpos_xy,
        TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, R_PATROL_DEG, "TARGETID",
    ))
    targets_id_list_alltiles = {tile_key: targets_id_list}   # just one tile
    
    _, first_idx = np.unique(gal_in_tile["TARGETID"], return_index=True)
    first_rows = gal_in_tile[np.sort(first_idx)]
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

    flow_dict, cost, forbidden, n_assigned, n_used_fibers = _solve_singletile_mcf(
        tile_id,
        target_positions,
        targets_id_list_alltiles,
        target_id_array_unique,
        priority_total,
        collision_constraints,
        cost_overflow,
        n_fibers,
        max_iterations,
        eval_workers,
        tiles_id_arr,
        mcf_timeout_sec,
    )

    fba_result = aggregate_group_assignments_with_pairwise_repair(
        flow_dict=flow_dict,
        target_ids=target_id_array_unique,
        all_forbidden=set(),
        collision_constraints=collision_constraints,
        gal_cat=gal_in_tile,
        apply_repair=True,
        verbose=True,
    )
    
    return fba_result, targets_id_list_alltiles

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmagcut", type=float, default=20.5)
    parser.add_argument("--Npasses", type=int, default=3)
    parser.add_argument("--ra0", type=float, default=30.0)
    parser.add_argument("--ra1", type=float, default=40.0)
    parser.add_argument("--dec0", type=float, default=10.0)
    parser.add_argument("--dec1", type=float, default=20.0)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="Workers for collision-repair eval inside solve_tile_group (use 1 on PBS)",
    )
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument(
        "--mcf_timeout_sec",
        type=float,
        default=240.0,
        help=(
            "Per-tile time limit (seconds) for solve_tile_group with collision "
            "gadgets; on timeout, retry without gadgets and use pairwise repair"
        ),
    )
    parser.add_argument("--rand_seed", type=int, default=3)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing output FITS and rerun from scratch",
    )
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside for MTL pixel split")
    args = parser.parse_args()

    rmagcut = args.rmagcut
    Npasses = args.Npasses
    ra0 = args.ra0
    ra1 = args.ra1
    dec0 = args.dec0
    dec1 = args.dec1
    n_workers = args.n_workers
    eval_workers = max(1, args.eval_workers)
    max_iterations = args.max_iterations
    mcf_timeout_sec = args.mcf_timeout_sec
    rand_seed = args.rand_seed
    no_resume = args.no_resume
    nside = args.nside

    pool_workers = max(1, n_workers - 2) if n_workers > 2 else n_workers

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    rng = Generator(PCG64(seed=rand_seed))
    # Constants
    _log(
        f"rand_seed={rand_seed}, pool_workers={pool_workers}, "
        f"eval_workers={eval_workers}, mcf_timeout_sec={mcf_timeout_sec}"
    )
    _log(f"Patrol radius: {R_PATROL_DEG:.6f} degrees")
    _log(f"Collision separation: {COLLISION_SEPARATION_ARCSEC} arcsec = {COLLISION_SEPARATION_DEG:.6f} deg = {COLLISION_SEPARATION_MM:.3f} mm")
    
    # Get fiber positions and set random seed
    fiberpos_xy = get_fiberpos()
    N_fibers = fiberpos_xy.shape[0]
    _log(f"Number of fibers: {N_fibers}")
    
    # Define overflow cost
    COST_OVERFLOW = 10.0
    
    
    
    dir_root = "/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/"
    ifile = dir_root + f"data/lightcone_ra_0_90_dec_0_90_rmagcut{rmagcut}.fits"
    columns = ["ra", "dec", "idx"]
    #input_cat = fitsio.read(ifile, columns=columns)
    input_cat = Table.read(ifile)
    mask = (input_cat["ra"]>ra0)&(input_cat["ra"]<ra1)&(input_cat["dec"]>dec0)&(input_cat["dec"]<dec1)
    input_cat = input_cat[mask]
    
    gal_MTL = Table()
    priority_initial = 4.0      # initial priority for all targets
    priority_degraded = 1.0     # degraded priority for targets that have been fiber assigned
    columns=["TARGETID", "RA", "DEC", "PRIORITY", "SUBPRIORITY"]
    
    for col in columns:
        if col == "TARGETID":
            gal_MTL[col] = input_cat["idx"]
        elif col == "RA":
            gal_MTL[col] = input_cat["ra"]
        elif col == "DEC":
            gal_MTL[col] = input_cat["dec"]
        elif col == "PRIORITY":
            gal_MTL[col] = np.ones(len(input_cat), dtype=float)*priority_initial
        elif col == "SUBPRIORITY":
            gal_MTL[col] = rng.random(len(input_cat))
        else:
            gal_MTL[col] = input_cat[col]
    del input_cat; gc.collect()
    
    
    # load tiles
    ifile = dir_root + "fba/input/tiles_4passes.fits"
    tile_data = Table.read(ifile)
    tile_ra0, tile_ra1 = ra0 + TILE_OUTER_RADIUS_DEG, ra1 - TILE_OUTER_RADIUS_DEG
    tile_dec0, tile_dec1 = dec0 + TILE_OUTER_RADIUS_DEG, dec1 - TILE_OUTER_RADIUS_DEG
    
    mask = (tile_data["RA"]>tile_ra0)&(tile_data["RA"]<tile_ra1)&(tile_data["DEC"]>tile_dec0)&(tile_data["DEC"]<tile_dec1)
    mask &= (tile_data["PASS"]<Npasses)   # PASS starts from 0
    tiles_sub = tile_data[mask]
    N_tiles = len(tiles_sub)
    
    neighboring_fiber_pairs = find_neighboring_fibers(fiberpos_xy, patrol_center_separation=12.0)
    _log(f"There are {len(neighboring_fiber_pairs)} paired fibers in one tile")
    
    out_dir = dir_root + f"fba/output/{Npasses}passes/{N_tiles}tiles_{ra0:.1f}ra{ra1:.1f}_{dec0:.1f}dec{dec1:.1f}/seed{rand_seed}/"
    os.makedirs(out_dir, exist_ok=True)
    
    
    nest = False  # RING ordering (healpy default)
    mtl_dir = out_dir + f"/mtl_nside{nside}/"
    mtl_files_exist = bool(glob.glob(os.path.join(mtl_dir, "mtl_healpix_*.fits")))
    # split the galaxies into pixels for the case running fiber assignment from the beginning
    if no_resume or not mtl_files_exist:
        _log(f"Split the MTL galaxies into pixels for the case running fiber assignment from the scratch.")
        npix = hp.nside2npix(nside)
        pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
        _log(f"HEALPix nside={nside}: {npix} pixels, ~{np.sqrt(pix_area_deg2):.2f} deg/pixel side")

        ra = np.asarray(gal_MTL["RA"], dtype=np.float64)
        dec = np.asarray(gal_MTL["DEC"], dtype=np.float64)
        pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)

        gal_MTL["HEALPIXID"] = pix.astype(np.int64)
        _log(f"Assigned {len(gal_MTL)} galaxies to {len(np.unique(pix))} non-empty pixels")

        sort_idx = np.argsort(pix, kind="stable")
        pix_sorted = pix[sort_idx]
        unique_pix, start_idx, counts = np.unique(
            pix_sorted, return_index=True, return_counts=True
        )

        pixel_cats = {}
        for pix_id, start, count in zip(unique_pix, start_idx, counts):
            rows = sort_idx[start : start + count]
            pixel_cats[int(pix_id)] = gal_MTL[rows]

        _log(f"Built pixel_cats with {len(pixel_cats)} entries")
        _log(
            "Galaxies per pixel: "
            f"min={counts.min()}, median={np.median(counts):.0f}, max={counts.max()}"
        )

        os.makedirs(mtl_dir, exist_ok=True)
        for pix_id, cat_pix in pixel_cats.items():
            ofile = os.path.join(mtl_dir, f"mtl_healpix_{pix_id:05d}.fits")
            cat_pix.write(ofile, overwrite=True)
        _log(f"Wrote {len(pixel_cats)} pixel catalogs to {mtl_dir}")

        del gal_MTL, pixel_cats
        gc.collect()
    else:
        _log(f"Resume: reusing existing MTL pixel files in {mtl_dir}")
        del gal_MTL
        gc.collect()

    if no_resume:
        completed_tiles = set()
        _log("Resume disabled (--no_resume); starting from scratch")
    else:
        completed_tiles = _list_completed_tiles(out_dir)
        if completed_tiles:
            _log(f"Resume: found {len(completed_tiles)} completed tile(s) in {out_dir}")
        else:
            _log(f"Resume: no completed tiles found in {out_dir}")

    near_radius_deg = 1.2 * TILE_OUTER_RADIUS_DEG
    _log(
        f"Load the sample from healpixID files, where some galaxies are within "
        f"{near_radius_deg:.6f} deg of the tile center."
    )

    for passid in range(Npasses):
        t0 = time.time()
        _log(f"passid: {passid}")
        mask = (tiles_sub["PASS"] == passid)
        tiles_ra_pass = np.asarray(tiles_sub["RA"][mask], dtype=np.float64)
        tiles_dec_pass = np.asarray(tiles_sub["DEC"][mask], dtype=np.float64)
        tiles_id_pass = np.asarray(tiles_sub["TILEID"][mask], dtype=np.int64)
        _log(f"Number of tiles in pass {passid}: {len(tiles_ra_pass)}")

        for tile_ra, tile_dec, tile_id in zip(tiles_ra_pass, tiles_dec_pass, tiles_id_pass):
            out_fits = _fits_path_for_tile(out_dir, tile_id)
            if not no_resume and tile_id in completed_tiles:
                _log(f"Tile {tile_id}: skip (already exists) {out_fits}")
                continue

            pix_ids = _healpix_ids_for_disc(tile_ra, tile_dec, near_radius_deg, nside, nest=nest)
            gal_mtl = load_galaxies_from_mtlpix(
                tile_ra,
                tile_dec,
                mtl_dir,
                near_radius_deg,
                nside=nside,
                nest=nest,
            )
            _log(
                f"Tile {tile_id} ({tile_ra:.4f}, {tile_dec:.4f}): "
                f"loaded {len(gal_mtl)} galaxies from {len(pix_ids)} HEALPix file(s) "
                f"pix={pix_ids.tolist()}"
            )

            t_fba_start = time.time()
            fba_result, targets_id_list_alltiles = fba_onetile(
                tile_ra,
                tile_dec,
                tile_id,
                gal_mtl,
                neighboring_fiber_pairs,
                fiberpos_xy,
                N_fibers,
                COST_OVERFLOW,
                max_iterations,
                eval_workers,
                mcf_timeout_sec,
            )
            t_fba_end = time.time()
            _log(f"  fba process finished in {t_fba_end - t_fba_start:.2f}s")

            per_tile_assigned = defaultdict(lambda: {"target_ids": [], "fiber_ids": []})
            for _tid, _node in fba_result["target_to_fiber"].items():
                _node = str(_node)
                if "_fiber_" not in _node:
                    continue
                _tile, _fid = _node.rsplit("_fiber_", 1)
                try:
                    _fid = int(_fid)
                except Exception:
                    continue
                per_tile_assigned[_tile]["target_ids"].append(int(_tid))
                per_tile_assigned[_tile]["fiber_ids"].append(_fid)

            assigned_tarids_all = []
            for _tile, _vals in per_tile_assigned.items():
                assigned_targets_id = _vals["target_ids"]
                assigned_tarids_all.extend(assigned_targets_id)
                write_fba_onetile(
                    tile_id=_tile,
                    assigned_targets_id=assigned_targets_id,
                    assigned_fiber_id=_vals["fiber_ids"],
                    targets_id_list_alltiles=targets_id_list_alltiles,
                    out_fits_path=out_fits,
                    overwrite=True,
                )
            t_outfits_end = time.time()
            _log(
                f"  Wrote {out_fits} ({len(assigned_targets_id)} assignments); "
                f"takes {t_outfits_end - t_fba_end:.2f}s"
            )

            if assigned_tarids_all:
                affected_pix = np.unique(gal_mtl["HEALPIXID"][np.isin(gal_mtl["TARGETID"], assigned_tarids_all)])
                n_disk = _degrade_mtl_priorities_on_disk(
                    mtl_dir, assigned_tarids_all, priority_degraded, affected_pix
                )
                _log(
                    f"  Updated PRIORITY on disk for {n_disk} row(s) "
                    f"in {len(affected_pix)} pixel file(s): {affected_pix.tolist()}"
                )
            completed_tiles.add(int(tile_id))
            _log(f"  Update the MTL PRIORITY files in {time.time() - t_outfits_end:.2f}s\n")
            _log("=====================")

        _log(f"pass {passid} running time (s): {time.time() - t0:.1f}")

if __name__ == "__main__":
    main()




