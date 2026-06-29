#!/usr/bin/env python
# coding: utf-8
# Single-tile fiber assignment with priority-weighted deblending before MCF.
# Based on mcf_singletiles_speedup_v0.14.ipynb; core tile logic lives in fba_single_tile.py.
#
# Example:
#   python mcf_singletiles_debelend.py \
#     --mock_version "v1" \
#     --input_mockpath "/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/data/v1/lightcone_ra_0_90_dec_0_90_rmagcut20.5.fits" \
#     --input_tilepath "/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/fba/input/tiles_4passes.fits" \
#     --output_fba_path "/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/fba/output/" \
#     --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
#     --Npasses 3 \
#     --n_workers 8 --eval_workers 1 \
#     --max_iterations 1 --rand_seed 100 --nside 32

import argparse
import glob
import gc
import os
import signal
import sys
import time
from collections import defaultdict

import healpy as hp
import numpy as np
from astropy.table import Table
from numpy.random import Generator, PCG64

sys.path.append("/home/zjding/installed_packages/JUST_fiberassign/py/")
from parameters import (
    COLLISION_SEPARATION_ARCSEC,
    COLLISION_SEPARATION_DEG,
    COLLISION_SEPARATION_MM,
    TILE_OUTER_RADIUS_DEG,
    R_PATROL_DEG,
)
from utils import _log, get_fiberpos, write_fba_onetile, find_neighboring_fibers
from fba_single_tile import (
    _degrade_mtl_priorities_on_disk,
    _fits_path_for_tile,
    _healpix_ids_for_disc,
    fba_onetile_deblend,
    fba_onetile,
    load_galaxies_from_mtlpix,
)


DEBLEND_TIMEOUT_SEC = 10.0


class FBAOnetileTimeout(TimeoutError):
    """Raised when fba_onetile_deblend exceeds the per-tile time limit."""


def _run_with_timeout(timeout_sec, func, *args, **kwargs):
    if timeout_sec is None or timeout_sec <= 0:
        return func(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise FBAOnetileTimeout(
            f"{func.__name__} exceeded {timeout_sec:.0f}s"
        )

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _deblend_fail_path(out_dir):
    return os.path.join(out_dir, "fba_deblend_timeout_tiles.txt")


def _record_deblend_fail(tile_id, fail_path):
    """Append a TILEID that timed out in fba_onetile_deblend."""
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(f"{int(tile_id)}\n")
        f.flush()
        os.fsync(f.fileno())


def _fba_onetile_deblend_or_fallback(
    tile_ra,
    tile_dec,
    tile_id,
    gal_mtl,
    neighboring_fiber_pairs,
    fiberpos_xy,
    eval_workers,
    deblend_fail_path,
    deblend_timeout_sec=DEBLEND_TIMEOUT_SEC,
):
    """Try fba_onetile_deblend; on timeout fall back to fba_onetile."""
    t0 = time.time()
    try:
        result = _run_with_timeout(
            deblend_timeout_sec,
            fba_onetile_deblend,
            tile_ra,
            tile_dec,
            tile_id,
            gal_mtl,
            neighboring_fiber_pairs,
            fiberpos_xy,
            eval_workers,
        )
        _log(
            f"  fba_onetile_deblend finished in {time.time() - t0:.2f}s"
        )
        return result
    except FBAOnetileTimeout as exc:
        elapsed = time.time() - t0
        _record_deblend_fail(tile_id, deblend_fail_path)
        _log(
            f"  Tile {tile_id}: {exc} (elapsed {elapsed:.1f}s); "
            f"falling back to fba_onetile (recorded in {deblend_fail_path})"
        )
        t1 = time.time()
        result = fba_onetile(
            tile_ra,
            tile_dec,
            tile_id,
            gal_mtl,
            neighboring_fiber_pairs,
            fiberpos_xy,
            eval_workers,
        )
        _log(f"  fba_onetile finished in {time.time() - t1:.2f}s")
        return result



def _list_completed_tiles(out_dir):
    completed = set()
    for path in glob.glob(os.path.join(out_dir, "fba_tile_*.fits")):
        base = os.path.basename(path)
        try:
            completed.add(int(base[len("fba_tile_") : -len(".fits")]))
        except ValueError:
            continue
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mockpath", type=str, default="/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/data/v1/lightcone_ra_0_90_dec_0_90_rmagcut20.5.fits")
    parser.add_argument("--input_tilepath", type=str, default="/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/fba/input/tiles_4passes.fits")
    parser.add_argument("--output_fba_path", type=str, default="/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/fba/output/v1/")
    parser.add_argument("--mock_version", type=str, default="v1")
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
    parser.add_argument("--rand_seed", type=int, default=100)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing output FITS and rerun from scratch",
    )
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside for MTL pixel split")
    args = parser.parse_args()

    input_mockpath = args.input_mockpath
    input_tilepath = args.input_tilepath
    output_fba_path = args.output_fba_path
    mock_version = args.mock_version
    Npasses = args.Npasses
    ra0 = args.ra0
    ra1 = args.ra1
    dec0 = args.dec0
    dec1 = args.dec1
    n_workers = args.n_workers
    eval_workers = max(1, args.eval_workers)
    max_iterations = args.max_iterations
    rand_seed = args.rand_seed
    no_resume = args.no_resume
    nside = args.nside

    pool_workers = max(1, n_workers - 2) if n_workers > 2 else n_workers

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    rng = Generator(PCG64(seed=rand_seed))
    _log(
        f"rand_seed={rand_seed}, pool_workers={pool_workers}, "
        f"eval_workers={eval_workers}, max_iterations={max_iterations}"
    )
    _log(f"Patrol radius: {R_PATROL_DEG:.6f} degrees")
    _log(
        f"Collision separation: {COLLISION_SEPARATION_ARCSEC} arcsec = "
        f"{COLLISION_SEPARATION_DEG:.6f} deg = {COLLISION_SEPARATION_MM:.3f} mm"
    )

    fiberpos_xy = get_fiberpos()
    N_fibers = fiberpos_xy.shape[0]
    _log(f"Number of fibers: {N_fibers}")


    input_cat = Table.read(input_mockpath)
    mask = (
        (input_cat["ra"] > ra0)
        & (input_cat["ra"] < ra1)
        & (input_cat["dec"] > dec0)
        & (input_cat["dec"] < dec1)
    )
    input_cat = input_cat[mask]

    gal_MTL = Table()
    priority_initial = 100.0
    priority_degraded = 2.0
    columns = ["TARGETID", "RA", "DEC", "PRIORITY", "SUBPRIORITY"]

    for col in columns:
        if col == "TARGETID":
            gal_MTL[col] = input_cat["idx"]
        elif col == "RA":
            gal_MTL[col] = input_cat["ra"]
        elif col == "DEC":
            gal_MTL[col] = input_cat["dec"]
        elif col == "PRIORITY":
            gal_MTL[col] = np.ones(len(input_cat), dtype=float) * priority_initial
        elif col == "SUBPRIORITY":
            gal_MTL[col] = rng.random(len(input_cat))
        else:
            gal_MTL[col] = input_cat[col]
    del input_cat
    gc.collect()


    tile_data = Table.read(input_tilepath)
    tile_ra0, tile_ra1 = ra0 + TILE_OUTER_RADIUS_DEG, ra1 - TILE_OUTER_RADIUS_DEG
    tile_dec0, tile_dec1 = dec0 + TILE_OUTER_RADIUS_DEG, dec1 - TILE_OUTER_RADIUS_DEG

    mask = (
        (tile_data["RA"] > tile_ra0)
        & (tile_data["RA"] < tile_ra1)
        & (tile_data["DEC"] > tile_dec0)
        & (tile_data["DEC"] < tile_dec1)
    )
    mask &= tile_data["PASS"] < Npasses
    tiles_sub = tile_data[mask]
    N_tiles = len(tiles_sub)

    neighboring_fiber_pairs = find_neighboring_fibers(
        fiberpos_xy, patrol_center_separation=12.0
    )
    _log(f"There are {len(neighboring_fiber_pairs)} paired fibers in one tile")

    out_dir = (
        output_fba_path
        + f"{Npasses}passes/{N_tiles}tiles_{ra0:.1f}ra{ra1:.1f}_{dec0:.1f}dec{dec1:.1f}/seed{rand_seed}/"
    )
    os.makedirs(out_dir, exist_ok=True)
    deblend_fail_path = _deblend_fail_path(out_dir)
    if no_resume and os.path.isfile(deblend_fail_path):
        os.remove(deblend_fail_path)
    _log(f"Deblend timeout tile list: {deblend_fail_path}")

    nest = False
    mtl_dir = out_dir + f"/mtl_nside{nside}/"
    mtl_files_exist = bool(glob.glob(os.path.join(mtl_dir, "mtl_healpix_*.fits")))
    if no_resume or not mtl_files_exist:
        _log("Split the MTL galaxies into pixels for a fresh fiber-assignment run.")
        npix = hp.nside2npix(nside)
        pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
        _log(
            f"HEALPix nside={nside}: {npix} pixels, "
            f"~{np.sqrt(pix_area_deg2):.2f} deg/pixel side"
        )

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
        mask = tiles_sub["PASS"] == passid
        tiles_ra_pass = np.asarray(tiles_sub["RA"][mask], dtype=np.float64)
        tiles_dec_pass = np.asarray(tiles_sub["DEC"][mask], dtype=np.float64)
        tiles_id_pass = np.asarray(tiles_sub["TILEID"][mask], dtype=np.int64)
        _log(f"Number of tiles in pass {passid}: {len(tiles_ra_pass)}")

        for tile_ra, tile_dec, tile_id in zip(
            tiles_ra_pass, tiles_dec_pass, tiles_id_pass
        ):
            out_fits = _fits_path_for_tile(out_dir, tile_id)
            if not no_resume and tile_id in completed_tiles:
                _log(f"Tile {tile_id}: skip (already exists) {out_fits}")
                continue

            pix_ids = _healpix_ids_for_disc(
                tile_ra, tile_dec, near_radius_deg, nside, nest=nest
            )
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
            fba_result, targets_id_list_alltiles = _fba_onetile_deblend_or_fallback(
                tile_ra,
                tile_dec,
                tile_id,
                gal_mtl,
                neighboring_fiber_pairs,
                fiberpos_xy,
                eval_workers,
                deblend_fail_path,
            )
            t_fba_end = time.time()
            _log(f"  fba total time: {t_fba_end - t_fba_start:.2f}s")

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
                affected_pix = np.unique(
                    gal_mtl["HEALPIXID"][
                        np.isin(gal_mtl["TARGETID"], assigned_tarids_all)
                    ]
                )
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

    if os.path.isfile(deblend_fail_path):
        with open(deblend_fail_path, encoding="utf-8") as f:
            n_fail = sum(1 for line in f if line.strip())
        _log(f"Total tiles that timed out in fba_onetile_deblend: {n_fail} ({deblend_fail_path})")


if __name__ == "__main__":
    main()
