import time, sys
import numpy as np
import random
from astropy.table import Table
from matplotlib import pyplot as plt
from multiprocessing import Pool
sys.path.append("/home/zjding/fiberassignment/algorithm")
from utils import get_fiberpos, is_point_in_just, radec2xy, plot_patrol_circle, get_spherearea
from assign import assign_targets_greedy
from functools import partial
from argparse import ArgumentParser

def assign_fun(tid, tileid_idx, idx, tiles, ra_array, dec_array, fibers_center, targets_ID, priority, subpriority):
    tilemask = (tileid_idx==tid)
    target_mask = np.logical_and(idx, tilemask)
    target_ra = ra_array[target_mask]
    target_dec = dec_array[target_mask]
    target_x, target_y = radec2xy(tiles['RA'][tid], tiles['DEC'][tid], target_ra, target_dec)
    targets_pos = np.vstack((target_x, target_y)).T
    
    assigned_target_ID = assign_targets_greedy(fibers_center, targets_pos, targets_ID[target_mask], priority, subpriority)
    return assigned_target_ID

def main():
    parser = ArgumentParser(description="Running fiber assignment on galaxy catalog.")
    parser.add_argument("--ncores", default=4, type=int, help="Number of processes (uses multiprocessing).")
    args = parser.parse_args()
    
    
    fibers_center = get_fiberpos()  # load fiber center position

    input_tiles = "./tiles.fits"
    tiles = Table.read(input_tiles)

    completeness_list = []
    target_skyden_list = []
    rmagcut_list = [19.5, 20.0, 20.5, 21.0, 21.5]
    for rmagcut in rmagcut_list:
        input_file = f"./catalog/DESIDR9_galaxy_ngc_rmagcut{rmagcut}.fits"
        cat = Table.read(input_file)

        ra_min, ra_max = np.min(tiles['RA'])-0.8, np.max(tiles['RA'])+0.8
        dec_min, dec_max = np.min(tiles['DEC'])-0.8, np.max(tiles['DEC'])+0.8

        ## select target_cat with similar (slightly larger) sky area of tile coverage
        mask = (cat['RA']>ra_min)&(cat['RA']<ra_max)&(cat['DEC']>dec_min)&(cat['DEC']<dec_max)
        target_cat = cat[mask]

        idx, tileid_idx = is_point_in_just(tiles['RA'], tiles['DEC'], target_cat['RA'], target_cat['DEC'], return_tile_index=True)

        ## select paraent_cat covered by tiles
        parent_cat = target_cat[idx]

        ## set priority and subpriority
        target_cat['priority'] = np.ones(len(target_cat))*1000.0
        seed = 13
        random.seed(seed)
        target_cat['subpriority'] = np.random.rand(len(target_cat))

        nproc = args.ncores
        chunksize = int(np.ceil(len(tiles)/nproc))
        print("chunksize:", chunksize)

        t0 = time.perf_counter()
        with Pool(processes=int(nproc)) as pool:
            partial_func = partial(assign_fun, tileid_idx=tileid_idx, idx=idx, tiles=tiles, ra_array=target_cat['RA'], dec_array=target_cat['DEC'], fibers_center=fibers_center, targets_ID=target_cat['gal_id'], priority=target_cat['priority'], subpriority=target_cat['subpriority'])
            assigned_id_list = pool.map(partial_func, tiles['TILEID'], chunksize=chunksize)
            t1 = time.perf_counter()
        print("Time of assignment: %.2f s"%(t1-t0))

        # append ID of assigned targets 
        assign_id_array =  np.array([item for sublist in assigned_id_list for item in sublist])
        print("Total number of assigned galaxies:", len(assign_id_array))
        ofile = f"./output/assigned_gal_id_DESIDR9_ngc_rmagcut{rmagcut}.npz"
        np.savez(ofile, assign_id_array=assign_id_array)

        ## the current algorithm divides targets into sub-sets for different tiles. Different sub-sets have no overlap. That's why the ID of assigned targets are unique.

        ## select sky region for statistical analysis, based on the input tile distribution
        ra0, ra1 = 150, 168
        dec0, dec1 = 0, 50
        ## estimate sky area without considering imaging mask
        area_cut = get_spherearea(ra0, ra1, dec0, dec1)
        print("sky area (deg^2):", area_cut)

        sky_mask = (parent_cat['RA']>ra0)&(parent_cat['RA']<ra1)&(parent_cat['DEC']>dec0)&(parent_cat['DEC']<dec1)
        parent_cat_cut = parent_cat[sky_mask]
        target_skyden = len(parent_cat_cut)/area_cut 
        target_skyden_list.append(target_skyden)

        unique_id_assigned = np.unique(assign_id_array)
        Nassigned_cut = np.sum(np.isin(unique_id_assigned, parent_cat_cut['gal_id']))
        completeness = Nassigned_cut/len(parent_cat_cut)
        print(f"rmagcut<{rmagcut:.2f}, completeness={completeness:.2f}")  # fiber assigned completeness on the galaxy sample
        completeness_list.append(completeness)
        
    ofile = "./output/completeness_diffmagcut.txt"
    output = np.array([rmagcut_list, target_skyden_list, completeness_list]).T
    header = " rmagcut  sky_density(deg^-2)  completeness"
    np.savetxt(ofile, output, fmt="%.7f", header=header)                                     
                                     
if __name__ == '__main__':
    main()

