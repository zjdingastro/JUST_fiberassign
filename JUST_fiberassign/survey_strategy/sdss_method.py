#!/usr/bin/env python
# coding: utf-8


"""
SDSS Tiling Algorithm Implementation
Based on Blanton et al. 2003, AJ, 125, 2276
"""

import os
import numpy as np
from astropy.table import Table
import networkx as nx
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Set
import random
from dataclasses import dataclass
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor


# ============================================================================ #
# Constants and Configuration
# ============================================================================ #
TILE_INNER_RADIUS_DEG = 0.1085       # Tile inner radius in degrees
TILE_OUTER_RADIUS_DEG = 0.5968       # Tile outer radius in degrees
TILE_RADIUS_RATIO = TILE_INNER_RADIUS_DEG/TILE_OUTER_RADIUS_DEG
TILE_MIDDLE_RADIUS_DEG = (TILE_INNER_RADIUS_DEG+TILE_OUTER_RADIUS_DEG)/2.0
FIBER_COLLISION_DISTANCE = 2.0  # Minimum fiber separation in mm
JUST_PLATESCALE = 128.0         # JUST plate scale um/arcsec
FIBER_COLLISION_ARCSEC = FIBER_COLLISION_DISTANCE * 1000.0/JUST_PLATESCALE   # Minimum fiber separation in arcseconds (~15.6 arcsec)
FIBER_COLLISION_DEG = FIBER_COLLISION_ARCSEC / 3600.0  
FIBERS_PER_TILE = 2184          # Available fibers for science targets (naive guess, 20% reserved for sky and standard stars)
COST_OVERFLOW = 1000.0          # Penalty cost for unassigned targets

class Priority(IntEnum):
    """Target priority levels (higher = more important)"""
    GALAXY = 10
    STAR = 1
    # LRG = 1
    # LOW_SB_GALAXY = 1
    # QSO = 2
    # HOT_STANDARD = 3
    # BROWN_DWARF = 3

# ============================================================================ #
# Data Structures
# ============================================================================ #

@dataclass
class Target:
    """Spectroscopic target"""
    id: int
    ra: float          # Right ascension (degrees)
    dec: float         # Declination (degrees)
    priority: Priority
    assigned_tile: Optional[int] = None
    in_decollided_set: bool = False
    collision_group_id: int = -1

    def distance_to(self, other: 'Target') -> float:
        """Angular distance to another target (small-patch approximation)"""
        return np.sqrt((self.ra - other.ra)**2 + (self.dec - other.dec)**2)

@dataclass
class Tile:
    """Spectroscopic tile (fiber plate)"""
    id: int
    ra: float          # Center position
    dec: float
    capacity: int = FIBERS_PER_TILE
    assigned_targets: List[Target] = None

    def __post_init__(self):
        if self.assigned_targets is None:
            self.assigned_targets = []

    def contains(self, target: Target) -> bool:
        """Check if target is within tile radius"""
        dist = np.sqrt((self.ra - target.ra)**2 + (self.dec - target.dec)**2)
        mask = (dist > TILE_INNER_RADIUS_DEG) & (dist <= TILE_OUTER_RADIUS_DEG)
        return mask

    def distance_to(self, target: Target) -> float:
        """Distance from tile center to target"""
        return np.sqrt((self.ra - target.ra)**2 + (self.dec - target.dec)**2)

    def remaining_capacity(self) -> int:
        return self.capacity - len(self.assigned_targets)

class CollisionGroup:
    """Group of targets within collision distance"""
    def __init__(self, group_id: int):
        self.group_id = group_id
        self.targets: List[Target] = []
        self.possible_tiles: Set[int] = set()

    def add_target(self, target: Target):
        self.targets.append(target)
        target.collision_group_id = self.group_id


@dataclass
class CostContext:
    """Reusable data needed by the positioning cost calculator."""
    coords: np.ndarray
    min_norm_dist: float = TILE_INNER_RADIUS_DEG/TILE_OUTER_RADIUS_DEG
    max_norm_dist: float = 2.5
    chunk_size: int = 512
    n_workers: int = 1


def build_cost_context(decollided_targets: List[Target],
                       chunk_size: int = 512,
                       n_workers: Optional[int] = None) -> CostContext:
    """
    Prepare cached coordinate arrays and threading parameters for cost evals.
    """
    coords = np.array([[t.ra, t.dec] for t in decollided_targets], dtype=np.float64)
    if n_workers is None:
        cpu_count = os.cpu_count() or 1
        n_workers = min(4, cpu_count)
    return CostContext(
        coords=coords,
        chunk_size=max(32, chunk_size),
        n_workers=max(1, n_workers),
    )

# ============================================================================ #
# Core Algorithm Components
# ============================================================================ #

def find_collision_groups(targets: List[Target], 
                         collision_radius: float = FIBER_COLLISION_DEG) -> List[CollisionGroup]:
    """
    Friends-of-friends algorithm to find collision groups.
    Uses KD-tree for O(N log N) performance.
    """
    positions = np.array([[t.ra, t.dec] for t in targets])
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=collision_radius)
    
    # Build adjacency graph
    adjacency = {i: set() for i in range(len(targets))}
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Find connected components
    visited = set()
    groups = []
    group_id = 0
    
    for i in range(len(targets)):
        if i not in visited:
            stack = [i]
            group = CollisionGroup(group_id)
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.add_target(targets[node])
                stack.extend(adjacency[node] - visited)
            
            if group.targets:
                groups.append(group)
                group_id += 1
    
    return groups

def find_decollided_set(targets: List[Target], 
                       groups: List[CollisionGroup]) -> List[Target]:
    """
    Find maximal decollided set (no internal collisions).
    Uses priority-weighted selection for each group.
    """
    decollided = []
    
    for group in groups:
        best_subset = _resolve_group_collisions(group.targets)
        for target in best_subset:
            target.in_decollided_set = True
            decollided.append(target)
    
    return decollided

def _resolve_group_collisions(targets: List[Target]) -> List[Target]:
    """Resolve collisions within a single group (brute force for small groups)"""
    if len(targets) == 1:
        return targets
    
    # Sort by priority for greedy fallback
    targets_sorted = sorted(targets, key=lambda t: t.priority, reverse=True)
    
    if len(targets) <= 4:
        # Brute force for small groups
        best_score = -1
        best_subset = []
        
        for k in range(1, len(targets) + 1):
            for combo in combinations(targets, k):
                # Check if valid (no collisions)
                if _is_valid_subset(combo):
                    # Score by sum of priorities
                    score = sum(t.priority for t in combo)
                    if score > best_score:
                        best_score = score
                        best_subset = list(combo)
        
        return best_subset if best_subset else [targets_sorted[0]]
    else:
        # Greedy for larger groups
        selected = []
        for target in targets_sorted:
            if all(target.distance_to(t) >= FIBER_COLLISION_DEG for t in selected):
                selected.append(target)
        return selected

def _is_valid_subset(targets: List[Target]) -> bool:
    """Check if all targets in subset are separated by > collision distance"""
    for i, t1 in enumerate(targets):
        for t2 in targets[i+1:]:
            if t1.distance_to(t2) < FIBER_COLLISION_DEG:
                return False
    return True

def build_assignment_network(targets: List[Target], 
                            tiles: List[Tile],
                            decollided_only: bool = False) -> nx.DiGraph:
    """
    Construct network flow graph for target-tile assignment.
    Source → Targets → Tiles → Sink
    """
    G = nx.DiGraph()
    
    # Count valid targets
    valid_targets = [t for t in targets if not decollided_only or t.in_decollided_set]
    n_targets = len(valid_targets)
    
    if n_targets == 0:
        return G
    
    # Source node: supply of n_targets
    G.add_node('source', demand=-n_targets)
    
    # Sink node: demand of n_targets (can be satisfied by tiles or overflow)
    G.add_node('sink', demand=n_targets)
    
    # Target nodes (demand = 0, they just pass flow through)
    for target in valid_targets:
        target_node = f't_{target.id}'
        G.add_node(target_node, demand=0)
        G.add_edge('source', target_node, capacity=1, weight=0)
        
        # Connect to covering tiles
        for tile in tiles:
            if tile.contains(target):
                tile_node = f'p_{tile.id}'
                if tile_node not in G:
                    G.add_node(tile_node, demand=0)
                G.add_edge(target_node, tile_node, capacity=1, weight=0)
    
    # Tile nodes to sink
    for tile in tiles:
        tile_node = f'p_{tile.id}'
        if tile_node in G:
            G.add_edge(tile_node, 'sink', capacity=tile.capacity, weight=0)
    
    # Overflow arc (unassigned targets go directly to sink with high cost)
    G.add_edge('source', 'sink', capacity=n_targets, weight=COST_OVERFLOW)
    
    return G

def solve_assignment(targets: List[Target],
                    tiles: List[Tile],
                    decollided_only: bool = False) -> int:
    """
    Solve network flow assignment problem.
    Returns number of successfully assigned targets.
    """
    # Build and solve network
    G = build_assignment_network(targets, tiles, decollided_only)
    
    if len(G.nodes()) == 0:
        return 0
    
    try:
        flow_dict = nx.min_cost_flow(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as e:
        print(f"Network flow error: {e}")
        return 0
    
    # Apply assignments
    if not decollided_only:
        # Clear previous assignments
        for tile in tiles:
            tile.assigned_targets.clear()
        for target in targets:
            target.assigned_tile = None
    
    assigned = 0
    valid_targets = [t for t in targets if not decollided_only or t.in_decollided_set]
    
    for target in valid_targets:
        target_node = f't_{target.id}'
        if target_node not in flow_dict:
            continue
        
        # Find which tile received the flow from this target
        # flow_dict[target_node] contains outgoing edges from target_node
        for tile in tiles:
            tile_node = f'p_{tile.id}'
            if (tile_node in flow_dict.get(target_node, {}) and 
                flow_dict[target_node][tile_node] > 0):
                # Check tile capacity
                if len(tile.assigned_targets) < tile.capacity:
                    tile.assigned_targets.append(target)
                    target.assigned_tile = tile.id
                    assigned += 1
                    break
    
    return assigned

def resolve_collision_groups(targets: List[Target],
                           tiles: List[Tile],
                           groups: List[CollisionGroup]) -> int:
    """
    Second network flow to resolve collisions in tile overlaps.
    """
    G = nx.DiGraph()
    
    total_demand = 0
    group_nodes = []
    
    for group in groups:
        # Only process groups in overlaps with remaining capacity
        overlap_tiles = set()
        for target in group.targets:
            for tile in tiles:
                if tile.contains(target) and tile.remaining_capacity() > 0:
                    overlap_tiles.add(tile.id)
        
        if len(overlap_tiles) <= 1:
            continue
        
        group_node = f'g_{group.group_id}'
        group_nodes.append((group, group_node, overlap_tiles))
        
        # Calculate capacities
        unassigned_in_group = sum(1 for t in group.targets if t.assigned_tile is None)
        c_max = min(unassigned_in_group, sum(tiles[tid].remaining_capacity() for tid in overlap_tiles))
        
        if c_max > 0:
            G.add_node(group_node, demand=0)
            G.add_edge('source', group_node, capacity=c_max, weight=0)
            total_demand += c_max
            
            for tile_id in overlap_tiles:
                tile_node = f'p_{tile_id}'
                if tile_node not in G:
                    G.add_node(tile_node, demand=0)
                
                max_to_tile = sum(1 for t in group.targets 
                                 if t.assigned_tile is None and tiles[tile_id].contains(t))
                if max_to_tile > 0:
                    G.add_edge(group_node, tile_node, capacity=max_to_tile, weight=0)
    
    if total_demand == 0 or len(G.nodes()) < 3:
        return 0
    
    # Source and sink nodes
    G.add_node('source', demand=-total_demand)
    G.add_node('sink', demand=total_demand)
    
    # Tile to sink edges
    for tile in tiles:
        if tile.remaining_capacity() > 0:
            tile_node = f'p_{tile.id}'
            if tile_node in G:
                G.add_edge(tile_node, 'sink', capacity=tile.remaining_capacity(), weight=0)
    
    try:
        flow_dict = nx.min_cost_flow(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as e:
        return 0
    
    # Apply collision resolutions
    assigned = 0
    for group, group_node, overlap_tiles in group_nodes:
        if group_node not in flow_dict:
            continue
        
        # Get flow to each tile
        for tile_id in overlap_tiles:
            tile = tiles[tile_id]
            tile_node = f'p_{tile_id}'
            
            if (tile_node in flow_dict.get(group_node, {}) and 
                flow_dict[group_node][tile_node] > 0):
                
                flow_amount = int(flow_dict[group_node][tile_node])
                
                # Assign targets from this group to tile
                for target in group.targets:
                    if flow_amount <= 0:
                        break
                    if (target.assigned_tile is None and 
                        tile.contains(target) and
                        tile.remaining_capacity() > 0 and
                        all(target.distance_to(t) >= FIBER_COLLISION_DEG 
                            for t in tile.assigned_targets)):
                        
                        tile.assigned_targets.append(target)
                        target.assigned_tile = tile.id
                        assigned += 1
                        flow_amount -= 1
    
    return assigned

# ============================================================================ #
# Tile Position Optimization
# ============================================================================ #

def _compute_min_distances(coords: np.ndarray,
                           tile_centers: np.ndarray,
                           chunk_size: int,
                           n_workers: int) -> np.ndarray:
    """
    Use a KDTree with chunked queries to find the nearest valid tile distance
    for each decollided target. Distances outside the annulus keep inf.
    """
    if coords.size == 0 or tile_centers.size == 0:
        return np.array([], dtype=np.float64)
    
    tree = cKDTree(tile_centers)
    n_targets = coords.shape[0]
    min_dists = np.full(n_targets, np.inf, dtype=np.float64)
    
    def process_chunk(start: int, chunk_coords: np.ndarray) -> Tuple[int, np.ndarray]:
        neighbor_lists = tree.query_ball_point(chunk_coords, TILE_OUTER_RADIUS_DEG)
        chunk_result = np.full(len(chunk_coords), np.inf, dtype=np.float64)
        for idx, neighbors in enumerate(neighbor_lists):
            if not neighbors:
                continue
            subset = tile_centers[neighbors]
            deltas = subset - chunk_coords[idx]
            dists = np.hypot(deltas[:, 0], deltas[:, 1])
            valid = (dists > TILE_INNER_RADIUS_DEG) & (dists <= TILE_OUTER_RADIUS_DEG)
            if np.any(valid):
                chunk_result[idx] = float(np.min(dists[valid]))
        return start, chunk_result
    
    # Sequential path for small problems
    if n_workers <= 1 or n_targets <= chunk_size:
        _, chunk_vals = process_chunk(0, coords)
        min_dists[:len(chunk_vals)] = chunk_vals
        return min_dists
    
    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for start in range(0, n_targets, chunk_size):
            chunk = coords[start:start + chunk_size]
            futures.append(executor.submit(process_chunk, start, chunk))
        for future in futures:
            start_idx, chunk_vals = future.result()
            min_dists[start_idx:start_idx + len(chunk_vals)] = chunk_vals
    
    return min_dists


def calculate_positioning_cost(tile_positions: np.ndarray,
                              context: CostContext,
                              alpha: float = 1.0) -> float:
    """
    Vectorized/parallel cost function for tile positioning optimization.
    Relies on a KDTree to find the nearest valid tile per decollided target.
    """
    if context.coords.size == 0:
        return 0.0
    
    tile_centers = tile_positions.reshape(-1, 2)
    min_dists = _compute_min_distances(
        coords=context.coords,
        tile_centers=tile_centers,
        chunk_size=context.chunk_size,
        n_workers=context.n_workers,
    )
    
    # Targets with no covering tile get overflow penalty
    n_targets = context.coords.shape[0]
    total_cost = np.full(n_targets, COST_OVERFLOW, dtype=np.float64)
    
    finite_mask = np.isfinite(min_dists)
    if not np.any(finite_mask):
        return float(np.sum(total_cost))
    
    r_norm = min_dists[finite_mask] / TILE_OUTER_RADIUS_DEG
    
    cost_values = np.zeros_like(r_norm)
    mask_far = r_norm > context.max_norm_dist
    cost_values[mask_far] = COST_OVERFLOW
    
    mask_outer = (~mask_far) & (r_norm > 1.0)
    cost_values[mask_outer] = COST_OVERFLOW * (np.power(r_norm[mask_outer], alpha) - 1.0)
    
    mask_mid = (~mask_far) & (~mask_outer) & (r_norm > context.min_norm_dist)
    cost_values[mask_mid] = 0.0
    
    mask_inner = ~(mask_far | mask_outer | mask_mid)
    if np.any(mask_inner):
        term = TILE_RADIUS_RATIO+1.0 - r_norm[mask_inner]
        cost_values[mask_inner] = COST_OVERFLOW * (np.power(term, alpha) - 1.0)
    
    total_cost[finite_mask] = cost_values
    return float(np.sum(total_cost))

def optimize_tile_positions(tiles: List[Tile],
                           targets: List[Target],
                           decollided_targets: List[Target],
                           alpha: float = 1.0,
                           max_iter: int = 100) -> Tuple[List[Tile], float]:
    """
    Iterative optimization of tile positions using Powell's method.
    """
    initial_positions = np.array([[t.ra, t.dec] for t in tiles]).flatten()
    
    # Define bounds to keep tiles near survey area
    ra_min, ra_max = min(t.ra for t in targets), max(t.ra for t in targets)
    dec_min, dec_max = min(t.dec for t in targets), max(t.dec for t in targets)
    
    bounds = []
    for i in range(len(tiles)):
        bounds.append((ra_min - TILE_OUTER_RADIUS_DEG, ra_max + TILE_OUTER_RADIUS_DEG))
        bounds.append((dec_min - TILE_OUTER_RADIUS_DEG, dec_max + TILE_OUTER_RADIUS_DEG))
    
    cost_context = build_cost_context(decollided_targets)
    if cost_context.coords.size == 0:
        return tiles, 0.0
    
    result = minimize(
        calculate_positioning_cost,
        initial_positions,
        args=(cost_context, alpha),
        method='Powell',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-4}
    )
    
    # Update final positions
    final_positions = result.x.reshape(-1, 2)
    for tile, (ra, dec) in zip(tiles, final_positions):
        tile.ra, tile.dec = ra, dec
    
    return tiles, result.fun

# ============================================================================ #
# Initialization and Main Algorithm
# ============================================================================ #

def initialize_uniform_tiles(targets: List[Target], 
                             ra_min: float = 150.0, ra_max: float = 154.0, 
                             dec_min: float = 0.0, dec_max: float = 4.0,
                           n_tiles: Optional[int] = None) -> List[Tile]:
    
    if n_tiles is None:
        # Estimate from target density and desired efficiency
        area = (ra_max - ra_min) * (dec_max - dec_min)
        target_density = len(targets) / area
        n_tiles_needed = int(len(targets) / FIBERS_PER_TILE)
        n_tiles_needed = max(n_tiles_needed, 1)
        print(f"Estimated {n_tiles_needed} tiles needed")
    
    # load tile file
    ifile = "./input/tiles_4passes_149.0ra161.0_-1.0dec11.0.fits"
    tiles = Table.read(ifile)
    mask = (tiles['RA'] > ra_min) & (tiles['RA'] < ra_max) & (tiles['DEC'] > dec_min) & (tiles['DEC'] < dec_max)
    
    ## modify it accordingly
    mask &= (tiles['PASSID'] < 2)
    
    
    tiles = tiles[mask]
    n_tiles = len(tiles)
    print(f"Loaded {n_tiles} tiles from file")
    
    tiles_sub = []

    for i in range(n_tiles):
        tiles_sub.append(Tile(i, tiles['RA'][i], tiles['DEC'][i]))
    
    # Add remaining tiles randomly
    # while len(tiles) < n_tiles:
    #     tiles.append(Tile(tile_id, 
    #                      np.random.uniform(ra_min, ra_max),
    #                      np.random.uniform(dec_min, dec_max)))
    #     tile_id += 1
    
    return tiles_sub

def run_tiling_pipeline(targets: List[Target],
                        ra_min: float = 150.0, ra_max: float = 154.0, 
                        dec_min: float = 0.0, dec_max: float = 4.0,
                        initial_tiles: Optional[List[Tile]] = None,
                        alpha: float = 1.0,
                        max_opt_iter: int = 100) -> Tuple[List[Tile], List[Target], dict]:
    """
    Execute complete tiling algorithm pipeline.
    
    Returns:
        tiles: Optimized tile positions
        targets: Targets with assignments
        stats: Dictionary of performance metrics
    """
    print("=== SDSS Tiling Algorithm ===")
    print(f"Processing {len(targets)} targets...")
    
    # Step 1: Initialize tiles
    if initial_tiles is None:
        tiles = initialize_uniform_tiles(targets, ra_min, ra_max, dec_min, dec_max)
    else:
        tiles = initial_tiles
    
    print(f"Initialized {len(tiles)} tiles")
    
    # Step 2: Find collision groups
    print("Finding collision groups...")
    groups = find_collision_groups(targets)
    multi_groups = [g for g in groups if len(g.targets) > 1]
    print(f"Found {len(groups)} groups ({len(multi_groups)} multi-target)")
    
    # Step 3: Determine decollided set
    print("Finding decollided set...")
    decollided = find_decollided_set(targets, groups)
    print(f"Decollided: {len(decollided)} ({len(decollided)/len(targets):.1%})")
    
    # Debug: Check tile coverage
    covered_targets = sum(1 for t in decollided 
                          if any(tile.contains(t) for tile in tiles))
    print(f"Decollided targets covered by tiles: {covered_targets}/{len(decollided)}")
    
    # Step 4: Initial assignment to decollided targets
    print("First network flow (decollided)...")
    assigned_decollided = solve_assignment(targets, tiles, decollided_only=True)
    print(f"Assigned: {assigned_decollided}/{len(decollided)} decollided targets")
    
    # Step 5: Optimize tile positions
    print("Optimizing tile positions...")
    tiles, final_cost = optimize_tile_positions(
        tiles, targets, decollided, alpha, max_opt_iter
    )
    print(f"Final optimization cost: {final_cost:.2f}")
    
    # Step 6: Reassign decollided targets with optimized tiles
    print("Reassigning with optimized tiles...")
    for tile in tiles:
        tile.assigned_targets.clear()
    for target in targets:
        target.assigned_tile = None
    
    assigned_decollided = solve_assignment(targets, tiles, decollided_only=True)
    
    # Step 7: Resolve collisions in overlaps
    print("Second network flow (collision resolution)...")
    resolved = resolve_collision_groups(targets, tiles, groups)
    print(f"Resolved {resolved} additional targets from collisions")
    
    # Calculate statistics
    total_assigned = sum(1 for t in targets if t.assigned_tile is not None)
    total_capacity = sum(t.capacity for t in tiles)
    used_capacity = sum(len(t.assigned_targets) for t in tiles)
    
    stats = {
        'n_targets': len(targets),
        'n_tiles': len(tiles),
        'decollided_fraction': len(decollided) / len(targets),
        'decollided_completeness': assigned_decollided / len(decollided),
        'total_completeness': total_assigned / len(targets),
        'efficiency': used_capacity / total_capacity,
        'collisions_resolved': resolved
    }
    
    print("\n=== Final Results ===")
    print(f"Targets assigned: {total_assigned}/{len(targets)} ({stats['total_completeness']:.1%})")
    print(f"Decollided completeness: {stats['decollided_completeness']:.1%}")
    print(f"Fiber efficiency: {stats['efficiency']:.1%}")
    
    return tiles, targets, stats

# ============================================================================ #
# Utility Functions
# ============================================================================ #

from itertools import combinations

def generate_test_sample(ra_min: float = 150.0, ra_max: float = 154.0, 
                        dec_min: float = 0.0, dec_max: float = 4.0) -> List[Target]:
    """Generate realistic test distribution with clustering."""
    targets = []
    
    ifile = "./input/gal_DR9_rmag19.5_149.0ra161.0_-1.0dec11.0.fits"
    gal_cat = Table.read(ifile)
    mask = (gal_cat['RA'] > ra_min) & (gal_cat['RA'] < ra_max) & (gal_cat['DEC'] > dec_min) & (gal_cat['DEC'] < dec_max)
    gal_cat = gal_cat[mask]
    N_gal = len(gal_cat)
    
    for i in range(N_gal):
        ra = gal_cat['RA'][i]
        dec = gal_cat['DEC'][i]
        target_type = Priority.GALAXY
        targets.append(Target(i, ra, dec, target_type))

    ifile = "./input/gaia_dr3_gmag19.0_149.0ra161.0_-1.0dec11.0.fits"
    star_cat = Table.read(ifile)
    mask = (star_cat['ra'] > ra_min) & (star_cat['ra'] < ra_max) & (star_cat['dec'] > dec_min) & (star_cat['dec'] < dec_max)
    star_cat = star_cat[mask][::5]  ## downsamping it for quick test
    N_star = len(star_cat)

    for j in range(N_star):
        ra = star_cat['ra'][j]
        dec = star_cat['dec'][j]
        target_type = Priority.STAR
        targets.append(Target(N_gal+j, ra, dec, target_type))

    return targets

def visualize_results(tiles: List[Tile], targets: List[Target], 
                     show_decollided: bool = True):
    """Simple matplotlib visualization (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot tiles
        for tile in tiles:
            circle = plt.Circle((tile.ra, tile.dec), TILE_OUTER_RADIUS_DEG, 
                              fill=False, color='blue', alpha=1.0)
            ax.add_patch(circle)
        
        # Plot targets
        assigned = [t for t in targets if t.assigned_tile is not None]
        unassigned = [t for t in targets if t.assigned_tile is None]
        decollided = [t for t in targets if t.in_decollided_set]
        
        if show_decollided:
            ax.scatter([t.ra for t in decollided], [t.dec for t in decollided], 
                      c='C1', s=2, alpha=0.5, label='Decollided')
        
        ax.scatter([t.ra for t in assigned], [t.dec for t in assigned], 
                  c='C2', s=2, alpha=0.7, label='Assigned')
        ax.scatter([t.ra for t in unassigned], [t.dec for t in unassigned], 
                  c='red', s=2, marker='x', label='Unassigned')
        
        ax.set_aspect('equal')
        ax.legend()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")


# ============================================================================ #
# Main Execution
# ============================================================================ #

def main():
    """Run example tiling problem"""
    print("Generating test sample...")
    targets = generate_test_sample(ra_min=150.0, ra_max=154.0, dec_min=0.0, dec_max=4.0)
    
    print("\nRunning tiling algorithm...")
    alpha = np.log(2.0)/np.log(2.5)   # parameter in the cost function for tiling scheme
    print("alpha:", alpha)
    
    tiles, targets, stats = run_tiling_pipeline(
        targets,
        ra_min=150.0, ra_max=154.0, dec_min=0.0, dec_max=4.0,
        alpha=alpha,
        max_opt_iter=50
    )
    
    # Optional visualization
    visualize_results(tiles, targets)

if __name__ == "__main__":
    main()





