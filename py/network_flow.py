import sys
import numpy as np
import astropy.units as units
from collections import deque
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import multiprocessing
from utils import get_fiberpos, xy2radec
import networkx as nx


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

def find_targets_in_one_tile(args): 
    tile_id, tile_ra, tile_dec, gal_cat, fiberpos_xy, TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, r_patrol_deg, TARGETID = args
    
    targets_id_list_onetile = {}
    fibers_ra, fibers_dec = xy2radec(tile_ra, tile_dec, fiberpos_xy[:,0], fiberpos_xy[:, 1])

    ## find targets falling in the tile coverage 
    mask = mask_targets_in_tile(tile_ra, tile_dec, gal_cat['RA'], gal_cat['DEC'], 
                                 TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG)
    gal_in_tile = gal_cat[mask]
    targets_ra, targets_dec = gal_in_tile['RA'], gal_in_tile['DEC']

    ## find targets falling in each fiber patrol region
    fiber_id = 0    
    for ra, dec in zip(fibers_ra, fibers_dec):
        mask = find_targets_in_patrol_radius(ra, dec, targets_ra, targets_dec, r_patrol_deg)
        targets_id_list = gal_in_tile[TARGETID][mask].data.tolist()
        if len(targets_id_list)>0:
            targets_id_list_onetile[f'fiber_{fiber_id}']=targets_id_list
        fiber_id += 1
    
    return f'tile_{tile_id}', targets_id_list_onetile

def find_collided_pairs_in_one_tile(args):
    """Compute collision constraints for a single tile in parallel"""
    tile_id, tile_idx, tiles_ra, tiles_dec, targets_id_list_alltiles, neighboring_fiber_pairs, target_positions, fiberpos_xy, COLLISION_SEPARATION_ARCSEC = args
    
    tile_ra, tile_dec = tiles_ra[tile_idx], tiles_dec[tile_idx]
    
    # Get RA/DEC positions of all fibers in this tile
    fibers_ra, fibers_dec = xy2radec(tile_ra, tile_dec, fiberpos_xy[:,0], fiberpos_xy[:, 1])
    
    tile_collision_constraints = {}
    
    for fiber_i, fiber_j in neighboring_fiber_pairs:
        fiber_i_key = f'fiber_{fiber_i}'
        fiber_j_key = f'fiber_{fiber_j}'

        # Check if both fibers have assignable targets in this tile
        if (fiber_i_key in targets_id_list_alltiles[tile_id] and 
            fiber_j_key in targets_id_list_alltiles[tile_id]):
            
            targets_i = targets_id_list_alltiles[tile_id][fiber_i_key]
            targets_j = targets_id_list_alltiles[tile_id][fiber_j_key]
            
            collision_pairs = []
            for target_id_i in targets_i:
                for target_id_j in targets_j:
                    if target_id_i != target_id_j:
                        # Compute angular separation between the two targets
                        ra_i, dec_i = target_positions[target_id_i]
                        ra_j, dec_j = target_positions[target_id_j]
                        
                        coord_i = SkyCoord(ra_i, dec_i, frame='icrs', unit='deg')
                        coord_j = SkyCoord(ra_j, dec_j, frame='icrs', unit='deg')
                        sep_arcsec = coord_i.separation(coord_j).to(units.arcsec).value
                        
                        # Record as collision pair if angular separation < separation threshold
                        if sep_arcsec < COLLISION_SEPARATION_ARCSEC:
                            collision_pairs.append((target_id_i, target_id_j))
            
            if len(collision_pairs) > 0:
                tile_collision_constraints[(tile_id, fiber_i, fiber_j)] = collision_pairs
    
    return tile_collision_constraints

## Compute collision constraints
## For each neighboring fiber pair, check which target pairs would collide
## If angular separation between two targets < 15.625 arcsec, they cannot both be assigned to neighboring fibers
def find_collided_pairs_in_one_tile(args):
    """Compute collision constraints for a single tile, for parallelization"""
    tile_id, tile_idx, tiles_ra, tiles_dec, targets_id_list_alltiles, neighboring_fiber_pairs, target_positions, fiberpos_xy, COLLISION_SEPARATION_ARCSEC = args
    
    tile_ra, tile_dec = tiles_ra[tile_idx], tiles_dec[tile_idx]
    
    # Get RA/DEC positions of all fibers in this tile
    fibers_ra, fibers_dec = xy2radec(tile_ra, tile_dec, fiberpos_xy[:,0], fiberpos_xy[:, 1])
    
    tile_collision_constraints = {}
    
    for fiber_i, fiber_j in neighboring_fiber_pairs:
        fiber_i_key = f'fiber_{fiber_i}'
        fiber_j_key = f'fiber_{fiber_j}'

        # Check if both fibers have assignable targets in this tile
        if (fiber_i_key in targets_id_list_alltiles[tile_id] and 
            fiber_j_key in targets_id_list_alltiles[tile_id]):
            
            targets_i = targets_id_list_alltiles[tile_id][fiber_i_key]
            targets_j = targets_id_list_alltiles[tile_id][fiber_j_key]
            
            collision_pairs = []
            for target_id_i in targets_i:
                for target_id_j in targets_j:
                    if target_id_i != target_id_j:
                        # Compute angular separation between the two targets
                        ra_i, dec_i = target_positions[target_id_i]
                        ra_j, dec_j = target_positions[target_id_j]
                        
                        coord_i = SkyCoord(ra_i, dec_i, frame='icrs', unit='deg')
                        coord_j = SkyCoord(ra_j, dec_j, frame='icrs', unit='deg')
                        sep_arcsec = coord_i.separation(coord_j).to(units.arcsec).value
                        
                        # Record as collision pair if angular separation < separation threshold
                        if sep_arcsec < COLLISION_SEPARATION_ARCSEC:
                            collision_pairs.append((target_id_i, target_id_j))
            
            if len(collision_pairs) > 0:
                tile_collision_constraints[(tile_id, fiber_i, fiber_j)] = collision_pairs
    
    return tile_collision_constraints

## Verify that assignment results satisfy collision constraints

def check_collision_constraints(flow_dict, target_id_array_unique, targets_id_list_alltiles, 
                                collision_constraints, target_positions):
    """
    Check whether assignment results satisfy collision constraints.

    Args:
        flow_dict: Flow dict returned by NetworkX min_cost_flow
        target_id_array_unique: Array of all unique target IDs
        targets_id_list_alltiles: Target assignment dict
        collision_constraints: Collision constraints dict
        target_positions: Target positions dict

    Returns:
        violations: List of assignments that violate constraints
    """
    if flow_dict is None:
        return []
    
    # Extract assignment info: which targets are assigned to which fibers
    target_to_fiber = {}  # {target_id: (tile_id, fiber_id)}
    
    for target_id in target_id_array_unique:
        target_node = f't_{target_id}'
        if flow_dict.get('source', {}).get(target_node, 0) == 1:
            # Target is active, find which fiber it is assigned to
            target_flow = flow_dict.get(target_node, {})
            for next_node, flow_value in target_flow.items():
                if flow_value == 1 and next_node.startswith('tile_') and '_fiber_' in next_node:
                    target_to_fiber[target_id] = next_node
                    break
    
    violations = []
    
    # Check each collision constraint
    for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
        fiber_i_key = f'fiber_{fiber_i}'
        fiber_j_key = f'fiber_{fiber_j}'
        fiber_i_node = f"{tile_id}_{fiber_i_key}"
        fiber_j_node = f"{tile_id}_{fiber_j_key}"
        
        for target_id_i, target_id_j in collision_pairs:
            # Check if both targets are assigned to neighboring fibers
            if (target_id_i in target_to_fiber and target_id_j in target_to_fiber):
                fiber_i_assigned = target_to_fiber[target_id_i] == fiber_i_node
                fiber_j_assigned = target_to_fiber[target_id_j] == fiber_j_node
                
                if fiber_i_assigned and fiber_j_assigned:
                    # Constraint violation: both targets assigned to neighboring fibers
                    # Compute actual angular separation
                    ra_i, dec_i = target_positions[target_id_i]
                    ra_j, dec_j = target_positions[target_id_j]
                    coord_i = SkyCoord(ra_i, dec_i, frame='icrs', unit='deg')
                    coord_j = SkyCoord(ra_j, dec_j, frame='icrs', unit='deg')
                    sep_arcsec = coord_i.separation(coord_j).to(units.arcsec).value
                    
                    violations.append({
                        'tile_id': tile_id,
                        'fiber_i': fiber_i,
                        'fiber_j': fiber_j,
                        'target_id_i': target_id_i,
                        'target_id_j': target_id_j,
                        'separation_arcsec': sep_arcsec
                    })
    
    return violations

# Enforce global per-fiber capacity via node splitting.
# A physical fiber is modeled as:
#   fiber_in (receives target flow) -> fiber_out (capacity=1) -> downstream arcs
# so at most one unit can traverse each physical fiber.
def build_graph_with_forbidden_assignments(
    targets_id_list_alltiles,
    target_id_array_unique,
    priority_array,
    collision_constraints,
    COST_OVERFLOW,
    N_fibers,
    forbidden_assignments=None,
):
    if forbidden_assignments is None:
        forbidden_assignments = set()

    N_targets = len(target_id_array_unique)
    G = nx.DiGraph()

    G.add_node("source", demand=-N_targets)
    G.add_node("sink", demand=N_targets)

    priority_dict = {}
    for target_id, priority in zip(target_id_array_unique, priority_array):
        priority_dict[target_id] = float(priority)
        target_node = f"t_{target_id}"
        G.add_node(target_node, demand=0)
        G.add_edge("source", target_node, capacity=1, weight=0)

    # Build tile/fiber structure once. Target->fiber edges go to fiber_in.
    # All downstream edges originate at fiber_out and are bottlenecked by
    # fiber_in->fiber_out (capacity=1), enforcing global fiber usage.
    for tile_id in targets_id_list_alltiles.keys():
        G.add_node(tile_id, demand=0)
        for fiber_id in targets_id_list_alltiles[tile_id].keys():
            fiber_in = f"{tile_id}_{fiber_id}"
            fiber_out = f"{fiber_in}_out"
            G.add_node(fiber_in, demand=0)
            G.add_node(fiber_out, demand=0)
            G.add_edge(fiber_in, fiber_out, capacity=1, weight=0)

            for target_id in targets_id_list_alltiles[tile_id][fiber_id]:
                if (tile_id, fiber_id, target_id) in forbidden_assignments:
                    continue
                target_node = f"t_{target_id}"
                priority = max(priority_dict[target_id], 1e-12)
                G.add_edge(target_node, fiber_in, capacity=1, weight=1.0 / priority)

            G.add_edge(fiber_out, tile_id, capacity=1, weight=0)

        G.add_edge(tile_id, "sink", capacity=N_fibers, weight=0)

    # Collision gadgets: connect from fiber_out so they also consume the same
    # single fiber capacity budget.
    for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
        fiber_i_key = f"fiber_{fiber_i}"
        fiber_j_key = f"fiber_{fiber_j}"
        fiber_i_in = f"{tile_id}_{fiber_i_key}"
        fiber_j_in = f"{tile_id}_{fiber_j_key}"
        fiber_i_out = f"{fiber_i_in}_out"
        fiber_j_out = f"{fiber_j_in}_out"

        for target_id_i, target_id_j in collision_pairs:
            if (
                target_id_i in targets_id_list_alltiles[tile_id].get(fiber_i_key, [])
                and target_id_j in targets_id_list_alltiles[tile_id].get(fiber_j_key, [])
            ):
                if (
                    (tile_id, fiber_i_key, target_id_i) in forbidden_assignments
                    or (tile_id, fiber_j_key, target_id_j) in forbidden_assignments
                ):
                    continue

                aux_node_i = f"aux_{tile_id}_f{fiber_i}_t{target_id_i}"
                aux_node_j = f"aux_{tile_id}_f{fiber_j}_t{target_id_j}"
                constraint_node = (
                    f"collision_{tile_id}_f{fiber_i}_f{fiber_j}_t{target_id_i}_t{target_id_j}"
                )

                G.add_node(aux_node_i, demand=0)
                G.add_node(aux_node_j, demand=0)
                G.add_node(constraint_node, demand=0)

                G.add_edge(fiber_i_out, aux_node_i, capacity=1, weight=0)
                G.add_edge(aux_node_i, "sink", capacity=1, weight=COST_OVERFLOW)

                G.add_edge(fiber_j_out, aux_node_j, capacity=1, weight=0)
                G.add_edge(aux_node_j, "sink", capacity=1, weight=COST_OVERFLOW)

                G.add_edge(aux_node_i, constraint_node, capacity=1, weight=0)
                G.add_edge(aux_node_j, constraint_node, capacity=1, weight=0)
                G.add_edge(constraint_node, "sink", capacity=1, weight=0)

    G.add_edge("source", "sink", capacity=N_targets, weight=COST_OVERFLOW)
    return G

def eval_cost_extra(
    targets_id_list_alltiles,
    target_id_array_unique,
    priority_array,
    collision_constraints,
    COST_OVERFLOW,
    N_fibers,
    forbidden_base,
    opt,
):
    """Evaluate min-cost-flow with forbidden_base plus one extra forbidden assignment."""
    try:
        G = build_graph_with_forbidden_assignments(
            targets_id_list_alltiles,
            target_id_array_unique,
            priority_array,
            collision_constraints,
            COST_OVERFLOW,
            N_fibers,
            forbidden_base | {opt},
        )
        return nx.min_cost_flow_cost(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return float("inf")

def _is_physical_fiber_node(name: str) -> bool:
    """True for real fiber nodes (e.g. tile_0_fiber_12), not collision/auxiliary nodes."""
    s = str(name)
    if s.startswith("collision_"):
        return False
    return s.startswith("tile_") and "_fiber_" in s


def _fiber_reachable_via_flow(flow_dict, start_node, max_hops=512):
    """
    Walk outgoing edges with positive flow from start_node until a physical fiber
    node is found. Needed when the graph routes t_* -> auxiliary (e.g. collision_* ) -> tile_*_fiber_*.
    """
    if _is_physical_fiber_node(start_node):
        return start_node
    q = deque([(start_node, 0)])
    seen = {start_node}
    while q:
        u, h = q.popleft()
        if h >= max_hops:
            continue
        for v, f in (flow_dict.get(u) or {}).items():
            if f <= 0:
                continue
            if _is_physical_fiber_node(v):
                return v
            if v in ("source", "sink"):
                continue
            if v not in seen:
                seen.add(v)
                q.append((v, h + 1))
    return None

def extract_assigned_targets_from_flow(target_id_array_unique, flow_dict):
    """Extract assignment info from flow_dict, including paths through auxiliary nodes."""
    target_to_fiber = {}
    assigned_targets_id = []

    for target_id in target_id_array_unique:
        target_node = f"t_{target_id}"
        if flow_dict.get("source", {}).get(target_node, 0) != 1:
            continue
        fiber_node = _fiber_reachable_via_flow(flow_dict, target_node)
        if fiber_node is not None:
            assigned_targets_id.append(target_id)
            target_to_fiber[target_id] = fiber_node

    return assigned_targets_id, target_to_fiber


#  returns the candidate forbidden assignments that can resolve a single collision violation.
def opts_fun(v, tf):
    # v: A violation dict with keys:
    # tile_id, fiber_i, fiber_j, target_id_i, target_id_j, separation_arcsec
    # tf: target_to_fiber mapping {target_id: fiber_node} (e.g. "tile_0_fiber_123"), i.e. the current assignment from the flow solution.
    # returns the candidate forbidden assignments that can resolve a single collision violation.
    fi, fj = f"fiber_{v['fiber_i']}", f"fiber_{v['fiber_j']}"
    ni, nj = f"{v['tile_id']}_{fi}", f"{v['tile_id']}_{fj}"
    out = []
    if tf.get(v['target_id_i']) == ni: out.append((v['tile_id'], fi, v['target_id_i']))
    if tf.get(v['target_id_j']) == nj: out.append((v['tile_id'], fj, v['target_id_j']))
    return out


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

def solve_tile_group(target_positions, group_tile_indices, tiles_ra, tiles_dec, targets_id_list_alltiles,
                     target_id_array_unique, priority_array, collision_constraints,
                     COST_OVERFLOW, N_fibers, max_iterations=10, n_workers=4):
    """
    Solve fiber assignment for a single group of overlapped tiles.

    Returns:
        flow_dict, cost, forbidden_assignments, n_assigned, n_used_fibers
    """
    # Build tile_id list for this group
    group_tile_ids = [f'tile_{idx}' for idx in group_tile_indices]
    
    # Filter targets_id_list_alltiles to only include tiles in this group
    group_targets_id_list = {tid: targets_id_list_alltiles[tid] for tid in group_tile_ids if tid in targets_id_list_alltiles}
    
    if len(group_targets_id_list) == 0:
        return None, 0, set(), 0, 0
    
    # Find targets that belong to this group (targets covered by any tile in the group)
    group_target_ids_set = set()
    for tile_id in group_tile_ids:
        if tile_id in targets_id_list_alltiles:
            for fiber_id, target_list in targets_id_list_alltiles[tile_id].items():
                group_target_ids_set.update(target_list)
    
    # Filter target_id_array_unique and priority_array to only include targets in this group
    mask = np.isin(target_id_array_unique, list(group_target_ids_set))
    group_targets_id_unique = target_id_array_unique[mask]
    group_priority_unique = priority_array[mask]

    # Deduplicate by target id (keep first occurrence). Duplicate rows inflate n_assigned vs n_used_fibers.
    ## Is the following code necessary?
    # if len(group_targets_id_unique) > 0:
    #     _, first_idx = np.unique(group_targets_id_unique, return_index=True)
    #     keep = np.sort(first_idx)
    #     group_targets_id_unique = group_targets_id_unique[keep]
    #     group_priority_unique = group_priority_unique[keep]

    if len(group_targets_id_unique) == 0:
        return None, 0, set(), 0, 0
    
    # Filter collision_constraints to only include constraints for tiles in this group
    group_collision_constraints = {}
    for (tile_id, fiber_i, fiber_j), pairs in collision_constraints.items():
        if tile_id in group_tile_ids:
            group_collision_constraints[(tile_id, fiber_i, fiber_j)] = pairs
    
    # Run iterative optimization for this group
    forbidden_assignments = set()
    flow_dict_final = None
    cost_final = None
    
    for iteration in range(max_iterations):
        G = build_graph_with_forbidden_assignments(
            group_targets_id_list, group_targets_id_unique, group_priority_unique,
            group_collision_constraints, COST_OVERFLOW, N_fibers, forbidden_assignments
        )
        
        try:
            flow_dict = nx.min_cost_flow(G)
            cost = nx.min_cost_flow_cost(G)
        except (nx.NetworkXUnfeasible, nx.NetworkXError) as e:
            print(f"  Group solve failed: {e}")
            break
        
        # Check collision constraints - need to use full data structures but only check for this group's tiles
        violations = check_collision_constraints(
            flow_dict, group_targets_id_unique, group_targets_id_list,
            group_collision_constraints, target_positions
        )
        
        if len(violations) == 0:
            flow_dict_final = flow_dict
            cost_final = cost
            break
        
        _, target_to_fiber = extract_assigned_targets_from_flow(group_targets_id_unique, flow_dict)
        
        # Prepare evaluation tasks
        tasks = [(group_targets_id_list, group_targets_id_unique, group_priority_unique,
                  group_collision_constraints, COST_OVERFLOW, N_fibers, forbidden_assignments, opt)
                 for v in violations for opt in opts_fun(v, target_to_fiber)]
        
        best_costs = {}
        if tasks:
            with multiprocessing.Pool(processes=min(n_workers, len(tasks))) as pool:
                results = pool.starmap(eval_cost_extra, tasks)
            for task, c in zip(tasks, results):
                opt = task[-1]
                if opt not in best_costs or c < best_costs[opt][0]:
                    best_costs[opt] = (c, opt)
        
        new_forbidden = set()
        for v in violations:
            opts = opts_fun(v, target_to_fiber)
            best_opt, best_c = None, float('inf')
            for opt in opts:
                if opt in best_costs and best_costs[opt][0] < best_c:
                    best_c, best_opt = best_costs[opt][0], opt
            if best_opt:
                new_forbidden.add(best_opt)
        
        forbidden_assignments.update(new_forbidden)
        flow_dict_final, cost_final = flow_dict, cost
    
    # Count assigned targets / distinct fibers (follows same paths as extract_assigned_targets_from_flow)
    n_assigned = 0
    n_used_fibers = 0
    if flow_dict_final is not None:
        _, _tf = extract_assigned_targets_from_flow(
            group_targets_id_unique, flow_dict_final
        )
        n_assigned = len(_tf)
        n_used_fibers = len(set(_tf.values()))
    return flow_dict_final, cost_final, forbidden_assignments, n_assigned, n_used_fibers
