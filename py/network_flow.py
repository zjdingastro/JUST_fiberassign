import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
sys.path.append("/home/zjding/installed_packages/JUST_fiberassign/py/")
from utils import get_fiberpos, xy2radec
import networkx as nx


def find_targets_in_tile(tile_ra, tile_dec, gal_ra, gal_dec, r_s, r_l):
    tile_coord = SkyCoord(tile_ra, tile_dec, frame='icrs', unit='deg')
    gal_coord = SkyCoord(gal_ra, gal_dec, frame='icrs', unit='deg')
    sep = tile_coord.separation(gal_coord).to(u.deg)
    r_s_deg = r_s * u.deg
    r_l_deg = r_l * u.deg
    mask = (sep > r_s_deg)&(sep < r_l_deg)
    return mask

def find_targets_in_patrol_radius(fiber_ra, fiber_dec, targets_ra, targets_dec, r_patrol_deg):
    fiber_coord = SkyCoord(fiber_ra, fiber_dec, frame='icrs', unit='deg')
    targets_coord = SkyCoord(targets_ra, targets_dec, frame='icrs', unit='deg')
    sep = fiber_coord.separation(targets_coord).to(u.deg)
    mask = (sep<r_patrol_deg*u.deg)
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
    tile_id, tile_ra, tile_dec, gal_cat, fiberpos_xy, TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG, r_patrol_deg = args
    
    targets_id_list_onetile = {}
    fibers_ra, fibers_dec = xy2radec(tile_ra, tile_dec, fiberpos_xy[:,0], fiberpos_xy[:, 1])

    ## find targets falling in the tile coverage 
    mask = find_targets_in_tile(tile_ra, tile_dec, gal_cat['RA'], gal_cat['DEC'], 
                                 TILE_INNER_RADIUS_DEG, TILE_OUTER_RADIUS_DEG)
    gal_in_tile = gal_cat[mask]
    targets_ra, targets_dec = gal_in_tile['RA'], gal_in_tile['DEC']

    ## find targets falling in each fiber patrol region
    fiber_id = 0    
    for ra, dec in zip(fibers_ra, fibers_dec):
        mask = find_targets_in_patrol_radius(ra, dec, targets_ra, targets_dec, r_patrol_deg)
        targets_id_list = gal_in_tile['gal_id'][mask].data.tolist()
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
                        sep_arcsec = coord_i.separation(coord_j).to(u.arcsec).value
                        
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
                        sep_arcsec = coord_i.separation(coord_j).to(u.arcsec).value
                        
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
                    sep_arcsec = coord_i.separation(coord_j).to(u.arcsec).value
                    
                    violations.append({
                        'tile_id': tile_id,
                        'fiber_i': fiber_i,
                        'fiber_j': fiber_j,
                        'target_id_i': target_id_i,
                        'target_id_j': target_id_j,
                        'separation_arcsec': sep_arcsec
                    })
    
    return violations


# Run the cell below for iterative optimization (simplified + parallelized version)

## Iterative optimization: feedback violating assignments, re-solve until no violations
def build_graph_with_forbidden_assignments(targets_id_list_alltiles, target_id_array_unique, priority_array, collision_constraints, 
                                           COST_OVERFLOW, N_fibers, forbidden_assignments=None):
    """
    Build a directed graph with collision constraints; can forbid specific assignments.

    Args:
        forbidden_assignments: Set of forbidden assignments, format {(tile_id, fiber_id, target_id), ...}

    Returns:
        G: NetworkX directed graph
    """
    if forbidden_assignments is None:
        forbidden_assignments = set()
    
    N_targets = len(target_id_array_unique)
    
    G = nx.DiGraph()
    
    # Source node: supply of n_targets
    G.add_node('source', demand=-N_targets)
    
    # Sink node: demand of n_targets (can be satisfied by tiles or overflow)
    G.add_node('sink', demand=N_targets)
    
    priority_dict = {}
    # Target nodes (demand = 0, they just pass flow through)
    for target_id, priority in zip(target_id_array_unique, priority_array):
        priority_dict[target_id] = priority
        target_node = f't_{target_id}'
        G.add_node(target_node, demand=0)
        G.add_edge('source', target_node, capacity=1, weight=0)
    
    # Add all normal edges (target to fiber), but skip forbidden assignments
    for tile_id in targets_id_list_alltiles.keys():
        G.add_node(tile_id, demand=0)
        for fiber_id in targets_id_list_alltiles[tile_id].keys():
            fiber_node = f"{tile_id}_{fiber_id}"
            G.add_node(fiber_node, demand=0)
            
            for target_id in targets_id_list_alltiles[tile_id][fiber_id]:
                # Check if this assignment is forbidden
                if (tile_id, fiber_id, target_id) in forbidden_assignments:
                    continue  # Skip forbidden assignment
                
                target_node = f't_{target_id}'  
                priority = priority_dict[target_id]
                G.add_edge(target_node, fiber_node, capacity=1, weight=1.0/priority)
    
            G.add_edge(fiber_node, tile_id, capacity=1, weight=0)
        G.add_edge(tile_id, 'sink', capacity=N_fibers, weight=0)
    
    # Add collision constraints: for each collision pair, create mutual exclusion constraint
    for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
        fiber_i_key = f'fiber_{fiber_i}'
        fiber_j_key = f'fiber_{fiber_j}'
        fiber_i_node = f"{tile_id}_{fiber_i_key}"
        fiber_j_node = f"{tile_id}_{fiber_j_key}"
        
        # Create constraint node for each collision pair
        for target_id_i, target_id_j in collision_pairs:
            # Check if both targets can be assigned to the corresponding fibers
            if (target_id_i in targets_id_list_alltiles[tile_id].get(fiber_i_key, []) and
                target_id_j in targets_id_list_alltiles[tile_id].get(fiber_j_key, [])):
                
                # Check if these assignments are forbidden
                if ((tile_id, fiber_i_key, target_id_i) in forbidden_assignments or
                    (tile_id, fiber_j_key, target_id_j) in forbidden_assignments):
                    continue  # If one is forbidden, skip this constraint
                
                target_i_node = f't_{target_id_i}'
                target_j_node = f't_{target_id_j}'
                
                # Create auxiliary node: aux node 1 has flow only when target_id_i is assigned to fiber_i
                aux_node_i = f"aux_{tile_id}_f{fiber_i}_t{target_id_i}"
                G.add_node(aux_node_i, demand=0)
                # G.add_edge(target_i_node, aux_node_i, capacity=1, weight=0)
                # G.add_edge(fiber_i_node, aux_node_i, capacity=1, weight=0)
                target_i_weight = 1.0/priority_dict[target_id_i]
                G.add_edge(target_i_node, fiber_i_node, capacity=1, weight=target_i_weight)
                G.add_edge(fiber_i_node, aux_node_i, capacity=1, weight=0)
                G.add_edge(aux_node_i, 'sink', capacity=1, weight=COST_OVERFLOW)
                
                # Create auxiliary node: aux node 2 has flow only when target_id_j is assigned to fiber_j
                aux_node_j = f"aux_{tile_id}_f{fiber_j}_t{target_id_j}"
                G.add_node(aux_node_j, demand=0)
                # G.add_edge(target_j_node, aux_node_j, capacity=1, weight=0)
                # G.add_edge(fiber_j_node, aux_node_j, capacity=1, weight=0)
                target_j_weight = 1.0/priority_dict[target_id_j]
                G.add_edge(target_j_node, fiber_j_node, capacity=1, weight=target_j_weight)
                G.add_edge(fiber_j_node, aux_node_j, capacity=1, weight=0)
                G.add_edge(aux_node_j, 'sink', capacity=1, weight=COST_OVERFLOW)
                
                # Create constraint node: ensure both auxiliary nodes cannot have flow simultaneously
                constraint_node = f"collision_{tile_id}_f{fiber_i}_f{fiber_j}_t{target_id_i}_t{target_id_j}"
                G.add_node(constraint_node, demand=0)
                G.add_edge(aux_node_i, constraint_node, capacity=1, weight=0)
                G.add_edge(aux_node_j, constraint_node, capacity=1, weight=0)
                #G.add_edge(constraint_node, 'sink', capacity=1, weight=COST_OVERFLOW)
                G.add_edge(constraint_node, 'sink', capacity=1, weight=0)
    
    # Overflow arc (unassigned targets go directly to sink with high cost)
    G.add_edge('source', 'sink', capacity=N_targets, weight=COST_OVERFLOW)
    
    return G

def extract_assigned_targets_from_flow(target_id_array_unique, flow_dict):
    """Extract assignment info from flow_dict"""
    target_to_fiber = {}  # {target_id: (tile_id, fiber_node)}
    assigned_targets = []
    
    for target_id in target_id_array_unique:
        target_node = f't_{target_id}'
        if flow_dict.get('source', {}).get(target_node, 0) == 1:
            assigned_targets.append(target_id)
            
            target_flow = flow_dict.get(target_node, {})
            for next_node, flow_value in target_flow.items():
                if flow_value == 1 and next_node.startswith('tile_') and '_fiber_' in next_node:
                    target_to_fiber[target_id] = next_node
                    break
    
    return assigned_targets, target_to_fiber

# Simplified + parallelized: single greedy strategy, multiprocessing.Pool
def eval_cost_extra(targets_id_list_alltiles, target_id_array_unique, priority_array, 
                     collision_constraints, COST_OVERFLOW, N_fibers, forbidden_base, opt):
    """Evaluate min-cost flow with forbidden_base plus the single extra forbidden assignment opt."""
    try:
        G = build_graph_with_forbidden_assignments(targets_id_list_alltiles, target_id_array_unique, priority_array, 
                                                   collision_constraints, COST_OVERFLOW, N_fibers, forbidden_base | {opt})
        return nx.min_cost_flow_cost(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return float('inf')

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

