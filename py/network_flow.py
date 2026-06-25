import sys
import numpy as np
import astropy.units as units
from collections import deque
import multiprocessing
import networkx as nx


def _angular_sep_arcsec_scalar(ra_i_deg, dec_i_deg, ra_j_deg, dec_j_deg):
    """Great-circle separation in arcsec (matches SkyCoord to ~1e-5 arcsec)."""
    ra1, dec1 = np.deg2rad(ra_i_deg), np.deg2rad(dec_i_deg)
    ra2, dec2 = np.deg2rad(ra_j_deg), np.deg2rad(dec_j_deg)
    dot = np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2) + np.sin(dec1) * np.sin(dec2)
    return float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))) * 3600.0)

## Verify that assignment results satisfy collision constraints
def check_collision_constraints(flow_dict, target_id_array_unique,
                                collision_constraints, target_positions):
    """
    Check whether assignment results satisfy collision constraints.

    Args:
        flow_dict: Flow dict returned by NetworkX min_cost_flow
        target_id_array_unique: Array of all unique target IDs
        collision_constraints: Collision constraints dict
        target_positions: Target positions dict

    Returns:
        violations: List of assignments that violate constraints
    """
    if flow_dict is None:
        return []

    source_flow = flow_dict.get("source") or {}
    target_to_fiber = {}
    for target_id in target_id_array_unique:
        target_node = f"t_{target_id}"
        if source_flow.get(target_node, 0) != 1:
            continue
        for next_node, flow_value in (flow_dict.get(target_node) or {}).items():
            if flow_value == 1 and next_node.startswith("tile_") and "_fiber_" in next_node:
                target_to_fiber[target_id] = next_node
                break

    fiber_to_targets = {}
    for target_id, fiber_node in target_to_fiber.items():
        fiber_to_targets.setdefault(fiber_node, set()).add(target_id)

    violations = []
    for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
        fiber_i_node = f"{tile_id}_fiber_{fiber_i}"
        fiber_j_node = f"{tile_id}_fiber_{fiber_j}"
        targets_on_i = fiber_to_targets.get(fiber_i_node)
        targets_on_j = fiber_to_targets.get(fiber_j_node)
        if not targets_on_i or not targets_on_j:
            continue

        for target_id_i, target_id_j in collision_pairs:
            if target_id_i not in targets_on_i or target_id_j not in targets_on_j:
                continue

            ra_i, dec_i = target_positions[target_id_i]
            ra_j, dec_j = target_positions[target_id_j]
            violations.append({
                "tile_id": tile_id,
                "fiber_i": fiber_i,
                "fiber_j": fiber_j,
                "target_id_i": target_id_i,
                "target_id_j": target_id_j,
                "separation_arcsec": _angular_sep_arcsec_scalar(ra_i, dec_i, ra_j, dec_j),
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
                G.add_edge(target_node, fiber_in, capacity=1, weight = -1.0*priority)

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




def solve_tile_group(target_positions, group_tile_indices, targets_id_list_alltiles,
                     target_id_array_unique, priority_array, collision_constraints,
                     COST_OVERFLOW, N_fibers, max_iterations=10, n_workers=4, tiles_id=None):
    """
    Solve fiber assignment for a single group of overlapped tiles.

    Returns:
        flow_dict, cost, forbidden_assignments, n_assigned, n_used_fibers
    """
    # Build tile_id list for this group
    if tiles_id is None:
        group_tile_ids = [f'tile_{idx}' for idx in group_tile_indices]
    else:
        group_tile_ids = [f'tile_{int(tiles_id[idx])}' for idx in group_tile_indices]
    
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


    if len(group_targets_id_unique) == 0:
        return None, 0, set(), 0, 0
    
    # Filter collision_constraints to only include constraints for tiles in this group
    group_collision_constraints = {}
    for (tile_id, fiber_i, fiber_j), pairs in collision_constraints.items():
        if tile_id in group_tile_ids:
            group_collision_constraints[(tile_id, fiber_i, fiber_j)] = pairs
    
    group_priority_map = {
        int(tid): float(pr)
        for tid, pr in zip(group_targets_id_unique, group_priority_unique)
    }

    # Run iterative optimization for this group
    forbidden_assignments = set()
    flow_dict_final = None
    cost_final = None
    prev_n_violations = None
    
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
            flow_dict, group_targets_id_unique, group_collision_constraints, target_positions
        )
        
        if len(violations) == 0:
            flow_dict_final = flow_dict
            cost_final = cost
            break
        
        _, target_to_fiber = extract_assigned_targets_from_flow(group_targets_id_unique, flow_dict)
        
        # Prepare unique candidate options for evaluation.
        # Many violations map to the same (tile, fiber, target) forbid, so evaluating
        # duplicates wastes expensive min-cost-flow solves and causes large seed variance.
        unique_opts = []
        seen_opts = set()
        for v in violations:
            for opt in opts_fun(v, target_to_fiber):
                if opt in seen_opts:
                    continue
                seen_opts.add(opt)
                unique_opts.append(opt)

        # Bound expensive eval_cost_extra solves. Prefer forbidding lower-priority
        # targets first because these are more likely to be removed by repair anyway.
        if max_eval_options is not None and max_eval_options > 0 and len(unique_opts) > max_eval_options:
            unique_opts = sorted(
                unique_opts,
                key=lambda opt: group_priority_map.get(int(opt[2]), float("inf"))
            )[:max_eval_options]

        tasks = [
            (
                group_targets_id_list,
                group_targets_id_unique,
                group_priority_unique,
                group_collision_constraints,
                COST_OVERFLOW,
                N_fibers,
                forbidden_assignments,
                opt,
            )
            for opt in unique_opts
        ]

        print(
            f"  Iteration {iteration + 1}/{max_iterations}: "
            f"violations={len(violations)}, unique_opts={len(unique_opts)}, "
            f"eval_workers={eval_workers}",
            flush=True,
        )

        best_costs = {}
        if tasks:
            if eval_workers <= 1:
                results = [eval_cost_extra(*task) for task in tasks]
            else:
                with multiprocessing.Pool(processes=min(eval_workers, len(tasks))) as pool:
                    results = pool.starmap(eval_cost_extra, tasks)
            for task, c in zip(tasks, results):
                opt = task[-1]
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
        
        # No progress means the next iteration will repeat the same graph/evaluations.
        # Stop early to avoid spending the full max_iterations with identical work.
        if not new_forbidden:
            flow_dict_final, cost_final = flow_dict, cost
            break

        # If violations do not decrease, stop iterating and let pairwise repair
        # finish residual conflicts outside this optimizer loop.
        if prev_n_violations is not None and len(violations) >= prev_n_violations:
            flow_dict_final, cost_final = flow_dict, cost
            break
        prev_n_violations = len(violations)

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


def aggregate_group_assignments_with_pairwise_repair(
    flow_dict,
    target_ids,
    all_forbidden,
    collision_constraints,
    gal_cat,
    apply_repair=True,
    verbose=True,
):
    # Aggregate assigned targets/target_to_fiber from a single flow dictionary.
    node_to_target_id = {f"t_{tid}": int(tid) for tid in target_ids}
    assigned_targets_set = set()
    target_to_fiber = {}

    source_flow = flow_dict.get("source", {}) if flow_dict else {}
    for target_node, source_val in source_flow.items():
        if source_val != 1 or target_node not in node_to_target_id:
            continue

        target_flow = flow_dict.get(target_node, {})
        for next_node, flow_value in target_flow.items():
            if flow_value == 1 and next_node.startswith("tile_") and "_fiber_" in next_node:
                target_id = node_to_target_id[target_node]
                assigned_targets_set.add(target_id)
                target_to_fiber[target_id] = next_node
                break

    assigned_targets_id = np.asarray(list(assigned_targets_set))

    def count_violations(current_target_to_fiber):
        count = 0
        for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
            fiber_i_node = f"{tile_id}_fiber_{fiber_i}"
            fiber_j_node = f"{tile_id}_fiber_{fiber_j}"
            for target_id_i, target_id_j in collision_pairs:
                if (
                    current_target_to_fiber.get(target_id_i) == fiber_i_node
                    and current_target_to_fiber.get(target_id_j) == fiber_j_node
                ):
                    count += 1
        return count

    violations_before = count_violations(target_to_fiber)

    if verbose:
        print(f"\n=== Single Tile/Group Complete ===")
        print(
            "All constraints satisfied."
            if violations_before == 0
            else f"Remaining violations: {violations_before}"
        )

    removed_targets = set()
    violations_after = violations_before

    # Resolve each active collision pair by removing lower PRIORITY; if tied, remove lower
    # SUBPRIORITY; if still tied, remove larger TARGETID.
    if apply_repair and violations_before > 0:
        _pmask = np.isin(gal_cat["TARGETID"], target_ids)
        target_priority = {}
        target_subpriority = {}
        for _tid, _pri, _subpri in zip(
            gal_cat["TARGETID"][_pmask],
            gal_cat["PRIORITY"][_pmask],
            gal_cat["SUBPRIORITY"][_pmask],
        ):
            _tid = int(_tid)
            if _tid not in target_priority:
                target_priority[_tid] = float(_pri)
                target_subpriority[_tid] = float(_subpri)

        n_before = len(target_to_fiber)

        for (tile_id, fiber_i, fiber_j), collision_pairs in collision_constraints.items():
            fiber_i_node = f"{tile_id}_fiber_{fiber_i}"
            fiber_j_node = f"{tile_id}_fiber_{fiber_j}"

            for target_id_i, target_id_j in collision_pairs:
                ti, tj = int(target_id_i), int(target_id_j)

                if ti in removed_targets or tj in removed_targets:
                    continue

                active_i = target_to_fiber.get(ti) == fiber_i_node
                active_j = target_to_fiber.get(tj) == fiber_j_node
                if not (active_i and active_j):
                    continue

                pri_i = target_priority.get(ti, 0.0)
                pri_j = target_priority.get(tj, 0.0)

                if pri_i < pri_j:
                    remove_tid = ti
                elif pri_j < pri_i:
                    remove_tid = tj
                else:
                    subpri_i = target_subpriority.get(ti, 0.0)
                    subpri_j = target_subpriority.get(tj, 0.0)
                    if subpri_i < subpri_j:
                        remove_tid = ti
                    elif subpri_j < subpri_i:
                        remove_tid = tj
                    else:
                        remove_tid = max(ti, tj)

                removed_targets.add(remove_tid)
                target_to_fiber.pop(remove_tid, None)

        assigned_targets_set = set(target_to_fiber.keys())
        assigned_targets_id = np.asarray(list(assigned_targets_set))
        violations_after = count_violations(target_to_fiber)

        if verbose:
            print(
                f"Pairwise collision repair: removed {len(removed_targets)} lower-priority "
                f"targets; kept {len(assigned_targets_set)} / {n_before}. "
                f"Violations: {violations_before} -> {violations_after}."
            )

    return {
        "assigned_targets_id": assigned_targets_id,
        "assigned_targets_set": assigned_targets_set,
        "target_to_fiber": target_to_fiber,
        "violations_before": violations_before,
        "violations_after": violations_after,
        "removed_targets": removed_targets,
    }
