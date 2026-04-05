import numpy as np
from scipy.spatial import KDTree



# credit: Yunzhe
#ZD change minimum_separation from 1.6 to 2.0 based on the updated fiber positioner (08-04-2025)
def assign_targets_greedy(fibers_center, targets_pos, targets_ID, priorities, subpriorities, radius = 6, minimum_separation = 2.0):
    '''
    Parameters
    ----------
    fibers_center : np.array([N, 2]), (x, y) location of each fiber positoner
    targets_pos : np.array([N, 2]), projected location of targets on focalplane
    priorities : 1d array, priority of targets
    subpriorities : 1d array, subpriorities of targets
    radius : patrol radius
    minimum_separation : targets separated less than this distance will not be assigned by fibers in the same tile

    returns:
    --------
    #assigned_targets : coordiantes (x, y) of targets assigned by fibers; None means unassigned
    assigned_target_ids : indices of targets with fiber assigned
    '''
    target_tree = KDTree(targets_pos)
    #assigned_targets = []
    assigned_indices = set()
    assigned_positions = [] # Track positions of assigned targets for distance checks
    assigned_ID = []
    #for fiber in tqdm(fibers_center):
    for fiber in fibers_center:
        indices = target_tree.query_ball_point(fiber, r = radius)
        available = [i for i in indices if i not in assigned_indices]
        valid = []
        for idx in available:
            too_close = False
            for assigned_pos in assigned_positions:
                dist = ((targets_pos[idx][0] - assigned_pos[0]) **2 + 
                        (targets_pos[idx][1] - assigned_pos[1]) **2) **0.5
                if dist < minimum_separation:
                    too_close = True
                    break
            if not too_close:
                valid.append(idx)
        if valid:
            best_idx = max(valid, key = lambda i: (priorities[i], subpriorities[i]))
            assigned_positions.append(targets_pos[best_idx])
            assigned_indices.add(best_idx)
            assigned_ID.append(targets_ID[best_idx])  # added by ZD, 06-28-2025
        # else:
        #     assigned_targets.append(None)

    return assigned_ID

