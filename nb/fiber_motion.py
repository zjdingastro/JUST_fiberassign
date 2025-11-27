import numpy as np
#import random
from scipy.spatial import KDTree

def generate_hex_grid(radius, rows, cols):
    """
    生成一个二维正六边形网格。
    :param radius: 每个六边形的边长
    :param rows: 网格的行数
    :param cols: 网格的列数
    :return: 所有点的坐标列表
    """
    dx = radius * np.sqrt(3)
    dy = radius * 1.5
    points = []

    for row in range(rows):
        for col in range(cols):
            x = col * dx
            y = row * dy
            # if col % 2 == 1:
            #     y += radius * 0.75
            if row % 2 == 1:
                x += radius * 3**0.5/2
            points.append((x, y))
    return points

def get_neighbors(center, radius):
    """返回一个点的六个邻居坐标"""
    angles_deg = [30, 90, 150, 210, 270, 330]
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    neighbors = [(center[0] + radius * np.cos(a),
                  center[1] + radius * np.sin(a)) for a in angles_rad]
    return neighbors

def generate_random_point_in_circle(center, radius=6.0):
    # 生成随机角度（0~2π）
    theta = np.random.uniform(0, 2 * np.pi)
    # 调整半径分布（避免边缘稀疏）
    r = np.sqrt(np.random.uniform(0, 1)) * radius
    # 转换为笛卡尔坐标
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.array([x, y]) 

def get_ids_targets_in_collison_dis(targets_pos, dis_collision):
    target_tree = KDTree(targets_pos)
    collide_ids = target_tree.query_pairs(r=dis_collision)
    return collide_ids

# def fiber_pos_thetaphi2xy(patrol_pos, theta, phi, arm1=3.0, arm2=3.0):
#     # patrol_pos: (x, f) of patrol center
#     # theta: angle of the first arm
#     # phi: angle of the second arm extended with respect to patrol center
    
#     ang_matrix = np.array([[np.cos(theta), np.cos(theta+phi)],[np.sin(theta), np.sin(theta+phi)]])
#     arms = np.array([arm1, arm2])
    
#     fiber_pos = patrol_pos + np.dot(ang_matrix, arms)
    
#     return fiber_pos

## to do: adding Numba for faster calculation
def fiber_pos_thetaphi2xy(patrol_pos, theta, phi, arm1=3.0, arm2=3.0):
    """
    批量计算光纤位置（支持theta/phi为数组）
    
    参数：
    patrol_pos: (N, 2)数组，每个元素是(x, f)形式的巡逻中心坐标
    theta: (N,)数组，每个元素是第一个臂的角度（弧度）
    phi: (N,)数组，每个元素是第二个臂相对于第一个臂的偏移角（弧度）
    arm1/arm2: 臂长（标量或与theta同形状的数组）
    
    返回：
    (N, 2)数组，每个元素是光纤的最终坐标
    """
    patrol_pos = np.asarray(patrol_pos, dtype=np.float64)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # 支持 arm1/arm2 为标量或数组
    # arm1 = np.asarray(arm1)
    # arm2 = np.asarray(arm2)
    # if arm1.ndim == 0:
    #     arm1 = np.full_like(theta, arm1)
    # if arm2.ndim == 0:
    #     arm2 = np.full_like(theta, arm2)

    # 第一个臂的末端位置
    x1 = arm1 * np.cos(theta)
    y1 = arm1 * np.sin(theta)

    # 第二个臂的末端位置（相对于第一个臂）
    x2 = arm2 * np.cos(theta + np.pi - phi)
    y2 = arm2 * np.sin(theta + np.pi - phi)

    # 总偏移量
    dx = x1 + x2
    dy = y1 + y2

    # 最终位置
    return patrol_pos + np.stack([dx, dy], axis=-1)


def get_arm1_endpoint(patrol_pos, theta, arm1=3.0):
    dx = arm1 * np.cos(theta)
    dy = arm1 * np.sin(theta)
    return patrol_pos + np.stack([dx, dy], axis=-1)


def decen_navigation_fun(pos_fibers, pos_targets, id_fiber, id_neighbor_fibers, d_l, d_s, lambda_1, lambda_2):
    '''
    pos_fibers: (x, y) of N fibers, shape=(N, 2)
    pos_targets: (x, y) of N targets, shape=(N, 2)
    id_fiber: index of the fiber consider
    id_neighbor_fibers: indices of the neighboring fibers
    d_l: large D
    d_s: small d
    lambda_1, lambda_2: constants to be tuned
    '''
    #print(pos_targets[id_fiber])
    phi_1 = lambda_1 * np.sum((pos_fibers[id_fiber] - pos_targets[id_fiber])**2)
    phi_2 = 0
    
    for j in id_neighbor_fibers:
        dis2_fibers = np.sum((pos_fibers[id_fiber] - pos_fibers[j])**2)
        temp = (dis2_fibers - d_l**2.0)/(dis2_fibers - d_s**2.0)
        phi_2 += np.min([0.0, temp])
        
    phi_2 *= lambda_2
    
    return phi_1 + phi_2


def cal_fiber_angular_vel(psi1, psi2, fiber_pos1, fiber_pos2, theta, phi, la, lb):
    dxy = fiber_pos2 - fiber_pos1
    dx = dxy[:, 0]
    dy = dxy[:, 1]
    mask_dx = (np.abs(dx) > 1e-10)   # avoid 0 devision
    mask_dy = (np.abs(dy) > 1e-10)
    dpsi_dx = np.zeros_like(dx)
    dpsi_dy = np.zeros_like(dy)
    dx_dtheta = np.zeros_like(dx)
    dy_dtheta = np.zeros_like(dy)
    dx_dphi = np.zeros_like(dx)
    dy_dphi = np.zeros_like(dy)
    
    
    theta_mask_dx = theta[mask_dx]
    phi_mask_dx   = phi[mask_dx]
    
    theta_mask_dy = theta[mask_dy]
    phi_mask_dy   = phi[mask_dy]
    
    dpsi_dx[mask_dx] = (psi2[mask_dx] - psi1[mask_dx])/dx[mask_dx]
    dpsi_dy[mask_dy] = (psi2[mask_dy] - psi1[mask_dy])/dy[mask_dy]
    
    dx_dtheta[mask_dx] = -la*np.sin(theta_mask_dx) + lb*np.sin(theta_mask_dx - phi_mask_dx)
    dy_dtheta[mask_dy] =  la*np.cos(theta_mask_dy) - lb*np.cos(theta_mask_dy - phi_mask_dy)
    
    dx_dphi[mask_dx] = -lb*np.sin(theta_mask_dx - phi_mask_dx)
    dy_dphi[mask_dy] =  lb*np.cos(theta_mask_dy - phi_mask_dy)
    
    wtheta = -1.0*(dpsi_dx * dx_dtheta + dpsi_dy * dy_dtheta)
    wphi = -1.0*(dpsi_dx * dx_dphi + dpsi_dy * dy_dphi)
    
    return np.stack([wtheta, wphi], axis=-1)

def cal_decen_navigation_fun(patrol_centers, patrol_tree, pos_fibers, targets_pos, d_l, d_s, lambda_1, lambda_2, arm1, arm2, N_fibers):
    dnf_list = []
    for fiber_id in range(N_fibers): 
        neighbor_ids = np.array(patrol_tree.query_ball_point(patrol_centers[fiber_id], 2*(arm1+arm2)))   # return points within radius=12 mm
        #print(neighbor_ids)
        mask = (neighbor_ids != fiber_id)
        id_neighbor_fibers = neighbor_ids[mask]
        dnf = decen_navigation_fun(pos_fibers, targets_pos, fiber_id, id_neighbor_fibers, d_l, d_s, lambda_1, lambda_2)
        dnf_list.append(dnf)
    return np.array(dnf_list)
