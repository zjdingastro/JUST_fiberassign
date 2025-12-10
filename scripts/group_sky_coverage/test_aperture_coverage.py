#!/usr/bin/env python
# coding: utf-8

"""
计算星系团孔径覆盖天区比例
分析覆盖比例随质量、红移和孔径半径的变化
"""
import os, sys
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')
#from skycover_utils import angular_radius_from_physical, generate_uniform_points_on_sphere, compute_coverage_fraction_monte_carlo
sys.path.append("/home/zjding/fiberassignment/algorithm")
from utils import get_spherearea



# ============================================================================ #
# 宇宙学参数和常数
# ============================================================================ #

# 使用Planck 2018参数 (h=0.7)
H0 = 70.0  # km/s/Mpc
Om0 = 0.3
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

RA_MIN, RA_MAX = 150., 200.
DEC_MIN, DEC_MAX = 0., 40.

# 默认参数
SURVEY_AREA_DEG2 = get_spherearea(RA_MIN, RA_MAX, DEC_MIN, DEC_MAX)
print(SURVEY_AREA_DEG2)
LOG_MASS_THRESHOLD = 14.0  # 10^14 Msol/h
Z_MIN = 0.1
Z_MAX = 0.34
R_AP_DEFAULT = 20.0  # Mpc/h 


# load group catalog
idir_group = "/home/xhyang/work/Gfinder/DESIDR9/data/"
filename = "DESIDR9_NGC_group"
group_cat = np.loadtxt(idir_group + filename)

columns = ['group_id', 'richness', 'RA', 'DEC', 'Z', 'logMh', 'L_group']
group_cat = Table(group_cat, names = columns)



def load_cluster_catalog(group_cat,
                         mass_range: Tuple[float, float] = (LOG_MASS_THRESHOLD, 16.0),
                         z_range: Tuple[float, float] = (Z_MIN, Z_MAX),
                         ra_range: Tuple[float, float] = (RA_MIN, RA_MAX),
                         dec_range: Tuple[float, float] = (DEC_MIN, DEC_MAX)) -> np.ndarray:
    """
    load DR9 星系团目录
    
    参数:
        group_cat: input parent group catalog
        mass_range: log10(质量范围) (Msol/h)
        z_range: 红移范围
        ra_range: 赤经范围 (度)
        dec_range: 赤纬范围 (度)
        seed: 随机种子
    
    返回:
        catalog: 数组，列：[ra, dec, z, Mh]
    """
    
    group_mass = group_cat['logMh']
    mass_mask = (group_mass > mass_range[0])&(group_mass < mass_range[1])
    zmask = (group_cat['Z'] > z_range[0])&(group_cat['Z'] < z_range[1])
    sky_mask = (group_cat['RA']>ra_range[0])&(group_cat['RA']<ra_range[1])&(group_cat['DEC']>dec_range[0])&(group_cat['DEC']<dec_range[1])
    mask = mass_mask&zmask&sky_mask

    ras = group_cat['RA'][mask]
    decs = group_cat['DEC'][mask]
    redshifts = group_cat['Z'][mask]
    masses = group_mass[mask]
    
    catalog = np.column_stack([ras, decs, redshifts, masses])
    
    return catalog

# ============================================================================ #
# 数据结构和工具函数
# ============================================================================ #

def angular_radius_from_physical(r_physical_mpch: float, z: float, 
                                  h: float = 0.7) -> float:
    """
    从物理半径（Mpc/h）和红移计算角半径（度）
    
    参数:
        r_physical_mpch: 物理半径 (Mpc/h)
        z: 红移
        h: Hubble参数 (默认0.7)
    
    返回:
        角半径 (度)
    """
    # 转换为物理单位 (Mpc)
    r_physical_mpc = r_physical_mpch / h
    
    # 计算角直径距离
    d_A = cosmo.angular_diameter_distance(z).to(units.Mpc).value
    
    # 角半径 (弧度)
    theta_rad = r_physical_mpc / d_A
    
    # 转换为度
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg



def generate_uniform_points_on_sphere(n_points: int,
                                      ra_range: Tuple[float, float] = (0, 360),
                                      dec_range: Tuple[float, float] = (-90, 90),
                                      seed: Optional[int] = None) -> np.ndarray:
    """
    在天球上生成均匀分布的随机点 (RA, DEC)
    
    注意：在天球坐标系中，要保证面积均匀分布，DEC需要按照sin(dec)分布采样。
    对于给定的DEC范围，使用正确的采样方法。
    
    参数:
        n_points: 要生成的点数
        ra_range: RA范围 (度)，默认 (0, 360)
        dec_range: DEC范围 (度)，默认 (-90, 90)
        seed: 随机种子（可选）
    
    返回:
        points: 数组，形状为 (n_points, 2)，列：[RA, DEC] (度)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # RA均匀分布
    ra_min, ra_max = ra_range
    ras = np.random.uniform(ra_min, ra_max, n_points)
    
    # DEC需要按照面积元素均匀分布
    # 面积元素 dA = cos(dec) d(ra) d(dec)
    # 要均匀分布，需要从 sin(dec) 均匀分布采样
    dec_min, dec_max = dec_range
    
    # 转换为弧度
    dec_min_rad = np.radians(dec_min)
    dec_max_rad = np.radians(dec_max)
    
    # 从 sin(dec) 均匀分布采样
    sin_dec_min = np.sin(dec_min_rad)
    sin_dec_max = np.sin(dec_max_rad)
    
    # 均匀采样 sin(dec)
    u = np.random.uniform(sin_dec_min, sin_dec_max, n_points)
    
    # 转换为 dec
    decs_rad = np.arcsin(u)
    decs = np.degrees(decs_rad)
    
    return ras, decs

# ============================================================================ #
# 覆盖面积计算
# ============================================================================ #

def compute_coverage_fraction_monte_carlo(catalog,
                                          r_ap_mpch: float,
                                          n_samples: int=1000000,
                                          ra_range: Optional[Tuple[float, float]] = (RA_MIN, RA_MAX),
                                          dec_range: Optional[Tuple[float, float]] = (DEC_MIN, DEC_MAX)) -> float:
    """
    使用蒙特卡洛方法计算覆盖比例
    
    参数:
        catalog: 星系团目录 [ra, dec, z, Mh]
        r_ap_mpch: 孔径半径 (Mpc/h)
        n_samples: 蒙特卡洛采样点数
        ra_range: RA范围（如果None，从catalog估算）
        dec_range: Dec范围（如果None，从catalog估算）
    
    返回:
        覆盖比例 (0-1)
    """
    if len(catalog) == 0:
        return 0.0
    
    # 确定采样区域
    if ra_range is None:
        ra_range = (catalog[:, 0].min(), catalog[:, 0].max())
    if dec_range is None:
        dec_range = (catalog[:, 1].min(), catalog[:, 1].max())
    
    # 生成随机采样点
    ra_samples, dec_samples = generate_uniform_points_on_sphere(n_samples, ra_range, dec_range)
    
    # 计算每个采样点是否被覆盖
    covered = np.zeros(n_samples, dtype=bool)
    
    for i, (ra_cl, dec_cl, z_cl, _) in enumerate(catalog):
        # 计算该星系团的角半径
        theta_deg = angular_radius_from_physical(r_ap_mpch, z_cl)
        
        # 计算采样点到星系团中心的角距离
        # 简化：小角度近似
        dra = ra_samples - ra_cl
        ddec = dec_samples - dec_cl
        # 考虑RA的cos(dec)修正
        dec_cl_rad = np.deg2rad(dec_cl)
        dra_corr = dra * np.cos(dec_cl_rad)
        distances = np.sqrt(dra_corr**2 + ddec**2)
        
        # 检查是否在孔径内
        covered |= (distances <= theta_deg)
    
    coverage_fraction = np.sum(covered) / n_samples
    
    return coverage_fraction




"""
主程序：生成目录、计算覆盖比例、进行参数扫描
"""
print("=" * 60)
print("星系团孔径覆盖分析")
print("=" * 60)

# 生成星系团目录
print("\n1. load galaxy cluster...")

catalog = load_cluster_catalog(group_cat,
    mass_range=(LOG_MASS_THRESHOLD, 16.0),
    z_range=(Z_MIN, Z_MAX),
    ra_range = (RA_MIN, RA_MAX),
    dec_range = (DEC_MIN, DEC_MAX)
)
print(f"   contains {len(catalog)} 个星系团")
print(f"   log10(质量)范围: {catalog[:, 3].min():.2e} - {catalog[:, 3].max():.2e} Msol/h")
print(f"   红移范围: {catalog[:, 2].min():.3f} - {catalog[:, 2].max():.3f}")



# 计算默认参数下的覆盖比例
print("\n2. 计算覆盖比例...")
n_samples=1000000
coverage_default = compute_coverage_fraction_monte_carlo(
    catalog, R_AP_DEFAULT, n_samples, ra_range = (RA_MIN, RA_MAX), dec_range = (DEC_MIN, DEC_MAX))
print(f"   默认参数 (r_ap={R_AP_DEFAULT} Mpc/h): "
      f"覆盖比例 = {coverage_default:.4f} ({coverage_default*100:.2f}%)")


cluster_radii = angular_radius_from_physical(R_AP_DEFAULT, catalog[:, 2])



fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

# 设置颜色映射
norm = Normalize(vmin=catalog[:, 2].min(), vmax=catalog[:, 2].max())
cmap = plt.cm.viridis

# 为每个星系团绘制圆形孔径
for i in range(len(catalog)):
    ra, dec, z, _ = catalog[i]
    radius = cluster_radii[i]
    
    # 创建圆形，使用红移值确定颜色
    color = cmap(norm(z))
    circle = Circle((ra, dec), radius, 
                   facecolor='none', 
                   edgecolor=color, 
                   linewidth=0.5,
                   alpha=0.6)
    ax.add_patch(circle)

# 设置坐标轴范围
ax.set_xlim(catalog[:, 0].min() - cluster_radii.max(), 
            catalog[:, 0].max() + cluster_radii.max())
ax.set_ylim(catalog[:, 1].min() - cluster_radii.max(), 
            catalog[:, 1].max() + cluster_radii.max())

ax.plot([RA_MIN, RA_MAX, RA_MAX, RA_MIN, RA_MIN], [DEC_MIN, DEC_MIN, DEC_MAX, DEC_MAX, DEC_MIN], 'k--')
ax.set_xlabel('RA [deg]', fontsize=14)
ax.set_ylabel('DEC [deg]', fontsize=14)
ax.set_title(r'$M_\mathrm{cluster}>10^{14}M_{\odot}/h, R=%.0f\mathrm{Mpc/h}, f_{\mathrm{cover}}=%.1f$'%(R_AP_DEFAULT, coverage_default*100)+"%", fontsize=14)
ax.set_aspect('equal')

# 添加colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('redshift')
plt.savefig("./figs/group_skycover.png")




