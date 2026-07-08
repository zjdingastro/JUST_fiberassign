#!/usr/bin/env python
# coding: utf-8
# python cal_aperture_coverage.py --LOG_MASS_THRESHOLD 14.0 --R_AP_DEFAULT 8.0

"""
Compute the sky fraction covered by galaxy-cluster apertures.
Analyze how the covered fraction depends on mass, redshift, and aperture radius.
"""
import os, sys
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units
from typing import Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
sys.path.append("/home/zjding/installed_packages/JUST_fiberassign/py/")
from utils import get_spherearea
import argparse

# Default cosmology (Planck 2018-like, h = 0.7)
H0 = 70.0  # km/s/Mpc
Om0 = 0.3
H = H0 / 100.0
DEFAULT_COSMO = FlatLambdaCDM(H0=H0, Om0=Om0)


def load_cluster_catalog(group_cat, mass_range, z_range, ra_range, dec_range):

    """
    Load a galaxy cluster catalog.

    Parameters:
        group_cat: input parent group catalog
        mass_range: log10(mass range) in Msun/h
        z_range: redshift range
        ra_range: right ascension range in degrees
        dec_range: declination range in degrees
        seed: random seed

    Returns:
        catalog: array with columns [ra, dec, z, Mh]
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
# Data structures and utility functions
# ============================================================================ #

def _radec_to_unit_xyz(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def angular_radius_from_physical(r_comoving_mpch, z, h=0.7, cosmo=None):
    """
    Convert a physical radius (Mpc/h) and redshift to an angular radius in degrees.

    Parameters:
        r_comoving_mpch: comoving radius in Mpc/h
        z: redshift (scalar or array)
        h: Hubble parameter (default 0.7)
        cosmo: astropy cosmology object

    Returns:
        angular radius in degrees
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    r_comoving_mpc = r_comoving_mpch / h    
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    # comoving transverse distance, the same as the comoving distance if :math:`\Omega_k` is zero
    d_A = cosmo.comoving_transverse_distance(z_arr).to(units.Mpc).value
    #d_A = cosmo.comoving_distance(z).to(units.Mpc).value
    theta_deg = np.degrees(r_comoving_mpc / d_A)
    if np.ndim(z) == 0:
        return float(theta_deg[0])
    return theta_deg



def generate_uniform_points_on_sphere(n_points: int,
                                      ra_range: Tuple[float, float] = (0, 360),
                                      dec_range: Tuple[float, float] = (-90, 90),
                                      seed: Optional[int] = None) -> np.ndarray:
    """
    Generate uniformly distributed random points on the sphere (RA, DEC).

    Note: to preserve uniform area on the sphere, DEC must be sampled in sin(dec).
    This function uses the correct sampling method for a given DEC range.

    Parameters:
        n_points: number of points to generate
        ra_range: RA range in degrees, default (0, 360)
        dec_range: DEC range in degrees, default (-90, 90)
        seed: optional random seed

    Returns:
        points: tuple of arrays (RA, DEC) in degrees
    """
    if seed is not None:
        np.random.seed(seed)

    # Uniform in RA
    ra_min, ra_max = ra_range
    ras = np.random.uniform(ra_min, ra_max, n_points)

    # Uniform area element requires sampling in sin(dec)
    # Area element dA = cos(dec) d(ra) d(dec)
    # Uniform surface density => sample sin(dec) uniformly
    dec_min, dec_max = dec_range

    # Convert to radians
    dec_min_rad = np.radians(dec_min)
    dec_max_rad = np.radians(dec_max)

    # Sample uniformly in sin(dec)
    sin_dec_min = np.sin(dec_min_rad)
    sin_dec_max = np.sin(dec_max_rad)

    # Draw uniform sin(dec) values
    u = np.random.uniform(sin_dec_min, sin_dec_max, n_points)

    # Convert back to dec
    decs_rad = np.arcsin(u)
    decs = np.degrees(decs_rad)

    return ras, decs

# ============================================================================ #
# Coverage fraction calculation
# ============================================================================ #

def compute_coverage_fraction_monte_carlo(
    catalog,
    r_ap_mpch,
    n_samples,
    ra_range,
    dec_range,
    h=0.7,
    cosmo=None,
    cluster_radii=None,
    batch_size=32,
    seed=None,
    use_kdtree=True,
):
    """
    Estimate the covered sky fraction with Monte Carlo sampling.

    Parameters:
        catalog: cluster catalog with columns [ra, dec, z, Mh]
        r_ap_mpch: aperture radius in Mpc/h
        n_samples: number of Monte Carlo sample points
        ra_range: RA range; if None, inferred from catalog
        dec_range: DEC range; if None, inferred from catalog
        cluster_radii: optional precomputed angular radii in degrees
        batch_size: number of clusters processed per vectorized batch
        seed: optional random seed
        use_kdtree: use a 3D KD-tree on sample points (much faster for large catalogs)

    Returns:
        covered fraction between 0 and 1
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    if len(catalog) == 0:
        return 0.0

    if ra_range is None:
        ra_range = (catalog[:, 0].min(), catalog[:, 0].max())
    if dec_range is None:
        dec_range = (catalog[:, 1].min(), catalog[:, 1].max())

    ra_samples, dec_samples = generate_uniform_points_on_sphere(
        n_samples, ra_range, dec_range, seed=seed
    )

    if cluster_radii is None:
        cluster_radii = angular_radius_from_physical(
            r_ap_mpch, catalog[:, 2], h=h, cosmo=cosmo
        )
    cluster_radii = np.asarray(cluster_radii, dtype=np.float64)

    if use_kdtree:
        xyz_samples = _radec_to_unit_xyz(ra_samples, dec_samples)
        tree = cKDTree(xyz_samples)
        xyz_cl = _radec_to_unit_xyz(catalog[:, 0], catalog[:, 1])
        chord_limits = 2.0 * np.sin(np.deg2rad(cluster_radii) / 2.0)

        covered = np.zeros(n_samples, dtype=bool)
        for xyz, chord_limit in zip(xyz_cl, chord_limits):
            idx = tree.query_ball_point(xyz, chord_limit)
            if idx:
                covered[np.asarray(idx, dtype=np.int64)] = True
        return float(np.mean(covered))

    ra_cl = catalog[:, 0]
    dec_cl = catalog[:, 1]
    cos_dec_cl = np.cos(np.deg2rad(dec_cl))
    theta2 = cluster_radii * cluster_radii

    covered = np.zeros(n_samples, dtype=bool)
    n_clusters = len(catalog)

    for start in range(0, n_clusters, batch_size):
        end = min(n_clusters, start + batch_size)
        dra = ra_samples[None, :] - ra_cl[start:end, None]
        ddec = dec_samples[None, :] - dec_cl[start:end, None]
        dra_corr = dra * cos_dec_cl[start:end, None]
        dist2 = dra_corr * dra_corr + ddec * ddec
        covered |= (dist2 <= theta2[start:end, None]).any(axis=0)

    return float(np.mean(covered))


def plot_ra_dec_radii(ax, ra_cl, dec_cl, theta_deg, color='k', nphi=181):
    """Plot the boundary of a spherical cap at angular radius theta_deg from (ra_cl, dec_cl)."""
    theta = np.deg2rad(float(theta_deg))
    ra_rad, dec_rad = np.radians(float(ra_cl)), np.radians(float(dec_cl))
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    sin_dec = (
        np.sin(dec_rad) * np.cos(theta)
        + np.cos(dec_rad) * np.sin(theta) * np.cos(phi)
    )
    dra = np.arctan2(
        np.sin(theta) * np.sin(phi),
        np.cos(dec_rad) * np.cos(theta) - np.sin(dec_rad) * np.sin(theta) * np.cos(phi),
    )
    ra_deg = np.degrees(ra_rad + dra)
    dec_deg = np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    ax.plot(ra_deg, dec_deg, lw=0.5, color=color, alpha=0.6, zorder=10)


def main():
    parser = argparse.ArgumentParser(description='Compute aperture coverage for galaxy clusters')
    parser.add_argument("--LOG_MASS_THRESHOLD", type=float, default=14.0)
    parser.add_argument("--R_AP_DEFAULT", type=float, default=8.0)
    args = parser.parse_args()
    # ============================================================================ #
    # Cosmology and constants
    # ============================================================================ #

    # Planck 2018-like parameters with h = 0.7
    cosmo = DEFAULT_COSMO
    h = H

    RA_MIN, RA_MAX = 0., 90.
    DEC_MIN, DEC_MAX = 0., 90.

    # Default parameters
    SURVEY_AREA_DEG2 = get_spherearea(RA_MIN, RA_MAX, DEC_MIN, DEC_MAX)
    print("SURVEY_AREA_DEG2 = %.1f deg^2"%SURVEY_AREA_DEG2)
    LOG_MASS_THRESHOLD = args.LOG_MASS_THRESHOLD  # default 14.0 (10^14 Msol/h)
    R_AP_DEFAULT = args.R_AP_DEFAULT  # default 8.0 (Mpc/h)


    # Load group catalog
    ifile = "/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/data/v1/lightcone_ra_0_90_dec_0_90_rmagcut20.5.fits"
    gal_cat = Table.read(ifile)
    mask = (np.log10(gal_cat["Mh"])> LOG_MASS_THRESHOLD)&(gal_cat["type"]==1) # approximate the galaxy cluster sample
    group_cat = gal_cat[mask]
    print(f"   contains {len(group_cat)} clusters")
    Z_MIN = 0.0
    Z_MAX = 1.25

    group_cat["logMh"] = np.log10(group_cat["Mh"])
    group_cat.rename_column("ra", "RA")
    group_cat.rename_column("dec", "DEC")
    group_cat.rename_column("z", "Z")

    """
    Main workflow: build catalog, compute coverage fraction, and plot apertures.
    """
    print("=" * 60)
    print("Galaxy cluster aperture coverage analysis")
    print("=" * 60)

    # Build cluster catalog
    print("\n1. load galaxy cluster...")

    catalog = load_cluster_catalog(group_cat,
        mass_range=(LOG_MASS_THRESHOLD, 16.0),
        z_range=(Z_MIN, Z_MAX),
        ra_range = (RA_MIN, RA_MAX),
        dec_range = (DEC_MIN, DEC_MAX)
    )
    print(f"   contains {len(catalog)} clusters after redshift and sky footprint cuts")
    print(f"   log10(mass) range: {catalog[:, 3].min():.2e} - {catalog[:, 3].max():.2e} Msol/h")
    print(f"   redshift range: {catalog[:, 2].min():.3f} - {catalog[:, 2].max():.3f}")

    # Coverage fraction for default parameters
    print("\n2. compute coverage fraction...")
    n_samples = 1_000_000
    cluster_radii = angular_radius_from_physical(
        R_AP_DEFAULT, catalog[:, 2], h=h, cosmo=cosmo
    )
    coverage_default = compute_coverage_fraction_monte_carlo(
        catalog,
        R_AP_DEFAULT,
        n_samples,
        ra_range=(RA_MIN, RA_MAX),
        dec_range=(DEC_MIN, DEC_MAX),
        h=h,
        cosmo=cosmo,
        cluster_radii=cluster_radii,
    )
    print(f"   default parameters (r_ap={R_AP_DEFAULT} Mpc/h): "
        f"coverage fraction = {coverage_default:.4f} ({coverage_default*100:.2f}%)")


    # Plot the aperture coverage (subsample if the catalog is very large)
    max_plot_clusters = 100000
    if len(catalog) > max_plot_clusters:
        plot_idx = np.linspace(0, len(catalog) - 1, max_plot_clusters, dtype=int)
        print(f"   plotting {max_plot_clusters} of {len(catalog)} cluster apertures")
    else:
        plot_idx = np.arange(len(catalog))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # Color map by redshift
    norm = Normalize(vmin=catalog[:, 2].min(), vmax=catalog[:, 2].max())
    cmap = plt.cm.viridis

    for i in plot_idx:
        ra, dec, z, _ = catalog[i]
        color = cmap(norm(z))
        plot_ra_dec_radii(ax, ra, dec, cluster_radii[i], color=color)

    # Axis limits
    ax.set_xlim(catalog[:, 0].min() - cluster_radii.max(),
                catalog[:, 0].max() + cluster_radii.max())
    ax.set_ylim(catalog[:, 1].min() - cluster_radii.max(),
                catalog[:, 1].max() + cluster_radii.max())

    ax.plot([RA_MIN, RA_MAX, RA_MAX, RA_MIN, RA_MIN], [DEC_MIN, DEC_MIN, DEC_MAX, DEC_MAX, DEC_MIN], 'k--')
    ax.set_xlabel('RA [deg]', fontsize=14)
    ax.set_ylabel('DEC [deg]', fontsize=14)
    ax.set_title(r'$M_\mathrm{cluster}>10^{%s}M_{\odot}/h, R=%.1f\mathrm{Mpc/h}, f_{\mathrm{cover}}=%.1f$'%(str(LOG_MASS_THRESHOLD), R_AP_DEFAULT, coverage_default*100)+"%", fontsize=14)
    ax.set_aspect('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('redshift')
    os.makedirs("./figs", exist_ok=True)
    plt.savefig("./figs/group_skycover_logMh%s_R%.1f.png"%(str(LOG_MASS_THRESHOLD), R_AP_DEFAULT), dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()



