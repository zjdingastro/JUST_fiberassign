# JUST_fiberassign

**JUST_fiberassign** is a Python package for spectroscopic fiber assignment on the Jiao Tong University Spectroscopic Telescope (JUST). It provides focal-plane geometry utilities, greedy assignment, and a min-cost-flow (MCF) solver with fiber-collision constraints. The design takes inspiration from the [DESI fiberassign](https://github.com/desihub/fiberassign) package and SDSS method (Blanton et al. 2003). 

## Features

- **Focal-plane geometry**: fiber positions, plate scale, RA/Dec ↔ focal-plane (x, y) transforms, and tile coverage masks; instrument data in `parameters/`.
- **Greedy assignment**: fast per-tile assignment with priority, subpriority, patrol radius, and minimum target separation.
- **Network-flow assignment**: min-cost-flow optimization with in-graph collision gadgets, iterative forbidden-assignment repair for overlapping tile groups, and pairwise post-solve collision cleanup.
- **Single-tile MCF pipeline**: memory-friendly per-tile processing over multi-pass surveys, with HEALPix-split Merged Target Lists (MTL), resume from existing output, and per-tile timeout fallback (fast MCF without collision gadgets, then pairwise repair).
- **Survey analysis tools**: completeness studies and galaxy-cluster aperture sky-coverage estimates.

## Package layout

```
JUST_fiberassign/
├── JUST_fiberassign/           # Core library (installable package)
│   ├── __init__.py
│   ├── greedy_assign.py        # Greedy fiber assignment
│   ├── network_flow.py         # Min-cost-flow solver and collision handling
│   ├── fba_single_tile.py      # Single-tile MCF helpers (`fba_onetile`, decollided variant)
│   ├── utils.py                # Geometry, tile masks, I/O helpers
│   └── parameters.py           # Tile radii, patrol radius, collision separation
├── scripts/                    # Command-line workflows
│   ├── assign_targets.py       # Greedy assignment + completeness
│   ├── mcf_singletiles.py      # Single-tile MCF over multiple passes (timeout fallback)
│   ├── mcf_singletiles_decollided.py  # MCF with decolliding pre-filter (uses `fba_single_tile`)
│   ├── group_sky_coverage/
│   │   └── cal_aperture_coverage.py   # Cluster aperture coverage analysis
│   └── run_assign.pbs          # Example PBS job script
├── parameters/                 # Instrument data (fiberpos.csv, platescale.txt, rzsn.txt)
├── input/                      # Example tile-layout inputs
│   └── set_tiles/
├── nb/                         # Jupyter notebooks (development / analysis)
├── requirements.txt
├── MANIFEST.in
└── setup.py
```

## Requirements

- **Python**: >= 3.7
- **Core dependencies** (installed by `pip install -e .` or `pip install -r requirements.txt`):
  - `numpy` >= 1.19.0
  - `scipy` >= 1.7.0
  - `astropy` >= 4.0
  - `pandas` >= 1.3.0 — reads `parameters/fiberpos.csv`
  - `networkx` >= 2.6 — min-cost-flow solver
  - `healpy` >= 1.14.0 — HEALPix pixelization for MTL workflows (`fba_single_tile.py`, `mcf_singletiles*.py`)
- **Optional**:
  - `matplotlib` >= 3.3.0 — visualization and completeness scripts (`pip install -e ".[visualization]"`)

`requirements.txt` lists core dependencies plus optional `matplotlib`. Editable install is recommended:

```bash
pip install -e .
```

## Installation

### From a local clone

```bash
git clone git@github.com:zjdingastro/JUST_fiberassign.git
cd JUST_fiberassign

# Recommended: editable install (core dependencies + healpy)
pip install -e .

# Include matplotlib for plotting scripts
pip install -e ".[visualization]"
```

### Using scripts without installing

Scripts under `scripts/` import modules from the inner library directory (`JUST_fiberassign/JUST_fiberassign/`). Either install the package as above, or set `PYTHONPATH` before running:

```bash
export PYTHONPATH="/path/to/JUST_fiberassign/JUST_fiberassign:${PYTHONPATH}"
python scripts/mcf_singletiles.py --help
python scripts/mcf_singletiles_decollided.py --help
```

## Core library (`JUST_fiberassign/`)

| Module | Description |
|--------|-------------|
| `greedy_assign.py` | `assign_targets_greedy()` — greedy assignment on the focal plane using patrol radius and minimum target separation. |
| `network_flow.py` | Builds the assignment graph, runs `networkx.min_cost_flow`, `solve_tile_group()` for overlapping tiles, forbidden-assignment repair, and `aggregate_group_assignments_with_pairwise_repair()`. |
| `fba_single_tile.py` | Per-tile MCF helpers: `fba_onetile()` (standard) and `fba_onetile_decollided()` (priority-weighted pre-filter); used by `mcf_singletiles_decollided.py`. |
| `utils.py` | Fiber positions and plate scale from `parameters/`, coordinate transforms (`radec2xy`, `xy2radec`), tile/target masks, neighboring-fiber pairs, overlapped-tile grouping, and FITS output (`write_fba_onetile`). |
| `parameters.py` | Instrument constants: tile inner/outer radius, patrol radius, collision separation (15.625 arcsec ≈ 2 mm on focal plane). |

### Quick example (greedy assignment)

After `pip install -e .`:

```python
import numpy as np
from JUST_fiberassign.greedy_assign import assign_targets_greedy
from JUST_fiberassign.utils import get_fiberpos, radec2xy
from JUST_fiberassign.parameters import R_PATROL

fibers_xy = get_fiberpos()
tile_ra, tile_dec = 35.0, 15.0

# Project targets onto the focal plane
target_ra = np.array([...])
target_dec = np.array([...])
target_x, target_y = radec2xy(tile_ra, tile_dec, target_ra, target_dec)
targets_pos = np.column_stack([target_x, target_y])

priorities = np.ones(len(target_ra))
subpriorities = np.random.rand(len(target_ra))
target_ids = np.arange(len(target_ra))

assigned_ids = assign_targets_greedy(
    fibers_xy, targets_pos, target_ids, priorities, subpriorities,
    radius=R_PATROL, minimum_separation=2.0,
)
```

Without installing, set `PYTHONPATH` to `JUST_fiberassign/JUST_fiberassign/` and import from `greedy_assign` / `utils` directly.

## Scripts (`scripts/`)

### `assign_targets.py` — greedy assignment and completeness

Runs greedy fiber assignment in parallel over a tile list, then estimates assignment completeness vs. magnitude limit. Requires **matplotlib** (`pip install -e ".[visualization]"`).

```bash
export PYTHONPATH="/path/to/JUST_fiberassign/JUST_fiberassign:${PYTHONPATH}"
python scripts/assign_targets.py --ncores 8
```

Run from a working directory that contains:

- `./tiles.fits` — tile centers and IDs
- `./catalog/DESIDR9_galaxy_ngc_rmagcut*.fits` — galaxy catalogs for several `rmagcut` values

Writes `./output/assigned_gal_id_DESIDR9_ngc_rmagcut*.npz` and `./output/completeness_diffmagcut.txt`.

### `mcf_singletiles.py` — single-tile min-cost-flow assignment

Processes one tile at a time to limit memory use. Supports multi-pass surveys, HEALPix-split MTL catalogs, resume from existing output FITS, priority degradation for already-assigned targets, and per-tile timeout fallback (fast MCF without collision gadgets, then pairwise repair).

```bash
export PYTHONPATH="/path/to/JUST_fiberassign/JUST_fiberassign:${PYTHONPATH}"
python scripts/mcf_singletiles.py \
  --rmagcut 20.5 --Npasses 3 \
  --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
  --n_workers 8 --eval_workers 1 \
  --max_iterations 3 --mcf_timeout_sec 240 \
  --rand_seed 3 --nside 32
```

The script currently hardcodes input/output paths under `dir_root` in `main()`; edit that variable for your mock data layout.

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--Npasses` | 3 | Number of survey passes (tiles with `PASS < Npasses`) |
| `--eval_workers` | 1 | Workers inside collision-repair evaluation (use 1 on batch systems) |
| `--max_iterations` | 3 | Forbidden-assignment repair iterations inside `solve_tile_group` |
| `--mcf_timeout_sec` | 240 | Per-tile timeout; falls back to no-gadget solve + pairwise repair |
| `--no_resume` | off | Ignore existing `fba_tile_*.fits` and restart |
| `--nside` | 32 | HEALPix nside for on-disk MTL pixel files |

### `mcf_singletiles_decollided.py` — MCF with decolliding

Same multi-pass workflow as above, but uses `fba_onetile_decollided()` from `fba_single_tile.py` to pre-filter a priority-weighted collision-free target subset before MCF. Tile logic and timeouts are handled in the script:

- **`fba_onetile_decollided` timeout** (default 10 s): fall back to `fba_onetile()` (standard MCF + pairwise repair).
- **Fallback timeout** (default 180 s): if `fba_onetile` also fails, the tile is skipped and recorded in `fba_skipped_tiles.txt`.
- Decollided timeouts are logged in `fba_decollided_timeout_tiles.txt`.


```bash
export PYTHONPATH="/path/to/JUST_fiberassign/JUST_fiberassign:${PYTHONPATH}"
python scripts/mcf_singletiles_decollided.py \
  --input_mockpath "/path/to/lightcone.fits" \
  --input_tilepath "/path/to/tiles_4passes.fits" \
  --output_fba_path "/path/to/fba/output/" \
  --mock_version "v1" \
  --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
  --Npasses 3 \
  --n_workers 8 --eval_workers 1 \
  --max_iterations 3 --rand_seed 100 --nside 32
```

Output is written under `{output_fba_path}{Npasses}passes/{N_tiles}tiles_{ra0}ra{ra1}_{dec0}dec{dec1}/seed{rand_seed}/`.

### `group_sky_coverage/cal_aperture_coverage.py` — cluster aperture coverage

Monte Carlo estimate of the sky fraction covered by galaxy-cluster apertures as a function of mass, redshift, and aperture radius. Requires **matplotlib**. The cluster catalog path is hardcoded in `main()`; edit `ifile` for your data.

```bash
export PYTHONPATH="/path/to/JUST_fiberassign/JUST_fiberassign:${PYTHONPATH}"
python scripts/group_sky_coverage/cal_aperture_coverage.py \
  --LOG_MASS_THRESHOLD 14.0 --R_AP_DEFAULT 8.0 --Z_MIN 0.0 --Z_MAX 1.25
```

| Option | Default | Description |
|--------|---------|-------------|
| `--LOG_MASS_THRESHOLD` | 14.0 | log10 cluster mass cut (Msun/h) |
| `--R_AP_DEFAULT` | 8.0 | Aperture radius (Mpc/h) |
| `--Z_MIN`, `--Z_MAX` | 0.0, 1.25 | Redshift range for analysis |

### `run_assign.pbs` — batch job example

Example PBS script that loads Anaconda, activates the `desi_fiberassign` conda environment, and runs `assign_targets.py --ncores $PBS_NP` from the submit directory. Expects `assign_targets.py` (or a symlink) in the job working directory alongside `./tiles.fits` and `./catalog/`.

## Output format

MCF scripts write one FITS file per tile: `fba_tile_<TILEID>.fits`, with a primary HDU plus two table extensions:

- **ASSIGNED** — columns `TARGETID`, `FIBERID` for fibers that received a target.
- **REACHABLE** — all target–fiber pairs within patrol radius for that tile.

MTL pixel catalogs are stored under `mtl_nside<NSIDE>/mtl_healpix_<PIXID>.fits` inside the run output directory. Assigned targets have their `PRIORITY` degraded on disk between passes (`4.0 → 1.0` in `mcf_singletiles.py`; `100.0 → 2.0` in `mcf_singletiles_decollided.py`).

## Instrument parameters

Default values in `JUST_fiberassign/parameters.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `TILE_INNER_RADIUS_DEG` | 0.1085° | Inner edge of tile annulus |
| `TILE_OUTER_RADIUS_DEG` | 0.5968° | Outer edge of tile annulus |
| `R_PATROL` | 6.0 mm | Fiber patrol radius on focal plane |
| `COLLISION_SEPARATION_ARCSEC` | 15.625 | Minimum angular separation for neighboring fibers |

Instrument data files under `parameters/` (resolved relative to the repo root by `utils.py`):

| File | Purpose |
|------|---------|
| `fiberpos.csv` | Fiber (x, y) positions on the focal plane (2184 fibers) |
| `platescale.txt` | Radial plate scale vs. focal-plane radius |
| `rzsn.txt` | Arc-length lookup used when building the plate-scale table |

## Development

Jupyter notebooks under `nb/` cover tile layout, multi-tile MCF, and completeness plots. They are intended for interactive development.

## License

MIT License. See package metadata in `setup.py`.

## References

- DESI fiber assignment: [desihub/fiberassign](https://github.com/desihub/fiberassign)
- Network-flow fiber assignment methods follow ideas from the SDSS survey design [Blanton et al. 2003](https://iopscience.iop.org/article/10.1086/344761)
