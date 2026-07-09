# JUST_fiberassign

**JUST_fiberassign** is a Python package for spectroscopic fiber assignment on the Jiao Tong University Spectroscopic Telescope (JUST). It provides focal-plane geometry utilities, greedy assignment, and a min-cost-flow (MCF) solver with fiber-collision constraints. The design takes inspiration from the [DESI fiberassign](https://github.com/desihub/desi) package and is adapted to the JUST focal plane and patrol radius.

## Features

- **Focal-plane geometry**: fiber positions, plate scale, RA/Dec ↔ focal-plane (x, y) transforms, and tile coverage masks.
- **Greedy assignment**: fast per-tile assignment with priority, subpriority, patrol radius, and minimum target separation.
- **Network-flow assignment**: global min-cost-flow optimization with neighboring-fiber collision constraints and pairwise repair.
- **Single-tile MCF pipeline**: multi-pass fiber assignment over survey tiles, with HEALPix-split Master Target Lists (MTL), resume/checkpoint support, and optional priority-weighted decolliding.
- **Survey analysis tools**: completeness studies and galaxy-cluster aperture sky-coverage estimates.

## Package layout

```
JUST_fiberassign/
├── py/                         # Core library
│   ├── assign.py               # Greedy fiber assignment
│   ├── network_flow.py         # Min-cost-flow solver and collision handling
│   ├── fba_single_tile.py      # Single-tile MCF workflow (with decolliding)
│   ├── utils.py                # Geometry, tile masks, I/O helpers
│   └── parameters.py           # Tile radii, patrol radius, collision separation
├── scripts/                    # Command-line workflows
│   ├── assign_targets.py       # Greedy assignment + completeness
│   ├── mcf_singletiles.py      # Single-tile MCF over multiple passes
│   ├── mcf_singletiles_decollided.py  # MCF with decolliding pre-filter
│   ├── group_sky_coverage/
│   │   └── cal_aperture_coverage.py   # Cluster aperture coverage analysis
│   └── run_assign.pbs          # Example PBS job script
├── parameters/                 # Instrument parameters (fiberpos, platescale)
├── nb/                         # Jupyter notebooks (development / analysis)
├── requirements.txt
└── setup.py
```

## Requirements

- **Python**: >= 3.7
- **Core dependencies** (installed by `pip install -e .`):
  - `numpy` >= 1.19.0
  - `scipy` >= 1.7.0
  - `astropy` >= 4.0
  - `pandas` >= 1.3.0
  - `networkx` >= 2.6
  - `healpy` >= 1.14.0 — HEALPix pixelization for MTL workflows (`fba_single_tile.py`, `mcf_singletiles*.py`)
- **Optional**:
  - `matplotlib` >= 3.3.0 — visualization and completeness scripts (`pip install -e ".[visualization]"`)

Install from `requirements.txt` (includes optional `matplotlib`):

```bash
pip install -r requirements.txt
```

Or install the package in editable mode (see below), which pulls in core dependencies automatically.

## Installation

### From a local clone

```bash
git clone git@github.com:zjdingastro/JUST_fiberassign.git
cd JUST_fiberassign

# Recommended: editable install (core + healpy)
pip install -e .

# Include matplotlib for plotting scripts
pip install -e ".[visualization]"
```

### Using scripts without installing

The scripts under `scripts/` expect the `py/` directory on `PYTHONPATH`. Either install the package as above, or run from the repository root with:

```bash
export PYTHONPATH="/path/to/JUST_fiberassign/py:${PYTHONPATH}"
python scripts/mcf_singletiles_decollided.py --help
```

## Core library (`py/`)

| Module | Description |
|--------|-------------|
| `assign.py` | `assign_targets_greedy()` — greedy assignment on the focal plane using patrol radius and minimum target separation. |
| `network_flow.py` | Builds the assignment graph, runs `networkx.min_cost_flow`, enforces collision constraints, and applies pairwise repair. |
| `fba_single_tile.py` | End-to-end single-tile MCF: load MTL galaxies, optional decolliding, solve, and return assignments. |
| `utils.py` | Fiber positions, coordinate transforms (`radec2xy`, `xy2radec`), tile/target masks, neighboring-fiber pairs, overlapped-tile grouping, and FITS output (`write_fba_onetile`). |
| `parameters.py` | Instrument constants: tile inner/outer radius, patrol radius, collision separation (15.625 arcsec ≈ 2 mm on focal plane). |

### Quick example (greedy assignment)

```python
import numpy as np
from assign import assign_targets_greedy
from utils import get_fiberpos, radec2xy

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
    radius=6.0, minimum_separation=2.0,
)
```

## Scripts (`scripts/`)

### `assign_targets.py` — greedy assignment and completeness

Runs greedy fiber assignment over a tile list and galaxy catalog, then estimates assignment completeness vs. magnitude limit.

```bash
python scripts/assign_targets.py --ncores 8
```

Expects local input files such as `tiles.fits` and galaxy catalogs under `./catalog/`. Writes assigned target IDs and completeness tables under `./output/`.

### `mcf_singletiles.py` — single-tile min-cost-flow assignment

Processes one tile at a time to limit memory use. Supports multi-pass surveys, HEALPix-split MTL catalogs, resume from existing output FITS, and priority degradation for already-assigned targets.

```bash
python scripts/mcf_singletiles.py \
  --rmagcut 20.5 --Npasses 3 \
  --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
  --n_workers 8 --eval_workers 1 \
  --max_iterations 3 --rand_seed 3 --nside 32
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--Npasses` | 3 | Number of survey passes (tiles with `PASS < Npasses`) |
| `--eval_workers` | 1 | Workers inside collision-repair evaluation (use 1 on batch systems) |
| `--mcf_timeout_sec` | 240 | Per-tile timeout; falls back to no-gadget solve + pairwise repair |
| `--no_resume` | off | Ignore existing `fba_tile_*.fits` and restart |
| `--nside` | 32 | HEALPix nside for on-disk MTL pixel files |

### `mcf_singletiles_decollided.py` — MCF with decolliding

Same multi-pass workflow as above, but first applies a priority-weighted collision-free target subset (`fba_onetile_decollided`). On timeout, falls back to the standard single-tile solver.

```bash
python scripts/mcf_singletiles_decollided.py \
  --mock_version "v1" \
  --input_mockpath "/path/to/lightcone.fits" \
  --input_tilepath "/path/to/tiles_4passes.fits" \
  --output_fba_path "/path/to/fba/output/" \
  --ra0 30 --ra1 40 --dec0 10 --dec1 20 \
  --Npasses 3 \
  --n_workers 8 --eval_workers 1 \
  --max_iterations 1 --rand_seed 100 --nside 32
```

### `group_sky_coverage/cal_aperture_coverage.py` — cluster aperture coverage

Monte Carlo estimate of the sky fraction covered by galaxy-cluster apertures as a function of mass, redshift, and aperture radius.

```bash
python scripts/group_sky_coverage/cal_aperture_coverage.py \
  --LOG_MASS_THRESHOLD 14.0 --R_AP_DEFAULT 8.0 --Z_MIN 0.0 --Z_MAX 1.25
```

### `run_assign.pbs` — batch job example

Example PBS script that activates a conda environment and runs `assign_targets.py` with the allocated core count.

## Output format

MCF scripts write one FITS file per tile: `fba_tile_<TILEID>.fits`, with two table extensions:

- **ASSIGNED** — columns `TARGETID`, `FIBERID` for fibers that received a target.
- **REACHABLE** — all target–fiber pairs within patrol radius for that tile.

MTL pixel catalogs are stored under `mtl_nside<NSIDE>/mtl_healpix_<PIXID>.fits` inside the run output directory. Assigned targets have their `PRIORITY` degraded on disk between passes.

## Instrument parameters

Default values in `py/parameters.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `TILE_INNER_RADIUS_DEG` | 0.1085° | Inner edge of tile annulus |
| `TILE_OUTER_RADIUS_DEG` | 0.5968° | Outer edge of tile annulus |
| `R_PATROL` | 6.0 mm | Fiber patrol radius on focal plane |
| `COLLISION_SEPARATION_ARCSEC` | 15.625 | Minimum angular separation for neighboring fibers |
| `N_FIBERS` | 2184 | Fibers per tile (from `parameters/fiberpos.csv`) |

Fiber positions and plate-scale data live in `parameters/fiberpos.csv`, `parameters/platescale.txt`, and `parameters/rzsn.txt`.

## Development

Jupyter notebooks under `nb/` cover tile layout, multi-tile MCF, completeness plots, and fiber-motion animation. They are intended for interactive development and are not required for running the scripts.

## License

MIT License. See package metadata in `setup.py`.

## References

- DESI fiber assignment: [desihub/fiberassign](https://github.com/desihub/fiberassign)
- Network-flow fiber assignment methods follow ideas from the SDSS survey design [Blanton et al. 2003](https://iopscience.iop.org/article/10.1086/344761)
