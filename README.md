# LAMOS: Location Allocation Model Optimization System

LAMOS is an open-source desktop visual analytics tool for multi-objective
location optimization. It helps a decision-maker define a scenario-specific
demand set (Da) from a large point dataset (D0), compare the two with six
exploratory spatial data analysis (ESDA) methods, and derive non-dominated
facility locations that trade off coverage and access with a Monte Carlo
simulation approach.

It accompanies the paper:

> Cho, S., Dodge, S., Murray, A. T., and Church, R. L. Open-Source Visual
> Analytics for Multi-Objective Location Optimization: A Major League Baseball
> Case Study. *Journal of Spatial Information Science* (under review).

The sample data (batted balls at Dodger Stadium, 2015-2019, with MLB Statcast
fields) reproduce the paper's case study.

## Features

| Window | Method | Paper |
|---|---|---|
| Main window | Load study area and demand points, set parameters, define the scenario with up to three temporal ranges and two attribute values | Figures 3, 5 |
| ESDA 1 | Weighted quadrat analysis: box plots of the per-quadrat difference \|Da - D0\| / D0 for resolutions 1-30, and D0 / Da quadrat maps with a resolution slider | Figure 6 |
| ESDA 2 | Approximated demand surface difference map: relative difference of the ordinary-kriging surfaces of the demand weight, top view and rotatable 3D view | Figure 7 |
| ESDA 3 | LISA: global and local Moran's I of the demand weight with Voronoi-contiguity weights (hotspots / coldspots) | Figure 8 |
| ESDA 4 | Kernel density estimation by temporal unit, all panels on a common extent and aspect ratio | Figure 9 |
| ESDA 5 | Kolmogorov-Smirnov tests (x, y, weight), Monte Carlo Ripley's K with a simulated complete-spatial-randomness reference, Voronoi overlap | Figure 10 |
| ESDA 6 | Demand-weighted k-means clustering with a slider for the number of facilities | Figure 11 |
| 3. Analyze | Monte Carlo multi-objective optimization (coverage and access) and trade-off graphs | Figure 12 |
| 4. Save | Export the non-dominated facility locations as a shapefile | |

## Installation

LAMOS needs Python 3.10-3.13 with Tk (included in the python.org installers)
and Git.

**Windows.** Download `run.bat` to the folder where LAMOS should be installed
and double-click it. It clones (or updates) this repository into `lamos`,
creates a virtual environment, installs the packages and starts the tool.
Double-clicking `run.bat` inside an existing copy runs that copy.

**macOS / Linux.**

```bash
git clone https://github.com/manjukcus/lamos.git
cd lamos
./run.sh
```

With Homebrew Python on macOS, install Tk first (`brew install python-tk`); on
Ubuntu/Debian, `sudo apt install python3-tk python3-venv`.

**Manual installation.**

```bash
git clone https://github.com/manjukcus/lamos.git
cd lamos
python -m venv venv
venv\Scripts\activate          # Windows;  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python lamos_app.py
```

## Quick start: the paper's case study

1. **0. Area**: `sample_data/outfield_poly.shp`.
2. **1. Input**: `sample_data/dodger_of_20152019.shp` (8,788 batted balls).
3. Weight column `ewoba` (xwOBA). The other parameters are pre-filled with the
   values of the paper (3 facilities, coverage distance 90 ft, 30 partitions,
   threshold distance 5, maximum iteration 10, 5 iterations, 100 Voronoi
   iterations, random seed 42). Click **2. Parameter Setting**.
4. Temporal variables: type `game_year`, `month` and `inning` next to
   **Temporal Variable (1)-(3)**, click each button, and set the ranges
   2016-2019, 4-8 and 1-6. Attribute variables: `stand` = `L`, `p_thrws` = `R`.
5. **Visualize**: Da contains 1,228 batted balls.
6. **ESDA** opens the six analyses as sequential pop-up windows; a small menu re-opens any single view.
7. **3. Analyze** runs the Monte Carlo optimization (about 4 minutes with
   100 Voronoi iterations on a recent desktop CPU); **4. Save** exports the
   result.

## Reproducing the figures without the GUI

```bash
python scripts/reproduce_figures.py            # Figures 6-12 into ./figures
python scripts/reproduce_figures.py --only 6 11
python scripts/reproduce_figures.py --quick    # reduced settings, about 1 minute
python scripts/figure4_overview.py             # Figure 4 (data overview, outside the GUI)
```

## Tests

```bash
pip install pytest
python -m pytest tests
```

`tests/test_equivalence.py` runs a reference copy of the original
optimization code (`tests/legacy_reference.py`) next to `lamos_core.py` and
checks that both return identical solutions for the same random seed.

## Code structure

| File | Content |
|---|---|
| `lamos_app.py` | Tkinter GUI |
| `lamos_core.py` | Optimization (lambda-interval heuristic) and Monte Carlo simulation, no GUI dependency |
| `lamos_esda.py` | The six ESDA methods and the result figures, no GUI dependency |
| `scripts/reproduce_figures.py` | Headless reproduction of the case-study figures |
| `tests/` | Equivalence and smoke tests on the sample data |

## Changes in this version (2026-10)

- `lamos_app.py` was a Colab notebook saved under a `.py` name and could not
  be run with `python lamos_app.py`; it is now a Python script, split into the
  GUI, the optimization core and the ESDA module.
- Fixed a syntax error (escaped quotation marks from the notebook export)
  that stopped the program from starting.
- Fixed **3. Analyze**: the coverage step received the unallocated demand table
  and stopped with `KeyError: 'nearest_id'`. With this fixed, the optimization
  reproduces the original algorithm exactly (see Tests).
- Fixed a crash that could occur when a facility received no demand in a Monte Carlo run, and
  clicking a row of the attribute table no longer overwrites the data with
  text values.
- Added the weighted quadrat analysis, demand surface difference map, LISA and
  k-means windows described in the paper (ESDA 1, 2, 3 and 6). These were
  re-implemented for this public release; outputs that depend on random
  sampling can differ slightly from the figures in the paper.
- KDE panels now share a common spatial extent and aspect ratio. Ripley's K is
  compared with complete spatial randomness simulated inside the study area,
  and the Voronoi overlap is reported as a percentage of the study area.
- Distance computations are vectorised and repeated optimization steps are
  cached: the Monte Carlo analysis is about 100 times faster with identical
  results. A random seed (default 42) makes all analyses reproducible.
- Added `run.sh` for macOS/Linux, a figure reproduction script and tests.
