# Kappa Motion Planner

<div align="center">

<img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" />
<img src="https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white" />

</div>

## Description

Kappa Motion Planner is a Python package for motion planning and navigation of autonomous guided vehicles (AGVs), developed within the Arena research project.

The package provides tools and algorithms for agile and reliable robot navigation experiments.

**Authors:**  
- [Sonia De Santis](https://www.mech.kuleuven.be/en/pma/research/meco/people/00153320)  
- [Alejandro Astudillo](https://scholar.google.com/citations?user=9ONkJZAAAAAJ)

---

> [!WARNING]
> This package is part of ongoing research work and is currently under active development.
> The code is provided primarily for research and experimental purposes.

# Installation

The recommended installation method is using a Python virtual environment together with an editable installation.

## 1. Clone the repository

```bash
git clone https://github.com/soniadesantis/kappa-motion-planner.git
cd kappa-motion-planner
```

## 2. Create a virtual environment
Linux/macOS
```bash
python3 -m venv kappa-planner-env
source kappa-planner-env/bin/activate
```

Windows
```bash
python -m venv kappa-planner-env
kappa-planner-env\Scripts\activate
```

After activation, your terminal should display the environment name: 

```bash
(kappa-planner-env)
```

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

## 4. Install the package in editable mode
```bash
pip install -e .
```

This command:

- installs the package locally
- installs all dependencies from pyproject.toml
- keeps the installation linked to the source code

This means that modifications to the source files are immediately reflected without reinstalling the package.

## Running an example
After installation, you can run one of the example scripts: 
```bash
python examples/hello_world.py
```                                          

## Submitting an issue

Please submit an issue if you want to report a bug or propose new features.

## Axis-aligned bicycle baseline and refinement

The implementation lives in `kappa_planner.helpers.baseline_construction`,
`kappa_planner.helpers.refinement`, and
`kappa_planner.helpers.pose_to_circle_bicycle`. Collision checks and safe
corridor unions are in `kappa_planner.helpers.collision_avoidance`.
The routines construct model-specific bicycle primitives. The standing
unicycle planner remains available through `MotionPlanner` with
`assumptions="standing"`.

Use Python 3.10 or newer and run `python -m pip install -e .` in an activated
environment to install the dependencies, including Shapely. From the repository
root, reproduce the short-corridor random experiment with:

```bash
MPLBACKEND=Agg python experiments/bicycle_journal_paper/benchmark_new_bicycle_baseline.py \
    --seed 7 --cases 100 --corridors 10 20 30 40 50 \
    --output experiments/bicycle_journal_paper/results/new_bicycle_baseline/short_corridors_R1_500.json \
    --figure-output experiments/bicycle_journal_paper/figures/new_bicycle_baseline/short_corridors_R1_500.png
```

This uses turning radius `R=1`, footprint radius `0.5`, interior nominal lengths
from `2–8` (plus longitudinal end extensions), and endpoint maneuvering space.
Nonconsecutive corridors cannot overlap in area; boundary touching is allowed.
Construction uses bounded backtracking when a new corridor cannot be placed.

Each successful internal baseline is refined for the same endpoint poses.
If refinement fails, the complete baseline is retained when available. Reports
store all generated bounds, poses, outcomes, failure reasons, and timings.
`comparison` includes baseline fallbacks;
`successful_refinement_comparison` excludes them. The latter's
`traversal_time_reduction_percent.median` reports the successful-refinement-only
median improvement. Computation timings exclude generation, validation, and
plotting; fallback timings count the unsuccessful refinement attempt as well.
The run saves trajectory figures and a separate `_comparison.png` figure.
Computation times vary by PC.

For the focused benchmark checks:

```bash
python -m unittest discover -s tests -p test_random_baseline_boundary_benchmark.py
```
