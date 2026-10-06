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
python examples/hello_world_unicycle.py
python examples/standing_unicycle.py
python examples/axis_aligned_bicycle.py
```                                          

The bicycle example uses a fixed ten-corridor sequence from the random
benchmark, with explicit endpoint poses and turning radius `R=1`. It plots
the complete baseline and refined trajectories, highlights both boundary
connections, and reports traversal times, percentage improvement, and total
planning computation time. No experiment files or random generation are needed.
Optionally save the plot with `--save /tmp/axis_aligned_bicycle.png`.

Select the planner through `MotionPlanner(..., assumptions="standing")` for
the unicycle pipeline (the default), or `assumptions="axis-aligned"` for the
bicycle baseline and refinement pipeline. Other model/planner combinations
are currently unsupported. The former checks the standing assumptions; the
latter checks axis alignment, circular-footprint containment, safe overlaps,
passage directions, and the absence of reversals before constructing a baseline.
Geometrically valid inputs can still fail trajectory construction.

Axis-aligned rectangles can omit `tilt`, which defaults to zero. With the
existing `CorridorWorld` convention at zero tilt, `height` is the extent along
x and `width` is the extent along y. For example, a vertical rectangle with
x extent 4 and y extent 12 is `CorridorWorld(width=12, height=4, center=[4, 4])`.
The axis-aligned planner infers travel directions and lateral walls from
rectangle geometry; the stored tilt does not prescribe traversal.

Both accept optional `start_pose` and `end_pose` in world coordinates. Standing
defaults use the corridor tilt. Axis-aligned defaults lie toward the outer ends
of the first and last corridors, with headings inferred from their overlaps;
ambiguous geometry requires explicit poses. Default placement does not guarantee
a feasible boundary connection. Normalized `relative_start_pose` and
`relative_end_pose` retain the existing shrunken-corridor frame convention:
positions in `[-1, 1]`, with heading in radians relative to the frame's x axis.
Each endpoint accepts either an absolute or a relative pose.

For the bicycle planner, `compute_trajectory_analytical()` attempts refinement
and returns the complete baseline when refinement fails. If neither produces
a complete boundary-connected trajectory, it raises `ValueError`.
Inspect `baseline`, `refinement_result`, `refinement_failure_reason`, and
`solution_source` (`"refined"` or `"baseline"`) for the outcome.
`traversal_time` is the returned trajectory's duration in seconds;
`comp_time_analytical_sol` includes baseline construction, boundary attachment,
assembly, and the refinement attempt, excluding constructor validation and plotting.

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
virtual environment to install the dependencies, including Shapely.
Research experiments and benchmark tests are maintained on `develop`.
This branch provides the planner package and the usage examples listed above.
