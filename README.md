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

## Reproducing the random bicycle baseline/refinement experiment

Use Python 3.10 or newer (the reference run used Python 3.12.3). After pulling
`develop`, activate your virtual environment and run `python -m pip install -e .`
to install the dependencies, including Shapely.

Run from the repository root:

```bash
MPLBACKEND=Agg python experiments/bicycle_journal_paper/benchmark_new_bicycle_baseline.py \
    --seed 7 --cases 100 --corridors 10 20 30 40 50 \
    --output experiments/bicycle_journal_paper/results/new_bicycle_baseline/short_corridors_R1_500.json \
    --figure-output experiments/bicycle_journal_paper/figures/new_bicycle_baseline/short_corridors_R1_500.png
```

The run generates 100 valid scenarios per corridor count. Turning radius is
`R = 1.0`, footprint radius is `0.5`, and interior nominal lengths are sampled
from `2–8`, subject to connection geometry. Longitudinal end extensions are
added separately. Endpoint corridors reserve maneuvering space. Nonconsecutive
corridors cannot overlap in area; boundary touching is allowed. Blocked
construction backtracks over 1–3 recent corridors before restarting.

Both methods use the same corridors and endpoint poses. An unsuccessful
refinement retains the complete baseline when available. The JSON report stores
each scenario, timings, outcomes, and fallback use. Each group contains:

- `comparison`: paired complete solutions, including baseline fallbacks.
- `successful_refinement_comparison`: paired cases that actually produced a
  refined trajectory, excluding fallbacks. Its
  `traversal_time_reduction_percent.median` is the median of the per-case
  percentage improvements against each case's baseline.

Computation times exclude generation, validation, and plotting. Fallback totals
include the unsuccessful refinement attempt and baseline boundary attachment.
The figure command saves trajectory examples and a separate `_comparison.png`
plot. The seed reproduces scenario generation with the same numerical software;
computation times vary with the PC. Reference versions: NumPy 2.3.2, SciPy 1.16.1,
Matplotlib 3.10.3, Shapely 2.1.2.

To run the focused checks:

```bash
python -m unittest discover -s tests -p test_random_baseline_boundary_benchmark.py
```
