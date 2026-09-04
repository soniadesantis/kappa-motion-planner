# Kappa Motion Planner

<div align="center">

<img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" />
<img src="https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white" />

</div>

## Description

Kappa Motion Planner is a Python package for analytical motion planning and
navigation of autonomous guided vehicles (AGVs) in sequences of corridors.

The package provides tools and algorithms for agile and reliable robot navigation experiments.

**Authors:**  
- [Sonia De Santis](https://www.mech.kuleuven.be/en/pma/research/meco/people/00153320)  
- [Alejandro Astudillo](https://scholar.google.com/citations?user=9ONkJZAAAAAJ)

---

> [!NOTE]
> The `paper-version` branch contains the implementation prepared for the
> revised paper and its reproducibility experiments. Use the matching frozen
> release tag for published results; development work may continue on other
> branches.

## Paper experiments

The parameter sweeps and analysis scripts live in the companion
[`kappa-planner-benchmark`](https://github.com/soniadesantis/kappa-planner-benchmark)
repository. For reproducible runs, install the planner revision recorded by
the benchmark metadata and run experiments from the benchmark environment.

## Installation

The recommended installation method is using a Python virtual environment together with an editable installation.

### 1. Clone the repository

```bash
git clone https://github.com/soniadesantis/kappa-motion-planner.git
cd kappa-motion-planner
git switch paper-version
```

### 2. Create a virtual environment
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

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install the package in editable mode
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
