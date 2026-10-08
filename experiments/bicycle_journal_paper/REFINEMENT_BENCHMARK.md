# Reversible-skipping benchmark

The laptop run uses seed 7, 100 scenarios for each of 10, 20, 30, 40, and
50 corridors, turning radius R = 1, and circular footprint radius r = 0.5.
All 500 scenarios match the previous benchmark. Successful refinements rose
from 360 to 444, with no previously successful case lost.

| Corridors | Previous successes | Laptop successes |
| --- | ---: | ---: |
| 10 | 91 | 96 |
| 20 | 82 | 95 |
| 30 | 78 | 91 |
| 40 | 65 | 86 |
| 50 | 44 | 76 |

Run from the repository root, using the Python environment with the project
dependencies installed. Keep the workstation results separate from the laptop:

```bash
MPLBACKEND=Agg PYTHONPATH=src python experiments/bicycle_journal_paper/benchmark_new_bicycle_baseline.py \
  --corridors 10 20 30 40 50 --cases 100 --seed 7 \
  --output experiments/bicycle_journal_paper/results/short_corridors_R1_500_reversible_skipping_workstation.json \
  --figure-output experiments/bicycle_journal_paper/figures/new_bicycle_baseline/short_corridors_R1_500_reversible_skipping_workstation.png
```

The committed archive includes the previous report, laptop report, paired
comparison, and historical diagnostic experiments. Extract it to restore the
reports used by the failure-plot script:

```bash
python -m zipfile -e experiments/bicycle_journal_paper/results/refinement_performance/reversible_skipping_laptop_20261008.zip experiments/bicycle_journal_paper/results
```

The laptop JSON includes machine metadata and computation times. Hardware
differs from the previous run, so compare success counts and traversal times
directly; compare computation times using runs on the same machine.

`diagnose_skipping_failures.py` and the old report's failure plots reproduce the
investigation before reversible skipping was introduced. Their failure
assertions can no longer hold for repaired cases under the current planner.
Use `EXAMPLE_THESIS.py --example 36` to inspect the current stage trace.
