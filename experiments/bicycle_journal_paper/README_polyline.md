# Baseline-polyline circle and tangent experiments

`examples_maps_polyline.py` provides 35 axis-aligned corridor examples. Set
`EXAMPLE_NUM` near the top of that file to inspect an example interactively.
The helpers and diagnostics are experimental; they do not replace the main
motion planner or produce a fully certified start-to-end vehicle trajectory.

## Construction

1. Build safe overlaps `D_j`, infer alternating orthogonal directions, and find
   a baseline polyline with segment lengths at least `2R`. Preserve local
   guaranteed/coupled spacing flags before propagation. Endpoint extensions and
   fillet-aware waypoint selection reuse the same corridor geometry.
2. Place circles from baseline turn signs and internal corners using the legacy
   placement priorities. Circle indices retain the original corridor mapping.
3. Check circle membership in `A_j`, consecutive opposite-turn overlaps, and
   consecutive same-turn quarter-arc intersections. Locally repair same-turn
   conflicts by making centers coincide; separate opposite-turn circles using
   admissible shifts or restoration toward the certified baseline centers.
   Rejected proposals preserve the existing centers and diagnostic flags.
4. Build unrestricted directed tangents: external for equal turn signs,
   internal for opposite signs. When consecutive finite tangent segments
   intersect, try bypassing progressively larger blocks of circles. Ordinary
   shared endpoints are allowed. Skips never mutate the input circle sequence.
5. Intersect supporting tangent lines to construct a generally nonorthogonal
   polyline. Check each retained vertex against its **original `D_j`**, not
   `A_j`. Membership is diagnostic only; it does not trigger skipping.

The first and last supporting tangents use the baseline entry/exit headings.
Coincident equal-turn circles use the first circle's baseline exit tangent as
an explicit convention. Parallel lines or unavailable internal tangents are
reported rather than replaced with invented vertices.

The separate `bp_tangent_chain.py` experiment retains the stronger fixed-quarter
contact, corridor-clearance and used-arc checks. Its `CONNECTED` status concerns
only the internal circle chain. Its results should not be confused with the
unrestricted tangent-polyline diagnostic. Neither algorithm proves global
optimality or completeness; endpoint maneuvers remain outside this work.

## Reproduce the diagnostics

Run from the repository root with the project installed, or set `PYTHONPATH=src`.
For noninteractive runs, set `MPLBACKEND=Agg`.

```sh
python experiments/bicycle_journal_paper/overview_spacing_coupling.py
python experiments/bicycle_journal_paper/check_bp_circle_overlaps.py
python experiments/bicycle_journal_paper/overview_bp_circle_shifts.py
python experiments/bicycle_journal_paper/demo_opposite_circle_overlap.py --repair
python experiments/bicycle_journal_paper/overview_bp_tangent_chains.py
python experiments/bicycle_journal_paper/overview_bp_tangent_polylines.py
python experiments/bicycle_journal_paper/overview_bp_tangent_polylines.py --keep-all
python experiments/bicycle_journal_paper/benchmark_bp_circle_to_polyline.py
```

Scripts write PDFs, CSVs and explanations below `figures/`; generated research
outputs remain ignored by Git. The benchmark builds baselines outside the
timed section, then measures circle placement, diagnostics, both repair passes,
tangents, skipping, and `D_j` checks with one warm-up and five timed passes.
It excludes plotting and endpoint maneuvers.

## Tests

```sh
MPLBACKEND=Agg PYTHONPATH=src python -m unittest discover -s tests
```

Tests cover baseline and endpoint feasibility, fillet-region geometry, legacy
placement priorities, shift rollback, circle/arc overlaps, turn-directed
tangents, multi-circle skips, and original-door indexing after skips.
