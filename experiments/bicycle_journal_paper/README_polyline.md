# Baseline-polyline circle and tangent experiments

## Current baseline and refinement entry points

### Direct baseline pipeline

The example uses `compute_baseline(corridors, vehicle, initial_pose=None,
final_pose=None, connect_boundaries=False)` by default (`BASELINE_RULE = 'baseline'`). Its body calls the
construction steps in order:

1. Compute and store boundary intersection points and shared boundary segments.
2. Compute overlaps `I_j`, then footprint-safe overlaps `D_j`.
3. Determine unique internal passage directions and initial/final directions.
4. Select the stored intersection on the turn side and construct local `A_j`.
5. Propagate `R_j` by the translated-segment half-strip rule.
6. Reconstruct the polyline backward using compatible-slice midpoints.

The returned report exposes `overlaps`, `boundary_intersections`, `corner_points`,
`admissible_sets`, and `fillet_reachability.reachable_sets`. Intersections also
populate the shared geometry cache for later refinement. Empty `D_j`, empty
`A_j`, ambiguous/unavailable directions or corners, and empty `R_j` reject the
construction at their own stage. An absent pose selects one geometrically
available perpendicular direction, otherwise straight continuation; this
entry point does not search other direction pairs after a later failure.

The direct pipeline performs one input validation and no preliminary `2R`
reachability pass, nominal seed reconstruction, backtracking, optimization,
segment-hypothesis certification, or independent final audit. Local sets have
at most one rounded inequality; their extrema are computed in closed form
instead of enumerating general circle/edge intersection events. Fillets are
assembled directly from the selected points and directions. Existing reference
and diagnostic APIs remain available separately.

### Pose-aware boundary directions

`compute_boundary_directed_baseline(corridors, robot, initial_pose=start_pose,
final_pose=end_pose, method='exact')` constructs a baseline with directions in
the initial and final corridors as well as the internal passages. `method`
can be `heuristic`, `exact`, or `segment`.

For the initial boundary, the reference segment joins the pose position to
the centroid of the first safe overlap `I_1`. Its cross product with the first
internal passage determines the turn sign and selects a perpendicular cardinal
entry direction. At the final boundary, use the last passage followed by the
segment from the last overlap centroid to the final position. Pose headings
are used later by the bicycle boundary connection. Centroids are reference
points; baseline waypoints remain free to move within their local regions.

Without a pose, try feasible perpendicular directions before straight
continuation. A pose exactly at its overlap centroid uses this fallback too.
Parallel reversed reference segments report an unsupported 180-degree turn.
The selected directions are returned in `initial_direction`, `final_direction`,
and `segment_directions`. The internal-only `compute_filleted_baseline*`
constructors remain available with their previous behavior.

### Optional physical boundary connections

Use `compute_baseline(corridors, bicycle, start_pose, end_pose,
connect_boundaries=True)` to construct the baseline first and then try each
boundary independently. The result exposes `initial_connection` and
`final_connection`, each with `status`, `connected`, and `maneuvers`. A failed
connection leaves the baseline and the other connection intact. Missing poses
are skipped; this physical construction currently supports Bicycle vehicles.
With the flag off, no physical connection work runs.

The shared `bicycle_boundary_connections` module reuses the original constrained
pose-to-circle construction and only forward-alignment recovery (strategy 3).
A safe directed circle arc joins the tangent arrival to the first outgoing
baseline contact. The final connection uses the reversed problem and joins
the last incoming baseline contact to the goal pose. Successful connections
replace their endpoint quarter fillets when assembling the complete path.
Collinear boundaries use a direct segment when possible, otherwise a terminal
turning circle tangent to the baseline endpoint. Circle joins cannot cross
unsafe angular gaps. As in the original active pose-to-circle construction,
there is no additional straight-segment containment check.

The example produces two figures: provided corridors with circular footprint
erosion and local `A_j` regions (including both endpoint overlaps), then forward
reachable `R_j` sets with the selected polyline, filleted baseline and successful
boundary connections. `EXAMPLE_NUM`, `BASELINE_RULE`, `USE_BOUNDARY_DIRECTIONS`,
and `CONNECT_BOUNDARIES` at the top control VS Code Run. Connections default to
on in the example; `--no-connect-boundaries` disables them. To also use the
geometry-only boundary direction fallback, add `--no-boundary-directions`.
Physical connections require the map poses, whose positions also determine
the baseline boundary turn signs.

`baseline_time_ms` measures the baseline alone; `total_time_ms` measures the
same call including both boundary attempts, even when an attempt fails. The
second plot and console report both when connections are enabled. Timing
excludes map creation, display geometry and plotting. Refinement is not run.
`--save /tmp/example.png` writes `example_local_regions.png` and
`example_baseline.png`.

Both exact and segment propagation select the midpoint of the last reachable
set's bounding box when feasible, otherwise the midpoint of a feasible slice
through its central x coordinate. Reconstruction proceeds backward through
analytic compatible slices, clips them by the directed `2R` separation, and
takes each midpoint. Segment propagation still uses certified extreme faces
to propagate sets; only point reconstruction uses slices.

### Exact propagation of the full fillet regions

`compute_filleted_baseline_exact(corridors, robot)` is an alternative to the
bounded midpoint heuristic. It reuses the same internal feasibility test,
local `A_j` regions, optional virtual boundary directions, and final geometric
validator. The default heuristic remains available unchanged.

The existing local regions are convex boxes clipped by positive-part circular
inequalities. The new helper `helpers/fillet_reachability.py` retains these
analytic inequalities throughout propagation. For a rightward passage, it
translates the previous set by `2R`, extends it to the right, and intersects
with the next region. The exact transverse projection is retained; constraints
that impose upper rather than lower longitudinal bounds are discarded during
the extension. The other directions follow by symmetry. Projection extrema
are computed from line/arc and arc/arc events, without sampled polygons or an
optimizer. A nonempty terminal set supports backward reconstruction through
exact one-dimensional slices.

`result.fillet_reachability.reachable_sets` contains the curved forward sets;
each exposes `bounds`, `contains(point)`, `slice_interval(axis, value)`, and a
`witness`. `fillet_reachability_empty` certifies incompatibility of the local
region model, to the implementation tolerance. It is not a statement about
arbitrary free-space paths. Numerical reconstruction or validation failure
remains unresolved. Exactness is relative to the existing local safety predicate
and its prerequisites.

Both example scripts accept `--baseline-rule exact`. For the VS Code Run button,
set `BASELINE_RULE = 'exact'`. Random experiment results use an `_exact` filename
suffix to preserve heuristic reports.

### Shared geometry and performance comparison

Successful baseline construction owns a private `SequenceGeometry` context,
shared with circle placement, repair and green-arc connection. It stores the
original overlaps and eroded corridor bounds. Concave corners, boundary
intersections, circle clipping intervals, ordinary directed tangent contacts,
corridor-union boundaries and straight-footprint decisions are computed lazily.
The Boolean internal-feasibility test does not build this context.

Circle and tangent keys include actual centers and radii. The context also
checks corridor bounds, footprint radius and tolerance before reuse. Moving a
circle or changing the robot therefore cannot reuse a stale geometric decision.
Coincident circles keep distinct door records and local interval checks. Repair
memoizes pair validity only for unchanged immutable records within that repair
call; repositioned records trigger new checks.

The successful baseline retains a content snapshot of its validation inputs.
Placement reuses that certificate only while waypoints, regions, corridor bounds,
directions, robot radii and tolerance still match. Edited public arrays trigger
the original geometric validator. All original acceptance rules remain in place.

`benchmark_baseline_refinement.py` runs directly with the VS Code Run button.
It compares the current code to `results/refinement_performance/reference_20261002.zip`,
the preserved pre-optimization sources. It checks all public outputs on the 35
maps, then alternates reference/current timing runs. Every timed run builds a
fresh baseline and context; imports, fixture setup and plotting are excluded.
It saves per-map stage timings and comparisons to
`results/refinement_performance/optimized_comparison.json`. Command-line example:

```bash
python experiments/bicycle_journal_paper/benchmark_baseline_refinement.py --repetitions 20
```

Fine-grained graph timing is available with
`connect_safe_arc_circles(..., timings={})`: setup, tangent generation, candidate
checks, straight-footprint checks and DAG search/reconstruction are measured
separately. Fine-grained clocks are disabled by default and measured in a
separate benchmark pass. This optimization does not change path selection or
the baseline/refinement comparison policy.

### Placement and connection rules

`connect_simple_tangent_chain(placements, baseline, robot, safe_arcs=...)` is an
alternative to the complete graph. After the existing positioning/overlap repairs,
it first builds tangents between neighboring occupied circle groups. These links
check green contacts, directed arc traversal and straight-footprint containment,
including the first entry and final exit. Both aligned alternatives remain available. Consecutive means
neighboring occupied groups, even if aligned doors without circles lie between
them. Such gaps receive the same footprint check as other straight links.

If this initial chain works, return immediately. Otherwise try one bypass group
pair at a time, starting from reachable groups and increasing the number of
skipped groups. After each addition, check whether a complete chain exists.
Stop at the first chain; there is no later shortcutting/length-optimization pass.
The contact-state DP selects compatible aligned alternatives and prevents arcs
from crossing unsafe gaps. It minimizes length among the links already tried,
but the returned path is not globally shortest over all fixed-circle links.

Every straight link checks its radius-r capsule against the original corridor
union between its doors, including zero-length joins. First, two endpoint tests
in one eroded corridor certify the whole segment by convexity. If that fast
certificate fails, the exposed union boundary supplies the analytic containment
test. A failed endpoint certificate alone does not reject safe concave wedges.
The existing containment helper already has this fast path. Raw tangents,
checked contact states, union boundaries and capsule decisions are reused across
repair rounds. Coincident same-turn circles and opposite external tangency keep
their zero-link handling. Supporting disks are not obstacles. No circles move.

The result sets `selection_method='local_repair'`, `safe_arcs_checked=True`,
`footprint_scope='all'`, `tangent_containment_checked=True`, and
`ordered_overlap_crossings_checked=False`. `simple_local_chain_connected` reports
success under this policy. `repair_rounds` records added group pairs.
Corridor order and global intersections remain unchecked. Failure is unresolved.
Pass `check_straight_containment=False` to disable all straight footprint checks
for an explicit diagnostic comparison.

The earlier rule remains available with `strategy='legacy_halfplane'`: neighboring
links plus shortcuts proposed by the signed-distance predicate
`turn*d + radius < -tol` and its contiguous blocks. That strategy keeps footprint
checks on all links and reports `simple_safe_arcs_connected` on success.

The complete graph remains the default library rule, with its original checks.
For a fair comparison with local repair, explicitly pass
`check_overlap_order=False` to
`connect_safe_arc_circles`. `compare_tangent_connection_rules.py` runs all 35
fixed examples, comparing the standard graph, the graph with all straight-footprint
checks, the preserved half-plane strategy, and first-chain local repair with
identical repair-stage caches. It saves outcomes,
selected chains, lengths and separate connection timings. Run it directly in
VS Code or with `--repetitions 20`. Results are written to
`results/refinement_performance/simple_chain_checked_local_comparison.json`, preserving
the earlier benchmarks.

In `example_baseline_construction.py`, `CONNECTION_RULE='compare'` adds a
two-panel `_chain_comparison` figure under equal green-arc and straight-footprint conditions.
`'graph'` displays the standard graph, and `'simple'` displays the local rule.
The corresponding CLI option is `--connection-rule graph|simple|compare`.
Plot labels state which straight checks were performed.

The current incremental refinement step is
`kappa_planner.refinement.place_refinement_circles(baseline, robot)`.
`example_baseline_construction.py` displays three figures by default: the filleted
baseline, supporting-circle placements with their local A_j regions and overlap
repairs, then the collision-free portions of the repaired circles. Set `PLOT_FEASIBILITY=True`
for the additional exact-feasibility plot.
The five priorities are nominal safe-half, shifted safe-half, ordinary nominal,
basic (a,b), and the baseline circle. A proposed center o is accepted only if
its implied vertex p=o+R*u-R*v belongs to A_j. No graph or repairs run in this step.
At aligned overlaps, inspect all corridor-boundary intersections on each side
of the directed baseline line. A side with exactly one point can propose one
circle, using its own directional A_j; absent/multiple intersections or shared
edges are reported separately. The local quarter-circle predicate still requires
a suitable concave corner. There is no baseline-circle fallback at an aligned
overlap. Straight continuation remains available. Specified boundary turns are
included. Placement plots show left/right alternatives and their intersections.

After placement, `restore_opposite_turn_overlaps` restores conflicting opposite
turns to the baseline. `repair_same_turn_overlaps` retains valid same-turn
transitions, tries either existing center as a safe common center otherwise,
then restores unresolved pairs to the baseline. Aligned options implicated in
fallback are removed because they have no baseline circle.

`compute_circle_safe_arcs(placements, baseline, robot)` runs after these repairs.
For each circle at overlap j it clips the full supporting circle to the union
of eroded corridors j and j+1. Circle/rectangle-edge intersections define angular
events; interval midpoints classify the intervening arcs analytically. The
already-certified fillet quarter then bridges the small gap where the footprint
straddles the original corridor union. This yields one continuous interval in
map 1 without filling unrelated unsafe gaps. Raw `eroded_intervals` are retained;
`include_certified_fillet=False` exposes the eroded-only result. Output is
all closed angular intervals, global CCW from +x, with wraparound stored as an
end angle above 2pi. Full circles, disconnected ranges, and isolated contacts are
supported. Coincident circles retain separate results for their own corridor
pairs. The third plot draws these portions in green over dashed supporting
circles and shows the eroded corridor boundaries. `--save` uses `_safe_arcs`.
`connect_safe_arc_circles` then tests every ordered pair of occupied circle groups,
retaining both aligned alternatives and allowing any number of intermediate
groups to be skipped. `allow_skipping=False` (or `ALLOW_SAFE_ARC_SKIPPING=False`
in the example) restores consecutive-only connections. Contacts must lie on green intervals, and
`safe_directed_arc_sweep` checks that arrival-to-departure traversal in the circle's
turn direction stays within one safe component. The tangent-state DAG selects
the shortest compatible chain, including the first entry and last exit arcs.
For each directed tangent, both contacts must lie on the corresponding green
arcs. No additional incident eroded-corridor containment check is imposed: a
green contact can belong to a certified fillet portion bridging the two eroded
corridors. For a link from door a to door b, only intermediate doors a+1,...,b-1
impose crossing constraints, using the full overlaps
`I_j = C_j intersection C_{j+1}` rather than the safe overlaps `D_j`. This includes
aligned doors without a circle; adjacent doors need no additional crossing test.
Shortcuts must cross these overlaps in order. The test uses exact rectangle
clipping and one-dimensional interval propagation, not sampling. Full overlaps
are shared with the baseline geometry context. Zero-length coincident-circle links use
the same rule. These crossing constraints express route progression separately
from footprint safety.

Every tangent also requires its complete radius-r capsule to be contained in
the union of the original corridor subsequence. For a link between zero-based
doors a and b, this is `corridor_bounds[a:b+2]`: both endpoint circles' incident
corridor pairs and all corridors between them. `helpers.corridor_union.CorridorUnion`
first accepts any capsule contained in a single eroded rectangle by convexity.
Otherwise it checks centerline containment, then requires minimum segment-to-union
boundary distance at least r. This is analytic capsule containment, including end
caps, concave wedges and holes. Exact coordinate compression removes internal
rectangle seams; exposed collinear edges are merged. Boundary construction is
O(n^2) and happens lazily once per door pair; tests take O(n log n+B) for B exposed
edges. No sampled path or polygonal approximation of a footprint is used. The
test does not treat skipped supporting disks or the baseline polyline as obstacles.
First/last groups remain mandatory, and alternatives at the same door are never
linked to each other. The green-arc figure now also
shows tangent candidates, selected contacts/connections, and red rings at blocked
contact-order circles. `safe_arcs_connected` certifies green circular portions,
straight footprint clearance and ordered intermediate full-overlap crossings.
Global intersections between nonadjacent path primitives remain unchecked.
`ordered_overlap_crossings_checked` reports the route check;
`tangent_containment_checked` reports the straight footprint check. Rejections
include `straight_footprint_clearance` when green contacts and raw overlaps alone
would have accepted an unsafe line. Coincident same-turn circles retain separate
logical records with shared-contact zero links. Opposite-turn external tangency
also permits a zero-length contact; its radius-r disk must pass the union check.
If no chain exists for the positioned circles, the frozen baseline remains
available; graph failure is not environment infeasibility.
For these fixed circles, green components, entry/exit headings and directed
tangent rules, the search is complete up to numerical tolerance: all ordered
pairs are enumerated, and the tangent-state DAG retains the contact needed to
decide every continuation. Coincident-circle joins include interval boundaries
and incident tangent contacts as finite critical events. Safe-component
unwrapping permits angular prefix minima (plus suffix minima for full circles),
so the DP does not materialize every incoming/outgoing state pair. Ordinary
pair enumeration is O(M^2), and optimization is O(E log E) for bounded component
counts; coincident-circle events can increase E. There is no heuristic limit on
skipped groups or searched chains. This completeness includes the specified
ordered intermediate full-overlap, green-contact and straight footprint checks,
but not global intersections between nonadjacent primitives.
The exact internal polyline test and the green circular intervals continue to
use the original safe/eroded geometry.

### Parked tangent-connection experiments

`connect_refinement_circles(placements, baseline, robot)` tests every ordered pair
of occupied waypoint groups, allowing intermediate circles to be skipped.
Passing `allow_skipping=False` restores consecutive-only links;
the original `connect_consecutive_circles` API also retains that behavior.
Each aligned alternative is a separate node;
alternatives at the same overlap are never linked to each other. Tangent states
store both contacts and their quarter parameters. Directed common tangents must
have contacts in the permitted quarters and the proper eroded corridors, and
cross every intervening safe overlap in order. Coincident equal-turn circles
use zero-length links at critical shared contacts, including incident tangencies.
The DAG minimizes straight-plus-arc length subject to arrival <= departure at
each circle, starting at a first-circle quarter entry and ending at a last-circle
quarter exit. The first and last groups remain mandatory endpoints; only interior
groups can be skipped. Every skipped overlap remains an ordered crossing
constraint. `skipped_waypoints` lists unused groups, excluding unused alternatives
of a selected group. No prescribed vehicle poses are connected. A single group
returns one safe quarter; no circles returns `no_circles`.
`no_circle_chain` (or `no_consecutive_chain` when skipping is disabled) concerns
only the requested fixed-circle construction,
not the environment or the baseline. Global self-intersections are not audited
in this mode. The older connection plotting helper remains available but is
not invoked by the current example. A failed chain leaves the baseline available.
The skipping graph tests O(M^2) circle pairs; each safety check can inspect O(n)
intermediate overlaps. The existing O(E log E) contact-order dynamic program is
reused, with no geometric sampling or additional circle repositioning.

### Earlier graph experiment (separate from the current step)

`kappa_planner.baseline_construction` contains only the baseline feasibility,
waypoint construction and local boundary-fillet functionality.
`kappa_planner.refinement.refine_internal_baseline(baseline, robot)` is the new
refinement workflow: independently move genuine turning circles toward their
local corners, preserve fixed internal-anchor tangency, optionally add safe
quarter-circle choices at aligned doors, and search the tangent-state DAG.
Every link respects ordered safe-overlap crossings. All used arcs stay within
validated quarters. No neighboring-circle repair or merging runs in this workflow.

Run `example_internal_refinement.py` directly in VS Code for the thesis geometry.
It replaces `example_fixed_circle_connection.py`. The low-level fixed-circle
types and functions now live in `kappa_planner.refinement`, not the baseline module.
Use `--benchmark-json PATH` to replay bounds saved by the random baseline test.
`--max-skipped-doors K` optionally limits graph links; the default is exhaustive.
Failure of a limited search or of the global intersection audit is unresolved.
The baseline is returned whenever the new chain fails or is longer.

Aligned-door candidates use one clipped midpoint per perpendicular heading pair
and a continuous arc-in-eroded-rectangle-union certificate. This is a finite,
sufficient placement heuristic, not a complete search over all new turns.

## Earlier experiments (not used by the current refinement workflow)

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

`random_baseline_endpoint_fillets.py` reruns the exact random experiment with
mandatory fillets at both endpoint overlaps. The incoming direction follows
the sampled first corridor heading; the outgoing direction follows the last.
Sequences enter this sample only when both form quarter turns with the adjacent
internal passages. Geometric-class and endpoint-heading rejections are recorded
separately; subsequent empty local/propagated fillet sets remain in the sample.
The first and last A sets are included in propagation, not added after waypoint
selection. `compute_filleted_baseline_exact(..., initial_direction='up',
final_direction='down')` supplies these virtual headings directly; no dummy
boundary positions or boundary segment constraints are needed.

```sh
python experiments/bicycle_journal_paper/random_baseline_endpoint_fillets.py --no-show
```

Defaults compare 100 cases for each of 5, 8, 12 and 20 corridors with sampled
widths 1.1–5, 1.1–8 and 1.1–12 m. Walk lengths and overhangs retain their previous
ranges. Results and 20-case five/eight-corridor overview figures are saved under
`results/random_baseline/endpoint_widths`. Endpoint arcs are orange, internal arcs
green. These are conditional samples, not matched geometric cases across widths;
covered-edge extension can increase final widths beyond their sampled values.

`stress_exact_fillet_propagation.py` compares exact propagation and bounded
construction on the same accepted random corridor sequences (default seed 37,
100 cases each for 5, 8, 12 and 20 corridors). It saves per-case geometry,
statuses, warmed median timings and retained constraint counts in
`results/random_baseline/exact_propagation/stress_report.json`. Generation and
plotting are excluded; full construction includes final geometric validation.
Separate synthetic intersections retain up to 64 curved inequalities to expose
the cost hidden by simple corridor examples. This conditional random-walk sample
does not establish a general bound on runtime or completeness beyond the A model.

```sh
python experiments/bicycle_journal_paper/stress_exact_fillet_propagation.py
python -m unittest discover -s tests -p 'test_fillet_reachability*.py'
```

The stress tests compare 64,000 propagated membership queries against direct
existential predecessor slices in all four directions over multiple geometry
scales. Additional tests check original-region conversion, projection with
multiple constraints, thin lenses, near tangency and degenerate sets.

The running thesis example now uses `compute_filleted_baseline_exact()`:

```sh
python experiments/bicycle_journal_paper/example_planner_thesis.py --no-show
```

Its additional `figures/planner_thesis/planner_thesis_exact_fillet_propagation`
PDF/PNG shows (a) corridors, (b) independent local `A_j` sets over safe overlaps,
and (c) exact forward `F_j` sets with the backward-reconstructed polyline.
Curved display boundaries are sampled only for drawing; feasibility and
reconstruction use the analytic sets. These are internal fillet sets; endpoint
arc alternatives are shown separately in the existing arc-fillets figure.

```sh
MPLBACKEND=Agg PYTHONPATH=src python -m unittest discover -s tests
```

Tests cover baseline and endpoint feasibility, fillet-region geometry, legacy
placement priorities, shift rollback, circle/arc overlaps, turn-directed
tangents, multi-circle skips, and original-door indexing after skips.
