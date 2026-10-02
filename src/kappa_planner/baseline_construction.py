"""Thesis baseline: feasibility of an internal directed 2R-spaced polyline.

Exact polyline existence is separate from bounded fillet-aware construction.
Optional boundary positions infer virtual directions only; pose connections are
not included. compute_filleted_baseline_candidates instead enumerates geometric
boundary-direction alternatives without positions and retains every successful
pair. All indices in diagnostic data are zero-based.
"""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Optional, Tuple

import numpy as np

from .helpers.fillet_backtracking import region_slice
from .helpers.fillet_safety import (
    fillet_vertex_region,
    region_contains_point,
    region_margin,
    revised_corner_condition,
)
from .helpers.nominal_polyline import _bounds, _interval_intersection
from .helpers.sequence_geometry import SequenceGeometry, sequence_geometry

Interval = Tuple[float, float]
Rectangle = Tuple[float, float, float, float]
CARDINAL_DIRECTIONS = ("right", "left", "up", "down")
OPPOSITE = MappingProxyType({"right": "left", "left": "right", "up": "down", "down": "up"})


def _admissible_boundary_directions(corridor, door, internal_direction, tol, *, initial):
    """Constant-time side-extension test on closed, eroded rectangles."""
    c, d = np.asarray(corridor, dtype=float), np.asarray(door, dtype=float)
    if (c.shape != (4,) or d.shape != (4,) or not np.all(np.isfinite(c))
            or not np.all(np.isfinite(d)) or not np.isfinite(tol) or tol < 0
            or internal_direction not in CARDINAL_DIRECTIONS):
        raise ValueError("Require finite rectangle bounds, a cardinal direction and tol>=0.")
    if (c[0] > c[1] or c[2] > c[3] or d[0] > d[1] or d[2] > d[3]
            or d[0] < c[0]-tol or d[1] > c[1]+tol
            or d[2] < c[2]-tol or d[3] > c[3]+tol):
        raise ValueError("Require ordered rectangles with the door inside the eroded corridor.")
    extensions = (c[0] < d[0]-tol, c[1] > d[1]+tol,
                  c[2] < d[2]-tol, c[3] > d[3]+tol)
    if not initial:
        extensions = (extensions[1], extensions[0], extensions[3], extensions[2])
    return tuple(direction for direction, extends in zip(CARDINAL_DIRECTIONS, extensions)
                 if extends and direction != OPPOSITE[internal_direction])


def admissible_initial_directions(first_eroded_corridor, first_door,
                                  first_internal_direction, tol=1e-9):
    """Geometric entry directions, excluding reversal of the first passage.

    Bounds are (xmin, xmax, ymin, ymax). This checks positive upstream space
    beyond the door, using tol; it imposes no R, boundary point or segment length.
    A returned perpendicular direction still requires local fillet validation.
    """
    return _admissible_boundary_directions(first_eroded_corridor, first_door,
                                          first_internal_direction, tol, initial=True)


def admissible_final_directions(last_eroded_corridor, last_door,
                                last_internal_direction, tol=1e-9):
    """Geometric exit directions, excluding reversal; no radius/pose constraints."""
    return _admissible_boundary_directions(last_eroded_corridor, last_door,
                                          last_internal_direction, tol, initial=False)


_DIRECTION_VECTORS = {
    "right": np.array([1., 0.]), "left": np.array([-1., 0.]),
    "up": np.array([0., 1.]), "down": np.array([0., -1.]),
}


def infer_initial_boundary_direction(point, door, tol=1e-9):
    """Infer entry toward a closed safe door; diagonal/interior points give None.

    This O(1) test imposes no coordinate or distance constraint on a waypoint.
    Positions must be finite XY pairs; doors use (xmin, xmax, ymin, ymax).
    """
    point = np.asarray(point, dtype=float)
    bounds = np.asarray(door, dtype=float)
    if (point.shape != (2,) or bounds.shape != (4,)
            or not np.all(np.isfinite(point)) or not np.all(np.isfinite(bounds))
            or bounds[0] > bounds[1] or bounds[2] > bounds[3]
            or not np.isfinite(tol) or tol < 0):
        raise ValueError("Require a finite XY point, ordered door bounds and tol>=0.")
    x, y = point
    xmin, xmax, ymin, ymax = bounds
    if ymin - tol <= y <= ymax + tol:
        if x < xmin - tol:
            return 'right'
        if x > xmax + tol:
            return 'left'
    if xmin - tol <= x <= xmax + tol:
        if y < ymin - tol:
            return 'up'
        if y > ymax + tol:
            return 'down'
    return None


def infer_final_boundary_direction(door, point, tol=1e-9):
    """Infer exit from a safe door toward a point, without connecting to it."""
    incoming = infer_initial_boundary_direction(point, door, tol)
    return {'right': 'left', 'left': 'right', 'up': 'down', 'down': 'up'}.get(incoming)


def _robot_radii(robot):
    try:
        r = float(robot.r if hasattr(robot, "r") else robot.width / 2)
        R = float(robot.R if hasattr(robot, "R") else robot.max_radius)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Robot must provide r and R, or width and max_radius."
        ) from error
    if not np.isfinite(r) or not np.isfinite(R) or not 0 < r < R:
        raise ValueError("Robot radii must be finite and satisfy 0 < r < R.")
    return r, R


@dataclass(frozen=True)
class OrthogonalPolylineFeasibility:
    """Forward reachable intervals and optional complete-sequence diagnostics.

    Rectangles use (xmin, xmax, ymin, ymax). Directions are right, left, up,
    or down. A None reachable interval denotes the empty set. On early
    rejection, directions can be partial and reachable tuples are empty.
    None safe overlaps denote empty regions. Viable intervals are the values
    participating in a complete feasible sequence (None on spacing failure).
    Viable tuples are empty when disabled or on geometric rejection. They are
    diagnostic data, not part of the Boolean existence test.
    """

    feasible: bool
    status: str
    reason: str = ""
    corridor_bounds: Tuple[Rectangle, ...] = ()
    safe_overlaps: Tuple[Optional[Rectangle], ...] = ()
    passage_directions: Tuple[str, ...] = ()
    x_reachable: Tuple[Optional[Interval], ...] = ()
    y_reachable: Tuple[Optional[Interval], ...] = ()
    x_viable: Tuple[Optional[Interval], ...] = ()
    y_viable: Tuple[Optional[Interval], ...] = ()
    _geometry: Optional[SequenceGeometry] = field(default=None, repr=False, compare=False)


def _propagate_coordinate(intervals, relations, minimum_length, tol):
    """Propagate exact directed reachable intervals, one per waypoint.

    From [a,b] into [l,u], equality gives their intersection, increasing
    separation gives [max(l,a+2R),u], and decreasing separation gives
    [l,min(u,b-2R)]. Only one predecessor extreme is needed in each case.

    Exactness in real arithmetic: equality requires the same previous value;
    increasing separation is possible iff z >= a+2R (choose predecessor a);
    decreasing separation is possible iff z <= b-2R (choose predecessor b).
    Intersect with the target domain. Starting from the first domain, induction
    makes each interval exactly the values admitting a feasible prefix.
    An empty interval is absorbing. tol implements numerical boundary slack,
    rather than exact real arithmetic; no grid or optimization is used.
    """
    reachable = [intervals[0]]
    for target, relation in zip(intervals[1:], relations):
        previous = reachable[-1]
        if previous is None:
            reachable.append(None)
            continue
        if relation == "equal":
            candidate = previous
        elif relation == "increase":
            candidate = (previous[0] + minimum_length, target[1])
        elif relation == "decrease":
            candidate = (target[0], previous[1] - minimum_length)
        else:
            raise ValueError("Unknown coordinate relation: " + relation)
        reachable.append(_interval_intersection(candidate, target, tol))
    return tuple(reachable)


def _viable_coordinate(reachable, relations, minimum_length, tol):
    """Optional diagnostic: values belonging to complete coordinate sequences.

    Set V_m = R_m, then V_j = R_j intersect Pre_relation(V_{j+1}).
    Reversing a relation exchanges increase/decrease and preserves equality,
    so the forward routine computes this backward recurrence unchanged.

    Proof by backward induction: any complete sequence must use a value in
    R_j with a compatible successor in V_{j+1}. Conversely, such a value has
    a feasible prefix by forward exactness and a compatible feasible suffix
    by induction. These join at that value because constraints are adjacent.
    Therefore V_j is exactly the complete-sequence projection at position j.
    It remains an interval. If R_m is empty there is no complete sequence.
    """
    if reachable[-1] is None:
        return (None,) * len(reachable)
    inverse = {"increase": "decrease", "decrease": "increase", "equal": "equal"}
    return tuple(reversed(_propagate_coordinate(
        list(reversed(reachable)), [inverse[r] for r in reversed(relations)],
        minimum_length, tol,
    )))


def analyze_orthogonal_polyline_feasibility(
    corridor_list, robot, *, tol=1e-9, compute_viable=True
):
    """Return feasibility and reusable data following the three thesis steps.

    Input corridors must have four finite corners forming positive axis-aligned
    rectangles, and there must be at least three. Robot parameters are r and R
    if provided; otherwise existing planner vehicles use width/2 and max_radius.
    These describe a circular footprint and the prescribed minimum turn radius;
    they must satisfy 0 < r < R. Invalid inputs raise ValueError. Geometric or
    spacing failure returns a report with feasible=False.

    Consecutive safe overlaps must be disjoint, including their boundaries.
    These two assumptions define the nominal class: empty_safe_overlap and
    intersecting_safe_overlaps indicate an assumption failure. Within this
    class, no_orthogonal_connection is a feasibility failure, not an additional
    assumption. insufficient_spacing means passages exist but no complete
    directed 2R-spaced waypoint sequence exists.
    Rejection does not prove absence of paths outside this nominal construction.
    Nonconsecutive corridor overlaps are unrestricted. Closed, degenerate safe
    overlaps and segment lengths exactly 2R are admitted. No alternation of
    segment axes is required by the supplied proposition.

    Interval propagation is mathematically exact, without discretization or
    optimization. Floating-point comparisons use an absolute tolerance in
    coordinate units (tol=0 disables slack). Boundary discrepancies no greater
    than tol are normalized to points. All steps cost O(n) time and storage.
    A backward pass also computes globally viable coordinate intervals for
    visualization; it does not construct a particular waypoint sequence.
    Set compute_viable=False to skip this optional backward pass.
    """
    corridors = tuple(corridor_list)
    if len(corridors) < 3:
        raise ValueError("At least three corridors are required.")
    r, R = _robot_radii(robot)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    bounds = tuple(_bounds(c, tol) for c in corridors)
    doors = []
    directions = []

    def reject(status, reason):
        return OrthogonalPolylineFeasibility(
            False, status, reason, bounds, tuple(doors), tuple(directions)
        )

    # 1. Nonempty safe overlaps D_j = (C_j intersect C_{j+1}) eroded by B_r.
    for j, (a, b) in enumerate(zip(bounds, bounds[1:])):
        x = (max(a[0], b[0]) + r, min(a[1], b[1]) - r)
        y = (max(a[2], b[2]) + r, min(a[3], b[3]) - r)
        x = _interval_intersection(x, x, tol)
        y = _interval_intersection(y, y, tol)
        doors.append(None if x is None or y is None else (*x, *y))
    for j, door in enumerate(doors):
        if door is None:
            return reject("empty_safe_overlap", f"Safe overlap {j} is empty.")

    # 2. Disjoint consecutive safe overlaps, including boundary contacts.
    # Check the nominal assumptions before passage feasibility. Cache the
    # interval compatibility tests for direction inference below.
    compatibility = []
    for j, (a, b) in enumerate(zip(doors, doors[1:]), start=1):
        horizontal = _interval_intersection(a[2:], b[2:], tol) is not None
        vertical = _interval_intersection(a[:2], b[:2], tol) is not None
        if horizontal and vertical:
            return reject("intersecting_safe_overlaps",
                          f"Safe overlaps {j-1} and {j} intersect (contact included).")
        compatibility.append((horizontal, vertical))

    # Passage existence is a feasibility condition within the nominal class.
    for j, ((a, b), (horizontal, vertical)) in enumerate(
        zip(zip(doors, doors[1:]), compatibility), start=1
    ):
        if not horizontal and not vertical:
            return reject("no_orthogonal_connection",
                          f"Internal corridor {j} admits no orthogonal segment.")
        if horizontal:
            # Common Y plus disjoint rectangles implies disjoint closed X
            # intervals: a.xmax < b.xmin or b.xmax < a.xmin. Thus the sign
            # is unique. Common X gives the analogous proof for Y below.
            directions.append("right" if b[0] > a[1] else "left")
        else:
            directions.append("up" if b[2] > a[3] else "down")

    # 3. Exact independent 1-D forward propagation of directed constraints.
    relations = {
        "right": ("increase", "equal"),
        "left": ("decrease", "equal"),
        "up": ("equal", "increase"),
        "down": ("equal", "decrease"),
    }
    x_reachable = _propagate_coordinate(
        [d[:2] for d in doors], [relations[d][0] for d in directions], 2 * R, tol
    )
    y_reachable = _propagate_coordinate(
        [d[2:] for d in doors], [relations[d][1] for d in directions], 2 * R, tol
    )
    feasible = x_reachable[-1] is not None and y_reachable[-1] is not None
    x_viable = y_viable = (None,) * len(doors) if compute_viable else ()
    if feasible and compute_viable:
        x_viable = _viable_coordinate(
            x_reachable, [relations[d][0] for d in directions], 2 * R, tol
        )
        y_viable = _viable_coordinate(
            y_reachable, [relations[d][1] for d in directions], 2 * R, tol
        )
    return OrthogonalPolylineFeasibility(
        feasible, "feasible" if feasible else "insufficient_spacing",
        "" if feasible else "No waypoint sequence satisfies directed 2R constraints.",
        bounds, tuple(doors), tuple(directions), x_reachable, y_reachable,
        x_viable, y_viable,
    )


def check_orthogonal_polyline_feasibility(corridor_list, robot, *, tol=1e-9):
    """Return whether the nominal construction certifies an internal 2R polyline.

    False is returned both when the nominal assumptions are violated and when
    they hold but the orthogonal-passage or spacing constraints are infeasible.
    Use analyze_orthogonal_polyline_feasibility() to distinguish these cases.
    False does not imply that no feasible path exists in the environment.

    See analyze_orthogonal_polyline_feasibility for assumptions, robot parameter
    conventions, numerical tolerance, and diagnostic/reconstruction data.
    This checks only nonempty/disjoint safe overlaps, orthogonal passage
    existence, and terminal forward reachable intervals. It never runs the
    optional backward diagnostic or constructs waypoints or fillets.
    """
    return analyze_orthogonal_polyline_feasibility(
        corridor_list, robot, tol=tol, compute_viable=False
    ).feasible


@dataclass(frozen=True)
class QuarterCircleFillet:
    """A directed radius-R quarter arc replacing one genuine turning waypoint."""

    waypoint_index: int
    center: np.ndarray
    incoming_tangent: np.ndarray
    outgoing_tangent: np.ndarray
    radius: float
    signed_angle: float


@dataclass(frozen=True)
class FilletedBaselineConstruction:
    """Construction report; unsuccessful search is not automatically infeasibility.

    feasible=True certifies the returned waypoint chain and its local fillets,
    to numerical tolerance. certified_infeasible refers only to the nominal
    waypoint/local-fillet-region model, never arbitrary free-space paths.
    An unresolved result has both flags False. polyline is None on failure;
    orthogonal_polyline retains the independently feasible unfilleted seed.

    On success, fillet_regions and fillets have one entry per waypoint. None
    regions mean A_j=D_j (unspecified boundaries and collinear waypoints); None fillets mean
    no arc. On early rejection, region entries may be incomplete.
    Region dictionaries use the existing analytic representation: low/high,
    frame, offset, radius, corner, incoming/outgoing, empty and witness.
    remaining_lengths are the straight lengths after both neighboring trims.
    backtracking_attempts counts candidate waypoint proposals, not full chains.
    midpoint_only_success means the first candidate at every waypoint succeeded.
    first_rejected_waypoint is zero-based and records the first failed candidate
    or empty predecessor domain; None means no geometric rejection was observed.
    """

    feasible: bool
    status: str
    reason: str
    feasibility: OrthogonalPolylineFeasibility
    certified_infeasible: bool = False
    orthogonal_polyline: Optional[np.ndarray] = None
    polyline: Optional[np.ndarray] = None
    fillet_regions: Tuple[Optional[dict], ...] = ()
    fillets: Tuple[Optional[QuarterCircleFillet], ...] = ()
    remaining_lengths: Optional[np.ndarray] = None
    selection_method: Optional[str] = None
    backtracking_attempts: int = 0
    midpoint_only_success: bool = False
    first_rejected_waypoint: Optional[int] = None
    max_violation: Optional[float] = None
    initial_direction: Optional[str] = None
    final_direction: Optional[str] = None
    segment_directions: Tuple[Optional[str], ...] = ()
    _validation_key: Optional[tuple] = field(default=None, repr=False, compare=False)


def _recover_orthogonal_polyline(data, R, tol):
    """Recover an exact forward-pass witness, using predecessor midpoints."""
    if not data.feasible:
        return None
    points = np.empty((len(data.safe_overlaps), 2))
    for k, reachable in enumerate((data.x_reachable, data.y_reachable)):
        points[-1, k] = sum(reachable[-1]) / 2
        for j in range(len(points) - 2, -1, -1):
            direction = _DIRECTION_VECTORS[data.passage_directions[j]][k]
            successor = points[j + 1, k]
            lo, hi = reachable[j]
            if direction == 0:
                points[j, k] = successor
                continue
            bound = (lo, successor - 2 * R) if direction > 0 else (successor + 2 * R, hi)
            interval = _interval_intersection((lo, hi), bound, tol)
            if interval is None:
                raise RuntimeError("Forward reachable intervals failed reconstruction.")
            points[j, k] = sum(interval) / 2
    return points


def compute_orthogonal_polyline(corridor_list, robot, *, tol=1e-9):
    """Return one directed 2R-spaced internal waypoint chain, or None.

    The forward existence check is exact in real arithmetic. Backward midpoint
    reconstruction succeeds whenever it passes; no heuristic alternatives or
    optimizer are needed for rectangular doors alone. No fillets are checked.
    None also covers violations of the nominal assumptions; use the analyzer
    to distinguish rejection reasons. Time and storage are O(n).
    """
    data = analyze_orthogonal_polyline_feasibility(
        corridor_list, robot, tol=tol, compute_viable=False
    )
    return _recover_orthogonal_polyline(data, _robot_radii(robot)[1], tol)


def _build_fillet_regions(corridors, data, seed, r, R, tol, segment_directions=None):
    """Use a corner relevant throughout the safe door, independently of the seed.

    The signed cross products are affine in the waypoint. Checking their minima
    at all four door vertices establishes the same corner for every point in
    the door, including degenerate doors. Ambiguous geometry stays unresolved.
    """
    regions = [None] * len(seed)
    geometry = sequence_geometry(data, r, tol)
    if segment_directions is None:
        segment_directions = (None, *data.passage_directions, None)
    for j in range(len(seed)):
        entry, exit_direction = segment_directions[j:j + 2]
        if entry is None or exit_direction is None:
            continue  # Unspecified boundary retains the original endpoint behavior.
        incoming, outgoing = _DIRECTION_VECTORS[entry], _DIRECTION_VECTORS[exit_direction]
        if np.array_equal(incoming, outgoing):
            continue  # Aligned passages need no fillet; A_j remains D_j.
        turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if turn == 0:
            return regions, "unsupported_turn", f"Waypoint {j} requires a 180-degree turn."
        matches = []
        door = data.safe_overlaps[j]
        vertices = np.array([(x, y) for x in door[:2] for y in door[2:]])
        for corner in geometry.corners(j):
            delta = corner - vertices
            cross_in = incoming[0] * delta[:, 1] - incoming[1] * delta[:, 0]
            cross_out = outgoing[0] * delta[:, 1] - outgoing[1] * delta[:, 0]
            if np.all(turn * cross_in > tol) and np.all(turn * cross_out > tol):
                matches.append(corner)
        if len(matches) != 1:
            return regions, "corner_unresolved", (
                f"Waypoint {j}: expected one relevant concave corner, found {len(matches)}."
            )
        regions[j] = fillet_vertex_region(
            dict(x=door[:2], y=door[2:]), corridors[j:j + 2], matches[0].copy(),
            incoming, outgoing, r, R, tol,
            pair_bounds=data.corridor_bounds[j:j + 2], validate_inputs=False,
        )
    if any(region is not None and region['empty'] for region in regions):
        return regions, "empty_fillet_region", "At least one local fillet region is empty."
    return regions, None, ""


def _midpoint_choices(lo, hi):
    """Midpoint, quarter points, then endpoints; deterministic, no duplicates."""
    seen = set()
    for value in ((lo + hi) / 2, .75 * lo + .25 * hi,
                  .25 * lo + .75 * hi, lo, hi):
        value = float(value)
        if value not in seen:
            seen.add(value)
            yield value


def _backtrack_filleted_polyline(data, regions, R, tol, max_attempts, diagnostics=None):
    """Bounded iterative DFS through signed intervals and analytic curved slices.

    Every candidate lies in the forward reachable intervals. The known successor
    fixes one coordinate and bounds the other by signed 2R separation. Try a
    cheap midpoint first; if needed, intersect the exact curved slice and try
    its midpoint/quarter points/endpoints. Iteration avoids recursion limits.
    Budget counts all proposed points, including failed membership checks.
    Exhaustion means unresolved, not infeasible.
    """
    n = len(data.safe_overlaps)
    def rejected(j):
        if diagnostics is not None and diagnostics.get('first_rejected_waypoint') is None:
            diagnostics['first_rejected_waypoint'] = j

    low = np.array([[x[0], y[0]] for x, y in zip(data.x_reachable, data.y_reachable)])
    high = np.array([[x[1], y[1]] for x, y in zip(data.x_reachable, data.y_reachable)])
    for j, region in enumerate(regions):
        if region is not None:
            low[j] = np.maximum(low[j], region['low'])
            high[j] = np.minimum(high[j], region['high'])
    if np.any(low > high + tol):
        rejected(int(np.flatnonzero(np.any(low > high + tol, axis=1))[0]))
        return None, 0
    near = low > high
    low[near] = high[near] = (low[near] + high[near]) / 2
    points = np.empty((n, 2))
    vectors = [_DIRECTION_VECTORS[d] for d in data.passage_directions]

    def terminal_candidates():
        transverse = 1 - int(np.argmax(abs(vectors[-1])))
        interval = _interval_intersection(
            (low[-1, transverse], high[-1, transverse]),
            (low[-2, transverse], high[-2, transverse]), tol,
        )
        if interval is None:
            return
        lo, hi = low[-1].copy(), high[-1].copy()
        lo[transverse], hi[transverse] = interval
        midpoint = (lo + hi) / 2
        yield midpoint
        # A terminal fillet can exclude the box center. Try its monotone witness
        # clipped to the reachable box, then exact line slices and fixed choices.
        # Membership is always checked by the DFS before accepting a proposal.
        seen = {tuple(midpoint)}
        region = regions[-1]
        if region is not None:
            witness = np.clip(region['witness'], lo, hi)
            if tuple(witness) not in seen:
                seen.add(tuple(witness))
                yield witness
            for axis in (0, 1):
                for fixed in _midpoint_choices(lo[axis], hi[axis]):
                    sliced = region_slice(region, axis, fixed, tol)
                    if sliced is None:
                        continue
                    free = 1 - axis
                    admissible = _interval_intersection((lo[free], hi[free]), sliced, tol)
                    if admissible is None:
                        continue
                    point = np.empty(2)
                    point[axis], point[free] = fixed, sum(admissible) / 2
                    if tuple(point) not in seen:
                        seen.add(tuple(point))
                        yield point
        for x in _midpoint_choices(lo[0], hi[0]):
            for y in _midpoint_choices(lo[1], hi[1]):
                if (x, y) not in seen:
                    seen.add((x, y))
                    yield np.array([x, y])

    def predecessor_candidates(j):
        axis = int(np.argmax(abs(vectors[j])))
        transverse = 1 - axis
        fixed = points[j + 1, transverse]
        if not low[j, transverse] - tol <= fixed <= high[j, transverse] + tol:
            rejected(j)
            return
        lo, hi = low[j, axis], high[j, axis]
        if vectors[j][axis] > 0:
            hi = min(hi, points[j + 1, axis] - 2 * R)
        else:
            lo = max(lo, points[j + 1, axis] + 2 * R)
        # At a preceding perpendicular turn this coordinate is shared with
        # the earlier waypoint. A box look-ahead cheaply avoids dead branches.
        if j and vectors[j - 1][axis] == 0:
            lo, hi = max(lo, low[j - 1, axis]), min(hi, high[j - 1, axis])
        interval = _interval_intersection((lo, hi), (lo, hi), tol)
        if interval is None:
            rejected(j)
            return
        lo, hi = interval
        midpoint = (lo + hi) / 2
        candidate = np.empty(2)
        candidate[axis], candidate[transverse] = midpoint, fixed
        yield candidate.copy()
        if regions[j] is not None:
            sliced = region_slice(regions[j], transverse, fixed, tol)
            if sliced is None:
                return
            interval = _interval_intersection((lo, hi), sliced, tol)
            if interval is None:
                return
            lo, hi = interval
        for value in _midpoint_choices(lo, hi):
            if value == midpoint:
                continue
            candidate[axis] = value
            yield candidate.copy()

    attempts = 0
    stack = [iter(terminal_candidates())]
    while stack and attempts < max_attempts:
        try:
            point = next(stack[-1])
        except StopIteration:
            rejected(n - len(stack))
            stack.pop()
            continue
        j = n - len(stack)
        attempts += 1
        points[j] = point
        if regions[j] is not None and not region_contains_point(point, regions[j], tol):
            rejected(j)
            continue
        if j == 0:
            return points.copy(), attempts
        stack.append(iter(predecessor_candidates(j - 1)))
    return None, attempts


def _joint_fillet_fallback(data, regions, seed, R, tol):
    """Adapt signed reachable intervals to the existing validated joint solver.

    The solver infers signs from a verified seed. Its domains are the forward
    reachable intervals rather than the larger raw doors, so it cannot search
    values excluded by the exact forward pass. Collinear equalities are retained.
    """
    from .helpers.smoothed_polyline import _solve_fillet_waypoints_joint

    nominal = dict(
        polyline=seed,
        doors=[dict(x=x, y=y) for x, y in zip(data.x_reachable, data.y_reachable)],
        directions=['H' if d in ('right', 'left') else 'V'
                    for d in data.passage_directions],
    )
    return _solve_fillet_waypoints_joint(nominal, regions, R, tol)


def _validate_and_build_fillets(points, data, regions, r, R, tol):
    """Check original signed constraints, tangent containment and local arc rule.

    Local region membership alone is not blindly trusted: check the original
    corner predicate and both tangent points in their respective eroded
    corridors before constructing the quarter circles. Remaining straight
    pieces lie in convex eroded corridors. No sampled clearance test is used.
    """
    if points.shape != (len(data.safe_overlaps), 2) or not np.all(np.isfinite(points)):
        return None
    violation = 0.
    vectors = [_DIRECTION_VECTORS[d] for d in data.passage_directions]
    for point, door in zip(points, data.safe_overlaps):
        lo, hi = np.array([door[0], door[2]]), np.array([door[1], door[3]])
        violation = max(violation, float(np.max(lo - point)), float(np.max(point - hi)))
    for j, vector in enumerate(vectors):
        delta = points[j + 1] - points[j]
        transverse = 1 - int(np.argmax(abs(vector)))
        violation = max(violation, abs(float(delta[transverse])), 2 * R - float(delta @ vector))
    fillets = [None] * len(points)
    trims = np.zeros(len(points))
    for j, region in enumerate(regions):
        if region is None:
            continue
        violation = max(violation, -float(region_margin(points[j], region)))
        incoming, outgoing = region['incoming'], region['outgoing']
        prediction = revised_corner_condition(points[j], region['corner'], incoming,
                                             outgoing, R, r, tol)
        if not bool(prediction['predicted']):
            return None
        before, after = points[j] - R * incoming, points[j] + R * outgoing
        for tangent, bounds in zip((before, after), data.corridor_bounds[j:j + 2]):
            lo = np.array([bounds[0] + r, bounds[2] + r])
            hi = np.array([bounds[1] - r, bounds[3] - r])
            violation = max(violation, float(np.max(lo - tangent)), float(np.max(tangent - hi)))
        turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        fillets[j] = QuarterCircleFillet(
            j, points[j] - R * incoming + R * outgoing, before, after,
            R, float(turn * np.pi / 2),
        )
        trims[j] = R
    remaining = np.linalg.norm(np.diff(points, axis=0), axis=1) - trims[:-1] - trims[1:]
    violation = max(violation, -float(np.min(remaining)))
    if not np.isfinite(violation) or violation > tol:
        return None
    return tuple(fillets), remaining, violation


def _baseline_validation_key(points, data, regions, r, R, tol):
    """Content snapshot: mutable public arrays must invalidate validation reuse."""
    def array_key(value):
        a = np.asarray(value)
        return a.shape, a.dtype.str, a.tobytes()
    keys = ('low', 'high', 'corner', 'frame', 'offset', 'incoming', 'outgoing')
    region_keys = tuple(None if region is None else
                        (tuple(array_key(region[k]) for k in keys),
                         region['radius'], region['empty']) for region in regions)
    return (r, R, tol, array_key(points), tuple(map(tuple, data.corridor_bounds)),
            tuple(map(tuple, data.safe_overlaps)), tuple(data.passage_directions), region_keys)


def _baseline_is_valid(baseline, r, R, tol):
    """Reuse the constructor's certificate only when all validated inputs match."""
    key = _baseline_validation_key(baseline.polyline, baseline.feasibility,
                                   baseline.fillet_regions, r, R, tol)
    if getattr(baseline, '_validation_key', None) == key:
        return True
    return _validate_and_build_fillets(baseline.polyline, baseline.feasibility,
                                       baseline.fillet_regions, r, R, tol) is not None


def compute_filleted_baseline(corridor_list, robot, *, tol=1e-9,
                              max_backtracking_attempts=128, use_joint_solver=True,
                              initial_position=None, final_position=None):
    """Construct an internal directed 2R polyline with safe radius-R fillets.

    1. Run the exact forward polyline test and recover an unfilleted seed.
    2. Build analytic local fillet regions only at genuine 90-degree turns.
    3. Try bounded midpoint-first backward selection within reachable intervals.
    4. If needed and enabled, invoke the existing joint LP/SLSQP solver.
    5. Validate every accepted chain geometrically, then return its quarter arcs.

    Success certifies a construction, not completeness of the fillet search.
    Bounded failure and numerical solver failure remain unresolved. The joint
    solver certifies incompatibility only through an infeasible linear outer
    relaxation (to solver tolerance). Empty local regions reject this local
    construction model. A 180-degree turn or unresolved concave-corner geometry
    is reported as unsupported/unresolved, not as a free-space impossibility.

    max_backtracking_attempts is a nonnegative integer counting point proposals;
    zero forces the fallback, or returns unresolved if fallback is disabled.
    Work before selection is O(n); bounded selection uses O(n) storage and at
    most the given candidate budget. No boundary connections, global path optimization,
    arc sampling, or optional backward viable-interval pass are performed.

    Optional initial_position/final_position are finite XY pairs used ONLY to
    infer virtual boundary directions after the exact internal test. No boundary
    alignment, length, pose, or straight connection is imposed. Each provided
    direction enables the local fillet at its boundary door, with tangent
    containment in the two adjacent eroded corridors. Diagonal/interior points
    return boundary_direction_unresolved, leaving internal feasibility intact.
    Omitted positions preserve the original construction without endpoint arcs.
    """
    if (isinstance(max_backtracking_attempts, bool)
            or not isinstance(max_backtracking_attempts, (int, np.integer))
            or max_backtracking_attempts < 0):
        raise ValueError("max_backtracking_attempts must be a nonnegative integer.")
    corridors = tuple(corridor_list)
    data = analyze_orthogonal_polyline_feasibility(
        corridors, robot, tol=tol, compute_viable=False
    )
    r, R = _robot_radii(robot)
    if not data.feasible:
        return FilletedBaselineConstruction(
            False, data.status, data.reason, data,
            certified_infeasible=data.status in ('no_orthogonal_connection', 'insufficient_spacing'),
        )
    seed = _recover_orthogonal_polyline(data, R, tol)
    initial_direction = (None if initial_position is None else
                         infer_initial_boundary_direction(initial_position, data.safe_overlaps[0], tol))
    final_direction = (None if final_position is None else
                       infer_final_boundary_direction(data.safe_overlaps[-1], final_position, tol))
    segment_directions = (initial_direction, *data.passage_directions, final_direction)
    boundary_data = dict(initial_direction=initial_direction, final_direction=final_direction,
                         segment_directions=segment_directions)
    unresolved = []
    if initial_position is not None and initial_direction is None:
        unresolved.append('initial')
    if final_position is not None and final_direction is None:
        unresolved.append('final')
    if unresolved:
        return FilletedBaselineConstruction(
            False, 'boundary_direction_unresolved',
            'Cannot infer an orthogonal boundary direction for: ' + ', '.join(unresolved),
            data, orthogonal_polyline=seed, **boundary_data,
        )
    data = replace(data, _geometry=SequenceGeometry(data.corridor_bounds, r, tol))
    return _construct_filleted_pair(corridors, data, seed, r, R, tol,
                                   max_backtracking_attempts, use_joint_solver,
                                   initial_direction, final_direction)


def _construct_filleted_pair(corridors, data, seed, r, R, tol,
                             max_backtracking_attempts, use_joint_solver,
                             initial_direction, final_direction):
    """Construct one directional alternative using an already-computed forward pass."""
    segment_directions = (initial_direction, *data.passage_directions, final_direction)
    boundary_data = dict(initial_direction=initial_direction, final_direction=final_direction,
                         segment_directions=segment_directions)
    regions, status, reason = _build_fillet_regions(
        corridors, data, seed, r, R, tol, segment_directions
    )
    if status is not None:
        return FilletedBaselineConstruction(
            False, status, reason, data, certified_infeasible=status == 'empty_fillet_region',
            orthogonal_polyline=seed, fillet_regions=tuple(regions),
            **boundary_data,
        )
    diagnostics = {}
    points, attempts = _backtrack_filleted_polyline(
        data, regions, R, tol, max_backtracking_attempts, diagnostics
    )
    validated = None if points is None else _validate_and_build_fillets(
        points, data, regions, r, R, tol
    )
    method = 'midpoint_backtracking'
    status, reason = 'backtracking_unresolved', 'Bounded search found no validated filleted chain.'
    if validated is None and use_joint_solver:
        method = 'joint_solver_fallback'
        solution = _joint_fillet_fallback(data, regions, seed, R, tol)
        if solution['feasible']:
            points = solution['points']
            validated = _validate_and_build_fillets(points, data, regions, r, R, tol)
            status, reason = 'validation_failed', 'Joint solver witness failed geometric validation.'
        else:
            status, reason = solution['status'].lower(), solution['reason']
    if validated is None:
        return FilletedBaselineConstruction(
            False, status, reason, data,
            certified_infeasible=status == 'fillet_regions_globally_incompatible',
            orthogonal_polyline=seed, fillet_regions=tuple(regions),
            selection_method=method, backtracking_attempts=attempts,
            first_rejected_waypoint=diagnostics.get('first_rejected_waypoint'),
            **boundary_data,
        )
    fillets, remaining, violation = validated
    return FilletedBaselineConstruction(
        True, 'feasible', 'Waypoint chain and local quarter-circle fillets verified.', data,
        orthogonal_polyline=seed, polyline=points, fillet_regions=tuple(regions),
        fillets=fillets, remaining_lengths=remaining, selection_method=method,
        backtracking_attempts=attempts, max_violation=violation,
        midpoint_only_success=method == 'midpoint_backtracking' and attempts == len(seed),
        first_rejected_waypoint=diagnostics.get('first_rejected_waypoint'),
        _validation_key=_baseline_validation_key(points, data, regions, r, R, tol),
        **boundary_data,
    )


@dataclass(frozen=True)
class LocalBoundaryFilletOptions:
    """Independent quarter arcs at fixed endpoints; no pair search or waypoint changes.

    initial/final contain (boundary_direction, fillet) pairs. Empty options mean
    no locally validated arc at this chosen vertex, not global infeasibility.
    unresolved records ambiguous concave-corner geometry as (side, direction).
    """

    initial: tuple = ()
    final: tuple = ()
    unresolved: tuple = ()
    initial_regions: tuple = ()
    final_regions: tuple = ()


def compute_local_boundary_fillets(baseline, robot, *, tol=1e-9):
    """Check at most two perpendicular directions at each fixed endpoint.

    Requires a validated internal filleted baseline. Reuses its corridor bounds
    and waypoints. Checks local radius-R safety and both tangent containments;
    does not search waypoint locations or connect boundary poses. Work is O(1).
    The internal >=2R spacing accommodates an added R trim at either end.
    """
    if not baseline.feasible or baseline.polyline is None:
        raise ValueError("A successful internal filleted baseline is required.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    r, R = _robot_radii(robot)
    data, points = baseline.feasibility, baseline.polyline
    geometry = sequence_geometry(data, r, tol)
    results, regions, unresolved = [[], []], [[], []], []
    for side, j in enumerate((0, len(points)-1)):
        bounds = data.corridor_bounds[0 if side == 0 else -1]
        eroded = (bounds[0]+r, bounds[1]-r, bounds[2]+r, bounds[3]-r)
        internal = data.passage_directions[0 if side == 0 else -1]
        infer = admissible_initial_directions if side == 0 else admissible_final_directions
        directions = infer(eroded, data.safe_overlaps[j], internal, tol)
        pair = data.corridor_bounds[j:j+2]
        door = data.safe_overlaps[j]
        vertices = np.array([(x,y) for x in door[:2] for y in door[2:]])
        corners = geometry.corners(j)
        for direction in directions:
            if direction == internal:
                continue  # Only genuine quarter turns are requested.
            incoming, outgoing = (_DIRECTION_VECTORS[d] for d in
                                  ((direction, internal) if side == 0 else (internal, direction)))
            turn = incoming[0]*outgoing[1]-incoming[1]*outgoing[0]
            matches = []
            for corner in corners:
                delta = corner-vertices
                if (np.all(turn*(incoming[0]*delta[:,1]-incoming[1]*delta[:,0]) > tol)
                        and np.all(turn*(outgoing[0]*delta[:,1]-outgoing[1]*delta[:,0]) > tol)):
                    matches.append(corner)
            if len(matches) != 1:
                unresolved.append(('initial' if side == 0 else 'final', direction))
                continue
            region = fillet_vertex_region(dict(x=door[:2], y=door[2:]), (), matches[0],
                                          incoming, outgoing, r, R, tol,
                                          pair_bounds=pair, validate_inputs=False)
            point = points[j]
            if not region_contains_point(point, region, tol):
                continue
            if not revised_corner_condition(point, matches[0], incoming, outgoing,
                                            R, r, tol)['predicted']:
                continue
            before, after = point-R*incoming, point+R*outgoing
            if any(np.any(tangent < np.array([b[0]+r,b[2]+r])-tol)
                   or np.any(tangent > np.array([b[1]-r,b[3]-r])+tol)
                   for tangent,b in zip((before,after),pair)):
                continue
            results[side].append((direction, QuarterCircleFillet(
                j, point-R*incoming+R*outgoing, before, after, R, float(turn*np.pi/2))))
            regions[side].append(region)
    return LocalBoundaryFilletOptions(tuple(results[0]), tuple(results[1]), tuple(unresolved),
                                     tuple(regions[0]), tuple(regions[1]))


@dataclass(frozen=True)
class FilletedBaselineCandidates:
    """All successful boundary-direction pairs and diagnostics for every tried pair.

    candidates retains one validated chain per successful pair, without ranking.
    pair_results also retains empty-region and unresolved-search results. Empty
    candidates alone is not an infeasibility certificate. feasibility describes
    only the unchanged exact internal problem; no boundary poses are connected.
    """

    feasibility: OrthogonalPolylineFeasibility
    initial_directions: Tuple[str, ...] = ()
    final_directions: Tuple[str, ...] = ()
    candidates: Tuple[FilletedBaselineConstruction, ...] = ()
    pair_results: Tuple[FilletedBaselineConstruction, ...] = ()
    status: str = "unresolved"

    @property
    def feasible(self):
        return bool(self.candidates)


def compute_filleted_baseline_candidates(corridor_list, robot, *, tol=1e-9,
                                        max_backtracking_attempts=128,
                                        use_joint_solver=True):
    """Retain all successful geometrically admissible boundary-direction pairs.

    Perform the exact internal pass and seed reconstruction once, then enumerate
    at most nine pairs in right/left/up/down order. Each pair receives its own
    waypoint-proposal budget and optional solver fallback. Boundary directions
    impose local endpoint fillets, never boundary positions or segment lengths.
    Geometry-only direction sets can be nonempty even when radius-R fillets fail.
    The original compute_filleted_baseline API retains its existing semantics.
    """
    if (isinstance(max_backtracking_attempts, bool)
            or not isinstance(max_backtracking_attempts, (int, np.integer))
            or max_backtracking_attempts < 0):
        raise ValueError("max_backtracking_attempts must be a nonnegative integer.")
    corridors = tuple(corridor_list)
    data = analyze_orthogonal_polyline_feasibility(corridors, robot, tol=tol,
                                                compute_viable=False)
    if not data.feasible:
        return FilletedBaselineCandidates(data, status=data.status)
    r, R = _robot_radii(robot)
    data = replace(data, _geometry=SequenceGeometry(data.corridor_bounds, r, tol))
    def eroded(bounds):
        xmin, xmax, ymin, ymax = bounds
        return xmin+r, xmax-r, ymin+r, ymax-r
    initial = admissible_initial_directions(eroded(data.corridor_bounds[0]),
                                           data.safe_overlaps[0],
                                           data.passage_directions[0], tol)
    final = admissible_final_directions(eroded(data.corridor_bounds[-1]),
                                       data.safe_overlaps[-1],
                                       data.passage_directions[-1], tol)
    if not initial or not final:
        return FilletedBaselineCandidates(data, initial, final,
                                          status="no_admissible_boundary_directions")
    seed = _recover_orthogonal_polyline(data, R, tol)
    results = tuple(_construct_filleted_pair(corridors, data, seed, r, R, tol,
                                            max_backtracking_attempts, use_joint_solver,
                                            entry, exit_direction)
                    for entry in initial for exit_direction in final)
    candidates = tuple(result for result in results if result.feasible)
    status = ("feasible" if candidates else "all_pairs_locally_infeasible"
              if all(result.certified_infeasible for result in results)
              else "boundary_construction_unresolved")
    return FilletedBaselineCandidates(data, initial, final, candidates, results, status)
