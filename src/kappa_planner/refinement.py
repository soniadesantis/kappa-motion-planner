"""Refinement experiments, starting with independent five-priority placement.

The current incremental entry point is place_refinement_circles: it performs
placement, A_j checks, and consecutive overlap detection. The earlier tangent-graph experiment remains
separate and is not invoked by that entry point or the baseline example.
The example then restores conflicting opposite-turn neighbors to their frozen
baseline placements and checks/repairs same-turn overlapping pairs.
compute_circle_safe_arcs joins eroded-corridor portions through the certified
fillet. connect_safe_arc_circles connects ordered subsets while remaining on
these green arcs. Other tangent-graph routines remain separate experiments.

Same-turn repair tries the two existing centers, retaining local safety. Quarter arcs and
ordered safe-overlap crossings define the connection class. The original
filleted baseline remains available if preferred placement cannot improve it.
"""
from dataclasses import dataclass, replace
from bisect import bisect_right
from types import MappingProxyType
from typing import Optional, Tuple
from time import perf_counter
import numpy as np

from .baseline_construction import (
    Rectangle, _DIRECTION_VECTORS, _robot_radii, _validate_and_build_fillets, _baseline_is_valid,
)
from .helpers.fillet_safety import (
    region_contains_point, axis_aligned_concave_corners, fillet_vertex_region,
)
from .helpers.sequence_geometry import sequence_geometry

@dataclass(frozen=True)
class IndependentCirclePlacement:
    """One circle proposal; its implied vertex, not its center, belongs to A_j."""

    waypoint_index: int
    rule: str
    center: np.ndarray
    vertex: np.ndarray
    radius: float
    region: dict
    in_admissible_region: bool
    rejected_candidates: tuple = ()
    side: Optional[str] = None  # Left/right of an aligned directed passage.


@dataclass(frozen=True)
class IndependentCirclePlacements:
    """Local placements and optional repair diagnostics; no connected-path claim."""

    status: str
    circles: tuple = ()
    skipped_waypoints: tuple = ()
    elapsed_ms: float = 0.
    aligned_sides: tuple = ()
    overlap_pairs: tuple = ()  # Indices into circles, not waypoint indices.
    overlap_blocks: tuple = ()
    restored_waypoints: tuple = ()
    removed_aligned_options: tuple = ()
    coincident_waypoints: tuple = ()
    same_turn_transitions: tuple = ()  # (circle index, circle index, valid)


def _placement_turn(circle):
    u, v = circle.region['incoming'], circle.region['outgoing']
    return u[0]*v[1]-u[1]*v[0]


@dataclass(frozen=True)
class CircleSafeArcs:
    """Safe parts from eroded corridors plus the certified local fillet.

    Angles are global counterclockwise radians from +x, independent of turn.
    Closed intervals have 0 <= start < 2pi and start <= end <= start+2pi.
    An interval crossing angle zero has end > 2pi. (0,2pi) is a full circle;
    (a,a) is an isolated safe contact. Empty intervals means no safe portion.
    Coincident circles retain separate results for their different corridor pairs.
    """

    circle_index: int
    waypoint_index: int
    intervals: tuple
    eroded_corridors: tuple
    eroded_intervals: tuple = ()
    certified_quarter: Optional[tuple] = None


def circle_safe_angular_intervals(center, radius, eroded_corridors, *, tol=1e-9):
    """Analytically clip a circle to a union of closed axis-aligned rectangles.

    Boundary intersections partition the circle into constant-membership open
    arcs. One midpoint per arc classifies the entire arc; isolated boundary
    contacts are retained separately. No angular sampling or quarter constraint
    is used. For two rectangles this takes bounded work (eight edges). Comparisons
    use spatial tol; returned angles are on the actual rectangle boundaries.
    """
    center = np.asarray(center, dtype=float)
    if (center.shape != (2,) or not np.all(np.isfinite(center))
            or not np.isfinite(radius) or radius <= 0 or not np.isfinite(tol) or tol < 0):
        raise ValueError('Require a finite center, radius>0 and finite tol>=0.')
    boxes = []
    for bounds in eroded_corridors:
        if np.shape(bounds) != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError('Expected finite rectangle bounds.')
        if bounds[0] <= bounds[1] and bounds[2] <= bounds[3]:
            boxes.append(bounds)
    if not boxes:
        return ()
    tau = 2*np.pi
    events = [0., tau]
    for xmin, xmax, ymin, ymax in boxes:
        for axis, walls, span in ((0, (xmin, xmax), (ymin, ymax)),
                                   (1, (ymin, ymax), (xmin, xmax))):
            for wall in walls:
                delta = wall-center[axis]
                if abs(delta) > radius:
                    continue
                offset = np.sqrt(max(0., (radius-delta)*(radius+delta)))
                for sign in (-1, 1):
                    point = center.copy()
                    point[axis], point[1-axis] = wall, center[1-axis]+sign*offset
                    if span[0]-tol <= point[1-axis] <= span[1]+tol:
                        events.append(float(np.arctan2(point[1]-center[1], point[0]-center[0]) % tau))
    events = sorted(set(events))
    def contained(angle):
        p = center+radius*np.array([np.cos(angle), np.sin(angle)])
        return any(xmin-tol <= p[0] <= xmax+tol and ymin-tol <= p[1] <= ymax+tol
                   for xmin, xmax, ymin, ymax in boxes)
    pieces = [(a, b) for a, b in zip(events, events[1:]) if contained((a+b)/2)]
    pieces.extend((a, a) for a in events if contained(a))
    return _merge_circle_intervals(pieces)


def _merge_circle_intervals(intervals):
    """Union closed circular intervals, including intervals crossing angle zero."""
    tau = 2*np.pi
    pieces = []
    for start, end in intervals:
        if end-start >= tau:
            return ((0., tau),)
        a = start % tau
        b = a+(end-start)
        if b > tau:
            pieces.extend(((a, tau), (0., b-tau)))
        else:
            pieces.append((a, b))
    merged = []
    for a, b in sorted(pieces):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[0][0] == 0. and merged[-1][1] == tau:
        if merged[-1][0] == tau:
            merged.pop()  # Duplicate isolated representation of angle zero.
        elif merged[0][1] == 0.:
            merged.pop(0)
        else:
            wrapped = (merged[-1][0], tau+merged[0][1])
            merged = merged[1:-1]+[wrapped]
    # An isolated contact at zero can otherwise be represented as (2pi,2pi).
    return tuple(sorted((0., 0.) if a == b == tau else (a, b) for a, b in merged))


def compute_circle_safe_arcs(placements, baseline, robot, *, include_certified_fillet=True, tol=1e-9):
    """Compute safe angular intervals after all circle repositioning/merging.

    Start with individually eroded corridors j and j+1. Add the already-certified
    fillet quarter: its footprint can straddle the original corridor union,
    safely bridging the small artificial gap between eroded-corridor intervals.
    No other gaps are filled. Raw eroded intervals remain available for diagnosis.
    """
    r, R = _robot_radii(robot)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    results = []
    geometry = sequence_geometry(baseline.feasibility, r, tol)
    for i, circle in enumerate(placements.circles):
        j = circle.waypoint_index
        if not 0 <= j < len(baseline.feasibility.corridor_bounds)-1 or abs(circle.radius-R) > tol:
            raise ValueError('Circle does not match the corridor sequence or turning radius.')
        boxes = geometry.eroded[j:j+2]
        eroded = geometry.circle_intervals(j, circle.center, circle.radius, circle_safe_angular_intervals)
        intervals, quarter = eroded, None
        if include_certified_fillet:
            u, v = circle.region['incoming'], circle.region['outgoing']
            vertex = circle.center+R*u-R*v
            if not region_contains_point(vertex, circle.region, tol):
                raise ValueError('Circle has no valid local fillet certificate.')
            # CCW set representation, irrespective of the actual traversal sign.
            radial = -v if _placement_turn(circle) > 0 else u
            start = float(np.arctan2(radial[1], radial[0]) % (2*np.pi))
            quarter = (start, start+np.pi/2)
            intervals = _merge_circle_intervals((*eroded, quarter))
        results.append(CircleSafeArcs(i, j, intervals, boxes, eroded, quarter))
    return tuple(results)


def has_safe_circle_transition(first, second, baseline, robot, *, tol=1e-9):
    """Pair-local directed quarter/tangent check, including corridor crossings.

    Coincident circles must share a safe contact on both permitted quarters.
    This is not a certificate of contact order through a longer circle chain.
    """
    from .helpers.tangent_refinement import _directed_tangent_lines, certified_angle
    r, _ = _robot_radii(robot)
    if first.waypoint_index >= second.waypoint_index:
        return False
    nodes = [dict(index=c.waypoint_index, center=c.center, radius=c.radius,
                  turn=_placement_turn(c), region=c.region) for c in (first, second)]
    a, b = nodes
    bounds = baseline.feasibility.corridor_bounds
    doors = baseline.feasibility.safe_overlaps[first.waypoint_index+1:second.waypoint_index]
    if (a['turn'] == b['turn'] and abs(first.radius-second.radius) <= tol
            and np.linalg.norm(first.center-second.center) <= tol):
        boxes = list(doors) + [(x+r, X-r, y+r, Y-r) for x, X, y, Y in
                              (bounds[first.waypoint_index+1], bounds[second.waypoint_index])]
        contacts = [second.center-second.radius*second.region['outgoing'],
                    second.center+second.radius*second.region['incoming']]
        candidates = [dict(start=p, end=p) for p in _coincident_contacts(a, boxes, contacts, tol)]
    else:
        geometry = sequence_geometry(baseline.feasibility, r, tol)
        candidates = geometry.tangents(a, b, _directed_tangent_lines)
    for candidate in candidates:
        start, end = candidate['start'], candidate['end']
        first_angle, second_angle = certified_angle(a, start, tol), certified_angle(b, end, tol)
        if first_angle is None or second_angle is None:
            continue
        if (_point_in_eroded_bounds(start, bounds[first.waypoint_index+1], r, tol)
                and _point_in_eroded_bounds(end, bounds[second.waypoint_index], r, tol)
                and ordered_safe_overlap_crossings(start, end, doors, tol=tol) is not None):
            return True
    return False


def repair_same_turn_overlaps(placements, baseline, robot, *, tol=1e-9):
    """One ordered local sweep; retain valid tangents, otherwise try either center.

    A candidate common center must retain both original directional A_j safety
    tests and admit a safe coincident transition. Prefer no disk penetration of
    the baseline polyline, then less penetration and less movement. Tangency is
    allowed. Reject moves creating opposite-turn overlaps or breaking a formerly
    valid neighboring transition. No new center search or global chain claim is
    made. Unresolved overlapping pairs return to their baseline circles; aligned
    options without baseline circles are removed. Restoration repeats for newly
    exposed overlapping conflicts, without retrying shifts or cycling.
    """
    started = perf_counter()
    circles = list(placements.circles)
    moved = set(placements.coincident_waypoints)
    transition_cache = {}
    def transition(a, b):
        # Keep object references in each entry: replaced records invalidate the
        # check, and Python cannot recycle an identity while it remains cached.
        key = id(a), id(b)
        if key not in transition_cache:
            transition_cache[key] = a, b, has_safe_circle_transition(a, b, baseline, robot, tol=tol)
        return transition_cache[key][2]
    def adjacent(sequence):
        groups = []
        for i, circle in enumerate(sequence):
            if not groups or sequence[groups[-1][0]].waypoint_index != circle.waypoint_index:
                groups.append([])
            groups[-1].append(i)
        return [(i, k) for a, b in zip(groups, groups[1:]) for i in a for k in b]
    neighbors = adjacent(circles)
    for i, k in neighbors:
        a, b = circles[i], circles[k]
        if (_placement_turn(a) != _placement_turn(b)
                or np.linalg.norm(a.center-b.center) >= a.radius+b.radius-tol
                or transition(a, b)):
            continue
        options = []
        for center in (a.center, b.center):
            replacements = []
            for c in (a, b):
                vertex = center+c.radius*c.region['incoming']-c.radius*c.region['outgoing']
                if not region_contains_point(vertex, c.region, tol):
                    break
                replacements.append(replace(c, center=_frozen_array(center), vertex=_frozen_array(vertex)))
            if len(replacements) != 2 or not transition(*replacements):
                continue
            trial = circles.copy()
            trial[i], trial[k] = replacements
            acceptable = True
            for h, j in neighbors:
                if not ({h, j} & {i, k}):
                    continue
                c, d = trial[h], trial[j]
                if (_placement_turn(c) != _placement_turn(d)
                        and np.linalg.norm(c.center-d.center) < c.radius+d.radius-tol):
                    acceptable = False
                    break
                if transition(circles[h], circles[j]) and not transition(c, d):
                    acceptable = False
                    break
            if not acceptable:
                continue
            penetration = 0.
            for start, end in zip(baseline.polyline, baseline.polyline[1:]):
                delta = end-start
                t = np.clip((center-start)@delta/(delta@delta), 0., 1.) if delta@delta else 0.
                penetration += max(0., a.radius-np.linalg.norm(center-(start+t*delta))-tol)
            movement = np.linalg.norm(center-a.center)+np.linalg.norm(center-b.center)
            options.append(((penetration > 0., penetration, movement), replacements))
        if options:
            _, replacements = min(options, key=lambda option: option[0])
            for index, new in zip((i, k), replacements):
                if np.linalg.norm(circles[index].center-new.center) > tol:
                    moved.add(new.waypoint_index)
                    new = replace(new, rule='same_turn_coincident')
                circles[index] = new
    result = _restore_overlap_baselines(replace(placements, circles=tuple(circles)),
                                       baseline, robot=robot, tol=tol, transition=transition)
    circles = result.circles
    pairs, blocks = result.overlap_pairs, result.overlap_blocks
    transitions = tuple((i, k, transition(circles[i], circles[k]))
                        for i, k in pairs if _placement_turn(circles[i]) == _placement_turn(circles[k]))
    moved = {c.waypoint_index for c in circles if c.rule == 'same_turn_coincident'}
    return replace(result, coincident_waypoints=tuple(sorted(moved)), same_turn_transitions=transitions,
                   elapsed_ms=placements.elapsed_ms+1000*(perf_counter()-started))


def restore_opposite_turn_overlaps(placements, baseline, *, tol=1e-9):
    """Restore both members of opposite-turn overlapping pairs to the baseline.

    Same-turn pairs are left alone. Optional aligned circles have no baseline
    counterpart and are removed if implicated. Recheck after each simultaneous
    batch: restoration/removal can expose a new neighboring conflict. Each
    candidate is restored or removed at most once. A surviving overlap between
    two already-baseline circles remains visible in the returned diagnostics;
    this procedure makes no claim that the whole circle chain is connected.
    The input placements and baseline are not mutated.
    """
    return _restore_overlap_baselines(placements, baseline, tol=tol)


def _restore_overlap_baselines(placements, baseline, *, robot=None, tol=1e-9, transition=None):
    """Monotone fallback; supplying robot also checks same-turn transitions."""
    started = perf_counter()
    circles = list(placements.circles)
    restored = set(placements.restored_waypoints)
    removed = set(placements.removed_aligned_options)
    if robot is not None and transition is None:
        transition = lambda a, b: has_safe_circle_transition(a, b, baseline, robot, tol=tol)
    while True:
        pairs, blocks = find_consecutive_circle_overlaps(circles, tol=tol)
        affected = set()
        for i, k in pairs:
            if (_placement_turn(circles[i])*_placement_turn(circles[k]) < 0
                    or (robot is not None and not transition(circles[i], circles[k]))):
                affected.update((i, k))
        updated, changed = [], False
        for i, circle in enumerate(circles):
            if i not in affected or circle.rule == 'baseline':
                updated.append(circle)
                continue
            j = circle.waypoint_index
            fillet = baseline.fillets[j]
            if fillet is None:
                if circle.side is None:
                    raise ValueError('Missing baseline fillet for a turning circle.')
                removed.add((j, circle.side))
            else:
                region = baseline.fillet_regions[j]
                vertex = baseline.polyline[j]
                if not region_contains_point(vertex, region, tol):
                    raise ValueError('Baseline fallback vertex is outside A_j.')
                updated.append(replace(circle, rule='baseline',
                    center=_frozen_array(fillet.center), vertex=_frozen_array(vertex),
                    region=_frozen_region(region), in_admissible_region=True))
                restored.add(j)
            changed = True
        if not changed:
            break
        circles = updated
    occupied = {c.waypoint_index for c in circles}
    skipped = set(placements.skipped_waypoints) | {j for j, _ in removed if j not in occupied}
    sides = tuple(replace(d, status='removed_overlap_fallback' if robot is not None else 'removed_opposite_overlap')
                  if (d.waypoint_index, d.side) in removed else d
                  for d in placements.aligned_sides)
    return replace(placements, circles=tuple(circles),
                   skipped_waypoints=tuple(sorted(skipped)), aligned_sides=sides,
                   overlap_pairs=pairs, overlap_blocks=blocks,
                   restored_waypoints=tuple(sorted(restored)),
                   removed_aligned_options=tuple(sorted(removed)),
                   elapsed_ms=placements.elapsed_ms+1000*(perf_counter()-started))


def find_consecutive_circle_overlaps(circles, *, tol=1e-9):
    """Return overlapping index pairs and their connected blocks, without repairs.

    Compare adjacent occupied waypoint groups; missing circles do not interrupt
    the circle sequence. Same-waypoint alternatives are never compared with each
    other. Blocks involving alternatives describe potential conflicts, not a
    selected path. Positive disk overlap means distance < r_i + r_k - tol;
    external tangency is excluded, coincidence is included. This says nothing
    about whether the permitted arcs intersect or a directed tangent exists.

    Input must be ordered by waypoint index. With at most two alternatives per
    waypoint, detection and connected-component grouping take O(n) work.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    groups = []
    for i, circle in enumerate(circles):
        if i and circle.waypoint_index < circles[i-1].waypoint_index:
            raise ValueError('Circles must be ordered by waypoint index.')
        if not i or circle.waypoint_index != circles[i-1].waypoint_index:
            groups.append([])
        groups[-1].append(i)
    pairs, neighbors = [], {}
    for first, second in zip(groups, groups[1:]):
        for i in first:
            for k in second:
                if np.linalg.norm(circles[k].center-circles[i].center) < circles[i].radius+circles[k].radius-tol:
                    pairs.append((i, k))
                    neighbors.setdefault(i, []).append(k)
                    neighbors.setdefault(k, []).append(i)
    blocks, seen = [], set()
    for i in neighbors:
        if i in seen:
            continue
        pending, members = [i], set()
        seen.add(i)
        while pending:
            current = pending.pop()
            members.add(current)
            for other in neighbors[current]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        # Preserve sequence order without sorting each component.
        blocks.append(members)
    membership = {i: block for block, members in enumerate(blocks) for i in members}
    ordered_blocks = [[] for _ in blocks]
    for i in range(len(circles)):
        if i in membership:
            ordered_blocks[membership[i]].append(i)
    return tuple(pairs), tuple(tuple(block) for block in ordered_blocks)


@dataclass(frozen=True)
class AlignedSidePlacement:
    """A side-specific decision; absence/ambiguity differs from placement failure."""

    waypoint_index: int
    side: str
    status: str
    intersection_count: int
    corner: Optional[np.ndarray] = None
    region: Optional[dict] = None
    rejected_candidates: tuple = ()


def _place_circle_in_region(j, region, pair, r, R, tol, fallback=None, side=None):
    """Apply the shared four geometric priorities, optionally a baseline fifth."""
    from .helpers.arc_feasibility import (
        compute_basic_bounds, compute_safe_half_bounds, generate_local_center_candidates,
    )
    u, v = region['incoming'], region['outgoing']
    widths = [(b[3]-b[2]) if d[0] else (b[1]-b[0]) for b,d in zip(pair,(u,v))]
    a,b = compute_basic_bounds(*widths,R+r)
    h1,h2 = compute_safe_half_bounds(*widths,R)
    proposals = generate_local_center_candidates(a,b,h1,h2,R-r,tol)
    turn = u[0]*v[1]-u[1]*v[0]
    ex, ey = turn*np.array([-u[1],u[0]]), turn*np.array([-v[1],v[0]])
    rejected, selected = [], None
    for rule, local in proposals:
        center = region['corner']+local[0]*ex+local[1]*ey
        vertex = center+R*u-R*v
        if region_contains_point(vertex,region,tol):
            selected = rule, center, vertex
            break
        rejected.append(dict(rule=rule, center=_frozen_array(center),
                             vertex=_frozen_array(vertex), reason='outside_Aj'))
    if selected is None and fallback is not None:
        center, vertex = fallback
        selected = 'baseline', center, vertex
    if selected is None:
        return None, tuple(rejected)
    rule, center, vertex = selected
    if not region_contains_point(vertex,region,tol):
        raise ValueError(f'Baseline fallback at waypoint {j} is outside A_j.')
    return IndependentCirclePlacement(j,rule,_frozen_array(center),_frozen_array(vertex),
        R,_frozen_region(region),True,tuple(rejected),side), tuple(rejected)


def _corridor_boundary_intersections(first, second, tol):
    """Return distinct point intersections and shared boundary segments.

    Constant-size edge intersection calculation on axis-aligned rectangles.
    Shared edges are retained explicitly, not reduced to an arbitrary endpoint.
    """
    def edges(b):
        return [(0,x,b[2],b[3]) for x in b[:2]]+[(1,y,b[0],b[1]) for y in b[2:]]
    points, segments = [], []
    def add(point):
        point = np.asarray(point,dtype=float)
        if not any(np.max(abs(point-old)) <= tol for old in points):
            points.append(point)
    for axis,a,lo,hi in edges(first):
        for other,b,low,high in edges(second):
            if axis != other:
                if lo-tol <= b <= hi+tol and low-tol <= a <= high+tol:
                    point = np.empty(2)
                    point[axis],point[other] = a,b
                    add(point)
            elif abs(a-b) <= tol:
                left,right = max(lo,low),min(hi,high)
                if left > right+tol:
                    continue
                start,end = np.empty(2),np.empty(2)
                start[axis] = end[axis] = (a+b)/2
                start[1-axis],end[1-axis] = left,right
                if right-left > tol:
                    segments.append((start,end))
                    add(start); add(end)
                else:
                    add((start+end)/2)
    return np.asarray(points).reshape(-1,2), segments


def _place_aligned_sides(j, point, direction, door, pair, r, R, tol, geometry=None):
    """Find any boundary intersections on each side of the directed baseline line.

    Infer a quarter with one heading equal to the aligned passage; its missing
    quadrant must face the corner throughout the door, exactly as for baseline
    turns. This can be entry-aligned or exit-aligned depending on the step.
    Corners on the line are assigned to neither side. No artificial corners,
    baseline-circle fallback, or endpoint connection constraints are introduced.
    """
    corners, segments = (_corridor_boundary_intersections(*pair,tol) if geometry is None else
                         geometry.intersections(j, _corridor_boundary_intersections))
    signed = direction[0]*(corners[:,1]-point[1])-direction[1]*(corners[:,0]-point[0])
    concave = axis_aligned_concave_corners(*pair,tol) if geometry is None else geometry.corners(j)
    vertices = np.array([(x,y) for x in door[:2] for y in door[2:]])
    circles, diagnostics = [], []
    for side, mask in (('left', signed > tol), ('right', signed < -tol)):
        choices = corners[mask]
        sign = 1 if side == 'left' else -1
        shared_edge = any(any(sign*(direction[0]*(p[1]-point[1])-direction[1]*(p[0]-point[0])) > tol
                                  for p in segment) for segment in segments)
        if shared_edge or len(choices) != 1:
            status = ('shared_boundary' if shared_edge else
                      'no_intersection' if len(choices) == 0 else 'ambiguous_intersection')
            diagnostics.append(AlignedSidePlacement(j,side,status,len(choices)))
            continue
        corner = choices[0]
        # Identification uses ALL intersections. The existing local safety
        # condition still requires a concave corner; do not silently apply it
        # to a convex/touching intersection where its premises do not hold.
        if not any(np.max(abs(corner-c)) <= tol for c in concave):
            diagnostics.append(AlignedSidePlacement(j,side,'unsupported_intersection',1,
                                                    _frozen_array(corner)))
            continue
        orientations = []
        for transverse in _DIRECTION_VECTORS.values():
            if direction @ transverse != 0:
                continue
            for u,v in ((direction,transverse),(transverse,direction)):
                turn = u[0]*v[1]-u[1]*v[0]
                delta = corner-vertices
                if (np.all(turn*(u[0]*delta[:,1]-u[1]*delta[:,0]) > tol)
                        and np.all(turn*(v[0]*delta[:,1]-v[1]*delta[:,0]) > tol)):
                    orientations.append((u,v))
        if len(orientations) != 1:
            diagnostics.append(AlignedSidePlacement(j,side,'direction_unresolved',1,
                                                    _frozen_array(corner)))
            continue
        u,v = orientations[0]
        region = fillet_vertex_region(dict(x=door[:2],y=door[2:]),(None,None),corner,
                                      u,v,r,R,tol,pair_bounds=pair,validate_inputs=False)
        if region['empty']:
            circle, rejected = None, ()
            status = 'empty_fillet_region'
        else:
            circle, rejected = _place_circle_in_region(j,region,pair,r,R,tol,side=side)
            status = 'placed' if circle is not None else 'placement_unresolved'
        if circle is not None:
            circles.append(circle)
        diagnostics.append(AlignedSidePlacement(j,side,status,1,_frozen_array(corner),
                                                _frozen_region(region),rejected))
    return circles, diagnostics


def place_refinement_circles(baseline, robot, *, tol=1e-9):
    """Independently apply the four legacy priorities, then the baseline fallback.

    Priority: forty_five_safe_half, safe_half_shifted, forty_five_basic,
    basic_shifted, baseline. Reuse the original width-bound candidate generator;
    filter each center by its implied vertex's membership in the complete A_j,
    which also enforces finite corridor ends and tangent containment. No forward
    reachable-set constraint is imposed on independently placed circles.

    Every turning overlap with a validated fillet region is included, including
    specified boundary turns. At aligned overlaps, try one candidate per side
    with a unique boundary intersection and its own directional A_j. No baseline
    circle exists there, so a failed side stays empty. Unspecified boundaries
    remain circle-free. Work is O(n), with at most four proposals per side.
    """
    started = perf_counter()
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    r, R = _robot_radii(robot)
    if not baseline.feasible:
        return IndependentCirclePlacements('baseline_unavailable', elapsed_ms=1000*(perf_counter()-started))
    if any(f is not None and abs(f.radius-R) > tol for f in baseline.fillets):
        raise ValueError('Robot turning radius differs from the baseline.')
    if not _baseline_is_valid(baseline, r, R, tol):
        raise ValueError('Baseline is not valid for the supplied robot.')
    geometry = sequence_geometry(baseline.feasibility, r, tol)
    circles, skipped, aligned_sides = [], [], []
    for j, region in enumerate(baseline.fillet_regions):
        if region is None:
            # Both headings must be known: unspecified boundary directions
            # do not silently acquire a turning circle.
            directions = baseline.segment_directions
            entry, exit_direction = directions[j:j+2] if directions else (None,None)
            if entry is not None and entry == exit_direction:
                added, diagnostics = _place_aligned_sides(j,baseline.polyline[j],
                    _DIRECTION_VECTORS[entry],baseline.feasibility.safe_overlaps[j],
                    baseline.feasibility.corridor_bounds[j:j+2],r,R,tol,geometry)
                circles.extend(added)
                aligned_sides.extend(diagnostics)
                if not added:
                    skipped.append(j)
            else:
                skipped.append(j)
            continue
        if abs(region['radius']-(R-r)) > tol:
            raise ValueError('Robot radii differ from the baseline local regions.')
        fillet = baseline.fillets[j]
        if fillet is None or abs(fillet.radius-R) > tol:
            raise ValueError('Missing compatible baseline fallback circle.')
        circle, _ = _place_circle_in_region(j,region,baseline.feasibility.corridor_bounds[j:j+2],
            r,R,tol,fallback=(fillet.center,baseline.polyline[j]))
        circles.append(circle)
    overlap_pairs, overlap_blocks = find_consecutive_circle_overlaps(circles, tol=tol)
    return IndependentCirclePlacements('placed',tuple(circles),tuple(skipped),
                                       1000*(perf_counter()-started),tuple(aligned_sides),
                                       overlap_pairs, overlap_blocks)


@dataclass(frozen=True)
class InternalPathAnchor:
    """Fixed internal point/heading, door index and containing corridor index."""

    point: np.ndarray
    direction: np.ndarray
    waypoint_index: int
    corridor_index: int


@dataclass(frozen=True)
class FixedTurningCircle:
    """Fixed locally admissible circle and its original directed quarter."""

    waypoint_index: int
    center: np.ndarray
    radius: float
    turn: int
    region: dict
    origin: str = "baseline_turn"


@dataclass(frozen=True)
class FixedCircleConnectionProblem:
    """Frozen internal refinement domain; boundary poses play no role.

    Circle groups have increasing original door indices between the anchors;
    alternatives at one door are mutually exclusive graph nodes.
    The builder keeps boundary fillets outside this domain: start is the first
    door's outgoing tangent (or waypoint), end its last incoming tangent (or
    waypoint). Their internal baseline headings remain fixed. All interior
    turning circles are optional; all intervening safe doors remain mandatory.
    baseline_primitives/length retain the original certified internal fallback,
    independent of any supplied alternative circle centers.
    """

    start: InternalPathAnchor
    end: InternalPathAnchor
    circles: Tuple[FixedTurningCircle, ...]
    corridor_bounds: Tuple[Rectangle, ...]
    safe_overlaps: Tuple[Rectangle, ...]
    robot_radius: float
    baseline_primitives: Tuple[dict, ...] = ()
    baseline_length: Optional[float] = None


@dataclass(frozen=True)
class DirectedTangentState:
    """One admissible link between node indices (source, circles, sink)."""

    first: int
    second: int
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    departure_angle: float
    arrival_angle: float
    length: float
    crossings: Tuple[Tuple[int, float], ...]
    start_corridor: Optional[int] = None
    end_corridor: Optional[int] = None


@dataclass(frozen=True)
class ConsecutiveCircleConnection:
    """Local fixed-circle chain; legacy name also used by the skipping API.

    State endpoints index placements.circles. All candidate states are retained;
    chosen_states describes the shortest ordered chain from a first-circle
    quarter entry to a last-circle quarter exit. Global self-intersections are
    not audited here, and failure concerns only the requested fixed-circle class.
    """

    feasible: bool
    status: str
    reason: str
    groups: tuple = ()
    tangent_states: tuple = ()
    chosen_states: tuple = ()
    selected_circles: tuple = ()
    primitives: tuple = ()
    length: Optional[float] = None
    attempted_pairs: int = 0
    rejected_pairs: tuple = ()
    contact_order_blocks: tuple = ()
    reachable_circles: tuple = ()
    elapsed_ms: float = 0.
    allow_skipping: bool = False
    skipped_waypoints: tuple = ()  # Entire unselected groups, not unused alternatives.
    geometry_only: bool = False
    tangent_intersections_checked: bool = False
    search_expansions: int = 0
    safe_arcs_checked: bool = False
    tangent_containment_checked: bool = False
    ordered_overlap_crossings_checked: bool = False
    footprint_scope: str = 'none'  # 'all', 'shortcuts', or 'none'.


@dataclass(frozen=True)
class SimpleTangentChainConnection(ConsecutiveCircleConnection):
    """Local connection diagnostics; legacy half-plane selection stays available.

    Local repair stops at its first green-compatible, footprint-safe chain.
    The legacy strategy retains its original checks/proposals.
    No circle is moved; corridor order and global intersections are unchecked.
    Circle indices refer to the original placements, including aligned options.
    """

    selection_method: str = 'legacy_halfplane'
    halfplane_triples: tuple = ()  # (first, middle, last, clearance or None, bad)
    halfplane_blocks: tuple = ()  # First/last occupied group indices.
    proposed_shortcuts: tuple = ()  # First/last circle indices.
    reference_pairs_tested: int = 0
    repair_rounds: tuple = ()  # Bypassed occupied-group endpoint indices.


def _legacy_halfplane_pairs(groups, nodes, geometry, tol):
    """O(M) local proposals for a bounded number of alternatives per group.

    For each triple, use the directed tangent between the outer circles. The
    signed center distance d is positive to its right; the old predicate flags
    the middle record exactly when turn*d + radius < -tol. Contiguous flagged
    triples also propose a block-endpoint tangent, allowing multi-circle skips.
    Degenerate reference lines cannot define a half-plane and are not flagged.
    Neighboring links are always retained; green-arc search validates proposals.
    """
    from .helpers.tangent_refinement import _directed_tangent_lines
    pairs = dict.fromkeys((i, k) for a, b in zip(groups, groups[1:]) for i in a for k in b)
    triples, flags, references, shortcuts = [], [], set(), []
    for g in range(len(groups)-2):
        flagged = False
        for i in groups[g]:
            for k in groups[g+2]:
                references.add((i, k))
                tangents = geometry.tangents(nodes[i], nodes[k], _directed_tangent_lines)
                for j in groups[g+1]:
                    clearance = None
                    for tangent in tangents:
                        delta = tangent['end']-tangent['start']
                        length = float(np.linalg.norm(delta))
                        if length <= tol:
                            continue
                        normal = np.array([delta[1], -delta[0]])/length
                        distance = float((nodes[j]['center']-tangent['start']) @ normal)
                        clearance = nodes[j]['turn']*distance+nodes[j]['radius']
                        break  # Ordinary directed tangent is unique.
                    bad = clearance is not None and clearance < -tol
                    triples.append((i, j, k, clearance, bool(bad)))
                    if bad:
                        flagged = True
                        if (i, k) not in pairs:
                            pairs[i, k] = None
                            shortcuts.append((i, k))
        flags.append(flagged)
    blocks, start = [], None
    for g, flagged in enumerate((*flags, False)):
        if flagged and start is None:
            start = g
        elif not flagged and start is not None:
            blocks.append((start, g+1))
            start = None
    for first, last in blocks:
        for i in groups[first]:
            for k in groups[last]:
                if (i, k) not in pairs:
                    pairs[i, k] = None
                    shortcuts.append((i, k))
    return tuple(pairs), dict(halfplane_triples=tuple(triples), halfplane_blocks=tuple(blocks),
                              proposed_shortcuts=tuple(shortcuts), reference_pairs_tested=len(references))


def connect_simple_tangent_chain(placements, baseline, robot, *, safe_arcs=None, tol=1e-9,
                                 check_straight_containment=True, strategy='local_repair'):
    """Connect neighboring groups, repairing failures until the first chain exists.

    Neighboring occupied circle groups use green-contact, directed-arc and straight
    footprint checks, including first entry and last exit. Both aligned alternatives remain
    available. If that chain fails, try bypasses from reachable groups, increasing
    the number of skipped groups. Test one group pair at a time and stop as soon
    as a complete chain exists. Every straight link's radius-r capsule must fit in
    the original corridor union between its doors. The containment helper first
    certifies both endpoints in any one eroded corridor by convexity; only when
    that fails does it check the union boundary. Previously checked tangent states,
    footprint decisions and raw tangent geometry are reused during repair.

    An occupied group can be separated from its neighbor by aligned doors with
    no circle; this still counts as a consecutive connection. No later shortening
    pass is performed. The result is not a globally shortest chain. Overlap order
    and global intersections are unchecked. footprint_scope reports 'all' with
    containment enabled. check_straight_containment=False disables all straight
    footprint checks for explicit comparisons.

    strategy='legacy_halfplane' retains the previous half-plane proposals and
    footprint checks on all links, for comparison. Failure is unresolved.
    """
    if strategy not in ('local_repair', 'legacy_halfplane'):
        raise ValueError("strategy must be 'local_repair' or 'legacy_halfplane'.")
    options = dict(safe_arcs=safe_arcs, allow_skipping=True, tol=tol,
                   check_straight_containment=check_straight_containment,
                   check_overlap_order=False, result_type=SimpleTangentChainConnection)
    if strategy == 'legacy_halfplane':
        return _connect_green_arc_candidates(placements, baseline, robot,
                                            pair_rule=_legacy_halfplane_pairs, **options)
    started = perf_counter()
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    r, _ = _robot_radii(robot)
    geometry = sequence_geometry(baseline.feasibility, r, tol) if baseline.feasible else None
    if safe_arcs is None and baseline.feasible:
        options['safe_arcs'] = compute_circle_safe_arcs(placements, baseline, robot, tol=tol)
    groups = []
    for i, circle in enumerate(placements.circles):
        if not groups or placements.circles[groups[-1][0]].waypoint_index != circle.waypoint_index:
            groups.append([])
        groups[-1].append(i)
    pairs = [(i, k) for first, second in zip(groups, groups[1:]) for i in first for k in second]
    rounds, shortcuts, checked, tried_groups = [], [], {}, set()
    while True:
        def proposals(*_):
            return tuple(pairs), dict(selection_method='local_repair',
                                      repair_rounds=tuple(rounds), proposed_shortcuts=tuple(shortcuts))
        result = _connect_green_arc_candidates(placements, baseline, robot,
            pair_rule=proposals,
            _candidate_cache=checked, _geometry_context=geometry, **options)
        if result.feasible or result.status != 'simple_chain_unresolved':
            return replace(result, elapsed_ms=1000*(perf_counter()-started))
        reachable = set(result.reachable_circles)
        repair = next(((first, first+span) for span in range(2, len(groups))
                       for first in range(len(groups)-span)
                       if (first, first+span) not in tried_groups
                       and any(i in reachable for i in groups[first])), None)
        if repair is None:
            return replace(result, elapsed_ms=1000*(perf_counter()-started))
        tried_groups.add(repair)
        rounds.append(repair)
        first, last = repair
        additions = [(i, k) for i in groups[first] for k in groups[last]]
        pairs.extend(additions)
        shortcuts.extend(additions)


def tangent_corridor_crossings(start, end, first_door, last_door, overlaps, *, tol=1e-9):
    """Check full overlaps strictly between the endpoint doors, without sampling.

    Adjacent doors have no additional crossing constraint. A shortcut must cross
    every intermediate overlap, including aligned doors without a circle, in
    order. Green-arc membership is checked separately; no incident eroded-corridor
    endpoint constraint is imposed. Crossing points certify the middle portions
    by convexity, but full straight containment remains a separate check.
    Return ((door_index,t),...) or None.
    """
    indices = range(first_door+1, last_door)
    times = ordered_safe_overlap_crossings(start, end, (overlaps[j] for j in indices), tol=tol)
    return None if times is None else tuple(zip(indices, times))


def safe_directed_arc_sweep(intervals, arrival, departure, turn, *, angular_tol=1e-9):
    """Forward sweep iff the entire directed arc lies in one safe component.

    Input angles and intervals are global CCW radians; turn is +1 (CCW) or -1
    (CW). Membership of both endpoints alone does not permit crossing a gap.
    Returns radians in [0,2pi), or None. Equal contacts require no full lap.
    """
    if turn not in (-1, 1):
        raise ValueError('turn must be +1 or -1.')
    sweep = _forward_circle_angle(turn*arrival, turn*departure, angular_tol)
    for a, b in intervals:
        if b-a >= 2*np.pi-angular_tol:
            return sweep
        offset = (arrival-a) % (2*np.pi)
        if 2*np.pi-offset <= angular_tol:
            offset = 0.
        if offset <= b-a+angular_tol:
            if turn > 0 and offset+sweep <= b-a+angular_tol:
                return sweep
            if turn < 0 and offset-sweep >= -angular_tol:
                return sweep
    return None


def _safe_arc_predecessors(arrivals, departures, intervals, turn, R, tol):
    """Minimum incoming cost per outgoing contact using angular prefix minima.

    Each safe component is unwrapped along the turn direction. Full circles
    additionally need suffix minima for arrivals that wrap through angle zero.
    With a bounded number of components, this takes O(E log E), without building
    all incoming/outgoing graph edges. Returns outgoing_id -> (cost, incoming_id).
    """
    answers = {}
    eps, tau = tol/R, 2*np.pi
    for a, b in intervals:
        full = b-a >= tau-eps
        origin = a if turn > 0 else -b
        def position(angle):
            value = (turn*angle-origin) % tau
            return 0. if tau-value <= eps else value
        entries = sorted((position(angle), idx, cost, angle) for idx, cost, angle in arrivals
                         if full or position(angle) <= b-a+eps)
        if not entries:
            continue
        positions = [entry[0] for entry in entries]
        prefix, suffix = [], [None]*len(entries)
        best, value = None, np.inf
        for entry in entries:
            candidate = entry[2]-R*entry[0]
            if candidate < value:
                best, value = entry, candidate
            prefix.append(best)
        if full:
            best, value = None, np.inf
            for j in range(len(entries)-1, -1, -1):
                entry = entries[j]
                candidate = entry[2]-R*entry[0]
                if candidate < value:
                    best, value = entry, candidate
                suffix[j] = best
        for idx, departure in departures:
            target = position(departure)
            if not full and target > b-a+eps:
                continue
            j = bisect_right(positions, target+eps)
            choices = ([prefix[j-1]] if j else []) + ([suffix[j]] if full and j < len(entries) else [])
            for _, predecessor, cost, arrival in choices:
                delta = safe_directed_arc_sweep(((a, b),), arrival, departure, turn, angular_tol=eps)
                if delta is not None and (idx not in answers or cost+R*delta < answers[idx][0]):
                    answers[idx] = (cost+R*delta, predecessor)
    return answers


def connect_safe_arc_circles(placements, baseline, robot, *, safe_arcs=None, allow_skipping=True,
                             tol=1e-9, timings=None, check_straight_containment=True,
                             check_overlap_order=True, footprint_shortcuts_only=False):
    """Complete ordered tangent-state search within the fixed green-arc class.

    Retain both alternatives at aligned overlaps. A state stores global CCW
    departure/arrival angles; consecutive states are compatible only if the
    directed intervening arc stays in a single safe component. First entry and
    last exit retain their baseline-quarter headings. Coincident circles use
    shared contacts at safe-interval boundaries and incident tangent contacts.
    With allow_skipping=True, test every forward pair of different groups and
    minimize straight-plus-arc length through any ordered subset. First/last
    groups remain mandatory, and alternatives in one group are never connected.
    False recovers consecutive-only search. No circles are moved.

    Ordinary tangent generation tests O(M^2) pairs. Contact-order DP uses angular
    prefix/suffix minima, O(E log E) for bounded safe-component counts. Coincident
    circles add finite critical-contact states. No heuristic skip/search budget
    is used. Completeness and minimum length refer to these fixed circles, safe
    intervals and endpoint headings, up to floating-point tolerance.

    Both contacts must lie on their circles' green arcs. Shortcuts also cross
    every full overlap strictly between the endpoint doors, in order (O(n) per
    link), including aligned doors without circles. Each straight primitive's
    complete radius-r capsule must lie in the original corridor union from the
    first door's incoming corridor through the last door's outgoing corridor.
    This analytic test admits safe concave wedges outside individually eroded
    corridors. Union boundaries are built lazily and cached per door pair.
    Skipped supporting disks are not obstacles. Global self-intersections are
    not audited here.

    Optional timings is a caller-owned dictionary populated with milliseconds
    for setup, tangent generation, candidate checks, capsule checks and DAG
    search/reconstruction. Fine-grained clocks are disabled when it is omitted.
    Disabling overlap order gives the same acceptance checks as local repair.
    footprint_shortcuts_only=True recovers the earlier diagnostic policy with
    unchecked consecutive straight footprints.
    Disable straight containment too to recover
    the earlier full green-only graph. Safety-check flags and statuses report
    the chosen acceptance conditions.
    """
    return _connect_green_arc_candidates(placements, baseline, robot, safe_arcs=safe_arcs,
        allow_skipping=allow_skipping, tol=tol, timings=timings,
        check_straight_containment=check_straight_containment, check_overlap_order=check_overlap_order,
        footprint_shortcuts_only=footprint_shortcuts_only)


def _connect_green_arc_candidates(placements, baseline, robot, *, safe_arcs=None,
                                 allow_skipping=True, tol=1e-9, timings=None,
                                 check_straight_containment=True, check_overlap_order=True,
                                 pair_rule=None, result_type=ConsecutiveCircleConnection,
                                 footprint_shortcuts_only=False, _candidate_cache=None,
                                 _geometry_context=None):
    """Shared geometry, green-contact DP and reconstruction for both pair rules."""
    from .helpers.tangent_refinement import _directed_tangent_lines
    started = perf_counter()
    if timings is not None:
        timings.clear()
        timings.update(setup_ms=0., tangent_generation_ms=0., candidate_checks_ms=0.,
                       straight_containment_ms=0., graph_search_ms=0.)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    r, R = _robot_radii(robot)
    extra = {}
    def finish(feasible, status, reason, **data):
        return result_type(feasible, status, reason, safe_arcs_checked=True,
                                            ordered_overlap_crossings_checked=check_overlap_order,
                                            tangent_containment_checked=check_straight_containment and not footprint_shortcuts_only,
                                            footprint_scope=('shortcuts' if footprint_shortcuts_only else 'all')
                                                if check_straight_containment else 'none',
                                            allow_skipping=allow_skipping,
                                            elapsed_ms=1000*(perf_counter()-started), **extra, **data)
    if not baseline.feasible:
        return finish(False, 'baseline_unavailable', 'A valid baseline is required.')
    circles = placements.circles
    if not circles:
        return finish(False, 'no_circles', 'No circle groups to connect.')
    geometry = _geometry_context or sequence_geometry(baseline.feasibility, r, tol)
    overlaps = geometry.overlaps
    safe_arcs = compute_circle_safe_arcs(placements, baseline, robot, tol=tol) if safe_arcs is None else safe_arcs
    if len(safe_arcs) != len(circles):
        raise ValueError('Expected safe intervals for every positioned circle.')
    groups, nodes, starts, ends = [], [], [], []
    def angle(c, p):
        return float(np.arctan2(p[1]-c.center[1], p[0]-c.center[0]) % (2*np.pi))
    def point(c, a):
        return c.center+c.radius*np.array([np.cos(a), np.sin(a)])
    def sweep(i, a, b):
        return safe_directed_arc_sweep(safe_arcs[i].intervals, a, b,
                                      _placement_turn(circles[i]), angular_tol=tol/R)
    for i, c in enumerate(circles):
        if (safe_arcs[i].circle_index != i or safe_arcs[i].waypoint_index != c.waypoint_index
                or abs(c.radius-R) > tol or not np.all(np.isfinite(c.center))
                or (i and c.waypoint_index < circles[i-1].waypoint_index)):
            raise ValueError('Circle ordering, radius, or safe-arc mapping is inconsistent.')
        if not groups or circles[groups[-1][0]].waypoint_index != c.waypoint_index:
            groups.append([])
        groups[-1].append(i)
        nodes.append(dict(index=c.waypoint_index, center=c.center, radius=c.radius,
                          turn=_placement_turn(c), region=c.region))
        starts.append(angle(c, c.center-R*c.region['outgoing']))
        ends.append(angle(c, c.center+R*c.region['incoming']))
    group_index = {i: g for g, group in enumerate(groups) for i in group}
    raw, coincident, contacts = {}, [], []
    if timings is not None:
        generation_start = perf_counter()
        timings['setup_ms'] = 1000*(generation_start-started)
    if pair_rule is None:
        group_pairs = ((first, second) for gi, first in enumerate(groups)
                       for second in (groups[gi+1:] if allow_skipping else groups[gi+1:gi+2]))
        pairs = ((i, k) for first, second in group_pairs for i in first for k in second)
    else:
        pairs, extra = pair_rule(groups, nodes, geometry, tol)
    for i, k in pairs:
        if nodes[i]['turn'] == nodes[k]['turn'] and np.linalg.norm(circles[i].center-circles[k].center) <= tol:
            raw[i, k] = []
            coincident.append((i, k))
        else:
            raw[i, k] = geometry.tangents(nodes[i], nodes[k], _directed_tangent_lines)
    if coincident:
        for candidates in raw.values():
            for tangent in candidates:
                contacts.extend((tangent['start'], tangent['end']))
        for i, c in enumerate(circles):
            contacts.extend((point(c, starts[i]), point(c, ends[i])))
            for a, b in safe_arcs[i].intervals:
                contacts.extend((point(c, a), point(c, b)))
        contacts = [np.array(p) for p in dict.fromkeys(tuple(p) for p in contacts)]
    for i, k in coincident:
        # Ordered crossing feasibility can also change at overlap edges.
        for p in _coincident_contacts(nodes[i], overlaps if check_overlap_order else (), contacts, tol):
            if abs(np.linalg.norm(p-circles[i].center)-R) > tol:
                continue
            radial = (p-circles[i].center)/R
            raw[i, k].append(dict(start=p, end=p,
                                 direction=nodes[i]['turn']*np.array([-radial[1], radial[0]])))
    if timings is not None:
        checks_start = perf_counter()
        timings['tangent_generation_ms'] = 1000*(checks_start-generation_start)
    states, rejected = [], []
    incoming, outgoing = [[] for _ in circles], [[] for _ in circles]
    for (i, k), candidates in raw.items():
        seen, accepted, failures = set(), False, set()
        for t in candidates:
            key = (*t['start'], *t['end'])
            if key in seen:
                continue
            seen.add(key)
            cache_key = (i, k, *key)
            cached = _candidate_cache.get(cache_key) if _candidate_cache is not None else None
            if isinstance(cached, str):
                failures.add(cached)
                continue
            if isinstance(cached, DirectedTangentState):
                idx = len(states)
                states.append(cached)
                outgoing[i].append(idx); incoming[k].append(idx)
                accepted = True
                continue
            def reject(reason):
                failures.add(reason)
                if _candidate_cache is not None:
                    _candidate_cache[cache_key] = reason
            a, b = angle(circles[i], t['start']), angle(circles[k], t['end'])
            if sweep(i, a, a) is None or sweep(k, b, b) is None:
                reject('contact_outside_safe_arcs')
                continue
            length = float(np.linalg.norm(t['end']-t['start']))
            first, last = circles[i].waypoint_index, circles[k].waypoint_index
            crossings = (tangent_corridor_crossings(t['start'], t['end'], first, last,
                                                     overlaps, tol=tol) if check_overlap_order else ())
            if crossings is None:
                reject('ordered_overlap_crossing')
                continue
            if check_straight_containment and (not footprint_shortcuts_only
                                               or group_index[k] > group_index[i]+1):
                if timings is not None:
                    containment_start = perf_counter()
                contained = geometry.contains_capsule(first, last, t['start'], t['end'])
                if timings is not None:
                    timings['straight_containment_ms'] += 1000*(perf_counter()-containment_start)
                if not contained:
                    reject('straight_footprint_clearance')
                    continue
            idx = len(states)
            states.append(DirectedTangentState(i, k, _frozen_array(t['start']), _frozen_array(t['end']),
                _frozen_array(t['direction']), a, b, length, crossings))
            if _candidate_cache is not None:
                _candidate_cache[cache_key] = states[-1]
            outgoing[i].append(idx); incoming[k].append(idx)
            accepted = True
        if not accepted:
            rejected.append((i, k, ','.join(sorted(failures)) or 'no_directed_tangent'))
    if timings is not None:
        graph_start = perf_counter()
        timings['candidate_checks_ms'] = 1000*(graph_start-checks_start)-timings['straight_containment_ms']
    costs, prev = np.full(len(states), np.inf), np.full(len(states), -1, dtype=int)
    blocked = set()
    for gi, group in enumerate(groups):
        for i in group:
            arrivals = [(-1, 0., starts[i])] if gi == 0 else [
                (j, costs[j], states[j].arrival_angle) for j in incoming[i] if np.isfinite(costs[j])]
            best = _safe_arc_predecessors(arrivals,
                [(idx, states[idx].departure_angle) for idx in outgoing[i]],
                safe_arcs[i].intervals, _placement_turn(circles[i]), R, tol)
            for idx in outgoing[i]:
                if idx in best:
                    cost, predecessor = best[idx]
                    costs[idx], prev[idx] = cost+states[idx].length, predecessor
                elif arrivals:
                    blocked.add(i)
    common = dict(groups=tuple(map(tuple, groups)), tangent_states=tuple(states), attempted_pairs=len(raw),
                  rejected_pairs=tuple(rejected), contact_order_blocks=tuple(sorted(blocked)),
                  reachable_circles=tuple(i for i in range(len(circles)) if i in groups[0]
                      or any(np.isfinite(costs[j]) for j in incoming[i])))
    terminal = []
    if len(groups) == 1:
        for i in groups[0]:
            delta = sweep(i, starts[i], ends[i])
            if delta is not None:
                terminal.append((R*delta, -1, i))
    else:
        for i in groups[-1]:
            for idx in incoming[i]:
                delta = sweep(i, states[idx].arrival_angle, ends[i])
                if np.isfinite(costs[idx]) and delta is not None:
                    terminal.append((costs[idx]+R*delta, idx, i))
    if not terminal:
        if timings is not None:
            timings['graph_search_ms'] = 1000*(perf_counter()-graph_start)
        conditions = ['green arcs']
        if check_straight_containment:
            conditions.append('shortcut footprint clearance' if footprint_shortcuts_only else 'straight footprint clearance')
        if check_overlap_order:
            conditions.append('ordered full-overlap crossings')
        reason = ('No chain satisfies green arcs, straight footprint clearance and ordered full-overlap crossings.'
                  if check_straight_containment and check_overlap_order and not footprint_shortcuts_only else
                  'No chain through the proposed pairs satisfies '+', '.join(conditions)+'.')
        return finish(False, 'simple_chain_unresolved' if pair_rule else 'no_safe_arc_chain', reason, **common)
    length, idx, last = min(terminal)
    chosen = []
    while idx >= 0:
        chosen.append(idx); idx = int(prev[idx])
    chosen.reverse()
    selected = [states[chosen[0]].first]+[states[j].second for j in chosen] if chosen else [last]
    primitives = []
    for position, i in enumerate(selected):
        a = states[chosen[position-1]].arrival_angle if position else starts[i]
        b = states[chosen[position]].departure_angle if position < len(chosen) else ends[i]
        delta = sweep(i, a, b)
        c = circles[i]
        enter = (_placement_turn(c)*(a-starts[i])) % (2*np.pi)
        if delta*R > tol:
            fixed = FixedTurningCircle(c.waypoint_index, c.center, R, int(_placement_turn(c)), c.region)
            primitives.append(_quarter_primitive(fixed, enter, enter+delta))
        if position < len(chosen):
            t = states[chosen[position]]
            if t.length > tol:
                primitives.append(dict(kind='line', start=t.start, end=t.end, direction=t.direction,
                                       crossings=t.crossings, start_corridor=t.start_corridor,
                                       end_corridor=t.end_corridor))
    allowance = max(tol, 1e-12)*max(1, len(primitives))
    actual = sum(np.linalg.norm(p['end']-p['start']) if p['kind'] == 'line'
                 else R*(p['leave']-p['enter']) for p in primitives)
    valid = bool(primitives) and abs(actual-length) <= allowance
    def heading(p, end):
        if p['kind'] == 'line':
            return p['direction']
        radial = (p['end' if end else 'start']-p['center'])/R
        return p['turn']*np.array([-radial[1], radial[0]])
    if primitives:
        valid &= np.linalg.norm(primitives[0]['start']-point(circles[selected[0]], starts[selected[0]])) <= allowance
        valid &= np.linalg.norm(primitives[-1]['end']-point(circles[selected[-1]], ends[selected[-1]])) <= allowance
        valid &= all(np.linalg.norm(a['end']-b['start']) <= allowance
                     and np.linalg.norm(heading(a, True)-heading(b, False)) <= allowance/R
                     for a, b in zip(primitives, primitives[1:]))
    if timings is not None:
        timings['graph_search_ms'] = 1000*(perf_counter()-graph_start)
    status = (('simple_local_chain_connected' if extra.get('selection_method') == 'local_repair' else
               'simple_safe_arcs_connected' if check_straight_containment else
               'simple_green_arcs_connected') if pair_rule else
              'safe_arcs_connected' if check_straight_containment and check_overlap_order else 'green_arcs_connected')
    reason = ('Green arcs, straight footprint clearance and ordered full-overlap crossings checked; global intersections unaudited.'
              if check_straight_containment and check_overlap_order and not footprint_shortcuts_only else
              'Selected arcs remain green; '
              + ('shortcut footprints checked; consecutive footprints unchecked; '
                 if check_straight_containment and footprint_shortcuts_only else
                 'straight footprints checked; ' if check_straight_containment else 'straight footprints unchecked; ')
              + ('overlap order checked.' if check_overlap_order else 'overlap order unchecked.'))
    return finish(bool(valid), status if valid else 'reconstruction_validation_failed', reason,
                  chosen_states=tuple(chosen), selected_circles=tuple(selected),
                  skipped_waypoints=tuple(circles[g[0]].waypoint_index for g in groups
                                          if not any(i in selected for i in g)),
                  primitives=tuple(primitives), length=float(length), **common)


def _forward_circle_angle(arrival, departure, angular_tol):
    """First forward traversal, with coincident contacts requiring no full lap."""
    delta = (departure-arrival) % (2*np.pi)
    return 0. if min(delta, 2*np.pi-delta) <= angular_tol else float(delta)


def _nonintersecting_tangent_chain(states, groups, outgoing, R, tol, max_expansions):
    """History-aware best-first search; never merge geometrically different prefixes.

    The DAG suffix cost ignoring intersections is an admissible lower bound.
    First complete chain popped is shortest among the enumerated candidates.
    A finite expansion limit yields an unresolved status, not infeasibility.
    Only nonzero straight segments are audited; arcs are deliberately separate.
    """
    from heapq import heappush, heappop
    from itertools import count
    from .helpers.bp_tangent_chain import _conflict
    angle = lambda a, b: _forward_circle_angle(a, b, tol/R)
    suffix = np.full(len(states), np.inf)
    last = set(groups[-1])
    for group in reversed(groups):
        for i in group:
            for idx in outgoing[i]:
                state = states[idx]
                if state.second in last:
                    suffix[idx] = R*angle(state.arrival_angle, np.pi/2)
                else:
                    suffix[idx] = min((R*angle(state.arrival_angle, states[k].departure_angle)
                                       + states[k].length+suffix[k] for k in outgoing[state.second]),
                                      default=np.inf)
    heap, serial = [], count()
    for i in groups[0]:
        for idx in outgoing[i]:
            cost = R*angle(0., states[idx].departure_angle)+states[idx].length
            if np.isfinite(suffix[idx]):
                heappush(heap, (cost+suffix[idx], next(serial), cost, (idx,)))
    expansions, cache = 0, {}
    lines = [dict(kind='line', start=s.start, end=s.end) for s in states]
    while heap and expansions < max_expansions:
        estimate, _, cost, path = heappop(heap)
        expansions += 1
        current = states[path[-1]]
        if current.second in last:
            return path, float(estimate), expansions, False
        previous_lines = [j for j in path if states[j].length > tol]
        for idx in outgoing[current.second]:
            if not np.isfinite(suffix[idx]):
                continue
            safe = True
            if states[idx].length > tol:
                for j in previous_lines:
                    adjacent = j == previous_lines[-1]
                    key = (j, idx, adjacent)
                    if key not in cache:
                        cache[key] = _conflict(lines[j], lines[idx], adjacent, tol)
                    if cache[key]:
                        safe = False
                        break
            if safe:
                new_cost = cost+R*angle(current.arrival_angle, states[idx].departure_angle)+states[idx].length
                heappush(heap, (new_cost+suffix[idx], next(serial), new_cost, (*path, idx)))
    return (), None, expansions, bool(heap)


def connect_consecutive_circles(placements, baseline, robot, *, tol=1e-9):
    """Compatibility entry point: connect neighboring occupied groups only."""
    return connect_refinement_circles(placements, baseline, robot, allow_skipping=False, tol=tol)


def connect_refinement_circles(placements, baseline, robot, *, allow_skipping=True,
                               geometry_only=False, max_search_expansions=10000, tol=1e-9):
    """Shortest ordered tangent-state chain, optionally skipping internal groups.

    Enumerate both aligned alternatives independently, certify directed contacts
    and exact ordered-door crossings, then propagate minimum costs subject to
    arrival <= departure on each quarter. Skipping tests every ordered pair of
    distinct groups. First/last groups remain the domain endpoints; alternatives
    within a group are never joined. No circle movement occurs. Coincident circles use critical shared contacts including
    incident tangencies, quarter endpoints, and rectangle-edge intersections.
    Missing circle groups still contribute safe-door crossing constraints.

    With skipping disabled, at most four pairs per group boundary are tested.
    Skipping generates O(M^2) circle pairs; certifying each shortcut can inspect
    O(n) doors. DAG optimization uses sorted prefix minima (O(E log E)).
    Coincident pairs additionally enumerate critical shared contacts.
    Empty input has no circle-chain domain;
    a single occupied group returns one complete safe quarter.

    geometry_only=True disables quarter-contact, tangent-endpoint containment,
    and ordered-door filters. Arcs follow the prescribed turn through [0,2pi),
    and history-aware search rejects intersecting tangent segments (shared joins
    of consecutive segments are allowed). Its feasible flag means only geometric
    connectivity; containment and arc/arc or arc/line intersections are unaudited.
    The finite search limit bounds history expansion, not circle-pair generation.
    """
    from .helpers.tangent_refinement import _directed_tangent_lines, certified_angle
    started = perf_counter()
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    if isinstance(max_search_expansions, bool) or not isinstance(max_search_expansions, int) or max_search_expansions < 1:
        raise ValueError('max_search_expansions must be a positive integer.')
    r, R = _robot_radii(robot)
    def contact_angle(node, point, tolerance):
        if not geometry_only:
            return certified_angle(node, point, tolerance)
        radial = (point-node['center'])/node['radius']
        if abs(np.linalg.norm(point-node['center'])-node['radius']) > tolerance:
            return None
        return float(np.arctan2(radial@node['region']['incoming'],
                               radial@(-node['region']['outgoing'])) % (2*np.pi))
    def finish(feasible, status, reason, **data):
        return ConsecutiveCircleConnection(feasible, status, reason,
                                            elapsed_ms=1000*(perf_counter()-started),
                                            allow_skipping=allow_skipping, geometry_only=geometry_only,
                                            tangent_intersections_checked=geometry_only, **data)
    if not baseline.feasible:
        return finish(False, 'baseline_unavailable', 'A valid filleted baseline is required.')
    circles = placements.circles
    if not circles:
        return finish(False, 'no_circles', 'No turning circles to connect; retain the baseline.')
    bounds, doors = baseline.feasibility.corridor_bounds, baseline.feasibility.safe_overlaps
    groups, nodes = [], []
    for i, c in enumerate(circles):
        j, u, v = c.waypoint_index, c.region['incoming'], c.region['outgoing']
        if (not 0 <= j < len(doors) or (i and j < circles[i-1].waypoint_index)
                or abs(c.radius-R) > tol or abs(c.region['radius']-(R-r)) > tol
                or not np.all(np.isfinite(c.center))
                or not np.all(np.isin([u, v], [-1., 0., 1.]))
                or u@u != 1 or v@v != 1 or u@v != 0):
            raise ValueError('Expected ordered locally safe cardinal quarter circles for this robot.')
        vertex = c.center+R*u-R*v
        if (not region_contains_point(vertex, c.region, tol)
                or not _point_in_eroded_bounds(vertex, doors[j], 0., tol)
                or not _point_in_eroded_bounds(vertex-R*u, bounds[j], r, tol)
                or not _point_in_eroded_bounds(vertex+R*v, bounds[j+1], r, tol)):
            raise ValueError('Circle fails local region or tangent-point containment.')
        if not groups or j != circles[groups[-1][0]].waypoint_index:
            groups.append([])
        groups[-1].append(i)
        nodes.append(dict(index=j, center=c.center, radius=R, turn=_placement_turn(c), region=c.region))

    raw, coincident, contacts = {}, [], []
    group_pairs = ((first, second) for position, first in enumerate(groups)
                   for second in (groups[position+1:] if allow_skipping else groups[position+1:position+2]))
    for first, second in group_pairs:
        for i in first:
            for k in second:
                a, b = nodes[i], nodes[k]
                if a['turn'] == b['turn'] and np.linalg.norm(a['center']-b['center']) <= tol:
                    coincident.append((i, k))
                    raw[i, k] = []
                else:
                    raw[i, k] = _directed_tangent_lines(a, b, tol, restrict_quarters=False)
                    for tangent in raw[i, k]:
                        contacts.extend((tangent['start'], tangent['end']))
    for c in circles:
        contacts.extend((c.center-R*c.region['outgoing'], c.center+R*c.region['incoming']))
    boxes = [] if geometry_only else list(doors) + [(x+r, X-r, y+r, Y-r) for x, X, y, Y in bounds]
    for i, k in coincident:
        for point in _coincident_contacts(nodes[i], boxes, contacts, tol):
            if contact_angle(nodes[i], point, tol) is None or contact_angle(nodes[k], point, tol) is None:
                continue
            radial = (point-circles[i].center)/R
            raw[i, k].append(dict(start=point, end=point,
                                 direction=nodes[i]['turn']*np.array([-radial[1], radial[0]])))

    states, rejected = [], []
    incoming, outgoing = [[] for _ in circles], [[] for _ in circles]
    for (i, k), candidates in raw.items():
        a, b = nodes[i], nodes[k]
        seen, failures, accepted = set(), set(), 0
        for tangent in candidates:
            start, end = tangent['start'], tangent['end']
            departure, arrival = contact_angle(a, start, tol), contact_angle(b, end, tol)
            if departure is None or arrival is None:
                failures.add('contact_outside_quarter')
                continue
            if not geometry_only and (not _point_in_eroded_bounds(start, bounds[a['index']+1], r, tol)
                    or not _point_in_eroded_bounds(end, bounds[b['index']], r, tol)):
                failures.add('endpoint_containment')
                continue
            indices = () if geometry_only else range(a['index']+1, b['index'])
            crossing = ordered_safe_overlap_crossings(start, end, (doors[j] for j in indices), tol=tol)
            if crossing is None:
                failures.add('ordered_safe_overlap_crossing')
                continue
            key = tuple(np.r_[start, end, tangent['direction']])
            if key in seen:
                continue
            seen.add(key)
            idx = len(states)
            states.append(DirectedTangentState(i, k, _frozen_array(start), _frozen_array(end),
                _frozen_array(tangent['direction']), departure, arrival,
                float(np.linalg.norm(end-start)), tuple(zip(indices, crossing))))
            incoming[k].append(idx)
            outgoing[i].append(idx)
            accepted += 1
        if not accepted:
            rejected.append((i, k, ','.join(sorted(failures)) or 'no_directed_tangent'))

    costs, predecessor = np.full(len(states), np.inf), np.full(len(states), -1, dtype=int)
    order_blocks = set()
    for group_index, group in enumerate(groups):
        if geometry_only:
            break
        for i in group:
            if group_index == 0:
                for idx in outgoing[i]:
                    costs[idx] = R*states[idx].departure_angle+states[idx].length
                continue
            arrivals = sorted((states[idx].arrival_angle, idx) for idx in incoming[i]
                              if np.isfinite(costs[idx]))
            angles, best_indices = [], []
            best, value = -1, np.inf
            for angle, idx in arrivals:
                candidate = costs[idx]-R*angle
                if candidate < value:
                    best, value = idx, candidate
                angles.append(angle)
                best_indices.append(best)
            for idx in outgoing[i]:
                position = bisect_right(angles, states[idx].departure_angle+tol/R)-1
                if position < 0:
                    if arrivals:
                        order_blocks.add(i)
                    continue
                prev = best_indices[position]
                costs[idx] = costs[prev]+R*max(0., states[idx].departure_angle-states[prev].arrival_angle)+states[idx].length
                predecessor[idx] = prev
    common = dict(groups=tuple(map(tuple, groups)), tangent_states=tuple(states),
                  attempted_pairs=len(raw), rejected_pairs=tuple(rejected),
                  contact_order_blocks=tuple(sorted(order_blocks)),
                  reachable_circles=tuple(i for i in range(len(circles)) if i in groups[0]
                                          or any(np.isfinite(costs[s]) for s in incoming[i])))
    terminal = [idx for i in groups[-1] for idx in incoming[i] if np.isfinite(costs[idx])]
    chosen = []
    if geometry_only:
        reachable = set(groups[0])
        for group in groups:
            for i in group:
                if i in reachable:
                    reachable.update(states[idx].second for idx in outgoing[i])
        common['reachable_circles'] = tuple(sorted(reachable))
        if len(groups) > 1:
            chosen, length, expansions, limited = _nonintersecting_tangent_chain(
                states, groups, outgoing, R, tol, max_search_expansions)
            common['search_expansions'] = expansions
            if not chosen:
                return finish(False, 'geometry_search_unresolved' if limited else 'no_geometric_chain',
                              'Geometry search limit reached.' if limited else
                              'No nonintersecting tangent chain among the enumerated candidates.', **common)
            selected = [states[chosen[0]].first]+[states[s].second for s in chosen]
        else:
            selected, length = [groups[0][0]], R*np.pi/2
    elif len(groups) > 1 and not terminal:
        return finish(False, 'no_circle_chain' if allow_skipping else 'no_consecutive_chain',
                      'No contact-ordered chain for the requested fixed-circle links; baseline retained.', **common)
    elif terminal:
        idx = min(terminal, key=lambda s: costs[s]+R*(np.pi/2-states[s].arrival_angle))
        length = float(costs[idx]+R*(np.pi/2-states[idx].arrival_angle))
        while idx >= 0:
            chosen.append(idx)
            idx = int(predecessor[idx])
        chosen.reverse()
        selected = [states[chosen[0]].first]+[states[s].second for s in chosen]
    elif not geometry_only:
        selected, length = [groups[0][0]], R*np.pi/2
    primitives = []
    for position, i in enumerate(selected):
        enter = states[chosen[position-1]].arrival_angle if position else 0.
        leave = states[chosen[position]].departure_angle if position < len(chosen) else np.pi/2
        if geometry_only:
            leave = enter+_forward_circle_angle(enter, leave, tol/R)
        c = circles[i]
        fixed = FixedTurningCircle(c.waypoint_index, c.center, R, int(_placement_turn(c)), c.region)
        if R*(leave-enter) > tol:
            primitives.append(_quarter_primitive(fixed, enter, leave))
        if position < len(chosen):
            state = states[chosen[position]]
            if state.length > tol:
                primitives.append(dict(kind='line', start=state.start, end=state.end,
                                       direction=state.direction, crossings=state.crossings))
    # Recheck reconstructed continuity, headings, anchors and total length.
    allowance = max(tol, 1e-12)*max(1, len(primitives))
    actual = sum(np.linalg.norm(p['end']-p['start']) if p['kind'] == 'line'
                 else R*(p['leave']-p['enter']) for p in primitives)
    def heading(p, at_end):
        if p['kind'] == 'line':
            return p['direction']
        radial = (p['end' if at_end else 'start']-p['center'])/R
        return p['turn']*np.array([-radial[1], radial[0]])
    valid = bool(primitives) and abs(actual-length) <= allowance
    if primitives:
        first, last = circles[selected[0]], circles[selected[-1]]
        valid &= np.linalg.norm(primitives[0]['start']-(first.center-R*first.region['outgoing'])) <= allowance
        valid &= np.linalg.norm(primitives[-1]['end']-(last.center+R*last.region['incoming'])) <= allowance
        valid &= all(np.linalg.norm(a['end']-b['start']) <= allowance
                     and np.linalg.norm(heading(a, True)-heading(b, False)) <= allowance/R
                     for a, b in zip(primitives, primitives[1:]))
    status = 'geometry_connected_pending_containment' if geometry_only else 'connected'
    return finish(bool(valid), status if valid else 'reconstruction_validation_failed',
                  ('Directed chain with nonintersecting tangents found; containment and arc intersections unchecked.'
                   if geometry_only else 'Ordered tangent/quarter chain found; global self-intersections not audited.') if valid else
                  'Reconstructed chain failed continuity or length checks.',
                  chosen_states=tuple(chosen), selected_circles=tuple(selected),
                  skipped_waypoints=tuple(circles[group[0]].waypoint_index for group in groups
                                          if not any(i in selected for i in group)),
                  primitives=tuple(primitives), length=length, **common)


@dataclass(frozen=True)
class FixedCircleConnectionResult:
    """Shortest local chain and a separate, optional global intersection audit.

    local_feasible refers to exhaustive tangent/quarter/ordered-door conditions.
    feasible also requires the requested global audit. Failure of that audit
    is unresolved globally: other local chains are NOT searched in this version.
    certified_locally_infeasible applies only to this fixed-placement class.
    No circle moves, global curvature optimality, or boundary connections occur.
    """

    feasible: bool
    local_feasible: bool
    status: str
    reason: str
    problem: FixedCircleConnectionProblem
    tangent_states: Tuple[DirectedTangentState, ...] = ()
    chosen_states: Tuple[int, ...] = ()
    primitives: Tuple[dict, ...] = ()
    length: Optional[float] = None
    retained_waypoints: Tuple[int, ...] = ()
    skipped_waypoints: Tuple[int, ...] = ()
    reachable_waypoints: Tuple[int, ...] = ()
    rejected_pairs: Tuple[Tuple[int, int, str], ...] = ()
    intersections: Tuple[Tuple[int, int], ...] = ()
    global_audit_performed: bool = False
    certified_locally_infeasible: bool = False
    exhaustive: bool = True
    attempted_pairs: int = 0
    contact_order_blocks: Tuple[int, ...] = ()
    rejected_node_pairs: Tuple[Tuple[int, int, str], ...] = ()


def _frozen_array(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _frozen_region(region):
    return MappingProxyType({key: _frozen_array(value) if isinstance(value, np.ndarray)
                             else value for key, value in region.items()})


def _quarter_primitive(circle, enter, leave):
    region, R = circle.region, circle.radius
    def point(angle):
        return circle.center + R*(-np.cos(angle)*region['outgoing']
                                  + np.sin(angle)*region['incoming'])
    return dict(kind='arc', index=circle.waypoint_index, center=circle.center,
                radius=R, turn=circle.turn, region=region, enter=enter, leave=leave,
                start=point(enter), end=point(leave))


def freeze_internal_baseline(baseline, robot, *, circle_centers=None, tol=1e-9):
    """Expose internal anchors/circles and an immutable baseline fallback.

    circle_centers optionally maps interior turning waypoint indices to proposed
    centers. Each proposal must remain in its own translated A_j. No placement
    heuristic, neighboring-circle repair, or connection search runs here.
    O(n) snapshot work. Leaving centers unspecified freezes baseline circles.
    """
    if not baseline.feasible:
        raise ValueError('A validated filleted baseline is required.')
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    r, R = _robot_radii(robot)
    points, fillets = baseline.polyline, baseline.fillets
    data = baseline.feasibility
    if (any(fillet is not None and abs(fillet.radius-R) > tol for fillet in fillets)
            or any(region is not None and abs(region['radius']-(R-r)) > tol
                   for region in baseline.fillet_regions)):
        raise ValueError('Robot radii differ from the frozen baseline.')
    # This checks the robot against the baseline instead of trusting a new radius.
    if _validate_and_build_fillets(points, data, baseline.fillet_regions, r, R, tol) is None:
        raise ValueError('Baseline is not valid for the supplied robot.')
    m = len(points)
    start = InternalPathAnchor(
        _frozen_array(fillets[0].outgoing_tangent if fillets[0] else points[0]),
        _frozen_array(_DIRECTION_VECTORS[data.passage_directions[0]]), 0, 1)
    end = InternalPathAnchor(
        _frozen_array(fillets[-1].incoming_tangent if fillets[-1] else points[-1]),
        _frozen_array(_DIRECTION_VECTORS[data.passage_directions[-1]]), m-1, m-1)
    proposals = {} if circle_centers is None else dict(circle_centers)
    baseline_circles, circles = [], []
    for j in range(1, m-1):
        fillet = fillets[j]
        if fillet is None:
            continue
        region = _frozen_region(baseline.fillet_regions[j])
        turn = 1 if fillet.signed_angle > 0 else -1
        original = FixedTurningCircle(j, _frozen_array(fillet.center), R, turn, region)
        baseline_circles.append(original)
        center = _frozen_array(proposals.pop(j, fillet.center))
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError('Circle centers must be finite XY pairs.')
        vertex = center + R*region['incoming'] - R*region['outgoing']
        if not region_contains_point(vertex, region, tol):
            raise ValueError(f'Proposed circle at waypoint {j} is outside its local region.')
        circles.append(FixedTurningCircle(j, center, R, turn, region))
    if proposals:
        raise ValueError('Center proposals must refer to interior turning waypoints.')
    originals = {circle.waypoint_index: circle for circle in baseline_circles}
    primitives = []
    length = 0.
    for j in range(m-1):
        a = fillets[j].outgoing_tangent if fillets[j] else points[j]
        b = fillets[j+1].incoming_tangent if fillets[j+1] else points[j+1]
        length += float(np.linalg.norm(b-a))
        if np.linalg.norm(b-a) > tol:
            primitives.append(dict(kind='line', start=_frozen_array(a), end=_frozen_array(b)))
        if j+1 in originals:
            primitive = _quarter_primitive(originals[j+1], 0., np.pi/2)
            primitive['start'], primitive['end'] = map(_frozen_array,
                                                       (primitive['start'], primitive['end']))
            primitives.append(primitive)
            length += R*np.pi/2
    problem = FixedCircleConnectionProblem(start, end, tuple(circles), data.corridor_bounds,
                                           data.safe_overlaps, r,
                                           tuple(MappingProxyType(p) for p in primitives), length)
    _connection_nodes(problem, tol)
    return problem


def segment_rectangle_parameter_interval(start, end, bounds, *, tol=1e-9):
    """Closed t interval with start+t*(end-start) in a rectangle, or None.

    Slab clipping is exact in real arithmetic (tol=0), including zero-length
    segments and degenerate rectangles. tol expands the box in spatial units.
    """
    start, delta = np.asarray(start, dtype=float), np.asarray(end, dtype=float)-start
    low, high = 0., 1.
    for axis, (a, b) in enumerate((bounds[:2], bounds[2:])):
        if delta[axis] == 0:
            if not a-tol <= start[axis] <= b+tol:
                return None
        else:
            x, y = (a-tol-start[axis])/delta[axis], (b+tol-start[axis])/delta[axis]
            low, high = max(low, min(x, y)), min(high, max(x, y))
    return (low, high) if low <= high else None


def ordered_safe_overlap_crossings(start, end, doors, *, tol=1e-9):
    """Exact ordered-door existence test; return earliest crossing t values.

    Doors may be full corridor overlaps or eroded safe overlaps; the caller
    supplies the rectangles appropriate to its containment check.
    For each closed slab interval [l_h,u_h], choose t_h=max(t_{h-1},l_h).
    Any admissible ordered sequence has its h-th value at least this earliest
    one, by induction. Thus t_h>u_h rejects exactly when no sequence exists.
    No sampling; equality, point doors and zero-length segments are allowed.
    """
    previous, crossings = 0., []
    for door in doors:
        interval = segment_rectangle_parameter_interval(start, end, door, tol=tol)
        if interval is None:
            return None
        previous = max(previous, interval[0])
        if previous > interval[1]:
            return None
        crossings.append(previous)
    return tuple(crossings)


def _point_in_eroded_bounds(point, bounds, r, tol):
    a, b, c, d = bounds
    return bool(a+r-tol <= point[0] <= b-r+tol
                and c+r-tol <= point[1] <= d-r+tol)


def _connection_nodes(problem, tol):
    """Validate the fixed problem and adapt it to the existing tangent algebra."""
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(problem.robot_radius) or problem.robot_radius <= 0:
        raise ValueError('Require finite tol>=0 and robot_radius>0.')
    count = len(problem.safe_overlaps)
    if len(problem.corridor_bounds) != count+1:
        raise ValueError('Expected one safe overlap per consecutive corridor pair.')
    for bounds in (*problem.corridor_bounds, *problem.safe_overlaps):
        if (np.shape(bounds) != (4,) or not np.all(np.isfinite(bounds))
                or bounds[0] > bounds[1] or bounds[2] > bounds[3]):
            raise ValueError('Require finite ordered rectangle bounds.')
    for j, door in enumerate(problem.safe_overlaps):
        for bounds in problem.corridor_bounds[j:j+2]:
            for point in ((door[0], door[2]), (door[1], door[3])):
                if not _point_in_eroded_bounds(point, bounds, problem.robot_radius, tol):
                    raise ValueError('Safe overlaps must lie in both adjacent eroded corridors.')
    nodes = []
    for anchor in (problem.start, problem.end):
        if (np.shape(anchor.point) != (2,) or np.shape(anchor.direction) != (2,)
                or not np.all(np.isfinite([anchor.point, anchor.direction]))
                or abs(np.linalg.norm(anchor.direction)-1) > max(tol, 1e-12)
                or not 0 <= anchor.waypoint_index < count
                or not 0 <= anchor.corridor_index < len(problem.corridor_bounds)
                or not _point_in_eroded_bounds(anchor.point,
                                               problem.corridor_bounds[anchor.corridor_index],
                                               problem.robot_radius, tol)):
            raise ValueError('Invalid internal anchor or anchor outside its eroded corridor.')
    if (problem.start.corridor_index != problem.start.waypoint_index+1
            or problem.end.corridor_index != problem.end.waypoint_index):
        raise ValueError('Anchors must use the outgoing start and incoming end corridors.')
    nodes.append(dict(index=problem.start.waypoint_index, center=problem.start.point,
                      radius=0., turn=0, region=None))
    last = problem.start.waypoint_index
    for circle in problem.circles:
        if not (problem.start.waypoint_index < circle.waypoint_index < problem.end.waypoint_index
                and last <= circle.waypoint_index):
            raise ValueError('Circle groups must be ordered between internal anchors.')
        u, v = circle.region['incoming'], circle.region['outgoing']
        if (not np.all(np.isin([u, v], [-1., 0., 1.]))
                or u @ u != 1 or v @ v != 1 or u @ v != 0):
            raise ValueError('Quarter directions must be perpendicular signed axis vectors.')
        vertex = circle.center + circle.radius*u - circle.radius*v
        turn = u[0]*v[1]-u[1]*v[0]
        if (np.shape(circle.center) != (2,) or not np.all(np.isfinite(circle.center))
                or not np.isfinite(circle.radius) or circle.radius <= problem.robot_radius
                or circle.turn not in (-1, 1) or turn != circle.turn
                or (circle.origin != 'aligned_option'
                    and not region_contains_point(vertex, circle.region, tol))):
            raise ValueError('Invalid circle or implied vertex outside its local region.')
        if circle.origin == 'aligned_option' and not _quarter_in_eroded_pair(
                circle, problem.corridor_bounds[circle.waypoint_index:circle.waypoint_index+2],
                problem.robot_radius, tol):
            raise ValueError('Optional aligned quarter is not safe in its adjacent corridors.')
        door = problem.safe_overlaps[circle.waypoint_index]
        if not _point_in_eroded_bounds(vertex, door, 0., tol):
            raise ValueError('A circle must have its implied vertex in its associated safe overlap.')
        # The region was built for this R; prohibit changing radius independently.
        if abs(circle.region['radius']-(circle.radius-problem.robot_radius)) > tol:
            raise ValueError('Circle radius does not match its local fillet region.')
        if problem.circles and abs(circle.radius-problem.circles[0].radius) > tol:
            raise ValueError('This connection class requires a common turning radius.')
        nodes.append(dict(index=circle.waypoint_index, center=circle.center,
                          radius=circle.radius, turn=circle.turn, region=circle.region))
        last = circle.waypoint_index
    if last >= problem.end.waypoint_index:
        raise ValueError('Internal anchors must be strictly ordered.')
    nodes.append(dict(index=problem.end.waypoint_index, center=problem.end.point,
                      radius=0., turn=0, region=None))
    return nodes


def _coincident_contacts(node, boxes, contacts, tol):
    """Finite critical contacts for the continuum of coincident equal-turn joins.

    Feasibility changes only at quarter endpoints, incident tangent contacts,
    or rectangle-edge crossings. A zero-link contact can slide to one of those
    events without losing order. Arc costs telescope on coincident same-turn
    circles, so the move preserves total length. Enumerating these events avoids
    falsely claiming completeness from only the quarter endpoint tangents.
    """
    center, R = node['center'], node['radius']
    candidates = list(contacts)
    candidates.extend((center-R*node['region']['outgoing'], center+R*node['region']['incoming']))
    for bounds in boxes:
        for axis in (0, 1):
            for value in bounds[2*axis:2*axis+2]:
                gap = value-center[axis]
                if abs(gap) > R+tol:
                    continue
                offset = np.sqrt(max(0., R*R-gap*gap))
                for sign in (-1, 1):
                    point = center.copy()
                    point[axis], point[1-axis] = value, center[1-axis]+sign*offset
                    candidates.append(point)
    return candidates


def connect_fixed_circles(problem, *, tol=1e-9, audit_global=True, max_skipped_doors=None):
    """Shortest locally admissible tangent/arc chain for a fixed placement.

    Exhaustively enumerate directed tangents for every ordered pair of nodes,
    including fixed-heading source/sink and skips. A link must have departure
    in its outgoing eroded corridor, arrival in its incoming eroded corridor,
    and ordered crossings of ALL intermediate doors (including aligned ones).
    Tangent states form an implicit DAG. At each circle sort reachable incoming
    angles and use prefix minima of cost-R*arrival_angle: all compatible state
    transitions are optimized without materializing their quadratic adjacency.
    Nodes at the same door are alternatives and are never joined to each other.
    A finite max_skipped_doors limits links by intermediate door count; it
    restricts optimality/completeness and cannot certify full-graph failure.

    Standard enumeration is O(M^2*n); DP is O(E log E) time and O(E) storage.
    Coincident equal-turn circles add finite critical zero-link contacts. The
    optional O(P^2) global primitive audit is separate from local completeness.
    If its shortest local chain intersects, return global_audit_unresolved with
    the local optimum retained; do not claim no other simple chain exists.
    """
    from .helpers.tangent_refinement import _directed_tangent_lines, certified_angle
    from .helpers.bp_tangent_chain import _conflict

    if max_skipped_doors is not None and (
            isinstance(max_skipped_doors, bool) or not isinstance(max_skipped_doors, int)
            or max_skipped_doors < 0):
        raise ValueError('max_skipped_doors must be None or a nonnegative integer.')
    nodes = _connection_nodes(problem, tol)
    last_node = len(nodes)-1
    raw, coincident, contacts = {}, [], []
    boxes = list(problem.safe_overlaps)
    r = problem.robot_radius
    boxes.extend((a+r, b-r, c+r, d-r) for a, b, c, d in problem.corridor_bounds)
    for i in range(last_node):
        for k in range(i+1, last_node+1):
            a, b = nodes[i], nodes[k]
            if a['index'] == b['index']:
                continue  # Alternatives at one door are mutually exclusive.
            if (max_skipped_doors is not None
                    and b['index']-a['index']-1 > max_skipped_doors):
                continue
            if (a['radius'] and b['radius'] and a['turn'] == b['turn']
                    and abs(a['radius']-b['radius']) <= tol
                    and np.linalg.norm(a['center']-b['center']) <= tol):
                coincident.append((i, k))
                raw[i, k] = []
                continue
            if i == 0 and k == last_node:
                delta = problem.end.point-problem.start.point
                direction = problem.start.direction
                normal = np.array([-direction[1], direction[0]])
                raw[i, k] = ([dict(start=problem.start.point, end=problem.end.point,
                                   direction=direction, first_angle=0., second_angle=0.)]
                             if np.linalg.norm(direction-problem.end.direction) <= tol
                             and delta @ direction >= -tol and abs(delta @ normal) <= tol else [])
            else:
                raw[i, k] = _directed_tangent_lines(a, b, tol, restrict_quarters=False)
            for tangent in raw[i, k]:
                contacts.extend((tangent['start'], tangent['end']))
    # Include every quarter endpoint, not only those of the coincident pair.
    for node in nodes[1:-1]:
        contacts.extend((node['center']-node['radius']*node['region']['outgoing'],
                         node['center']+node['radius']*node['region']['incoming']))
    for i, k in coincident:
        a, b = nodes[i], nodes[k]
        for point in _coincident_contacts(a, boxes, contacts, tol):
            alpha, beta = certified_angle(a, point, tol), certified_angle(b, point, tol)
            if alpha is None or beta is None:
                continue
            radial = (point-a['center'])/a['radius']
            direction = a['turn']*np.array([-radial[1], radial[0]])
            raw[i, k].append(dict(start=point, end=point, direction=direction,
                                  first_angle=alpha, second_angle=beta))

    states, rejected, rejected_nodes = [], [], []
    incoming, outgoing = [[] for _ in nodes], [[] for _ in nodes]
    for (i, k), candidates in raw.items():
        a, b = nodes[i], nodes[k]
        valid, failures, seen = 0, set(), set()
        for tangent in candidates:
            start, end, direction = tangent['start'], tangent['end'], tangent['direction']
            if ((i == 0 and np.linalg.norm(direction-problem.start.direction) > tol)
                    or (k == last_node and np.linalg.norm(direction-problem.end.direction) > tol)):
                failures.add('anchor_heading')
                continue
            departure = certified_angle(a, start, tol)
            arrival = certified_angle(b, end, tol)
            if departure is None or arrival is None:
                failures.add('contact_outside_quarter')
                continue
            start_corridor = problem.start.corridor_index if i == 0 else a['index']+1
            end_corridor = problem.end.corridor_index if k == last_node else b['index']
            if (not _point_in_eroded_bounds(start, problem.corridor_bounds[start_corridor], r, tol)
                    or not _point_in_eroded_bounds(end, problem.corridor_bounds[end_corridor], r, tol)):
                failures.add('endpoint_containment')
                continue
            indices = range(a['index']+1, b['index'])
            crossing = ordered_safe_overlap_crossings(
                start, end, (problem.safe_overlaps[j] for j in indices), tol=tol)
            if crossing is None:
                failures.add('ordered_safe_overlap_crossing')
                continue
            key = tuple(np.r_[start, end, direction])
            if key in seen:
                continue
            seen.add(key)
            state = DirectedTangentState(i, k, _frozen_array(start), _frozen_array(end),
                                         _frozen_array(direction), departure,
                                         arrival, float(np.linalg.norm(end-start)),
                                         tuple(zip(indices, crossing)))
            idx = len(states)
            states.append(state)
            outgoing[i].append(idx)
            incoming[k].append(idx)
            valid += 1
        if not valid:
            rejected.append((a['index'], b['index'], ','.join(sorted(failures))
                             if failures else ('opposite_turn_overlap'
                             if a['radius'] and b['radius'] and a['turn'] != b['turn']
                             and np.linalg.norm(a['center']-b['center']) < a['radius']+b['radius']-tol
                             else 'no_directed_tangent')))
            rejected_nodes.append((i, k, rejected[-1][2]))

    costs = np.full(len(states), np.inf)
    predecessor = np.full(len(states), -1, dtype=int)
    order_blocks = set()
    for node_index in range(last_node):
        if node_index == 0:
            for idx in outgoing[0]:
                costs[idx] = states[idx].length
            continue
        R = nodes[node_index]['radius']
        arrivals = sorted((states[idx].arrival_angle, idx) for idx in incoming[node_index]
                          if np.isfinite(costs[idx]))
        angles, best_indices = [], []
        best, best_value = -1, np.inf
        for angle, idx in arrivals:
            value = costs[idx]-R*angle
            if value < best_value:
                best, best_value = idx, value
            angles.append(angle)
            best_indices.append(best)
        for idx in outgoing[node_index]:
            state = states[idx]
            position = bisect_right(angles, state.departure_angle+tol/R)-1
            if position < 0:
                if arrivals:
                    order_blocks.add(nodes[node_index]['index'])
                continue
            prev = best_indices[position]
            costs[idx] = costs[prev] + R*max(0., state.departure_angle-states[prev].arrival_angle) + state.length
            predecessor[idx] = prev
    reachable = tuple(sorted({nodes[k]['index'] for k in range(1, last_node)
                              if any(np.isfinite(costs[idx]) for idx in incoming[k])}))
    terminal = [idx for idx in incoming[last_node] if np.isfinite(costs[idx])]
    common = dict(problem=problem, tangent_states=tuple(states), reachable_waypoints=reachable,
                  rejected_pairs=tuple(rejected), exhaustive=max_skipped_doors is None,
                  attempted_pairs=len(raw), contact_order_blocks=tuple(sorted(order_blocks)),
                  rejected_node_pairs=tuple(rejected_nodes))
    if not terminal:
        return FixedCircleConnectionResult(
            False, False, 'no_local_chain' if max_skipped_doors is None else 'limited_search_unresolved',
            'No chain found within the requested fixed-circle connection search.',
            certified_locally_infeasible=max_skipped_doors is None, **common)
    chosen = []
    idx = min(terminal, key=lambda x: costs[x])
    length = float(costs[idx])
    while idx >= 0:
        chosen.append(idx)
        idx = int(predecessor[idx])
    chosen.reverse()
    primitives, retained = [], []
    for position, idx in enumerate(chosen):
        state = states[idx]
        if position:
            prev = states[chosen[position-1]]
            circle = problem.circles[state.first-1]
            enter, leave = prev.arrival_angle, max(prev.arrival_angle, state.departure_angle)
            retained.append(circle.waypoint_index)
            if (leave-enter)*circle.radius > tol:
                primitives.append(_quarter_primitive(circle, enter, leave))
        if state.length > tol:
            primitives.append(dict(kind='line', start=state.start, end=state.end,
                                   direction=state.direction, crossings=state.crossings))
    # Independently check the reconstructed endpoints, joins and cost. Local
    # circle admissibility and exact tangent crossings were checked above.
    reconstructed_length = sum(float(np.linalg.norm(p['end']-p['start']))
                               if p['kind'] == 'line' else p['radius']*(p['leave']-p['enter'])
                               for p in primitives)
    allowance = max(tol, 1e-12)*max(1, len(states), length)
    valid = abs(reconstructed_length-length) <= allowance
    if primitives:
        valid &= np.linalg.norm(primitives[0]['start']-problem.start.point) <= allowance
        valid &= np.linalg.norm(primitives[-1]['end']-problem.end.point) <= allowance
        valid &= all(np.linalg.norm(a['end']-b['start']) <= allowance
                     for a, b in zip(primitives, primitives[1:]))
    else:
        valid &= np.linalg.norm(problem.end.point-problem.start.point) <= allowance
    if not valid:
        return FixedCircleConnectionResult(
            False, True, 'reconstruction_validation_failed',
            'Reconstructed local optimum failed endpoint, continuity or length validation.',
            chosen_states=tuple(chosen), primitives=tuple(primitives), length=length, **common)
    intersections = []
    if audit_global:
        for i, primitive in enumerate(primitives):
            for k in range(i+1, len(primitives)):
                if _conflict(primitive, primitives[k], k == i+1, tol):
                    intersections.append((i, k))
    return FixedCircleConnectionResult(
        not intersections, True, 'global_audit_unresolved' if intersections else 'connected',
        'Shortest local chain intersects; alternatives have not been searched.' if intersections else
        'Shortest chain for the fixed-circle local construction class found.',
        chosen_states=tuple(chosen), primitives=tuple(primitives), length=length,
        retained_waypoints=tuple(retained),
        skipped_waypoints=tuple(sorted({c.waypoint_index for c in problem.circles
                                       if c.waypoint_index not in retained})),
        intersections=tuple(intersections), global_audit_performed=audit_global, **common)


def _quarter_in_eroded_pair(circle, pair, r, tol):
    """Exact arc-in-union check by rectangle-boundary events, not angle sampling.

    This sufficient footprint certificate uses the union of the two eroded
    rectangles. It can reject arcs safe only in the erosion of their union.
    Each coordinate is monotone on a cardinal quarter. Membership can change
    only at an analytically computed wall crossing; test each intervening cell.
    """
    u, v, R, center = (circle.region['incoming'], circle.region['outgoing'],
                       circle.radius, circle.center)
    boxes = [(a+r,b-r,c+r,d-r) for a,b,c,d in pair]
    angles = [0., np.pi/2]
    for box in boxes:
        for axis in (0,1):
            for wall in box[2*axis:2*axis+2]:
                coefficient = u[axis] if u[axis] else -v[axis]
                value = (wall-center[axis])/(R*coefficient)
                if 0 <= value <= 1:
                    angles.append(float(np.arcsin(value) if u[axis] else np.arccos(value)))
    angles = np.unique(angles)
    angles = np.r_[angles, (angles[:-1]+angles[1:])/2]
    points = center+R*(-np.cos(angles)[:,None]*v+np.sin(angles)[:,None]*u)
    inside = np.zeros(len(points), dtype=bool)
    for a,b,c,d in boxes:
        inside |= ((points[:,0] >= a-tol) & (points[:,0] <= b+tol)
                   & (points[:,1] >= c-tol) & (points[:,1] <= d+tol))
    return (bool(inside.all())
            and _point_in_eroded_bounds(center-R*v, pair[0], r, tol)
            and _point_in_eroded_bounds(center+R*u, pair[1], r, tol))


def _aligned_circle_options(baseline, r, R, tol):
    """At each aligned interior door try eight orientations, one midpoint each.

    These are optional geometric turns, not new constraints on the baseline.
    Clip the vertex box by both tangent containments before choosing a midpoint.
    At most one candidate per directed quarter is retained. No completeness over
    placements or multiple turns inside one corridor is claimed.
    """
    data = baseline.feasibility
    options = []
    for j in range(1,len(baseline.polyline)-1):
        if data.passage_directions[j-1] != data.passage_directions[j]:
            continue
        door, pair = data.safe_overlaps[j], data.corridor_bounds[j:j+2]
        for u in _DIRECTION_VECTORS.values():
            for v in _DIRECTION_VECTORS.values():
                if u @ v != 0:
                    continue
                low, high = np.array([door[0],door[2]]), np.array([door[1],door[3]])
                for bounds, shift in zip(pair, (R*u,-R*v)):
                    a,b,c,d = bounds
                    low = np.maximum(low,np.array([a+r,c+r])+shift)
                    high = np.minimum(high,np.array([b-r,d-r])+shift)
                if np.any(low > high):
                    continue
                point = (low+high)/2
                region = _frozen_region(dict(incoming=u, outgoing=v, radius=R-r))
                circle = FixedTurningCircle(j,_frozen_array(point-R*u+R*v),R,
                                           int(u[0]*v[1]-u[1]*v[0]),region,'aligned_option')
                if _quarter_in_eroded_pair(circle,pair,r,tol):
                    options.append(circle)
    return tuple(options)


@dataclass(frozen=True)
class InternalRefinement:
    """Preferred placement search and the selected path (possibly the baseline).

    A fallback status is not geometric infeasibility. graph retains the attempted
    connection, its rejection reasons, and whether all skips were considered.
    No automatic circle repair or endpoint-pose connection is performed.
    """
    status: str
    problem: FixedCircleConnectionProblem
    graph: FixedCircleConnectionResult
    primitives: tuple
    length: float
    used_fallback: bool
    placements: tuple
    placement_ms: float
    connection_ms: float
    aligned_options: int


def refine_internal_baseline(baseline, robot, *, add_aligned_turns=True,
                             max_skipped_doors=None, audit_global=True, tol=1e-9):
    """Earlier graph experiment; not the current placement-only pipeline.

    Independently place safe circles, then find the shortest tangent/arc chain.

    Each genuine turn moves toward its box-clipped concave corner along a ray
    inside its local admissible vertex region. No neighbor affects this choice.
    First/last placement also preserves tangency to the frozen internal anchors.
    Optional aligned-door quarters are independently certified against the union
    of eroded adjacent corridors. Every link still crosses intermediate doors.

    max_skipped_doors=None enumerates all ordered node pairs. A finite value is
    an explicit speed/completeness tradeoff; no-chain then remains unresolved.
    Path lengths are comparable because internal anchors remain fixed. Accept
    only a validated chain no longer than the baseline; otherwise retain it.
    Global self-intersection audit failure does not rule out alternative chains.
    """
    from .helpers.tangent_refinement import _corner_target

    started = perf_counter()
    original = freeze_internal_baseline(baseline,robot,tol=tol)
    preferred_circles, placements = [], []
    for circle in original.circles:
        j, region = circle.waypoint_index, circle.region
        # The outside path is frozen. Preserve a possible direct tangent to
        # its anchors when choosing the first/last preferred circle. These are
        # local linear placement bounds, independent of neighboring circles.
        placement_region = dict(region)
        low, high = region['low'].copy(), region['high'].copy()
        if circle is original.circles[0]:
            u = region['incoming']
            k = int(np.argmax(abs(u)))
            low[1-k] = high[1-k] = original.start.point[1-k]
            if u[k] > 0:
                low[k] = max(low[k], original.start.point[k]+circle.radius)
            else:
                high[k] = min(high[k], original.start.point[k]-circle.radius)
        if circle is original.circles[-1]:
            v = region['outgoing']
            k = int(np.argmax(abs(v)))
            low[1-k] = high[1-k] = original.end.point[1-k]
            if v[k] > 0:
                high[k] = min(high[k], original.end.point[k]-circle.radius)
            else:
                low[k] = max(low[k], original.end.point[k]+circle.radius)
        placement_region.update(low=low, high=high)
        vertex = _corner_target(baseline.polyline[j],placement_region,tol)
        # A numerical failure must not corrupt an otherwise valid baseline.
        accepted = region_contains_point(vertex,region,tol)
        center = (vertex-circle.radius*region['incoming']+circle.radius*region['outgoing']
                  if accepted else circle.center)
        preferred_circles.append(replace(circle, center=_frozen_array(center),
                                         origin='preferred_turn'))
        placements.append(dict(waypoint_index=j, original_center=circle.center,
                               preferred_center=_frozen_array(center),
                               rule='independent_corner_ray' if accepted else 'baseline_local_fallback'))
    preferred = replace(original, circles=tuple(preferred_circles))
    r,R = _robot_radii(robot)
    options = _aligned_circle_options(baseline,r,R,tol) if add_aligned_turns else ()
    problem = replace(preferred, circles=tuple(sorted((*preferred.circles,*options),
                                                     key=lambda c:c.waypoint_index)))
    placement_ms = 1000*(perf_counter()-started)
    started = perf_counter()
    graph = connect_fixed_circles(problem,tol=tol,audit_global=audit_global,
                                  max_skipped_doors=max_skipped_doors)
    connection_ms = 1000*(perf_counter()-started)
    accept = graph.feasible and graph.length <= original.baseline_length+tol
    return InternalRefinement(
        'refined' if accept else 'baseline_fallback', problem, graph,
        graph.primitives if accept else original.baseline_primitives,
        graph.length if accept else original.baseline_length, not accept,
        tuple(placements), placement_ms, connection_ms, len(options))
