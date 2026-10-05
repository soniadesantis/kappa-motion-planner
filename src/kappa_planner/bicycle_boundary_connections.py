"""Bicycle pose-to-circle connections shared by baseline and refinement.

The geometric constructions are those of the original bicycle planner: CS
or C_back CS, followed if necessary by a wall-tangent backward correction.
Boundary recovery uses only forward corridor alignment and extra separation
behind the same target circle (strategy 3 of the original recovery hierarchy).

These functions return trajectory primitive lists, or None for a failed
construction. They end at a directed tangent contact on the target circle;
the caller must connect that contact to the internal path's departure contact
through an admissible arc. build_baseline_boundary_connections completes that
arc and handles the reversed final-pose problem. No placement is modified.

For a baseline fillet at waypoint j::

    maneuvers = build_pose_to_circle_connection(
        first_corridor, second_corridor, start_pose, bicycle,
        baseline.fillets[j], corner_point=baseline.fillet_regions[j]['corner'])

The same call accepts a refinement circle directly; its region supplies the
corner. Select the corridor pair and forward parametrization for that circle.
"""

from dataclasses import dataclass
from math import cos, sin
from types import SimpleNamespace

import numpy as np

from .geometry import Point, Pose
from .helpers.collision_avoidance import (
    check_arc_collision,
)
from .helpers.pose_to_circle_bicycle import (
    _finalize_maneuver_sequence,
    compute_backward_arc,
    compute_backward_arc_optimal,
    compute_initial_turn_direction,
    compute_two_maneuvers_bicycle,
    rule_initial_backward_maneuver,
)
from .helpers.recovery_maneuvers_bicycle import (
    _compute_alignment_state,
    compute_segment_deeper_into_corridor,
)
from .helpers.trajectory_bicycle import compute_required_deeper_distance

__all__ = [
    "build_free_space_minimum_time_candidate",
    "build_pose_to_circle_connection",
    "build_forward_alignment_recovery",
    "BoundaryConnection",
    "build_baseline_boundary_connections",
]


def _boundary_circle(circle, bicycle, corner_point=None, tol=1e-9):
    """Adapt old intermediate circles, baseline fillets and refinement circles."""
    center = np.asarray(tuple(circle.center), dtype=float)
    radius = float(circle.radius)
    turn = getattr(circle, "turn_direction", getattr(circle, "turn", None))
    if turn is None and hasattr(circle, "signed_angle"):
        turn = np.sign(circle.signed_angle)
    region = getattr(circle, "region", None)
    if turn is None and region is not None:
        u, v = region['incoming'], region['outgoing']
        turn = np.sign(u[0]*v[1] - u[1]*v[0])
    if (center.shape != (2,) or not np.all(np.isfinite(center))
            or not np.isfinite(radius) or radius <= 0 or turn not in (-1, 1)):
        raise ValueError("Require a finite circle with positive radius and turn +/-1.")
    if abs(radius - bicycle.max_radius) > tol:
        raise ValueError("The original boundary construction requires the bicycle turning radius.")
    if corner_point is None:
        corner_point = getattr(circle, "corner_point", None)
    if corner_point is None and region is not None:
        corner_point = region['corner']
    if corner_point is not None:
        corner = np.asarray(tuple(corner_point), dtype=float)
        if corner.shape != (2,) or not np.all(np.isfinite(corner)):
            raise ValueError("corner_point must be a finite XY pair.")
        corner_point = Point(*corner)
    return SimpleNamespace(
        center=Point(*center), xc=center[0], yc=center[1], radius=radius,
        turn_direction=int(turn), corner_point=corner_point,
        index=getattr(circle, "index", getattr(circle, "waypoint_index", 0)),
    )


def _validate_pose(start_pose, tau0, tol):
    pose = np.asarray(tuple(start_pose), dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("start_pose must contain finite x, y and heading.")
    if tau0 not in (-1, 0, 1) or not np.isfinite(tol) or tol < 0:
        raise ValueError("Require tau0 in (-1, 0, 1) and finite tol >= 0.")
    return pose.tolist()


def build_free_space_minimum_time_candidate(start_pose, bicycle, circle, *,
                                           tau0=0, tol=1e-9):
    """Build the original analytical CS or C_back CS candidate in free space.

    'Minimum-time' refers to the original planner's candidate and backward-arc
    rule, not a global search over all Reeds-Shepp families. The circle must
    have the bicycle's nominal turning radius. Return None when this family
    has no construction. No corridor checks or boundary recovery run here.
    """
    pose = _validate_pose(start_pose, tau0, tol)
    target = _boundary_circle(circle, bicycle, tol=tol)
    if np.linalg.norm(np.asarray(pose[:2]) - tuple(target.center)) <= target.radius + tol:
        return None
    try:
        if tau0 == 0:
            tau0 = compute_initial_turn_direction(
                target.xc, target.yc, target.radius, *pose, target.turn_direction)
        backward_is_better, _, _, _ = rule_initial_backward_maneuver(pose, target, tau0)
        maneuvers = []
        if backward_is_better:
            backward_arc = compute_backward_arc_optimal(
                Pose(Point(*pose[:2]), pose[2]), tau0, target.turn_direction, target, bicycle)
            maneuvers.append(backward_arc)
            pose = list(backward_arc.end_pose)
        forward_arc, segment = compute_two_maneuvers_bicycle(
            pose, bicycle, target, target.turn_direction, tau1=tau0,
            t0=maneuvers[-1].tf if maneuvers else 0.)
    except ValueError:
        # asin/sqrt domains fail when the required geometric tangent is absent.
        return None
    if forward_arc is None or segment is None:
        return None
    maneuvers.extend((forward_arc, segment))
    _finalize_maneuver_sequence(maneuvers)
    return maneuvers


def _constrained_candidate(corridor1, corridor2, pose, bicycle, target, tau0, tol):
    """Original arc checks and one wall-tangent corrective backward maneuver."""
    candidate = build_free_space_minimum_time_candidate(
        pose, bicycle, target, tau0=tau0, tol=tol)
    if candidate is None:
        return None
    forward_arc = candidate[-2]
    colliding_wall = None
    for arc in candidate[:-1]:
        collision, wall = check_arc_collision(arc, corridor1, tol=tol)
        if collision:
            if wall == corridor1.FWD:
                return None
            colliding_wall = wall
            break
    if colliding_wall is not None:
        # Baseline fillets have no corner metadata: their caller supplies the
        # corresponding baseline.fillet_regions[j]['corner'] when needed.
        if target.corner_point is None:
            raise ValueError("Supply corner_point for wall correction of a baseline fillet.")
        corrective_arc = compute_backward_arc(
            corridor1, Pose(Point(*pose[:2]), pose[2]), bicycle,
            forward_arc.turn_direction, bicycle.max_radius,
            colliding_wall, target.corner_point, tol=tol)
        if corrective_arc is None or check_arc_collision(corrective_arc, corridor1, tol=tol)[0]:
            return None
        corrected_arc, segment = compute_two_maneuvers_bicycle(
            list(corrective_arc.end_pose), bicycle, target, target.turn_direction,
            tau1=forward_arc.turn_direction, t0=corrective_arc.tf)
        if corrected_arc is None or segment is None:
            return None
        if check_arc_collision(corrected_arc, corridor1, tol=tol)[0]:
            return None
        candidate = [corrective_arc, corrected_arc, segment]
    _finalize_maneuver_sequence(candidate)
    return candidate


def build_forward_alignment_recovery(corridor1, corridor2, start_pose, bicycle,
                                     circle, *, corner_point=None, tol=1e-9):
    """Try only strategy 3; return recovery + connection primitives, or None.

    Align to corridor1.tilt using at most two arcs. If needed, reverse straight
    along that axis until the target center is R ahead (same turns) or 2R plus
    the original safety margin ahead (opposite turns), then retry the original
    constrained connection. No corridor/circle is removed and no opposite
    alignment is attempted. Pass the effective, unshrunk corridor pair whose
    first longitudinal axis points toward the target transition.
    """
    pose = _validate_pose(start_pose, 0, tol)
    target = _boundary_circle(circle, bicycle, corner_point, tol)
    alignment = _compute_alignment_state(
        pose, corridor1, bicycle, preferred_direction=1, tol=tol)
    if alignment is None:
        return None
    recovery, aligned_pose, aligned_time = alignment
    recovery = list(recovery)
    try:
        tau0 = compute_initial_turn_direction(
            target.xc, target.yc, target.radius, *aligned_pose, target.turn_direction)
    except (ValueError, ZeroDivisionError):
        # Inside/on the target, first obtain conservative 2R separation.
        tau0 = -target.turn_direction
    distance = compute_required_deeper_distance(
        aligned_pose, corridor1, target, tau0, tol=tol)
    recovered_pose = list(aligned_pose)
    if distance > tol:
        segment = compute_segment_deeper_into_corridor(
            aligned_pose, corridor1, bicycle, distance, tol=tol, t0=aligned_time)
        if segment is None:
            return None
        recovery.append(segment)
        recovered_pose = list(segment.end_pose)
    connection = _constrained_candidate(
        corridor1, corridor2, recovered_pose, bicycle, target, 0, tol)
    if connection is None:
        return None
    maneuvers = recovery + connection
    _finalize_maneuver_sequence(maneuvers)
    return maneuvers


def build_pose_to_circle_connection(corridor1, corridor2, start_pose, bicycle,
                                   circle, *, tau0=0, corner_point=None,
                                   recovery=True, tol=1e-9):
    """Connect a pose to a directed circle under circular-footprint constraints.

    Accept an old IntermediateCircle, a baseline QuarterCircleFillet, a
    refinement FixedTurningCircle or IndependentCirclePlacement. For a bare
    baseline fillet, pass corner_point=baseline.fillet_regions[j]['corner'].
    Corridors must be unshrunk CorridorWorld objects; corridor1.tilt must
    represent the forward direction toward the circle (apply any stored
    corridor inversion before calling).

    Try the original constrained candidate when boundary separation permits,
    then only forward-alignment recovery. tau0 prescribes the nominal turn;
    after recovery the turn is recomputed for the new pose. Return a continuous
    primitive list or None. As in the original active construction, no
    additional straight-segment containment check runs.
    """
    pose = _validate_pose(start_pose, tau0, tol)
    target = _boundary_circle(circle, bicycle, corner_point, tol)
    separation = np.dot(
        np.asarray(pose[:2]) - tuple(target.center),
        [cos(corridor1.tilt), sin(corridor1.tilt)])
    candidate = None
    distance = np.linalg.norm(np.asarray(pose[:2]) - tuple(target.center))
    if separation <= -target.radius + tol and distance > target.radius + tol:
        turn = tau0 or compute_initial_turn_direction(
            target.xc, target.yc, target.radius, *pose, target.turn_direction)
        if turn == target.turn_direction or distance > 2*target.radius + tol:
            candidate = _constrained_candidate(
                corridor1, corridor2, pose, bicycle, target, turn, tol)
    if candidate is not None or not recovery:
        return candidate
    return build_forward_alignment_recovery(
        corridor1, corridor2, pose, bicycle, target, tol=tol)


@dataclass(frozen=True)
class BoundaryConnection:
    """One optional physical boundary connection, independent of baseline success."""

    status: str
    maneuvers: tuple = ()

    @property
    def connected(self):
        return self.status == 'connected'


def _directed_corridor(bounds, direction):
    """Parametrize the same unshrunk rectangle along the effective direction."""
    from .corridor import CorridorWorld
    xmin, xmax, ymin, ymax = bounds
    horizontal = direction in ('left', 'right')
    angles = dict(right=0., left=np.pi, up=np.pi/2, down=-np.pi/2)
    return CorridorWorld(ymax-ymin if horizontal else xmax-xmin,
                         xmax-xmin if horizontal else ymax-ymin,
                         [(xmin+xmax)/2, (ymin+ymax)/2], angles[direction])


def _complete_circle_connection(candidate, circle, anchor, intervals, bicycle, tol):
    """Append the directed safe circle arc to a specified internal-path contact."""
    from .refinement import safe_directed_arc_sweep
    from .trajectory import CurvilinearArcUnicycle
    if candidate is None:
        return None
    center = np.asarray(tuple(circle.center), dtype=float)
    turn = circle.turn
    last = candidate[-1]
    arrival = np.arctan2(last.yf-center[1], last.xf-center[0])
    departure = np.arctan2(anchor[1]-center[1], anchor[0]-center[0])
    sweep = safe_directed_arc_sweep(intervals, arrival, departure, turn,
                                   angular_tol=tol/circle.radius)
    if sweep is None:
        return None
    if sweep > tol/circle.radius:
        candidate.append(CurvilinearArcUnicycle(
            *center, last.xf, last.yf, last.thetaf,
            *anchor, last.thetaf+turn*sweep, circle.radius, turn,
            bicycle.v_max, turn*bicycle.v_max/circle.radius, bicycle, t0=last.tf))
    _finalize_maneuver_sequence(candidate)
    return candidate


def _build_baseline_boundary(result, bicycle, pose, *, initial, tol):
    from .helpers.primitives import invert_maneuvers
    from .refinement import circle_safe_angular_intervals, _merge_circle_intervals
    from .trajectory import LinearSegmentUnicycle
    from .vehicle import Bicycle
    if pose is None:
        return BoundaryConnection('missing_pose')
    if not result.feasible:
        return BoundaryConnection('baseline_unavailable')
    if not isinstance(bicycle, Bicycle):
        return BoundaryConnection('unsupported_vehicle')
    opposite = dict(right='left', left='right', up='down', down='up')
    bounds = result.feasibility.corridor_bounds
    geometry = result.feasibility._geometry
    j = 0 if initial else len(result.fillets)-1
    fillet = result.fillets[j]
    direction = result.initial_direction if initial else opposite[result.final_direction]
    adjacent = result.segment_directions[1] if initial else opposite[result.segment_directions[-2]]
    indices = (0, 1) if initial else (len(bounds)-1, len(bounds)-2)
    corridors = [_directed_corridor(bounds[i], d) for i, d in zip(indices, (direction, adjacent))]
    pose = np.asarray(pose, dtype=float).copy()
    if not initial:
        pose[2] += np.pi
    candidates = []
    if fillet is not None:
        turn = int(np.sign(fillet.signed_angle))*(1 if initial else -1)
        anchor = fillet.outgoing_tangent if initial else fillet.incoming_tangent
        circle = SimpleNamespace(center=fillet.center, radius=fillet.radius, turn=turn,
                                 corner_point=result.fillet_regions[j]['corner'])
        # Individually eroded rectangles are safe; the already constructed
        # baseline quarter also certifies the arc through the concave overlap.
        radial = fillet.incoming_tangent-fillet.center
        angle = np.arctan2(radial[1], radial[0])
        quarter = (angle, angle+fillet.signed_angle) if fillet.signed_angle > 0 else (angle+fillet.signed_angle, angle)
        intervals = _merge_circle_intervals((*circle_safe_angular_intervals(
            fillet.center, fillet.radius, [geometry.eroded[i] for i in indices], tol=tol), quarter))
        targets = [(circle, anchor, intervals)]
    else:
        anchor = result.polyline[j]
        heading = corridors[0].tilt
        forward = np.array([cos(heading), sin(heading)])
        delta = anchor-pose[:2]
        if (abs(np.sin(pose[2]-heading)) <= tol and np.cos(pose[2]-heading) > 0
                and abs(delta[0]*forward[1]-delta[1]*forward[0]) <= tol
                and delta@forward >= -tol):
            segment = LinearSegmentUnicycle(*pose[:2], *anchor, pose[2], bicycle.v_max, bicycle)
            candidates.append([segment])
        # For a straight endpoint, use terminal circles tangent to the anchor.
        normal = np.array([-forward[1], forward[0]])
        targets = []
        for turn in (-1, 1):
            center = anchor+turn*bicycle.max_radius*normal
            circle = SimpleNamespace(center=center, radius=bicycle.max_radius, turn=turn,
                                     corner_point=anchor)
            intervals = circle_safe_angular_intervals(
                center, circle.radius, [geometry.eroded[i] for i in indices], tol=tol)
            targets.append((circle, anchor, intervals))
    if not candidates:
        for circle, anchor, intervals in targets:
            try:
                candidate = build_pose_to_circle_connection(
                    *corridors, pose, bicycle, circle, tol=tol)
                candidate = _complete_circle_connection(candidate, circle, anchor, intervals, bicycle, tol)
            except (ValueError, ZeroDivisionError):
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return BoundaryConnection('failed')
    maneuvers = min(candidates, key=lambda pieces: pieces[-1].tf)
    if not initial:
        maneuvers = invert_maneuvers(maneuvers)
        _finalize_maneuver_sequence(maneuvers)
    return BoundaryConnection('connected', tuple(maneuvers))


def build_baseline_boundary_connections(result, bicycle, initial_pose=None, final_pose=None, *, tol=1e-9):
    """Connect to the first outgoing and last incoming baseline contacts.

    Both ends reuse pose-to-circle geometry and forward-alignment recovery.
    Complete each with a safe directed circle arc. Reverse the final problem
    and its primitives. A failed end does not discard the other end or baseline.
    Successful connections replace the corresponding endpoint quarter fillet
    when assembling or plotting the path; internal baseline segments stay fixed.
    """
    return (_build_baseline_boundary(result, bicycle, initial_pose, initial=True, tol=tol),
            _build_baseline_boundary(result, bicycle, final_pose, initial=False, tol=tol))
