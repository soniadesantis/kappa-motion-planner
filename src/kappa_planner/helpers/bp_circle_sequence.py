"""Legacy initial circle placement using a certified baseline polyline (bp).

Reuses the four-priority placement and optional forbidden-corner fallback.
Optional local shifting and internal tangent-chain passes follow placement;
no merging, endpoint maneuvers, or forward/backward refinement. Placement success does not
certify a connected path. Corridor geometry and the supplied bp are not mutated.
"""

from copy import copy
from time import perf_counter

import numpy as np

from ..corridor import CorridorWorld
from ..geometry import IntermediateCirclesSequence, Point
from .arc_feasibility import (
    build_intermediate_circle_from_geometry_result,
    compute_intermediate_circle_geometry,
    compute_vehicle_clearance_radii,
)
from .axis_aligned_int_circle_sequence import get_other_intersection_point_if_present
from .bp_circle_diagnostics import analyze_bp_circles
from .nominal_polyline import _bounds


def corridor_along_bp(corridor, direction, tol=1e-8):
    """Equivalent rectangle with its longitudinal axis following the bp."""
    direction = np.asarray(direction, dtype=float)
    if (direction.shape != (2,) or not np.all(np.isfinite(direction))
            or not np.isclose(np.linalg.norm(direction), 1, atol=tol, rtol=0)
            or np.count_nonzero(abs(direction) > tol) != 1):
        raise ValueError('The bp direction must be a signed axis unit vector.')
    xmin, xmax, ymin, ymax = _bounds(corridor, tol)
    spans = np.array([xmax-xmin, ymax-ymin])
    along = int(np.argmax(abs(direction)))
    return CorridorWorld(width=spans[1-along], height=spans[along],
                         center=[(xmin+xmax)/2, (ymin+ymax)/2],
                         tilt=float(np.arctan2(direction[1], direction[0])))


class _BpCorridorView:
    """Only the rectangle data consumed by legacy initial placement.

    Bounds and signed axis directions have already been validated by the bp.
    Corner order matches CorridorWorld; get_corners returns the cached array.
    No wall matrices, normal dictionaries, or corridor objects are rebuilt.
    """
    __slots__ = ('width', 'corners')

    def __init__(self, bounds, direction):
        xmin, xmax, ymin, ymax = bounds
        dx, dy = xmax-xmin, ymax-ymin
        self.width = dy if direction[0] else dx
        height = dx if direction[0] else dy
        center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        along = .5*height*direction
        right = .5*self.width*np.array([direction[1], -direction[0]])
        self.corners = np.array([center+along+right, center-along+right,
                                 center-along-right, center+along-right])

    def get_corners(self):
        return self.corners


def build_bp_circle_sequence(corridors, vehicle, geometry, *, radius=None, tol=1e-8,
                             diagnostic_all_pairs=False, shift_same_turn=False,
                             shift_opposite_turn=False, connect_tangents=False):
    """Place one legacy circle per turning bp vertex using its known corner.

    `geometry` is the output of build_trajectory_geometry. Original corridor
    pair indices are retained even for straight transitions and failed slots.
    Effective corridor tilts/widths follow bp segment directions, not the old
    stored tilts. Cached bp bounds and per-call corridor views avoid rebuilding
    physical corridors. Radii are computed once. Diagnostics check consecutive
    circles in O(n); diagnostic_all_pairs=True enables the O(n^2) overview audit.
    All caches are local to this call, so later circle moves cannot make them stale.

    report['diagnostics']['flags'] contains zero-based original circle-index
    pairs under 'opposite_turn_overlap' and 'same_turn_arc_conflict', plus the
    indices 'outside_Aj'. Flags describe consecutive placed circles only.
    Placement feasibility is distinct from these flags and from connectivity.
    placement_time excludes diagnostics; computation_time includes them.
    shift_same_turn=True additionally repairs aligned same-turn arc conflicts
    using the other-neighbor spacing flags; report['shifting'] logs proposals.
    shift_opposite_turn=True then tries local opposite-turn separation, logged
    under report['opposite_shifting']. Unresolved pairs retain their flags.
    connect_tangents=True constructs the fixed-circle internal tangent chain
    afterward; see report['tangent_chain']. Start/end poses remain unconnected.
    """
    started = perf_counter()
    sequence = IntermediateCirclesSequence()
    slots = [None] * (len(corridors)-1)
    report = dict(sequence=sequence, circle_slots=slots, placements=[],
                  feasible=False, connections_checked=False, status=None)

    def finish(status, reason):
        placement_seconds = perf_counter()-started
        report['diagnostics'] = analyze_bp_circles(
            report, geometry, tol=tol, all_pairs=diagnostic_all_pairs)
        report.update(status=status, reason=reason,
                      placement_time=placement_seconds,
                      computation_time=perf_counter()-started)
        if shift_same_turn and geometry['feasible']:
            from .bp_circle_shifting import shift_same_turn_circles
            shifts = shift_same_turn_circles(report, geometry, tol=tol)
            report['reason'] += (f" Same-turn shifts: {shifts['accepted']} accepted, "
                                 f"{shifts['rejected']} unresolved proposals.")
            report['computation_time'] = perf_counter()-started
        if shift_opposite_turn and geometry['feasible']:
            from .bp_opposite_circle_shifting import shift_opposite_turn_circles
            shifts = shift_opposite_turn_circles(report, geometry, tol=tol)
            report['reason'] += (f" Opposite-turn shifts: {shifts['accepted']} accepted, "
                                 f"{shifts['rejected']} unresolved proposals.")
            report['computation_time'] = perf_counter()-started
        if connect_tangents:
            from .bp_tangent_chain import build_bp_tangent_chain
            report['tangent_chain'] = build_bp_tangent_chain(
                report, geometry, corridors, vehicle.width/2, tol=tol)
            report['reason'] = report['reason'].replace(
                'Circle connections have not been checked.',
                'Full start-to-end trajectory has not been certified.')
            report['circle_connections_checked'] = True
            report['computation_time'] = perf_counter()-started
        return report

    if not geometry['feasible']:
        return finish('NO_FEASIBLE_BP', geometry['smoothed_check']['reason'])
    points = geometry['polyline']
    regions = geometry['smoothed_check']['regions']
    offset = geometry['check'].get('corridor_offset', 0)
    placement_vehicle = vehicle
    if radius is not None:
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Circle radius must be finite and positive.')
        if radius != vehicle.max_radius:
            placement_vehicle = copy(vehicle)
            placement_vehicle.max_radius = radius
    radii = compute_vehicle_clearance_radii(placement_vehicle)
    bounds = geometry['check'].get('corridor_bounds')
    views = {}
    def view(index, direction):
        key = (index, float(direction[0]), float(direction[1]))
        if key not in views:
            box = bounds[index] if bounds is not None else _bounds(corridors[index], tol)
            views[key] = _BpCorridorView(box, direction)
        return views[key]

    turns = geometry.get('turn_directions')
    for j in range(1, len(points)-1):
        pair_index = j + offset
        region = regions[j]
        if region is None:
            report['placements'].append(dict(bp_vertex=j, corridor_pair=(pair_index, pair_index+1),
                                              status='straight', rule=None, circle=None))
            continue
        u, v = region['incoming'], region['outgoing']
        tau = (int(turns[j-1]) if turns is not None
               else int(np.sign(u[0]*v[1]-u[1]*v[0])))
        corner = Point(*region['corner'])
        first = view(pair_index, u)
        second = view(pair_index+1, v)
        frame = (tau*np.array([-u[1], u[0]]), tau*np.array([-v[1], v[0]]))
        other, present = get_other_intersection_point_if_present(first, second)
        if present and not isinstance(other, Point):
            other = Point(*other)
        try:
            result = compute_intermediate_circle_geometry(
                first, second, corner, tau, placement_vehicle,
                other_intersection_point=other if present else None, tol=tol,
                _clearance_radii=radii, _transition_frame=frame)
        except ValueError as error:
            # E.g. the bp can exist for R<=r, but the legacy rule requires R>r.
            result = dict(feasible=False, reason=str(error), rule='infeasible')
        item = dict(bp_vertex=j, corridor_pair=(pair_index, pair_index+1),
                    incoming=u.copy(), outgoing=v.copy(), corner=region['corner'].copy(),
                    turn=tau, geometry=result, rule=result['rule'], circle=None,
                    status='placed' if result['feasible'] else 'placement_failed')
        if result['feasible']:
            circle = build_intermediate_circle_from_geometry_result(
                result, index=pair_index, door_point=Point(*points[j]), merged=False)
            circle.corridor_index_start = pair_index
            circle.corridor_index_end = pair_index+1
            circle.bp_vertex_index = j
            circle.construction_rule = result['rule']
            circle.geometry_result = result
            slots[pair_index] = circle
            sequence.append(circle)
            item['circle'] = circle
        report['placements'].append(item)
    failed = [item for item in report['placements'] if item['status'] == 'placement_failed']
    report['feasible'] = not failed
    return finish('PLACEMENT_INCOMPLETE' if failed else 'PLACED',
                  f'{len(sequence)} initial circles placed; {len(failed)} failed transitions. '
                  'Circle connections have not been checked.')
