"""Read-only local checks for the initial circles placed from a bp.

Tests the original A_j sets and full relevant pi/2 arcs analytically. No circle
moves, arc sampling, merging, or tangent-chain search occurs here. Pair results
are local diagnostics, not certification of a complete path.
"""

from itertools import combinations
from time import perf_counter

import numpy as np

from .tangent_refinement import _used_arcs_intersect


def analyze_bp_circles(placement, geometry, tol=1e-8, *, all_pairs=False):
    """Return A_j membership and consecutive-pair flags in O(n).

    all_pairs=True adds the nonconsecutive overview checks in O(n^2).

    A center o corresponds to the virtual orthogonal vertex p=o+R*u-R*v.
    `consecutive` refers to consecutive entries in the placed circle sequence;
    `neighboring_transitions` requires adjacent original bp turn vertices too.
    An isolated end-to-start contact is reported separately from an arc conflict.
    Nonconsecutive shared endpoints are not treated as permitted joins.
    """
    started = perf_counter()
    circles, arcs, pairs = [], [], []
    regions = geometry['smoothed_check'].get('regions', [])
    for circle in placement['sequence']:
        region = regions[circle.bp_vertex_index]
        u, v = region['incoming'], region['outgoing']
        center = np.array([circle.center.x, circle.center.y])
        R = circle.radius
        vertex = center + R*u - R*v
        box_margin = float(np.min(np.minimum(vertex-region['low'], region['high']-vertex)))
        local = (vertex+region['offset']) @ region['frame']
        corner_margin = (float(region['radius']-np.linalg.norm(np.maximum(local, 0)))
                         if region['radius'] >= 0 else None)
        margin = (-np.inf if region['empty'] else
                  min(box_margin, corner_margin) if corner_margin is not None else box_margin)
        endpoint_door_violation = False
        check = geometry.get('check', {})
        if check.get('endpoint_constraints'):
            door = check['doors'][circle.bp_vertex_index]
            endpoint_door_violation = any(
                vertex[k] < door[axis][0]-tol or vertex[k] > door[axis][1]+tol
                for k, axis in enumerate(('x', 'y')))
        circles.append(dict(circle_index=circle.index, bp_vertex=circle.bp_vertex_index,
                            turn=circle.turn_direction, center=center, radius=R,
                            implied_vertex=vertex, in_Aj=margin >= -tol,
                            Aj_margin=margin, box_margin=box_margin,
                            corner_margin=corner_margin,
                            endpoint_door_violation=endpoint_door_violation,
                            rule=circle.construction_rule))
        arcs.append(dict(center=center, radius=R, turn=circle.turn_direction,
                         region=region, enter=0., leave=np.pi/2,
                         start=center-R*v, end=center+R*u))
    indices = (combinations(range(len(circles)), 2) if all_pairs else
               ((i, i+1) for i in range(len(circles)-1)))
    for i, j in indices:
        first, second = circles[i], circles[j]
        R = first['radius']
        if abs(second['radius']-R) > tol:
            raise ValueError('This diagnostic expects the common bp circle radius.')
        distance = float(np.linalg.norm(second['center']-first['center']))
        relation = ('coincident' if distance <= tol else
                    'overlap' if distance < 2*R-tol else
                    'touch' if distance <= 2*R+tol else 'separate')
        # Disjoint supporting circles cannot have intersecting quarter-arcs.
        # Avoid the more detailed analytic check for this common case.
        join = relation != 'separate' and bool(np.linalg.norm(arcs[i]['end']-arcs[j]['start']) <= tol)
        conflict = relation != 'separate' and bool(_used_arcs_intersect(arcs[i], arcs[j], tol))
        consecutive = j == i+1
        # The refinement helper permits an end-to-start join; here only adjacent
        # circles can legitimately use that exception.
        conflict = conflict or (join and not consecutive)
        same_turn = first['turn'] == second['turn']
        opposite_overlap = not same_turn and relation in ('overlap', 'coincident')
        pairs.append(dict(first=first['circle_index'], second=second['circle_index'],
                          consecutive=consecutive,
                          neighboring_transitions=second['bp_vertex'] == first['bp_vertex']+1,
                          same_turn=same_turn, center_distance=distance,
                          required_separation=2*R, circle_relation=relation,
                          both_in_Aj=first['in_Aj'] and second['in_Aj'],
                          quarters_intersect=conflict or join,
                          quarter_conflict=conflict,
                          shared_join=join and not conflict,
                          opposite_turn_overlap=opposite_overlap,
                          consecutive_opposite_overlap=consecutive and opposite_overlap,
                          consecutive_same_turn_arc_conflict=consecutive and same_turn and conflict,
                          same_turn_local_ok=(same_turn and first['in_Aj']
                                              and second['in_Aj'] and not conflict)))
    return dict(circles=circles, arcs=arcs, pairs=pairs, connections_checked=False,
                pair_scope='all' if all_pairs else 'consecutive',
                flags=dict(
                    opposite_turn_overlap=[(p['first'], p['second']) for p in pairs
                                           if p['consecutive_opposite_overlap']],
                    same_turn_arc_conflict=[(p['first'], p['second']) for p in pairs
                                           if p['consecutive_same_turn_arc_conflict']],
                    outside_Aj=[c['circle_index'] for c in circles if not c['in_Aj']]),
                computation_time=perf_counter()-started)
