"""One bounded pass of coordinate-preserving same-turn circle shifts.

Mutates the placed circles, never the bp. Initial placement geometry and
canonical centers remain provenance; current centers and xc/yc are updated.
This is a local repair, not certification of a connected trajectory.
"""

from time import perf_counter

import numpy as np

from ..geometry import Point
from .bp_circle_diagnostics import analyze_bp_circles


def shift_same_turn_circles(report, geometry, tol=1e-8):
    """Resolve reversed, aligned same-turn pairs by making centers coincide.

    The other-neighbor spacing flag chooses equal or one-sided movement.
    Unknown spacing is conservatively treated as coupled. Missing external
    overlaps at sequence boundaries have no internal neighbor, but endpoint
    restrictions remain enforced through A_j. Each proposal checks at most
    four circles. Proposals introducing flags or worsening an existing A_j
    violation are recorded and rolled back. One forward pass is O(n); any
    unresolved pairs remain explicitly flagged, without iterative retries.
    """
    started = perf_counter()
    initial = report['diagnostics']
    circles = list(report['sequence'])
    regions = geometry['smoothed_check'].get('regions', [])
    spacing = geometry.get('internal_check', geometry.get('check', {})).get('spacing_pairs', [])
    statuses = {tuple(p['overlaps']): p['status'] for p in spacing}
    slot_count = len(report['circle_slots'])
    steps = []

    def risk(first, second):
        if first < 0 or second >= slot_count:
            return False
        return statuses.get((first, second)) != 'guaranteed'

    def set_center(circle, center):
        circle.center = Point(*center)
        circle.xc, circle.yc = map(float, center)

    for i in range(len(circles)-1):
        a, b = circles[i:i+2]
        if a.turn_direction != b.turn_direction:
            continue
        # Consecutive window preserves the meaning of adjacency in diagnostics.
        window = dict(sequence=circles[max(0, i-1):i+3])
        before = analyze_bp_circles(window, geometry, tol=tol)
        key = (a.index, b.index)
        if key not in before['flags']['same_turn_arc_conflict']:
            continue
        ca = np.array([a.center.x, a.center.y])
        cb = np.array([b.center.x, b.center.y])
        u = regions[a.bp_vertex_index]['outgoing']
        next_u = regions[b.bp_vertex_index]['incoming']
        delta = cb-ca
        gap = float(delta @ u)
        left, right = risk(a.index-1, a.index), risk(b.index, b.index+1)
        step = dict(pair=key, neighbor_risk=(left, right), accepted=False,
                    before_centers=(ca.copy(), cb.copy()))
        steps.append(step)
        if (b.bp_vertex_index != a.bp_vertex_index+1
                or not np.allclose(u, next_u, atol=tol, rtol=0)
                or np.linalg.norm(delta-gap*u) > tol or gap >= -tol):
            step['reason'] = 'Requires adjacent, aligned centers in reversed travel order.'
            continue
        weights = (.5, .5) if left == right else ((0., 1.) if left else (1., 0.))
        # Exact common target avoids numerical residual separation. The existing
        # transverse coordinates agree within tolerance and stay within that tolerance.
        target = ca + weights[0]*delta
        step.update(weights=weights, proposed_center=target.copy(),
                    displacements=(target-ca, target-cb))
        set_center(a, target)
        set_center(b, target)
        after = analyze_bp_circles(window, geometry, tol=tol)
        new_flags = {name: sorted(set(after['flags'][name])-set(before['flags'][name]))
                     for name in before['flags']}
        old_margins = {c['circle_index']: c['Aj_margin'] for c in before['circles']}
        worsened = [c['circle_index'] for c in after['circles']
                    if not c['in_Aj'] and c['Aj_margin'] < old_margins[c['circle_index']]-tol]
        step.update(new_flags=new_flags, worsened_Aj=worsened)
        if (any(new_flags.values()) or worsened
                or key in after['flags']['same_turn_arc_conflict']):
            set_center(a, ca)
            set_center(b, cb)
            step['reason'] = 'Proposal introduces or worsens a local constraint violation; rolled back.'
            continue
        step.update(accepted=True, reason='Quarter-arcs join at coincident centers.')
        for circle, old in ((a, ca), (b, cb)):
            if np.linalg.norm(target-old) > tol:
                circle.shifted = True
                circle.number_of_shifts = getattr(circle, 'number_of_shifts', 0)+1
    final = analyze_bp_circles(report, geometry, tol=tol,
                              all_pairs=initial['pair_scope'] == 'all')
    result = dict(steps=steps, accepted=sum(s['accepted'] for s in steps),
                  rejected=sum(not s['accepted'] for s in steps),
                  initial_diagnostics=initial, connections_checked=False,
                  computation_time=perf_counter()-started)
    report['diagnostics'] = final
    report['shifting'] = result
    return result
