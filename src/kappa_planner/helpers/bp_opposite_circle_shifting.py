"""Bounded local separation of opposite-turn circles using certified bp data."""
from time import perf_counter

import numpy as np

from ..geometry import Point
from .bp_circle_diagnostics import analyze_bp_circles
from .tangent_refinement import _directed_tangent_lines


def _separation_fraction(delta, change, distance, tol):
    """First t in [0,1] with |delta+t*change|=distance, starting inside."""
    a = float(change @ change)
    b = float(2*delta @ change)
    c = float(delta @ delta-distance*distance)
    if c >= 0:
        return 0.
    if a == 0 or np.linalg.norm(delta+change) < distance-tol:
        return None
    root = np.sqrt(b*b-4*a*c)
    t = -2*c/(b+root) if b >= 0 else (-b+root)/(2*a)
    return min(1., t) if t <= 1+tol/max(np.sqrt(a), tol) else None


def shift_opposite_turn_circles(report, geometry, tol=1e-8):
    """Separate consecutive opposite-turn circles, retaining unresolved flags.

    Try the neighbor-preferred radial move first. If inadmissible, interpolate
    toward the certified bp circle centers, first testing radius-sum separation.
    Also test the baseline targets: touching alone can put the tangent contact
    outside the relevant quarters, requiring a larger movement.
    A_j is convex, so this fallback stays admissible when both endpoints are
    admissible. The final positions are always checked independently. One-sided
    preferences are attempted before a two-circle fallback. Only a fixed number
    of candidates and at most four diagnostic circles are checked per pair.

    A directed tangent must touch both relevant quarters; its corridor clearance
    and compatibility with a complete tangent chain are NOT certified here.
    Rejected proposals are rolled back. No global search or recursive repair.
    """
    started = perf_counter()
    initial = report['diagnostics']
    circles = list(report['sequence'])
    regions = geometry['smoothed_check'].get('regions', [])
    spacing = geometry.get('internal_check', geometry.get('check', {})).get('spacing_pairs', [])
    statuses = {tuple(p['overlaps']): p['status'] for p in spacing}
    slots = len(report['circle_slots'])
    steps = []

    def risk(first, second):
        return first >= 0 and second < slots and statuses.get((first, second)) != 'guaranteed'

    def set_center(circle, point):
        circle.center = Point(*point)
        circle.xc, circle.yc = map(float, point)

    for i in range(len(circles)-1):
        a, b = circles[i:i+2]
        if a.turn_direction == b.turn_direction:
            continue
        ca, cb = (np.array([c.center.x, c.center.y]) for c in (a, b))
        delta = cb-ca
        distance = a.radius+b.radius
        norm = float(np.linalg.norm(delta))
        if norm >= distance-tol:
            continue
        window = dict(sequence=circles[max(0, i-1):i+3])
        before = analyze_bp_circles(window, geometry, tol=tol)
        left, right = risk(a.index-1, a.index), risk(b.index, b.index+1)
        weights = (.5, .5) if left == right else ((0., 1.) if left else (1., 0.))
        step = dict(pair=(a.index, b.index), neighbor_risk=(left, right),
                    before_centers=(ca.copy(), cb.copy()), accepted=False, attempts=[])
        steps.append(step)
        targets = []
        for c in (a, b):
            region = regions[c.bp_vertex_index]
            targets.append(geometry['polyline'][c.bp_vertex_index]
                           -c.radius*region['incoming']+c.radius*region['outgoing'])
        candidates = []
        if norm > tol:
            movement = (distance-norm)*delta/norm
            candidates.append(('radial', ca-weights[0]*movement, cb+weights[1]*movement))
        modes = [(True, True)] if left == right else [(not left, not right), (True, True)]
        for move_a, move_b in modes:
            da = targets[0]-ca if move_a else np.zeros(2)
            db = targets[1]-cb if move_b else np.zeros(2)
            fraction = _separation_fraction(delta, db-da, distance, tol)
            name = 'bp_both' if move_a and move_b else 'bp_first' if move_a else 'bp_second'
            if fraction is None:
                step['attempts'].append(dict(method=name, accepted=False,
                    reason='This movement cannot reach the required separation.'))
            else:
                candidates.append((name, ca+fraction*da, cb+fraction*db))
                if fraction < 1:
                    candidates.append((name+'_full', ca+da, cb+db))
        old_margins = {c['circle_index']: c['Aj_margin'] for c in before['circles']}
        for method, na, nb in candidates:
            set_center(a, na)
            set_center(b, nb)
            after = analyze_bp_circles(window, geometry, tol=tol)
            new_flags = {name: sorted(set(after['flags'][name])-set(before['flags'][name]))
                         for name in before['flags']}
            moved_checks = [c for c in after['circles'] if c['circle_index'] in step['pair']]
            nodes = [dict(index=c.index, center=p, radius=c.radius, turn=c.turn_direction,
                          region=regions[c.bp_vertex_index]) for c, p in ((a, na), (b, nb))]
            tangents = _directed_tangent_lines(*nodes, tol)
            valid = (np.linalg.norm(nb-na) >= distance-tol
                     and all(c['in_Aj'] for c in moved_checks)
                     and not any(new_flags.values())
                     and not any(not c['in_Aj'] and c['Aj_margin'] < old_margins[c['circle_index']]-tol
                                 for c in after['circles'])
                     and bool(tangents))
            step['attempts'].append(dict(method=method, accepted=valid, new_flags=new_flags,
                                         directed_tangent_found=bool(tangents)))
            if valid:
                step.update(accepted=True, method=method, after_centers=(na.copy(), nb.copy()),
                            displacements=(na-ca, nb-cb), center_distance=float(np.linalg.norm(nb-na)),
                            reason='Separated with admissible centers and directed quarter contacts.')
                for circle, old, new in ((a, ca, na), (b, cb, nb)):
                    if np.linalg.norm(new-old) > tol:
                        circle.shifted = True
                        circle.number_of_shifts = getattr(circle, 'number_of_shifts', 0)+1
                break
            set_center(a, ca)
            set_center(b, cb)
        if not step['accepted']:
            step['reason'] = 'No locally admissible candidate; centers unchanged and overlap retained.'
    report['diagnostics'] = analyze_bp_circles(report, geometry, tol=tol,
                                              all_pairs=initial['pair_scope'] == 'all')
    result = dict(steps=steps, accepted=sum(s['accepted'] for s in steps),
                  rejected=sum(not s['accepted'] for s in steps), initial_diagnostics=initial,
                  connections_checked=False, computation_time=perf_counter()-started)
    report['opposite_shifting'] = result
    return result
