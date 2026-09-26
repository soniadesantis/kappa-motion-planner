"""Incremental directed tangent chain through fixed, repaired bp circles.

The first and last circles are retained. End-pose maneuvers are out of scope.
No circle is moved or deleted. Skips belong only to the returned chain.
"""
from time import perf_counter

import numpy as np

from .nominal_polyline import _bounds
from .tangent_refinement import _directed_tangent_lines, certified_angle, _used_arcs_intersect


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _segment_hits(a, b, tol):
    """Intersection witnesses, including overlap endpoints and point segments."""
    p, q = a['start'], b['start']
    u, v = a['end']-p, b['end']-q
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu <= tol:
        if lv <= tol:
            return [p] if np.linalg.norm(p-q) <= tol else []
        t = np.clip((p-q) @ v/(v @ v), 0, 1)
        return [p] if np.linalg.norm(p-q-t*v) <= tol else []
    if lv <= tol:
        return _segment_hits(b, a, tol)
    det = _cross(u, v)
    if abs(det) > tol*max(lu, lv):
        t, s = _cross(q-p, v)/det, _cross(q-p, u)/det
        return [p+t*u] if -tol/lu <= t <= 1+tol/lu and -tol/lv <= s <= 1+tol/lv else []
    if abs(_cross(q-p, u)) > tol*lu:
        return []
    ends = sorted(((q-p) @ u/(u @ u), (b['end']-p) @ u/(u @ u)))
    lo, hi = max(0., ends[0]), min(1., ends[1])
    return [p+lo*u, p+hi*u] if hi >= lo-tol/lu else []


def _segment_arc_hits(segment, arc, tol):
    p, d = segment['start'], segment['end']-segment['start']
    z = p-arc['center']
    aa = float(d @ d)
    if aa <= tol*tol:
        probes = [p]
    else:
        t = -float(z @ d)/aa
        residual = z+t*d
        h2 = arc['radius']**2-float(residual @ residual)
        if h2 < -tol*max(arc['radius'], tol):
            return []
        half = np.sqrt(max(0., h2)/aa)
        probes = [p+s*d for s in (t-half, t+half)
                  if -tol/np.sqrt(aa) <= s <= 1+tol/np.sqrt(aa)]
    hits = []
    for point in probes:
        angle = certified_angle(arc, point, tol)
        if angle is not None and arc['enter']-tol/arc['radius'] <= angle <= arc['leave']+tol/arc['radius']:
            hits.append(point)
    return hits


def _conflict(a, b, adjacent, tol):
    if a['kind'] == b['kind'] == 'arc':
        if _used_arcs_intersect(a, b, tol):
            return True
        # The shared-join exception belongs only to consecutive primitives.
        return not adjacent and np.linalg.norm(a['end']-b['start']) <= tol
    if a['kind'] == b['kind'] == 'line':
        hits = _segment_hits(a, b, tol)
    else:
        hits = _segment_arc_hits(a, b, tol) if a['kind'] == 'line' else _segment_arc_hits(b, a, tol)
    return any(not (adjacent and np.linalg.norm(p-a['end']) <= tol
                    and np.linalg.norm(p-b['start']) <= tol) for p in hits)


def _covered(segment, boxes, tol):
    """Analytic interval cover by the union of eroded axis-aligned rectangles."""
    p, d = segment['start'], segment['end']-segment['start']
    intervals = []
    for low, high in boxes:
        if np.any(low > high):
            continue
        lo, hi = 0., 1.
        for axis in range(2):
            if abs(d[axis]) <= tol:
                if p[axis] < low[axis]-tol or p[axis] > high[axis]+tol:
                    hi = -1.
                    break
            else:
                ends = sorted(((low[axis]-tol-p[axis])/d[axis],
                               (high[axis]+tol-p[axis])/d[axis]))
                lo, hi = max(lo, ends[0]), min(hi, ends[1])
        if lo <= hi:
            intervals.append((lo, hi))
    end = 0.
    for lo, hi in sorted(intervals):
        if lo > end:
            return False
        end = max(end, hi)
        if end >= 1.:
            return True
    return False


def _union_edges(boxes, tol):
    """Exposed edges of an axis-aligned rectangle union, with internal seams removed."""
    edges = []
    for low, high in boxes:
        for axis in range(2):
            along = 1-axis
            for side in (-1, 1):
                value = low[axis] if side < 0 else high[axis]
                cuts = sorted({float(low[along]), float(high[along])} |
                              {float(p[along]) for box in boxes for p in box
                               if low[along] < p[along] < high[along]})
                for a, b in zip(cuts, cuts[1:]):
                    mid = (a+b)/2
                    covered = any(l[along]-tol <= mid <= h[along]+tol
                                  and l[axis]-tol <= value <= h[axis]+tol
                                  and (h[axis] > value+tol if side > 0 else l[axis] < value-tol)
                                  for l, h in boxes)
                    if not covered and b-a > tol:
                        p, q = np.zeros(2), np.zeros(2)
                        p[axis] = q[axis] = value
                        p[along], q[along] = a, b
                        edges.append(dict(start=p, end=q))
    return edges


def _union_clear(segment, boxes, radius, tol):
    """Exact disk clearance from exposed union edges; no arc/line sampling."""
    if not _covered(segment, boxes, tol):
        return False
    def point_distance(p, edge):
        d = edge['end']-edge['start']
        t = np.clip((p-edge['start']) @ d/(d @ d), 0, 1) if d @ d else 0.
        return np.linalg.norm(p-edge['start']-t*d)
    for edge in _union_edges(boxes, tol):
        if _segment_hits(segment, edge, tol):
            distance = 0.
        else:
            distance = min(point_distance(p, b) for a, b in ((segment, edge), (edge, segment))
                           for p in (a['start'], a['end']))
        if distance < radius-tol:
            return False
    return True


def build_bp_tangent_chain(report, geometry, corridors, r, *, tol=1e-8, audit_all=True):
    """Connect fixed circles with quarter-restricted directed common tangents.

    On failure try popping departure circles from a stack, transactionally.
    Local checks include contact order, preceding tangents and used arcs.
    Fast eroded-rectangle coverage is followed, when needed, by exact disk
    clearance from exposed union edges; internal rectangle seams are ignored.
    Cached pair calculations avoid rebuilding tangents. Ordinary append work
    is local; repeated rejected skips can cost O(n^2). The optional final
    all-primitive intersection audit is O(n^2) and enabled by default.
    A failed greedy chain is not a proof of geometric infeasibility.
    """
    started = perf_counter()
    checks = {c['circle_index']: c for c in report['diagnostics']['circles']}
    regions = geometry['smoothed_check'].get('regions', [])
    nodes = [dict(index=c.index, center=np.array([c.center.x, c.center.y]), radius=c.radius,
                  turn=c.turn_direction, region=regions[c.bp_vertex_index]) for c in report['sequence']]
    bounds = geometry.get('check', {}).get('corridor_bounds')
    if bounds is None:
        bounds = [_bounds(c, tol) for c in corridors]
    boxes = [(np.array([b[0], b[2]])+r, np.array([b[1], b[3]])-r) for b in bounds]
    original_boxes = [(lo-r, hi+r) for lo, hi in boxes]
    stack, links, events, cache = [], [], [], {}
    deferred, failures = [], []
    reason = 'Circle-to-circle chain connected; endpoint poses are not connected.'
    blocked = None

    def arc(node, enter, leave):
        u, v = node['region']['incoming'], node['region']['outgoing']
        def point(angle):
            return node['center']+node['radius']*(-np.cos(angle)*v+np.sin(angle)*u)
        return dict(node, kind='arc', enter=enter, leave=leave,
                    start=point(enter), end=point(leave))

    def choices(a, b):
        key = (a['index'], b['index'])
        if key not in cache:
            cache[key] = []
            failure = 'OUTSIDE_AJ'
            if checks[a['index']]['in_Aj'] and checks[b['index']]['in_Aj']:
                directed = _directed_tangent_lines(a, b, tol)
                failure = 'CORRIDOR_CLEARANCE' if directed else 'NO_DIRECTED_QUARTER_TANGENT'
                for tangent in directed:
                    # Contacts can lie in either supporting corridor of each
                    # circle, not just the corridor shared by adjacent turns.
                    span = slice(a['index'], b['index']+2)
                    if (_covered(tangent, boxes[span], tol)
                            or _union_clear(tangent, original_boxes[span], r, tol)):
                        cache[key].append(dict(tangent, kind='line'))
            cache[key].sort(key=lambda t: (np.linalg.norm(t['end']-t['start']), t['first_angle']))
            if not cache[key]:
                failures.append(dict(pair=key, reason=failure))
        return cache[key]

    if not geometry['feasible'] or not report['feasible'] or not nodes:
        return dict(feasible=False, status='NO_CIRCLE_CHAIN', reason='No complete circle placement.',
                    tangents=[], arcs=[], retained=[], skipped=[], events=[],
                    endpoint_connections_checked=False, computation_time=perf_counter()-started)
    stack.append(nodes[0])
    for target in nodes[1:]:
        accepted = False
        for k in range(len(stack)-1, -1, -1):
            departure = stack[k]
            for tangent in choices(departure, target):
                enter = links[k-1]['second_angle'] if k else 0.
                leave = tangent['first_angle']
                if leave < enter-tol/departure['radius']:
                    continue
                used = arc(departure, enter, max(enter, leave))
                if k:
                    previous_link = links[k-1]
                    if _conflict(previous_link, tangent, True, tol):
                        continue
                    previous_arc = arc(stack[k-1], links[k-2]['second_angle'] if k > 1 else 0.,
                                       previous_link['first_angle'])
                    if _conflict(previous_arc, tangent,
                                 np.linalg.norm(previous_link['end']-previous_link['start']) <= tol
                                 and used['leave']-used['enter'] <= tol/departure['radius'], tol):
                        continue
                    if _conflict(previous_arc, used,
                                 np.linalg.norm(previous_link['end']-previous_link['start']) <= tol, tol):
                        continue
                popped = [n['index'] for n in stack[k+1:]]+deferred
                if popped:
                    events.append(dict(target=target['index'], bypassed=popped,
                                       replacement=(departure['index'], target['index'])))
                stack = stack[:k+1]+[target]
                links = links[:k]+[tangent]
                deferred = []
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            if target is not nodes[-1]:
                # The current circle may itself be unnecessary. Defer it and
                # commit its skip only if a later target can be connected.
                deferred.append(target['index'])
                continue
            blocked = target['index']
            reason = f'No admissible greedy extension to O{blocked+1}; partial chain only.'
            break
    arcs = [arc(n, links[j-1]['second_angle'] if j else 0.,
                links[j]['first_angle'] if j < len(links) else np.pi/2)
            for j, n in enumerate(stack)]
    primitives = []
    for j, used in enumerate(arcs):
        if used['leave']-used['enter'] > tol/used['radius']:
            primitives.append(used)
        if j < len(links) and np.linalg.norm(links[j]['end']-links[j]['start']) > tol:
            primitives.append(links[j])
    intersections = []
    if audit_all:
        for j, a in enumerate(primitives):
            for k in range(j+1, len(primitives)):
                if _conflict(a, primitives[k], k == j+1, tol):
                    intersections.append((j, k))
    retained = [n['index'] for n in stack]
    safe = all(checks[j]['in_Aj'] for j in retained)
    feasible = blocked is None and safe and not intersections
    if not safe:
        reason = 'A retained circle is outside A_j; chain is not certified.'
    elif intersections:
        reason = 'Final intersection audit failed; the greedy candidate is not accepted.'
    return dict(feasible=feasible, status='CONNECTED' if feasible else 'UNRESOLVED', reason=reason,
                tangents=links, arcs=arcs, primitives=primitives, retained=retained,
                skipped=[j for event in events for j in event['bypassed']], events=events,
                blocked_circle=blocked, intersections=intersections,
                unprocessed=[n['index'] for n in nodes if n['index'] > retained[-1]],
                endpoint_connections_checked=False, all_intersections_checked=audit_all,
                failed_pairs=failures,
                pair_evaluations=len(cache), computation_time=perf_counter()-started)
