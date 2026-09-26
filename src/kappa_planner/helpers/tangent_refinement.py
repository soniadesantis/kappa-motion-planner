"""Sequential geometric placement with explicit tangent/arc collision certificates.

The supplied orthogonal solution and its quarter-arc regions are the baseline.
This module never substitutes sampled collision checking for those certificates.
"""

from time import perf_counter

import numpy as np

from .fillet_safety import region_margin


def turn_blocks(points):
    segments = np.diff(np.asarray(points), axis=0)
    signs = np.sign(segments[:-1, 0]*segments[1:, 1]-segments[:-1, 1]*segments[1:, 0])
    blocks = []
    for j, sign in enumerate(signs, 1):
        if not sign:
            continue
        if blocks and blocks[-1]['turn'] == sign and blocks[-1]['vertices'][-1] == j-1:
            blocks[-1]['vertices'].append(j)
        else:
            blocks.append(dict(turn=int(sign), vertices=[j]))
    return blocks


def _rotate(vector):
    return np.array([-vector[1], vector[0]])


def circle_node(index, point, region, R):
    if region is None:
        return dict(index=index, center=np.asarray(point), radius=0., turn=0, region=None)
    u, v = region['incoming'], region['outgoing']
    return dict(index=index, center=point-R*u+R*v, radius=R,
                turn=int(np.sign(u[0]*v[1]-u[1]*v[0])), region=region)


def certified_angle(node, contact, tol=1e-8):
    """Angular position inside this specific certified quarter, not any 90° arc."""
    if node['radius'] == 0:
        return 0. if np.linalg.norm(contact-node['center']) <= tol else None
    radial = (contact-node['center'])/node['radius']
    if abs(np.linalg.norm(contact-node['center'])-node['radius']) > tol:
        return None
    region = node['region']
    angle = np.arctan2(radial @ region['incoming'], radial @ (-region['outgoing']))
    if angle < -tol/node['radius'] or angle > np.pi/2+tol/node['radius']:
        return None
    return float(np.clip(angle, 0, np.pi/2))


def _in_eroded_corridor(point, corridor, r, tol):
    normals = corridor.W[:2]
    return bool(np.all(point @ normals+corridor.W[2]
                       <= -r*np.linalg.norm(normals, axis=0)+tol))


def _directed_tangent_lines(first, second, tol, *, restrict_quarters=True):
    """Algebraic directed contacts; corridor containment is checked separately."""
    delta = second['center']-first['center']
    distance = np.linalg.norm(delta)
    signed_difference = second['turn']*second['radius']-first['turn']*first['radius']
    directions = []
    if distance <= tol:
        if abs(signed_difference) > tol:
            return []
        # Coincident equal-turn circles can meet at a common certified endpoint.
        for node in (first, second):
            if node['region'] is not None:
                directions.extend([node['region']['incoming'], node['region']['outgoing']])
        if not restrict_quarters and first['region'] is not None:
            # Coincident equal-turn circles have infinitely many common
            # tangents. Use the first circle's bp exit heading explicitly.
            directions = [first['region']['outgoing']]
        if not directions:
            directions = [np.array([1., 0.])]
    elif abs(signed_difference) <= distance+tol:
        phi = np.arctan2(delta[1], delta[0])
        offset = np.arcsin(np.clip(signed_difference/distance, -1, 1))
        directions = [np.array([np.cos(angle), np.sin(angle)])
                      for angle in (phi-offset, phi-np.pi+offset)]
    results = []
    for direction in directions:
        q0 = first['center']-first['turn']*first['radius']*_rotate(direction)
        q1 = second['center']-second['turn']*second['radius']*_rotate(direction)
        difference = q1-q0
        if difference @ direction < -tol or abs(difference @ _rotate(direction)) > tol:
            continue
        angle0, angle1 = certified_angle(first, q0, tol), certified_angle(second, q1, tol)
        if restrict_quarters and (angle0 is None or angle1 is None):
            continue
        results.append(dict(start=q0, end=q1, direction=direction,
                            first=first['index'], second=second['index'],
                            first_angle=angle0, second_angle=angle1))
    return results


def common_tangents(first, second, corridors, r, tol=1e-8, exclude_circle_overlap=False,
                    corridor_offset=0):
    """Directed quarter-arc contacts whose segment lies in one eroded corridor.

    Containment of both endpoints is sufficient by convexity. For skips this
    deliberately sufficient test requires one intervening corridor to contain
    the complete shortcut. Supporting circles may overlap by default.
    """
    if (exclude_circle_overlap and first['radius'] and second['radius']
            and np.linalg.norm(second['center']-first['center'])
                <= first['radius']+second['radius']+tol):
        return []
    results = []
    for tangent in _directed_tangent_lines(first, second, tol):
        container = next((k for k in range(first['index']+1+corridor_offset,
                                           second['index']+1+corridor_offset)
                          if _in_eroded_corridor(tangent['start'], corridors[k], r, tol)
                          and _in_eroded_corridor(tangent['end'], corridors[k], r, tol)), None)
        if container is not None:
            results.append(dict(tangent, corridor=container))
    return results


def certify_path(points, regions, corridors, r, R, *, skipped=(),
                 exclude_circle_overlap=False, tol=1e-8, corridor_offset=0, _cache=None):
    """Certify by region membership and analytical geometry, without arc sampling.

    A_j certifies each complete quarter; tangent contacts restrict traversal to
    an ordered subset. Returned arcs contain exact endpoints, not plot samples.
    The private cache is scoped to one refinement with fixed geometry/options.
    Exact coordinate keys permit reuse across rejected trials and backtracking;
    changed links also invalidate the used arcs on BOTH sides of the link.
    Global contact ordering is still checked for each assembled chain.
    """
    if (not np.all(np.isfinite(points)) or len(points) != len(regions)
            or len(points) < 2 or 0 in skipped or len(points)-1 in skipped):
        return dict(feasible=False, reason='Invalid waypoint chain or skipped endpoint.')
    indices = [j for j in range(len(points)) if j not in skipped]
    cache = {} if _cache is None else _cache
    node_cache = cache.setdefault('nodes', {})
    link_cache = cache.setdefault('links', {})
    arc_cache = cache.setdefault('arcs', {})
    intersection_cache = cache.setdefault('intersections', {})
    keys = [(j, float(points[j][0]), float(points[j][1])) for j in indices]
    nodes = []
    for j, key in zip(indices, keys):
        if key not in node_cache:
            safe = regions[j] is None or region_margin(points[j], regions[j]) >= -tol
            # Own the coordinates: callers may reuse/mutate their trial arrays.
            node_cache[key] = circle_node(j, np.array(points[j], copy=True), regions[j], R) if safe else None
        node = node_cache[key]
        if node is None:
            return dict(feasible=False, reason=f'Vertex {j} outside its certified region.')
        nodes.append(node)
    links = []
    for position, (a, b) in enumerate(zip(nodes, nodes[1:])):
        key = (keys[position], keys[position+1])
        if key not in link_cache:
            link_cache[key] = common_tangents(a, b, corridors, r, tol,
                                              exclude_circle_overlap, corridor_offset)
        links.append(link_cache[key])
    if any(not choices for choices in links):
        return dict(feasible=False, reason='No directed tangent inside the intervening eroded corridor.')

    def connect(j, incoming, chosen):
        if j == len(links):
            return chosen
        for tangent in links[j]:
            if j and not nodes[j]['radius'] and np.linalg.norm(chosen[-1]['direction']-tangent['direction']) > tol:
                continue  # A non-fillet interior point must remain straight.
            if nodes[j]['radius'] and incoming > tangent['first_angle']+tol/R:
                continue
            result = connect(j+1, tangent['second_angle'], chosen+[tangent])
            if result is not None:
                return result
        return None

    tangents = connect(0, 0., [])
    if tangents is None:
        return dict(feasible=False, reason='Tangent contacts reverse order within a certified quarter arc.')
    arcs, arc_keys = [], []
    for j, node in enumerate(nodes[1:-1], 1):
        if not node['radius']:
            continue
        # Cached links own their tangent objects for the cache's entire lifetime.
        key = (keys[j], id(tangents[j-1]), id(tangents[j]))
        if key not in arc_cache:
            enter, leave = tangents[j-1]['second_angle'], tangents[j]['first_angle']
            arc_cache[key] = dict(index=node['index'], center=node['center'], radius=R,
                                 turn=node['turn'], enter=enter, leave=max(enter, leave),
                                 start=tangents[j-1]['end'].copy(),
                                 end=tangents[j]['start'].copy(), region=node['region'])
        arcs.append(arc_cache[key])
        arc_keys.append(key)
    for j, (a, b) in enumerate(zip(arcs, arcs[1:])):
        key = (arc_keys[j], arc_keys[j+1])
        if key not in intersection_cache:
            intersection_cache[key] = _used_arcs_intersect(a, b, tol)
        if intersection_cache[key]:
            return dict(feasible=False, reason='Neighboring used arcs intersect away from their joining contact.')
    return dict(feasible=True, reason='Every segment and used arc is certified.',
                tangents=tangents, arcs=arcs, skipped=list(skipped))


def _used_arcs_intersect(first, second, tol=1e-8):
    """Circle intersections restricted to used portions; a common join is OK."""
    a, b, R = first['center'], second['center'], first['radius']
    delta, distance = b-a, np.linalg.norm(b-a)
    if distance > 2*R+tol:
        return False
    if distance <= tol:
        # Coincident circles: compare continuous angular intervals directly.
        # Express both traversals counterclockwise, then account for wraparound.
        def interval(arc):
            endpoint = arc['start'] if arc['turn'] > 0 else arc['end']
            radial = endpoint-arc['center']
            start = np.arctan2(radial[1], radial[0])
            return start, start+arc['leave']-arc['enter']
        lo, hi = interval(first)
        other_lo, other_hi = interval(second)
        for shift in (-2*np.pi, 0., 2*np.pi):
            lower, upper = max(lo, other_lo+shift), min(hi, other_hi+shift)
            if upper < lower-tol/R:
                continue
            if upper-lower > tol/R:
                return True
            point = a+R*np.array([np.cos(lower), np.sin(lower)])
            if not (np.linalg.norm(point-first['end']) <= tol
                    and np.linalg.norm(point-second['start']) <= tol):
                return True
        return False
    else:
        midpoint = (a+b)/2
        offset = np.sqrt(max(0., R*R-distance*distance/4))*_rotate(delta/distance)
        probes = [midpoint+offset, midpoint-offset]
    for point in probes:
        contained = []
        for arc in (first, second):
            node = dict(center=arc['center'], radius=R, region=arc['region'])
            angle = certified_angle(node, point, tol)
            contained.append(angle is not None and arc['enter']-tol/R <= angle <= arc['leave']+tol/R)
        if all(contained):
            common_join = (np.linalg.norm(point-first['end']) <= tol
                           and np.linalg.norm(point-second['start']) <= tol)
            if not common_join:
                return True
    return False


def _region_line_interval(origin, direction, region, tol=1e-8):
    """Exact interval of t with origin+t*direction in A_j (roundoff tolerance).

    Clip to the box, split at the two positive-part sign changes, then solve
    the quadratic disk inequality on each piece. No sampling or optimizer.
    """
    if region['empty'] or not np.any(direction):
        return None
    low, high = -np.inf, np.inf
    for k in (0, 1):
        if direction[k] == 0:
            if not region['low'][k]-tol <= origin[k] <= region['high'][k]+tol:
                return None
        else:
            a, b = (region['low'][k]-origin[k])/direction[k], (region['high'][k]-origin[k])/direction[k]
            low, high = max(low, min(a, b)), min(high, max(a, b))
    if low > high:
        return None
    if region['radius'] < 0:
        return low, high
    base = (origin+region['offset']) @ region['frame']
    slope = direction @ region['frame']
    cuts = [low, high]
    cuts.extend(-base[k]/slope[k] for k in (0, 1)
                if slope[k] != 0 and low < -base[k]/slope[k] < high)
    cuts.sort()
    intervals = []
    for left, right in zip(cuts, cuts[1:]):
        active = base+(left+right)/2*slope > 0
        v, w = base[active], slope[active]
        quadratic = float(w @ w)
        if quadratic == 0:
            if v @ v <= region['radius']**2:
                intervals.append((left, right))
            continue
        center = -float(v @ w)/quadratic
        residual = v+center*w
        allowance = region['radius']**2-float(residual @ residual)
        if allowance < -tol*max(tol, region['radius']):
            continue
        half = np.sqrt(max(0., allowance)/quadratic)
        left, right = max(left, center-half), min(right, center+half)
        if left <= right:
            intervals.append((left, right))
    if not intervals:
        return None
    return intervals[0][0], intervals[-1][1]


def _corner_target(point, region, tol):
    """Follow the ray toward the box-clipped corner until A_j's boundary."""
    direction = np.clip(region['corner'], region['low'], region['high'])-point
    interval = _region_line_interval(point, direction, region, tol)
    if interval is None:
        return point.copy()
    fraction = min(1., interval[1])
    if fraction < 0:
        return point.copy()
    return point+fraction*direction


def _neighbor_repairs(points, j, k, active, regions, R, tol):
    """Place one neighbor on an admissible offset of a reference tangent line.

    First use a tangent bypassing k, if its outer neighbor exists. Also try the
    appropriate certified endpoint tangent of j (needed for same-turn links).
    Project k's current placement onto the admissible line interval, so this
    repair moves it as little as possible along that line. Every proposal must
    still pass complete connection certification, including k's outer link.
    """
    node = circle_node(j, points[j], regions[j], R)
    position = active.index(k)
    outer_position = position-1 if k < j else position+1
    lines = []
    if 0 <= outer_position < len(active):
        h = active[outer_position]
        outer = circle_node(h, points[h], regions[h], R)
        first, second = (outer, node) if k < j else (node, outer)
        lines.extend((t['start'], t['direction'])
                     for t in _directed_tangent_lines(first, second, tol))
    direction = regions[j]['incoming' if k < j else 'outgoing']
    lines.append((node['center']-node['turn']*R*_rotate(direction), direction))
    region = regions[k]
    shift = -R*region['incoming']+R*region['outgoing']
    turn = circle_node(k, points[k], region, R)['turn']
    seen = set()
    for contact, direction in lines:
        # Center = contact + tau R Jd; subtract the fixed vertex-to-center shift.
        origin = contact+turn*R*_rotate(direction)-shift
        interval = _region_line_interval(origin, direction, region, tol)
        if interval is None:
            continue
        parameter = np.clip((points[k]-origin) @ direction, *interval)
        candidate = origin+parameter*direction
        key = tuple(candidate)
        if key not in seen and region_margin(candidate, region) >= -tol:
            seen.add(key)
            yield candidate


def refine_forward_backward(orthogonal, regions, corridors, r, R, *,
                            exclude_circle_overlap=False, tol=1e-8, corridor_offset=0):
    """One forward and one backward pass with bounded, certified local moves.

    For each circle try a cornerward ray target, then halfway to that target.
    At each position try unchanged neighbors, then repair at most one neighbor
    using tangent-line geometry (previous in traversal order first). If no move
    succeeds, try skipping the current circle, then keep its current placement.
    No 2R spacing constraint is imposed here. Safety follows from A_j membership,
    directed tangent containment, ordered quarter contacts and used-arc checks.
    All trials reuse unchanged certificates; the final check uses a fresh cache.
    """
    started = perf_counter()
    original = np.asarray(orthogonal, dtype=float).copy()
    baseline = certify_path(original, regions, corridors, r, R, tol=tol, corridor_offset=corridor_offset)
    if not baseline['feasible']:
        return dict(baseline, status='BASELINE_CERTIFICATION_FAILED', points=original,
                    computation_time=perf_counter()-started)
    current = original.copy()
    history, skipped, cache = [], [], {}
    turns = [j for j in range(1, len(current)-1) if regions[j] is not None]

    def validate(candidate, omitted):
        return certify_path(candidate, regions, corridors, r, R, skipped=omitted,
                            exclude_circle_overlap=exclude_circle_overlap, tol=tol,
                            corridor_offset=corridor_offset, _cache=cache)

    for pass_name, order in (('forward', turns), ('backward', turns[::-1])):
        for j in order:
            if j in skipped:
                continue
            active = [k for k in range(len(current)) if k not in skipped]
            position = active.index(j)
            neighbors = [active[position-1], active[position+1]]
            if pass_name == 'backward':
                neighbors.reverse()
            neighbors = [k for k in neighbors if 0 < k < len(current)-1 and regions[k] is not None]
            target = _corner_target(current[j], regions[j], tol)
            accepted = False
            for fraction in (1., .5):
                if np.max(abs(target-current[j])) <= tol:
                    break
                candidate = current.copy()
                candidate[j] += fraction*(target-current[j])
                action, repaired = 'corner_move', []
                tested = validate(candidate, skipped)
                if not tested['feasible']:
                    for k in neighbors:
                        for repair in _neighbor_repairs(candidate, j, k, active, regions, R, tol):
                            trial = candidate.copy()
                            trial[k] = repair
                            tested = validate(trial, skipped)
                            if tested['feasible']:
                                candidate, action, repaired = trial, 'neighbor_repair', [k]
                                break
                        if tested['feasible']:
                            break
                if tested['feasible']:
                    current, accepted = candidate, True
                    history.append(dict(pass_name=pass_name, action=action, vertices=[j],
                                        fraction=fraction, neighbor_repair=repaired,
                                        accepted_points=current[[j]+repaired].copy()))
                    break
            if not accepted:
                omitted = sorted(skipped+[j])
                if validate(current, omitted)['feasible']:
                    skipped = omitted
                    history.append(dict(pass_name=pass_name, action='skip', vertices=[j], skipped=True))
                else:
                    history.append(dict(pass_name=pass_name, action='keep', vertices=[j]))
    certificate = certify_path(current, regions, corridors, r, R, skipped=skipped,
                               exclude_circle_overlap=exclude_circle_overlap, tol=tol,
                               corridor_offset=corridor_offset)
    if not certificate['feasible']:
        return dict(baseline, status='ORTHOGONAL_FALLBACK', points=original,
                    history=history, reason='Final certification failed; retained the original baseline.',
                    supporting_circle_exclusion_enforced=False,
                    computation_time=perf_counter()-started)
    changed = bool(skipped or np.max(abs(current-original)) > tol)
    return dict(certificate, status='REFINED' if changed else 'ORTHOGONAL_RETAINED',
                points=current, history=history,
                supporting_circle_exclusion_enforced=exclude_circle_overlap,
                computation_time=perf_counter()-started)


def refine_by_blocks(*args, **kwargs):
    """Compatibility entry point; refinement now uses forward/backward local moves."""
    return refine_forward_backward(*args, **kwargs)
