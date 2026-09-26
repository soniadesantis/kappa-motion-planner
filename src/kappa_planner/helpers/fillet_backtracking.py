"""Bounded midpoint-first backtracking using the unchanged nominal supports."""

import numpy as np

from .fillet_safety import region_margin, region_contains_point
from .nominal_polyline import _propagate_coordinate


def region_slice(region, fixed_axis, fixed_value, tol=1e-8):
    """Exact free-coordinate interval of an axis-aligned line through A_j."""
    free_axis = 1-fixed_axis
    if region['empty'] or not (region['low'][fixed_axis]-tol <= fixed_value
                               <= region['high'][fixed_axis]+tol):
        return None
    low, high = float(region['low'][free_axis]), float(region['high'][free_axis])
    radius = region['radius']
    if radius >= 0:
        frame, offset = region['frame'], region['offset']
        column = int(np.argmax(abs(frame[free_axis])))
        fixed = max(0., (fixed_value+offset[fixed_axis])*frame[fixed_axis, 1-column])
        if fixed > radius+tol:
            return None
        bound = np.sqrt(max(0., radius*radius-fixed*fixed))*frame[free_axis, column]-offset[free_axis]
        if frame[free_axis, column] > 0:
            high = min(high, bound)
        else:
            low = max(low, bound)
    if low > high+tol:
        return None
    return (low, high) if low <= high else ((low+high)/2,)*2


def _choices(intervals, preferred=None):
    """Midpoint first, then a bounded set of interior and boundary alternatives."""
    for low, high in intervals:
        values = [(low+high)/2]
        if preferred is not None and low <= preferred <= high:
            values.append(preferred)
        values.extend([.75*low+.25*high, .25*low+.75*high, low, high])
        seen = set()
        for value in values:
            value = float(value)
            if value not in seen:
                seen.add(value)
                yield value


def backtrack_fillet_waypoints(nominal, regions, R, tol=1e-8, *, max_attempts=128):
    """Find a witness cheaply; failure means unresolved, never infeasible.

    Forward reachable intervals still come solely from rectangular doors D_j.
    Curved regions are intersected analytically only during point selection.
    The finite retry budget is followed by the caller's joint-solver fallback.
    """
    initial, doors, axes = nominal['polyline'], nominal['doors'], nominal['directions']
    n = len(doors)
    if len(regions) != n:
        raise ValueError('Expected one optional fillet region per door.')
    attempts = 0

    def failure(reason):
        return dict(feasible=False, status='BACKTRACKING_UNRESOLVED', reason=reason,
                    points=None, backtracking_attempts=attempts)

    if max_attempts <= 0:
        return failure('Midpoint backtracking budget is zero.')
    if any(region is not None and region['empty'] for region in regions):
        return failure('A local fillet region is empty.')
    reachable = []
    for k, name in enumerate(('x', 'y')):
        supports = nominal.get(name+'_reachable')
        if not supports:
            relations = ['separate' if axis == ('H' if k == 0 else 'V') else 'equal' for axis in axes]
            supports = _propagate_coordinate([door[name] for door in doors], relations, 2*R, tol)
        reachable.append(supports)
    low = np.array([[door['x'][0], door['y'][0]] for door in doors], dtype=float)
    high = np.array([[door['x'][1], door['y'][1]] for door in doors], dtype=float)
    for j, region in enumerate(regions):
        if region is not None:
            low[j] = np.maximum(low[j], region['low'])
            high[j] = np.minimum(high[j], region['high'])
    points = np.empty((n, 2))

    def intervals(j, k, lo, hi):
        result = []
        for a, b in reachable[k][j]:
            a, b = max(a, lo, low[j, k]), min(b, hi, high[j, k])
            if a <= b+tol:
                result.append((a, b) if a <= b else ((a+b)/2,)*2)
        return result

    def choose(j):
        nonlocal attempts
        if j < 0:
            return True
        if attempts >= max_attempts:
            return False
        k = 0 if axes[j] == 'H' else 1
        t = 1-k
        fixed = points[j+1, t]
        if not any(a-tol <= fixed <= b+tol for a, b in intervals(j, t, -np.inf, np.inf)):
            return False
        lo, hi = low[j, k], high[j, k]
        sign = np.sign(initial[j+1, k]-initial[j, k])
        if sign > 0:
            hi = min(hi, points[j+1, k]-2*R)
        else:
            lo = max(lo, points[j+1, k]+2*R)
        # In an alternating chain this coordinate is shared with the preceding
        # vertex. A local box look-ahead avoids choices that immediately fail.
        if j and axes[j-1] != axes[j]:
            lo, hi = max(lo, low[j-1, k]), min(hi, high[j-1, k])
        # Try the interval center with a squared-distance region test first.
        # Only compute curved-boundary intersections if that cheap attempt fails.
        tried = set()
        for a, b in intervals(j, k, lo, hi):
            if attempts >= max_attempts:
                return False
            value = (a+b)/2
            tried.add(float(value))
            attempts += 1
            points[j, k], points[j, t] = value, fixed
            if regions[j] is not None and not region_contains_point(points[j], regions[j], tol):
                continue
            if choose(j-1):
                return True
        if regions[j] is not None:
            sliced = region_slice(regions[j], t, fixed, tol)
            if sliced is None:
                return False
            lo, hi = max(lo, sliced[0]), min(hi, sliced[1])
        for value in _choices(intervals(j, k, lo, hi), initial[j, k]):
            if value in tried:
                continue
            if attempts >= max_attempts:
                return False
            attempts += 1
            points[j, k], points[j, t] = value, fixed
            if regions[j] is not None and not region_contains_point(points[j], regions[j], tol):
                continue
            if choose(j-1):
                return True
        return False

    # The last door has no corner constraint in the production pipeline.
    # Filter it too for callers supplying an explicit terminal region.
    terminal = [intervals(n-1, k, -np.inf, np.inf) for k in (0, 1)]
    t = 1-(0 if axes[-1] == 'H' else 1)
    terminal[t] = intervals(n-1, t, low[n-2, t], high[n-2, t])
    for x in _choices(terminal[0], initial[-1, 0]):
        for y in _choices(terminal[1], initial[-1, 1]):
            if attempts >= max_attempts:
                return failure('Midpoint backtracking exhausted its retry budget.')
            attempts += 1
            points[-1] = x, y
            if regions[-1] is not None and region_margin(points[-1], regions[-1]) < -tol:
                continue
            if choose(n-2):
                violation = max(0., float(np.max(low-points)), float(np.max(points-high)))
                for j, axis in enumerate(axes):
                    k = 0 if axis == 'H' else 1
                    signed_length = (points[j+1, k]-points[j, k])*np.sign(initial[j+1, k]-initial[j, k])
                    violation = max(violation, abs(points[j+1, 1-k]-points[j, 1-k]), 2*R-signed_length)
                for j, region in enumerate(regions):
                    if region is not None:
                        violation = max(violation, -float(region_margin(points[j], region)))
                if np.all(np.isfinite(points)) and violation <= tol:
                    return dict(feasible=True, status='FEASIBLE', points=points.copy(),
                                reason='Fillet-aware midpoint backtracking found a compatible chain.',
                                max_violation=float(violation), backtracking_attempts=attempts)
                return failure('Backtracking witness failed final validation.')
    return failure('No witness found among the midpoint and alternate choices.')
