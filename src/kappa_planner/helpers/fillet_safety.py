"""Revised local fillet rule and independent continuous arc validation."""

from __future__ import annotations

from math import isfinite

import numpy as np


def revised_corner_condition(vertices, corner, incoming, outgoing, R, r, tol=1e-9):
    """Evaluate the proposed OR condition in the canonical corner frame.

    e1=-incoming and e2=outgoing point into the missing quadrant. In this frame
    p=(-xp,-yp), and o_local=p_local+(R,R). The global circle center is
    o=p-R*incoming+R*outgoing. The component-wise OR must use o_local, not the
    global coordinates. This handles translations, rotations and reflections.

    This is a conjecture predicate only. Safe-overlap membership, safe tangent
    segments, and selection of the relevant concave corner are prerequisites.
    """
    u, v = np.asarray(incoming, dtype=float), np.asarray(outgoing, dtype=float)
    if (u.shape != (2,) or v.shape != (2,) or not np.all(np.isfinite([u, v]))
            or not np.isclose(u @ u, 1.) or not np.isclose(v @ v, 1.)
            or not np.isclose(u @ v, 0.)):
        raise ValueError('Directions must be perpendicular unit vectors.')
    if (not np.isfinite(R) or R <= 0 or not np.isfinite(r) or r < 0
            or not np.isfinite(tol) or tol < 0):
        raise ValueError('Require finite R>0, r>=0, tol>=0.')
    vertices, corner = np.asarray(vertices, dtype=float), np.asarray(corner, dtype=float)
    if (vertices.ndim < 1 or vertices.shape[-1] != 2 or corner.shape != (2,)
            or not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(corner))):
        raise ValueError('Expected finite XY vertices and one corner.')
    centers = vertices-R*u+R*v
    local_centers = np.stack(((centers-corner) @ (-u), (centers-corner) @ v), axis=-1)
    far = np.any(local_centers <= tol, axis=-1)
    circular = (R >= r) & (np.linalg.norm(local_centers, axis=-1) <= R-r+tol)
    return dict(predicted=far | circular, far=far, circular=circular,
                centers=centers, local_centers=local_centers)


def quarter_arc_points(vertices, incoming, outgoing, R, angles):
    centers = np.asarray(vertices)-R*incoming+R*outgoing
    return centers[..., None, :] + R*(-np.cos(angles)[..., None]*outgoing
                                      + np.sin(angles)[..., None]*incoming)


def continuous_arc_clearance(vertices, incoming, outgoing, R, corridors, boundary,
                             clearance_function):
    """Independent whole-quarter-arc check, to floating-point accuracy.

    Arc/edge distance minima occur at arc endpoints, radial directions to edge
    endpoints, coordinate extrema, or crossings of edge supporting lines.
    Here the quarter arcs and rectangle edges are axis-aligned, so coordinate
    extrema occur at the arc endpoints. Evaluate all these critical angles.
    Test midpoints between them too, to detect outside portions between boundary
    crossings. For contained arcs the minimum clearance is continuous, not a
    coarse angular sample. Outside arcs return a negative clearance witness.
    """
    u, v = np.asarray(incoming), np.asarray(outgoing)
    if (not np.all(np.isin([u, v], [-1., 0., 1.])) or u @ v != 0
            or u @ u != 1 or v @ v != 1):
        raise ValueError('The critical-angle check requires perpendicular signed axis vectors.')
    minima, witnesses = [], []
    for start in range(0, len(vertices), 256):
        points = np.asarray(vertices[start:start+256])
        centers = points-R*u+R*v
        delta = boundary.reshape(-1, 2)[None, :, :]-centers[:, None, :]
        local_x, local_y = delta @ (-v), delta @ u
        angles = np.concatenate((np.zeros((len(points), 1)),
                                 np.full((len(points), 1), np.pi/2),
                                 np.clip(np.arctan2(local_y, local_x), 0, np.pi/2),
                                 np.arccos(np.clip(local_x/R, 0, 1)),
                                 np.arcsin(np.clip(local_y/R, 0, 1))), axis=1)
        angles.sort(axis=1)
        angles = np.concatenate((angles, (angles[:, :-1]+angles[:, 1:])/2), axis=1)
        arcs = quarter_arc_points(points, u, v, R, angles)
        clearances = clearance_function(arcs, corridors, boundary)
        indices = np.argmin(clearances, axis=1)
        minima.extend(clearances[np.arange(len(points)), indices])
        witnesses.extend(arcs[np.arange(len(points)), indices])
    return np.asarray(minima), np.asarray(witnesses).reshape(-1, 2)


def fillet_vertex_region(door, pair, corner, incoming, outgoing, r, R, tol=1e-9,
                         *, pair_bounds=None, validate_inputs=True):
    """Clip a safe door by the tangent and canonical-frame prerequisites.

    On this clipped box, local p<=-r and hence local o<=R-r componentwise.
    For R>=r the revised OR rule is exactly ||max(local_o,0)|| <= R-r:
    if one component is nonpositive, the other is already <=R-r. For R<r
    both components are negative, so every point of the box passes the OR.
    This representation retains the far-arm regions, not just the old disk.
    The validated nominal pipeline can supply stored bounds and skip repeated
    direction/scalar validation; standalone callers retain validation by default.
    """
    u, v, c = map(lambda x: np.asarray(x, dtype=float), (incoming, outgoing, corner))
    if validate_inputs:
        revised_corner_condition(np.empty((0, 2)), c, u, v, R, r, tol)
        if not np.all(np.isin([u, v], [-1., 0., 1.])):
            raise ValueError('The waypoint-region representation requires axis directions.')
    low = np.array([door['x'][0], door['y'][0]])
    high = np.array([door['x'][1], door['y'][1]])
    for index, (corridor, shift) in enumerate(zip(pair, (R*u, -R*v))):
        if pair_bounds is None:
            corners = np.asarray(corridor.corners)
            c_low, c_high = corners.min(axis=0), corners.max(axis=0)
        else:
            a, b, c0, d = pair_bounds[index]
            c_low, c_high = np.array([a, c0]), np.array([b, d])
        low = np.maximum(low, c_low+r+shift)
        high = np.minimum(high, c_high-r+shift)
    frame = np.column_stack((-u, v))
    for direction in (-u, v):
        k = int(np.argmax(abs(direction)))
        if direction[k] > 0:
            high[k] = min(high[k], c[k]-r)
        else:
            low[k] = max(low[k], c[k]+r)
    empty = bool(np.any(low > high+tol))
    # Normalize only floating-point boundary contacts.
    near = (low > high) & (low <= high+tol)
    low[near] = high[near] = (low[near]+high[near])/2
    result = dict(low=low, high=high, corner=c, frame=frame,
                  offset=-R*u+R*v-c, radius=R-r, empty=empty, witness=None,
                  incoming=u, outgoing=v)
    if not empty:
        # Each canonical coordinate depends on a different world coordinate,
        # so both can attain their minimum simultaneously within the rectangle.
        minimum_point = np.where(frame.sum(axis=1) > 0, low, high)
        local_o = (minimum_point+result['offset']) @ frame
        empty = R >= r and np.linalg.norm(np.maximum(local_o, 0)) > R-r+tol
        result.update(empty=bool(empty), witness=None if empty else minimum_point)
    return result


def region_margin(points, region):
    """Signed margin for the complete local region, including prerequisites."""
    points = np.asarray(points)
    if region['empty']:
        return np.full(points.shape[:-1], -np.inf)
    margin = np.min(np.minimum(points-region['low'], region['high']-points), axis=-1)
    if region['radius'] >= 0:
        local_o = (points+region['offset']) @ region['frame']
        margin = np.minimum(margin, region['radius']-np.linalg.norm(np.maximum(local_o, 0), axis=-1))
    return margin


def region_contains_point(point, region, tol=1e-8):
    """Scalar membership test: box comparisons and a squared corner-frame norm."""
    x, y = float(point[0]), float(point[1])
    if (region['empty'] or not isfinite(x) or not isfinite(y)
            or x < region['low'][0]-tol or x > region['high'][0]+tol
            or y < region['low'][1]-tol or y > region['high'][1]+tol):
        return False
    radius = region['radius']
    if radius < 0:
        return True
    x += region['offset'][0]
    y += region['offset'][1]
    frame = region['frame']
    a = max(0., x*frame[0, 0]+y*frame[1, 0])
    b = max(0., x*frame[0, 1]+y*frame[1, 1])
    return bool(a*a+b*b <= (radius+tol)**2)


def axis_aligned_concave_corners(first, second, tol=1e-9):
    """Concave corners of two validated axis-aligned rectangles, from bounds.

    Candidates are the four corners of their overlap. A concave corner has
    exactly three occupied local quadrants. No edge splitting or union polygon
    is required. Bounds use (xmin, xmax, ymin, ymax).
    """
    xmin, xmax = max(first[0], second[0]), min(first[1], second[1])
    ymin, ymax = max(first[2], second[2]), min(first[3], second[3])
    if xmin > xmax+tol or ymin > ymax+tol:
        return np.empty((0, 2))
    result = []
    for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        occupied = [False]*4
        for a, b, c, d in (first, second):
            if a-tol <= x <= b+tol and c-tol <= y <= d+tol:
                left, right = x > a+tol, x < b-tol
                down, up = y > c+tol, y < d-tol
                quadrants = (left and down, left and up, right and down, right and up)
                occupied = [old or new for old, new in zip(occupied, quadrants)]
        if sum(occupied) == 3 and (x, y) not in result:
            result.append((x, y))
    return np.asarray(result, dtype=float).reshape(-1, 2)
