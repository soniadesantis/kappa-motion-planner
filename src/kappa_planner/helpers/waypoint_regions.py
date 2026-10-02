"""Exact rectangle-union propagation for safe orthogonal waypoint chains."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .centroid_polyline_validity import check_centroid_polyline


def _rectangles(items):
    return np.asarray(items, dtype=float).reshape(-1, 4)


def _union(items, tol):
    """Remove contained pieces; merge only when the union is itself a rectangle.

    Bounds are (xmin, xmax, ymin, ymax). Lines and points are retained because
    equality at the clearance/length limits is feasible.
    """
    result = []
    for item in items:
        box = np.array(item, dtype=float)
        if box[0] > box[1] + tol or box[2] > box[3] + tol:
            continue
        for lo, hi in ((0, 1), (2, 3)):
            if box[lo] > box[hi]:
                box[lo] = box[hi] = (box[lo] + box[hi]) / 2
        result.append(box)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                a, b = result[i], result[j]
                if (a[0] >= b[0]-tol and a[1] <= b[1]+tol
                        and a[2] >= b[2]-tol and a[3] <= b[3]+tol):
                    result.pop(i)
                elif (b[0] >= a[0]-tol and b[1] <= a[1]+tol
                      and b[2] >= a[2]-tol and b[3] <= a[3]+tol):
                    result.pop(j)
                elif (np.all(abs(a[2:]-b[2:]) <= tol)
                      and max(a[0], b[0]) <= min(a[1], b[1])+tol):
                    result[i] = np.array([min(a[0], b[0]), max(a[1], b[1]), a[2], a[3]])
                    result.pop(j)
                elif (np.all(abs(a[:2]-b[:2]) <= tol)
                      and max(a[2], b[2]) <= min(a[3], b[3])+tol):
                    result[i] = np.array([a[0], a[1], min(a[2], b[2]), max(a[3], b[3])])
                    result.pop(j)
                else:
                    continue
                changed = True
                break
            if changed:
                break
    return _rectangles(result)


def _intersection(first, second, tol):
    return _union([
        [max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])]
        for a in first for b in second
    ], tol)


def aligned_overlap_pairs(entrance, exit_region, axis, tol=1e-9):
    """Restrict pairs of overlap pieces to their common transverse range.

    For a horizontal corridor both pieces get the intersection of their y
    intervals; for a vertical corridor they get the intersection of their x
    intervals. Returns a list of (entrance_rectangle, exit_rectangle) pairs.
    The pairing is retained rather than combining unrelated alternatives.
    This is only the alignment stage; no longitudinal distance is imposed.
    """
    if axis not in ("horizontal", "vertical"):
        raise ValueError("axis must be horizontal or vertical.")
    transverse = 2 if axis == "horizontal" else 0
    pairs = []
    for first in _rectangles(entrance):
        for second in _rectangles(exit_region):
            low = max(first[transverse], second[transverse])
            high = min(first[transverse+1], second[transverse+1])
            if low > high + tol:
                continue
            if low > high:
                low = high = (low+high)/2
            a, b = first.copy(), second.copy()
            a[transverse] = b[transverse] = low
            a[transverse+1] = b[transverse+1] = high
            pairs.append((a, b))
    return pairs


def _supported(source, destination, axes, distance, tol):
    """Destination points with at least one compatible source point."""
    pieces = []
    for a in source:
        for b in destination:
            for axis in axes:
                longitudinal = 0 if axis == "horizontal" else 2
                transverse = 2-longitudinal
                shared_low = max(a[transverse], b[transverse])
                shared_high = min(a[transverse+1], b[transverse+1])
                if shared_low > shared_high + tol:
                    continue
                left, right = b.copy(), b.copy()
                left[transverse] = right[transverse] = shared_low
                left[transverse+1] = right[transverse+1] = shared_high
                left[longitudinal+1] = min(b[longitudinal+1], a[longitudinal+1]-distance)
                right[longitudinal] = max(b[longitudinal], a[longitudinal]+distance)
                pieces.extend((left, right))
    return _union(pieces, tol)


def _midpoint(region):
    # Prefer a positive-area piece; lines and points remain selectable.
    areas = (region[:, 1]-region[:, 0]) * (region[:, 3]-region[:, 2])
    box = region[np.argmax(areas)]
    return np.array([(box[0]+box[1])/2, (box[2]+box[3])/2])


def _separated_pair(first, second, axis, distance, tol):
    """Trim a before b, preserving a longitudinal gap >= distance.

    Choose the balanced cut maximizing the product of retained longitudinal
    lengths. This is one conservative member of the continuum of valid cuts.
    """
    coordinate = 0 if axis == "horizontal" else 2
    a0, a1 = first[coordinate:coordinate+2]
    b0, b1 = second[coordinate:coordinate+2]
    if b1-a0 < distance-tol:
        return None
    cut = (a0+b1-distance)/2
    cut = np.clip(cut, min(a1, b0-distance), max(a1, b0-distance))
    cut = np.clip(cut, a0, max(a0, b1-distance))
    a, b = first.copy(), second.copy()
    a[coordinate+1] = min(a1, cut)
    b[coordinate] = max(b0, cut+distance)
    return a, b


def compute_corridor_point_sets(corridors, r, R, *, tol=1e-9):
    """Paired entrance/exit sets with EVERY cross-pair at distance >= R.

    Uses safe overlaps (inset r). Each intermediate corridor offers up to four
    alternatives: either axis, either travel direction. Both rectangles share
    exactly the same transverse interval and have a longitudinal gap >= R.
    Thus every cross-pair has Euclidean distance >= R; equal-coordinate partners
    can be chosen anywhere in their shared transverse interval.

    Overlapping longitudinal intervals are trimmed with a balanced cut. Those
    finite candidates are conservative, NOT the complete continuous family of
    possible cuts. Chain propagation/backtracking selects compatible candidates
    where possible. Failure proves only that these candidates do not connect.

    Returns candidates and selected_pairs (one per internal corridor), plus
    shared_regions: intersections of neighboring selected sets at each overlap.
    A sequence-wide equality check is also performed, since nonempty local
    intersections alone are insufficient. The witness excludes start/end poses;
    neither 2R fillet compatibility nor arc clearance is certified here.
    """
    started = perf_counter()
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive.")
    _, basic = check_centroid_polyline(corridors, r, tol)
    if not basic["axis_aligned_ok"]:
        raise ValueError("Local point sets require axis-aligned rectangles.")
    corners = np.asarray([c.corners for c in corridors])
    low, high = corners.min(axis=1), corners.max(axis=1)
    lo, hi = np.maximum(low[:-1], low[1:]), np.minimum(high[:-1], high[1:])
    raw, safe = [], []
    for j in range(len(lo)):
        box = [lo[j, 0], hi[j, 0], lo[j, 1], hi[j, 1]]
        exists = basic["overlaps"][j]["exists"]
        raw.append(_rectangles([box]) if exists else _rectangles([]))
        safe.append(_union([[box[0]+r, box[1]-r, box[2]+r, box[3]-r]], tol)
                    if exists else _rectangles([]))
    candidates = []
    for j in range(len(safe)-1):
        alternatives = []
        for axis in ("horizontal", "vertical"):
            for entrance, exit_region in aligned_overlap_pairs(safe[j], safe[j+1], axis, tol):
                for direction in (1, -1):
                    pair = (_separated_pair(entrance, exit_region, axis, R, tol)
                            if direction == 1 else
                            _separated_pair(exit_region, entrance, axis, R, tol))
                    if pair is None:
                        continue
                    a, b = pair if direction == 1 else pair[::-1]
                    alternatives.append(dict(corridor=j+1, axis=axis, direction=direction,
                                             entrance=a, exit=b))
        def coverage(candidate):
            a, b = candidate["entrance"], candidate["exit"]
            area_a, area_b = (a[1]-a[0])*(a[3]-a[2]), (b[1]-b[0])*(b[3]-b[2])
            return -(area_a+area_b)
        alternatives.sort(key=coverage)
        candidates.append(alternatives)

    forward = [safe[0].copy()]
    per_candidate = []
    for j, alternatives in enumerate(candidates):
        reachable = []
        for candidate in alternatives:
            source = _intersection(forward[j], [candidate["entrance"]], tol)
            reachable.append(_supported(source, [candidate["exit"]],
                                        [candidate["axis"]], 0, tol))
        per_candidate.append(reachable)
        forward.append(_union([box for region in reachable for box in region], tol))
    feasible = bool(len(forward[-1]))
    selected = [None] * len(candidates)
    points = np.empty((0, 2))
    if feasible:
        points = np.empty((len(safe), 2))
        points[-1] = _midpoint(forward[-1])
        for j in range(len(candidates)-1, -1, -1):
            x, y = points[j+1]
            singleton = _rectangles([[x, x, y, y]])
            for candidate, reachable in zip(candidates[j], per_candidate[j]):
                if not len(_intersection(singleton, reachable, tol)):
                    continue
                source = _intersection(forward[j], [candidate["entrance"]], tol)
                compatible = _supported(singleton, source, [candidate["axis"]], 0, tol)
                points[j] = _midpoint(compatible)
                selected[j] = candidate
                break
    else:
        selected = [items[0] if items else None for items in candidates]
    shared = [region.copy() for region in safe]
    for j, candidate in enumerate(selected):
        if candidate is None:
            shared[j] = shared[j+1] = _rectangles([])
        else:
            shared[j] = _intersection(shared[j], [candidate["entrance"]], tol)
            shared[j+1] = _intersection(shared[j+1], [candidate["exit"]], tol)
    errors = [f"Corridor {j+1}: no aligned entrance/exit sets separated by R."
              for j, items in enumerate(candidates) if not items]
    if not feasible and not errors:
        errors.append("No complete chain found among the balanced-cut candidates.")
    return dict(raw_overlaps=raw, safe_regions=safe, candidates=candidates,
                selected_pairs=selected, shared_regions=shared, feasible=feasible,
                local_intersections_nonempty=all(len(region) > 0 for region in shared),
                waypoints=points, separation=R, errors=errors,
                computation_time=perf_counter()-started)


def propagate_waypoint_regions(corridors, r, R, *, axes=None, tol=1e-9):
    """Return all waypoint locations belonging to a complete feasible chain.

    Safe regions D[j] are consecutive rectangle overlaps eroded by a disk of
    radius r (bounds inset by r). Each internal segment must share x or y and
    have length >= 2R. ``axes`` may specify 'horizontal', 'vertical', or 'either'
    for each of the N-2 INTERNAL corridors. By default either axis is allowed,
    matching the orientation-free rectangle model. Axes are undirected.

    Forward/backward passes retain finite unions of rectangles, never their
    bounding boxes. G[j] = F[j] intersect B[j] is exact for these constraints,
    up to tol. A compatible witness is reconstructed from forward supports;
    independent midpoints of G[j] would not necessarily be compatible.

    Returns a dict containing raw_overlaps, safe_regions, forward_regions,
    backward_regions, feasible_regions, feasible, waypoints, segment_axes,
    allowed_axes, errors, and computation_time. Every region is a (K,4) array
    of (xmin,xmax,ymin,ymax); K=0 denotes empty. Closed lines/points are valid.

    The witness contains overlap waypoints only. Start/end poses, their segment
    lengths, U-turn avoidance, and fillet-arc clearance are NOT constraints of
    this problem. Infeasibility here does not prove all possible paths impossible.
    """
    started = perf_counter()
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive.")
    _, basic = check_centroid_polyline(corridors, r, tol)
    if not basic["axis_aligned_ok"]:
        raise ValueError("Waypoint-region propagation requires axis-aligned rectangles.")
    count = len(corridors)-1
    if axes is None:
        axes = ["either"] * (count-1)
    if len(axes) != count-1 or any(a not in ("horizontal", "vertical", "either") for a in axes):
        raise ValueError("axes needs one horizontal/vertical/either entry per internal corridor.")
    allowed = [("horizontal", "vertical") if a == "either" else (a,) for a in axes]
    corners = np.asarray([c.corners for c in corridors])
    low, high = corners.min(axis=1), corners.max(axis=1)
    lo, hi = np.maximum(low[:-1], low[1:]), np.minimum(high[:-1], high[1:])
    raw, safe, errors = [], [], []
    for j in range(count):
        overlap = [lo[j, 0], hi[j, 0], lo[j, 1], hi[j, 1]]
        exists = basic["overlaps"][j]["exists"]
        raw.append(_rectangles([overlap]) if exists else _rectangles([]))
        region = _union([[overlap[0]+r, overlap[1]-r, overlap[2]+r, overlap[3]-r]], tol)
        safe.append(region if exists else _rectangles([]))
        if not len(safe[-1]):
            errors.append(f"Overlap {j}: safe overlap is empty.")
    forward = [safe[0].copy()]
    for j in range(1, count):
        forward.append(_supported(forward[-1], safe[j], allowed[j-1], 2*R, tol))
    backward = [None] * count
    backward[-1] = safe[-1].copy()
    for j in range(count-2, -1, -1):
        backward[j] = _supported(backward[j+1], safe[j], allowed[j], 2*R, tol)
    feasible_regions = [_intersection(f, b, tol) for f, b in zip(forward, backward)]
    feasible = bool(len(forward[-1]))
    points, chosen_axes = np.empty((0, 2)), []
    if feasible:
        points = np.empty((count, 2))
        points[-1] = _midpoint(forward[-1])
        for j in range(count-2, -1, -1):
            x, y = points[j+1]
            supported = _supported(_rectangles([[x, x, y, y]]), forward[j],
                                   allowed[j], 2*R, tol)
            points[j] = _midpoint(supported)
        for j, delta in enumerate(np.diff(points, axis=0)):
            chosen_axes.append("horizontal" if abs(delta[1]) <= tol
                               and "horizontal" in allowed[j] else "vertical")
    elif not errors:
        errors.append("No complete waypoint chain satisfies the equality and 2R constraints.")
    return dict(
        raw_overlaps=raw, safe_regions=safe, forward_regions=forward,
        backward_regions=backward, feasible_regions=feasible_regions,
        feasible=feasible, waypoints=points, segment_axes=chosen_axes,
        allowed_axes=allowed, errors=errors, computation_time=perf_counter()-started,
    )
