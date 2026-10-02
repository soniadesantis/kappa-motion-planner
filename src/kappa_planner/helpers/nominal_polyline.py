"""Nominal orthogonal path check with unique, alternating H/V directions."""

from __future__ import annotations

from time import perf_counter

import numpy as np

def _bounds(corridor, tol=1e-9):
    corners = np.asarray(corridor.corners, dtype=float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Corridors need four finite XY corners.")
    low, high = corners.min(axis=0), corners.max(axis=0)
    on_low, on_high = abs(corners-low) <= tol, abs(corners-high) <= tol
    codes = on_high[:, 0].astype(int) + 2*on_high[:, 1]
    if (np.any(high-low <= tol) or not np.all(on_low | on_high)
            or not np.array_equal(np.sort(codes), np.arange(4))):
        raise ValueError("Nominal check requires positive axis-aligned rectangles.")
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def _interval_intersection(a, b, tol=1e-9):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo-tol:
        return None
    if hi < lo:  # A numerical boundary contact is a point, not an inverted interval.
        lo = hi = (lo+hi)/2
    return float(lo), float(hi)


def _normalize_intervals(intervals, tol=1e-9):
    normalized = []
    for lo, hi in sorted(intervals):
        if hi < lo-tol:
            continue
        if hi < lo:
            lo = hi = (lo+hi)/2
        if normalized and lo <= normalized[-1][1]+tol:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], hi))
        else:
            normalized.append((lo, hi))
    return normalized


def _propagate_coordinate(intervals, relations, minimum_length, tol=1e-9):
    """Exact one-coordinate forward supports as unions of closed intervals."""
    reachable = [[intervals[0]]]
    for target, relation in zip(intervals[1:], relations):
        previous = reachable[-1]
        if not previous:
            reachable.append([])
            continue
        if relation == "equal":
            candidates = [_interval_intersection(item, target, tol) for item in previous]
            current = [item for item in candidates if item is not None]
        elif relation == "separate":
            # Only extrema are needed for existential absolute separation,
            # even when the previous reachable set contains gaps.
            previous_min, previous_max = previous[0][0], previous[-1][1]
            current = [(target[0], min(target[1], previous_max-minimum_length)),
                       (max(target[0], previous_min+minimum_length), target[1])]
        else:
            raise ValueError(f"Unknown relation: {relation}")
        reachable.append(_normalize_intervals(current, tol))
    return reachable


def _backtrack_coordinate(reachable, relations, minimum_length, tol=1e-9):
    if not reachable[-1]:
        return None
    values = np.empty(len(reachable))
    values[-1] = sum(reachable[-1][0])/2
    for j in range(len(reachable)-1, 0, -1):
        value = values[j]
        for lo, hi in reachable[j-1]:
            if relations[j-1] == "equal":
                if lo-tol <= value <= hi+tol:
                    values[j-1] = value
                    break
            elif relations[j-1] == "separate":
                candidate_hi = min(hi, value-minimum_length)
                candidate_lo = max(lo, value+minimum_length)
                if candidate_hi >= lo-tol:
                    values[j-1] = max(lo, candidate_hi)
                    break
                if candidate_lo <= hi+tol:
                    values[j-1] = min(hi, candidate_lo)
                    break
            else:
                raise ValueError(f"Unknown relation: {relations[j-1]}")
        else:
            raise RuntimeError("Coordinate backtracking failed.")
    return values


def classify_door_spacing(doors, directions, R, tol=1e-9):
    """Local 2R diagnostics for original safe doors, before propagation.

    For each uniquely directed pair, the transverse intervals already overlap.
    The minimum/maximum aligned lengths therefore depend only on the two
    longitudinal intervals. `guaranteed` means every aligned choice satisfies
    2R; `coupled` means some do and some do not; `impossible` means none do.
    Other directions or missing doors are `unavailable`, never guaranteed.

    This is O(number of pairs) time and space. It does not certify global
    feasibility, transverse-coordinate independence, endpoints, or fillets.
    Door and corridor indices refer to the original sequence, even if an
    endpoint extension later copies these diagnostics into its result.
    """
    pairs = []
    for j, (first, second, direction) in enumerate(
            zip(doors, doors[1:], directions)):
        pair = dict(corridor=j+1, overlaps=(j, j+1), direction=direction,
                    minimum_length=None, maximum_length=None,
                    required_length=2*R, status="unavailable")
        if first is not None and second is not None and direction in ("H", "V"):
            axis = "x" if direction == "H" else "y"
            a, b = first[axis], second[axis]
            minimum = max(0., b[0]-a[1], a[0]-b[1])
            maximum = max(abs(b[1]-a[0]), abs(a[1]-b[0]))
            status = ("impossible" if maximum < 2*R-tol else
                      "guaranteed" if minimum >= 2*R-tol else "coupled")
            pair.update(minimum_length=float(minimum), maximum_length=float(maximum),
                        status=status)
        pairs.append(pair)
    return pairs


def check_orthogonal_polyline(corridors, r, R, tol=1e-9):
    """Check the nominal class and return one internal waypoint per door.

    1. D[j] = (C[j] intersect C[j+1]) eroded by the radius-r disk.
    2. Infer H/V from common transverse ranges. Reject BOTH and NEITHER cases.
    3. Require strict alternation of these uniquely inferred directions.
    4. Propagate independent x/y equality and >=2R separation constraints.
    5. Backtrack a compatible polyline; never choose door midpoints independently.

    Feasibility is exact for this nominal class, up to tol. Rejecting ambiguity
    or nonalternation does not imply that every orthogonal path is impossible.
    No start/end pose connection or fillet-arc clearance is checked. Closed
    zero-width/height safe doors are accepted, as is length exactly 2R.

    Result fields are consistent on failure: feasible, status, reason, polyline,
    doors, directions, raw_overlaps, x_reachable, y_reachable, computation_time,
    ambiguous_overlaps, spacing_pairs. The spacing_pairs diagnostics classify
    local 2R coupling on the original safe doors before global propagation;
    see classify_door_spacing. Each ambiguity diagnostic records the internal corridor,
    adjacent overlap indices, raw_intersection and safe_intersection bounds
    (xmin, xmax, ymin, ymax), and whether the safe intersection has positive
    area rather than just boundary contact. For axis-aligned rectangular doors,
    BOTH H and V possible is equivalent to a nonempty safe-door intersection,
    up to tol. All ambiguities are recorded before returning a rejection.
    Door/direction/reason indices are zero-based. Bad scalar arguments or fewer
    than three corridors raise ValueError; geometric rejection returns a report.
    """
    started = perf_counter()
    if len(corridors) < 3:
        raise ValueError("At least three corridors are required for an internal polyline.")
    if not np.isfinite(r) or r < 0 or not np.isfinite(R) or R <= 0:
        raise ValueError("r must be finite and nonnegative; R must be finite and positive.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    result = dict(feasible=False, status=None, reason=None, polyline=None,
                  doors=[], directions=[], raw_overlaps=[],
                  x_reachable=[], y_reachable=[], ambiguous_overlaps=[], spacing_pairs=[])

    def finish(status, reason=None):
        result.update(status=status, reason=reason, feasible=status == "feasible",
                      computation_time=perf_counter()-started)
        return result

    try:
        bounds = [_bounds(c, tol) for c in corridors]
    except ValueError as error:
        return finish("unsupported_geometry", str(error))
    result['corridor_bounds'] = bounds  # Reuse for endpoint slices and local fillet geometry.
    for a, b in zip(bounds, bounds[1:]):
        xmin, xmax = max(a[0], b[0]), min(a[1], b[1])
        ymin, ymax = max(a[2], b[2]), min(a[3], b[3])
        result["raw_overlaps"].append((xmin, xmax, ymin, ymax))
        dx, dy = xmax-xmin, ymax-ymin
        if dx < 2*r-tol or dy < 2*r-tol:
            result["doors"].append(None)
        else:
            # Normalize an interval inverted only by floating-point tolerance.
            x = _interval_intersection((xmin+r, xmax-r), (xmin+r, xmax-r), tol)
            y = _interval_intersection((ymin+r, ymax-r), (ymin+r, ymax-r), tol)
            result["doors"].append(dict(x=x, y=y, raw_dx=dx, raw_dy=dy))
    for j, (previous, following) in enumerate(
            zip(result["doors"], result["doors"][1:]), start=1):
        if previous is None or following is None:
            result["directions"].append("UNAVAILABLE")
            continue
        horizontal = _interval_intersection(previous["y"], following["y"], tol) is not None
        vertical = _interval_intersection(previous["x"], following["x"], tol) is not None
        result["directions"].append(
            "AMBIGUOUS" if horizontal and vertical else
            "H" if horizontal else "V" if vertical else "INFEASIBLE"
        )
        if horizontal and vertical:
            x = _interval_intersection(previous["x"], following["x"], tol)
            y = _interval_intersection(previous["y"], following["y"], tol)
            a, b = result["raw_overlaps"][j-1:j+1]
            result["ambiguous_overlaps"].append(dict(
                corridor=j, overlaps=(j-1, j),
                raw_intersection=(max(a[0], b[0]), min(a[1], b[1]),
                                  max(a[2], b[2]), min(a[3], b[3])),
                safe_intersection=(*x, *y),
                positive_area=(x[1]-x[0] > tol and y[1]-y[0] > tol),
            ))
    result["spacing_pairs"] = classify_door_spacing(
        result["doors"], result["directions"], R, tol)
    for j, door in enumerate(result["doors"]):
        if door is None:
            return finish("empty_overlap", f"Overlap {j}-{j+1} does not contain a 2r × 2r square.")
    for j, direction in enumerate(result["directions"], start=1):
        if direction == "AMBIGUOUS":
            return finish("ambiguous_direction", f"Internal corridor {j} has ambiguous H/V direction; its adjacent safe overlaps intersect.")
        if direction == "INFEASIBLE":
            return finish("no_orthogonal_connection", f"Internal corridor {j} admits no orthogonal connection.")
    directions = result["directions"]
    if any(a == b for a, b in zip(directions, directions[1:])):
        return finish("nonalternating", "Internal segment directions do not alternate H/V.")
    x_relations = ["separate" if d == "H" else "equal" for d in directions]
    y_relations = ["equal" if d == "H" else "separate" for d in directions]
    for name, relations in (("x", x_relations), ("y", y_relations)):
        result[f"{name}_reachable"] = _propagate_coordinate(
            [door[name] for door in result["doors"]], relations, 2*R, tol
        )
    if not result["x_reachable"][-1] or not result["y_reachable"][-1]:
        return finish("spacing", "No orthogonal waypoint sequence satisfies the 2R spacing.")
    xs = _backtrack_coordinate(result["x_reachable"], x_relations, 2*R, tol)
    ys = _backtrack_coordinate(result["y_reachable"], y_relations, 2*R, tol)
    result["polyline"] = np.column_stack((xs, ys))
    return finish("feasible")
