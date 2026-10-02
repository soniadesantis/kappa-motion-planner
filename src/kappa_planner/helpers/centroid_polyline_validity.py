"""Validate centered overlap-centroid paths without a turning-radius condition."""

from __future__ import annotations

import numpy as np


class InvalidCentroidSequence(ValueError):
    """A sequence failed the nominal centroid-path checks; report explains why."""

    def __init__(self, report):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def check_centroid_polyline(corridors, r, tol=1e-9, *, centerline_mode="either"):
    """Return overlap centroids and a detailed report, with zero-based indices.

    Every rectangle and consecutive overlap must have both dimensions >= 2r.
    Rectangles must be axis-aligned, and each overlap centroid must lie on the
    selected centerline of BOTH rectangles. No corridors are added or modified.

    ``either`` chooses either symmetry axis, consistently across both neighbors
    of a corridor. ``longest`` uses the longer axis and rejects ambiguous squares.
    R is deliberately absent: these checks do not certify fillet feasibility.

    On success, points has shape (N-1, 2), excluding start/end poses. On ANY
    validation failure it has shape (0, 2); individual candidate centroids and
    explanations remain in the report. Repeated centroids and collinear
    backtracking are reported, but can be simplified by the caller: they do not
    invalidate the centered-overlap conditions.
    """
    if len(corridors) < 2:
        raise ValueError("At least two corridors are required.")
    if not np.isfinite(r) or r < 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("r and tol must be finite and nonnegative.")
    if centerline_mode not in ("either", "longest"):
        raise ValueError("centerline_mode must be 'either' or 'longest'.")
    corners = np.asarray([c.corners for c in corridors], dtype=float)
    if corners.shape != (len(corridors), 4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Each corridor must have four finite XY corners.")

    # Compute every bound once; all consecutive overlaps are vectorized.
    low, high = corners.min(axis=1), corners.max(axis=1)
    sizes, centers = high - low, (low + high) / 2
    on_low = abs(corners - low[:, None, :]) <= tol
    on_high = abs(corners - high[:, None, :]) <= tol
    corner_codes = on_high[:, :, 0].astype(int) + 2 * on_high[:, :, 1]
    aligned = (np.all(on_low | on_high, axis=(1, 2))
               & np.all(np.sort(corner_codes, axis=1) == np.arange(4), axis=1)
               & np.all(sizes > tol, axis=1))
    dimensions_ok = aligned & np.all(sizes + tol >= 2*r, axis=1)
    overlap_low = np.maximum(low[:-1], low[1:])
    overlap_high = np.minimum(high[:-1], high[1:])
    overlap_sizes = overlap_high - overlap_low
    exists = (np.all(overlap_sizes > tol, axis=1) & aligned[:-1] & aligned[1:])
    overlap_dimensions_ok = exists & np.all(overlap_sizes + tol >= 2*r, axis=1)
    centroids = (overlap_low + overlap_high) / 2

    # Axis 0 is horizontal (equal y); axis 1 is vertical (equal x).
    on_first = abs((centroids - centers[:-1])[:, ::-1]) <= tol
    on_second = abs((centroids - centers[1:])[:, ::-1]) <= tol
    allowed = np.ones((len(corridors), 2), dtype=bool)
    if centerline_mode == "longest":
        allowed[:, 0] = sizes[:, 0] > sizes[:, 1] + tol
        allowed[:, 1] = sizes[:, 1] > sizes[:, 0] + tol
    allowed[:-1] &= on_first & exists[:, None]
    allowed[1:] &= on_second & exists[:, None]
    allowed &= aligned[:, None]
    axes = np.full(len(corridors), -1, dtype=int)
    for index in range(len(corridors)):
        candidates = np.flatnonzero(allowed[index])
        if len(candidates):
            axes[index] = candidates[np.argmax(sizes[index, candidates])]

    errors, corridor_checks, overlap_checks = [], [], []
    for index, (size, axis) in enumerate(zip(sizes, axes)):
        if not aligned[index]:
            errors.append(f"Corridor {index}: not a positive axis-aligned rectangle")
        elif not dimensions_ok[index]:
            errors.append(f"Corridor {index}: dimensions smaller than 2r={2*r:g}")
        if aligned[index] and axis < 0:
            errors.append(f"Corridor {index}: no consistent admissible centerline")
        corridor_checks.append(dict(
            corridor=index, dx=float(size[0]), dy=float(size[1]),
            axis_aligned=bool(aligned[index]), ok=bool(dimensions_ok[index]),
            centerline=("horizontal", "vertical")[axis] if axis >= 0 else None,
        ))
    for index in range(len(corridors)-1):
        first_axis, second_axis = axes[index:index+2]
        first_ok = bool(exists[index] and first_axis >= 0 and on_first[index, first_axis])
        second_ok = bool(exists[index] and second_axis >= 0
                         and on_second[index, second_axis])
        centered = first_ok and second_ok
        if not exists[index]:
            errors.append(f"Pair ({index}, {index+1}): no valid positive-area overlap")
        elif not overlap_dimensions_ok[index]:
            errors.append(f"Pair ({index}, {index+1}): overlap dimensions smaller than 2r")
        if exists[index] and not centered:
            errors.append(f"Pair ({index}, {index+1}): centroid is not on both centerlines")
        overlap_checks.append(dict(
            pair=(index, index+1), exists=bool(exists[index]),
            dimensions_ok=bool(overlap_dimensions_ok[index]),
            dx=float(overlap_sizes[index, 0]), dy=float(overlap_sizes[index, 1]),
            centroid=centroids[index].copy() if exists[index] else None,
            on_first_centerline=first_ok, on_second_centerline=second_ok,
            centered_overlap=centered,
        ))

    turns = []
    if np.all(exists):
        segments = np.diff(centroids, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        for index, (u, v) in enumerate(zip(segments, segments[1:]), start=1):
            degenerate = bool(min(lengths[index-1:index+1]) <= tol)
            cross = float(u[0]*v[1] - u[1]*v[0])
            tau = int(np.sign(cross)) if abs(cross) > tol else 0
            turns.append(dict(
                vertex=index, point=centroids[index].copy(), tau=tau,
                degenerate=degenerate,
                reversal=bool(not degenerate and tau == 0 and u @ v < 0),
            ))
    report = dict(
        valid=not errors, errors=errors, centerline_mode=centerline_mode,
        axis_aligned_ok=bool(np.all(aligned)),
        corridor_dimensions_ok=bool(np.all(dimensions_ok)),
        overlap_dimensions_ok=bool(np.all(overlap_dimensions_ok)),
        all_overlap_centroids_on_both_centerlines=all(
            check["centered_overlap"] for check in overlap_checks
        ),
        corridors=corridor_checks, overlaps=overlap_checks, turn_directions=turns,
    )
    return (centroids.copy() if report["valid"] else np.empty((0, 2))), report


def check_orthogonal_polyline(corridors, r, tol=1e-9):
    """Use centered centroids when possible, otherwise align overlap waypoints.

    Retains the axis-alignment, positive-overlap, and 2r dimension checks of
    check_centroid_polyline. A waypoint stays inside each consecutive overlap.
    Through each intermediate corridor, the two waypoints share x or y; the
    longer corridor axis is preferred, but either is allowed. No auxiliary
    rectangles are inserted and no turning-radius R condition is imposed.

    Equal-coordinate constraints propagate across an entire horizontal or
    vertical run. Dynamic programming finds feasible runs, minimizing use of
    shorter corridor axes, then number of runs. Each shared coordinate is the
    midpoint of the intersection of ALL overlap ranges in that run. This avoids
    independently moving a waypoint twice to satisfy its two neighbors.

    Returns (points, report), with an empty (0, 2) array on failure. The original
    centroid/centerline fields remain diagnostic; report['valid'] describes the
    new waypoint construction. Start/end points are added by the caller.
    """
    points, report = check_centroid_polyline(corridors, r, tol)
    report["waypoint_method"] = "centroids"
    report["alignment_runs"] = []
    if report["valid"]:
        report["waypoints"] = points.copy()
        return points, report

    basic_valid = (report["axis_aligned_ok"] and report["corridor_dimensions_ok"]
                   and report["overlap_dimensions_ok"])
    if not basic_valid:
        # Centerline failures are diagnostic only under the relaxed rule.
        report["errors"] = []
        for item in report["corridors"]:
            if not item["axis_aligned"]:
                report["errors"].append(
                    f"Corridor {item['corridor']}: not a positive axis-aligned rectangle"
                )
            elif not item["ok"]:
                report["errors"].append(
                    f"Corridor {item['corridor']}: dimensions smaller than 2r={2*r:g}"
                )
        for item in report["overlaps"]:
            if not item["exists"]:
                report["errors"].append(f"Pair {item['pair']}: no valid positive-area overlap")
            elif not item["dimensions_ok"]:
                report["errors"].append(f"Pair {item['pair']}: overlap dimensions smaller than 2r")
        report["waypoint_method"] = "unavailable"
        report["waypoints"] = np.empty((0, 2))
        return report["waypoints"], report

    corners = np.asarray([c.corners for c in corridors], dtype=float)
    low, high = corners.min(axis=1), corners.max(axis=1)
    overlap_low = np.maximum(low[:-1], low[1:])
    overlap_high = np.minimum(high[:-1], high[1:])
    sizes = high - low
    count = len(overlap_low)
    points = (overlap_low + overlap_high) / 2
    # A run from overlap a through b uses corridors a+1 through b.
    short_axis = np.column_stack((sizes[1:-1, 0] < sizes[1:-1, 1] - tol,
                                  sizes[1:-1, 1] < sizes[1:-1, 0] - tol))
    penalties = np.vstack([np.zeros(2, dtype=int), np.cumsum(short_axis, axis=0)])
    costs, parents = {}, {}
    for end in range(1, count):
        for axis in (0, 1):
            transverse = 1-axis
            lower, upper = overlap_low[end, transverse], overlap_high[end, transverse]
            for start in range(end-1, -1, -1):
                lower = max(lower, overlap_low[start, transverse])
                upper = min(upper, overlap_high[start, transverse])
                if lower > upper + tol:
                    break
                previous = (0, 0) if start == 0 else costs.get((start, 1-axis))
                if previous is None:
                    continue
                cost = (previous[0] + int(penalties[end, axis]-penalties[start, axis]),
                        previous[1] + 1)
                if (end, axis) not in costs or cost < costs[end, axis]:
                    costs[end, axis] = cost
                    parents[end, axis] = (start, lower, upper)

    if count > 1:
        choices = [(costs[count-1, axis], axis) for axis in (0, 1)
                   if (count-1, axis) in costs]
        if not choices:
            report["valid"] = False
            report["errors"] = [
                "No globally consistent common x/y ranges connect all overlaps "
                "with one axis-aligned segment per intermediate corridor."
            ]
            report["waypoint_method"] = "unavailable"
            report["turn_directions"] = []
            report["waypoints"] = np.empty((0, 2))
            return report["waypoints"], report
        _, axis = min(choices)
        end = count-1
        runs = []
        while end > 0:
            start, lower, upper = parents[end, axis]
            coordinate = (lower + upper) / 2
            points[start:end+1, 1-axis] = coordinate
            runs.append(dict(
                first_overlap=start, last_overlap=end,
                axis=("horizontal", "vertical")[axis],
                coordinate=("y", "x")[axis], common_range=(float(lower), float(upper)),
                value=float(coordinate),
            ))
            for corridor_index in range(start+1, end+1):
                report["corridors"][corridor_index]["alignment_axis"] = (
                    "horizontal", "vertical"
                )[axis]
            end, axis = start, 1-axis
        report["alignment_runs"] = runs[::-1]

    report["valid"] = True
    report["errors"] = []
    report["waypoint_method"] = "common_range_midpoints"
    report["waypoints"] = points.copy()
    # Report turns of the chosen waypoints, not the superseded centroid path.
    report["turn_directions"] = []
    for index in range(1, len(points)-1):
        u, v = points[index]-points[index-1], points[index+1]-points[index]
        cross = float(u[0]*v[1]-u[1]*v[0])
        tau = int(np.sign(cross)) if abs(cross) > tol else 0
        degenerate = bool(min(np.linalg.norm(u), np.linalg.norm(v)) <= tol)
        report["turn_directions"].append(dict(
            vertex=index, point=points[index].copy(), tau=tau, degenerate=degenerate,
            reversal=bool(not degenerate and tau == 0 and u @ v < 0),
        ))
    return points, report
