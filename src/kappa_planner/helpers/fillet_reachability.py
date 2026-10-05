"""Analytic full-set propagation for the existing local fillet regions.

Each set is a box intersected with rounded-corner inequalities
    max(s_x (x-c_x), 0)^2 + max(s_y (y-c_y), 0)^2 <= rho^2.
These are convex and have only straight and quarter-circular boundaries.

A rightward passage takes F + (2R, 0) + {(t,0): t>=0}, then intersects
the next A. At each y, F has a closed interval slice. Its left endpoint
is the maximum of the lower bounds imposed by the box and inequalities
whose x sign is negative. Thus extrusion discards the inequalities with
positive x sign, shifts the others, and retains the EXACT y projection.
The other three directions are symmetric. This proves closure of the
representation and implements the set-valued recurrence without a grid,
polygonal approximation, candidate budget, or numerical optimizer.

Projection extrema occur at box vertices, line/arc or arc/arc intersections,
or coordinate extrema of arcs (their endpoints). Enumerating these events
and testing membership gives the exact projection in real arithmetic.
Indeed, the bounded convex intersection has a boundary assembled from portions
of those lines/arcs. A coordinate extremum on the interior of a straight portion
also occurs at its endpoints; an interior arc extremum is an axis endpoint.
Every remaining endpoint is an intersection of boundary pieces. The straight
rays of rounded inequalities are already included as tightened box walls.
This covers lower-dimensional segments and singletons as well. A nonempty
compact set must have such an extremum, so absence of feasible events proves
emptiness. With k retained constraints, O(k^2) events each require O(k)
membership work: O(k^3) worst-case time per intersection, not a linear-time claim.
All computations here use floating point and the supplied comparison tolerance.
Exactness is relative to the supplied A regions, not a completeness claim
for arbitrary collision-free paths or for the underlying local safety lemma.
"""

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RoundedCornerConstraint:
    center: tuple
    signs: tuple
    radius: float

    def contains(self, point, tol):
        qx = max(self.signs[0] * (point[0] - self.center[0]), 0.)
        qy = max(self.signs[1] * (point[1] - self.center[1]), 0.)
        return hypot(qx, qy) <= self.radius + tol

    def on_quarter(self, point, tol):
        return all(s * (p-c) >= -tol for s, p, c in zip(self.signs, point, self.center))


@dataclass(frozen=True)
class FilletReachableSet:
    """A nonempty F_j, with its exact bounding box and analytic constraints.

    bounds are (xmin, xmax, ymin, ymax); witness belongs to the full set.
    slice_interval fixes one world coordinate and returns the other interval.
    """

    bounds: tuple
    constraints: tuple
    witness: tuple

    def contains(self, point, tol=1e-9):
        a, b, c, d = self.bounds
        return (a-tol <= point[0] <= b+tol and c-tol <= point[1] <= d+tol
                and all(g.contains(point, tol) for g in self.constraints))

    def slice_interval(self, fixed_axis, value, tol=1e-9):
        free = 1-fixed_axis
        if not self.bounds[2*fixed_axis]-tol <= value <= self.bounds[2*fixed_axis+1]+tol:
            return None
        low, high = self.bounds[2*free:2*free+2]
        for g in self.constraints:
            qfixed = max(g.signs[fixed_axis] * (value-g.center[fixed_axis]), 0.)
            if qfixed > g.radius + tol:
                return None
            remaining = sqrt(max(0., g.radius*g.radius-qfixed*qfixed))
            bound = g.center[free] + g.signs[free] * remaining
            if g.signs[free] > 0:
                high = min(high, bound)
            else:
                low = max(low, bound)
        if low > high+tol:
            return None
        return (low, high) if low <= high else ((low+high)/2,)*2


@dataclass(frozen=True)
class FilletReachability:
    feasible: bool
    status: str
    reachable_sets: tuple
    polyline: Optional[np.ndarray] = None
    empty_waypoint: Optional[int] = None


def _circle_intersections(first, second, tol):
    """Finite circle/circle boundary events, including a tangency point."""
    dx = second.center[0]-first.center[0]
    dy = second.center[1]-first.center[1]
    distance = hypot(dx, dy)
    r1, r2 = first.radius, second.radius
    if (distance == 0 or distance > r1+r2+tol
            or distance < abs(r1-r2)-tol):
        return
    # Factor the difference of squares: nearly equal radii/centers otherwise
    # lose their small separation through cancellation. Do not identify distinct
    # centers using tol: tiny circles and thin lenses can still be meaningful.
    along = .5*(distance+(r1-r2)*(r1+r2)/distance)
    height_squared = (r1-along)*(r1+along)
    if height_squared < -tol * max(1., r1, r2):
        return
    height = sqrt(max(0., height_squared))
    x = first.center[0] + along*dx/distance
    y = first.center[1] + along*dy/distance
    for sign in (-1, 1):
        point = (x-sign*height*dy/distance, y+sign*height*dx/distance)
        if first.on_quarter(point, tol) and second.on_quarter(point, tol):
            yield point


def _make_set(bounds, constraints, tol):
    """Intersect analytically, decide emptiness and tighten all four extrema."""
    bounds = list(map(float, bounds))
    # The straight rays of each rounded constraint give global coordinate bounds.
    for g in constraints:
        for axis in (0, 1):
            bound = g.center[axis]+g.signs[axis]*g.radius
            if g.signs[axis] > 0:
                bounds[2*axis+1] = min(bounds[2*axis+1], bound)
            else:
                bounds[2*axis] = max(bounds[2*axis], bound)
    for axis in (0, 1):
        low, high = bounds[2*axis:2*axis+2]
        if low > high+tol:
            return None
        if low > high:
            bounds[2*axis:2*axis+2] = [(low+high)/2]*2
    # Remove constraints satisfied by the entire box, avoiding repeated work.
    active = []
    for g in constraints:
        farthest = tuple(bounds[2*k+(g.signs[k] > 0)] for k in (0, 1))
        if not g.contains(farthest, 0.):
            active.append(g)
    constraints = tuple(dict.fromkeys(active))
    xmin, xmax, ymin, ymax = bounds
    candidates = [(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    for g in constraints:
        cx, cy = g.center
        sx, sy = g.signs
        radius = g.radius
        candidates.extend([(cx+sx*radius, cy), (cx, cy+sy*radius)])
        for fixed_axis in (0, 1):
            free = 1-fixed_axis
            for value in bounds[2*fixed_axis:2*fixed_axis+2]:
                local = g.signs[fixed_axis]*(value-g.center[fixed_axis])
                if local < -tol or local > radius+tol:
                    continue
                remaining = sqrt(max(0., radius*radius-local*local))
                point = [0., 0.]
                point[fixed_axis] = value
                point[free] = g.center[free]+g.signs[free]*remaining
                candidates.append(tuple(point))
    for i, first in enumerate(constraints):
        for second in constraints[i+1:]:
            candidates.extend(_circle_intersections(first, second, tol))
    feasible = []
    for point in candidates:
        x, y = point
        if not (xmin-tol <= x <= xmax+tol and ymin-tol <= y <= ymax+tol):
            continue
        point = (min(xmax, max(xmin, x)), min(ymax, max(ymin, y)))
        if all(g.contains(point, tol) for g in constraints):
            feasible.append(point)
    if not feasible:
        return None
    xs, ys = zip(*feasible)
    # A convex combination of feasible events gives a deterministic witness.
    witness = (sum(xs)/len(xs), sum(ys)/len(ys))
    return FilletReachableSet((min(xs), max(xs), min(ys), max(ys)), constraints, witness)


def _local_set(data, regions, j):
    """Return a local box and its optional rounded constraint, without sampling."""
    x, y = data.x_reachable[j], data.y_reachable[j]
    bounds = [x[0], x[1], y[0], y[1]]
    region = regions[j]
    constraints = []
    if region is not None:
        if region['empty']:
            return None, ()
        for k in (0, 1):
            bounds[2*k] = max(bounds[2*k], region['low'][k])
            bounds[2*k+1] = min(bounds[2*k+1], region['high'][k])
        if region['radius'] >= 0:
            frame = np.asarray(region['frame'], dtype=float)
            rounded = np.rint(frame)
            if (frame.shape != (2, 2) or not np.all(np.isfinite(frame))
                    or not np.allclose(frame, rounded, atol=1e-12, rtol=0.)
                    or not np.all(np.sum(abs(rounded), axis=0) == 1)
                    or not np.all(np.sum(abs(rounded), axis=1) == 1)):
                raise ValueError('Fillet propagation requires a signed-permutation local frame.')
            signs = tuple(map(int, rounded.sum(axis=1)))
            constraints.append(RoundedCornerConstraint(tuple(-region['offset']), signs,
                                                        float(region['radius'])))
    return bounds, tuple(constraints)


_DIRECTIONS = {'right': (0, 1), 'left': (0, -1), 'up': (1, 1), 'down': (1, -1)}


def _midpoint_polyline(reachable, directions, R, tol, *, verify=True):
    """Start centrally in the last set, then bisect compatible predecessor slices.

    A curved set need not contain its bounding-box midpoint. In that case fix
    the midpoint of its x projection and bisect the resulting feasible y slice.
    Reconstruction queries analytic intervals only; no sampled search is used.
    """
    points = np.empty((len(reachable), 2))
    last = reachable[-1]
    xmin, xmax, ymin, ymax = last.bounds
    terminal = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
    if not last.contains(terminal, tol):
        interval = last.slice_interval(0, terminal[0], tol)
        if interval is None:
            return None
        terminal[1] = sum(interval)/2
    if verify and not last.contains(terminal, tol):
        return None
    points[-1] = terminal
    for j in range(len(reachable)-2, -1, -1):
        axis, sign = _DIRECTIONS[directions[j]]
        transverse = 1-axis
        fixed = points[j+1, transverse]
        interval = reachable[j].slice_interval(transverse, fixed, tol)
        if interval is None:
            return None
        low, high = interval
        if sign > 0:
            high = min(high, points[j+1, axis]-2*R)
        else:
            low = max(low, points[j+1, axis]+2*R)
        if low > high+tol:
            return None
        points[j, transverse] = fixed
        points[j, axis] = (low+high)/2
        if verify and not reachable[j].contains(points[j], tol):
            return None
    return points


def propagate_fillet_regions(data, regions, R, *, tol=1e-9):
    """Compute every F_j and reconstruct a chain iff terminal F is nonempty.

    Requires a successful exact internal-polyline analysis and one optional A
    per waypoint. None means A_j=D_j. Intersecting with the existing rectangular
    forward reachable sets is redundant mathematically but cheaply prunes boxes.
    The output sets retain the complete curved regions, not bounding boxes alone.
    Empty propagation certifies incompatibility of this local-region model,
    to numerical tolerance. Reconstruction roundoff failure stays unresolved.
    """
    if not data.feasible or len(regions) != len(data.safe_overlaps):
        raise ValueError('Require feasible internal analysis and one region per waypoint.')
    if not np.isfinite(R) or R <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError('Require finite R>0 and tol>=0.')
    reachable = []
    for j in range(len(regions)):
        bounds, constraints = _local_set(data, regions, j)
        if bounds is None:
            current = None
        else:
            if j:
                previous = reachable[-1]
                axis, sign = _DIRECTIONS[data.passage_directions[j-1]]
                transverse = 1-axis
                bounds[2*transverse] = max(bounds[2*transverse], previous.bounds[2*transverse])
                bounds[2*transverse+1] = min(bounds[2*transverse+1], previous.bounds[2*transverse+1])
                if sign > 0:
                    bounds[2*axis] = max(bounds[2*axis], previous.bounds[2*axis]+2*R)
                else:
                    bounds[2*axis+1] = min(bounds[2*axis+1], previous.bounds[2*axis+1]-2*R)
                shifted = []
                for g in previous.constraints:
                    if g.signs[axis] == -sign:
                        center = list(g.center)
                        center[axis] += sign*2*R
                        shifted.append(RoundedCornerConstraint(tuple(center), g.signs, g.radius))
                constraints = (*constraints, *shifted)
            current = _make_set(bounds, constraints, tol)
        reachable.append(current)
        if current is None:
            return FilletReachability(False, 'fillet_reachability_empty', tuple(reachable),
                                      empty_waypoint=j)
    points = _midpoint_polyline(reachable, data.passage_directions, R, tol)
    if points is None:
        return FilletReachability(False, 'exact_reconstruction_unresolved', tuple(reachable))
    return FilletReachability(True, 'feasible', tuple(reachable), points)
