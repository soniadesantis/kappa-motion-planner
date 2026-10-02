"""Analytic containment of a segment's circular footprint in rectangle unions.

Internal rectangle seams are removed before measuring clearance. The exposed
boundary includes concave corners, hole boundaries and disconnected components.
No polygon buffering, rasterization or path sampling is used.
"""

import numpy as np


def _line_box_intervals(start, end, low, high):
    """Vectorized slab clipping of one segment against closed rectangle bounds."""
    entry, exit = np.zeros(len(low)), np.ones(len(low))
    for axis, delta in enumerate(end-start):
        if delta == 0:
            outside = (start[axis] < low[:, axis]) | (start[axis] > high[:, axis])
            entry[outside], exit[outside] = np.inf, -np.inf
        else:
            a = (low[:, axis]-start[axis])/delta
            b = (high[:, axis]-start[axis])/delta
            entry = np.maximum(entry, np.minimum(a, b))
            exit = np.minimum(exit, np.maximum(a, b))
    return entry, exit


def _exposed_boundary(bounds):
    """Rectangle union boundary from exact coordinate compression, O(n^2) space.

    Rectangle-edge coordinates partition the plane into cells on which union
    membership is constant. Difference-array updates count coverage; boundaries
    are precisely the interfaces between occupied and unoccupied cells. Merge
    contiguous collinear interfaces to keep later distance checks small.
    """
    xs, ys = np.unique(bounds[:, :2]), np.unique(bounds[:, 2:])
    left, right = np.searchsorted(xs, bounds[:, :2]).T
    bottom, top = np.searchsorted(ys, bounds[:, 2:]).T
    counts = np.zeros((len(xs), len(ys)), dtype=int)
    for i, j, value in ((left, bottom, 1), (right, bottom, -1),
                        (left, top, -1), (right, top, 1)):
        np.add.at(counts, (i, j), value)
    occupied = counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0
    vertical = np.empty((len(xs), len(ys)-1), dtype=bool)
    horizontal = np.empty((len(ys), len(xs)-1), dtype=bool)
    vertical[0], vertical[-1] = occupied[0], occupied[-1]
    vertical[1:-1] = occupied[:-1] != occupied[1:]
    horizontal[0], horizontal[-1] = occupied[:, 0], occupied[:, -1]
    horizontal[1:-1] = (occupied[:, :-1] != occupied[:, 1:]).T
    pieces = []
    for mask, fixed, along, axis in ((vertical, xs, ys, 0),
                                    (horizontal, ys, xs, 1)):
        starts_mask, ends_mask = mask.copy(), mask.copy()
        starts_mask[:, 1:] &= ~mask[:, :-1]
        ends_mask[:, :-1] &= ~mask[:, 1:]
        rows, starts = np.nonzero(starts_mask)
        _, ends = np.nonzero(ends_mask)
        edges = np.empty((len(starts), 2, 2))
        edges[:, :, axis] = fixed[rows, None]
        edges[:, 0, 1-axis], edges[:, 1, 1-axis] = along[starts], along[ends+1]
        pieces.append(edges)
    boundary = np.concatenate(pieces)
    boundary.setflags(write=False)
    return boundary


def _segment_boundary_distance_squared(start, end, boundary):
    """Exact segment-to-boundary distance, including crossings and point links."""
    a, b = boundary[:, 0], boundary[:, 1]
    entry, exit = _line_box_intervals(start, end, np.minimum(a, b), np.maximum(a, b))
    if np.any(entry <= exit):
        return 0.
    edge = b-a
    edge_length_squared = np.einsum('ij,ij->i', edge, edge)
    distances = []
    for point in (start, end):
        t = np.clip(np.einsum('ij,ij->i', point-a, edge)/edge_length_squared, 0., 1.)
        delta = point-(a+t[:, None]*edge)
        distances.append(np.min(np.einsum('ij,ij->i', delta, delta)))
    delta = end-start
    length_squared = delta @ delta
    for points in (a, b):
        if length_squared:
            t = np.clip((points-start) @ delta/length_squared, 0., 1.)
        else:
            t = np.zeros(len(points))
        offsets = points-(start+t[:, None]*delta)
        distances.append(np.min(np.einsum('ij,ij->i', offsets, offsets)))
    return float(min(distances))


class CorridorUnion:
    """Reusable union of positive axis-aligned rectangles, with a lazy boundary.

    Bounds use (xmin, xmax, ymin, ymax). A single eroded rectangle containing
    both endpoints certifies a capsule immediately by convexity. Otherwise the
    exposed union boundary is built once and cached. Coordinate compression is
    O(n^2); subsequent tests clip n rectangles and measure distance to B exposed
    edges, taking O(n log n+B) work. All caches belong to this union instance.
    """

    def __init__(self, bounds):
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 4 or not len(self.bounds)
                or not np.all(np.isfinite(self.bounds))
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])
                or np.any(self.bounds[:, 2] >= self.bounds[:, 3])):
            raise ValueError('Expected nonempty finite positive rectangle bounds.')
        self.bounds.setflags(write=False)
        self._low, self._high = self.bounds[:, (0, 2)], self.bounds[:, (1, 3)]
        self._eroded_bounds = {}
        self._boundary = None

    @property
    def boundary(self):
        """Read-only exposed edges, including holes; internal seams are absent."""
        if self._boundary is None:
            self._boundary = _exposed_boundary(self.bounds)
        return self._boundary

    def contains_capsule(self, start, end, radius, *, tol=1e-9):
        """Whether [start,end] + B_radius is contained in the original union.

        Centerline containment plus minimum exposed-boundary distance >= radius
        is necessary and sufficient in real arithmetic. Closed-boundary contact
        is allowed. A zero-length link is a disk, and radius zero is a segment.
        tol is a spatial tolerance on containment and required clearance.
        """
        points = np.asarray((start, end), dtype=float)
        if (points.shape != (2, 2) or not np.all(np.isfinite(points))
                or not np.isfinite(radius) or radius < 0
                or not np.isfinite(tol) or tol < 0):
            raise ValueError('Expected finite XY endpoints, radius >= 0 and tol >= 0.')
        low, high = self._low, self._high
        key = radius, tol
        if key not in self._eroded_bounds:
            self._eroded_bounds[key] = low+radius-tol, high-radius+tol, low-tol, high+tol
        eroded_low, eroded_high, expanded_low, expanded_high = self._eroded_bounds[key]
        eroded_inside = np.all((points[:, None, :] >= eroded_low)
                               & (points[:, None, :] <= eroded_high), axis=2)
        if np.any(np.all(eroded_inside, axis=0)):
            return True
        start, end = points
        entry, exit = _line_box_intervals(start, end, expanded_low, expanded_high)
        valid = entry <= exit
        intervals = sorted(zip(entry[valid], exit[valid]))
        covered = 0.
        for a, b in intervals:
            if a > covered:
                return False
            covered = max(covered, b)
            if covered >= 1.:
                break
        if covered < 1.:
            return False
        clearance = max(0., radius-tol)
        distance_squared = _segment_boundary_distance_squared(start, end, self.boundary)
        return distance_squared >= clearance**2
