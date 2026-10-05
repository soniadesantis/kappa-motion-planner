"""Geometry shared by one baseline and its refinement, never a global cache."""

import numpy as np

from .corridor_union import CorridorUnion
from .fillet_safety import axis_aligned_concave_corners


class SequenceGeometry:
    """Own immutable corridor bounds; compute local geometry only when requested.

    The footprint radius and tolerance are part of this context. Circle clipping
    additionally keys on the actual center and radius, so repairs cannot reuse
    stale angular intervals. Logical records still keep their own door indices.
    """

    def __init__(self, bounds, footprint_radius, tol, *, overlaps=None):
        self.bounds = tuple(tuple(b) for b in bounds)
        self.r, self.tol = footprint_radius, tol
        self.overlaps = (tuple(overlaps) if overlaps is not None else
                         tuple((max(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), min(a[3], b[3]))
                               for a, b in zip(self.bounds, self.bounds[1:])))
        self.eroded = tuple((a+self.r, b-self.r, c+self.r, d-self.r)
                            for a, b, c, d in self.bounds)
        self._corners, self._intersections, self._circle_intervals, self._unions = {}, {}, {}, {}
        self._tangents = {}
        self._capsules = {}

    def matches(self, bounds, footprint_radius, tol):
        return self.r == footprint_radius and self.tol == tol and self.bounds == tuple(map(tuple, bounds))

    def corners(self, j):
        if j not in self._corners:
            corners = axis_aligned_concave_corners(*self.bounds[j:j+2], self.tol)
            corners.setflags(write=False)
            self._corners[j] = corners
        return self._corners[j]

    def intersections(self, j, compute):
        if j not in self._intersections:
            points, segments = compute(*self.bounds[j:j+2], self.tol)
            points.setflags(write=False)
            for segment in segments:
                for endpoint in segment:
                    endpoint.setflags(write=False)
            self._intersections[j] = points, segments
        return self._intersections[j]

    def circle_intervals(self, j, center, radius, compute):
        key = (j, float(center[0]), float(center[1]), radius)
        if key not in self._circle_intervals:
            self._circle_intervals[key] = compute(center, radius, self.eroded[j:j+2], tol=self.tol)
        return self._circle_intervals[key]

    def union(self, first, last):
        key = first, last
        if key not in self._unions:
            self._unions[key] = CorridorUnion(self.bounds[first:last+2])
        return self._unions[key]

    def tangents(self, first, second, compute):
        """Share ordinary directed contact geometry between repair and graph.

        Coincident-circle critical contacts are constructed by their callers.
        Quarter/green admissibility is checked separately, so a changed local
        region cannot inherit an old acceptance decision from this cache.
        """
        def key(node):
            return (node['index'], *node['center'], node['radius'], node['turn'])
        pair = key(first), key(second)
        if pair not in self._tangents:
            self._tangents[pair] = compute(first, second, self.tol,
                                           restrict_quarters=False, compute_quarter_angles=False)
        return self._tangents[pair]

    def contains_capsule(self, first, last, start, end):
        """Reuse footprint checks when a later graph keeps the same straight link."""
        key = first, last, *start, *end
        if key not in self._capsules:
            self._capsules[key] = self.union(first, last).contains_capsule(start, end, self.r, tol=self.tol)
        return self._capsules[key]


def sequence_geometry(data, footprint_radius, tol):
    """Reuse a construction's context, or build one for external diagnostic data."""
    geometry = getattr(data, '_geometry', None)
    if geometry is None or not geometry.matches(data.corridor_bounds, footprint_radius, tol):
        geometry = SequenceGeometry(data.corridor_bounds, footprint_radius, tol)
    return geometry


def corridor_boundary_intersections(first, second, tol):
    """Return distinct point intersections and shared boundary segments.

    Constant-size edge intersection calculation on axis-aligned rectangles.
    Shared edges are retained explicitly, not reduced to an arbitrary endpoint.
    """
    def edges(b):
        return [(0,x,b[2],b[3]) for x in b[:2]]+[(1,y,b[0],b[1]) for y in b[2:]]
    points, segments = [], []
    def add(point):
        point = np.asarray(point,dtype=float)
        if not any(np.max(abs(point-old)) <= tol for old in points):
            points.append(point)
    for axis,a,lo,hi in edges(first):
        for other,b,low,high in edges(second):
            if axis != other:
                if lo-tol <= b <= hi+tol and low-tol <= a <= high+tol:
                    point = np.empty(2)
                    point[axis],point[other] = a,b
                    add(point)
            elif abs(a-b) <= tol:
                left,right = max(lo,low),min(hi,high)
                if left > right+tol:
                    continue
                start,end = np.empty(2),np.empty(2)
                start[axis] = end[axis] = (a+b)/2
                start[1-axis],end[1-axis] = left,right
                if right-left > tol:
                    segments.append((start,end))
                    add(start); add(end)
                else:
                    add((start+end)/2)
    return np.asarray(points).reshape(-1,2), segments
