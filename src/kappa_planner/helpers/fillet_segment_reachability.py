"""Certified extreme-segment alternative to generic curved-set propagation.

For convex S with exact coordinate extrema, its upstream extreme face spans
its full transverse projection iff BOTH endpoints of that proposed face lie
in S. Convexity then contains their entire segment. This is a finite analytic
certificate of the user's hypothesis; sampling slices is unnecessary.

Why the nominal orthogonal corridor model has this property (real arithmetic):
A local turning region is a box intersected with a single positive-part rounded
inequality. Its signs come from frame=(-incoming, outgoing). On the outgoing
axis the sign equals the passage sign. Moving a point upstream decreases its
positive-part component, so cannot violate that inequality. The box supplies
the constant upstream extreme. A straight junction has no rounded constraint.
Starting with A_0, segment extrusion imposes only box bounds on A_{j+1}.
Inductively each reachable set remains a box intersected with just its own
local rounded inequality, with the same outgoing monotonicity. Its full
upstream extreme segment therefore exists. Endpoint fillets obey the same
argument; the last set has no outgoing internal propagation to certify.

This does NOT hold for arbitrary rounded sets. The generic solver supports
those too. This alternative rejects a failed certificate, rather than silently
overestimating reachability or incorrectly reporting geometric infeasibility.
The existing analytic intersection routine is reused; curved A_j boundaries
still determine true projection extrema. Floating-point checks use tol.
"""
from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np

from .fillet_reachability import (
    FilletReachability, FilletReachableSet, _DIRECTIONS, _local_set, _make_set,
    _midpoint_polyline,
)


def intersect_baseline_region(bounds, constraint=None, *, tol=1e-9):
    """Intersect a box with its one local rounded constraint in constant time.

    Baseline A_j and R_j have at most one rounded inequality. In its signed
    frame it is hypot(max(qx,0), max(qy,0)) <= rho, which is monotone in both
    coordinates. The minimum corner decides emptiness; fixing either minimum
    coordinate gives the other exact projection maximum. No event enumeration,
    certificate, or inherited-constraint validation is needed in this model.
    """
    xmin, xmax, ymin, ymax = bounds
    if xmin > xmax+tol or ymin > ymax+tol:
        return None
    if xmin > xmax:
        xmin = xmax = (xmin+xmax)/2
    if ymin > ymax:
        ymin = ymax = (ymin+ymax)/2
    if constraint is None:
        return FilletReachableSet((xmin, xmax, ymin, ymax), (),
                                  ((xmin+xmax)/2, (ymin+ymax)/2))
    cx, cy = constraint.center
    sx, sy = constraint.signs
    rho = constraint.radius
    qx_min, qx_max = (xmin-cx, xmax-cx) if sx > 0 else (cx-xmax, cx-xmin)
    qy_min, qy_max = (ymin-cy, ymax-cy) if sy > 0 else (cy-ymax, cy-ymin)
    positive_x, positive_y = max(qx_min, 0.), max(qy_min, 0.)
    if hypot(positive_x, positive_y) > rho+tol:
        return None
    qx_max = min(qx_max, sqrt(max(0., rho*rho-positive_y*positive_y)))
    qy_max = min(qy_max, sqrt(max(0., rho*rho-positive_x*positive_x)))
    if sx > 0:
        xmax = cx+qx_max
    else:
        xmin = cx-qx_max
    if sy > 0:
        ymax = cy+qy_max
    else:
        ymin = cy-qy_max
    if xmin > xmax:
        xmin = xmax = (xmin+xmax)/2
    if ymin > ymax:
        ymin = ymax = (ymin+ymax)/2
    witness = (xmin if sx > 0 else xmax, ymin if sy > 0 else ymax)
    active = () if hypot(max(qx_max, 0.), max(qy_max, 0.)) <= rho else (constraint,)
    return FilletReachableSet((xmin, xmax, ymin, ymax), active, witness)


@dataclass(frozen=True)
class ExtremeSegmentCertificate:
    direction: str
    endpoints: tuple
    valid: bool


class SegmentHypothesisError(ValueError):
    """The supplied reachable set does not have the required extreme segment."""

    def __init__(self, waypoint, certificate):
        self.waypoint = waypoint
        self.certificate = certificate
        super().__init__(f'Extreme-segment hypothesis failed at waypoint {waypoint} '
                         f'for passage {certificate.direction}: {certificate.endpoints}')


def extreme_segment_certificate(reachable, direction, *, tol=1e-9):
    """Certify the whole upstream segment using convexity and its endpoints.

    Requires a nonempty convex FilletReachableSet with tight projection bounds,
    as returned by _make_set. A user-constructed loose box is not sufficient.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('Require finite tol>=0.')
    axis, sign = _DIRECTIONS[direction]
    transverse = 1-axis
    extreme = reachable.bounds[2*axis+(sign < 0)]
    endpoints = []
    for value in reachable.bounds[2*transverse:2*transverse+2]:
        point = [0., 0.]
        point[axis], point[transverse] = extreme, value
        endpoints.append(tuple(point))
    endpoints = tuple(endpoints)
    return ExtremeSegmentCertificate(direction, endpoints,
                                     all(reachable.contains(p, tol) for p in endpoints))



def _half_strip_intersection_bounds(bounds, previous, direction, R):
    bounds = list(bounds)
    axis, sign = _DIRECTIONS[direction]
    transverse = 1-axis
    bounds[2*transverse] = max(bounds[2*transverse], previous.bounds[2*transverse])
    bounds[2*transverse+1] = min(bounds[2*transverse+1], previous.bounds[2*transverse+1])
    if sign > 0:
        bounds[2*axis] = max(bounds[2*axis], previous.bounds[2*axis]+2*R)
    else:
        bounds[2*axis+1] = min(bounds[2*axis+1], previous.bounds[2*axis+1]-2*R)
    return bounds


def propagate_reachable_set(A_next, R_prev, direction, R, *, tol=1e-9):
    """One certified segment -> translated half-strip -> A_next intersection.

    A_next and R_prev are analytic FilletReachableSet objects. Returns the
    next set, or None for an empty intersection. This matches the proposed
    stepwise API; indices in the full-sequence API remain zero-based.
    """
    if not np.isfinite(R) or R <= 0:
        raise ValueError('Require finite R>0.')
    certificate = extreme_segment_certificate(R_prev, direction, tol=tol)
    if not certificate.valid:
        raise SegmentHypothesisError(None, certificate)
    if A_next is None:
        return None
    bounds = _half_strip_intersection_bounds(A_next.bounds, R_prev, direction, R)
    return _make_set(bounds, A_next.constraints, tol)


def propagate_fillet_regions_segment(data, regions, R, *, tol=1e-9):
    """Propagate certified extreme segments, intersect A_j, reconstruct a chain.

    Same inputs and result type as propagate_fillet_regions. Every outgoing
    segment is certified before use. A failed structural hypothesis raises
    SegmentHypothesisError and is distinct from an empty reachable set.
    The generic reference implementation is left unchanged.
    """
    if not data.feasible or len(regions) != len(data.safe_overlaps):
        raise ValueError('Require feasible internal analysis and one region per waypoint.')
    if not np.isfinite(R) or R <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError('Require finite R>0 and tol>=0.')
    reachable = []
    for j in range(len(regions)):
        certificate = None
        if j:
            certificate = extreme_segment_certificate(
                reachable[-1], data.passage_directions[j-1], tol=tol)
            if not certificate.valid:
                raise SegmentHypothesisError(j-1, certificate)
        bounds, constraints = _local_set(data, regions, j)
        if bounds is None:
            current = None
        else:
            if j:
                bounds = _half_strip_intersection_bounds(
                    bounds, reachable[-1], certificate.direction, R)
                # No inherited curved inequalities: only the certified half-strip.
            current = _make_set(bounds, constraints, tol)
        reachable.append(current)
        if current is None:
            return FilletReachability(False, 'fillet_reachability_empty', tuple(reachable),
                                      empty_waypoint=j)
    # Propagation still uses certified extreme segments. Reconstruction selects
    # central points by cheap analytic slices, shared with the exact reference.
    points = _midpoint_polyline(reachable, data.passage_directions, R, tol)
    if points is None:
        return FilletReachability(False, 'exact_reconstruction_unresolved', tuple(reachable))
    return FilletReachability(True, 'feasible', tuple(reachable), points)
