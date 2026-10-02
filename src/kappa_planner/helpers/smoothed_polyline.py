"""Global waypoint compatibility for the revised fillet-admissible regions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from .fillet_safety import region_margin
from .fillet_backtracking import backtrack_fillet_waypoints


def solve_fillet_waypoints(nominal, regions, R, tol=1e-8, *, max_backtracking_attempts=128):
    """Try cheap fillet-aware backtracking before the existing joint solver."""
    quick = backtrack_fillet_waypoints(nominal, regions, R, tol,
                                      max_attempts=max_backtracking_attempts)
    if quick['feasible']:
        return dict(quick, selection_method='MIDPOINT_BACKTRACKING')
    result = _solve_fillet_waypoints_joint(nominal, regions, R, tol)
    return dict(result, selection_method='JOINT_SOLVER_FALLBACK',
                backtracking_attempts=quick['backtracking_attempts'],
                backtracking_reason=quick['reason'])


def _solve_fillet_waypoints_joint(nominal, regions, R, tol=1e-8):
    """Re-select a complete chain, never independent local representative points.

    Unique nominal H/V axes imply disjoint longitudinal door intervals, hence
    fixed signs. Shared transverse coordinates are eliminated analytically.
    The revised corner regions are convex on their prerequisite boxes. A
    minimum-slack convex search uses SLSQP; every accepted witness is checked.
    Failed searches are called incompatible only if a linear outer relaxation
    is infeasible (up to LP solver tolerance); otherwise status is unresolved.
    """
    initial, doors, axes = nominal['polyline'], nominal['doors'], nominal['directions']
    n = len(doors)
    if len(regions) != n:
        raise ValueError('Expected one optional fillet region per door.')

    def failure(status, reason):
        return dict(feasible=False, status=status, reason=reason, points=None)

    if any(region is not None and region['empty'] for region in regions):
        return failure('NO_SAFE_FILLET_REGION', 'At least one local fillet-admissible region is empty.')
    groups = np.empty((n, 2), dtype=int)
    lower, upper, seed = [], [], []
    for k, key in enumerate(('x', 'y')):
        for j, door in enumerate(doors):
            lo, hi = door[key]
            if regions[j] is not None:
                lo, hi = max(lo, regions[j]['low'][k]), min(hi, regions[j]['high'][k])
            if j and axes[j-1] == ('V' if k == 0 else 'H'):
                g = groups[j-1, k]
                lower[g], upper[g] = max(lower[g], lo), min(upper[g], hi)
            else:
                g = len(lower)
                lower.append(lo)
                upper.append(hi)
                seed.append(initial[j, k])
            groups[j, k] = g
    lower, upper, seed = map(lambda x: np.asarray(x, dtype=float), (lower, upper, seed))
    if np.any(lower > upper+tol):
        return failure('FILLET_REGIONS_GLOBALLY_INCOMPATIBLE', 'Shared-coordinate ranges do not intersect.')
    near = lower > upper
    lower[near] = upper[near] = (lower[near]+upper[near])/2
    linear = np.zeros((n-1, len(lower)))
    for j, axis in enumerate(axes):
        k = 0 if axis == 'H' else 1
        sign = np.sign(initial[j+1, k]-initial[j, k])
        linear[j, groups[j+1, k]] += sign
        linear[j, groups[j, k]] -= sign
    bounds = list(zip(lower, upper))
    lp = linprog(np.zeros(len(lower)), A_ub=-linear, b_ub=np.full(n-1, -2*R),
                 bounds=bounds, method='highs')
    if lp.status == 2:
        return failure('FILLET_REGIONS_GLOBALLY_INCOMPATIBLE',
                       'Reduced waypoint ranges cannot satisfy alignment and 2R spacing.')
    if not lp.success:
        return failure('NUMERICAL_SEARCH_UNRESOLVED', 'The linear compatibility search did not converge.')
    turns = [j for j, region in enumerate(regions) if region is not None and region['radius'] > 0]
    seed = np.clip(seed, lower, upper)
    if np.any(linear @ seed < 2*R-tol):
        seed = lp.x
    free = upper > lower
    base = (lower+upper)/2

    def unpack(z):
        values = base.copy()
        values[free] = z[:-1]
        return values

    def curvature(values):
        points = values[groups]
        margins, gradients = [], []
        for j in turns:
            region = regions[j]
            positive = np.maximum((points[j]+region['offset']) @ region['frame'], 0)
            norm = np.linalg.norm(positive)
            margins.append(region['radius']-norm)
            gradient = np.zeros(len(base))
            if norm > 0:
                gradient[groups[j]] = -(region['frame'] @ (positive/norm))
            gradients.append(gradient)
        return np.asarray(margins), np.asarray(gradients).reshape(-1, len(base))

    optimizer_message = 'Linear constraints only.'
    values = seed
    if turns:
        initial_margins, _ = curvature(seed)
        z0 = np.r_[seed[free], max(0., -float(initial_margins.min()))]
        result = minimize(lambda z: z[-1], z0, method='SLSQP',
            jac=lambda z: np.r_[np.zeros(free.sum()), 1.],
            bounds=list(zip(lower[free], upper[free]))+[(0., None)],
            constraints=[
                dict(type='ineq', fun=lambda z: linear @ unpack(z)-2*R,
                     jac=lambda z: np.column_stack((linear[:, free], np.zeros(n-1)))),
                dict(type='ineq', fun=lambda z: curvature(unpack(z))[0]+z[-1],
                     jac=lambda z: np.column_stack((curvature(unpack(z))[1][:, free], np.ones(len(turns)))))],
            options=dict(ftol=1e-12, maxiter=1000))
        values = unpack(result.x)
        optimizer_message = str(result.message)
    points = values[groups]
    violation = max(0., float(np.max(lower-values)), float(np.max(values-upper)),
                    float(np.max(2*R-linear @ values)))
    for j, region in enumerate(regions):
        if region is not None:
            violation = max(violation, -float(region_margin(points[j], region)))
    if np.all(np.isfinite(points)) and np.isfinite(violation) and violation <= tol:
        return dict(feasible=True, status='FEASIBLE', reason='Compatible fillet-admissible waypoints found.',
                    points=points, max_violation=violation, optimizer_message=optimizer_message)
    # Supporting planes to convex constraints are necessary conditions. Their
    # infeasibility can reject a chain without mistaking optimizer failure for proof.
    margins, gradients = curvature(values)
    if np.all(np.isfinite(gradients)) and np.all(np.isfinite(values)):
        outer = linprog(np.zeros(len(base)), A_ub=np.vstack((-linear, -gradients)),
                        b_ub=np.r_[np.full(n-1, -2*R), margins-gradients @ values],
                        bounds=bounds, method='highs')
        if outer.status == 2:
            return failure('FILLET_REGIONS_GLOBALLY_INCOMPATIBLE',
                           'A linear outer relaxation of the coupled fillet regions is infeasible '
                           '(to numerical solver tolerance).')
    return failure('NUMERICAL_SEARCH_UNRESOLVED',
                   'No validated witness found; infeasibility has not been established.')
