"""Deterministic randomized checks against direct existential slice queries.

The oracle never extrudes or translates inequalities. It asks whether the
original predecessor slice contains ANY coordinate satisfying the separation.
Boundary, projection, conversion and reconstruction checks complement point
queries; randomized agreement is evidence, not a proof of completeness.
"""

import unittest
from math import sqrt
from types import SimpleNamespace

import numpy as np

from kappa_planner.helpers.fillet_reachability import (
    RoundedCornerConstraint, _make_set, _local_set, propagate_fillet_regions,
)
from kappa_planner.helpers.fillet_safety import region_contains_point


def direct_slice(bounds, constraints, axis, fixed):
    """Solve the original rounded inequalities on a line, without projection."""
    if not bounds[2*axis] <= fixed <= bounds[2*axis+1]:
        return None
    free = 1-axis
    low, high = bounds[2*free:2*free+2]
    for g in constraints:
        component = max(0., g.signs[axis]*(fixed-g.center[axis]))
        if component > g.radius:
            return None
        extent = sqrt(max(0., (g.radius-component)*(g.radius+component)))
        boundary = g.center[free]+g.signs[free]*extent
        if g.signs[free] == 1:
            high = min(high, boundary)
        else:
            low = max(low, boundary)
    return None if low > high else (low, high)


class FilletReachabilityStressTest(unittest.TestCase):
    def test_64000_membership_queries_against_existential_predecessors(self):
        rng = np.random.default_rng(20261003)
        directions = ('right', 'left', 'up', 'down')
        vectors = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        checks = 0
        for case in range(160):
            scale = 10.**rng.uniform(-3, 3)
            tol = scale*1e-10
            R = .5*scale
            point = rng.uniform(-20, 20, size=2)*scale
            boxes, regions, passages = [], [], []
            for j in range(6):
                if j:
                    d = int(rng.integers(4))
                    passages.append(directions[d])
                    point = point+vectors[d]*scale*rng.uniform(1., 2.)
                low = point-rng.uniform(.1, 3, size=2)*scale
                high = point+rng.uniform(.1, 3, size=2)*scale
                box = (low[0], high[0], low[1], high[1])
                boxes.append(box)
                center = point+rng.uniform(-1, 1, size=2)*scale
                signs = rng.choice([-1, 1], size=2)
                radius = np.linalg.norm(np.maximum(signs*(point-center), 0))+.1*scale
                # Alternate diagonal and exchanged-axis local frames.
                frame = np.diag(signs.astype(float))
                if case % 2:
                    frame = frame[:, ::-1]
                regions.append(dict(low=low, high=high, offset=-center,
                                    frame=frame, radius=radius, empty=False))
            data = SimpleNamespace(feasible=True, safe_overlaps=boxes,
                                   x_reachable=[b[:2] for b in boxes],
                                   y_reachable=[b[2:] for b in boxes],
                                   passage_directions=passages)
            result = propagate_fillet_regions(data, regions, R, tol=tol)
            self.assertTrue(result.feasible, (case, result.status))
            for j, current in enumerate(result.reachable_sets):
                local_bounds, local_constraints = _local_set(data, regions, j)
                # Check conversion directly against the original matrix predicate.
                for q in rng.uniform(regions[j]['low'], regions[j]['high'], (10, 2)):
                    converted = all(g.contains(q, tol) for g in local_constraints)
                    self.assertEqual(converted, region_contains_point(q, regions[j], tol))
                q = result.polyline[j]
                self.assertTrue(current.contains(q, tol))
                self.assertTrue(region_contains_point(q, regions[j], tol))
                if not j:
                    continue
                previous = result.reachable_sets[j-1]
                axis = 0 if passages[j-1] in ('right', 'left') else 1
                sign = 1 if passages[j-1] in ('right', 'up') else -1
                self.assertAlmostEqual(result.polyline[j, 1-axis], result.polyline[j-1, 1-axis])
                self.assertGreaterEqual(sign*(q[axis]-result.polyline[j-1, axis]), 2*R-tol)
                a,b,c,d = current.bounds
                probes = np.vstack((rng.uniform([a,c], [b,d], (40,2)),
                                    rng.uniform(regions[j]['low'], regions[j]['high'], (40,2))))
                for q in probes:
                    interval = direct_slice(previous.bounds, previous.constraints, 1-axis, q[1-axis])
                    expected = (region_contains_point(q, regions[j], 0.)
                                and interval is not None
                                and (q[axis] >= interval[0]+2*R if sign > 0
                                     else q[axis] <= interval[1]-2*R))
                    self.assertEqual(current.contains(q, 0.), bool(expected),
                                     (case, j, passages[j-1], q, interval))
                    checks += 1
        self.assertEqual(checks, 64000)

    def test_raw_projection_and_slices_with_many_constraints(self):
        rng = np.random.default_rng(728)
        for count in (1, 2, 4, 8, 16):
            for _ in range(15):
                constraints = []
                for k in range(count):
                    center = rng.uniform(-1, 1, 2)
                    signs = rng.choice([-1, 1], 2)
                    radius = np.linalg.norm(np.maximum(-signs*center, 0))+rng.uniform(.05, 1)
                    constraints.append(RoundedCornerConstraint(tuple(center), tuple(signs), radius))
                result = _make_set((-2, 2, -2, 2), constraints, 1e-10)
                self.assertIsNotNone(result)
                self.assertTrue(result.contains((0, 0)))
                self.assertTrue(result.contains(result.witness))
                for axis in (0, 1):
                    # A reported extremum must have a nonempty original slice.
                    for value in result.bounds[2*axis:2*axis+2]:
                        self.assertIsNotNone(direct_slice((-2, 2, -2, 2), constraints,
                                                         axis, value-1e-12 if value > 0 else value+1e-12))
                    for value in rng.uniform(-2, 2, 80):
                        expected = direct_slice((-2, 2, -2, 2), constraints, axis, value)
                        actual = result.slice_interval(axis, value, 0.)
                        self.assertEqual(expected is None, actual is None)
                        if expected is not None:
                            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.)


if __name__ == '__main__':
    unittest.main()
