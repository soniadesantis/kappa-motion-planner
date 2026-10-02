"""Pose-coordinate endpoint slices for the orthogonal waypoint chain."""

from time import perf_counter

import numpy as np

from .nominal_polyline import _bounds, _propagate_coordinate, _backtrack_coordinate


def extend_nominal_with_pose_coordinates(nominal, corridors, start_pose, end_pose, r, R, tol=1e-9,
                                        *, exact_start=False, exact_end=False):
    """Add endpoints on pose-coordinate lines inside the eroded end corridors.

    The corridor axis picks the copied pose coordinate; headings are not used.
    Endpoints have no fillet region. Door vertices, including the first/last
    overlaps, will receive any necessary local fillet constraints downstream.
    An exact endpoint fixes both coordinates to the input pose position.
    """
    if not nominal['feasible']:
        return nominal
    started = perf_counter()
    result = dict(nominal, doors=[dict(d) for d in nominal['doors']],
                  directions=list(nominal['directions']), polyline=None,
                  feasible=False, corridor_offset=-1, endpoint_constraints=True,
                  endpoint_modes=('exact' if exact_start else 'projected',
                                  'exact' if exact_end else 'projected'))

    def fail(status, reason):
        result.update(status=status, reason=reason,
                      computation_time=nominal['computation_time']+perf_counter()-started)
        return result

    endpoints, axes = [], []
    for at_start, corridor, pose in ((True, corridors[0], start_pose),
                                     (False, corridors[-1], end_pose)):
        pose = np.asarray(pose, dtype=float)
        if pose.ndim != 1 or len(pose) < 2 or not np.all(np.isfinite(pose[:2])):
            raise ValueError('Endpoint poses need finite x and y coordinates.')
        stored = nominal.get('corridor_bounds')
        a, b, c, d = stored[0 if at_start else -1] if stored else _bounds(corridor, tol)
        low, high = np.array([a+r, c+r]), np.array([b-r, d-r])
        k = int(np.argmax(np.abs(corridor.unit_vector)))
        axis, key = ('H', 'x') if k == 0 else ('V', 'y')
        if np.any(low > high+tol) or not low[k]-tol <= pose[k] <= high[k]+tol:
            return fail('endpoint_outside_corridor',
                        f"{'Start' if at_start else 'End'} pose-coordinate line misses the eroded corridor.")
        low[k] = high[k] = pose[k]
        exact = exact_start if at_start else exact_end
        if exact:
            t = 1-k
            if not low[t]-tol <= pose[t] <= high[t]+tol:
                return fail('endpoint_outside_corridor',
                            f"Exact {'start' if at_start else 'end'} position is outside the eroded corridor.")
            low[t] = high[t] = pose[t]
        endpoint = dict(x=(float(low[0]), float(high[0])), y=(float(low[1]), float(high[1])))
        door = result['doors'][0 if at_start else -1]
        lo, hi = door[key]
        # Orient travel toward the first overlap, or from the last overlap.
        sign = np.sign((lo+hi)/2-pose[k]) if at_start else np.sign(pose[k]-(lo+hi)/2)
        adjacent_axis = nominal['directions'][0 if at_start else -1]
        if axis == adjacent_axis:
            a, b = nominal['polyline'][:2] if at_start else nominal['polyline'][-2:]
            sign = np.sign(b[k]-a[k])  # Continue straight, never reverse.
        if sign == 0:
            sign = np.sign(corridor.unit_vector[k])
        if (at_start and sign > 0) or (not at_start and sign < 0):
            lo = max(lo, pose[k]+2*R)
        else:
            hi = min(hi, pose[k]-2*R)
        if lo > hi+tol:
            return fail('endpoint_spacing',
                        f"{'Initial' if at_start else 'Final'} segment cannot reach length 2R in the required direction.")
        door[key] = (lo, hi) if lo <= hi else ((lo+hi)/2,)*2
        endpoints.append(endpoint)
        axes.append(axis)
    result['doors'] = [endpoints[0]]+result['doors']+[endpoints[1]]
    result['directions'] = [axes[0]]+result['directions']+[axes[1]]
    coordinates = []
    for name, axis in (('x', 'H'), ('y', 'V')):
        relations = ['separate' if direction == axis else 'equal' for direction in result['directions']]
        reachable = _propagate_coordinate([door[name] for door in result['doors']], relations, 2*R, tol)
        result[name+'_reachable'] = reachable
        if not reachable[-1]:
            return fail('endpoint_spacing', 'No complete orthogonal sequence satisfies pose coordinates and 2R spacing.')
        coordinates.append(_backtrack_coordinate(reachable, relations, 2*R, tol))
    points = np.column_stack(coordinates)
    delta = np.diff(points, axis=0)
    for a, b in zip(delta, delta[1:]):
        if a @ b < -tol and abs(a[0]*b[1]-a[1]*b[0]) <= tol:
            return fail('endpoint_reversal', 'An endpoint extension would create a 180-degree reversal.')
    result.update(polyline=points, feasible=True, status='feasible', reason=None,
                  endpoint_axes=axes, computation_time=nominal['computation_time']+perf_counter()-started)
    return result
