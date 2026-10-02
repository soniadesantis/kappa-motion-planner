"""Unrestricted tangent-line intersections and safe-overlap D_j membership.

Diagnostic only: optionally bypass blocks at consecutive tangent intersections.
No clearance filtering or quarter-contact restriction. Circles and bp are fixed.
"""
from time import perf_counter

import numpy as np

from .tangent_refinement import _directed_tangent_lines
from .bp_tangent_chain import _conflict


def build_bp_tangent_polyline(report, geometry, tol=1e-8, *, skip_intersections=True):
    """Intersect incoming/outgoing supporting tangent lines at each circle.

    First/last boundary lines use the bp entry/exit headings, not pose maneuvers.
    Coincident equal-turn circles use the first circle's bp exit heading as an
    explicit common-tangent convention. Parallel lines yield no unique vertex.
    D_j tests use the actual line-intersection vertex, not o+R*u-R*v. Since these
    vertices need not be orthogonal, membership alone is not a fillet certificate.
    Skips are triggered ONLY by intersecting consecutive finite tangent segments,
    excluding their ordinary common endpoint. Retry progressively larger blocks
    transactionally; keep the first and last circles. D_j is diagnostic only.
    Use original radius-r eroded overlaps from internal_check, without the
    endpoint or fillet restrictions added to the bp's A_j regions. Original
    circle indices identify doors even after skipping or endpoint extension.
    Set skip_intersections=False to reproduce the unfiltered original polyline.
    """
    started = perf_counter()
    regions = geometry['smoothed_check'].get('regions', [])
    circles = list(report['sequence'])
    nodes = [dict(index=c.index, center=np.array([c.center.x, c.center.y]), radius=c.radius,
                  turn=c.turn_direction, region=regions[c.bp_vertex_index]) for c in circles]
    all_nodes = nodes
    circles_by_index = {c.index: c for c in circles}
    cache, events, vertices = {}, [], []
    def link(a, b):
        key = (a['index'], b['index'])
        if key in cache:
            return cache[key]
        choices = _directed_tangent_lines(a, b, tol, restrict_quarters=False)
        tangent = dict(choices[0], kind='line') if choices else None
        cache[key] = dict(pair=key, tangent=tangent,
                             status='DEFINED' if tangent is not None else 'NO_REAL_DIRECTED_TANGENT',
                             coincident_convention=bool(np.linalg.norm(a['center']-b['center']) <= tol
                                                        and a['turn'] == b['turn']))
        return cache[key]
    def crosses(first, second):
        return (first['tangent'] is not None and second['tangent'] is not None
                and _conflict(first['tangent'], second['tangent'], True, tol))
    nodes, tangents = all_nodes[:1], []
    for target in all_nodes[1:]:
        candidate = link(nodes[-1], target)
        if skip_intersections and tangents and crosses(tangents[-1], candidate):
            event = dict(trigger=(tangents[-1]['pair'], candidate['pair']),
                         target=target['index'], accepted=False, attempts=[])
            events.append(event)
            for keep in range(len(nodes)-2, -1, -1):
                replacement = link(nodes[keep], target)
                valid = (replacement['tangent'] is not None and
                         (keep == 0 or (tangents[keep-1]['tangent'] is not None
                                        and not crosses(tangents[keep-1], replacement))))
                event['attempts'].append(dict(pair=replacement['pair'], accepted=valid))
                if valid:
                    event.update(accepted=True, skipped=[n['index'] for n in nodes[keep+1:]],
                                 replacement=replacement['pair'])
                    nodes, tangents = nodes[:keep+1], tangents[:keep]
                    candidate = replacement
                    break
        nodes.append(target)
        tangents.append(candidate)
    circles = [circles_by_index[n['index']] for n in nodes]
    def boundary(node, direction):
        normal = np.array([-direction[1], direction[0]])
        p = node['center']-node['turn']*node['radius']*normal
        return dict(start=p, end=p, direction=direction.copy())
    lines = ([boundary(nodes[0], nodes[0]['region']['incoming'])]
             + [t['tangent'] for t in tangents]
             + [boundary(nodes[-1], nodes[-1]['region']['outgoing'])]) if nodes else []
    polyline = np.full((len(nodes)+2, 2), np.nan)
    if nodes:
        polyline[0], polyline[-1] = lines[0]['start'], lines[-1]['end']
    for j, (node, circle) in enumerate(zip(nodes, circles)):
        a, b = lines[j:j+2]
        row = dict(circle_index=node['index'], bp_vertex=circle.bp_vertex_index,
                   point=None, in_Dj=None, Dj_margin=None, status='MISSING_TANGENT')
        if a is not None and b is not None:
            u, v = a['direction'], b['direction']
            delta = b['start']-a['start']
            det = float(u[0]*v[1]-u[1]*v[0])
            if abs(det) <= 1e-12:
                offset = abs(float(delta[0]*u[1]-delta[1]*u[0]))
                row['status'] = 'COINCIDENT_LINES' if offset <= tol else 'PARALLEL_LINES'
            else:
                t = float(delta[0]*v[1]-delta[1]*v[0])/det
                point = a['start']+t*u
                if 'internal_check' in geometry:
                    door = geometry['internal_check']['doors'][circle.index]
                else:
                    # Fallback for callers supplying just the bp check. Its
                    # unextended door list follows bp vertex indices.
                    if geometry['check'].get('endpoint_constraints'):
                        raise ValueError('Original D_j doors are required in internal_check; '
                                         'endpoint-restricted doors are not interchangeable.')
                    door = geometry['check']['doors'][circle.bp_vertex_index]
                margin = (-np.inf if door is None else float(min(
                    point[0]-door['x'][0], door['x'][1]-point[0],
                    point[1]-door['y'][0], door['y'][1]-point[1])))
                row.update(status='DEFINED', point=point, in_Dj=margin >= -tol, Dj_margin=margin)
                polyline[j+1] = point
        vertices.append(row)
    return dict(tangents=tangents, boundary_lines=([lines[0], lines[-1]] if nodes else []),
                vertices=vertices, polyline=polyline,
                outside_Dj=[v['circle_index'] for v in vertices if v['in_Dj'] is False],
                undefined=[v['circle_index'] for v in vertices if v['point'] is None],
                retained=[n['index'] for n in nodes],
                skipped=sorted(set(circles_by_index)-{n['index'] for n in nodes}), events=events,
                remaining_intersections=[(a['pair'], b['pair']) for a, b in zip(tangents, tangents[1:])
                                         if crosses(a, b)] if skip_intersections else [],
                missing_tangents=[t['pair'] for t in tangents if t['tangent'] is None],
                intersections_checked=skip_intersections, intersection_scope='consecutive finite tangent segments',
                clearance_checked=False, pair_evaluations=len(cache),
                computation_time=perf_counter()-started)
