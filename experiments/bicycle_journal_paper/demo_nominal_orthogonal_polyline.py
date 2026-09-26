"""Revised local fillet-condition experiment and nominal propagation demo.

Default: test the canonical OR condition against continuous arc clearance.
Use --demo propagation for the original interval-propagation figures.
Use --output-dir PATH --no-show to save the numerical comparison headlessly.

Original propagation demo:

Run this file directly or with VS Code's Run button. These rectangles are already
safe overlaps; no further erosion is applied. The full nominal checker would
reject the first direction as ambiguous (D1 and D2 share both coordinate ranges).
This teaching demo deliberately starts AFTER direction selection and uses the
same propagation and backtracking functions as the checker. A separate figure
shows actual corridors whose eroded overlaps reproduce the supplied doors.
The checker's direction-inference policy is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.fillet_safety import (
    revised_corner_condition, quarter_arc_points,
    continuous_arc_clearance as _continuous_arc_clearance,
)

from kappa_planner.helpers.nominal_polyline import (
    _backtrack_coordinate,
    _interval_intersection,
    _normalize_intervals,
    _propagate_coordinate,
    check_orthogonal_polyline,
)

TURNING_RADIUS = 1.0
FOOTPRINT_RADIUS = 0.2
COLORS = ['#0072B2', '#D55E00', '#009E73']
PURPLE = '#743CAE'


def make_example():
    return [dict(x=(0., 3.), y=(4., 6.)),
            dict(x=(1., 5.), y=(5., 7.)),
            dict(x=(4., 6.), y=(8., 10.))]


def make_failure_example():
    return [dict(x=(0., 3.), y=(4., 6.)),
            dict(x=(1., 3.), y=(5., 7.)),
            dict(x=(1.2, 1.8), y=(8., 10.))]


def make_corridors(doors, r=FOOTPRINT_RADIUS):
    """Realize these three doors as eroded consecutive rectangle intersections.

    First construct eroded corridors S: extend the end doors vertically and
    take the rectangular hull of each adjacent pair for the middle corridors.
    Expand every S edge by r to obtain the original C. Verify the construction
    using the production overlap code; arbitrary door triples may not admit it.
    """
    first, _, last = doors
    safe = [dict(x=first['x'], y=(first['y'][0]-3, first['y'][1]))]
    for a, b in zip(doors, doors[1:]):
        safe.append({c: (min(a[c][0], b[c][0]), max(a[c][1], b[c][1]))
                     for c in ('x', 'y')})
    safe.append(dict(x=last['x'], y=(last['y'][0], last['y'][1]+3)))
    corridors = []
    for region in safe:
        a, b = region['x']
        c, d = region['y']
        corridors.append(SimpleNamespace(corners=np.array([
            [a-r, c-r], [b+r, c-r], [b+r, d+r], [a-r, d+r]])))
    report = check_orthogonal_polyline(corridors, r, TURNING_RADIUS)
    for expected, actual in zip(doors, report['doors']):
        if actual is None:
            raise ValueError('Corridor construction produced an empty safe overlap.')
        for coordinate in ('x', 'y'):
            np.testing.assert_allclose(actual[coordinate], expected[coordinate], atol=1e-9)
    return corridors


def plot_corridor_examples(success, failure):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), layout='constrained')
    fig.suptitle(f'Corridors realizing the two examples, footprint radius r = {FOOTPRINT_RADIUS:g} m\n'
                 'Colored doors are exactly (Cj ∩ Cj+1) eroded by r. H → V is prescribed.', fontsize=13)
    corridor_colors = ['#555555', '#B47B20', '#7656A6', '#477C72']
    for ax, result, title in zip(axes, (success, failure),
                                ('Feasible with prescribed H → V', 'Spacing failure with prescribed H → V')):
        corridors = make_corridors(result['doors'])
        for j, corridor in enumerate(corridors):
            low, high = corridor.corners.min(axis=0), corridor.corners.max(axis=0)
            box(ax, (low[0], high[0]), (low[1], high[1]),
                facecolor='none', edgecolor=corridor_colors[j], linewidth=2)
            # Label the exposed end for C1/C4, and the right edge for C2/C3.
            if j in (0, 3):
                anchor = ((low[0]+high[0])/2, low[1]+.3 if j == 0 else high[1]-.3)
            else:
                anchor = (high[0]+.15, high[1]-.45)
            ax.text(*anchor, f'C{j+1}', color=corridor_colors[j], weight='bold',
                    ha='center' if j in (0, 3) else 'left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        draw_doors(ax, result['doors'])
        if result['polyline'] is not None:
            ax.plot(*result['polyline'].T, 'o-', color='black', lw=2, label='Recovered polyline')
            ax.legend(loc='lower right')
        ax.set(xlim=(-.6, 7), ylim=(.3, 13.7), title=title)
        ax.set_yticks(range(1, 14, 2))
    return fig


def solve_example(doors=None):
    """Use the production coordinate routines with prescribed H/V relations."""
    doors = make_example() if doors is None else doors
    minimum = 2*TURNING_RADIUS
    relations = dict(x=['separate', 'equal'], y=['equal', 'separate'])
    forward, complete, values = {}, {}, {}
    for coordinate in ('x', 'y'):
        intervals = [door[coordinate] for door in doors]
        rules = relations[coordinate]
        forward[coordinate] = _propagate_coordinate(intervals, rules, minimum)
        # Backward supports distinguish a reachable prefix from a complete path.
        backward = _propagate_coordinate(intervals[::-1], rules[::-1], minimum)[::-1]
        complete[coordinate] = []
        for prefix, suffix in zip(forward[coordinate], backward):
            intersections = [_interval_intersection(a, b) for a in prefix for b in suffix]
            complete[coordinate].append(_normalize_intervals(
                [item for item in intersections if item is not None]))
        values[coordinate] = _backtrack_coordinate(forward[coordinate], rules, minimum)
    feasible = all(values[c] is not None for c in ('x', 'y'))
    points = np.column_stack((values['x'], values['y'])) if feasible else None
    return dict(doors=doors, forward=forward, complete=complete,
                polyline=points, feasible=feasible)


def box(ax, x, y, **style):
    if x[0] == x[1] or y[0] == y[1]:
        ax.plot(x, y, color=style.get('edgecolor', PURPLE), lw=4,
                marker='o', markersize=4, zorder=4)
    else:
        ax.add_patch(Rectangle((x[0], y[0]), x[1]-x[0], y[1]-y[0], **style))


def draw_doors(ax, doors):
    for j, door in enumerate(doors):
        box(ax, door['x'], door['y'], facecolor=COLORS[j], edgecolor=COLORS[j],
            alpha=.12, linestyle='--', linewidth=2)
        ax.annotate(f'D{j+1}', (door['x'][1], door['y'][1]), xytext=(5, 4),
                    textcoords='offset points', color=COLORS[j], weight='bold')
    ax.set(xlim=(-.5, 6.8), ylim=(3.5, 10.8), xlabel='x [m]', ylabel='y [m]')
    ax.set_aspect('equal')
    ax.set_xticks(range(7))
    ax.set_yticks(range(4, 11))
    ax.grid(alpha=.15)


def highlight(ax, xs, ys, color=PURPLE):
    for x in xs:
        for y in ys:
            box(ax, x, y, facecolor=color, edgecolor=color, alpha=.45, linewidth=2)


def plot_steps(result):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout='constrained')
    fig.suptitle('Safe-door propagation with prescribed H → V,  2R = 2 m\n'
                 'Dashed rectangles: supplied safe doors. Solid purple: retained sets.', fontsize=15)
    for ax in axes.flat:
        draw_doors(ax, result['doors'])
    ax = axes[0, 0]
    ax.set_title('1. Start with the three safe rectangles\n'
                 'D1 = [0, 3] × [4, 6]; D2 = [1, 5] × [5, 7]\n'
                 'D3 = [4, 6] × [8, 10]', fontsize=11)
    ax.text(.03, .95, 'Assume p1 → p2 is horizontal.\nAssume p2 → p3 is vertical.',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='0.8'))

    ax = axes[0, 1]
    ax.set_title('2. Horizontal connection: y1 = y2\n'
                 '[4, 6] ∩ [5, 7] = [5, 6]\n'
                 'Both points must use the same y in this band.', fontsize=11)
    for door in result['doors'][:2]:
        highlight(ax, [door['x']], [(5, 6)])
    ax.axhline(5, color=PURPLE, lw=1)
    ax.axhline(6, color=PURPLE, lw=1)

    ax = axes[1, 0]
    ax.set_title('3. Horizontal spacing: |x2 − x1| ≥ 2\n'
                 'x2 ≤ 3 − 2 = 1  OR  x2 ≥ 0 + 2 = 2\n'
                 'In D2, retain {1} ∪ [2, 5]; remove the open gap (1, 2).', fontsize=11)
    highlight(ax, result['forward']['x'][1], result['forward']['y'][1])
    ax.annotate('x2 = 1 survives\n(use x1 = 3)', xy=(1, 5.5), xytext=(0, 8.4),
                arrowprops=dict(arrowstyle='->', color=PURPLE), fontsize=10,
                bbox=dict(facecolor='white', edgecolor='0.8'))
    ax.plot(1.5, 5.5, 'x', color='crimson', markersize=9, mew=2)
    ax.annotate('x2 = 1.5 fails:\nmaximum distance = 1.5 < 2', xy=(1.5, 5.5),
                xytext=(2.1, 4.1), arrowprops=dict(arrowstyle='->', color='crimson'),
                fontsize=9, color='crimson')

    ax = axes[1, 1]
    ax.set_title('4. Vertical connection: x3 = x2\n'
                 '({1} ∪ [2, 5]) ∩ [4, 6] = [4, 5]\n'
                 'Only x2 = x3 in [4, 5] can finish the whole chain.', fontsize=11)
    for j in (1, 2):
        highlight(ax, result['complete']['x'][j], result['complete']['y'][j])
    ax.axvline(4, color=PURPLE, lw=1)
    ax.axvline(5, color=PURPLE, lw=1)
    ax.text(.03, .03, 'The isolated x2 = 1 cannot reach D3.\n'
            'The vertical spacing is also valid:\ny3 ≥ 8 and y2 ≤ 6 give y3 − y2 ≥ 2.',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', edgecolor='0.8'))
    return fig


def plot_solution(result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
    fig.suptitle('Forward sets retain possible prefixes; backtracking chooses a complete path', fontsize=14)
    ax = axes[0]
    rows = [(0, 3), (1, 5), (4, 6)]
    for j, (original, reachable) in enumerate(zip(rows, result['forward']['x'])):
        ax.plot(original, [j, j], color='0.82', lw=14, solid_capstyle='butt')
        for lo, hi in reachable:
            ax.plot([lo, hi], [j, j], color=PURPLE, lw=6, solid_capstyle='butt')
            ax.plot([lo, hi], [j, j], '|', color=PURPLE, markersize=14, mew=2)
        if j == 1:
            ax.plot(1, j, 'o', color=PURPLE, markersize=7)
    xs = result['polyline'][:, 0]
    ax.plot(xs, range(3), 'o--', color='black', lw=1.5)
    ax.set_yticks(range(3), ['D1: [0, 3]', 'D2: {1} ∪ [2, 5]', 'D3: [4, 5]'])
    ax.set(xlim=(-.5, 6.5), ylim=(2.6, -.6), xlabel='x coordinate [m]')
    ax.set_title('Read purple sets downward; recover black points upward', fontsize=11)
    ax.text(0, .55, '|x2 − x1| ≥ 2', bbox=dict(facecolor='white', edgecolor='0.8'))
    ax.text(0, 1.55, 'x3 = x2', bbox=dict(facecolor='white', edgecolor='0.8'))
    ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Line2D([], [], color='0.82', lw=8, label='Original x range'),
                       Line2D([], [], color=PURPLE, lw=5, label='Forward-reachable x set'),
                       Line2D([], [], color='black', marker='o', linestyle='--', label='Backtracked x values')],
              loc='lower left', fontsize=9)
    ax = axes[1]
    draw_doors(ax, result['doors'])
    points = result['polyline']
    ax.plot(*points.T, 'o-', color='black', lw=2.5)
    for j, point in enumerate(points, 1):
        ax.annotate(f'p{j} = ({point[0]:g}, {point[1]:g})', point, xytext=(8, 5),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    ax.set_title('Recovered points in the supplied safe doors\n'
                 'p1 → p2: horizontal, 2 m; p2 → p3: vertical, 3 m', fontsize=11)
    return fig


def plot_failure(result):
    """Show the nonempty doors, the spacing gap, and the empty final support."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), layout='constrained')
    fig.suptitle('Failure example: the 2R spacing removes every possible continuation\n'
                 'Prescribed H → V, 2R = 2 m. All three safe doors are nonempty.', fontsize=14)
    for ax in axes[:2]:
        draw_doors(ax, result['doors'])
        ax.set_xlim(-.4, 3.7)
        ax.set_xticks(range(4))
    axes[0].set_title('1. Supplied safe doors\n'
                      'D1 = [0, 3] × [4, 6]\n'
                      'D2 = [1, 3] × [5, 7]\n'
                      'D3 = [1.2, 1.8] × [8, 10]', fontsize=11)
    ax = axes[1]
    ax.set_title('2. Retain x2 = 1 or x2 ∈ [2, 3]\n'
                 'D3 requires x2 = x3 ∈ [1.2, 1.8].\n'
                 'This entire range is inside the removed gap.', fontsize=11)
    ax.axvspan(1.2, 1.8, color='crimson', alpha=.1)
    ax.axvline(1.2, color='crimson', linestyle=':', lw=1.5)
    ax.axvline(1.8, color='crimson', linestyle=':', lw=1.5)
    highlight(ax, result['forward']['x'][1], result['forward']['y'][1])
    ax.annotate('Removed gap\n1 < x2 < 2', xy=(1.5, 5.5), xytext=(.05, 7.2),
                arrowprops=dict(arrowstyle='->', color='crimson'),
                color='crimson', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none'))
    ax = axes[2]
    for j, door in enumerate(result['doors']):
        ax.plot(door['x'], [j, j], color='0.82', lw=14, solid_capstyle='butt')
        for lo, hi in result['forward']['x'][j]:
            ax.plot([lo, hi], [j, j], color=PURPLE, lw=6, solid_capstyle='butt')
            ax.plot([lo, hi], [j, j], '|', color=PURPLE, markersize=14, mew=2)
            if lo == hi:
                ax.plot(lo, j, 'o', color=PURPLE, markersize=7)
    ax.text(1.5, 2, 'EMPTY', ha='center', va='center', color='crimson', weight='bold',
            bbox=dict(facecolor='white', edgecolor='crimson'))
    ax.text(.02, .55, '|x2 − x1| ≥ 2', bbox=dict(facecolor='white', edgecolor='0.8'))
    ax.text(.02, 1.55, 'x3 = x2', bbox=dict(facecolor='white', edgecolor='0.8'))
    ax.set_yticks(range(3), ['D1', 'D2', 'D3'])
    ax.set(xlim=(-.3, 3.4), ylim=(2.85, -.6), xlabel='x coordinate [m]')
    ax.set_title('3. Forward x sets, read downward\n'
                 '[0, 3] → {1} ∪ [2, 3] → EMPTY\n'
                 'No compatible polyline exists.', fontsize=11)
    ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Line2D([], [], color='0.82', lw=8, label='Original safe x range'),
                       Line2D([], [], color=PURPLE, lw=5, label='Forward-reachable x set')],
              loc='lower left', fontsize=9)
    return fig


def propagation_main(show=True):
    result = solve_example()
    np.testing.assert_allclose(result['forward']['x'][1], [(1, 1), (2, 5)])
    np.testing.assert_allclose(result['forward']['x'][2], [(4, 5)])
    np.testing.assert_allclose(result['complete']['x'][1], [(4, 5)])
    points = result['polyline']
    for point, door in zip(points, result['doors']):
        assert all(door[c][0] <= point[k] <= door[c][1] for k, c in enumerate(('x', 'y')))
    assert points[0, 1] == points[1, 1] and points[1, 0] == points[2, 0]
    assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) >= 2*TURNING_RADIUS)
    print('Propagation demo: supplied safe doors, prescribed H → V (not inferred).')
    print('Forward x sets:', result['forward']['x'])
    print('Forward y sets:', result['forward']['y'])
    print('Complete-path x sets:', result['complete']['x'])
    print('Backtracked polyline:\n', points)
    plot_steps(result)
    plot_solution(result)
    failure = solve_example(make_failure_example())
    np.testing.assert_allclose(failure['forward']['x'][1], [(1, 1), (2, 3)])
    assert failure['forward']['x'][2] == []
    assert failure['forward']['y'][2]  # Failure is specifically in x spacing.
    assert not failure['feasible'] and failure['polyline'] is None
    print('Failure example forward x sets:', failure['forward']['x'])
    print('No valid 2R-spaced H → V polyline exists for the failure example.')
    plot_failure(failure)
    plot_corridor_examples(result, failure)
    if show:
        plt.show()
    else:
        plt.close('all')


def _union_geometry():
    """Read existing union geometry without changing the example-map checker."""
    import importlib.util
    import sys

    name = '_nominal_demo_union_geometry'
    if name not in sys.modules:
        path = Path(__file__).with_name('examples_maps_polyline.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def local_turn_examples():
    definitions = [
        ('L: long arms', [(-6, 0, -2, 0), (-2, 0, -2, 6)], (-2, 0), False),
        ('L: narrow arms', [(-6, 0, -.6, 0), (-.6, 0, -.6, 6)], (-.6, 0), False),
        ('T: left approach', [(-6, 6, -2, 0), (-1, 1, -2, 6)], (-1, 0), False),
        ('L: short arms', [(-2.2, 0, -2, 0), (-2, 0, -2, .3)], (-2, 0), False),
        ('L: reflected right turn', [(-6, 0, -2, 0), (-2, 0, -2, 6)], (-2, 0), True),
        ('L: wide overlap', [(-10, 0, -4, 0), (-4, 0, -4, 10)], (-4, 0), False),
    ]
    examples = []
    for name, bounds, corner, reflected in definitions:
        corridors = []
        for xmin, xmax, ymin, ymax in bounds:
            if reflected:
                xmin, xmax = -xmax, -xmin
            corridors.append(CorridorWorld(ymax-ymin, xmax-xmin,
                                           [(xmin+xmax)/2, (ymin+ymax)/2], 0))
        reflection = np.array([-1., 1.]) if reflected else np.ones(2)
        examples.append(dict(name=name, corridors=corridors,
                             corner=np.array(corner)*reflection,
                             incoming=np.array([1., 0.])*reflection,
                             outgoing=np.array([0., 1.])*reflection))
    examples.append(dict(examples[0], name='L: vertex on curved admissible boundary',
                         boundary_example=True))
    return examples


def continuous_arc_clearance(vertices, incoming, outgoing, R, corridors, boundary):
    return _continuous_arc_clearance(vertices, incoming, outgoing, R, corridors, boundary,
                                     _union_geometry().union_signed_clearance)


def evaluate_local_turn(example, r=FOOTPRINT_RADIUS, R=TURNING_RADIUS, grid_size=61,
                        tol=1e-9):
    """Compare the conjecture with actual safety only on eligible safe vertices.

    Both tangencies must lie in their respective eroded rectangles. Convexity
    and p in the safe overlap then guarantee the two radius-R straight pieces.
    The canonical prerequisites xp,yp>=r are checked as well. Excluded vertices
    are not counted as safe/unsafe: their clearance entries remain NaN.
    """
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError('grid_size must be an integer >= 2.')
    geometry = _union_geometry()
    corridors, u, v, c = (example[k] for k in ('corridors', 'incoming', 'outgoing', 'corner'))
    revised_corner_condition(np.empty((0, 2)), c, u, v, R, r, tol)
    if not geometry.is_union_concave_corner(c, *corridors):
        raise ValueError('Expected a relevant concave corner of the union.')
    corners = np.asarray([corridor.corners for corridor in corridors])
    low = corners.min(axis=1).max(axis=0)+r
    high = corners.max(axis=1).min(axis=0)-r
    if np.any(low > high):
        raise ValueError('Safe overlap is empty.')
    x, y = np.meshgrid(np.linspace(low[0], high[0], grid_size),
                       np.linspace(low[1], high[1], grid_size))
    points = np.unique(np.column_stack((x.ravel(), y.ravel())), axis=0)
    boundary_point = None
    if example.get('boundary_example', False):
        if R < r:
            raise ValueError('The curved-boundary example requires R >= r.')
        # In the canonical frame choose o=((R-r)/sqrt(2), (R-r)/sqrt(2)).
        # This is an exact analytic boundary point, not the nearest grid sample.
        a = (R-r)/np.sqrt(2)
        boundary_point = c+(a-R)*(-u+v)
        if np.any(boundary_point < low-tol) or np.any(boundary_point > high+tol):
            raise ValueError('The selected curved-boundary vertex is outside the safe overlap.')
        points = np.unique(np.vstack((points, boundary_point)), axis=0)
    prediction = revised_corner_condition(points, c, u, v, R, r, tol)
    local_p = np.column_stack(((points-c) @ (-u), (points-c) @ v))
    canonical_ok = np.all(local_p <= -r+tol, axis=1)
    tangent_ok = []
    for tangents, corridor in zip((points-R*u, points+R*v), corridors):
        margin = tangents @ corridor.W[:2]+corridor.W[2]
        tangent_ok.append(np.all(margin <= -r*np.linalg.norm(corridor.W[:2], axis=0)+tol, axis=1))
    eligible = canonical_ok & tangent_ok[0] & tangent_ok[1]
    boundary = geometry.corridor_union_boundary(corridors)
    minimum = np.full(len(points), np.nan)
    witnesses = np.full_like(points, np.nan)
    minimum[eligible], witnesses[eligible] = continuous_arc_clearance(
        points[eligible], u, v, R, corridors, boundary)
    actual = minimum >= r-tol
    predicted = prediction['predicted']
    masks = dict(both_safe=eligible & predicted & actual,
                 false_positive=eligible & predicted & ~actual,
                 false_negative=eligible & ~predicted & actual,
                 both_unsafe=eligible & ~predicted & ~actual)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    candidates = np.flatnonzero(masks['false_positive'] | masks['false_negative'])
    if not len(candidates):
        # Illustrate the newly admitted far-arm cases that failed the old disk rule.
        candidates = np.flatnonzero(masks['both_safe'] & prediction['far'] & ~prediction['circular'])
    if not len(candidates):
        candidates = np.flatnonzero(eligible)
    selected = int(candidates[len(candidates)//2]) if len(candidates) else None
    if boundary_point is not None:
        selected = int(np.argmin(np.linalg.norm(points-boundary_point, axis=1)))
        if not eligible[selected]:
            raise ValueError('The boundary example lacks safe tangent segments.')
    return dict(example=example, r=r, R=R, tol=tol, points=points, low=low, high=high,
                prediction=prediction, eligible=eligible, canonical_ok=canonical_ok,
                incoming_ok=tangent_ok[0], outgoing_ok=tangent_ok[1], actual=actual,
                minimum_clearance=minimum, witnesses=witnesses, masks=masks,
                counts=counts, selected=selected)


_CLASS_STYLES = {
    'both_safe': ('#009E73', 'Predicted safe / actually safe'),
    'false_positive': ('#D55E00', 'Predicted safe / actually unsafe'),
    'false_negative': ('#0072B2', 'Predicted unsafe / actually safe'),
    'both_unsafe': ('#bdbdbd', 'Predicted unsafe / actually unsafe'),
}


def plot_admissible_vertex_region(ax, result, resolution=301):
    """Overlay the proposed p-region, clipped by overlap and tangent constraints.

    A continuous signed margin represents the two half-planes OR disk, then
    intersects it with the prerequisites. Only rendering uses a plotting grid;
    the area is not inferred from the sampled actual arc-safety classifications.
    """
    ex, r, R = result['example'], result['r'], result['R']
    low, high = result['low'], result['high']
    padding = max(float(np.max(high-low))*.025, .02)
    x = np.linspace(low[0]-padding, high[0]+padding, resolution)
    y = np.linspace(low[1]-padding, high[1]+padding, resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    u, v, c = ex['incoming'], ex['outgoing'], ex['corner']
    center = points-R*u+R*v
    local_center = np.stack(((center-c) @ (-u), (center-c) @ v), axis=-1)
    # Positive means inside the union of the two far-arm half-planes and disk.
    margin = np.maximum(-local_center[..., 0], -local_center[..., 1])
    if R >= r:
        margin = np.maximum(margin, R-r-np.linalg.norm(local_center, axis=-1))
    margin = np.minimum(margin, np.min(np.minimum(points-low, high-points), axis=-1))
    local_p = np.stack(((points-c) @ (-u), (points-c) @ v), axis=-1)
    margin = np.minimum(margin, np.min(-r-local_p, axis=-1))
    for tangent, corridor in zip((points-R*u, points+R*v), ex['corridors']):
        normals = corridor.W[:2]
        clearance = -(tangent @ normals+ corridor.W[2])/np.linalg.norm(normals, axis=0)
        margin = np.minimum(margin, np.min(clearance-r, axis=-1))
    if np.max(margin) > 0:
        ax.contourf(x, y, margin, levels=[0, float(margin.max())+1],
                    colors=['#2563eb'], alpha=.30, zorder=2)
        ax.contour(x, y, margin, levels=[0], colors=['#1d4ed8'], linewidths=1.6, zorder=3)
    else:
        ax.text(.02, .02, 'No admissible vertex area', transform=ax.transAxes,
                color='#1d4ed8', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85))


def plot_local_turn_experiment(results):
    geometry = _union_geometry()
    figure, axes = plt.subplots(len(results), 2, figsize=(13, 4*len(results)),
                                squeeze=False, layout='constrained')
    figure.suptitle('Revised local rule: ox ≤ 0 OR oy ≤ 0 OR ‖o‖ ≤ R − r (canonical frame)\n'
                   'Blue area: admissible p positions including safe-overlap and safe-tangent checks.', fontsize=12)
    for (ax, physical), result in zip(axes, results):
        ex, r, R = result['example'], result['r'], result['R']
        excluded = ~result['eligible']
        ax.scatter(*result['points'][excluded].T, marker='x', s=8, color='#d6b5dc',
                   linewidths=.5, label=f'Excluded prerequisites: {excluded.sum()}', rasterized=True)
        for name, (color, label) in _CLASS_STYLES.items():
            ax.scatter(*result['points'][result['masks'][name]].T, s=7, color=color,
                       label=f'{label}: {result["counts"][name]}', rasterized=True)
        ax.set(title=ex['name'], xlabel='vertex p: global x [m]', ylabel='vertex p: global y [m]')
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
        ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(1, 1))
        geometry.plot_union_erosion(physical, ex['corridors'], r, resolution=220)
        low, high = result['low'], result['high']
        box(physical, (low[0], high[0]), (low[1], high[1]),
            facecolor='none', edgecolor=PURPLE, linewidth=1.2, linestyle='--')
        plot_admissible_vertex_region(physical, result)
        j = result['selected']
        if j is None:
            physical.set_title('No eligible vertices; no comparison')
            continue
        p, o, c = result['points'][j], result['prediction']['centers'][j], ex['corner']
        u, v = ex['incoming'], ex['outgoing']
        arc = quarter_arc_points(p, u, v, R, np.linspace(0, np.pi/2, 301))
        ax.plot(*p, '*', color='black', ms=10)
        physical.plot(*np.array([p-R*u, p, p+R*v]).T, ':', color='0.35')
        physical.plot(*arc.T, color='#009E73' if result['actual'][j] else '#D55E00', lw=2, zorder=4)
        for point, label, marker in ((p, 'p', 'o'), (o, 'o', '+'), (c, 'c', 'x')):
            physical.plot(*point, marker, color='black', zorder=5)
            physical.annotate(label, point, xytext=(5, 5), textcoords='offset points')
        physical.add_patch(Circle(result['witnesses'][j], r, fill=False, ec='#D55E00', lw=1.5))
        local = result['prediction']['local_centers'][j]
        physical.set_title(f'Predicted={bool(result["prediction"]["predicted"][j])}, '
                           f'actual={bool(result["actual"][j])}; canonical o=({local[0]:.2f}, {local[1]:.2f})\n'
                           f'Arc clearance={result["minimum_clearance"][j]:.4f} m; required r={r:g}', fontsize=10)
        if ex.get('boundary_example', False):
            physical.set_title('p on the curved admissible boundary: ‖o − c‖ = R − r\n'
                               f'Minimum arc clearance = {result["minimum_clearance"][j]:.6f} m = r',
                               fontsize=10)
            physical.plot(*np.array([o, c, result['witnesses'][j]]).T,
                          '--', color='0.35', lw=1, zorder=4)
        physical.set_xlim(min(low[0], o[0]-R)-.3, max(high[0], o[0]+R)+.3)
        physical.set_ylim(min(low[1], o[1]-R)-.3, max(high[1], o[1]+R)+.3)
        physical.legend(handles=[Patch(facecolor='#a7d9ce', label='Eroded union'),
                                 Patch(facecolor='#2563eb', edgecolor='#1d4ed8', alpha=.3,
                                       label='Admissible vertex area (revised rule)'),
                                 Line2D([], [], color=PURPLE, ls='--', label='Safe overlap'),
                                 Line2D([], [], color='#D55E00', marker='o', mfc='none',
                                        ls='none', label='Radius-r disk at clearance witness')],
                        fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    return figure


def run_local_turn_experiment(grid_size=61, output_dir=None, show=True):
    results = [evaluate_local_turn(example, grid_size=grid_size) for example in local_turn_examples()]
    summary = []
    for result in results:
        eligible = result['eligible']
        prediction = result['prediction']
        item = dict(name=result['example']['name'], r=result['r'], R=result['R'],
                    sampled=len(eligible), eligible=int(eligible.sum()),
                    excluded=int((~eligible).sum()), counts=result['counts'],
                    far_arm_only=int((eligible & prediction['far'] & ~prediction['circular']).sum()))
        print(item)
        if result['example'].get('boundary_example', False):
            j = result['selected']
            item['boundary_vertex'] = dict(p=result['points'][j].tolist(),
                center=prediction['centers'][j].tolist(),
                corner=result['example']['corner'].tolist(),
                center_corner_distance=float(np.linalg.norm(
                    prediction['centers'][j]-result['example']['corner'])),
                arc_clearance=float(result['minimum_clearance'][j]))
            print('  Boundary vertex:', item['boundary_vertex'])
        mismatches = np.flatnonzero(result['masks']['false_positive'] | result['masks']['false_negative'])
        if len(mismatches):
            j = int(mismatches[0])
            item['mismatch_example'] = dict(p=result['points'][j].tolist(),
                canonical_center=prediction['local_centers'][j].tolist(),
                predicted=bool(prediction['predicted'][j]), actual=bool(result['actual'][j]),
                clearance=float(result['minimum_clearance'][j]))
            print('  Mismatch:', item['mismatch_example'])
        summary.append(item)
    figure = plot_local_turn_experiment(results)
    boundary_figure = plot_local_turn_experiment([results[-1]])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir/'revised_fillet_condition.png', dpi=150)
        figure.savefig(output_dir/'revised_fillet_condition.pdf')
        boundary_figure.savefig(output_dir/'boundary_vertex.png', dpi=170)
        boundary_figure.savefig(output_dir/'boundary_vertex.pdf')
        (output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        with (output_dir/'vertices.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['example', 'px', 'py', 'ox_local', 'oy_local', 'canonical_ok',
                             'incoming_tangent_ok', 'outgoing_tangent_ok', 'eligible',
                             'far_branch', 'circular_branch', 'predicted', 'actual', 'clearance'])
            for result in results:
                prediction = result['prediction']
                for j, p in enumerate(result['points']):
                    eligible = result['eligible'][j]
                    writer.writerow([result['example']['name'], *p, *prediction['local_centers'][j],
                        bool(result['canonical_ok'][j]), bool(result['incoming_ok'][j]),
                        bool(result['outgoing_ok'][j]), bool(eligible), bool(prediction['far'][j]),
                        bool(prediction['circular'][j]), bool(prediction['predicted'][j]),
                        bool(result['actual'][j]) if eligible else '',
                        result['minimum_clearance'][j] if eligible else ''])
        print(f'Saved comparison to {output_dir}')
    print('Finite vertex sampling tests the revised conjecture; it does not prove it.')
    if show:
        plt.show()
    else:
        plt.close(figure)
        plt.close(boundary_figure)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', choices=('fillet', 'propagation'), default='fillet')
    parser.add_argument('--grid-size', type=int, default=61)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.demo == 'propagation':
        propagation_main(show=not args.no_show)
    else:
        run_local_turn_experiment(args.grid_size, args.output_dir, not args.no_show)


if __name__ == '__main__':
    main()
