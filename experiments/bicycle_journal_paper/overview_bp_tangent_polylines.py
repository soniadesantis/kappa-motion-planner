"""Plot directed-tangent polylines, bypassing blocks at consecutive intersections.

Use --keep-all to reproduce the original raw construction. Green/red vertices indicate
literal membership in the existing D_j, not certification of a new fillet.
"""
import argparse
import csv
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
import numpy as np

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence
from kappa_planner.helpers.bp_tangent_polyline import build_bp_tangent_polyline


def plot_tangent_polyline(corridors, geometry, placement, number, *, skip_intersections=True):
    result = build_bp_tangent_polyline(placement, geometry, skip_intersections=skip_intersections)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for c in corridors:
        points = np.vstack((c.corners, c.corners[0]))
        ax.plot(*points.T, color='.7', lw=.8)
    ax.plot(*geometry['polyline'].T, '--', color='.65', lw=1, label='Baseline polyline')
    for c in placement['sequence']:
        skipped = c.index in result['skipped']
        ax.add_patch(Circle((c.center.x, c.center.y), c.radius, fill=False,
                            ec='.7' if skipped else '.55', lw=.8, ls='--' if skipped else '-'))
        ax.plot(c.center.x, c.center.y, 'x' if skipped else '+', color='.4', ms=5)
        if skipped:
            side = 1 if c.index % 2 else -1
            ax.annotate(f'O{c.index+1}', (c.center.x, c.center.y),
                        xytext=(side*7, -12), textcoords='offset points', fontsize=7,
                        ha='left' if side > 0 else 'right', color='.45')
    for j, link in enumerate(result['tangents']):
        t = link['tangent']
        if t is not None:
            ax.plot(*np.array([t['start'], t['end']]).T, color='#e69f00', lw=2,
                    label='Directed circle tangents' if j == 0 else None)
            ax.plot(*np.array([t['start'], t['end']]).T, '.', color='#e69f00', ms=4)
    ax.plot(*result['polyline'].T, '-', color='#0072b2', lw=1.3,
            label='Tangent-line intersection polyline')
    for inside, color, label in ((True, '#009e73', r'Vertex in $D_j$'),
                                  (False, '#c62828', r'Vertex outside $D_j$')):
        points = [v['point'] for v in result['vertices'] if v['in_Dj'] is inside]
        if points:
            ax.scatter(*np.array(points).T, s=35, c=color, zorder=6, label=label)
    for v in result['vertices']:
        if v['point'] is not None:
            side = 1 if v['circle_index'] % 2 else -1
            ax.annotate(f"p{v['circle_index']+1}", v['point'], xytext=(side*7, 8),
                        textcoords='offset points', fontsize=8,
                        ha='left' if side > 0 else 'right',
                        color='#009e73' if v['in_Dj'] else '#c62828')
    ax.set(aspect='equal', xlabel='x [m]', ylabel='y [m]',
           title=f'Example {number}: tangent polyline' +
                 (' with intersection-triggered skips' if skip_intersections else ' (all circles)'))
    ax.grid(alpha=.15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    undefined = ', '.join(f'p{i+1}' for i in result['undefined']) or 'none'
    policy = (f"Skipped circles: {', '.join(str(i+1) for i in result['skipped']) or 'none'}\n"
              f"Remaining consecutive intersections: {len(result['remaining_intersections'])}\n"
              'Only segment intersections trigger skips.\n'
              'D_j membership does not trigger skips.\n') if skip_intersections else 'All circles retained; no intersection rejection.\n'
    ax.text(1.02, .4, f"Outside D_j: {len(result['outside_Dj'])}\n"
            f"Undefined vertices: {undefined}\n\n"
            + policy +
            'No fixed-quarter restriction.\n\n'
            'Boundary tangents use bp headings.\n'
            'Coincident circles use the bp\nexit tangent as a convention.\n\n'
            'D_j membership alone does not\ncertify this nonorthogonal polyline.',
            transform=ax.transAxes, va='top', fontsize=9)
    return fig, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-all', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    examples = runpy.run_path(str(here/'examples_maps_polyline.py'))
    output = here/'figures'/'bp_circles'/('tangent_polylines' if args.keep_all else 'tangent_polylines_skipped')
    output.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    with PdfPages(output/'overview.pdf') as pdf:
        for n in examples['EXAMPLE_NUMBERS']:
            c, s, e, v = examples['example_corridor_sequence'](n)
            g = examples['build_trajectory_geometry'](c, v.width/2, v.max_radius,
                                                       start_pose=s, end_pose=e)
            if not g['feasible']:
                continue
            p = build_bp_circle_sequence(c, v, g, shift_same_turn=True, shift_opposite_turn=True)
            fig, result = plot_tangent_polyline(c, g, p, n, skip_intersections=not args.keep_all)
            pdf.savefig(fig)
            if n in (1, 5, 7, 13, 20, 21, 23, 29, 35):
                fig.savefig(output/f'example_{n}.pdf')
            if n in (1, 23):
                fig.savefig(output/f'example_{n}.png', dpi=130)
            plt.close(fig)
            summaries.append(dict(example=n, vertices=len(result['vertices']),
                skipped=';'.join(str(i+1) for i in result['skipped']),
                remaining_intersections=len(result['remaining_intersections']),
                outside_Dj=';'.join(str(i+1) for i in result['outside_Dj']),
                undefined=';'.join(str(i+1) for i in result['undefined'])))
            for a in result['vertices']:
                rows.append(dict(example=n, circle=a['circle_index']+1, bp_vertex=a['bp_vertex']+1,
                    status=a['status'], x=a['point'][0] if a['point'] is not None else '',
                    y=a['point'][1] if a['point'] is not None else '',
                    in_Dj=a['in_Dj'], margin=a['Dj_margin']))
    for name, data in (('vertices.csv', rows), ('summary.csv', summaries)):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print('Defined:', sum(r['status'] == 'DEFINED' for r in rows),
          'outside:', sum(r['in_Dj'] is False for r in rows),
          'undefined:', sum(r['status'] != 'DEFINED' for r in rows))
    print(output/'overview.pdf')
    (output/'README.md').write_text(
        '# Tangent polylines\n\n'
        'Reproduce with `MPLBACKEND=Agg python experiments/bicycle_journal_paper/overview_bp_tangent_polylines.py` '
        '(add `--keep-all` for the original unfiltered construction).\n\n'
        'Only intersections of consecutive finite tangent segments trigger block skipping. '
        'Ordinary shared endpoints are allowed. Each replacement is checked against its preceding '
        'tangent, trying progressively larger blocks without a one-circle limit. '
        'First and last circles stay. Impossible replacements leave explicit unresolved intersections. '
        'Missing tangents do not silently trigger skips.\n\n'
        'D_j membership is diagnostic only; red vertices do not trigger a skip. '
        'D_j is the original overlap of corridors C_j and C_(j+1), eroded by '
        'the vehicle radius r. It excludes the additional endpoint and fillet '
        'restrictions used in A_j. Surviving vertices keep their original door '
        'indices after block skipping. '
        'There is no fixed-quarter restriction or clearance filter. '
        'Only consecutive tangent-segment intersections are checked, not global polyline '
        'self-intersections or intersections of infinite supporting lines.\n\n'
        'The first/last boundary lines use bp headings. Coincident circles use the first '
        "circle's bp exit tangent. Green/red labels use the actual tangent-line intersection "
        'vertex in its original D_j, not the orthogonal implied vertex. Membership alone '
        'does not certify the generally nonorthogonal path. CSV circle indices are one-based.\n')


if __name__ == '__main__':
    main()
