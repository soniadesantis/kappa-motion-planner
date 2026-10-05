"""Plot exact polylines and fillets with segments close to the 2R bound.

Run in VS Code or use --no-show. Cases refer to the seed-37 stress experiment;
each selected segment has fillets at both ends and a short remaining straight.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from kappa_planner.baseline_construction import compute_filleted_baseline_exact


DIRECTORY = Path(__file__).parent / 'results/random_baseline/exact_propagation'
CASES = ((5, 29, 2), (8, 64, 2), (12, 96, 2), (20, 34, 12))
STYLE = {'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 14,
         'axes.titlesize': 16, 'axes.labelsize': 15, 'pdf.fonttype': 42}


def draw_case(ax, n, case, segment, result, robot):
    for a, b, c, d in case['bounds']:
        ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor='#EDF5FC',
                               edgecolor='#555555', linewidth=.6, zorder=1))
    for a, b, c, d in result.feasibility.safe_overlaps:
        ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor='#0072B2',
                               alpha=.2, edgecolor='none', zorder=2))
    points = result.polyline
    ax.plot(*points.T, '--o', color='#64748b', linewidth=1,
            markersize=3, alpha=.7, zorder=3)
    j = segment-1
    ax.plot(*points[j:j+2].T, '--o', color='#D55E00', linewidth=1.7,
            markersize=5, zorder=4)
    for k in range(len(points)-1):
        first, second = result.fillets[k:k+2]
        start = first.outgoing_tangent if first is not None else points[k]
        end = second.incoming_tangent if second is not None else points[k+1]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color='#333333', linewidth=2, zorder=5)
    for fillet in result.fillets:
        if fillet is None:
            continue
        radial = fillet.incoming_tangent-fillet.center
        angles = np.arctan2(radial[1], radial[0])+np.linspace(0, fillet.signed_angle, 151)
        arc = fillet.center+fillet.radius*np.column_stack((np.cos(angles), np.sin(angles)))
        ax.plot(*arc.T, color='#009E73', linewidth=2.6, zorder=6)
    for index in (j, j+1):
        ax.annotate(rf'$\mathbf{{p}}_{{{index+1}}}$',
                    points[index], xytext=(7, 7), textcoords='offset points',
                    color='#A13E00', fontsize=14, zorder=7,
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
    length = np.linalg.norm(points[j+1]-points[j])
    remaining = result.remaining_lengths[j]
    ax.set_title(f'{n} corridors · case {case["case"]}\n'
                 f'$L={length:.5f}$ m; remaining straight = {1000*remaining:.2f} mm')
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    ax.margins(.08)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.grid(alpha=.12)
    print(f'{n} corridors, case {case["case"]}, p{j+1}→p{j+2}: '
          f'L={length:.9f} m, 2R={2*robot.R:g} m, remaining={remaining:.9f} m')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--input', type=Path, default=DIRECTORY/'stress_report.json')
    parser.add_argument('--output-dir', type=Path, default=DIRECTORY/'near_spacing')
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if report['seed'] != 37:
        raise ValueError('The selected case numbers refer to seed 37.')
    robot = SimpleNamespace(r=report['r'], R=report['R'])
    figures = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        figures.append(('short_segments_overview', fig))
        for ax, (n, number, segment) in zip(axes.flat, CASES):
            group = next(g for g in report['groups'] if g['corridors'] == n)
            case = next(c for c in group['cases'] if c['case'] == number)
            corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]], float))
                         for a,b,c,d in case['bounds']]
            result = compute_filleted_baseline_exact(corridors, robot)
            if not result.feasible:
                raise RuntimeError(f'Replay failed: {n} corridors, case {number}.')
            lengths = np.linalg.norm(np.diff(result.polyline, axis=0), axis=1)
            if np.any(lengths < 2*robot.R-1e-9):
                raise RuntimeError('Polyline violates the 2R bound.')
            draw_case(ax, n, case, segment, result, robot)
            detail, detail_ax = plt.subplots(figsize=(9, 7))
            draw_case(detail_ax, n, case, segment, result, robot)
            detail.tight_layout()
            figures.append((f'short_segment_n{n}_case{number}', detail))
        fig.legend(handles=[
            Patch(facecolor='#EDF5FC', edgecolor='#555555', label='Corridors'),
            Patch(facecolor='#0072B2', alpha=.2, label=r'Safe overlaps $\mathcal{D}_j$'),
            Line2D([], [], color='#64748b', linestyle='--', marker='o', label='Polyline'),
            Line2D([], [], color='#D55E00', linestyle='--', label=r'Segment close to $2R=4$ m'),
            Line2D([], [], color='#333333', linewidth=2, label='Remaining straights'),
            Line2D([], [], color='#009E73', linewidth=2.6, label='Radius-R fillets'),
        ], loc='lower center', ncol=3, frameon=False, fontsize=14)
        fig.tight_layout(rect=(0, .07, 1, 1), h_pad=2, w_pad=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures:
        for suffix in ('pdf', 'png'):
            path = args.output_dir/f'{name}.{suffix}'
            figure.savefig(path, dpi=200, bbox_inches='tight')
            print('Saved:', path)
    if args.no_show:
        for _, figure in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
