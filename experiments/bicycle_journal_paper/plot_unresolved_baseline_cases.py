"""Replay and plot bounded-search failures from saved random experiments.

Run directly in VS Code, or use --no-show. Dashed polylines are exact internal
2R witnesses, not validated filleted paths. No solver or larger budget is used.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from example_baseline_construction import draw_fillet_region, draw_region, plot_filleted_baseline
from example_random_baseline_construction import corridors_from_bounds
from kappa_planner.baseline_construction import compute_filleted_baseline


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'results' / 'random_baseline' / 'unresolved_review'
INPUTS = [OUTPUT_DIRECTORY / f'random_baseline_n{n}_seed7.json' for n in (5, 8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', default=INPUTS)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    unresolved = []
    for source in args.inputs:
        experiment = json.loads(source.read_text())
        config = experiment['configuration']
        robot = SimpleNamespace(r=config['r'], R=config['R'])
        for case in experiment['cases']:
            if case['status'] != 'backtracking_unresolved':
                continue
            result = compute_filleted_baseline(corridors_from_bounds(case['bounds']), robot,
                                              use_joint_solver=False,
                                              max_backtracking_attempts=config['max_attempts'])
            if result.status != case['status']:
                raise RuntimeError(f"Replay changed status for n={config['corridors']}, case {case['case']}.")
            unresolved.append((config['corridors'], case, result, robot))
    if not unresolved:
        raise RuntimeError('No bounded-search unresolved cases in the input experiments.')
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = (len(unresolved) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 5.1 * rows), squeeze=False)
    for ax, (n, case, result, robot) in zip(axes.flat, unresolved):
        for a, b, c, d in case['bounds']:
            ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor='#EDF5FC',
                                   edgecolor='0.3', linewidth=.5, zorder=1))
        for door in result.feasibility.safe_overlaps:
            draw_region(ax, door, '#a855f7', alpha=.2, zorder=2)
        for region in result.fillet_regions:
            if region is not None:
                draw_fillet_region(ax, region)
        points = result.orthogonal_polyline
        ax.plot(*points.T, '--o', color='#486F91', linewidth=1,
                markersize=3, zorder=5)
        for j, point in enumerate(points, 1):
            ax.annotate(f'p{j}', point, xytext=(4, 5), textcoords='offset points', fontsize=9)
        ax.set_title(f'{n} corridors · case {case["case"]}\n'
                     f'{result.backtracking_attempts} proposals; search unresolved', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.margins(.08)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(alpha=.15)
        detail = plot_filleted_baseline(case['case'], result, case['internal_median_ms'], robot)
        detail.axes[0].set_title(f'{n} corridors · case {case["case"]}: bounded search unresolved')
        detail_path = OUTPUT_DIRECTORY / f'unresolved_n{n}_case{case["case"]}'
        for suffix in ('pdf', 'png'):
            detail.savefig(detail_path.with_suffix('.'+suffix), dpi=200, bbox_inches='tight')
        plt.close(detail)
        print(f'n={n}, case {case["case"]}, generated sequence {case["generated_index"]}: '
              f'{result.backtracking_attempts} proposals; first rejected waypoint '
              f'p{result.first_rejected_waypoint + 1}')
    for ax in list(axes.flat)[len(unresolved):]:
        ax.set_axis_off()
    fig.legend(handles=[
        Patch(facecolor='#EDF5FC', edgecolor='0.3', label='Corridors'),
        Patch(facecolor='#a855f7', alpha=.3, label='Safe overlaps D_j'),
        Patch(facecolor='#16a34a', alpha=.4, label='Local fillet-admissible regions A_j'),
        Line2D([], [], color='#486F91', linestyle='--', marker='o', markersize=3,
               label='Unfilleted 2R witness'),
    ], loc='lower center', ncol=2, fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, 1), h_pad=2, w_pad=2)
    for suffix in ('pdf', 'png'):
        path = OUTPUT_DIRECTORY / f'unresolved_baseline_overview.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        print(f'Saved {path}')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
