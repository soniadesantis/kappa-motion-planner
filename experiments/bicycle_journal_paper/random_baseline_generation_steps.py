"""Illustrate one random test from its sampled walk to a filleted baseline.

Run directly in VS Code; --no-show exports without a window. This illustration
selects a successful case with edge extensions and at least two fillets. The
actual experiment retains all polyline-feasible cases, including fillet failures.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import example_random_baseline_construction as generator
from corridor_dimensions_study import STYLE
from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility,
    compute_filleted_baseline,
    compute_orthogonal_polyline,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "thesis_figures"
STYLE = {**STYLE, "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"}


def select_case(seed, corridor_count, limit=10000):
    """Use the existing sampler; choose an informative successful illustration."""
    generator.NUMBER_OF_CORRIDORS = corridor_count
    generator.validate_configuration()
    rng = np.random.default_rng(seed)
    robot = SimpleNamespace(r=generator.ROBOT_RADIUS, R=generator.TURNING_RADIUS)
    rejected = Counter()
    for index in range(1, limit + 1):
        bounds, sampled = generator.sample_corridors(rng)
        corridors = generator.corridors_from_bounds(bounds)
        report = analyze_orthogonal_polyline_feasibility(corridors, robot, compute_viable=False)
        if not report.feasible:
            rejected[report.status] += 1
            continue
        baseline = compute_filleted_baseline(
            corridors, robot, use_joint_solver=False,
            max_backtracking_attempts=generator.MAX_BACKTRACKING_ATTEMPTS)
        if (baseline.feasible and sampled['edge_extensions']
                and sum(fillet is not None for fillet in baseline.fillets) >= 2):
            witness = compute_orthogonal_polyline(corridors, robot)
            return index, bounds, sampled, report, baseline, witness, dict(rejected)
    raise RuntimeError("No illustrative successful case found within the generation limit.")


def rectangle(ax, bounds, *, fill="white", edge="0.25", alpha=1, lw=0.6, zorder=1):
    a, b, c, d = bounds
    ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor=fill, edgecolor=edge,
                           alpha=alpha, linewidth=lw, zorder=zorder))


def draw_corridors(ax, bounds):
    # Fill first, then outline so overlaps retain visible corridor boundaries.
    for box in bounds:
        rectangle(ax, box, fill="#EDF5FC", edge="none")
    for box in bounds:
        rectangle(ax, box, fill="none", zorder=2)


def create_figure(case, seed):
    index, bounds, sampled, report, baseline, witness, rejected = case
    axes_vectors = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    increments = axes_vectors[sampled['headings']] * np.array(sampled['walk_lengths'])[:, None]
    walk = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.4))
        fig.subplots_adjust(left=.025, right=.98, bottom=.10, top=.94,
                            wspace=.10, hspace=.34)
        axes = axes.ravel()
        titles = ["(a) Sample an orthogonal walk", "(b) Add widths and end extensions",
                  "(c) Extend fully covered edges", "(d) Check exact polyline feasibility",
                  "(e) Recover a feasible polyline", "(f) Construct the filleted baseline"]
        axes[0].plot(*walk.T, color="0.3", linewidth=1, marker="o", markersize=3)
        for j, (first, last) in enumerate(zip(walk[:-1], walk[1:]), 1):
            middle = (first + last) / 2
            axes[0].annotate(str(j), middle, xytext=(5, 5), textcoords="offset points",
                             fontsize=11, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        draw_corridors(axes[1], sampled['original_bounds'])
        axes[1].plot(*walk.T, ":", color="0.5", linewidth=.7)
        for ax in axes[2:]:
            draw_corridors(ax, bounds)
        for box in sampled['original_bounds']:
            rectangle(axes[2], box, fill="none", edge="0.6", lw=.6, zorder=3)
        edge_coordinates = {'xmin': 0, 'xmax': 1, 'ymin': 2, 'ymax': 3}
        for change in sampled['edge_extensions']:
            a, b, c, d = bounds[change['corridor']]
            edge = edge_coordinates[change['edge']]
            if edge < 2:
                x = change['after']
                axes[2].plot([x, x], [c, d], color="#486F91", linewidth=1.5, zorder=4)
            else:
                y = change['after']
                axes[2].plot([a, b], [y, y], color="#486F91", linewidth=1.5, zorder=4)
        for door, xs, ys in zip(report.safe_overlaps, report.x_reachable, report.y_reachable):
            rectangle(axes[3], door, fill="#E6E6E6", edge="0.5", zorder=3)
            rectangle(axes[3], (xs[0], xs[1], ys[0], ys[1]),
                      fill="#9DBBD0", edge="#486F91", zorder=4)
        axes[4].plot(*witness.T, color="0.3", linewidth=1.1, marker="o", markersize=3, zorder=5)
        points = baseline.polyline
        axes[5].plot(*points.T, ":", color="0.6", linewidth=.7, zorder=3)
        for j in range(len(points)-1):
            first = baseline.fillets[j].outgoing_tangent if baseline.fillets[j] else points[j]
            last = baseline.fillets[j+1].incoming_tangent if baseline.fillets[j+1] else points[j+1]
            axes[5].plot([first[0], last[0]], [first[1], last[1]],
                         color="0.3", linewidth=1.15, zorder=5)
        for fillet in baseline.fillets:
            if fillet is None:
                continue
            radial = fillet.incoming_tangent - fillet.center
            angles = np.arctan2(radial[1], radial[0]) + np.linspace(0, fillet.signed_angle, 100)
            arc = fillet.center + fillet.radius*np.column_stack([np.cos(angles), np.sin(angles)])
            axes[5].plot(*arc.T, color="#486F91", linewidth=1.8, zorder=6)
        all_boxes = np.array(bounds + sampled['original_bounds'])
        low = np.array([all_boxes[:, 0].min(), all_boxes[:, 2].min()])
        high = np.array([all_boxes[:, 1].max(), all_boxes[:, 3].max()])
        pad = .06 * max(high-low)
        for ax, title in zip(axes, titles):
            ax.set(xlim=(low[0]-pad, high[0]+pad), ylim=(low[1]-pad, high[1]+pad), aspect="equal")
            ax.set_title(title, fontsize=13, pad=9)
            ax.set_axis_off()
        fig.text(.5, .045,
                 rf"$n={len(bounds)},\ r={generator.ROBOT_RADIUS:g},\ R={generator.TURNING_RADIUS:g}$"
                 + f"; seed {seed}, generated case {index}; "
                 + f"{baseline.backtracking_attempts} waypoint proposals",
                 fontsize=12, ha="center")
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--corridors', type=int, default=5)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    case = select_case(args.seed, args.corridors)
    fig = create_figure(case, args.seed)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for suffix in ('pdf', 'png'):
            path = OUTPUT_DIRECTORY / f'random_baseline_generation_steps.{suffix}'
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.06)
            print(f'Saved {path}')
    index, bounds, sampled, report, baseline, witness, rejected = case
    metadata = dict(seed=args.seed, generated_index=index, corridor_count=args.corridors,
                    r=generator.ROBOT_RADIUS, R=generator.TURNING_RADIUS,
                    bounds=bounds, **sampled, polyline_rejections_before_selection=rejected,
                    baseline_status=baseline.status, attempts=baseline.backtracking_attempts,
                    selection_method=baseline.selection_method,
                    illustration_selection='Successful baseline, nonempty edge extensions, at least two fillets')
    (OUTPUT_DIRECTORY / 'random_baseline_generation_steps.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    print(f'Generated case {index}: {baseline.status}, {baseline.backtracking_attempts} proposals')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
