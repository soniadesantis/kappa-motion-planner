"""Reproduce example 35: nominal opposite-turn overlap with a feasible bp.

Run with MPLBACKEND=Agg to save the PDF/PNG without opening a window.
"""
import argparse
from pathlib import Path
import runpy

import matplotlib.pyplot as plt

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', action='store_true', help='Apply opposite-turn separation.')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    examples = runpy.run_path(str(here/'examples_maps_polyline.py'))
    corridors, start, end, vehicle = examples['example_corridor_sequence'](35)
    geometry = examples['build_trajectory_geometry'](
        corridors, vehicle.width/2, vehicle.max_radius,
        start_pose=start, end_pose=end, validate_arcs=True)
    report = build_bp_circle_sequence(corridors, vehicle, geometry,
                                      shift_opposite_turn=args.repair)
    assert geometry['feasible'] and report['feasible']
    assert report['diagnostics']['flags']['opposite_turn_overlap'] == ([] if args.repair else [(0, 1)])
    assert not report['diagnostics']['flags']['outside_Aj']
    figure, _ = examples['plot_bp_circle_sequence'](
        corridors, geometry, vehicle, vehicle.max_radius, 35, report=report)
    checks = report['diagnostics']['circles']
    a, b = (c['center'] for c in checks)
    distance = report['diagnostics']['pairs'][0]['center_distance']
    ax = figure.axes[0]
    figure.set_size_inches(12, 4.8)
    notes = figure.axes[2]
    for text in list(notes.texts):
        text.remove()
    notes.text(0, 1, ('Feasible bp; independent arc audit passes.\n\n'
               'Both circles belong to their own A_j.\n'
               'Nominal rule: 45 degrees, wall clearance.\n\n'
               'O1: left turn. O2: right turn.\n'
               'Opposite-turn overlap: no internal tangent.\n\n'
               'No circle shifts applied.') if not args.repair else
               'Opposite-turn overlap resolved.\n\n'
               'Circle centers restored from the certified bp.\n'
               'Both remain admissible in A_j.\n\n'
               'A directed vertical tangent joins the quarters.\n'
               'Full trajectory connections remain unchecked.\n\n'
               'Dashed gray circles: nominal placement.\n'
               'Arrows: center displacements.', va='top', fontsize=9,
               transform=notes.transAxes)
    ax.plot([a[0], b[0]], [a[1], b[1]], ':', color='#b2188b', lw=1.2)
    comparison = '>' if args.repair else '<'
    ax.text(0, 2.75, rf'$d={distance:.3f}\,\mathrm{{m}}{comparison}2R=2\,\mathrm{{m}}$',
            ha='center', color='#b2188b', fontsize=10)
    ax.set_ylim(-.65, 3.05)
    if args.repair:
        ax.plot([0, 0], [.8, 1.2], color='black', lw=1.4)
    output = here/'figures'/'bp_circles'/('opposite_repaired' if args.repair else 'opposite_overlap')
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output/'example_35.pdf')
    figure.savefig(output/'example_35.png', dpi=150)
    plt.close(figure)
    if args.repair:
        (output/'README.md').write_text(
            '# Opposite-turn repair: example 35\n\n'
            'Reproduce using `python experiments/bicycle_journal_paper/demo_opposite_circle_overlap.py --repair`.\n\n'
            'The radial proposal violates A_j. Interpolation to distance 2R '
            'is admissible but its contact lies outside the relevant quarters. '
            'Restoring the certified bp centers (-1,0.8) and (1,1.2) gives '
            f'distance {distance:.9f}, admissible quarters and a vertical tangent '
            'from (0,0.8) to (0,1.2). Neighbor flags are rechecked. '
            'The bp is unchanged. No full tangent-chain certification is claimed.\n')
        print(output/'example_35.pdf')
        return
    (output/'README.md').write_text(
        '# Example 35: nominal opposite-turn overlap\n\n'
        'Reproduce with `python experiments/bicycle_journal_paper/demo_opposite_circle_overlap.py`.\n\n'
        'Three corridors form an east–north–east S-bend. Their widths are '
        '0.65, 0.50, and 0.65 m. The vehicle has r=0.1 m and R=1 m.\n\n'
        'The certified baseline vertices are (-4,-0.2), (0,-0.2), (0,2.2), '
        '(4,2.2): segment lengths 4, 2.4, 4 m. The independent continuous '
        'fillet-clearance audit also passes.\n\n'
        'Both nominal circles use forty_five_basic, with q=0.9/sqrt(2). '
        'The preferred-half candidate requires local offsets (0.675,0.75) '
        '(or swapped), whose norm exceeds 0.9. The nominal offsets (q,q) '
        'satisfy the ordinary clearance constraints.\n\n'
        f'Centers: {tuple(a)} and {tuple(b)}; turns +1 and -1. '
        f'Their distance is {distance:.9f} m < 2R=2 m. Both implied vertices '
        'belong to their own A_j sets, but they are not a jointly compatible '
        'orthogonal-polyline placement. The original bp remains feasible.\n\n'
        'The relevant quarters do not cross, but overlapping opposite-turn '
        'supporting circles cannot have a direct common internal tangent. '
        'The same-turn shifting pass leaves this pair unchanged.\n')
    print(output/'example_35.pdf')


if __name__ == '__main__':
    main()
