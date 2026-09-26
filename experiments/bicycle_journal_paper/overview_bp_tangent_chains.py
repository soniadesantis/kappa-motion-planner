"""One-pass tangent-chain audit and plots for the bp example set.

MPLBACKEND=Agg python experiments/bicycle_journal_paper/overview_bp_tangent_chains.py
"""
import csv
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence


def main():
    here = Path(__file__).resolve().parent
    example = runpy.run_path(str(here/'examples_maps_polyline.py'))
    output = here/'figures'/'bp_circles'/'tangent_chains'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with PdfPages(output/'overview.pdf') as pdf:
        for number in example['EXAMPLE_NUMBERS']:
            c, s, e, v = example['example_corridor_sequence'](number)
            g = example['build_trajectory_geometry'](
                c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
            p = build_bp_circle_sequence(c, v, g, shift_same_turn=True,
                                        shift_opposite_turn=True, connect_tangents=True)
            t = p['tangent_chain']
            rows.append(dict(example=number, status=t['status'], links=len(t['tangents']),
                skipped=';'.join(str(i+1) for i in t['skipped']),
                retained=';'.join(str(i+1) for i in t['retained']),
                milliseconds=1000*t['computation_time'], reason=t['reason']))
            if not g['feasible']:
                continue
            fig, _ = example['plot_bp_circle_sequence'](c, g, v, v.max_radius, number, report=p)
            pdf.savefig(fig)
            if number in (1, 7, 15, 23, 35):
                fig.savefig(output/f'example_{number}.pdf')
            if number in (1, 35):
                fig.savefig(output/f'example_{number}.png', dpi=120)
            plt.close(fig)
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Incremental bp tangent chains', '',
        'Fixed circles after same-turn and opposite-turn repair. Only the internal '
        'circle chain is constructed: no start/end pose maneuvers or timing. '
        'All used arcs are within their certified quarter and at most pi/2. '
        'Lines pass disk-clearance checks against the relevant corridor union. '
        'Each candidate extension checks local order/intersections; a final audit '
        'checks all retained line/arc pairs analytically. Ordinary shared joins '
        'and zero-length coincident-circle joins are allowed.', '',
        'A stack tries bypassing departure circles when extension fails. A currently '
        'unconnectable intermediate target may be deferred and skipped if a later '
        'connection succeeds. First and last circles are retained. The greedy '
        'construction is not a completeness or optimality proof. UNRESOLVED plots '
        'show diagnostic candidate/partial paths in amber, not accepted trajectories.', '',
        '| Example | Status | Tangents | Skipped circles (one-based) | Reason |',
        '| --- | --- | ---: | --- | --- |']
    lines.extend(f"| {r['example']} | {r['status']} | {r['links']} | {r['skipped'] or '—'} | {r['reason']} |"
                 for r in rows)
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(output/'overview.pdf')


if __name__ == '__main__':
    main()
