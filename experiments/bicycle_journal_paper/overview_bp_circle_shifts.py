"""Export a same-turn shift overview; build each example once, plot affected maps.

Run with MPLBACKEND=Agg for headless export. Initial-circle audits are preserved.
"""
import csv
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence


def main():
    here = Path(__file__).resolve().parent
    output = here/'figures'/'bp_circles'/'shifted'
    output.mkdir(parents=True, exist_ok=True)
    example = runpy.run_path(str(here/'examples_maps_polyline.py'))
    rows = []
    with PdfPages(output/'overview.pdf') as pdf:
        for number in example['EXAMPLE_NUMBERS']:
            c, s, e, v = example['example_corridor_sequence'](number)
            g = example['build_trajectory_geometry'](
                c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
            report = build_bp_circle_sequence(c, v, g, shift_same_turn=True)
            shift = report.get('shifting')
            if shift is None:
                continue
            for step in shift['steps']:
                rows.append(dict(example=number, first=step['pair'][0]+1,
                    second=step['pair'][1]+1, accepted=step['accepted'],
                    neighbor_risk=step['neighbor_risk'], weights=step.get('weights'),
                    displacements=tuple(tuple(d) for d in step.get('displacements', ())),
                    reason=step['reason']))
            if shift['steps']:
                fig, _ = example['plot_bp_circle_sequence'](
                    c, g, v, v.max_radius, number, report=report)
                pdf.savefig(fig)
                fig.savefig(output/f'example_{number}.pdf')
                if number == 23:
                    fig.savefig(output/'example_23.png', dpi=140)
                plt.close(fig)
    with (output/'shifts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['example'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['accepted'] for r in rows)} accepted shifts; outputs: {output}")


if __name__ == '__main__':
    main()
