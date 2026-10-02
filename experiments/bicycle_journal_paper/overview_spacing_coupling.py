"""Export the local 2R coupling diagnostics for all example maps.

Run from any directory:
    python experiments/bicycle_journal_paper/overview_spacing_coupling.py

Outputs a multi-page vector PDF, per-example and per-pair CSVs, and a Markdown
summary. Only the nominal safe-door check runs: no endpoint or fillet solver.
Orange means some aligned choices fail 2R, not that the drawn witness fails it.
Rejected maps show safe doors only; ambiguous/unavailable pairs are unclassified.
"""

import argparse
import csv
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = runpy.run_path(str(HERE / 'examples_maps_polyline.py'))
STATUSES = ('guaranteed', 'coupled', 'impossible', 'unavailable')


def collect():
    records, summaries, details = [], [], []
    for number in EXAMPLES['EXAMPLE_NUMBERS']:
        corridors, _, _, vehicle = EXAMPLES['example_corridor_sequence'](number)
        r, R = vehicle.width / 2, vehicle.max_radius
        result = EXAMPLES['check_orthogonal_polyline'](corridors, r, R)
        counts = {status: sum(pair['status'] == status for pair in result['spacing_pairs'])
                  for status in STATUSES}
        row = dict(example=number, radius_r=r, radius_R=R, required_length=2*R,
                   nominal_status=result['status'], **counts,
                   coupled_pairs='; '.join(
                       f"D{p['overlaps'][0]+1}-D{p['overlaps'][1]+1}"
                       for p in result['spacing_pairs'] if p['status'] == 'coupled'))
        summaries.append(row)
        records.append((corridors, result, row))
        for pair in result['spacing_pairs']:
            details.append(dict(example=number, corridor=pair['corridor']+1,
                                first_door=pair['overlaps'][0]+1,
                                second_door=pair['overlaps'][1]+1,
                                direction=pair['direction'], status=pair['status'],
                                minimum_length=pair['minimum_length'],
                                maximum_length=pair['maximum_length'],
                                required_length=pair['required_length']))
    return records, summaries, details


def draw_map(ax, corridors, result, summary):
    for corridor in corridors:
        closed = np.vstack([corridor.corners, corridor.corners[0]])
        ax.plot(*closed.T, color='0.6', lw=.65)
    centers = []
    for door in result['doors']:
        if door is None:
            centers.append([np.nan, np.nan])
            continue
        x, y = door['x'], door['y']
        centers.append([sum(x)/2, sum(y)/2])
        ax.add_patch(Rectangle((x[0], y[0]), x[1]-x[0], y[1]-y[0],
                               facecolor='#dbeafe', edgecolor='#93b6d9', lw=.6))
    points = result['polyline'] if result['feasible'] else np.asarray(centers)
    if result['feasible']:
        ax.plot(*points.T, color='#2563eb', lw=1.2, zorder=3)
    if len(points):
        ax.scatter(*points.T, color='#2563eb', s=10, zorder=5)
        EXAMPLES['plot_spacing_coupling'](
            ax, result, points, draw_segments=result['feasible'])
    mode = 'nominal witness' if result['feasible'] else 'doors only'
    ax.set_title(f"Example {summary['example']} — {mode}\n"
                 f"G {summary['guaranteed']}   C {summary['coupled']}   "
                 f"I {summary['impossible']}   U {summary['unavailable']}"
                 f"  |  {result['status'].replace('_', ' ')}", fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.12)
    ax.set_xlabel('x [m]', fontsize=8)
    ax.set_ylabel('y [m]', fontsize=8)
    ax.margins(.07)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=HERE / 'figures' / 'spacing_coupling')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, summaries, details = collect()
    for name, rows in [('summary.csv', summaries), ('pairs.csv', details)]:
        with (args.output_dir / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    legend = [
        Line2D([], [], color='#2563eb', marker='o', ms=4, label='Nominal witness / door center'),
        Line2D([], [], color='#d97706', ls='--', marker='o', markerfacecolor='none',
               label='Coupled: some aligned choices < 2R'),
        Line2D([], [], color='#c62828', ls='--', marker='o', markerfacecolor='none',
               label='Impossible: all aligned choices < 2R'),
    ]
    pdf = args.output_dir / 'overview.pdf'
    with PdfPages(pdf) as document:
        for start in range(0, len(records), 6):
            fig, axes = plt.subplots(3, 2, figsize=(11.7, 12.5))
            fig.subplots_adjust(left=.07, right=.97, bottom=.06, top=.87,
                                hspace=.55, wspace=.25)
            fig.suptitle('Local 2R spacing between consecutive safe overlaps', y=.985, fontsize=15)
            fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .96),
                       fontsize=9, frameon=False, ncol=1)
            for ax, record in zip(axes.flat, records[start:start+6]):
                draw_map(ax, *record)
            for ax in list(axes.flat)[len(records[start:start+6]):]:
                ax.set_axis_off()
            fig.text(.5, .018,
                     'G: guaranteed   C: coupled   I: locally impossible   U: unclassified\n'
                     'Original eroded overlaps; flags precede global propagation, endpoint constraints and fillet constraints.',
                     ha='center', fontsize=9)
            document.savefig(fig)
            fig.savefig(args.output_dir / f'overview_{start//6+1}.png', dpi=140)
            plt.close(fig)
    lines = [
        '# Local 2R coupling overview', '',
        'Computed from the original radius-r-eroded overlap rectangles, before '
        'propagation or endpoint/fillet constraints. Distances are in metres. '
        'Each unique H/V pair is classified by the minimum and maximum aligned '
        'distance. Guaranteed means min >= 2R; coupled means min < 2R <= max; '
        'impossible means max < 2R (all comparisons use the nominal tolerance). '
        'Unclassified means missing safe doors or no unique H/V direction. '
        'Guaranteed applies only to longitudinal spacing: transverse alignment '
        'and global feasibility still matter.', '',
        '| Example | Nominal status | Guaranteed | Coupled | Impossible | Unclassified | Coupled doors |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in summaries:
        lines.append(f"| {row['example']} | {row['nominal_status']} | {row['guaranteed']} | "
                     f"{row['coupled']} | {row['impossible']} | {row['unavailable']} | {row['coupled_pairs']} |")
    (args.output_dir / 'README.md').write_text('\n'.join(lines)+'\n')
    for status in STATUSES:
        print(f"{status}: {sum(row[status] for row in summaries)} pairs in "
              f"{sum(row[status] > 0 for row in summaries)} examples")
    print(pdf.resolve())


if __name__ == '__main__':
    main()
