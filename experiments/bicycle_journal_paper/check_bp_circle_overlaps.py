"""One-pass audit of initial circles across all bp examples (no plotting).

Run: python experiments/bicycle_journal_paper/check_bp_circle_overlaps.py
Produces CSV details and a Markdown summary in figures/bp_circles/diagnostics.
No repeated benchmarks or refinement are run.
"""

import argparse
import csv
from pathlib import Path
import runpy

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=HERE/'figures'/'bp_circles'/'diagnostics')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    example = runpy.run_path(str(HERE/'examples_maps_polyline.py'))
    circle_rows, pair_rows, summary = [], [], []
    for number in example['EXAMPLE_NUMBERS']:
        c, s, e, v = example['example_corridor_sequence'](number)
        geometry = example['build_trajectory_geometry'](
            c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
        placement = build_bp_circle_sequence(c, v, geometry, diagnostic_all_pairs=True)
        if not geometry['feasible']:
            summary.append(dict(example=number, status=placement['status'], circles=0,
                                outside_Aj='', opposite_overlaps='', same_overlaps='',
                                same_arc_conflicts='', same_Aj_failures=''))
            continue
        report = placement['diagnostics']
        for item in report['circles']:
            circle_rows.append(dict(example=number, circle=item['circle_index']+1,
                bp_vertex=item['bp_vertex']+1, turn=item['turn'], radius=item['radius'],
                center_x=item['center'][0], center_y=item['center'][1],
                vertex_x=item['implied_vertex'][0], vertex_y=item['implied_vertex'][1],
                in_Aj=item['in_Aj'], Aj_margin=item['Aj_margin'],
                box_margin=item['box_margin'], corner_margin=item['corner_margin'],
                endpoint_door_violation=item['endpoint_door_violation'], rule=item['rule']))
        for item in report['pairs']:
            pair_rows.append(dict(example=number, **dict(item, first=item['first']+1,
                                                         second=item['second']+1)))
        overlaps = [p for p in report['pairs'] if p['circle_relation'] in ('overlap', 'coincident')]
        same = [p for p in overlaps if p['same_turn']]
        def names(pairs):
            return '; '.join(f"O{p['first']+1}-O{p['second']+1}"
                             + ('' if p['consecutive'] else ' [nonconsecutive]') for p in pairs)
        summary.append(dict(example=number, status=placement['status'],
            circles=len(report['circles']),
            outside_Aj='; '.join(f"O{p['circle_index']+1}" for p in report['circles'] if not p['in_Aj']),
            opposite_overlaps=names([p for p in overlaps if not p['same_turn']]),
            same_overlaps=names(same),
            same_arc_conflicts=names([p for p in same if p['quarter_conflict']]),
            same_Aj_failures=names([p for p in same if not p['both_in_Aj']])))
    write_csv(args.output_dir/'circles.csv', circle_rows)
    write_csv(args.output_dir/'pairs.csv', pair_rows)
    write_csv(args.output_dir/'summary.csv', summary)
    lines = ['# Initial bp-circle diagnostics', '',
        'One construction per example; no circle moves or connections. Circle O_i belongs '
        'to corridors C_i/C_(i+1). A_j membership is checked at the implied vertex '
        'p = o + R*u - R*v, using the existing bp region. Pair checks use the full relevant '
        'quarter-arcs analytically. A common end-to-start contact of consecutive circles is '
        'reported as a shared join, not an arc conflict. Opposite-turn overlap prevents a '
        'direct internal tangent when those circles must be connected; nonconsecutive '
        'overlaps are reported separately. The current A_j sets also include the door '
        'bounds tightened by the bp endpoint-spacing construction; violating those '
        'bounds alone is not proof of obstacle collision. These local checks do not '
        'certify a full path.', '',
        '| Example | Status | Circles | Outside A_j | Opposite-turn overlaps | Same-turn overlaps | Same-turn arc conflicts | Same-turn pairs failing A_j |',
        '| --- | --- | ---: | --- | --- | --- | --- | --- |']
    for row in summary:
        lines.append('| '+' | '.join(str(value) or '—' for value in row.values())+' |')
    (args.output_dir/'README.md').write_text('\n'.join(lines)+'\n')
    for row in summary:
        if row['opposite_overlaps'] or row['same_overlaps'] or row['outside_Aj']:
            print(row)
    print('Circle count:', len(circle_rows))
    print('Outside A_j:', sum(not c['in_Aj'] for c in circle_rows))
    print('All pair count:', len(pair_rows))
    print(args.output_dir.resolve())


if __name__ == '__main__':
    main()
