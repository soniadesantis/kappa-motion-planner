"""Show the baseline, repaired circle placements, and collision-free circle arcs.

The second figure applies the four existing placement rules plus baseline
fallback, checking each implied waypoint against A_j, then overlap repairs.
The third figure highlights eroded-corridor arcs joined through the certified fillet.
Tangency contacts must lie on these green intervals. Shortcuts also cross the
full overlaps at intermediate doors in order. Each straight segment's circular
footprint is checked analytically in the original corridor union, including safe
concave wedges. Global tangent intersections remain unchecked.
Enable PLOT_FEASIBILITY for the optional exact-test figure.

Use VS Code Run/Debug and choose EXAMPLE_NUM below. No arguments are needed.
CLI example: python experiments/bicycle_journal_paper/example_baseline_construction.py \
    --example 7 --repetitions 100 --save /tmp/baseline.png

Timing covers the complete Boolean check, excluding imports, map creation,
plotting and the optional backward pass for globally viable regions.
"""

import argparse
from pathlib import Path
from statistics import median
from textwrap import fill
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility,
    check_orthogonal_polyline_feasibility,
    compute_filleted_baseline,
)
from kappa_planner.helpers.fillet_backtracking import region_slice
from kappa_planner.refinement import (
    place_refinement_circles, restore_opposite_turn_overlaps,
    repair_same_turn_overlaps, connect_refinement_circles, compute_circle_safe_arcs,
    connect_safe_arc_circles, connect_simple_tangent_chain,
)

EXAMPLE_NUM = 21  # Choose one map from 1 through 35, then press Run in VS Code.
REPETITIONS = 100  # Calls per timing batch.
BATCHES = 7
WARMUP = 5
PLOT_RESULTS = True
PLOT_FEASIBILITY = False  # Optional extra figure beyond baseline, placement, connections.
ALLOW_SAFE_ARC_SKIPPING = True  # False compares against consecutive-only connections.
CONNECTION_RULE = 'compare'  # 'graph', 'simple', or 'compare'; VS Code Run uses this.
USE_JOINT_SOLVER = False  # Evaluate the cheap heuristic alone; enable for fallback.
USE_BOUNDARY_DIRECTIONS = True  # Infer virtual entry/exit from the map's poses.
SAVE_FIGURE = None  # Optional filename, e.g. "/tmp/baseline.png".


def benchmark_check(corridors, robot, repetitions, batches, warmup):
    """Return per-call batch averages in microseconds (min, median, max)."""
    for _ in range(warmup):
        check_orthogonal_polyline_feasibility(corridors, robot)
    samples = []
    for _ in range(batches):
        start = perf_counter_ns()
        for _ in range(repetitions):
            check_orthogonal_polyline_feasibility(corridors, robot)
        samples.append((perf_counter_ns() - start) / repetitions / 1000)
    return min(samples), median(samples), max(samples)


def draw_region(ax, bounds, color, *, alpha=.2, linestyle="-", hatch=None, zorder=3):
    """Draw a closed rectangle, line, or point without inflating its geometry."""
    xmin, xmax, ymin, ymax = bounds
    if xmax == xmin and ymax == ymin:
        ax.scatter([xmin], [ymin], marker="s", color=color, s=45, zorder=zorder)
    elif xmax == xmin or ymax == ymin:
        ax.plot([xmin, xmax], [ymin, ymax], color=color, linewidth=3,
                linestyle=linestyle, zorder=zorder)
    else:
        ax.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, facecolor=color,
            edgecolor=color, alpha=alpha, linewidth=2, linestyle=linestyle,
            hatch=hatch, zorder=zorder,
        ))


def separation_cut(report, index, R):
    """Part of D_j excluded directly by its incoming signed 2R constraint.

    Equality constraints may also shrink forward supports; these are shown by
    the blue outline instead. Orange is drawn only for a strict local reduction.
    """
    if index == 0 or not report.x_reachable:
        return None
    direction = report.passage_directions[index - 1]
    axis = 0 if direction in ("right", "left") else 2
    previous = (report.x_reachable if axis == 0 else report.y_reachable)[index - 1]
    if previous is None:
        return None
    door = report.safe_overlaps[index]
    low, high = door[axis:axis + 2]
    increasing = direction in ("right", "up")
    threshold = previous[0] + 2 * R if increasing else previous[1] - 2 * R
    if (increasing and threshold <= low + 1e-9
            or not increasing and threshold >= high - 1e-9):
        return None
    cut = list(door)
    cut[axis:axis + 2] = ((low, min(high, threshold)) if increasing
                          else (max(low, threshold), high))
    return tuple(cut)


def draw_boundary_positions(ax, initial_position, final_position):
    """Mark prescribed positions without drawing unvalidated connections."""
    handles = []
    for point, label, marker, color in (
        (initial_position, r"$p_0$", '^', '#0f766e'),
        (final_position, r"$p_f$", '*', '#be123c'),
    ):
        if point is None:
            continue
        ax.scatter(point[0], point[1], marker=marker, color=color, s=110,
                   edgecolors='white', linewidths=.7, zorder=10)
        ax.annotate(label, point, xytext=(7, 7), textcoords='offset points',
                    color=color, fontsize=11, zorder=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
        handles.append(Line2D([], [], marker=marker, color=color, linestyle='none',
                              markersize=9, label='Initial position' if marker == '^'
                              else 'Final position'))
    return handles


def plot_example(number, report, robot, timings, initial_position=None, final_position=None):
    """Plot D_j, forward supports, complete-sequence supports and 2R cuts."""
    R = robot.R if hasattr(robot, "R") else robot.max_radius
    r = robot.r if hasattr(robot, "r") else robot.width / 2
    fig, (ax, notes) = plt.subplots(
        1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [3, 1.4]}
    )
    notes.axis("off")
    for j, bounds in enumerate(report.corridor_bounds, start=1):
        draw_region(ax, bounds, "#64748b", alpha=.1, zorder=1)
        ax.annotate(f"C{j}", (bounds[0], bounds[3]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points", color="#475569")
    reductions = []
    for j, door in enumerate(report.safe_overlaps):
        if door is None:
            continue
        draw_region(ax, door, "#a855f7", alpha=.25, zorder=2)
        ax.annotate(f"D{j+1}", (door[0], door[2]), fontsize=8,
                    xytext=(3, -12), textcoords="offset points")
        cut = separation_cut(report, j, R)
        if cut is not None:
            reductions.append(j + 1)
            draw_region(ax, cut, "#ea580c", alpha=.35, hatch="///", zorder=3)
        if report.x_reachable:
            x, y = report.x_reachable[j], report.y_reachable[j]
            if x is None or y is None:
                ax.scatter([(door[0] + door[1]) / 2], [(door[2] + door[3]) / 2],
                           marker="x", color="red", s=65, zorder=6)
            else:
                draw_region(ax, (*x, *y), "#2563eb", alpha=.35,
                            linestyle="--", zorder=4)
        if report.x_viable:
            x, y = report.x_viable[j], report.y_viable[j]
            if x is not None and y is not None:
                draw_region(ax, (*x, *y), "#16a34a", alpha=.45, zorder=5)
    # Closest aligned points in each original pair of safe overlaps. The
    # shared transverse midpoint keeps the dimension segment inside both
    # transverse ranges; its longitudinal extent is exactly the minimum gap.
    for j, direction in enumerate(report.passage_directions):
        a, b = report.safe_overlaps[j:j + 2]
        if direction in ("right", "left"):
            transverse = (max(a[2], b[2]) + min(a[3], b[3])) / 2
            start, end = ((a[1], b[0]) if direction == "right" else (a[0], b[1]))
            xs, ys = [start, end], [transverse, transverse]
            marker, offset = "|", (0, -18)
        else:
            transverse = (max(a[0], b[0]) + min(a[1], b[1])) / 2
            start, end = ((a[3], b[2]) if direction == "up" else (a[2], b[3]))
            xs, ys = [transverse, transverse], [start, end]
            marker, offset = "_", (7, 0)
        gap = abs(end - start)
        short = gap < 2 * R - 1e-9
        color = "#dc2626" if short else "#334155"
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker,
                markersize=10, markeredgewidth=2, zorder=7)
        ax.annotate(f"{gap:.3g} m" + (" < 2R" if short else ""),
                    ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2),
                    xytext=offset, textcoords="offset points", color=color,
                    fontsize=8, ha="center" if marker == "|" else "left",
                    va="top" if marker == "|" else "center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1),
                    zorder=8)
    boundary_handles = draw_boundary_positions(ax, initial_position, final_position)
    status = "FEASIBLE" if report.feasible else "INFEASIBLE (nominal class)"
    ax.set_title(f"Example {number}: {status}",
                 color="#15803d" if report.feasible else "#b91c1c")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=.2)
    handles = [
        Patch(facecolor="#64748b", alpha=.2, label="Corridors"),
        Patch(facecolor="#a855f7", alpha=.3, label="Safe overlap D_j"),
        Patch(facecolor="#ea580c", alpha=.35, hatch="///", label="Removed by 2R"),
        Patch(facecolor="#2563eb", alpha=.35, linestyle="--",
              label="Forward reachable region"),
        Patch(facecolor="#16a34a", alpha=.45, label="Complete-sequence region"),
        Line2D([], [], color="#334155", marker="|", linewidth=2,
               label="Minimum gap between D_j sets"),
        Line2D([], [], color="#dc2626", marker="|", linewidth=2,
               label="Minimum gap < 2R"),
    ]
    notes.legend(handles=handles + boundary_handles, loc="lower left", fontsize=9)
    messages = [
        status, report.status, report.reason,
        f"r = {r:g} m; R = {R:g} m; minimum segment length = {2*R:g} m.",
        f"Boolean check: median {timings[1]:.2f} µs/call\n"
        f"min {timings[0]:.2f}, max {timings[2]:.2f} µs/call.",
        "Directions: " + (", ".join(report.passage_directions) or "unavailable"),
        "Direct 2R reductions at: " +
        (", ".join(f"D{j}" for j in reductions) or "none"),
        "Blue regions satisfy preceding constraints. Green regions contain values "
        "that participate in at least one complete feasible sequence.",
        "Waypoint p_j belongs to safe overlap D_j.",
        "Waypoints cannot be chosen independently from these regions: adjacent "
        "points must still satisfy equality and separation constraints.",
        "Red crosses mark empty forward regions. Fillet construction is shown "
        "in a separate figure; endpoint connections are not checked.",
        "A gap below 2R means some aligned choices are too short; a feasible "
        "polyline may still exist with points farther apart.",
    ]
    notes.text(0, 1, "\n\n".join(fill(text, 43) for text in messages if text),
               va="top", fontsize=8.5, transform=notes.transAxes)
    fig.tight_layout()
    return fig


def draw_fillet_region(ax, region, *, color='#16a34a'):
    """Render the actual curved A_j via analytic slices, including degeneracies.

    The favorable y endpoint minimizes its canonical center coordinate; slicing
    there gives the full x projection. Sample only the curved display boundary,
    never the feasibility test. Polygon chords lie within this convex region.
    """
    if region['empty']:
        return
    low, high = region['low'], region['high']
    best_y = low[1] if region['frame'][1].sum() > 0 else high[1]
    projection = region_slice(region, 1, best_y)
    if projection is None:
        return
    xs = np.linspace(*projection, 201)
    # Include the straight/curved junction exactly in the displayed boundary.
    junction = -region['offset'][0]
    if projection[0] < junction < projection[1]:
        xs = np.unique(np.r_[xs, junction])
    lower, upper = [], []
    for x in xs:
        sliced = region_slice(region, 0, x)
        if sliced is not None:
            lower.append((x, sliced[0]))
            upper.append((x, sliced[1]))
    if not lower:
        return
    vertices = np.array(lower + upper[::-1])
    spans = np.ptp(vertices, axis=0)
    if np.any(spans <= 1e-10):
        draw_region(ax, (vertices[:, 0].min(), vertices[:, 0].max(),
                         vertices[:, 1].min(), vertices[:, 1].max()), color, zorder=3)
    else:
        ax.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor=color,
                             alpha=.4, linewidth=1.5, zorder=3))


def plot_filleted_baseline(number, result, elapsed_ms, robot=None,
                          initial_position=None, final_position=None):
    """Show the validated trimmed segments and quarter-circle fillets."""
    fig, (ax, notes) = plt.subplots(
        1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [3, 1.4]}
    )
    notes.axis("off")
    for j, bounds in enumerate(result.feasibility.corridor_bounds, start=1):
        draw_region(ax, bounds, "#64748b", alpha=.1, zorder=1)
        ax.annotate(f"C{j}", (bounds[0], bounds[3]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    for j, door in enumerate(result.feasibility.safe_overlaps, start=1):
        if door is not None:
            draw_region(ax, door, "#a855f7", alpha=.2, zorder=2)
            ax.annotate(f"D{j}", (door[0], door[2]), fontsize=8,
                        xytext=(3, -12), textcoords="offset points")
    for j, region in enumerate(result.fillet_regions):
        if region is None:
            continue
        door = result.feasibility.safe_overlaps[j]
        if region['empty']:
            center = ((door[0]+door[1])/2, (door[2]+door[3])/2)
            ax.scatter(*center, marker='x', color='#b91c1c', s=60, zorder=6)
            ax.annotate(f'A{j+1} empty', center, xytext=(5, 10),
                        textcoords='offset points', color='#b91c1c', fontsize=8)
        else:
            draw_fillet_region(ax, region)
            ax.annotate(f'A{j+1}', region['high'], xytext=(3, 3),
                        textcoords='offset points', color='#15803d', fontsize=8, zorder=7)

    points = result.polyline if result.feasible else result.orthogonal_polyline
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], "--", color="#2563eb",
                linewidth=1.3, zorder=3)
        ax.scatter(points[:, 0], points[:, 1], color="#2563eb", s=30, zorder=6)
        for j, point in enumerate(points, start=1):
            ax.annotate(f"p{j}", point, xytext=(5, 6),
                        textcoords="offset points", fontsize=9, zorder=7)
    if result.feasible:
        for j in range(len(points) - 1):
            start = (result.fillets[j].outgoing_tangent
                     if result.fillets[j] is not None else points[j])
            end = (result.fillets[j + 1].incoming_tangent
                   if result.fillets[j + 1] is not None else points[j + 1])
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="#172033", linewidth=2.5, zorder=4)
        for fillet in result.fillets:
            if fillet is None:
                continue
            radial = fillet.incoming_tangent - fillet.center
            angles = (np.arctan2(radial[1], radial[0])
                      + np.linspace(0, fillet.signed_angle, 100))
            arc = fillet.center + fillet.radius * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            ax.plot(arc[:, 0], arc[:, 1], color="#ea580c",
                    linewidth=3, zorder=5)
            tangents = np.array([fillet.incoming_tangent, fillet.outgoing_tangent])
            ax.scatter(tangents[:, 0], tangents[:, 1], marker="s",
                       color="#ea580c", s=30, zorder=6)
        if robot is not None:
            R = robot.R if hasattr(robot, 'R') else robot.max_radius
            vectors = {'right': np.array([1., 0.]), 'left': np.array([-1., 0.]),
                       'up': np.array([0., 1.]), 'down': np.array([0., -1.])}
            for incoming, direction in ((True, result.initial_direction),
                                        (False, result.final_direction)):
                if direction is None:
                    continue
                j = 0 if incoming else -1
                vertex = points[j]
                vector = 2 * R * vectors[direction]
                start, end = ((vertex-vector, vertex) if incoming else
                              (vertex, vertex+vector))
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color='#0891b2', linestyle='--', linewidth=2, zorder=4)
                ax.annotate('', end, xytext=start, zorder=5,
                            arrowprops=dict(arrowstyle='->', color='#0891b2',
                                            linewidth=2, shrinkA=0, shrinkB=0))

    boundary_handles = draw_boundary_positions(ax, initial_position, final_position)
    title = "FILLETED BASELINE FOUND" if result.feasible else "NO FILLETED BASELINE FOUND"
    ax.set_title(f"Example {number}: {title}",
                 color="#15803d" if result.feasible else "#b91c1c")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=.2)
    handles = [
        Patch(facecolor="#64748b", alpha=.2, label="Corridors"),
        Patch(facecolor="#a855f7", alpha=.3, label="Safe overlap D_j"),
        Patch(facecolor="#16a34a", alpha=.4, label="Local fillet-admissible A_j"),
        Line2D([], [], color="#2563eb", linestyle="--", marker="o",
               label="Orthogonal polyline / waypoints"),
    ]
    if result.feasible:
        handles.extend([
            Line2D([], [], color="#172033", linewidth=2.5, label="Trimmed straight segments"),
            Line2D([], [], color="#ea580c", linewidth=3, label="Radius-R arc fillets"),
            Line2D([], [], color="#ea580c", marker="s", linestyle="none",
                   label="Tangent points"),
        ])
        if result.initial_direction is not None or result.final_direction is not None:
            handles.append(Line2D([], [], color='#0891b2', linestyle='--',
                                  label='Boundary direction guide (2R)'))
    fig.legend(handles=handles + boundary_handles, loc="lower center", ncol=3, fontsize=8)
    messages = [result.status, result.reason,
                f"Construction time: {elapsed_ms:.3f} ms (one call; excludes plotting)."]
    messages.append(f'Candidate waypoint attempts: {result.backtracking_attempts}. '
                    'Attempts count individual point proposals, not complete polylines.')
    messages.append('Green A_j regions impose local fillet safety; alignment and 2R spacing '
                    'must still hold jointly. Where no fillet is required, A_j = D_j.')
    if result.segment_directions:
        messages.append('Boundary directions: entry '
                        + (result.initial_direction or 'unspecified') + '; exit '
                        + (result.final_direction or 'unspecified') + '.')
    if result.feasible:
        messages.extend([
            f"Selection: {result.selection_method}.",
            f"Quarter-circle fillets: {sum(f is not None for f in result.fillets)}.",
            "Remaining straight lengths [m]: "
            + ", ".join(f"{length:.3g}" for length in result.remaining_lengths),
            "Dashed lines show the selected orthogonal polyline. The solid "
            "segments and orange arcs form the validated filleted path.",
            "Turning waypoints locate the original corners. The rounded path "
            "passes through the square tangent markers.",
        ])
    else:
        messages.append(
            "Failure is certified only for the nominal construction model."
            if result.certified_infeasible else
            "This result is not a certificate that no filleted path exists."
        )
        if points is not None:
            messages.append("The dashed polyline satisfies the orthogonal 2R "
                            "constraints, but is not a validated filleted path.")
    messages.append("Initial and final pose connections are not included.")
    if result.initial_direction is not None or result.final_direction is not None:
        messages.append('Cyan direction guides are 2R long and attached to the '
                        'first/last vertices. They are illustrative, not validated path segments.')
    notes.text(0, 1, "\n\n".join(fill(text, 43) for text in messages if text),
               va="top", fontsize=8.5, transform=notes.transAxes)
    fig.tight_layout(rect=(0, .13, 1, .96))
    return fig


def plot_independent_circle_placement(number, baseline, placements):
    """Display supporting circles and implied vertices in A_j; no connections."""
    fig, (ax, notes) = plt.subplots(1,2,figsize=(13,8),gridspec_kw={'width_ratios':[3,1.4]})
    notes.axis('off')
    for bounds in baseline.feasibility.corridor_bounds:
        draw_region(ax,bounds,'#64748b',alpha=.1,zorder=1)
    for door in baseline.feasibility.safe_overlaps:
        if door is not None:
            draw_region(ax,door,'#a855f7',alpha=.2,zorder=2)
    for region in baseline.fillet_regions:
        if region is not None and not region['empty']:
            draw_fillet_region(ax,region)
    for decision in placements.aligned_sides:
        color = '#0891b2' if decision.side == 'left' else '#7c3aed'
        if decision.region is not None and not decision.region['empty']:
            draw_fillet_region(ax,decision.region,color=color)
        if decision.corner is not None:
            ax.scatter(*decision.corner,marker='D',s=20,color=color,zorder=8)
            ax.annotate(f'I{decision.waypoint_index+1}-{decision.side[0].upper()}',
                        decision.corner,xytext=(-32,5),textcoords='offset points',
                        fontsize=8,color=color)
    if baseline.polyline is not None:
        ax.plot(*baseline.polyline.T,':',color='#94a3b8',lw=1,zorder=3)
    colors={'forty_five_safe_half':'#2563eb','safe_half_shifted':'#0891b2',
            'forty_five_basic':'#9333ea','basic_shifted':'#ea580c','baseline':'#64748b','same_turn_coincident':'#059669'}
    names={'forty_five_safe_half':'1. Nominal, safe halves',
           'safe_half_shifted':'2. Shifted into halves',
           'forty_five_basic':'3. Ordinary nominal',
           'basic_shifted':'4. (a, b)', 'baseline':'5. Baseline fallback', 'same_turn_coincident':'Coincident-center shift'}
    messages=[]
    if placements.restored_waypoints:
        messages.append('Overlap fallback: restored baseline circles at '+
                        ', '.join(f'D{j+1}' for j in placements.restored_waypoints)+'.')
    rejected_colors={'forty_five_safe_half':'#dc2626', 'safe_half_shifted':'#db2777',
                     'forty_five_basic':'#b91c1c', 'basic_shifted':'#9f1239'}
    for i, k, valid in placements.same_turn_transitions:
        a, b = placements.circles[i], placements.circles[k]
        messages.append(f'D{a.waypoint_index+1} → D{b.waypoint_index+1}: same-turn transition {"valid" if valid else "unresolved"}.')
    if placements.coincident_waypoints:
        messages.append("Shifted to common center: "+", ".join(f"D{j+1}" for j in placements.coincident_waypoints)+".")
    rejected_handles={}
    overlap_color = '#b8860b'
    overlapping = {i for block in placements.overlap_blocks for i in block}
    for circle_index, circle in enumerate(placements.circles):
        color=colors[circle.rule] if circle.side is None else ('#0891b2' if circle.side == 'left' else '#7c3aed')
        if circle_index in overlapping:
            color = overlap_color
        tag = f'{circle.waypoint_index+1}' + (f'-{circle.side[0].upper()}' if circle.side else '')
        ax.add_patch(Circle(circle.center,circle.radius,fill=False,edgecolor=color,
                            lw=2 if circle_index in overlapping else 1,
                            linestyle='--',alpha=.9 if circle_index in overlapping else .6,zorder=4))
        alpha=np.linspace(0,np.pi/2,151)
        region=circle.region
        points=circle.center+circle.radius*(-np.cos(alpha)[:,None]*region['outgoing']
                                           +np.sin(alpha)[:,None]*region['incoming'])
        ax.plot(*points.T,color=color,lw=2.4,zorder=5)
        ax.scatter(*circle.center,marker='+',s=65,color=color,zorder=6)
        ax.scatter(*circle.vertex,marker='s',s=25,color=color,zorder=7)
        ax.annotate(f'O{tag}',circle.center,xytext=(5,5),
                    textcoords='offset points',fontsize=9,color=color)
        # Rules 1 and 3 can propose the identical nominal circle. Draw that
        # geometry once and label both rules, rather than hiding one underneath.
        groups={}
        for rejected in circle.rejected_candidates:
            key=tuple(np.r_[rejected['center'],rejected['vertex']])
            groups.setdefault(key,[]).append(rejected)
        for group in groups.values():
            rejected=group[0]
            rejected_color=rejected_colors[rejected['rule']]
            rule_numbers='/'.join(names[item['rule']].split('.')[0] for item in group)
            label=f'Rejected rule {rule_numbers}'
            rejected_handles[label]=rejected_color
            ax.add_patch(Circle(rejected['center'],circle.radius,fill=False,
                                edgecolor=rejected_color,lw=1.3,linestyle=':',alpha=.8,zorder=6))
            arc=rejected['center']+circle.radius*(-np.cos(alpha)[:,None]*region['outgoing']
                                                 +np.sin(alpha)[:,None]*region['incoming'])
            ax.plot(*arc.T,'--',color=rejected_color,lw=2,zorder=7)
            ax.scatter(*rejected['center'],marker='+',s=60,color=rejected_color,zorder=8)
            ax.scatter(*rejected['vertex'],marker='x',s=65,color=rejected_color,zorder=9)
            ax.annotate(f'D{tag}: rejected {rule_numbers}',
                        rejected['vertex'],xytext=(8,-18-14*(len(rejected_handles)-1)),
                        textcoords='offset points',fontsize=8,color=rejected_color,
                        arrowprops=dict(arrowstyle='-',color=rejected_color,lw=.7),zorder=10)
            messages.append(f'D{tag}: rejected rules {rule_numbers}; '
                            f'p=({rejected["vertex"][0]:.4f}, {rejected["vertex"][1]:.4f}) outside A_j.')

        messages.append(f'D{tag}: {names[circle.rule]}; implied vertex in directional A_j: yes.'
                        + (f' {len(circle.rejected_candidates)} earlier candidates outside A_j.'
                           if circle.rejected_candidates else ''))
    for decision in placements.aligned_sides:
        if decision.status != 'placed':
            messages.append(f'D{decision.waypoint_index+1}-{decision.side}: {decision.status}.')
    if placements.skipped_waypoints:
        messages.append('No placed circle: '+', '.join(f'D{j+1}' for j in placements.skipped_waypoints)+'.')
    def circle_label(i):
        circle = placements.circles[i]
        return f'O{circle.waypoint_index+1}' + (f'-{circle.side[0].upper()}' if circle.side else '')
    messages.append(f'Consecutive overlaps: {len(placements.overlap_pairs)} pairs, '
                    f'{len(placements.overlap_blocks)} blocks (gold). Touching alone is excluded.')
    for block_number, block in enumerate(placements.overlap_blocks, 1):
        messages.append(f'Block {block_number}: '+', '.join(circle_label(i) for i in block)+'.')
    notes.text(0,1,'\n\n'.join(fill(t,42) for t in [
        f'Independent placement: {placements.status}',
        f'{placements.elapsed_ms:.3f} ms; opposite-turn restoration and same-turn pair checks.',
        'Dashed: supporting circles. Solid: permitted quarter-arcs. Squares: implied vertices p = o + R u - R v.',
        'A_j is a waypoint region, not a center region. Red/pink dotted circles and dashed arcs are rejected proposals; crosses show their implied vertices. They are not accepted arcs.',
        *messages,
        'Gold marks remaining overlaps. See pair checks for valid or unresolved same-turn transitions. Local placements need not form a connected path.'
    ]),va='top',fontsize=9,transform=notes.transAxes)
    used={c.rule for c in placements.circles if c.side is None}
    handles=[Patch(facecolor='#16a34a',alpha=.4,label='Local regions A_j'),
             Line2D([],[],color='#94a3b8',ls=':',label='Baseline polyline')]
    if overlapping:
        handles.append(Line2D([],[],color=overlap_color,lw=2,label='Consecutive overlapping circles'))
    handles += [Line2D([],[],color=colors[rule],lw=2,label=label)
                for rule,label in names.items() if rule in used]
    for side,color in (('left','#0891b2'),('right','#7c3aed')):
        if any(c.side == side for c in placements.circles):
            handles.append(Line2D([],[],color=color,lw=2,label=f'Aligned: {side} option'))
    handles += [Line2D([],[],color=color,ls='--',marker='x',label=label)
                for label,color in rejected_handles.items()]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9)
    ax.set_title(f'Example {number}: independent circle placement')
    ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(.08)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,.1,1,1))
    return fig


def plot_circle_connections(number, baseline, placements, connection):
    """Display every safe candidate link and the contact-ordered selected chain."""
    boxes = baseline.feasibility.corridor_bounds
    width = max(b[1] for b in boxes)-min(b[0] for b in boxes)
    height = max(b[3] for b in boxes)-min(b[2] for b in boxes)
    fig, ax = plt.subplots(figsize=(10, np.clip(9*height/max(width, 1e-9)+2, 5, 10)))
    for bounds in baseline.feasibility.corridor_bounds:
        draw_region(ax, bounds, '#64748b', alpha=.12, zorder=1)
    for door in baseline.feasibility.safe_overlaps:
        if door is not None:
            draw_region(ax, door, '#a855f7', alpha=.15, zorder=2)
    if baseline.polyline is not None:
        ax.plot(*baseline.polyline.T, ':', color='#94a3b8', lw=1, zorder=3)
    selected = set(connection.selected_circles) if connection.feasible else set()
    center_labels = {}
    for i, circle in enumerate(placements.circles):
        color = '#2563eb' if i in selected else '#94a3b8'
        ax.add_patch(Circle(circle.center, circle.radius, fill=False, edgecolor=color,
                            lw=1, linestyle='--', alpha=.5, zorder=3))
        ax.scatter(*circle.center, marker='+', s=35, color=color, zorder=4)
        suffix = f'-{circle.side[0].upper()}' if circle.side else ''
        center_labels.setdefault(tuple(circle.center), []).append(f'O{circle.waypoint_index+1}{suffix}')
    for center, labels in center_labels.items():
        ax.annotate('/'.join(labels), center, xytext=(5, 5), textcoords='offset points',
                    color='#475569', fontsize=10)
    for state in connection.tangent_states:
        ax.plot(*np.array([state.start, state.end]).T, color='#0891b2', lw=1, alpha=.35, zorder=4)
        ax.scatter(*state.start, s=10, color='#0891b2', alpha=.35, zorder=4)
        ax.scatter(*state.end, s=10, color='#0891b2', alpha=.35, zorder=4)
    if connection.feasible:
        for primitive in connection.primitives:
            if primitive['kind'] == 'line':
                ax.plot(*np.array([primitive['start'], primitive['end']]).T,
                        color='#111827', lw=2.6, zorder=6)
            else:
                angles = np.linspace(primitive['enter'], primitive['leave'], 100)
                region = primitive['region']
                points = primitive['center']+primitive['radius']*(
                    -np.cos(angles)[:, None]*region['outgoing']+
                    np.sin(angles)[:, None]*region['incoming'])
                ax.plot(*points.T, color='#2563eb', lw=3, zorder=6)
        for idx in connection.chosen_states:
            state = connection.tangent_states[idx]
            ax.scatter(*state.start, s=30, facecolor='white', edgecolor='#111827', zorder=7)
            ax.scatter(*state.end, s=30, facecolor='white', edgecolor='#111827', zorder=7)
        start, end = connection.primitives[0]['start'], connection.primitives[-1]['end']
        ax.scatter(*start, marker='s', color='#16a34a', s=55, zorder=8)
        ax.scatter(*end, marker='s', color='#dc2626', s=55, zorder=8)
    handles = [Line2D([], [], color='#0891b2', alpha=.5, label='Admissible tangent candidates'),
               Line2D([], [], color='#111827', lw=2.6, label='Selected tangents'),
               Line2D([], [], color='#2563eb', lw=3, label='Selected directed arcs' if connection.geometry_only else 'Selected quarter-arc portions'),
               Line2D([], [], marker='o', color='#111827', markerfacecolor='white', ls='', label='Selected contacts'),
               Line2D([], [], marker='s', color='#16a34a', ls='', label='First-circle entry'),
               Line2D([], [], marker='s', color='#dc2626', ls='', label='Last-circle exit')]
    if not connection.feasible:
        handles = handles[:1]
    elif connection.skipped_waypoints:
        handles.append(Line2D([], [], color='#94a3b8', ls='--', label='Unused supporting circles'))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10, frameon=False)
    length = f'; length {connection.length:.3f} m' if connection.feasible else ''
    ax.set_title(f'Example {number}: {connection.status.replace("_", " ")}\n'
                 f'{connection.attempted_pairs} circle pairs; {len(connection.tangent_states)} tangent states'
                 f'; {connection.elapsed_ms:.2f} ms{length}', fontsize=12)
    details = ('Circle skipping enabled; first/last groups retained.' if connection.allow_skipping else
               'Consecutive groups only.')
    if connection.skipped_waypoints:
        details += ' Skipped: '+', '.join(f'D{j+1}' for j in connection.skipped_waypoints)+'.'
    details += (' Tangent intersections checked; containment and arc intersections pending.'
                if connection.geometry_only else ' No boundary-pose connections or global self-intersection audit.')
    if not connection.feasible:
        blocked = sorted({placements.circles[i].waypoint_index+1 for i in connection.contact_order_blocks})
        details = connection.reason + (f' Contact-order blocks at D: {blocked}.' if blocked else '')
        failures = sorted({reason for _, _, reasons in connection.rejected_pairs for reason in reasons.split(',')})
        if failures:
            details += ' Rejected links: '+', '.join(reason.replace('_', ' ') for reason in failures)+'.'
    fig.text(.5, .14, fill(details, 110), ha='center', fontsize=10)
    ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(.08)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, .2, 1, 1))
    return fig


def plot_circle_safe_arcs(number, baseline, placements, safe_arcs, elapsed_ms, connection=None,
                         *, axis=None, title=None):
    """Highlight green intervals and optional compatible tangent-chain shortcuts."""
    own_figure = axis is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax, fig = axis, axis.figure
    for bounds in baseline.feasibility.corridor_bounds:
        draw_region(ax, bounds, '#64748b', alpha=.09, zorder=1)
    boxes = {box for result in safe_arcs for box in result.eroded_corridors}
    for xmin, xmax, ymin, ymax in sorted(boxes):
        if xmin <= xmax and ymin <= ymax:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False,
                                   edgecolor='#64748b', linestyle=':', lw=1, alpha=.7, zorder=2))
    if baseline.polyline is not None:
        ax.plot(*baseline.polyline.T, ':', color='#94a3b8', lw=1, zorder=2)
    labels = {}
    for result in safe_arcs:
        circle = placements.circles[result.circle_index]
        ax.add_patch(Circle(circle.center, circle.radius, fill=False,
                            edgecolor='#b8860b', linestyle='--', lw=1.2, alpha=.65, zorder=3))
        ax.scatter(*circle.center, marker='+', s=45, color='#475569', zorder=5)
        tag = f'O{circle.waypoint_index+1}' + (f'-{circle.side[0].upper()}' if circle.side else '')
        labels.setdefault(tuple(circle.center), []).append(tag)
        for start, end in result.intervals:
            # Sampling is for rendering only; interval endpoints are analytic.
            angles = np.linspace(start, end, max(2, int(100*(end-start))+2))
            points = circle.center+circle.radius*np.column_stack((np.cos(angles), np.sin(angles)))
            ax.plot(*points.T, color='#009e73', lw=2.8, solid_capstyle='butt', zorder=4)
            if start == end:
                ax.scatter(*points[0], color='#009e73', s=18, zorder=5)
    for center, names in labels.items():
        ax.annotate('/'.join(names), center, xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='#334155')
    if connection is not None:
        for state in connection.tangent_states:
            ax.plot(*np.array([state.start, state.end]).T, color='#38bdf8', lw=1, alpha=.4, zorder=3)
            ax.scatter(*np.array([state.start, state.end]).T, color='#0284c7', s=12, zorder=5)
        for i in (() if connection.feasible else connection.contact_order_blocks):
            circle = placements.circles[i]
            ax.scatter(*circle.center, s=150, facecolors='none', edgecolors='#dc2626', lw=1.5, zorder=6)
        if connection.feasible:
            for primitive in connection.primitives:
                if primitive['kind'] != 'arc':
                    continue
                angles = np.linspace(primitive['enter'], primitive['leave'], 100)
                region = primitive['region']
                points = primitive['center']+primitive['radius']*(
                    -np.cos(angles)[:, None]*region['outgoing']+
                    np.sin(angles)[:, None]*region['incoming'])
                ax.plot(*points.T, color='#2563eb', lw=3.1, zorder=6)
            for idx in connection.chosen_states:
                state = connection.tangent_states[idx]
                ax.plot(*np.array([state.start, state.end]).T, color='#111827', lw=2, zorder=6)
                ax.scatter(*np.array([state.start, state.end]).T, s=22,
                           facecolor='white', edgecolor='#111827', zorder=7)
    handles = [Line2D([], [], color='#b8860b', ls='--', label='Repositioned supporting circles'),
               Line2D([], [], color='#009e73', lw=3.5, label='Collision-free arc portions'),
               Line2D([], [], color='#64748b', ls=':', label='Eroded corridor boundaries'),
               Line2D([], [], color='#94a3b8', ls=':', label='Baseline polyline')]
    if connection is not None:
        handles.extend((Line2D([], [], color='#38bdf8', alpha=.5, label='Admissible tangent candidates'),
                        Line2D([], [], color='#111827', lw=2, marker='o', markerfacecolor='white',
                               label='Selected tangents and contacts'),
                        Line2D([], [], color='#2563eb', lw=3, label='Selected directed arcs')))
        if connection.contact_order_blocks and not connection.feasible:
            handles.append(Line2D([], [], color='#dc2626', marker='o', markerfacecolor='none',
                                  ls='', label='Incompatible contact order'))
    status = f'\n{connection.status}; connection {connection.elapsed_ms:.3f} ms' if connection is not None else ''
    ax.set_title(title or f'Example {number}: collision-free arcs after circle repairs\n'
                 f'{len(safe_arcs)} circles; angular-interval computation {elapsed_ms:.3f} ms{status}', fontsize=12)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(.08); ax.grid(alpha=.2)
    if own_figure:
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10, frameon=False)
    note = 'Safe arcs include the certified fillet bridging the gap between eroded corridors.'
    if connection is not None:
        note = 'Green arcs checked. '
        note += ('Shortcut footprints checked; consecutive footprints unchecked. '
                 if connection.footprint_scope == 'shortcuts' else
                 'Straight footprints checked. ' if connection.tangent_containment_checked else
                 'Straight footprints unchecked. ')
        note += ('Overlap order checked. ' if connection.ordered_overlap_crossings_checked else
                 'Overlap order unchecked. ')
        note += 'Global intersections unchecked.'
    if connection is not None and connection.contact_order_blocks:
        if not connection.feasible:
            note = 'Contact order blocked at '+', '.join(f'O{placements.circles[i].waypoint_index+1}'
                                                       for i in connection.contact_order_blocks)+'. '+note
    if connection is not None and connection.skipped_waypoints:
        note = 'Skipped: '+', '.join(f'O{j+1}' for j in connection.skipped_waypoints)+'. '+note
    if own_figure:
        fig.text(.5, .12, fill(note, 125), ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .17, 1, 1))
    return fig


def plot_tangent_rule_comparison(number, baseline, placements, safe_arcs, graph, simple):
    """Compare complete search and first-chain repair with all straight footprints checked."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    for ax, name, result in zip(axes, ('Full graph: shortest chain', 'Local repair: first valid chain'),
                               (graph, simple)):
        length = f'{result.length:.3f} m' if result.feasible else 'unresolved'
        title = f'{name}\n{length}; {result.attempted_pairs} pairs; {result.elapsed_ms:.3f} ms'
        plot_circle_safe_arcs(number, baseline, placements, safe_arcs, 0., result, axis=ax, title=title)
        chain = ', '.join(f'O{placements.circles[i].waypoint_index+1}'
                          + (f'-{placements.circles[i].side[0].upper()}' if placements.circles[i].side else '')
                          for i in result.selected_circles)
        ax.text(.5, -.12, fill('Selected: '+chain if result.feasible else result.reason, 70),
                transform=ax.transAxes, ha='center', va='top', fontsize=10)
    axes[1].set_ylabel('')
    fig.suptitle(f'Example {number}: alternative tangent-chain rules', fontsize=14)
    fig.legend(handles=[Line2D([], [], color='#009e73', lw=3, label='Available green arcs'),
                        Line2D([], [], color='#2563eb', lw=3, label='Selected directed arcs'),
                        Line2D([], [], color='#111827', lw=2, label='Selected tangents'),
                        Line2D([], [], color='#38bdf8', lw=1, label='Admissible candidate tangents')],
               loc='lower center', ncol=4, frameon=False)
    fig.text(.5, .055, 'Green contacts, directed arcs and all straight footprints checked. '
             'Overlap order and global intersections unchecked.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .95))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=list(EXAMPLE_NUMBERS),
                        default=EXAMPLE_NUM)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--save", type=Path,
                        default=Path(SAVE_FIGURE) if SAVE_FIGURE else None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument('--connection-rule', choices=('graph', 'simple', 'compare'), default=CONNECTION_RULE)
    parser.add_argument("--boundary-directions", action="store_true", default=USE_BOUNDARY_DIRECTIONS,
                        help="Infer virtual boundary directions from the map's start/end positions")
    args = parser.parse_args()
    if args.repetitions < 1 or args.batches < 1 or args.warmup < 0:
        parser.error("repetitions and batches must be positive; warmup nonnegative")
    corridors, start_pose, end_pose, robot = example_corridor_sequence(args.example)
    report = analyze_orthogonal_polyline_feasibility(corridors, robot)
    timings = benchmark_check(corridors, robot, args.repetitions,
                              args.batches, args.warmup)
    print(f"Example {args.example}: {report.status}; feasible={report.feasible}")
    if report.reason:
        print(report.reason)
    print("Passage directions:", ", ".join(report.passage_directions) or "unavailable")
    print(f"Boolean feasibility check (µs/call): min={timings[0]:.2f}, "
          f"median={timings[1]:.2f}, max={timings[2]:.2f}")
    print(f"{args.batches} batches × {args.repetitions} calls; "
          "setup and backward visualization pass excluded.")
    start = perf_counter_ns()
    baseline = compute_filleted_baseline(
        corridors, robot, use_joint_solver=USE_JOINT_SOLVER,
        initial_position=start_pose[:2] if args.boundary_directions else None,
        final_position=end_pose[:2] if args.boundary_directions else None,
    )
    construction_ms = (perf_counter_ns() - start) / 1e6
    print(f"Fillet construction: {baseline.status}; "
          f"time={construction_ms:.3f} ms; method={baseline.selection_method}")
    if baseline.reason:
        print(baseline.reason)
    placements = place_refinement_circles(baseline, robot)
    placements = restore_opposite_turn_overlaps(placements, baseline)
    placements = repair_same_turn_overlaps(placements, baseline, robot)
    arc_start = perf_counter_ns()
    safe_arcs = compute_circle_safe_arcs(placements, baseline, robot)
    arc_ms = (perf_counter_ns()-arc_start)/1e6
    simple = footprint_graph = None
    if args.connection_rule == 'simple':
        connection = connect_simple_tangent_chain(placements, baseline, robot, safe_arcs=safe_arcs)
    else:
        connection = connect_safe_arc_circles(placements, baseline, robot, safe_arcs=safe_arcs,
                                             allow_skipping=ALLOW_SAFE_ARC_SKIPPING)
    if args.connection_rule == 'compare':
        simple = connect_simple_tangent_chain(placements, baseline, robot, safe_arcs=safe_arcs)
        footprint_graph = connect_safe_arc_circles(placements, baseline, robot, safe_arcs=safe_arcs,
                                                  check_overlap_order=False)
        print(f'Local first-chain repair, all straight footprints: {simple.status}; {simple.elapsed_ms:.3f} ms; '
              f'{simple.attempted_pairs} pairs; length={simple.length}')
        print(f'Full graph, all straight footprints: {footprint_graph.status}; {footprint_graph.elapsed_ms:.3f} ms; '
              f'{footprint_graph.attempted_pairs} pairs; length={footprint_graph.length}')
        print('Same selected circles (straight-footprint comparison):',
              simple.feasible == footprint_graph.feasible and simple.selected_circles == footprint_graph.selected_circles)
        print('Local repair rounds (occupied group indices):', simple.repair_rounds)
    print(f'Independent circle placement: {placements.status}; '
          f'{len(placements.circles)} circles; {placements.elapsed_ms:.3f} ms')
    print(f'Consecutive circle overlaps: {len(placements.overlap_pairs)} pairs; '
          f'{len(placements.overlap_blocks)} blocks')
    print('Restored baseline circles:', tuple(j+1 for j in placements.restored_waypoints))
    print('Shifted to common center:', tuple(j+1 for j in placements.coincident_waypoints))
    print('Same-turn pair checks:', placements.same_turn_transitions)
    print(f'Collision-free angular intervals: {arc_ms:.3f} ms.')
    print(f'Green-arc tangent chain: {connection.status}; {connection.elapsed_ms:.3f} ms; '
          f'{len(connection.tangent_states)} candidates; length={connection.length}')
    print('Skipped circles:', tuple(j+1 for j in connection.skipped_waypoints))
    clearance_rejections = tuple((placements.circles[i].waypoint_index+1,
                                  placements.circles[k].waypoint_index+1)
                                 for i, k, reason in connection.rejected_pairs
                                 if 'straight_footprint_clearance' in reason)
    if clearance_rejections:
        print('Tangent pairs rejected by straight footprint clearance:', clearance_rejections)
    if connection.contact_order_blocks:
        print('Some continuations rejected by contact order at circles:',
              tuple(placements.circles[i].waypoint_index+1 for i in connection.contact_order_blocks))
    for result in safe_arcs:
        circle = placements.circles[result.circle_index]
        tag = f'D{result.waypoint_index+1}' + (f'-{circle.side}' if circle.side else '')
        print(f'  {tag} safe angles [degrees, CCW from +x]: '
              f'{[(round(float(np.degrees(a)), 3), round(float(np.degrees(b)), 3)) for a, b in result.intervals]}')
    for circle in placements.circles:
        print(f'  D{circle.waypoint_index+1}: {circle.rule}; '
              f'implied vertex in A_j={circle.in_admissible_region}; '
              f'earlier A_j rejections={len(circle.rejected_candidates)}; side={circle.side}')
    for decision in placements.aligned_sides:
        print(f'  Aligned D{decision.waypoint_index+1} {decision.side}: {decision.status}; '
              f'boundary intersection points={decision.intersection_count}')
    if PLOT_RESULTS and not args.no_plot or args.save:
        figures = [('', plot_filleted_baseline(args.example, baseline, construction_ms, robot,
                                               start_pose[:2], end_pose[:2])),
                   ('_placement', plot_independent_circle_placement(args.example, baseline, placements)),
                   ('_safe_arcs', plot_circle_safe_arcs(args.example, baseline, placements, safe_arcs, arc_ms, connection))]
        if simple is not None:
            figures.append(('_chain_comparison', plot_tangent_rule_comparison(
                           args.example, baseline, placements, safe_arcs, footprint_graph, simple)))
        if PLOT_FEASIBILITY:
            figures.append(('_feasibility',plot_example(args.example,report,robot,timings,
                                                       start_pose[:2],end_pose[:2])))
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            for suffix,fig in figures:
                path=args.save.with_name(args.save.stem+suffix+args.save.suffix)
                fig.savefig(path,dpi=180,bbox_inches='tight')
                print(f'Saved figure to {path}')
        if PLOT_RESULTS and not args.no_plot:
            plt.show()
        else:
            for _,fig in figures:
                plt.close(fig)


if __name__ == "__main__":
    main()
