"""Two views of baseline construction: local A_j, then reachable R_j and path.

Choose EXAMPLE_NUM, USE_BOUNDARY_DIRECTIONS and CONNECT_BOUNDARIES below.
Optional physical connections reuse the bicycle boundary construction.
Timing covers one complete baseline construction, excluding imports, map creation,
display geometry and plotting. Refinement is not run by this example.

CLI: python experiments/bicycle_journal_paper/example_baseline_construction.py \
    --example 7 --save /tmp/baseline.png --no-plot
"""

import argparse
from dataclasses import replace
from pathlib import Path
from statistics import median
from textwrap import fill
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Polygon, Rectangle

from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility,
    check_orthogonal_polyline_feasibility,
    compute_boundary_directed_baseline,
    compute_baseline,
)
from kappa_planner.bicycle_boundary_connections import build_baseline_boundary_connections
from kappa_planner.helpers.fillet_backtracking import region_slice
from kappa_planner.helpers.fillet_reachability import propagate_fillet_regions
from kappa_planner.helpers.corridor_union import CorridorUnion

EXAMPLE_NUM = 22  # Choose one map from 1 through 35, then press Run in VS Code.
PLOT_RESULTS = True
USE_JOINT_SOLVER = False  # Evaluate the cheap heuristic alone; enable for fallback.
BASELINE_RULE = 'baseline'  # Direct pipeline; 'segment', 'exact', 'heuristic' retain the older routines for comparison.
USE_BOUNDARY_DIRECTIONS = True  # Pose-to-centroid turn signs; otherwise perpendicular then straight.
CONNECT_BOUNDARIES = True  # Try initial/final bicycle connections after constructing the baseline.
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
        for j, fillet in enumerate(result.fillets):
            connection = (result.initial_connection if j == 0 else
                          result.final_connection if j == len(result.fillets)-1 else None)
            if fillet is None or (connection is not None and connection.connected):
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


def _erosion_display(bounds, radius):
    """Sample exact union-boundary clearance for display, including concave arcs.

    This grid is used only for drawing; construction never uses it. Eroding
    each rectangle separately would incorrectly remove safe overlap corners.
    """
    union = CorridorUnion(bounds)
    bounds = np.asarray(bounds)
    low = bounds[:, (0, 2)].min(axis=0)
    high = bounds[:, (1, 3)].max(axis=0)
    span = high-low
    counts = np.maximum(100, np.ceil(600*span/span.max()).astype(int))
    xs, ys = [np.linspace(a, b, count) for a, b, count in zip(low, high, counts)]
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = np.zeros(len(points), dtype=bool)
    for xmin, xmax, ymin, ymax in bounds:
        inside |= ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                   & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
    distance_squared = np.full(len(points), np.inf)
    for a, b in union.boundary:
        edge = b-a
        parameter = np.clip((points-a) @ edge/(edge @ edge), 0., 1.)
        delta = points-(a+parameter[:, None]*edge)
        distance_squared = np.minimum(distance_squared, np.einsum('ij,ij->i', delta, delta))
    clearance = np.sqrt(distance_squared)
    clearance[~inside] *= -1
    return xx, yy, clearance.reshape(xx.shape), union.boundary


def _draw_baseline_scene(ax, report, erosion, radius):
    xx, yy, clearance, boundary = erosion
    if clearance.max() > radius:
        ax.contourf(xx, yy, clearance, levels=[radius, clearance.max()+1.],
                    colors=['#e2e8f0'], zorder=0)
        ax.contour(xx, yy, clearance, levels=[radius], colors=['#94a3b8'],
                   linewidths=.7, zorder=1)
    for xmin, xmax, ymin, ymax in report.corridor_bounds:
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                              fill=False, edgecolor='#94a3b8', linewidth=.7,
                              linestyle='--', alpha=.7, zorder=1))
    ax.add_collection(LineCollection(boundary, colors='#475569', linewidths=1., zorder=2))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(color='#cbd5e1', alpha=.25, linewidth=.5)
    ax.set_axisbelow(True)
    low = boundary.reshape(-1, 2).min(axis=0)
    high = boundary.reshape(-1, 2).max(axis=0)
    padding = .06*(high-low)
    ax.set_xlim(low[0]-padding[0], high[0]+padding[0])
    ax.set_ylim(low[1]-padding[1], high[1]+padding[1])
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')


def _draw_reachable_set(ax, reachable):
    """Draw analytic R_j slices rather than its rectangular bounding box."""
    xmin, xmax, ymin, ymax = reachable.bounds
    if xmax-xmin <= 1e-10 or ymax-ymin <= 1e-10:
        draw_region(ax, reachable.bounds, '#2563eb', zorder=3)
        return
    lower, upper = [], []
    for x in np.linspace(xmin, xmax, 241):
        interval = reachable.slice_interval(0, x)
        if interval is not None:
            lower.append((x, interval[0]))
            upper.append((x, interval[1]))
    if lower:
        vertices = np.array(lower+upper[::-1])
        ax.add_patch(Polygon(vertices, facecolor='#60a5fa', edgecolor='#2563eb',
                             alpha=.38, linewidth=1.1, zorder=3))


def _region_label(ax, j, point, symbol, color):
    ax.annotate(rf'${symbol}_{{{j+1}}}$', point,
                xytext=(5, 8 if j % 2 == 0 else -13), textcoords='offset points',
                fontsize=9, color=color, zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.8))


def plot_baseline_construction(number, result, elapsed_ms, robot,
                               initial_position=None, final_position=None):
    """Return exactly two clean figures: local A_j and reachable R_j + baseline."""
    report = result.feasibility
    r = robot.r if hasattr(robot, 'r') else robot.width/2
    R = robot.R if hasattr(robot, 'R') else robot.max_radius
    erosion = _erosion_display(report.corridor_bounds, r)
    local_fig, local_ax = plt.subplots(figsize=(9, 7))
    path_fig, path_ax = plt.subplots(figsize=(9, 7))
    for ax in (local_ax, path_ax):
        _draw_baseline_scene(ax, report, erosion, r)
    for j, door in enumerate(report.safe_overlaps):
        if door is None:
            continue
        center = ((door[0]+door[1])/2, (door[2]+door[3])/2)
        if j >= len(result.fillet_regions):
            continue  # A construction rejected earlier has no local A_j here.
        region = result.fillet_regions[j]
        if region is None:
            directions = result.segment_directions[j:j+2]
            if (len(directions) == 2 and all(d is not None for d in directions)
                    and directions[0] != directions[1]):
                # No local region was constructed for this unresolved turn.
                local_ax.annotate(rf'$A_{{{j+1}}}$ unresolved', center, fontsize=8,
                                  color='#b91c1c', zorder=8)
                continue
            draw_region(local_ax, door, '#16a34a', alpha=.35, zorder=3)
        elif region['empty']:
            local_ax.scatter(*center, marker='x', color='#b91c1c', s=45, zorder=5)
        else:
            draw_fillet_region(local_ax, region)
        _region_label(local_ax, j, center, 'A', '#15803d')

    reachability = result.fillet_reachability
    if (reachability is None and report.feasible
            and len(result.fillet_regions) == len(report.safe_overlaps)):
        # Heuristic reports omit full R_j. Compute them only for the display,
        # after the timed baseline call; this does not change the selected path.
        reachability = propagate_fillet_regions(report, result.fillet_regions, R)
    if reachability is not None:
        for j, reachable in enumerate(reachability.reachable_sets):
            if reachable is None:
                door = report.safe_overlaps[j]
                center = ((door[0]+door[1])/2, (door[2]+door[3])/2)
                path_ax.scatter(*center, marker='x', color='#b91c1c', s=45, zorder=5)
            else:
                _draw_reachable_set(path_ax, reachable)
                bounds = reachable.bounds
                center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2)
            _region_label(path_ax, j, center, 'R', '#1d4ed8')
    path_handles = [Patch(facecolor='#60a5fa', edgecolor='#2563eb', alpha=.38,
                          label=r'Reachable sets $R_j$')]
    if result.feasible:
        points = result.polyline
        path_ax.plot(points[:, 0], points[:, 1], '--', color='#d97706',
                     linewidth=1.2, alpha=.8, zorder=4)
        path_ax.scatter(points[:, 0], points[:, 1], color='#d97706',
                        edgecolors='white', linewidths=.5, s=22, zorder=6)
        for j in range(len(points)-1):
            start = result.fillets[j].outgoing_tangent if result.fillets[j] else points[j]
            end = result.fillets[j+1].incoming_tangent if result.fillets[j+1] else points[j+1]
            path_ax.plot([start[0], end[0]], [start[1], end[1]],
                         color='#172033', linewidth=2.2, zorder=5)
        for fillet in result.fillets:
            if fillet is None:
                continue
            radial = fillet.incoming_tangent-fillet.center
            angles = np.arctan2(radial[1], radial[0])+np.linspace(0, fillet.signed_angle, 100)
            arc = fillet.center+fillet.radius*np.column_stack((np.cos(angles), np.sin(angles)))
            path_ax.plot(arc[:, 0], arc[:, 1], color='#172033', linewidth=2.2, zorder=5)
        path_handles.extend([
            Line2D([], [], color='#d97706', linestyle='--', marker='o', markersize=4,
                   label='Selected polyline'),
            Line2D([], [], color='#172033', linewidth=2.2, label='Filleted baseline'),
        ])
    for connection, color, label in (
            (result.initial_connection, '#7c3aed', 'Initial connection'),
            (result.final_connection, '#db2777', 'Final connection')):
        if connection is None or not connection.connected:
            continue
        for maneuver in connection.maneuvers:
            coordinates = maneuver.path_coordinates
            path_ax.plot(coordinates[:, 0], coordinates[:, 1],
                         color=color, linewidth=2.2, zorder=6)
        path_handles.append(Line2D([], [], color=color, linewidth=2.2, label=label))
    for ax in (local_ax, path_ax):
        draw_boundary_positions(ax, initial_position, final_position)
    local_ax.set_title(f'Example {number} · local admissible regions', loc='left', fontsize=13)
    path_ax.set_title(f'Example {number} · reachable sets and baseline', loc='left', fontsize=13)
    local_fig.legend(handles=[
        Line2D([], [], color='#475569', linewidth=1., label='Provided corridors'),
        Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label='Circular-footprint erosion'),
        Patch(facecolor='#16a34a', alpha=.4, label=r'Local regions $A_j$'),
    ], loc='lower center', ncol=3, frameon=False, fontsize=9)
    path_fig.legend(handles=path_handles, loc='lower center', ncol=3,
                    frameon=False, fontsize=9)
    status = 'Baseline found' if result.feasible else f'No baseline found ({result.status})'
    timing = f'baseline: {elapsed_ms:.3f} ms'
    if result.boundary_connections_requested:
        timing += f' · baseline + boundaries: {result.total_time_ms:.3f} ms'
    path_fig.text(.5, .065, f'{status} · {timing}',
                  ha='center', fontsize=10, color='#334155')
    local_fig.tight_layout(rect=(0, .065, 1, 1))
    path_fig.tight_layout(rect=(0, .105, 1, 1))
    return local_fig, path_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', type=int, choices=list(EXAMPLE_NUMBERS), default=EXAMPLE_NUM)
    parser.add_argument('--save', type=Path,
                        default=Path(SAVE_FIGURE) if SAVE_FIGURE else None)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--baseline-rule', choices=('baseline', 'heuristic', 'exact', 'segment'), default=BASELINE_RULE)
    parser.add_argument('--boundary-directions', dest='boundary_directions', action='store_true',
                        default=USE_BOUNDARY_DIRECTIONS,
                        help='Choose boundary turn signs from pose-to-overlap-centroid segments')
    parser.add_argument('--no-boundary-directions', dest='boundary_directions', action='store_false',
                        help='Choose perpendicular directions when feasible, otherwise straight')
    parser.add_argument('--connect-boundaries', dest='connect_boundaries', action='store_true',
                        default=CONNECT_BOUNDARIES, help='Try physical boundary connections after the baseline')
    parser.add_argument('--no-connect-boundaries', dest='connect_boundaries', action='store_false')
    args = parser.parse_args()
    corridors, start_pose, end_pose, robot = example_corridor_sequence(args.example)
    options = dict(use_joint_solver=USE_JOINT_SOLVER) if args.baseline_rule == 'heuristic' else {}
    start = perf_counter_ns()
    poses = dict(initial_pose=start_pose if args.boundary_directions or args.connect_boundaries else None,
                 final_pose=end_pose if args.boundary_directions or args.connect_boundaries else None)
    if args.baseline_rule == 'baseline':
        baseline = compute_baseline(corridors, robot, connect_boundaries=args.connect_boundaries, **poses)
    else:
        baseline = compute_boundary_directed_baseline(
            corridors, robot, method=args.baseline_rule, **options, **poses)
        baseline_ms = (perf_counter_ns()-start)/1e6
        initial_connection = final_connection = None
        if args.connect_boundaries:
            initial_connection, final_connection = build_baseline_boundary_connections(
                baseline, robot, start_pose, end_pose)
        baseline = replace(baseline, baseline_time_ms=baseline_ms,
                           total_time_ms=(perf_counter_ns()-start)/1e6,
                           boundary_connections_requested=args.connect_boundaries,
                           initial_connection=initial_connection, final_connection=final_connection)
    construction_ms = baseline.baseline_time_ms
    print(f'Example {args.example}: {baseline.status}')
    print(f'Baseline computation: {construction_ms:.3f} ms')
    if args.connect_boundaries:
        print(f'Baseline + boundary connections: {baseline.total_time_ms:.3f} ms')
        print(f'Initial connection: {baseline.initial_connection.status}; '
              f'final connection: {baseline.final_connection.status}')
    if not baseline.feasible and baseline.reason:
        print(baseline.reason)
    if (PLOT_RESULTS and not args.no_plot) or args.save:
        figures = plot_baseline_construction(
            args.example, baseline, construction_ms, robot,
            start_pose[:2] if args.boundary_directions or args.connect_boundaries else None,
            end_pose[:2] if args.boundary_directions or args.connect_boundaries else None)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            for suffix, figure in zip(('_local_regions', '_baseline'), figures):
                path = args.save.with_name(args.save.stem+suffix+args.save.suffix)
                figure.savefig(path, dpi=180, bbox_inches='tight')
                print(f'Saved figure to {path}')
        if PLOT_RESULTS and not args.no_plot:
            plt.show()
        else:
            for figure in figures:
                plt.close(figure)


if __name__ == '__main__':
    main()
