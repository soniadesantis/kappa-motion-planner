"""An integer-coordinate running example for the thesis planner procedure.

Run this file with VS Code's Run button. The first stage uses the finalized
feasibility algorithm. The polyline uses exact full fillet-set propagation and
backward reconstruction, so its waypoints can be reused in the fillet stage. Geometry is
shared by all panels and can be reused for subsequent planner stages.

CLI: python experiments/bicycle_journal_paper/example_planner_thesis.py --no-show
Exports vector PDF and PNG figures, plus an interval table figure.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon

from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility, compute_filleted_baseline_exact,
    compute_local_boundary_fillets,
)
from kappa_planner.helpers.fillet_backtracking import region_slice
from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_extension import extend_perpendicular_corridors

SHOW_POLYLINE = True
SAVE_FIGURES = True
SHOW_FIGURES = True
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures" / "planner_thesis"

# (xmin, xmax, ymin, ymax), metres, before covered-edge extension.
# C4 protrudes below C3; C5 ends inside C4 in x but protrudes above it.
# These partly covered edges stay put, unlike C5's fully covered left edge.
SEED_CORRIDOR_BOUNDS = (
    (0, 4, 0, 8),
    (0, 13, 0, 4),
    (9, 19, -1, 5),
    (14, 18, -2, 12),
    (8, 17, 9, 13),
    (7, 11, 6, 13),
)
CORRIDOR_HEADINGS = (1, 0, 0, 1, 2, 3)  # Integer quarter turns.
CORRIDOR_TILTS = tuple(h * np.pi / 2 for h in CORRIDOR_HEADINGS)
CORRIDOR_BOUNDS, EDGE_EXTENSIONS = extend_perpendicular_corridors(
    SEED_CORRIDOR_BOUNDS, CORRIDOR_HEADINGS
)
FOOTPRINT_RADIUS = 1
TURNING_RADIUS = 3
STYLE = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 23,
    "axes.titlesize": 22, "axes.labelsize": 23,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}
DOOR_COLOR = "#0072B2"
REACHABLE_COLOR = "#009E73"
SPACING_COLOR = "#D55E00"
EQUALITY_COLOR = "#CC79A7"


def build_example():
    """Return fresh corridor objects and the explicit circular-robot model."""
    corridors = []
    for (xmin, xmax, ymin, ymax), tilt in zip(CORRIDOR_BOUNDS, CORRIDOR_TILTS):
        dx, dy = xmax - xmin, ymax - ymin
        vertical = abs(np.sin(tilt)) > .5
        corridors.append(CorridorWorld(
            dx if vertical else dy, dy if vertical else dx,
            [(xmin + xmax) / 2, (ymin + ymax) / 2],
            tilt,
        ))
    return corridors, SimpleNamespace(r=FOOTPRINT_RADIUS, R=TURNING_RADIUS)


def rectangle(ax, bounds, color, *, alpha=.2, hatch=None, zorder=2,
              edgecolor=None, linewidth=1.4):
    xmin, xmax, ymin, ymax = bounds
    if abs(xmax - xmin) < 1e-9 and abs(ymax - ymin) < 1e-9:
        ax.scatter([(xmin + xmax) / 2], [(ymin + ymax) / 2], color=color,
                   s=50, marker="s", zorder=zorder)
        return
    if abs(xmax - xmin) < 1e-9 or abs(ymax - ymin) < 1e-9:
        ax.plot([xmin, xmax], [ymin, ymax], color=color, linewidth=3,
                zorder=zorder)
        return
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor=to_rgba(color, alpha),
                           edgecolor=edgecolor or to_rgba(color, alpha),
                           hatch=hatch, linewidth=linewidth, zorder=zorder))


def base_panel(ax, title, *, corridor_labels=False):
    for bounds in CORRIDOR_BOUNDS:
        rectangle(ax, bounds, "#64748b", alpha=.10, zorder=1,
                  edgecolor="black", linewidth=.6)
    if corridor_labels:
        # Drawing-style callouts: each leader ends on the indicated corridor wall.
        callouts = (
            ((2, 8), (2, 9.5)),
            ((6, 0), (6, -1.5)),
            ((11, -1), (11, -1.85)),
            ((14, 6), (12, 6)),
            ((12, 9), (12, 8)),
            ((7, 10.5), (5.3, 10.5)),
        )
        for j, (attachment, label) in enumerate(callouts, start=1):
            ax.annotate(rf"$\mathcal{{C}}_{j}$", xy=attachment, xytext=label,
                        ha="center", va="center", zorder=5, fontsize=23,
                        bbox=dict(facecolor="white", edgecolor="#444444",
                                  linewidth=.7, boxstyle="square,pad=0.2"),
                        arrowprops=dict(arrowstyle="-", color="#444444",
                                        linewidth=.8, shrinkA=0, shrinkB=0))
    ax.set(title=title, xlim=(-.7, 19.7), ylim=(-2.7, 13.7),
           xlabel="$x$ [m]", ylabel="$y$ [m]")
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 20, 2))
    ax.set_yticks(range(-2, 14, 2))
    ax.grid(alpha=.14)
    ax.set_axisbelow(True)


def draw_doors(ax, report, *, labels=True):
    for j, door in enumerate(report.safe_overlaps, start=1):
        rectangle(ax, door, DOOR_COLOR, alpha=.2)
        if labels:
            ax.text((door[0] + door[1]) / 2, (door[2] + door[3]) / 2,
                    rf"$\mathcal{{D}}_{j}$", ha="center", va="center",
                    color=DOOR_COLOR, zorder=5)



def draw_passages(ax, report):
    for a, b, direction in zip(report.safe_overlaps, report.safe_overlaps[1:],
                               report.passage_directions):
        if direction in ("right", "left"):
            y = (max(a[2], b[2]) + min(a[3], b[3])) / 2
            start = (a[1], y) if direction == "right" else (a[0], y)
            end = (b[0], y) if direction == "right" else (b[1], y)
        else:
            x = (max(a[0], b[0]) + min(a[1], b[1])) / 2
            start, end = (x, a[3]), (x, b[2])
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6))


def create_progressive_figure(report, points=None):
    """Four views of the same environment, with the constructed waypoint chain."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, title in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        base_panel(ax, title, corridor_labels=ax is axes[0, 0])
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    draw_doors(axes[0, 1], report)
    draw_passages(axes[0, 1], report)
    ax = axes[1, 0]
    draw_doors(ax, report, labels=False)
    # Separate longitudinal cuts (spacing) from transverse cuts (equality).
    for j, direction in enumerate(report.passage_directions, start=1):
        door = report.safe_overlaps[j]
        intervals = (report.x_reachable[j], report.y_reachable[j])
        longitudinal = 0 if direction in ("right", "left") else 1
        transverse = 1 - longitudinal
        for axis, color in ((longitudinal, SPACING_COLOR),
                            (transverse, EQUALITY_COLOR)):
            lo, hi = door[2*axis:2*axis+2]
            kept_lo, kept_hi = intervals[axis]
            for cut_lo, cut_hi in ((lo, kept_lo), (kept_hi, hi)):
                if cut_hi - cut_lo <= 1e-9:
                    continue
                cut = list(door)
                cut[2*axis:2*axis+2] = (cut_lo, cut_hi)
                if axis == transverse:
                    cut[2*longitudinal:2*longitudinal+2] = intervals[longitudinal]
                rectangle(ax, cut, color, alpha=.65, zorder=3)
    for x, y in zip(report.x_reachable, report.y_reachable):
        rectangle(ax, (*x, *y), REACHABLE_COLOR, alpha=.45, zorder=4)
    ax = axes[1, 1]
    draw_doors(ax, report, labels=False)
    if SHOW_POLYLINE:
        if points is None:
            raise ValueError("The polyline panel needs a successful construction.")
        ax.plot(*points.T, color="#222222", lw=2, zorder=5)
        ax.scatter(*points.T, color="#222222", s=35, zorder=6)
    else:
        draw_passages(ax, report)
    handles = [
        Patch(facecolor=DOOR_COLOR, alpha=.25, label="Safe overlap"),
        Patch(facecolor=REACHABLE_COLOR, alpha=.45, label="Reachable region"),
        Patch(facecolor=SPACING_COLOR, alpha=.65, label="Removed by 2R"),
        Patch(facecolor=EQUALITY_COLOR, alpha=.65,
              label="Removed by equality"),
    ]
    if SHOW_POLYLINE:
        handles.append(Line2D([], [], color="#222222", marker="o",
                              label="Polyline"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=21,
               bbox_to_anchor=(.5, .015), frameon=False)
    # Reserve space for the legend; match panel height to the equal-aspect maps.
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.95,
                        wspace=.20, hspace=.22)
    # Inset slightly so the complete thin frame survives PNG and PDF export.
    fig.add_artist(Rectangle((.003, .003), .994, .994,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=.8, clip_on=False,
                             zorder=100))
    return fig


def draw_reachable_fillet_region(ax, region, reachable):
    """Draw A_j intersected with R_j, sampling only the curved display boundary."""
    if region is None:  # Collinear waypoint: A_j = D_j.
        rectangle(ax, reachable, "#ea580c", alpha=.7, zorder=4)
        return
    clipped = dict(region)
    clipped['low'] = np.maximum(region['low'], [reachable[0], reachable[2]])
    clipped['high'] = np.minimum(region['high'], [reachable[1], reachable[3]])
    if np.any(clipped['low'] > clipped['high']):
        return
    best_y = clipped['low'][1] if region['frame'][1].sum() > 0 else clipped['high'][1]
    projection = region_slice(clipped, 1, best_y)
    if projection is None:
        return
    xs = np.linspace(*projection, 401)
    junction = -region['offset'][0]
    if projection[0] < junction < projection[1]:
        xs = np.unique(np.r_[xs, junction])
    lower, upper = [], []
    for x in xs:
        interval = region_slice(clipped, 0, x)
        if interval is not None:
            lower.append((x, interval[0]))
            upper.append((x, interval[1]))
    if not lower:
        return
    vertices = np.array(lower + upper[::-1])
    if np.any(np.ptp(vertices, axis=0) < 1e-9):
        rectangle(ax, (vertices[:,0].min(), vertices[:,0].max(),
                       vertices[:,1].min(), vertices[:,1].max()),
                  "#ea580c", zorder=4)
    else:
        ax.add_patch(Polygon(vertices, facecolor="#ea580c", edgecolor="#ea580c",
                             alpha=.7, linewidth=1.2, zorder=4))


def draw_exact_reachable_set(ax, reachable):
    """Render F_j using analytic slices; sampling is only for the display."""
    xmin, xmax, ymin, ymax = reachable.bounds
    if min(xmax-xmin, ymax-ymin) < 1e-9:
        rectangle(ax, reachable.bounds, REACHABLE_COLOR, alpha=.7, zorder=4)
        return
    # Include constraint junctions so straight/curved boundary changes are shown.
    xs = np.unique(np.r_[np.linspace(xmin, xmax, 501),
                         [g.center[0] for g in reachable.constraints
                          if xmin < g.center[0] < xmax]])
    lower, upper = [], []
    for x in xs:
        interval = reachable.slice_interval(0, x)
        if interval is not None:
            lower.append((x, interval[0]))
            upper.append((x, interval[1]))
    vertices = np.array(lower + upper[::-1])
    ax.add_patch(Polygon(vertices, facecolor=REACHABLE_COLOR,
                         edgecolor=REACHABLE_COLOR, alpha=.7,
                         linewidth=1.2, zorder=4))


def create_exact_propagation_figure(report, baseline):
    """Corridors, independent local A_j, then full F_j and a recovered witness.

    Endpoint directions are deliberately absent from this internal problem.
    At aligned and endpoint waypoints A_j=D_j. F_j includes all preceding
    local fillet conditions and signed 2R constraints, not future constraints.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.7))
    for ax, title in zip(axes, ('(a)', '(b)', '(c)')):
        base_panel(ax, title, corridor_labels=ax is axes[0])
    for ax in axes[1:]:
        ax.set_ylabel("")
        draw_doors(ax, report, labels=False)
    for door, region in zip(report.safe_overlaps, baseline.fillet_regions):
        # Use D_j, not R_j: this panel depicts the independent local sets.
        draw_reachable_fillet_region(axes[1], region, door)
        draw_reachable_fillet_region(axes[2], region, door)
    for reachable in baseline.fillet_reachability.reachable_sets:
        draw_exact_reachable_set(axes[2], reachable)
    axes[2].plot(*baseline.polyline.T, '-o', color='#222222', lw=2,
                 markersize=5, zorder=6)
    fig.legend(handles=[
        Patch(facecolor=DOOR_COLOR, alpha=.2, label=r'$\mathcal{D}_j$'),
        Patch(facecolor='#ea580c', alpha=.7, label=r'$\mathcal{A}_j$'),
        Patch(facecolor=REACHABLE_COLOR, alpha=.7, label=r'$\mathcal{F}_j$'),
        Line2D([], [], color='#222222', marker='o', label='Polyline'),
    ], loc='lower center', ncol=4, fontsize=21,
       bbox_to_anchor=(.5, .015), frameon=False)
    fig.subplots_adjust(left=.06, right=.985, bottom=.21, top=.93, wspace=.16)
    fig.add_artist(Rectangle((.003, .003), .994, .994,
                             transform=fig.transFigure, fill=False,
                             edgecolor='black', linewidth=.8, zorder=100))
    return fig


def create_fillet_figure(report, baseline, boundary):
    """Local admissible regions and the reconstructed chain with quarter arcs.

    This running geometry has exactly one validated arc at each fixed endpoint.
    Those arcs impose no connection to a prescribed initial/final pose.
    """
    if len(boundary.initial) != 1 or len(boundary.final) != 1:
        raise ValueError("This figure expects one local arc option at each endpoint.")
    regions, fillets = list(baseline.fillet_regions), list(baseline.fillets)
    regions[0], regions[-1] = boundary.initial_regions[0], boundary.final_regions[0]
    fillets[0], fillets[-1] = boundary.initial[0][1], boundary.final[0][1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7))
    for ax, title in zip(axes, ('(a)', '(b)')):
        base_panel(ax, title)
    axes[1].set_ylabel("")
    for j, (x, y) in enumerate(zip(report.x_reachable, report.y_reachable)):
        reachable = (*x, *y)
        rectangle(axes[0], reachable, DOOR_COLOR, alpha=.35, zorder=2)
        draw_reachable_fillet_region(axes[0], regions[j], reachable)
    points = baseline.polyline
    axes[1].plot(*points.T, '--o', color='#64748b', lw=1.1, markersize=4, zorder=3)
    for j in range(len(points)-1):
        start = fillets[j].outgoing_tangent if fillets[j] else points[j]
        end = fillets[j+1].incoming_tangent if fillets[j+1] else points[j+1]
        axes[1].plot([start[0], end[0]], [start[1], end[1]],
                     color='#222222', lw=2.2, zorder=5)
    for fillet in fillets:
        if fillet is None:
            continue
        radial = fillet.incoming_tangent-fillet.center
        angles = np.arctan2(radial[1], radial[0]) + np.linspace(0, fillet.signed_angle, 151)
        arc = fillet.center + fillet.radius*np.column_stack((np.cos(angles), np.sin(angles)))
        axes[1].plot(*arc.T, color='#ea580c', lw=2.8, zorder=6)
    fig.legend(handles=[
        Patch(facecolor=DOOR_COLOR, alpha=.35, label=r'$\mathcal{R}_j$'),
        Patch(facecolor='#ea580c', alpha=.7, label=r'$\mathcal{A}_j\cap\mathcal{R}_j$'),
        Line2D([], [], color='#64748b', linestyle='--', marker='o', label='Polyline'),
        Line2D([], [], color='#222222', lw=2.2, label='Straights'),
        Line2D([], [], color='#ea580c', lw=2.8, label='Fillets'),
    ], loc='lower center', ncol=5, fontsize=19, bbox_to_anchor=(.5,.015), frameon=False)
    fig.subplots_adjust(left=.085, right=.98, bottom=.21, top=.92, wspace=.20)
    fig.add_artist(Rectangle((.003,.003),.994,.994,transform=fig.transFigure,
                             fill=False,edgecolor='black',linewidth=.8,zorder=100))
    return fig


def interval_text(interval):
    # Suppress rotation roundoff in labels only; retain the computed geometry.
    lo, hi = (0.0 if abs(value) < 1e-9 else value for value in interval)
    return f"[{lo:g}, {hi:g}]"


def create_interval_table(report):
    """Numerical companion exposing each forward-propagation step."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    rows = []
    explanations = ["Initialization", "Right: spacing inactive",
                    "Right: spacing + equality",
                    "Up: equality restricts x", "Left: equality restricts y"]
    for j, (door, x, y) in enumerate(zip(report.safe_overlaps, report.x_reachable,
                                        report.y_reachable), start=1):
        rows.append([str(j), interval_text(door[:2]), interval_text(door[2:]),
                     interval_text(x), interval_text(y), explanations[j-1]])
    table = ax.table(cellText=rows,
                     colLabels=["j", "$X_j$", "$Y_j$", "$\\mathcal{R}_j^x$",
                                "$\\mathcal{R}_j^y$", "Incoming constraint"],
                     cellLoc="center", colWidths=[.05, .14, .14, .14, .14, .29],
                     bbox=[0, .28, 1, .6])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_facecolor("#e2e8f0" if row == 0 else "white")
    ax.set_title("Exact interval propagation, with 2R = 6 m", fontsize=21)
    ax.text(.5, .14, r"$\mathcal{R}_3^x=[15,17]\cap[10+6,\infty)=[16,17]$",
            ha="center", fontsize=20)
    ax.text(.5, .04, r"$\mathcal{R}_4^x=[15,16]\cap[16,17]=\{16\}$",
            ha="center", fontsize=20)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    corridors, robot = build_example()
    report = analyze_orthogonal_polyline_feasibility(corridors, robot,
                                                    compute_viable=False)
    if not report.feasible:
        raise ValueError("The running example is infeasible: " + report.reason)
    expected = ("right", "right", "up", "left")
    if report.passage_directions != expected:
        raise ValueError("Running example passage sequence changed.")
    print("Feasibility:", report.status)
    for change in EDGE_EXTENSIONS:
        print(f"Extended C{change['corridor']+1} {change['edge']}: "
              f"{change['before']} -> {change['after']} "
              f"(covered by C{change['neighbor']+1})")
    print("Corridors (xmin, xmax, ymin, ymax):")
    for j, bounds in enumerate(CORRIDOR_BOUNDS, start=1):
        print(f"  C{j}: {bounds}")
    print("Safe overlaps and forward reachable intervals:")
    for j, (door, x, y) in enumerate(zip(report.safe_overlaps, report.x_reachable,
                                        report.y_reachable), start=1):
        print(f"  D{j}: X={interval_text(door[:2])}, Y={interval_text(door[2:])}; "
              f"reachable x={interval_text(x)}, y={interval_text(y)}")
    baseline = compute_filleted_baseline_exact(corridors, robot)
    if not baseline.feasible:
        raise ValueError("Running example fillet construction failed: " + baseline.reason)
    boundary = compute_local_boundary_fillets(baseline, robot)
    points = baseline.polyline
    print("Waypoint construction:", baseline.selection_method,
          "(analytic forward sets and backward reconstruction)")
    print("Full fillet reachable sets (bounding intervals):")
    for j, reachable in enumerate(baseline.fillet_reachability.reachable_sets, start=1):
        print(f"  F{j}: x={interval_text(reachable.bounds[:2])}, "
              f"y={interval_text(reachable.bounds[2:])}; "
              f"{len(reachable.constraints)} curved constraints")
    print("Waypoints:\n", points)
    print("Local endpoint arc directions:", [d for d,_ in boundary.initial],
          [d for d,_ in boundary.final])
    with plt.rc_context(STYLE):
        figures = {
            "planner_thesis_exact_fillet_propagation": create_exact_propagation_figure(
                report, baseline
            ),
            "planner_thesis_polyline_feasibility": create_progressive_figure(
                report, points
            ),
            "planner_thesis_arc_fillets": create_fillet_figure(report, baseline, boundary),
            "planner_thesis_interval_propagation": create_interval_table(report),
        }
    if SAVE_FIGURES:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                path = args.output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=220)
                print("Saved:", path)
    if SHOW_FIGURES and not args.no_show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)


if __name__ == "__main__":
    main()
