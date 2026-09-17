"""Eight candidate constructions in one grid, sharing the overview parameters.

Run with arena-env. Edit the display settings below to tune the figure;
problem parameters are read from the adjacent figure_all_paths.py.
"""

import importlib.util
from pathlib import Path
from math import cos, sin, hypot, pi, degrees

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge

from kappa_planner.geometry import Point, Pose
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory
from kappa_planner.helpers.pose_to_pose_unicycle import compute_all_pose_to_pose_trajectories
from kappa_planner.vehicle import Unicycle


DIRECTORY = Path(__file__).resolve().parent


def load_plot_module(name, path):
    """Load sibling plotting helpers without running their main blocks."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overview = load_plot_module("overview_style", DIRECTORY / "figure_all_paths.py")
construction = load_plot_module(
    "family_constructions", DIRECTORY.parent / "plot_candidate_family_examples.py"
)

# Display configuration. Geometry and vehicle settings follow the overview.
FAMILIES = ("CSC", "TCSC", "CSCT", "TCSCT")
DIRECTIONS = ((+1, +1), (+1, -1))
FIGURE_WIDTH = 11.0
FIGURE_HEIGHT = 10.5
GEOMETRY_PADDING = 0.035
AMPLITUDE_ARC_RADIUS_RATIO = 0.4
AMPLITUDE_SECTOR_COLOR = "#CC79A7"  # Reddish purple from the Okabe–Ito palette.
AMPLITUDE_SECTOR_ALPHA = 0.3
AMPLITUDE_LABEL_COLOR = "#884D70"  # Darker purple for readable angle labels.
TURN_SECTOR_COLOR = "#B87400"  # Dark orange for visible turn-angle shading.
TURN_SECTOR_ALPHA = 0.25
TURN_LABEL_COLOR = "#805100"  # Darker orange for readable turn-angle labels.
FONT_SIZE_PANEL_LABEL = 20
FONT_SIZE_SYMBOL = 17
PATH_COLOR = overview.NORMAL_TRAJECTORY_COLOR
PATH_LINEWIDTH = 2.0
CONSTRUCTION_COLOR = "0.55"
CONSTRUCTION_LINEWIDTH = 0.9
POINT_MARKERSIZE = 3
PANEL_HORIZONTAL_SPACE = 0.04
PANEL_VERTICAL_SPACE = 0.14
SHOW_SYMBOLS = True
SHOW_AUXILIARY_GEOMETRY = True
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_FILENAME = "candidate_family_geometry"

# Configure the imported construction helpers for this denser overview.
construction.ANNOTATION_FONTSIZE = FONT_SIZE_SYMBOL
construction.AUXILIARY_LINEWIDTH = CONSTRUCTION_LINEWIDTH
construction.AUXILIARY_GEOMETRY_COLOR = CONSTRUCTION_COLOR
construction.AUXILIARY_TANGENCY_MARKERSIZE = POINT_MARKERSIZE


def label_point(ax, xy, text, offset=(5, 5), color="0.2", **alignment):
    ax.annotate(
        text, xy, xytext=offset, textcoords="offset points",
        fontsize=FONT_SIZE_SYMBOL, color=color, zorder=110, **alignment,
    )


def label_outside_circle(ax, xy, center, text, color="0.2", gap=5):
    """Place an unshaded label just beyond the point along its outward radius."""
    dx, dy = xy[0] - center[0], xy[1] - center[1]
    radius = hypot(dx, dy)
    ux, uy = dx / radius, dy / radius
    label_point(
        ax, xy, text, (gap * ux, gap * uy), color,
        ha="left" if ux > 1e-10 else "right" if ux < -1e-10 else "center",
        va="bottom" if uy > 1e-10 else "top" if uy < -1e-10 else "center",
    )


def refine_top_right_labels(ax, renderer):
    """Center selected labels on their outward radii with a small clear gap."""
    labels = {text.get_text(): text for text in ax.texts}
    for suffix, names in (
        ("f", (r"$\boldsymbol{p}_f$",)),
    ):
        center = labels[rf"$\boldsymbol{{o}}_{suffix}$"].xy
        for name in names:
            label = labels[name]
            point_px = ax.transData.transform(label.xy)
            center_px = ax.transData.transform(center)
            dx, dy = point_px - center_px
            radius = hypot(dx, dy)
            ux, uy = dx / radius, dy / radius
            bounds = label.get_window_extent(renderer)
            # Project the text box onto the radius to keep it beyond the
            # tangent at the point, with only two points of extra clearance.
            distance = (abs(ux) * bounds.width + abs(uy) * bounds.height) / 2
            distance = distance * 72 / ax.figure.dpi + 2
            label.set_ha("center")
            label.set_va("center")
            label.set_position((distance * ux, distance * uy))


def annotate_geometry(ax, trajectory, start_pose, end_pose, direction_labels=None,
                      initial_label_offset=(-17, 9)):
    """Label positions, supporting centers and arc amplitudes."""
    if direction_labels is None:
        label_point(ax, (start_pose.x, start_pose.y), r"$\boldsymbol{p}_0$",
                    initial_label_offset, overview.START_POSE_COLOR)
        label_point(ax, (end_pose.x, end_pose.y), r"$\boldsymbol{p}_f$",
                    (7, 9), overview.END_POSE_COLOR)
    arcs = construction.get_arc_primitives(trajectory)
    segment = construction.get_segment_primitive(trajectory)
    if len(arcs) < 2 or segment is None:
        return
    centers = [construction.get_circle_center(arcs[0]),
               construction.get_circle_center(arcs[-1])]
    if direction_labels is not None:
        for suffix, pose, center, color in zip(
            ("0", "f"), (start_pose, end_pose), centers,
            (overview.START_POSE_COLOR, overview.END_POSE_COLOR),
        ):
            label_outside_circle(
                ax, (pose.x, pose.y), center, rf"$\boldsymbol{{p}}_{suffix}$", color,
            )
    tangencies = [(float(segment.x0), float(segment.y0)),
                  (float(segment.xf), float(segment.yf))]
    for suffix, center, tangent, arc in zip(
        ("0", "f"), centers, tangencies, (arcs[0], arcs[-1])
    ):
        ax.plot(*center, marker="o", color="black", markersize=POINT_MARKERSIZE)
        # Opposite the swept arc's midpoint is the unused sector's bisector.
        angle = arc.epsilon + arc.turn_direction * arc.iota / 2 + pi
        label_point(
            ax, center, rf"$\boldsymbol{{o}}_{suffix}$",
            (14 * cos(angle), 14 * sin(angle)),
            ha="center", va="center",
        )
        amplitude_angle = arc.epsilon + arc.turn_direction * arc.iota / 2
        label_radius = (AMPLITUDE_ARC_RADIUS_RATIO + 0.2) * float(arc.radius)
        label_point(
            ax,
            (center[0] + label_radius * cos(amplitude_angle),
             center[1] + label_radius * sin(amplitude_angle)),
            rf"$\iota_{suffix}$", (0, 0),
            color=AMPLITUDE_LABEL_COLOR,
            ha="center", va="center",
        )
        ax.plot([center[0], tangent[0]], [center[1], tangent[1]],
                color=CONSTRUCTION_COLOR, linestyle=":", linewidth=0.8, zorder=1)
        ax.plot(*tangent, marker="o", color=PATH_COLOR, markersize=POINT_MARKERSIZE)


def plot_panel(ax, family, candidate, start_pose, end_pose, radius):
    trajectory = candidate["trajectory"]
    arcs = construction.get_arc_primitives(trajectory)
    for arc in arcs:
        ax.add_patch(Circle(
            construction.get_circle_center(arc), radius=float(arc.radius),
            fill=False, edgecolor=CONSTRUCTION_COLOR, linestyle="--",
            linewidth=CONSTRUCTION_LINEWIDTH, zorder=1,
        ))
    if SHOW_AUXILIARY_GEOMETRY:
        if arcs:
            for pose, arc in ((start_pose, arcs[0]), (end_pose, arcs[-1])):
                center = construction.get_circle_center(arc)
                ax.plot([pose.x, center[0]], [pose.y, center[1]],
                        color=CONSTRUCTION_COLOR, linestyle=":",
                        linewidth=0.8, zorder=1)
            for arc_index, arc in enumerate((arcs[0], arcs[-1])):
                start_angle = degrees(arc.epsilon)
                end_angle = degrees(arc.epsilon + arc.turn_direction * arc.iota)
                style = dict(
                    facecolor=AMPLITUDE_SECTOR_COLOR,
                    edgecolor=AMPLITUDE_SECTOR_COLOR,
                    alpha=AMPLITUDE_SECTOR_ALPHA,
                    linewidth=CONSTRUCTION_LINEWIDTH, zorder=2,
                    gid="arc-amplitude",
                )
                center = construction.get_circle_center(arc)
                marker_radius = AMPLITUDE_ARC_RADIUS_RATIO * float(arc.radius)
                square_angle = (
                    (family == "TCSC" and candidate["tau0"] == +1 and arc_index == 0)
                    or (family == "CSCT" and candidate["tau0"] == +1
                        and arc_index == 1)
                    or family == "TCSCT"
                )
                if square_angle and abs(float(arc.iota) - pi / 2) < 1e-8:
                    # Align the square with the two radii enclosing the right angle.
                    side = marker_radius / 2 ** 0.5
                    angles = (arc.epsilon, arc.epsilon + arc.turn_direction * arc.iota)
                    u, v = [(side * cos(angle), side * sin(angle)) for angle in angles]
                    vertices = [center,
                                (center[0] + u[0], center[1] + u[1]),
                                (center[0] + u[0] + v[0], center[1] + u[1] + v[1]),
                                (center[0] + v[0], center[1] + v[1])]
                    ax.add_patch(Polygon(vertices, closed=True, **style))
                else:
                    ax.add_patch(Wedge(
                        center, r=marker_radius,
                        theta1=min(start_angle, end_angle),
                        theta2=max(start_angle, end_angle), **style,
                    ))
        arguments = dict(ax=ax, family_name=family, trajectory=trajectory,
                         start_pose=start_pose, end_pose=end_pose, radius=radius,
                         tau0=candidate["tau0"], tauf=candidate["tauf"])
        construction.plot_reflected_geometry(**arguments)
        construction.plot_tcsct_auxiliary_geometry(**arguments)
        if family == "TCSC" and candidate["tau0"] == +1 and candidate["tauf"] == +1 and arcs:
            final_center = construction.get_circle_center(arcs[-1])
            ax.plot(
                [start_pose.x, final_center[0]], [start_pose.y, final_center[1]],
                linestyle=":", linewidth=construction.AUXILIARY_LINEWIDTH,
                color=construction.AUXILIARY_TANGENT_COLOR, zorder=2,
            )
        if family == "CSCT" and candidate["tau0"] == +1 and candidate["tauf"] == +1 and arcs:
            initial_center = construction.get_circle_center(arcs[0])
            ax.plot(
                [end_pose.x, initial_center[0]], [end_pose.y, initial_center[1]],
                linestyle=":", linewidth=construction.AUXILIARY_LINEWIDTH,
                color=construction.AUXILIARY_TANGENT_COLOR, zorder=2,
            )
        if family == "TCSCT" and candidate["tau0"] == +1 and candidate["tauf"] == +1:
            ax.plot(
                [start_pose.x, end_pose.x], [start_pose.y, end_pose.y],
                linestyle=":", linewidth=construction.AUXILIARY_LINEWIDTH,
                color=construction.AUXILIARY_TANGENT_COLOR, zorder=2,
            )
        for text in list(ax.texts):
            if text.get_text() == r"$2R$":
                text.remove()
    patch_count = len(ax.patches)
    plot_analytical_trajectory(
        trajectory, figure=ax, plot_circles=False, color=PATH_COLOR,
        linewidth=PATH_LINEWIDTH, plot_primitive_arrows=False,
        plot_turn_sectors=True, turn_sector_color=TURN_SECTOR_COLOR,
    )
    turn_sectors = [patch for patch in list(ax.patches)[patch_count:]
                    if isinstance(patch, Wedge)]
    for patch in turn_sectors:
        patch.set_alpha(TURN_SECTOR_ALPHA)
    if SHOW_SYMBOLS:
        turn_suffixes = {"CSC": (), "TCSC": ("0",), "CSCT": ("f",),
                         "TCSCT": ("0", "f")}[family]
        for sector, suffix in zip(turn_sectors, turn_suffixes):
            angle = (sector.theta1 + (sector.theta2 - sector.theta1) % 360 / 2) * pi / 180
            label_radius = 1.5 * sector.r
            if family == "CSCT" and candidate["tau0"] == +1 and candidate["tauf"] == +1:
                label_radius = 1.75 * sector.r
            if (family in ("CSCT", "TCSCT") and suffix == "f"
                    and candidate["tau0"] == +1 and candidate["tauf"] == -1):
                angle += 20 * pi / 180
            label_point(
                ax,
                (sector.center[0] + label_radius * cos(angle),
                 sector.center[1] + label_radius * sin(angle)),
                rf"$\phi_{suffix}$", (0, 0), color=TURN_LABEL_COLOR,
                ha="center", va="center",
            )
    for primitive in trajectory:
        overview.annotate_boundary_pose(
            ax, Pose(Point(primitive.xf, primitive.yf), primitive.thetaf), PATH_COLOR,
            marker_size=overview.PRIMITIVE_POSITION_MARKERSIZE,
            heading_length=overview.PRIMITIVE_HEADING_LENGTH,
            heading_linewidth=overview.PRIMITIVE_HEADING_LINEWIDTH,
            arrowhead_size=overview.PRIMITIVE_HEADING_ARROWHEAD_SIZE,
        )
    overview.annotate_boundary_pose(ax, start_pose, overview.START_POSE_COLOR)
    overview.annotate_boundary_pose(ax, end_pose, overview.END_POSE_COLOR)
    if SHOW_SYMBOLS:
        direction_labels = (
            f"{overview.tau_symbol(candidate['tau0'])},{overview.tau_symbol(candidate['tauf'])}"
            if family == "CSC" else None
        )
        annotate_geometry(
            ax, trajectory, start_pose, end_pose, direction_labels,
            initial_label_offset=(5, 10) if family == "TCSC" else (-17, 9),
        )
        if family == "TCSC" and candidate["tau0"] == +1 and candidate["tauf"] == -1:
            for text in list(ax.texts):
                if text.get_text() == r"$\boldsymbol{p}_0$":
                    text.set_position((-5, 10))
                elif text.get_text() == r"$\boldsymbol{p}_f$":
                    text.remove()
            label_outside_circle(
                ax, (end_pose.x, end_pose.y), construction.get_circle_center(arcs[-1]),
                r"$\boldsymbol{p}_f$", overview.END_POSE_COLOR, gap=2,
            )
            reflected_circle = next(
                (patch for patch in ax.patches
                 if isinstance(patch, Circle) and patch.get_label().startswith("Reflected circle ")),
                None,
            )
            if reflected_circle is not None:
                ax.plot(*reflected_circle.center, marker="o", color=CONSTRUCTION_COLOR,
                        markersize=POINT_MARKERSIZE, zorder=100)
                label_point(ax, reflected_circle.center, r"$\tilde{\boldsymbol{o}}_f$",
                            (-2, 2), color=CONSTRUCTION_COLOR, ha="right", va="bottom")
        if family == "CSCT" and candidate["tau0"] == +1 and candidate["tauf"] == +1:
            for text in list(ax.texts):
                if text.get_text() == r"$\boldsymbol{p}_0$":
                    text.remove()
                elif text.get_text() == r"$\boldsymbol{p}_f$":
                    text.set_position((7, 2))
            label_outside_circle(
                ax, (start_pose.x, start_pose.y), construction.get_circle_center(arcs[0]),
                r"$\boldsymbol{p}_0$", overview.START_POSE_COLOR, gap=5,
            )
        if family == "CSCT" and candidate["tau0"] == +1 and candidate["tauf"] == -1:
            for text in list(ax.texts):
                if text.get_text() == r"$\boldsymbol{p}_0$":
                    text.remove()
                elif text.get_text() == r"$\boldsymbol{p}_f$":
                    text.set_position((5, 7))
            label_outside_circle(
                ax, (start_pose.x, start_pose.y), construction.get_circle_center(arcs[0]),
                r"$\boldsymbol{p}_0$", overview.START_POSE_COLOR, gap=4,
            )
            reflected_circle = next(
                (patch for patch in ax.patches
                 if isinstance(patch, Circle) and patch.get_label().startswith("Reflected circle ")),
                None,
            )
            if reflected_circle is not None:
                ax.plot(*reflected_circle.center, marker="o", color=CONSTRUCTION_COLOR,
                        markersize=POINT_MARKERSIZE, zorder=100)
                label_point(ax, reflected_circle.center, r"$\tilde{\boldsymbol{o}}_0$",
                            (2, -2), color=CONSTRUCTION_COLOR, ha="left", va="top")
        if family == "TCSCT" and candidate["tau0"] == +1 and candidate["tauf"] == +1:
            for text in ax.texts:
                if text.get_text() == r"$\boldsymbol{p}_0$":
                    text.set_position((5, 10))
                    text.set_ha("center")
                    text.set_va("bottom")
                elif text.get_text() == r"$\boldsymbol{p}_f$":
                    text.set_position((7, 0))
                    text.set_va("center")
        if family == "TCSCT" and candidate["tau0"] == +1 and candidate["tauf"] == -1:
            for text in ax.texts:
                if text.get_text() == r"$\boldsymbol{p}_0$":
                    text.set_position((2, 12))
                elif text.get_text() == r"$\boldsymbol{p}_f$":
                    text.set_position((7, 4))
            reflected_circle = next(
                (patch for patch in ax.patches
                 if isinstance(patch, Circle) and patch.get_label().startswith("Reflected circle ")),
                None,
            )
            if reflected_circle is not None:
                ax.plot(*reflected_circle.center, marker="o", color=CONSTRUCTION_COLOR,
                        markersize=POINT_MARKERSIZE, zorder=100)
                label_point(ax, reflected_circle.center, r"$\tilde{\boldsymbol{o}}_f$",
                            (-2, 2), color=CONSTRUCTION_COLOR, ha="right", va="bottom")
        if family != "CSC" and candidate["tau0"] == +1 and candidate["tauf"] == -1:
            auxiliary_circle = next(
                (patch for patch in ax.patches
                 if isinstance(patch, Circle)
                 and patch.get_label().startswith("Auxiliary circle ")),
                None,
            )
            if auxiliary_circle is not None:
                symbol = {"TCSC": r"$\mathcal{A}_{\boldsymbol{o}_f}$",
                          "CSCT": r"$\mathcal{A}_{\boldsymbol{o}_0}$",
                          "TCSCT": r"$\mathcal{A}_{\boldsymbol{p}_f}$"}[family]
                angle = (225 if family == "CSCT" else 55) * pi / 180
                point = (auxiliary_circle.center[0] + auxiliary_circle.radius * cos(angle),
                         auxiliary_circle.center[1] + auxiliary_circle.radius * sin(angle))
                ax.annotate(
                    symbol, point,
                    xytext=(-8, -6) if family == "CSCT" else (8, 6),
                    textcoords="offset points", fontsize=FONT_SIZE_SYMBOL,
                    color=CONSTRUCTION_COLOR,
                    ha="right" if family == "CSCT" else "left",
                    va="top" if family == "CSCT" else "bottom",
                    arrowprops=dict(arrowstyle="-", color=CONSTRUCTION_COLOR,
                                    linewidth=CONSTRUCTION_LINEWIDTH,
                                    relpos=(1, 1) if family == "CSCT" else (0, 0),
                                    patchA=None, shrinkA=0, shrinkB=0),
                    zorder=110,
                )
    ax.set_axis_off()


def annotate_segment_length(ax, trajectory, radius, offset_ratio=0.45):
    """Dimension the straight segment with an offset line and extension lines."""
    segment = construction.get_segment_primitive(trajectory)
    if segment is None:
        return
    start = (float(segment.x0), float(segment.y0))
    end = (float(segment.xf), float(segment.yf))
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0:
        return
    normal = (dy / length, -dx / length)
    offset = offset_ratio * float(radius)
    side = 1 if offset_ratio >= 0 else -1
    dimension_points = []
    for point in (start, end):
        dimension_points.append(tuple(point[i] + offset * normal[i] for i in (0, 1)))
        extension = tuple(point[i] + (offset + side * 0.07 * float(radius)) * normal[i]
                          for i in (0, 1))
        ax.plot([point[0], extension[0]], [point[1], extension[1]],
                color=CONSTRUCTION_COLOR, linestyle="--", linewidth=0.7,
                scalex=False, scaley=False, zorder=1)
    ax.add_patch(FancyArrowPatch(
        dimension_points[0], dimension_points[1], arrowstyle="<->",
        shrinkA=0, shrinkB=0, mutation_scale=10,
        color="black", linewidth=0.9, zorder=2,
    ))
    midpoint = tuple((dimension_points[0][i] + dimension_points[1][i]) / 2
                     for i in (0, 1))
    label_point(ax, midpoint, r"$d$", (0, -5 * side), color="black",
                ha="center", va="top" if side > 0 else "bottom")


def create_figure():
    start_pose = Pose(Point(overview.START_X, overview.START_Y),
                      overview.degrees_to_radians(overview.START_THETA_DEG))
    end_pose = Pose(Point(overview.END_X, overview.END_Y),
                    overview.degrees_to_radians(overview.END_THETA_DEG))
    vehicle = Unicycle(
        state=[0, 0, 0], width=overview.VEHICLE_WIDTH, length=overview.VEHICLE_LENGTH,
        v_max=overview.V_MAX, v_min=0, omega_max=overview.OMEGA_MAX,
        omega_min=-overview.OMEGA_MAX,
    )
    trajectories = compute_all_pose_to_pose_trajectories(start_pose, end_pose, vehicle)
    lookup = overview.get_candidate_lookup(trajectories)
    figure, axes = plt.subplots(4, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    for row, family in enumerate(FAMILIES):
        for column, (tau0, tauf) in enumerate(DIRECTIONS):
            candidate = lookup.get((family, tau0, tauf))
            ax = axes[row, column]
            if candidate is None:
                ax.text(0.5, 0.5, "Candidate unavailable", ha="center",
                        transform=ax.transAxes, fontsize=FONT_SIZE_SYMBOL)
                ax.set_axis_off()
                continue
            plot_panel(ax, family, candidate, start_pose, end_pose, vehicle.max_radius)
            print(f"{family} ({tau0:+d},{tauf:+d}): T={float(candidate['time']):.6f} s")
    # Center each construction independently, with identical spans in every panel.
    # dataLim includes paths, circles and construction lines, but excludes text.
    data_axes = [ax for ax in axes.flat if ax.has_data()]
    x_span = max(ax.dataLim.width for ax in data_axes) * (1 + 2 * GEOMETRY_PADDING)
    y_span = max(ax.dataLim.height for ax in data_axes) * (1 + 2 * GEOMETRY_PADDING)
    for ax in data_axes:
        center_x = (ax.dataLim.x0 + ax.dataLim.x1) / 2
        center_y = (ax.dataLim.y0 + ax.dataLim.y1) / 2
        ax.set_xlim(center_x - x_span / 2, center_x + x_span / 2)
        ax.set_ylim(center_y - y_span / 2, center_y + y_span / 2)
        ax.set_aspect("equal", adjustable="box")
    # Lower panel f's construction to leave room for its upper-left label.
    ax = axes[2, 1]
    if ax.has_data():
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower + 0.06 * y_span, upper + 0.06 * y_span)
        for artist in (*ax.lines, *ax.patches):
            artist.set_clip_on(False)
    candidate = lookup.get(("CSC", +1, +1))
    if candidate is not None:
        annotate_segment_length(axes[0, 0], candidate["trajectory"],
                                vehicle.max_radius)
    candidate = lookup.get(("TCSC", +1, +1))
    if candidate is not None:
        annotate_segment_length(axes[1, 0], candidate["trajectory"], vehicle.max_radius)
    candidate = lookup.get(("TCSC", +1, -1))
    if candidate is not None:
        annotate_segment_length(axes[1, 1], candidate["trajectory"],
                                vehicle.max_radius, offset_ratio=-0.45)
    candidate = lookup.get(("CSC", +1, -1))
    if candidate is not None:
        annotate_segment_length(axes[0, 1], candidate["trajectory"],
                                vehicle.max_radius, offset_ratio=-0.45)
    candidate = lookup.get(("CSCT", +1, +1))
    if candidate is not None:
        annotate_segment_length(axes[2, 0], candidate["trajectory"], vehicle.max_radius)
    candidate = lookup.get(("CSCT", +1, -1))
    if candidate is not None:
        annotate_segment_length(axes[2, 1], candidate["trajectory"], vehicle.max_radius)
    candidate = lookup.get(("TCSCT", +1, +1))
    if candidate is not None:
        annotate_segment_length(axes[3, 0], candidate["trajectory"], vehicle.max_radius)
    candidate = lookup.get(("TCSCT", +1, -1))
    if candidate is not None:
        annotate_segment_length(axes[3, 1], candidate["trajectory"], vehicle.max_radius)
    for row, family in enumerate(FAMILIES):
        for column, (tau0, tauf) in enumerate(DIRECTIONS):
            trajectory_label = rf"C^{{{overview.tau_symbol(tau0)}}}SC^{{{overview.tau_symbol(tauf)}}}"
            if family.startswith("T"):
                trajectory_label = rf"T^{{{overview.tau_symbol(tau0)}}}" + trajectory_label
            if family.endswith("T"):
                trajectory_label += rf"T^{{{overview.tau_symbol(tauf)}}}"
            letter = chr(ord("a") + 2 * row + column)
            axes[row, column].text(
                0.02, 0.98, rf"{letter}) ${trajectory_label}$",
                transform=axes[row, column].transAxes, ha="left", va="top",
                fontsize=FONT_SIZE_PANEL_LABEL,
            )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.055,
                           wspace=PANEL_HORIZONTAL_SPACE, hspace=PANEL_VERTICAL_SPACE)
    # Give the rows enough height to fill their available width at equal aspect.
    panel = axes[0, 0].get_position(original=True)
    required_height = FIGURE_WIDTH * panel.width / panel.height * y_span / x_span
    figure.set_size_inches(FIGURE_WIDTH, max(FIGURE_HEIGHT, required_height))
    figure.canvas.draw()
    if axes[0, 1].has_data():
        refine_top_right_labels(axes[0, 1], figure.canvas.get_renderer())
    positions = [ax.get_position() for ax in axes.flat]
    center = (positions[0].x1 + positions[1].x0) / 2
    figure.add_artist(Line2D([center, center], [positions[-1].y0, positions[0].y1],
                            transform=figure.transFigure, color=overview.GRID_COLOR,
                            linewidth=overview.GRID_LINEWIDTH))
    for row in range(3):
        y = (axes[row, 0].get_position().y0 + axes[row + 1, 0].get_position().y1) / 2 - 0.006
        figure.add_artist(Line2D([0.065, 0.98], [y, y], transform=figure.transFigure,
                                color=overview.GRID_COLOR, linewidth=overview.GRID_LINEWIDTH))
    return figure


if __name__ == "__main__":
    figure = create_figure()
    if SAVE_FIGURE:
        for extension in ("pdf", "png"):
            output = DIRECTORY / f"{OUTPUT_FILENAME}.{extension}"
            figure.savefig(output, dpi=300, bbox_inches="tight")
            print(f"Saved {output}")
    if SHOW_FIGURE:
        plt.show(block=True)
    else:
        plt.close(figure)
