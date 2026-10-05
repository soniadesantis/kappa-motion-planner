import argparse

import matplotlib.pyplot as plt

import numpy as np

from time import perf_counter_ns

from examples_maps_polyline import (

    EXAMPLE_NUMBERS,

    example_corridor_sequence,

)

from kappa_planner.baseline_construction_new import (

    compute_bicycle_baseline,

)

from kappa_planner.refinement_new import (

    compute_safe_corridor_union,

    refine_bicycle_baseline,

)

# ===================================================================

# Select example here when running directly from VS Code

# ===================================================================

EXAMPLE_NUM = 5

PLOT_RESULTS = True

# ===================================================================

# Plotting

# ===================================================================

def plot_baseline_and_refinement(
    corridors,
    baseline,
    circle_groups,
    straight_passage_groups,
    active_circle_indices,
    tangents,
    example_number,
    robot_radius,

    start_pose=None,

    end_pose=None,

):

    """

    Plot:

        - corridor sequence,

        - safe overlaps D_j,

        - erosion of the complete corridor union,

        - baseline waypoints,

        - straight-passage groups containing aligned transitions,

        - baseline straight segments and fillets,

        - refinement circles at genuine turns, distinguishing retained
          and skipped circles,
        - accepted circle-to-circle tangents after chain simplification.

    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # ---------------------------------------------------------------

    # 1. Corridors

    # ---------------------------------------------------------------

    for j, corridor in enumerate(corridors):

        corners = np.asarray(

            corridor.corners,

            dtype=float,

        )

        closed_corners = np.vstack([

            corners,

            corners[0],

        ])

        ax.plot(

            closed_corners[:, 0],

            closed_corners[:, 1],

            "--",

            linewidth=1,

        )

        ax.text(

            corridor.center[0],

            corridor.center[1],

            f"C{j + 1}",

        )

    # ---------------------------------------------------------------

    # 2. Safe overlaps D_j

    # ---------------------------------------------------------------

    for j, safe_overlap in enumerate(

        baseline.safe_overlaps

    ):

        (x_min, x_max), (y_min, y_max) = safe_overlap

        rectangle = np.array([

            [x_min, y_min],

            [x_max, y_min],

            [x_max, y_max],

            [x_min, y_max],

            [x_min, y_min],

        ])

        ax.plot(

            rectangle[:, 0],

            rectangle[:, 1],

            ":",

            linewidth=1.5,

        )

        ax.text(

            0.5 * (x_min + x_max),

            0.5 * (y_min + y_max),

            f"D{j + 1}",

        )

    # ---------------------------------------------------------------
    # 3. Eroded corridor union W_free
    # ---------------------------------------------------------------
    safe_union = compute_safe_corridor_union(
        corridor_list=corridors,
        r=robot_radius,
    )

    safe_union_label_used = False

    if safe_union is not None:

        if safe_union.geom_type == "Polygon":
            safe_polygons = [safe_union]
        else:
            safe_polygons = list(safe_union.geoms)

        for polygon in safe_polygons:

            exterior = np.asarray(
                polygon.exterior.coords,
                dtype=float,
            )

            ax.plot(
                exterior[:, 0],
                exterior[:, 1],
                linewidth=1.8,
                alpha=0.8,
                label=(
                    "Eroded corridor union"
                    if not safe_union_label_used
                    else None
                ),
            )

            safe_union_label_used = True

            for interior in polygon.interiors:

                interior_points = np.asarray(
                    interior.coords,
                    dtype=float,
                )

                ax.plot(
                    interior_points[:, 0],
                    interior_points[:, 1],
                    linewidth=1.2,
                    alpha=0.8,
                )

    # ---------------------------------------------------------------

    # 3. Baseline waypoints / orthogonal polyline

    # ---------------------------------------------------------------

    waypoints = np.asarray(

        baseline.waypoints,

        dtype=float,

    )

    ax.scatter(

        waypoints[:, 0],

        waypoints[:, 1],

        s=45,

        zorder=6,

        label="Baseline waypoints",

    )

    if len(waypoints) > 1:

        ax.plot(

            waypoints[:, 0],

            waypoints[:, 1],

            "--",

            linewidth=1.2,

            label="Orthogonal polyline",

        )

    for j, waypoint in enumerate(

        waypoints,

        start=1,

    ):

        ax.annotate(

            f"p{j}",

            waypoint,

            xytext=(5, 5),

            textcoords="offset points",

        )

    # ---------------------------------------------------------------

    # 4. Straight-passage groups

    # ---------------------------------------------------------------

    straight_group_label_used = False

    for group_index, group in enumerate(

        straight_passage_groups,

        start=1,

    ):

        indices = np.arange(

            group.start_index,

            group.end_index + 1,

        )

        group_points = waypoints[indices]

        if len(group_points) > 1:

            ax.plot(

                group_points[:, 0],

                group_points[:, 1],

                linewidth=4.0,

                alpha=0.35,

                label=(

                    "Straight-passage groups"

                    if not straight_group_label_used

                    else None

                ),

            )

        ax.scatter(

            group_points[:, 0],

            group_points[:, 1],

            marker="s",

            s=35,

            alpha=0.65,

            zorder=7,

        )

        straight_group_label_used = True

        group_midpoint = np.mean(

            group_points,

            axis=0,

        )

        ax.annotate(

            f"G{group_index}",

            group_midpoint,

            xytext=(8, -12),

            textcoords="offset points",

            fontsize=9,

        )

    # ---------------------------------------------------------------

    # 5. Trimmed baseline straight portions

    # ---------------------------------------------------------------

    for j in range(len(waypoints) - 1):

        if baseline.fillets[j] is None:

            start = waypoints[j]

        else:

            start = baseline.fillets[j].end_point

        if baseline.fillets[j + 1] is None:

            end = waypoints[j + 1]

        else:

            end = baseline.fillets[j + 1].start_point

        ax.plot(

            [start[0], end[0]],

            [start[1], end[1]],

            linewidth=2.5,

            label=(

                "Baseline path"

                if j == 0

                else None

            ),

        )

    # ---------------------------------------------------------------

    # 6. Baseline radius-R fillets

    # ---------------------------------------------------------------

    for fillet in baseline.fillets:

        if fillet is None:

            continue

        radial_start = (

            fillet.start_point

            - fillet.center

        )

        start_angle = np.arctan2(

            radial_start[1],

            radial_start[0],

        )

        angles = (

            start_angle

            + fillet.turn_direction

            * np.linspace(

                0.0,

                np.pi / 2.0,

                100,

            )

        )

        radius = np.linalg.norm(

            radial_start

        )

        arc = (

            fillet.center

            + radius

            * np.column_stack([

                np.cos(angles),

                np.sin(angles),

            ])

        )

        ax.plot(

            arc[:, 0],

            arc[:, 1],

            linewidth=2.5,

        )

    # ---------------------------------------------------------------

    # 7. Refinement circles

    # ---------------------------------------------------------------

    angle_array = np.linspace(
        0.0,
        2.0 * np.pi,
        200,
    )

    active_circle_indices = set(
        active_circle_indices
    )

    active_label_used = False
    skipped_label_used = False

    for j, group in enumerate(circle_groups):

        for tau, circle in group.items():

            center = np.asarray(
                circle.center,
                dtype=float,
            )

            radius = circle.radius
            is_active = (
                j in active_circle_indices
            )

            x_circle = (
                center[0]
                + radius * np.cos(angle_array)
            )

            y_circle = (
                center[1]
                + radius * np.sin(angle_array)
            )

            if is_active:
                label = (
                    "Retained refinement circles"
                    if not active_label_used
                    else None
                )
                active_label_used = True
                linestyle = "-."
                linewidth = 1.7
                alpha = 1.0
                center_marker = "x"
                center_size = 65
            else:
                label = (
                    "Skipped refinement circles"
                    if not skipped_label_used
                    else None
                )
                skipped_label_used = True
                linestyle = ":"
                linewidth = 1.2
                alpha = 0.30
                center_marker = "o"
                center_size = 30

            ax.plot(
                x_circle,
                y_circle,
                linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

            ax.scatter(
                center[0],
                center[1],
                marker=center_marker,
                s=center_size,
                alpha=alpha,
                zorder=8,
            )

            if circle.corner_point is not None:

                corner = np.asarray(
                    circle.corner_point,
                    dtype=float,
                )

                ax.scatter(
                    corner[0],
                    corner[1],
                    marker="+",
                    s=50,
                    alpha=alpha,
                    zorder=7,
                )

                ax.plot(
                    [corner[0], center[0]],
                    [corner[1], center[1]],
                    ":",
                    linewidth=0.8,
                    alpha=alpha,
                )

            status = (
                "active"
                if is_active
                else "skipped"
            )

            ax.annotate(
                (
                    f"O{j + 1}, "
                    f"tau={tau}\n"
                    f"{circle.placement_rule}\n"
                    f"{status}"
                ),
                center,
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                alpha=alpha,
            )

    # ---------------------------------------------------------------
    # 8. Accepted refinement tangents
    # ---------------------------------------------------------------
    tangent_label_used = False

    for tangent_index, tangent in enumerate(
        tangents,
        start=1,
    ):

        start_point = np.asarray(
            tangent.start_point,
            dtype=float,
        )

        end_point = np.asarray(
            tangent.end_point,
            dtype=float,
        )

        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linewidth=2.8,
            label=(
                "Accepted refinement tangents"
                if not tangent_label_used
                else None
            ),
        )

        tangent_label_used = True

        ax.scatter(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            s=28,
            zorder=9,
        )

        midpoint = 0.5 * (
            start_point
            + end_point
        )

        ax.annotate(
            f"T{tangent_index}",
            midpoint,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # ---------------------------------------------------------------
    # 9. Initial and final positions
    # ---------------------------------------------------------------

    if start_pose is not None:

        ax.scatter(

            start_pose[0],

            start_pose[1],

            marker="^",

            s=70,

            zorder=10,

            label="Initial position",

        )

        ax.annotate(

            "p0",

            start_pose[:2],

            xytext=(6, 6),

            textcoords="offset points",

        )

    if end_pose is not None:

        ax.scatter(

            end_pose[0],

            end_pose[1],

            marker="*",

            s=90,

            zorder=10,

            label="Final position",

        )

        ax.annotate(

            "pf",

            end_pose[:2],

            xytext=(6, 6),

            textcoords="offset points",

        )

    # ---------------------------------------------------------------

    # Figure formatting

    # ---------------------------------------------------------------

    ax.set_aspect(

        "equal",

        adjustable="box",

    )

    ax.set_xlabel("x [m]")

    ax.set_ylabel("y [m]")

    ax.set_title(

        f"Bicycle baseline, refinement circles, and accepted tangents "

        f"— example {example_number}"

    )

    ax.grid(alpha=0.25)

    ax.legend()

    fig.tight_layout()

    return fig

# ===================================================================

# Printing

# ===================================================================

def print_baseline_information(

    baseline,

    baseline_time_ms,

):

    """

    Print the main geometric quantities produced by the baseline.

    """

    print()

    print("=" * 70)

    print("BASELINE")

    print("=" * 70)

    print(

        f"Computation time: "

        f"{baseline_time_ms:.3f} ms"

    )

    print()

    print("Corridor directions:")

    for j, direction in enumerate(

        baseline.corridor_directions,

        start=1,

    ):

        print(

            f"  C{j}: "

            f"{np.asarray(direction).tolist()}"

        )

    print()

    print("Turn directions:")

    for j, tau in enumerate(

        baseline.turn_directions,

        start=1,

    ):

        print(

            f"  tau_{j} = {tau}"

        )

    print()

    print("Selected waypoints:")

    for j, waypoint in enumerate(

        baseline.waypoints,

        start=1,

    ):

        print(

            f"  p_{j} = "

            f"{np.asarray(waypoint)}"

        )

    print()

    number_of_fillets = sum(

        fillet is not None

        for fillet in baseline.fillets

    )

    print(

        f"Number of baseline fillets: "

        f"{number_of_fillets}"

    )

def print_refinement_information(
    circle_groups,
    straight_passage_groups,
    active_circle_indices,
    tangents,
    refinement_time_ms,

):

    """

    Print the straight-passage groups and the resulting refinement circles.

    """

    print()

    print("=" * 70)

    print("REFINEMENT GROUPING AND CIRCLE POSITIONING")

    print("=" * 70)

    print(

        f"Computation time: "

        f"{refinement_time_ms:.3f} ms"

    )

    total_circles = sum(

        len(group)

        for group in circle_groups

    )

    print(

        f"Number of refinement circles: "

        f"{total_circles}"

    )

    active_circle_indices = list(
        active_circle_indices
    )

    active_set = set(
        active_circle_indices
    )

    all_circle_indices = [
        j
        for j, group in enumerate(circle_groups)
        if group
    ]

    skipped_circle_indices = [
        j
        for j in all_circle_indices
        if j not in active_set
    ]

    print(
        "Active circle transitions: ",
        [j + 1 for j in active_circle_indices],
    )

    print(
        "Skipped circle transitions: ",
        [j + 1 for j in skipped_circle_indices],
    )

    print()

    print("Straight-passage groups:")

    if not straight_passage_groups:

        print("  none")

    else:

        for group_index, group in enumerate(

            straight_passage_groups,

            start=1,

        ):

            left_circle = (

                None

                if group.left_circle_index is None

                else group.left_circle_index + 1

            )

            right_circle = (

                None

                if group.right_circle_index is None

                else group.right_circle_index + 1

            )

            print(

                f"  G{group_index}: "

                f"transitions {group.start_index + 1}"

                f" -> {group.end_index + 1}"

            )

            print(

                f"    zero run = "

                f"{group.zero_start_index + 1}"

                f" -> {group.zero_end_index + 1}"

            )

            print(

                f"    left baseline circle transition = "

                f"{left_circle}"

            )

            print(

                f"    right baseline circle transition = "

                f"{right_circle}"

            )

    print()

    print("Refinement circles:")

    for j, group in enumerate(

        circle_groups,

        start=1,

    ):

        print(

            f"Transition {j}:"

        )

        if not group:

            print(

                "  no refinement circle"

            )

            continue

        for tau, circle in group.items():

            status = (
                "active"
                if (j - 1) in active_set
                else "skipped"
            )

            print(
                f"  status = {status}"
            )

            print(

                f"  tau = {tau}"

            )

            print(

                f"    center = "

                f"{np.asarray(circle.center)}"

            )

            print(

                f"    radius = "

                f"{circle.radius}"

            )

            print(

                f"    rule = "

                f"{circle.placement_rule}"

            )

            print(

                f"    local coordinates = "

                f"{circle.local_coordinates}"

            )

            print(

                f"    effective dimensions = "

                f"{circle.effective_dimensions}"

            )

            print(

                f"    corner = "

                f"{np.asarray(circle.corner_point)}"

            )

    print()
    print("Accepted tangents:")

    if not tangents:
        print("  none")

    else:
        for tangent_index, tangent in enumerate(
            tangents,
            start=1,
        ):

            length = np.linalg.norm(
                tangent.end_point
                - tangent.start_point
            )

            print(
                f"  T{tangent_index}: "
                f"transition {tangent.start_circle_index + 1}"
                f" -> {tangent.end_circle_index + 1}"
            )

            print(
                f"    start point = "
                f"{np.asarray(tangent.start_point)}"
            )

            print(
                f"    end point = "
                f"{np.asarray(tangent.end_point)}"
            )

            print(
                f"    length = "
                f"{length:.6f}"
            )

            print(
                f"    heading = "
                f"{tangent.start_heading:.6f} rad"
            )


# ===================================================================

# Main

# ===================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--example",

        type=int,

        choices=list(EXAMPLE_NUMBERS),

        default=EXAMPLE_NUM,

    )

    parser.add_argument(

        "--no-plot",

        action="store_true",

    )

    args = parser.parse_args()

    # ---------------------------------------------------------------

    # Load example

    # ---------------------------------------------------------------

    (

        corridors,

        start_pose,

        end_pose,

        bicycle,

    ) = example_corridor_sequence(

        args.example

    )

    print("=" * 70)

    print(

        f"TESTING NEW BICYCLE REFINEMENT "

        f"— EXAMPLE {args.example}"

    )

    print("=" * 70)

    print(

        f"Number of corridors: "

        f"{len(corridors)}"

    )

    print(

        f"R = {bicycle.max_radius}"

    )

    print(

        f"r = {bicycle.width / 2}"

    )

    # ---------------------------------------------------------------

    # 1. Compute baseline

    # ---------------------------------------------------------------

    start_time = perf_counter_ns()

    baseline = compute_bicycle_baseline(

        corridor_list=corridors,

        bicycle=bicycle,

        initial_pose=start_pose,

        final_pose=end_pose,

    )

    baseline_time_ms = (

        perf_counter_ns()

        - start_time

    ) / 1e6

    if baseline is None:

        print()

        print("NO BASELINE FOUND")

        print(

            f"Baseline computation time: "

            f"{baseline_time_ms:.3f} ms"

        )

        return

    print_baseline_information(

        baseline=baseline,

        baseline_time_ms=baseline_time_ms,

    )

    # ---------------------------------------------------------------

    # 2. Compute independent refinement circles

    # ---------------------------------------------------------------

    start_time = perf_counter_ns()

    refinement_result = refine_bicycle_baseline(

        corridor_list=corridors,

        bicycle=bicycle,

        baseline=baseline,

    )

    refinement_time_ms = (

        perf_counter_ns()

        - start_time

    ) / 1e6

    if refinement_result is None:

        print()

        print("REFINEMENT FAILED")

        print(

            f"Refinement computation time: "

            f"{refinement_time_ms:.3f} ms"

        )

        return

    (
        circle_groups,
        straight_passage_groups,
        active_circle_indices,
        tangents,
    ) = refinement_result

    print_refinement_information(

        circle_groups=circle_groups,

        straight_passage_groups=straight_passage_groups,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        refinement_time_ms=refinement_time_ms,

    )

    # ---------------------------------------------------------------

    # 3. Total time so far

    # ---------------------------------------------------------------

    total_time_ms = (

        baseline_time_ms

        + refinement_time_ms

    )

    print()

    print("=" * 70)

    print("TIMING")

    print("=" * 70)

    print(

        f"Baseline:   "

        f"{baseline_time_ms:.3f} ms"

    )

    print(

        f"Refinement: "

        f"{refinement_time_ms:.3f} ms"

    )

    print(

        f"Total:      "

        f"{total_time_ms:.3f} ms"

    )

    # ---------------------------------------------------------------

    # 4. Plot

    # ---------------------------------------------------------------

    if (

        PLOT_RESULTS

        and not args.no_plot

    ):

        plot_baseline_and_refinement(

            corridors=corridors,

            baseline=baseline,

            circle_groups=circle_groups,

            straight_passage_groups=straight_passage_groups,
            active_circle_indices=active_circle_indices,
            tangents=tangents,
            example_number=args.example,
            robot_radius=bicycle.width / 2,

            start_pose=start_pose,

            end_pose=end_pose,

        )

        plt.show()

if __name__ == "__main__":

    main()