import numpy as np
import matplotlib.pyplot as plt

from math import atan2, cos, pi, sin

from kappa_planner.corridor import CorridorWorld
from kappa_planner.geometry import Circle, Point
from kappa_planner.helpers.collision_avoidance import allowed_maneuvers
from kappa_planner.helpers.plot_helpers import plot_corridors
from kappa_planner.helpers.recovery_maneuvers_bicycle import (
    compute_corridor_alignment_maneuvers,
    compute_segment_to_next_corridor,
    compute_segment_deeper_into_corridor,
)
from kappa_planner.vehicle import Bicycle


def example_corridor_sequence(num):
    """
    Return two connected corridors, an initial pose, and a bicycle model.
    """
    if num == 1:
        first_corridor = CorridorWorld(
            width=1.5,
            height=12.0,
            center=[11.2, 6.55],
            tilt=pi / 2.0,
        )

        # Horizontal corridor overlapping the front part of corridor 1.
        second_corridor = CorridorWorld(
            width=1.5,
            height=8.0,
            center=[14.45, 12.0],
            tilt=0.0,
        )

        start_pose = [
            11.24559211730957,
            2.833745002746582,
            -0.6947391079002443,
        ]

        vehicle = Bicycle(
            [0, 0, 0],
            width=0.1,
            length=0.1,
            wheelbase=0.25,
            v_max=1.0,
            v_min=-1.0,
            delta_max=0.5,
            delta_min=-0.5,
        )

        return (
            first_corridor,
            second_corridor,
            start_pose,
            vehicle,
        )

    raise ValueError(
        f"Example {num} is not defined"
    )


def plot_allowed_arc(
    ax,
    circle,
    start_pose,
    amplitude,
    angular_direction,
    color,
    label,
    samples_number=100,
):
    """
    Plot the allowed portion of a turning circle.

    Parameters
    ----------
    angular_direction:
        +1 for counterclockwise motion.
        -1 for clockwise motion.
    """
    x0, y0, _ = start_pose

    start_angle = atan2(
        y0 - circle.yc,
        x0 - circle.xc,
    )

    angles = np.linspace(
        start_angle,
        start_angle
        + angular_direction * amplitude,
        samples_number,
    )

    x_coordinates = (
        circle.xc
        + circle.radius * np.cos(angles)
    )

    y_coordinates = (
        circle.yc
        + circle.radius * np.sin(angles)
    )

    ax.plot(
        x_coordinates,
        y_coordinates,
        linewidth=2.0,
        color=color,
        label=label,
        alpha=0.5,
    )


def wrapped_angle_difference(
    angle_1,
    angle_2,
):
    """
    Return angle_1 - angle_2 wrapped to [-pi, pi).
    """
    return atan2(
        sin(angle_1 - angle_2),
        cos(angle_1 - angle_2),
    )


def plot_alignment_solution(
    ax,
    maneuvers,
    preferred_direction,
    corridor,
    color,
    label,
):
    """
    Plot a zero-, one-, or two-arc corridor-alignment solution.
    """
    if maneuvers is None:
        print(
            f"No alignment solution found for "
            f"preferred_direction={preferred_direction}."
        )
        return

    if len(maneuvers) == 0:
        print(
            f"The vehicle is already aligned for "
            f"preferred_direction={preferred_direction}."
        )
        return

    print(
        f"\nAlignment solution for "
        f"preferred_direction={preferred_direction}:"
    )

    for index, maneuver in enumerate(maneuvers):
        maneuver_label = (
            label
            if index == 0
            else None
        )

        ax.plot(
            maneuver.path_coordinates[:, 0],
            maneuver.path_coordinates[:, 1],
            color=color,
            linestyle="solid",
            linewidth=4.0,
            label=maneuver_label,
            zorder=10,
        )

        print(
            f"  Arc {index + 1}:"
        )

        print(
            f"    class      : "
            f"{type(maneuver).__name__}"
        )

        print(
            f"    amplitude  : "
            f"{maneuver.iota:.6f} rad"
        )

        print(
            f"    start pose : "
            f"{maneuver.start_pose}"
        )

        print(
            f"    end pose   : "
            f"{maneuver.end_pose}"
        )

        if index < len(maneuvers) - 1:
            ax.plot(
                maneuver.xf,
                maneuver.yf,
                marker="x",
                markersize=9,
                markeredgewidth=2.5,
                color=color,
                zorder=11,
            )

    final_maneuver = maneuvers[-1]

    ax.plot(
        final_maneuver.xf,
        final_maneuver.yf,
        marker="o",
        markersize=8,
        color=color,
        zorder=11,
    )

    ax.arrow(
        final_maneuver.xf,
        final_maneuver.yf,
        0.4 * cos(final_maneuver.thetaf),
        0.4 * sin(final_maneuver.thetaf),
        width=0.012,
        head_width=0.09,
        head_length=0.11,
        color=color,
        length_includes_head=True,
        zorder=11,
    )

    target_heading = (
        corridor.tilt
        if preferred_direction == 1
        else corridor.tilt + pi
    )

    alignment_error = (
        wrapped_angle_difference(
            final_maneuver.thetaf,
            target_heading,
        )
    )

    print(
        f"  Alignment error: "
        f"{alignment_error:.3e} rad"
    )


def get_pose_after_maneuvers(
    initial_pose,
    maneuvers,
):
    """
    Return the pose reached after a maneuver list.
    """
    if maneuvers is None:
        return None

    if len(maneuvers) == 0:
        return initial_pose

    return maneuvers[-1].end_pose


def get_time_after_maneuvers(
    initial_time,
    maneuvers,
):
    """
    Return the final time of a maneuver list.
    """
    if (
        maneuvers is None
        or len(maneuvers) == 0
    ):
        return initial_time

    return maneuvers[-1].tf


def plot_straight_segment(
    ax,
    segment,
    color,
    label,
    preferred_direction,
    segment_description,
):
    """
    Plot a recovery straight segment.
    """
    if segment is None:
        print(
            f"No {segment_description} segment found for "
            f"preferred_direction={preferred_direction}."
        )
        return

    ax.plot(
        segment.path_coordinates[:, 0],
        segment.path_coordinates[:, 1],
        color=color,
        linestyle="--",
        linewidth=4.0,
        label=label,
        zorder=9,
    )

    ax.plot(
        segment.xf,
        segment.yf,
        marker="s",
        markersize=8,
        color=color,
        zorder=11,
    )

    ax.arrow(
        segment.xf,
        segment.yf,
        0.4 * cos(segment.thetaf),
        0.4 * sin(segment.thetaf),
        width=0.012,
        head_width=0.09,
        head_length=0.11,
        color=color,
        length_includes_head=True,
        zorder=11,
    )

    motion_direction = (
        "forward"
        if segment.v > 0.0
        else "backward"
    )

    print(
        f"\n{segment_description.capitalize()} segment for "
        f"preferred_direction={preferred_direction}:"
    )

    print(
        f"  Motion direction : "
        f"{motion_direction}"
    )

    print(
        f"  Start pose       : "
        f"{segment.start_pose}"
    )

    print(
        f"  End pose         : "
        f"{segment.end_pose}"
    )

    print(
        f"  Length           : "
        f"{segment.path_length:.6f}"
    )


def prepare_base_figure(
    corridor_list,
    first_corridor,
    start_pose,
    vehicle,
    allowed_result,
    alignment_positive,
    alignment_negative,
):
    """
    Plot the geometry common to both recovery figures.
    """
    figure = plot_corridors(
        corridor_list,
    )

    ax = figure.gca()

    x0, y0, theta0 = start_pose
    radius = vehicle.max_radius

    # ---------------------------------------------------------------
    # Reconstruct the initial turning circles
    # ---------------------------------------------------------------
    left_center = Point(
        x0
        + radius * cos(theta0 + pi / 2.0),
        y0
        + radius * sin(theta0 + pi / 2.0),
    )

    right_center = Point(
        x0
        + radius * cos(theta0 - pi / 2.0),
        y0
        + radius * sin(theta0 - pi / 2.0),
    )

    left_circle = Circle(
        left_center,
        radius,
    )

    right_circle = Circle(
        right_center,
        radius,
    )

    # ---------------------------------------------------------------
    # Plot complete initial turning circles
    # ---------------------------------------------------------------
    circle_angles = np.linspace(
        0.0,
        2.0 * pi,
        300,
    )

    ax.plot(
        left_circle.xc
        + radius * np.cos(circle_angles),
        left_circle.yc
        + radius * np.sin(circle_angles),
        linestyle="--",
        linewidth=1.0,
        color="grey",
        label="left turning circle",
    )

    ax.plot(
        right_circle.xc
        + radius * np.cos(circle_angles),
        right_circle.yc
        + radius * np.sin(circle_angles),
        linestyle="--",
        linewidth=1.0,
        color="darkgrey",
        label="right turning circle",
    )

    # ---------------------------------------------------------------
    # Plot the initial pose
    # ---------------------------------------------------------------
    ax.plot(
        x0,
        y0,
        marker="o",
        markersize=6,
        color="black",
        label="start position",
        zorder=12,
    )

    ax.arrow(
        x0,
        y0,
        0.4 * cos(theta0),
        0.4 * sin(theta0),
        width=0.01,
        head_width=0.08,
        head_length=0.10,
        color="black",
        length_includes_head=True,
        zorder=12,
    )

    # ---------------------------------------------------------------
    # Plot maximum allowed initial maneuvers
    # ---------------------------------------------------------------
    amplitudes = allowed_result["amplitudes"]

    plot_allowed_arc(
        ax=ax,
        circle=left_circle,
        start_pose=start_pose,
        amplitude=amplitudes["forward_left"],
        angular_direction=1,
        color="tab:blue",
        label="forward left",
    )

    plot_allowed_arc(
        ax=ax,
        circle=left_circle,
        start_pose=start_pose,
        amplitude=amplitudes["backward_left"],
        angular_direction=-1,
        color="tab:orange",
        label="backward left",
    )

    plot_allowed_arc(
        ax=ax,
        circle=right_circle,
        start_pose=start_pose,
        amplitude=amplitudes["forward_right"],
        angular_direction=-1,
        color="tab:green",
        label="forward right",
    )

    plot_allowed_arc(
        ax=ax,
        circle=right_circle,
        start_pose=start_pose,
        amplitude=amplitudes["backward_right"],
        angular_direction=1,
        color="tab:red",
        label="backward right",
    )

    # ---------------------------------------------------------------
    # Plot circle-wall intersections
    # ---------------------------------------------------------------
    for point in allowed_result[
        "left_circle"
    ]["points"]:
        ax.plot(
            point[0],
            point[1],
            marker="x",
            markersize=7,
            color="tab:blue",
        )

    for point in allowed_result[
        "right_circle"
    ]["points"]:
        ax.plot(
            point[0],
            point[1],
            marker="x",
            markersize=7,
            color="tab:green",
        )

    # ---------------------------------------------------------------
    # Plot both alignment solutions
    # ---------------------------------------------------------------
    plot_alignment_solution(
        ax=ax,
        maneuvers=alignment_positive,
        preferred_direction=1,
        corridor=first_corridor,
        color="magenta",
        label="alignment: preferred direction +1",
    )

    plot_alignment_solution(
        ax=ax,
        maneuvers=alignment_negative,
        preferred_direction=-1,
        corridor=first_corridor,
        color="cyan",
        label="alignment: preferred direction -1",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    return figure, ax


if __name__ == "__main__":
    example_num = 1

    (
        first_corridor,
        second_corridor,
        start_pose,
        vehicle,
    ) = example_corridor_sequence(
        example_num
    )

    margin = vehicle.width / 2.0

    first_effective_corridor = (
        first_corridor.shrink(
            margin
        )
    )

    second_effective_corridor = (
        second_corridor.shrink(
            margin
        )
    )

    corridor_list = [
        first_corridor,
        second_corridor,
        first_effective_corridor,
        second_effective_corridor,
    ]

    # ---------------------------------------------------------------
    # Compute maximum allowed initial arc amplitudes
    # ---------------------------------------------------------------
    allowed_result = allowed_maneuvers(
        start_pose=start_pose,
        corridor=first_corridor,
        vehicle=vehicle,
    )

    # ---------------------------------------------------------------
    # Compute alignment in both corridor directions
    # ---------------------------------------------------------------
    alignment_positive = (
        compute_corridor_alignment_maneuvers(
            start_pose=start_pose,
            corridor=first_corridor,
            bicycle=vehicle,
            preferred_direction=1,
        )
    )

    alignment_negative = (
        compute_corridor_alignment_maneuvers(
            start_pose=start_pose,
            corridor=first_corridor,
            bicycle=vehicle,
            preferred_direction=-1,
        )
    )

    # ---------------------------------------------------------------
    # Determine the aligned poses and final alignment times
    # ---------------------------------------------------------------
    aligned_pose_positive = (
        get_pose_after_maneuvers(
            initial_pose=start_pose,
            maneuvers=alignment_positive,
        )
    )

    aligned_pose_negative = (
        get_pose_after_maneuvers(
            initial_pose=start_pose,
            maneuvers=alignment_negative,
        )
    )

    aligned_time_positive = (
        get_time_after_maneuvers(
            initial_time=0.0,
            maneuvers=alignment_positive,
        )
    )

    aligned_time_negative = (
        get_time_after_maneuvers(
            initial_time=0.0,
            maneuvers=alignment_negative,
        )
    )

    # ===============================================================
    # Recovery option 1:
    # move toward and into the second corridor
    # ===============================================================
    distance_inside_second = 0.25

    if aligned_pose_positive is not None:
        segment_to_second_positive = (
            compute_segment_to_next_corridor(
                start_pose=aligned_pose_positive,
                first_corridor=first_corridor,
                second_corridor=second_corridor,
                bicycle=vehicle,
                distance_inside_second=(
                    distance_inside_second
                ),
                t0=aligned_time_positive,
            )
        )
    else:
        segment_to_second_positive = None

    if aligned_pose_negative is not None:
        segment_to_second_negative = (
            compute_segment_to_next_corridor(
                start_pose=aligned_pose_negative,
                first_corridor=first_corridor,
                second_corridor=second_corridor,
                bicycle=vehicle,
                distance_inside_second=(
                    distance_inside_second
                ),
                t0=aligned_time_negative,
            )
        )
    else:
        segment_to_second_negative = None

    # ===============================================================
    # Recovery option 2:
    # move deeper inside the first corridor
    # ===============================================================
    distance_deeper_inside = 1.0

    if aligned_pose_positive is not None:
        segment_deeper_positive = (
            compute_segment_deeper_into_corridor(
                start_pose=aligned_pose_positive,
                corridor=first_corridor,
                bicycle=vehicle,
                distance=distance_deeper_inside,
                t0=aligned_time_positive,
            )
        )
    else:
        segment_deeper_positive = None

    if aligned_pose_negative is not None:
        segment_deeper_negative = (
            compute_segment_deeper_into_corridor(
                start_pose=aligned_pose_negative,
                corridor=first_corridor,
                bicycle=vehicle,
                distance=distance_deeper_inside,
                t0=aligned_time_negative,
            )
        )
    else:
        segment_deeper_negative = None

    # ===============================================================
    # Figure 1:
    # alignment followed by transition to corridor 2
    # ===============================================================
    figure_transition, ax_transition = (
        prepare_base_figure(
            corridor_list=corridor_list,
            first_corridor=first_corridor,
            start_pose=start_pose,
            vehicle=vehicle,
            allowed_result=allowed_result,
            alignment_positive=alignment_positive,
            alignment_negative=alignment_negative,
        )
    )

    plot_straight_segment(
        ax=ax_transition,
        segment=segment_to_second_positive,
        color="darkmagenta",
        label="transition after alignment +1",
        preferred_direction=1,
        segment_description="transition",
    )

    plot_straight_segment(
        ax=ax_transition,
        segment=segment_to_second_negative,
        color="darkcyan",
        label="transition after alignment -1",
        preferred_direction=-1,
        segment_description="transition",
    )

    ax_transition.set_title(
        "Alignment and transition to the second corridor"
    )

    ax_transition.legend()

    # ===============================================================
    # Figure 2:
    # alignment followed by movement deeper into corridor 1
    # ===============================================================
    figure_deeper, ax_deeper = (
        prepare_base_figure(
            corridor_list=corridor_list,
            first_corridor=first_corridor,
            start_pose=start_pose,
            vehicle=vehicle,
            allowed_result=allowed_result,
            alignment_positive=alignment_positive,
            alignment_negative=alignment_negative,
        )
    )

    plot_straight_segment(
        ax=ax_deeper,
        segment=segment_deeper_positive,
        color="darkmagenta",
        label="deeper recovery after alignment +1",
        preferred_direction=1,
        segment_description="deeper-corridor recovery",
    )

    plot_straight_segment(
        ax=ax_deeper,
        segment=segment_deeper_negative,
        color="darkcyan",
        label="deeper recovery after alignment -1",
        preferred_direction=-1,
        segment_description="deeper-corridor recovery",
    )

    ax_deeper.set_title(
        "Alignment and recovery deeper inside the first corridor"
    )

    ax_deeper.legend()

    plt.show(
        block=True
    )