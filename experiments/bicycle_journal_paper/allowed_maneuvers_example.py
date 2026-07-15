
import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, pi, atan2

from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Bicycle
from kappa_planner.helpers.collision_avoidance import allowed_maneuvers
from kappa_planner.geometry import Point, Circle
from kappa_planner.helpers.plot_helpers import plot_corridors


def example_corridor_sequence(num):

    if num == 1:
        corridor = CorridorWorld(
            1.5000000223517425,
            12.000000178813934,
            [11.200000166893005, 6.550000097602606],
            1.5707963267948966,
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

        return corridor, start_pose, vehicle

    raise ValueError(f"Example {num} is not defined")


def plot_allowed_arc(
    ax,
    circle,
    start_pose,
    amplitude,
    angular_direction,
    color,
    label,
):
    """
    Plot an arc starting from the vehicle position.

    angular_direction:
        +1 -> counterclockwise
        -1 -> clockwise
    """
    x0, y0, _ = start_pose

    start_angle = atan2(
        y0 - circle.yc,
        x0 - circle.xc,
    )

    angles = np.linspace(
        start_angle,
        start_angle + angular_direction * amplitude,
        100,
    )

    x = circle.xc + circle.radius * np.cos(angles)
    y = circle.yc + circle.radius * np.sin(angles)

    ax.plot(
        x,
        y,
        linewidth=2.5,
        color=color,
        label=label,
    )


if __name__ == "__main__":

    example_num = 1

    corridor, start_pose, vehicle = example_corridor_sequence(example_num)
    corridor_list = [corridor, corridor.shrink(vehicle.width/2)]

    result = allowed_maneuvers(
        start_pose=start_pose,
        corridor=corridor,
        vehicle=vehicle,
    )

    x0, y0, theta0 = start_pose
    R = vehicle.max_radius

    # Reconstruct the two turning circles.
    center_left = Point(
        x0 + R * cos(theta0 + pi / 2.0),
        y0 + R * sin(theta0 + pi / 2.0),
    )

    center_right = Point(
        x0 + R * cos(theta0 - pi / 2.0),
        y0 + R * sin(theta0 - pi / 2.0),
    )

    circle_left = Circle(center_left, R)
    circle_right = Circle(center_right, R)

    # Plot the corridor.
    plot_corridors(
        corridor_list,
    )

    ax = plt.gca()

    # Plot complete turning circles as dashed grey lines.
    circle_angles = np.linspace(0.0, 2.0 * pi, 300)

    ax.plot(
        circle_left.xc + R * np.cos(circle_angles),
        circle_left.yc + R * np.sin(circle_angles),
        linestyle="--",
        linewidth=1.0,
        color="grey",
        label="left turning circle",
    )

    ax.plot(
        circle_right.xc + R * np.cos(circle_angles),
        circle_right.yc + R * np.sin(circle_angles),
        linestyle="--",
        linewidth=1.0,
        color="darkgrey",
        label="right turning circle",
    )

    # Plot the initial vehicle position.
    ax.plot(
        x0,
        y0,
        marker="o",
        markersize=6,
        color="black",
        label="start position",
    )

    amplitudes = result["amplitudes"]

    # Left-circle maneuvers.
    plot_allowed_arc(
        ax=ax,
        circle=circle_left,
        start_pose=start_pose,
        amplitude=amplitudes["forward_left"],
        angular_direction=+1,
        color="tab:blue",
        label="forward left",
    )

    plot_allowed_arc(
        ax=ax,
        circle=circle_left,
        start_pose=start_pose,
        amplitude=amplitudes["backward_left"],
        angular_direction=-1,
        color="tab:orange",
        label="backward left",
    )

    # Right-circle maneuvers.
    plot_allowed_arc(
        ax=ax,
        circle=circle_right,
        start_pose=start_pose,
        amplitude=amplitudes["forward_right"],
        angular_direction=-1,
        color="tab:green",
        label="forward right",
    )

    plot_allowed_arc(
        ax=ax,
        circle=circle_right,
        start_pose=start_pose,
        amplitude=amplitudes["backward_right"],
        angular_direction=+1,
        color="tab:red",
        label="backward right",
    )

    # Optionally plot all detected circle-wall intersections.
    for point in result["left_circle"]["points"]:
        ax.plot(point[0], point[1], marker="x", color="tab:blue")

    for point in result["right_circle"]["points"]:
        ax.plot(point[0], point[1], marker="x", color="tab:green")

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Allowed minimum-radius arc maneuvers")

    plt.show(block=True)

