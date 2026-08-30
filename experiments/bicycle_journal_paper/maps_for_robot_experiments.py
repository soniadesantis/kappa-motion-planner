from math import sin, cos, pi

import matplotlib.pylab as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
    plot_corridors,
    plot_velocity_profiles,
)

# Bounds of x-y plane on the floor
# x_min = -0.9 m
# x_max = 2.65 m
# y_min = -1.25 m
# y_max = 4.9 m

# Unicycle model parameters
# v_max = 0.5 m/s
# v_min = 0.0 m/s
# omega_max = 2.0 rad/s
# omega_min = -2.0 rad/s
# vehicle_width = 0.34 m
# vehicle_length = 0.34 m

X_MIN = -0.9
X_MAX = 2.65
Y_MIN = -1.25
Y_MAX = 4.9

FLOOR_CLEARANCE = 0.20

ROBOT_V_MAX = 0.5
ROBOT_V_MIN = 0.0
ROBOT_OMEGA_MAX = 2.0
ROBOT_OMEGA_MIN = -2.0
ROBOT_WIDTH = 0.34
ROBOT_LENGTH = 0.34

# ============================================================
# Vehicle
# ============================================================

unicycle = Unicycle(model='Rosbot circular')

unicycle.update(
    width=ROBOT_WIDTH,
    length=ROBOT_LENGTH,
    v_max=ROBOT_V_MAX,
    v_min=ROBOT_V_MIN,
    omega_max=ROBOT_OMEGA_MAX,
    omega_min=ROBOT_OMEGA_MIN,
)

R = unicycle.v_max / unicycle.omega_max
ROBOT_RADIUS = 0.5 * max(unicycle.width, unicycle.length)

print(f"Turning radius R = {R:.2f} m")


# ============================================================
# Small plotting helper for start/end footprint
# ============================================================

def plot_robot_pose(ax, pose, radius, label=None):
    x, y, theta = pose

    footprint = Circle(
        (x, y),
        radius,
        fill=False,
        linewidth=2
    )
    ax.add_patch(footprint)

    ax.arrow(
        x,
        y,
        radius * cos(theta),
        radius * sin(theta),
        head_width=0.05,
        length_includes_head=True
    )

    if label is not None:
        ax.text(
            x,
            y - 1.4 * radius,
            label,
            ha='center'
        )


# ============================================================
# Projection-area helper
# ============================================================

def plot_projection_area(ax):
    projection = Rectangle(
        (X_MIN, Y_MIN),
        X_MAX - X_MIN,
        Y_MAX - Y_MIN,
        fill=False,
        edgecolor='black',
        linestyle='--',
        linewidth=2,
        label='Projection area'
    )
    ax.add_patch(projection)


# ============================================================
# Bounds helper
# ============================================================

def assert_experiment_fits_floor(corridor_list, initial_pose, final_pose, robot_radius):
    for idx, corridor in enumerate(corridor_list, start=1):
        x_values = corridor.corners[:, 0]
        y_values = corridor.corners[:, 1]

        if (
            x_values.min() < X_MIN or
            x_values.max() > X_MAX or
            y_values.min() < Y_MIN or
            y_values.max() > Y_MAX
        ):
            raise ValueError(f"Corridor {idx} exceeds the projected floor bounds.")

    for pose_name, pose in (('start', initial_pose), ('goal', final_pose)):
        x, y, _ = pose

        if (
            x - robot_radius < X_MIN or
            x + robot_radius > X_MAX or
            y - robot_radius < Y_MIN or
            y + robot_radius > Y_MAX
        ):
            raise ValueError(f"The {pose_name} pose footprint exceeds the projected floor bounds.")


# ============================================================
# Experiment 1: Right-angle turn
# ============================================================

def experiment_1():

    width = 0.90
    right_clearance = FLOOR_CLEARANCE

    horizontal_y = Y_MIN + FLOOR_CLEARANCE + 0.5 * width
    horizontal_start_x = X_MIN + FLOOR_CLEARANCE
    vertical_center_x = X_MAX - right_clearance - 0.5 * width
    horizontal_end_x = vertical_center_x + 0.5 * width

    horizontal_length = horizontal_end_x - horizontal_start_x
    corridor1 = CorridorWorld(
        width=width,
        height=horizontal_length,
        center=[
            horizontal_start_x + 0.5 * horizontal_length,
            horizontal_y,
        ],
        tilt=0.0
    )

    vertical_tail_y = horizontal_y - 0.5 * width
    vertical_top_y = Y_MAX - FLOOR_CLEARANCE - 0.5 * width - FLOOR_CLEARANCE
    vertical_length = vertical_top_y - vertical_tail_y
    corridor2 = CorridorWorld(
        width=width,
        height=vertical_length,
        center=[
            vertical_center_x,
            vertical_tail_y + 0.5 * vertical_length,
        ],
        tilt=pi / 2
    )

    corridor_list = [
        corridor1,
        corridor2
    ]

    # initial_pose = compute_start_pose(
    #     corridor1,
    #     unicycle,
    #     0.5
    # )

    # final_pose = compute_end_pose(
    #     corridor2,
    #     unicycle,
    #     0.35
    # )

    initial_pose = [
        -0.15,
        -0.35,
        -3 * pi / 4
    ]

    final_pose = [
        2.1,
        3.1, 
        -pi/6
    ]

    assert_experiment_fits_floor(
        corridor_list,
        initial_pose,
        final_pose,
        ROBOT_RADIUS,
    )

    return corridor_list, initial_pose, final_pose


# ============================================================
# Experiment 2: Same corridors as experiment 1
# ============================================================

def experiment_2():

    width = 0.90
    right_clearance = FLOOR_CLEARANCE

    horizontal_y = Y_MIN + FLOOR_CLEARANCE + 0.5 * width
    horizontal_start_x = X_MIN + FLOOR_CLEARANCE
    vertical_center_x = X_MAX - right_clearance - 0.5 * width
    horizontal_end_x = vertical_center_x + 0.5 * width

    horizontal_length = horizontal_end_x - horizontal_start_x
    corridor1 = CorridorWorld(
        width=width,
        height=horizontal_length,
        center=[
            horizontal_start_x + 0.5 * horizontal_length,
            horizontal_y,
        ],
        tilt=0.0
    )

    vertical_tail_y = horizontal_y - 0.5 * width
    vertical_top_y = Y_MAX - FLOOR_CLEARANCE - 0.5 * width - FLOOR_CLEARANCE
    vertical_length = vertical_top_y - vertical_tail_y
    corridor2 = CorridorWorld(
        width=width,
        height=vertical_length,
        center=[
            vertical_center_x,
            vertical_tail_y + 0.5 * vertical_length,
        ],
        tilt=pi / 2
    )

    corridor_list = [
        corridor1,
        corridor2
    ]

    initial_pose = [
        1.33,
        -0.34,
        3 * pi / 4
    ]

    final_pose = [
        1.76,
        0.22,
        -pi / 6
    ]

    assert_experiment_fits_floor(
        corridor_list,
        initial_pose,
        final_pose,
        ROBOT_RADIUS,
    )

    return corridor_list, initial_pose, final_pose


# ============================================================
# Experiment 3: Multiple corridors with different angles
# ============================================================

def experiment_3():

    width = 0.80
    add_height = 0.45

    # Corridor orientations
    phi1 = 0.0
    phi2 = pi / 3       # 60 deg
    phi3 = pi           # 180 deg
    phi4 = pi / 2       # 90 deg
    phi5 = phi4 - pi / 2

    # Corridor lengths
    length1 = 1.30
    length2 = 2.00
    length3 = 1.20
    length4 = 2.70
    length5 = 1.20

    # --------------------------------------------------------
    # Corridor 1
    # Keep it safely within the lower-left part of projection
    # --------------------------------------------------------

    corridor1 = CorridorWorld(
        width=width,
        height=length1,
        center=[
            -0.05,
            -0.70
        ],
        tilt=phi1
    )

    # --------------------------------------------------------
    # Corridor 2
    # --------------------------------------------------------

    tail = corridor1.head

    head = [
        tail[0] + length2 * cos(phi2),
        tail[1] + length2 * sin(phi2)
    ]

    corridor2 = get_corridor_from_vector(
        tail,
        head,
        width,
        add_height=add_height
    )

    # --------------------------------------------------------
    # Corridor 3
    #
    # Difference between phi2 and phi3 = 120 deg:
    # deliberately obtuse junction
    # --------------------------------------------------------

    tail = corridor2.head

    head = [
        tail[0] + length3 * cos(phi3),
        tail[1] + length3 * sin(phi3)
    ]

    corridor3 = get_corridor_from_vector(
        tail,
        head,
        width,
        add_height=add_height
    )

    # --------------------------------------------------------
    # Corridor 4
    # Use vertical direction to exploit long projection axis
    # --------------------------------------------------------

    tail = corridor3.head

    head = [
        tail[0] + length4 * cos(phi4),
        tail[1] + length4 * sin(phi4)
    ]

    corridor4 = get_corridor_from_vector(
        tail,
        head,
        width,
        add_height=add_height
    )

    # --------------------------------------------------------
    # Corridor 5
    # Final right turn relative to corridor 4
    # --------------------------------------------------------

    tail = corridor4.head

    head = [
        tail[0] + length5 * cos(phi5),
        tail[1] + length5 * sin(phi5)
    ]

    corridor5 = get_corridor_from_vector(
        tail,
        head,
        width,
        add_height=add_height
    )

    corridor_list = [
        corridor1,
        corridor2,
        corridor3,
        corridor4,
        corridor5
    ]

    initial_pose = compute_start_pose(
        corridor1,
        unicycle,
        0.35
    )
    initial_pose[2] = -pi/2

    final_pose = compute_end_pose(
        corridor5,
        unicycle,
        0.35
    )
    final_pose[2] = -pi/2

    assert_experiment_fits_floor(
        corridor_list,
        initial_pose,
        final_pose,
        ROBOT_RADIUS,
    )

    return corridor_list, initial_pose, final_pose


# ============================================================
# Experiment 4: Narrow corridors
# ============================================================

def experiment_4():

    # ========================================================
    # Geometry
    # ========================================================

    phi1 = 0.0
    phi2 = pi / 2       # 90 deg relative to C1
    phi3 = pi / 6       # 60 deg relative to C2

    length1 = 1.40
    length2 = 2.30
    length3 = 1.70

    add_height = 0.40

    # ========================================================
    # Minimum admissible widths
    # ========================================================

    rho = R + ROBOT_RADIUS

    beta1 = abs(phi2 - phi1) / 2
    beta2 = abs(phi3 - phi2) / 2

    q1 = (R - ROBOT_RADIUS) * cos(beta1)
    q2 = (R - ROBOT_RADIUS) * cos(beta2)

    width_min_90 = rho - q1
    width_min_60 = rho - q2

    # C2 participates in both junctions
    width2 = max(width_min_90, width_min_60)

    # Keep C1 and C3 narrow, but give them some experimental margin
    width1 = 0.50
    width3 = 0.50

    print(f"Minimum width at 90 deg junction: {width_min_90:.3f} m")
    print(f"Minimum width at 60 deg junction: {width_min_60:.3f} m")
    print(f"Corridor 2 width: {width2:.3f} m")

    # ========================================================
    # Corridor 1
    # ========================================================

    corridor1 = CorridorWorld(
        width=width1,
        height=length1,
        center=[
            -0.05,
            -0.70
        ],
        tilt=phi1
    )

    # ========================================================
    # Corridor 2
    # ========================================================

    tail = corridor1.head

    head = [
        tail[0] + length2 * cos(phi2),
        tail[1] + length2 * sin(phi2)
    ]

    corridor2 = get_corridor_from_vector(
        tail,
        head,
        width2,
        add_height=add_height
    )

    # ========================================================
    # Corridor 3
    # ========================================================

    tail = corridor2.head

    head = [
        tail[0] + length3 * cos(phi3),
        tail[1] + length3 * sin(phi3)
    ]

    corridor3 = get_corridor_from_vector(
        tail,
        head,
        width3,
        add_height=add_height
    )

    corridor_list = [
        corridor1,
        corridor2,
        corridor3
    ]

    # ========================================================
    # Boundary poses
    # ========================================================

    initial_pose = compute_start_pose(
        corridor1,
        unicycle,
        0.35
    )

    final_pose = compute_end_pose(
        corridor3,
        unicycle,
        0.35
    )

    assert_experiment_fits_floor(
        corridor_list,
        initial_pose,
        final_pose,
        ROBOT_RADIUS,
    )

    return corridor_list, initial_pose, final_pose


# Select experiment
# ============================================================

experiment_number = 3


if experiment_number == 1:
    corridor_list, initial_pose, final_pose = experiment_1()
    title = 'Experiment 1 - Right-Angle Turn'

elif experiment_number == 2:
    corridor_list, initial_pose, final_pose = experiment_2()
    title = 'Experiment 2 - Right-Angle Turn'

elif experiment_number == 3:
    corridor_list, initial_pose, final_pose = experiment_3()
    title = 'Experiment 3 - Multiple Corridors'

elif experiment_number == 4:
    corridor_list, initial_pose, final_pose = experiment_4()
    title = 'Experiment 4 - Narrow Corridors'

analytical_trajectory = None
motion_planner = None
if experiment_number in (1, 2, 3, 4):
    motion_planner = MotionPlanner(
        unicycle,
        corridor_list,
        start_pose=initial_pose,
        end_pose=final_pose,
        assumptions="standing",
    )
    analytical_trajectory = motion_planner.compute_trajectory_analytical()
    print(
        "Analytical trajectory computed in "
        f"{motion_planner.comp_time_analytical_sol:.3f} s."
    )


# ============================================================
# Plot
# ============================================================

if motion_planner is not None:
    figure = motion_planner.plot_planner_inputs(
        plot_intermediate_circles=False,
        plot_shrunken_corridors=True,
    )
else:
    figure = plot_corridors(
        corridor_list,
        plot_vectors=True,
        plot_corridor_index=True,
        linestyle='-'
    )

ax = figure.axes[0]

plot_projection_area(ax)

plot_robot_pose(
    ax,
    initial_pose,
    ROBOT_RADIUS,
    label='Start'
)

plot_robot_pose(
    ax,
    final_pose,
    ROBOT_RADIUS,
    label='Goal'
)

if analytical_trajectory is not None:
    plot_analytical_trajectory(
        analytical_trajectory,
        figure=figure,
        plot_circles=True,
    )

ax.set_title(title)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

if analytical_trajectory is not None:
    plot_velocity_profiles(analytical_trajectory, unicycle)

plt.show(block=True)
