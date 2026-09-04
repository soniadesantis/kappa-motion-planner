from math import cos, degrees, sin

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from ..vehicle import Bicycle, Unicycle
from .helper_functions import (
    get_vehicle_vertices_no_casadi,
)


def plot_corridors(corridor_list,
                   figure = None,
                   plot_vectors = False,
                   linestyle = '--',
                   color = 'k',
                   linewidth = 1.5,
                   label = None, 
                   colormap = False):
    '''This function will plot any corridors provided as arguments'''
    import matplotlib.pyplot as plt

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)
    else:
        # ax = plt.gca()
        ax = figure.axes[0]

    # Color of the rectangles is fixed
    if colormap == False:
        ind = 0
        for corridor in corridor_list:
            corners = corridor.corners
            corners = np.vstack([corners, corners[0]])

            if ind == 0:
                ax.plot([corner[0] for corner in corners], 
                [corner[1] for corner in corners],
                color = color, linewidth= linewidth, linestyle = linestyle, label = label)
                # ax.legend()
            else:
                ax.plot([corner[0] for corner in corners], 
                [corner[1] for corner in corners],
                color = color, linewidth= linewidth, linestyle = linestyle)
            ind +=1

            if plot_vectors:
                ax.quiver(corridor.center[0], corridor.center[1], cos(corridor.tilt), sin(corridor.tilt), color='k', pivot='middle', angles='xy', scale_units='xy', scale=1/corridor.height) 
            
        # Colors of the rectangles have a gradient from yellow to purple
    else: 
        # Define a colormap from yellow to purple
        cmap = mcolors.LinearSegmentedColormap.from_list("yellow_purple", ["yellow", "purple"])
        # Normalize indices so they map to [0, 1]
        norm = mcolors.Normalize(vmin=0, vmax=len(corridor_list)-1)
        # Generate colors for each rectangle
        colors = [cmap(norm(i)) for i in range(len(corridor_list))]

        ind = 0
        for i, corridor in enumerate(corridor_list):
            corners = corridor.corners
            corners.append(corners[0])

            if ind == 0:
                ax.plot([corner[0] for corner in corners], 
                [corner[1] for corner in corners],
                color = colors[i], linewidth= linewidth, linestyle = linestyle, label = label)
                # ax.legend()
            else:
                ax.plot([corner[0] for corner in corners], 
                [corner[1] for corner in corners],
                color = colors[i], linewidth= linewidth, linestyle = linestyle)
            ind +=1

            if plot_vectors:
                ax.quiver(corridor.center[0], corridor.center[1], cos(corridor.tilt), sin(corridor.tilt), color='k', pivot='middle', angles='xy', scale_units='xy', scale=1/corridor.height) 
            
    ax.set_aspect('equal')
    return figure

def vehic_to_plot(figure, x, y, theta, w_left, w_right, l_front, l_back, color='b'):
    import matplotlib as mlt
    import matplotlib.pylab as plt

    if figure is None: figure = plt.figure()
    ax = figure if isinstance(figure, mlt.axes._axes.Axes) else figure.axes[0]

    vertices_veh = get_vehicle_vertices_no_casadi(x, y, theta, w_left, w_right, l_front, l_back)
    vertices_veh_ext = np.append(vertices_veh, np.array([[vertices_veh[0][0]],[ vertices_veh[1][0]]]), axis=1)
    ax.plot(vertices_veh_ext[0,:],vertices_veh_ext[1,:], color=color)

    return figure


def plot_vehicle_rotating_on_the_spot(figure, maneuver, unicycle):
    for i in range(maneuver.samples_number):
        figure = vehic_to_plot(figure, maneuver.path_coordinates[i,0], maneuver.path_coordinates[i,1], maneuver.theta_trajectory[i], unicycle.width/2, unicycle.width/2, unicycle.length/2, unicycle.length/2, color='g')
    return figure

def plot_vehicle_curvilinear_arc(figure, maneuver, unicycle):
    for i in range(maneuver.samples_number):
        figure = vehic_to_plot(figure, maneuver.path_coordinates[i,0], maneuver.path_coordinates[i,1], maneuver.theta_trajectory[i], unicycle.width/2, unicycle.width/2, unicycle.length/2, unicycle.length/2, color='g')
    return figure

def plot_vehicle_linear_segment(figure, maneuver, unicycle):
    for i in range(maneuver.samples_number):
        figure = vehic_to_plot(figure, maneuver.path_coordinates[i,0], maneuver.path_coordinates[i,1], maneuver.theta_trajectory[i], unicycle.width/2, unicycle.width/2, unicycle.length/2, unicycle.length/2, color='g')
    return figure

def plot_unicycle_along_trajectory(figure, unicycle, path_coordinates, theta_vector, color='g'):
    for i in range(theta_vector.shape[0]):
        figure = vehic_to_plot(figure, path_coordinates[i,0], path_coordinates[i,1], theta_vector[i], unicycle.width/2, unicycle.width/2, unicycle.length/2, unicycle.length/2, color='g')
    return figure

def plot_vehicle(vehicle, figure, color='b', plot_vector=False):
    import matplotlib.pylab as plt
    
    state = vehicle.state
    width = vehicle.width
    length = vehicle.length

    vertices_veh = get_vehicle_vertices_no_casadi(state[0], state[1], state[2], width/2, width/2, length/2, length/2)
    vertices_veh_ext = np.append(vertices_veh, np.array([[vertices_veh[0][0]],[ vertices_veh[1][0]]]), axis=1)

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)
    else:
        # ax = plt.gca()
        ax = figure.axes[0]

    ax.plot(vertices_veh_ext[0,:],vertices_veh_ext[1,:], color=color)
    if plot_vector:
        ax.quiver(state[0], state[1], cos(state[2]), sin(state[2]), color=color, angles='xy', scale_units='xy', scale=2/length) 

def plot_planner_inputs(planner, figure=None, plot_intermediate_circles = False, plot_shrunken_corridors = True ):
    """
    Plot the corridors and start/end poses of a MotionPlanner instance.

    :param planner: The motion planner to visualize.
    :type planner: MotionPlanner
    :param figure: Figure to plot on. If ``None``, a new one is created.
    :type figure: matplotlib.figure.Figure, optional
    :returns: The resulting matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """

    figure = plot_corridors(planner.corridor_list, figure)
    if plot_shrunken_corridors:
        plot_corridors(planner.shrunken_corridor_list, figure)

    r = planner.vehicle.width * 0.5
    l = r + 0.3
    x0, y0, theta0 = planner.start_pose
    xf, yf, thetaf = planner.end_pose

    plt.plot(x0, y0, 'ro')
    plt.arrow(x0, y0, l * cos(theta0), l * sin(theta0), head_width=0.1, color='r')
    plt.plot(xf, yf, 'ro')
    plt.arrow(xf, yf, l * cos(thetaf), l * sin(thetaf), head_width=0.1, color='r')

    angle_array = np.linspace(0, 2 * np.pi, 10000)
    plt.plot(x0 + r * np.cos(angle_array), y0 + r * np.sin(angle_array), 'r-', linewidth=0.5)
    plt.plot(xf + r * np.cos(angle_array), yf + r * np.sin(angle_array), 'r-', linewidth=0.5)
    if plot_intermediate_circles:
        R = planner.vehicle.max_radius
        centers = planner.intermediate_circle_centers
        for center in centers:
            plt.plot(center[0], center[1], 'ko', markersize = 0.7)
            plt.plot(center[0] + R * np.cos(angle_array), center[1] + R * np.sin(angle_array), 'r--', linewidth=0.5)

    return figure

def plot_path_all_trajectories(trajectory, figure = None, color = 'k', linestyle = 'solid', linewidth = 2.5, label = None):
    import matplotlib as mlt
    import matplotlib.pyplot as plt

    if isinstance(figure, mlt.axes._axes.Axes): # If figure is an axis instead of an actual figure object
        if label is not None:
            figure.plot(trajectory.path_coordinates[:,0], trajectory.path_coordinates[:,1], color = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            figure.plot(trajectory.path_coordinates[:,0], trajectory.path_coordinates[:,1], color = color, linestyle = linestyle, linewidth = linewidth)
    else:
        if figure is None: figure = plt.figure()
        plt.plot(trajectory.path_coordinates[:,0], trajectory.path_coordinates[:,1], color = color, linestyle = linestyle, linewidth = linewidth, label = label)
    return figure

def plot_circle(Arc, figure = None, color = 'b', linestyle = 'dashed', linewidth = 0.5):
    import matplotlib as mlt
    import matplotlib.pyplot as plt
    
    if figure is not None:
        if isinstance(figure, mlt.axes._axes.Axes): # If figure is an axis instead of an actual figure object
            figure.plot(Arc.xc, Arc.yc, 'ko', markersize = 0.7)
            figure.plot(Arc.xc + Arc.radius * np.cos(np.linspace(0,2*np.pi,100)), Arc.yc + Arc.radius * np.sin(np.linspace(0,2*np.pi,100)), color = color, linestyle = linestyle, linewidth = linewidth)
        else:
            plt.plot(Arc.xc, Arc.yc, 'ko', markersize = 0.7)
            plt.plot(Arc.xc + Arc.radius * np.cos(np.linspace(0,2*np.pi,100)), Arc.yc + Arc.radius * np.sin(np.linspace(0,2*np.pi,100)), color = color, linestyle = linestyle, linewidth = linewidth)
    else:
        figure = plt.figure()
        plt.plot(Arc.xc, Arc.yc, 'ko', markersize = 0.7)
        plt.plot(Arc.xc + Arc.radius * np.cos(np.linspace(0,2*np.pi,100)), Arc.yc + Arc.radius * np.sin(np.linspace(0,2*np.pi,100)), color = color, linestyle = linestyle, linewidth = linewidth)
    return figure


def plot_turn_on_spot_sector(
    primitive,
    ax,
    radius=0.45,
    color="orange",
    alpha=0.25,
    zorder=9,
):
    """
    Plot a transparent sector showing the angular motion
    of a turn-on-the-spot primitive.
    """

    theta0_deg = degrees(primitive.theta0)
    thetaf_deg = degrees(primitive.thetaf)

    if primitive.turn_direction > 0:
        theta1 = theta0_deg
        theta2 = thetaf_deg
    else:
        theta1 = thetaf_deg
        theta2 = theta0_deg

    sector = Wedge(
        center=(primitive.x0, primitive.y0),
        r=radius,
        theta1=theta1,
        theta2=theta2,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_patch(sector)


def plot_primitive_arrows_and_markers(
    trajectory,
    ax,
    arrow_length=0.35,
    marker_size=35,
    start_color="black",
    end_color="black",
    zorder=10,
    plot_turn_sectors=True,
):
    for primitive in trajectory:

        if primitive.label == "turn on-the-spot" and plot_turn_sectors:
            plot_turn_on_spot_sector(
                primitive,
                ax,
                radius=arrow_length * 1.4,
                zorder=zorder - 1,
            )

        x0, y0 = primitive.x0, primitive.y0
        xf, yf = primitive.xf, primitive.yf

        theta0 = primitive.theta0
        thetaf = primitive.thetaf

        for x, y, theta, color in [
            (x0, y0, theta0, start_color),
            (xf, yf, thetaf, end_color),
        ]:
            ax.scatter(
                x,
                y,
                s=marker_size,
                color=color,
                zorder=zorder,
            )

            ax.arrow(
                x,
                y,
                arrow_length * cos(theta),
                arrow_length * sin(theta),
                head_width=0.08,
                head_length=0.08,
                fc=color,
                ec=color,
                length_includes_head=True,
                zorder=zorder + 1,
            )


def plot_analytical_trajectory(
    trajectory,
    figure=None,
    plot_circles=False,
    linewidth=2,
    color="b",
    plot_primitive_arrows=False,
    plot_turn_sectors=True,
):
    """
    Plot an analytical trajectory composed of multiple trajectory segments.
    """
    from ..trajectory import BackwardArc, CurvilinearArcUnicycle

    if figure is None:
        fig, ax = plt.subplots()
        figure = ax
    else:
        ax = figure

    for trajectory_piece in trajectory:
        trajectory_piece.plot_path(
            ax,
            color=color,
            linewidth=linewidth,
        )

    if plot_circles:
        for trajectory_piece in trajectory:
            if (
                isinstance(trajectory_piece, CurvilinearArcUnicycle)
                or isinstance(trajectory_piece, BackwardArc)
            ):
                trajectory_piece.plot_circle(ax)

    if plot_primitive_arrows:
        plot_primitive_arrows_and_markers(
            trajectory,
            ax,
            plot_turn_sectors=plot_turn_sectors,
        )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    return figure


def plot_velocity_profiles(trajectory, vehicle=None):
    """
    Plot control profiles for a unicycle or bicycle trajectory.

    If the vehicle is a Unicycle:
        Plots v(t) and ω(t).
    If the vehicle is a Bicycle:
        Plots v(t) and δ(t), where δ is inferred from ω by sign.

    Parameters
    ----------
    trajectory : list
        Sequence of trajectory segments, each with:
            - time_grid : array-like
            - forward_velocity : array-like
            - angular_velocity : array-like
    vehicle : Unicycle or Bicycle object (optional)
        Used to show saturation limits.
    """

    time_global = []
    v_global = []
    omega_global = []

    # Build continuous time and control profiles
    motion_time = 0.0
    for segment in trajectory:
        t = np.array(segment.time_grid)
        v = np.array(segment.forward_velocity)
        w = np.array(segment.angular_velocity)
        motion_time += segment.maneuver_time

        time_global.append(t)
        v_global.append(v)
        omega_global.append(w)

    time_global = np.concatenate(time_global)
    v_global = np.concatenate(v_global)
    omega_global = np.concatenate(omega_global)

    # If bicycle, convert ω → δ
    if vehicle is not None and isinstance(vehicle, Bicycle):
        delta_global = np.zeros_like(omega_global)
        delta_global[omega_global > 0] = vehicle.delta_max
        delta_global[omega_global < 0] = vehicle.delta_min
        # when omega == 0 -> 0 (already set)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ---- Forward velocity plot ----
    ax1.set_ylabel("Forward velocity v(t) [m/s]")
    ax1.grid(True)
    if vehicle is not None:
        ax1.axhline(vehicle.v_max, color='red', linestyle='--', label='v_max')
        ax1.axhline(vehicle.v_min, color='red', linestyle='--', label='v_min')
        ax1.legend(loc='upper right')
    ax1.step(time_global, v_global, where='post', linewidth=2)

    # ---- Angular velocity or steering angle plot ----
    if vehicle is None or isinstance(vehicle, Unicycle):
        # Unicycle → plot omega
        ax2.set_ylabel("Angular velocity ω(t) [rad/s]")
        if vehicle is not None:
            ax2.axhline(vehicle.omega_max, color='red', linestyle='--', label='ω_max')
            ax2.axhline(vehicle.omega_min, color='red', linestyle='--', label='ω_min')
            ax2.legend(loc='upper right')
        ax2.step(time_global, omega_global, where='post', linewidth=2)

    else:  # Bicycle → plot delta
        ax2.set_ylabel("Steering angle δ(t) [rad]")
        ax2.axhline(vehicle.delta_max, color='red', linestyle='--', label='δ_max')
        ax2.axhline(vehicle.delta_min, color='red', linestyle='--', label='δ_min')
        ax2.legend(loc='upper right')
        ax2.step(time_global, delta_global, where='post', linewidth=2)

    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    # Figure title
    fig.suptitle(f"Control Profiles — Total motion time: {motion_time:.2f} s", fontsize=14)

    plt.tight_layout()
    return fig


def get_analytical_control_profiles(trajectory):
    time_global = []
    v_global = []
    omega_global = []

    for segment in trajectory:
        time_global.append(np.array(segment.time_grid))
        v_global.append(np.array(segment.forward_velocity))
        omega_global.append(np.array(segment.angular_velocity))

    return (
        np.concatenate(time_global),
        np.concatenate(v_global),
        np.concatenate(omega_global),
    )


def plot_velocity_profiles_comparison(
    analytical_trajectory,
    ocp_result,
    vehicle=None,
    analytical_label="Best analytical",
    ocp_label="OCP",
):
    time_analytical, v_analytical, omega_analytical = get_analytical_control_profiles(
        analytical_trajectory
    )

    time_ocp = ocp_result["ts_ctrl"]
    v_ocp = ocp_result["vs"]
    omega_ocp = ocp_result["omegas"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax1.set_ylabel("Forward velocity v(t) [m/s]")
    ax1.grid(True)

    if vehicle is not None:
        ax1.axhline(vehicle.v_max, color="gray", linestyle="--", linewidth=1)
        ax1.axhline(vehicle.v_min, color="gray", linestyle="--", linewidth=1)

    ax1.step(
        time_analytical,
        v_analytical,
        where="post",
        linewidth=2,
        label=analytical_label,
    )

    ax1.step(
        time_ocp,
        v_ocp,
        where="post",
        linewidth=2,
        linestyle="--",
        label=ocp_label,
    )

    ax1.legend()

    ax2.set_ylabel("Angular velocity ω(t) [rad/s]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    if vehicle is not None:
        ax2.axhline(vehicle.omega_max, color="gray", linestyle="--", linewidth=1)
        ax2.axhline(vehicle.omega_min, color="gray", linestyle="--", linewidth=1)
        ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1)

    ax2.step(
        time_analytical,
        omega_analytical,
        where="post",
        linewidth=2,
        label=analytical_label,
    )

    ax2.step(
        time_ocp,
        omega_ocp,
        where="post",
        linewidth=2,
        linestyle="--",
        label=ocp_label,
    )

    ax2.legend()

    analytical_time = time_analytical[-1]
    ocp_time = ocp_result["time"]

    fig.suptitle(
        f"Control comparison\n"
        f"{analytical_label}: {analytical_time:.3f} s | "
        f"{ocp_label}: {ocp_time:.3f} s"
    )

    plt.tight_layout()

    return fig


def plot_circular_footprint(
    trajectory,
    radius,
    ax=None,
    step=1,
    color="k",
    linewidth=0.5,
    linestyle="-",
    alpha=0.2,
):
    """
    Plot circular footprints along the trajectory path.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for trajectory_piece in trajectory:

        if not hasattr(trajectory_piece, "path_coordinates"):
            continue

        path_coordinates = trajectory_piece.path_coordinates

        for point in path_coordinates[::step]:

            x, y = point

            footprint = plt.Circle(
                (x, y),
                radius,
                fill=False,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )

            ax.add_patch(footprint)

    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    return ax


def plot_rectangular_footprint(
    trajectory,
    width,
    front_overhang,
    rear_overhang,
    ax=None,
    step=1,
    color="k",
    linewidth=0.5,
    linestyle="-",
    alpha=0.2,
):
    """
    Plot rectangular vehicle footprints along a trajectory.

    The path coordinates are assumed to represent the midpoint of the rear axle.
    The rectangle extends:
    - front_overhang forward from the rear axle midpoint
    - rear_overhang backward from the rear axle midpoint
    - width / 2 laterally on each side
    """

    if ax is None:
        fig, ax = plt.subplots()

    half_width = width / 2.0

    for trajectory_piece in trajectory:
        if not hasattr(trajectory_piece, "path_coordinates"):
            continue

        if not hasattr(trajectory_piece, "theta_trajectory"):
            continue

        path_coordinates = trajectory_piece.path_coordinates
        theta_trajectory = trajectory_piece.theta_trajectory

        for point, theta in zip(
            path_coordinates[::step],
            theta_trajectory[::step],
        ):
            x, y = point

            # Rectangle corners in body frame, relative to rear axle midpoint
            body_corners = np.array([
                [front_overhang,  half_width],
                [front_overhang, -half_width],
                [-rear_overhang, -half_width],
                [-rear_overhang,  half_width],
                [front_overhang,  half_width],
            ])

            c = np.cos(theta)
            s = np.sin(theta)

            rotation = np.array([
                [c, -s],
                [s,  c],
            ])

            world_corners = body_corners @ rotation.T
            world_corners[:, 0] += x
            world_corners[:, 1] += y

            ax.plot(
                world_corners[:, 0],
                world_corners[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )

    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    return ax


def plot_turning_front_corner_path(
    trajectory,
    width,
    front_overhang,
    ax=None,
    color="r",
    linewidth=2,
    linestyle="--",
):
    """
    Plot the front-left or front-right footprint corner path
    for CurvilinearArcUnicycle pieces.

    If turn_direction == 1, plot the front-left corner.
    If turn_direction == -1, plot the front-right corner.
    """

    from ..trajectory import CurvilinearArcUnicycle

    if ax is None:
        fig, ax = plt.subplots()

    half_width = width / 2.0

    for trajectory_piece in trajectory:
        if not isinstance(trajectory_piece, CurvilinearArcUnicycle):
            continue

        path_coordinates = trajectory_piece.path_coordinates
        theta_trajectory = trajectory_piece.theta_trajectory
        turn_direction = trajectory_piece.turn_direction

        x = path_coordinates[:, 0]
        y = path_coordinates[:, 1]
        theta = theta_trajectory

        front_x = front_overhang * np.cos(theta)
        front_y = front_overhang * np.sin(theta)

        left_x = -half_width * np.sin(theta)
        left_y = half_width * np.cos(theta)

        if turn_direction == -1:
            corner_x = x + front_x + left_x
            corner_y = y + front_y + left_y
        elif turn_direction == 1:
            corner_x = x + front_x - left_x
            corner_y = y + front_y - left_y
        else:
            continue

        ax.plot(
            corner_x,
            corner_y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    ax.set_aspect("equal", adjustable="box")

    return ax
    #### TO-DO
    # def plot_forward_velocity(self, figure = None, color = 'k', linestyle = 'solid', label = 'Forward_velocity', step=False, legend=True):
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mlt
        
    #     if figure is None: figure = plt.figure()
    #     ax = figure if isinstance(figure, mlt.axes._axes.Axes) else figure.axes[0]
    #     if not step:
    #         ax.plot(self.time_grid, self.forward_velocity, color = color, linestyle = linestyle, label = label)
    #     else:
    #         ax.step(self.time_grid, self.forward_velocity, color = color, linestyle = linestyle,  label = label)
    #     if legend:
    #         ax.legend()
    #     return figure
    
    # def plot_theta_trajectory(self, figure = None, plot_style = 'k-', label = 'Theta_trajectory', step=False, legend=True):
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mlt
        
    #     if figure is None: figure = plt.figure()
    #     ax = figure if isinstance(figure, mlt.axes._axes.Axes) else figure.axes[0]
    #     if not step:
    #         ax.plot(self.time_grid, self.theta_trajectory, plot_style, label = label)
    #     else:
    #         ax.step(self.time_grid, self.theta_trajectory, plot_style, label = label)
    #     if legend:
    #         ax.legend()
    #     return figure

    # def plot_angular_velocity(self, figure = None, color = 'k', linestyle = 'solid', label = 'Angular_velocity', step = False, legend=True):
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mlt
        
    #     if figure is None: figure = plt.figure()
    #     ax = figure if isinstance(figure, mlt.axes._axes.Axes) else figure.axes[0]
    #     if not step:
    #         ax.plot(self.time_grid, self.angular_velocity, color = color, linestyle = linestyle, label = label)
    #     else:
    #         ax.step(self.time_grid, self.angular_velocity, color = color, linestyle = linestyle, label = label)

    #     if legend:
    #         ax.legend()

    #     return figure
    
    # def plot_steering_angle(self, figure = None, color = 'k', linestyle = 'solid', label = 'Steering_angle', step = False, legend=True):
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mlt
        
    #     if figure is None: figure = plt.figure()
    #     ax = figure if isinstance(figure, mlt.axes._axes.Axes) else figure.axes[0]
    #     if not step:
    #         ax.plot(self.time_grid, self.steering_angle, color = color, linestyle = linestyle, label = label)
    #     else:
    #         ax.step(self.time_grid, self.steering_angle, color = color, linestyle = linestyle, label = label)

    #     if legend:
    #         ax.legend()

    #     return figure
