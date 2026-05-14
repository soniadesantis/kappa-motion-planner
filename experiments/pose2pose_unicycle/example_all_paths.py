from math import cos, sin, degrees, pi

from matplotlib import pyplot as plt

from kappa_planner.helpers.pose_to_pose_unicycle import compute_CSC_trajectory, compute_TCSC_trajectory, compute_CSCT_trajectory, compute_TCSCT_trajectory
from kappa_planner.helpers.geometry_operations import compute_distance_two_points
from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory, plot_velocity_profiles, get_analytical_control_profiles, plot_velocity_profiles_comparison
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import (
    compute_ocp_pose_to_pose_trajectory,
    compute_initial_guess_from_analytical_trajectory,
)

from math import degrees


def get_trajectory_info(trajectory):
    """
    Return useful information about a trajectory composed of motion primitives.
    """

    info = {
        "primitive_sequence": [],
        "primitives": [],
        "total_duration": 0.0,
    }

    for i, primitive in enumerate(trajectory, start=1):
        primitive_info = {
            "index": i,
            "type": primitive.label,
            "duration": primitive.maneuver_time,
        }

        info["primitive_sequence"].append(primitive.label)
        info["total_duration"] += primitive.maneuver_time

        if primitive.label == "turn on-the-spot":
            primitive_info.update({
                "turn_angle_rad": primitive.delta_angle,
                "turn_angle_deg": degrees(primitive.delta_angle),
                "turn_direction": primitive.turn_direction,
            })

        elif primitive.label == "arc":
            primitive_info.update({
                "arc_angle_rad": primitive.iota,
                "arc_angle_deg": degrees(primitive.iota),
                "turn_direction": primitive.turn_direction,
                "radius": primitive.radius,
                "arc_length": primitive.path_length,
            })

        elif primitive.label == "segment":
            primitive_info.update({
                "length": primitive.path_length,
            })

        info["primitives"].append(primitive_info)

    return info


def print_trajectory_info(name, trajectory):
    info = get_trajectory_info(trajectory)

    print(f"\n{name}")
    print("-" * len(name))
    print("Primitive sequence:", " -> ".join(info["primitive_sequence"]))
    print(f"Total duration: {info['total_duration']:.3f} s")

    for primitive in info["primitives"]:
        print(f"\nPrimitive {primitive['index']}: {primitive['type']}")
        print(f"  Duration: {primitive['duration']:.3f} s")

        if primitive["type"] == "turn on-the-spot":
            print(f"  Turn angle: {primitive['turn_angle_rad']:.3f} rad "
                  f"({primitive['turn_angle_deg']:.1f} deg)")
            print(f"  Turn direction: {primitive['turn_direction']}")

        elif primitive["type"] == "arc":
            print(f"  Arc amplitude: {primitive['arc_angle_rad']:.3f} rad "
                  f"({primitive['arc_angle_deg']:.1f} deg)")
            print(f"  Turn direction: {primitive['turn_direction']}")
            print(f"  Radius: {primitive['radius']:.3f} m")
            print(f"  Arc length: {primitive['arc_length']:.3f} m")

        elif primitive["type"] == "segment":
            print(f"  Segment length: {primitive['length']:.3f} m")


def tau_to_string(tau):
    return "L" if tau == 1 else "R"


def get_primitive_amplitudes_string(trajectory):
    parts = []

    for primitive in trajectory:
        if primitive.label == "turn on-the-spot":
            parts.append(f"T={degrees(abs(primitive.delta_angle)):.1f}°")

        elif primitive.label == "arc":
            parts.append(f"C={degrees(abs(primitive.iota)):.1f}°")

        elif primitive.label == "segment":
            parts.append(f"S={primitive.path_length:.2f}m")

    return ", ".join(parts)


def plot_all_trajectories_grid(trajectories, rows=4, cols=4):
    """
    Plot all candidate trajectories in a grid.

    trajectories should be a dictionary like:

    {
        "CSC left-left": {
            "trajectory": CSC_left_left,
            "time": total_time_CSC_left_left,
            "type": "CSC",
            "tau0": tau_left,
            "tauf": tau_left,
        },
        ...
    }
    """

    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    axes = axes.flatten()

    best_name = min(
        trajectories,
        key=lambda name: trajectories[name]["time"],
    )

    for ax, (name, data) in zip(axes, trajectories.items()):
        trajectory = data["trajectory"]
        total_time = data["time"]
        trajectory_type = data["type"]
        tau0 = data["tau0"]
        tauf = data["tauf"]

        is_best = name == best_name
        color = "red" if is_best else "blue"
        linewidth = 3 if is_best else 1.8

        plot_analytical_trajectory(
            trajectory,
            figure=ax,
            plot_circles=True,
            color=color,
            linewidth=linewidth,
            plot_primitive_arrows=True,
            plot_turn_sectors=True,
        )

        amplitudes = get_primitive_amplitudes_string(trajectory)

        title = (
            f"{trajectory_type} "
            f"{tau_to_string(tau0)}-{tau_to_string(tauf)}\n"
            f"time = {total_time:.2f} s\n"
            f"{amplitudes}"
        )

        if is_best:
            title = "BEST\n" + title

        ax.set_title(title, fontsize=9)
        ax.axis("equal")
        ax.grid(True)

    for ax in axes[len(trajectories):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    start_pose = Pose(Point(0, 0), pi)
    end_pose = Pose(Point(0, 10), pi)

    ### Define Unicycle vehicle ###
    vehicle_width = 0.430
    vehicle_length = 0.430
    vehicle_vmax = 0.5
    vehicle_omegamax = 0.5

    unicycle = Unicycle(
        state = [0,0,0],
        width = vehicle_width,
        length = vehicle_length,
        v_max = vehicle_vmax,
        v_min = 0,
        omega_max = vehicle_omegamax,
        omega_min = -vehicle_omegamax)

    # Analytical trajectories
    try:
        trajectories = compute_all_pose_to_pose_trajectories(
            start_pose,
            end_pose,
            unicycle,
        )

    except ValueError as e:
        print(e)

    fig = plot_all_trajectories_grid(trajectories)

    best_name, best_data = min(
        trajectories.items(),
        key=lambda item: item[1]["time"],
    )

    best_trajectory = best_data["trajectory"]
    best_time = best_data["time"]

    print(f"Shortest trajectory: {best_name}")
    print(f"Total time: {best_time:.3f} s")

    analytical_initial_guess = best_trajectory
    # OCP trajectory
    ocp_result = compute_ocp_pose_to_pose_trajectory(
        start_pose,
        end_pose,
        unicycle,
        analytical_initial_guess=best_trajectory,
        T_guess=best_time,
        N = 30,
    )
    initial_guess = ocp_result["initial_guess"]

    print(f"OCP time: {ocp_result['time']:.3f} s")
    print("OCP sequence:", " - ".join(ocp_result["sequence"]))
    print(f"OCP solve time: {ocp_result['solve_time']:.3f} s")

    comparison_ax = plot_analytical_trajectory(
        best_trajectory,
        plot_circles=True,
        color="blue",
        linewidth=2.5,
        plot_primitive_arrows=True,
    )

    if analytical_initial_guess is not None:
        comparison_ax.plot(
            initial_guess["x"],
            initial_guess["y"],
            color="gray",
            linestyle=":",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Initial guess (OCP grid)",
        )

    comparison_ax.plot(
        ocp_result["xs"],
        ocp_result["ys"],
        "k--",
        linewidth=2,
        label=f"OCP, T={ocp_result['time']:.3f} s",
    )

    comparison_ax.set_title(
        f"Best analytical vs OCP\n"
        f"{best_name}: {best_time:.3f} s | "
        f"OCP: {ocp_result['time']:.3f} s"
    )

    comparison_ax.legend()
    comparison_ax.axis("equal")
    comparison_ax.grid(True)

    print(f"Best analytical time: {best_time:.3f} s")
    print(f"OCP time: {ocp_result['time']:.3f} s")
    print(f"Difference: {ocp_result['time'] - best_time:.6f} s")

    figure_vel = plot_velocity_profiles_comparison(
        best_trajectory,
        ocp_result,
        vehicle=unicycle,
        analytical_label=best_name,
        ocp_label="OCP",
    )


    plt.axis("equal")
    plt.show(block=True)
