"""Three-corridor standing-assumptions example used in the thesis."""

import argparse
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
    plot_velocity_profiles,
)
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle


def draw_problem(ax, planner, trajectory):
    # The input plotting helper also uses pyplot's current axes.
    plt.sca(ax)
    planner.plot_planner_inputs(
        figure=ax.figure, plot_shrunken_corridors=False, plot_corridor_numbers=True,
    )
    plot_analytical_trajectory(trajectory, figure=ax, color="#2563eb", linewidth=2.5)
    ax.set(xlabel="x [m]", ylabel="y [m]")
    ax.grid(alpha=0.15)


def draw_controls(ax_v, ax_w, trajectory, vehicle):
    time = np.concatenate([p.time_grid for p in trajectory])
    for ax, values, limits, labels, ylabel in (
        (ax_v, np.concatenate([p.forward_velocity for p in trajectory]),
         (vehicle.v_max, vehicle.v_min), ("v_max", "v_min"), "$v(t)$ [m/s]"),
        (ax_w, np.concatenate([p.angular_velocity for p in trajectory]),
         (vehicle.omega_max, vehicle.omega_min), ("ω_max", "ω_min"),
         r"$\omega(t)$ [rad/s]"),
    ):
        for limit, label in zip(limits, labels):
            ax.axhline(limit, color="red", linestyle="--", label=label)
        ax.step(time, values, where="post", linewidth=2)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(labelsize=10)
    ax_v.tick_params(labelbottom=False)
    ax_w.set_xlabel("Time [s]", fontsize=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    corridor1 = CorridorWorld(width=1, height=4, center=[0, 0], tilt=pi/6)
    corridor2 = get_corridor_from_vector(
        corridor1.head,
        [corridor1.head[0] + 3*cos(pi/2), corridor1.head[1] + 3*sin(pi/2)],
        width=1,
        add_height=0.7,
    )
    corridor3 = get_corridor_from_vector(
        corridor2.head,
        [corridor2.head[0] + 4*cos(-pi/6), corridor2.head[1] + 4*sin(-pi/6)],
        width=1,
        add_height=0.7,
    )
    corridor_list = [corridor1, corridor2, corridor3]

    unicycle = Unicycle(model="Rosbot circular", footprint_radius=0.1185)
    unicycle.update(v_max=0.8)
    unicycle.update(omega_max=1)

    # Positions are normalized fractions of the shrunken corridor dimensions.
    # Heading pi/2 points along the corridor's longitudinal axis.
    relative_start_pose = [0.7, -0.65, 5*pi/4]
    relative_end_pose = [-0.7, 0.65, pi]

    mp = MotionPlanner(
        unicycle,
        corridor_list,
        relative_start_pose=relative_start_pose,
        relative_end_pose=relative_end_pose,
        assumptions="standing",
    )
    analytical_trajectory = mp.compute_trajectory_analytical()
    if not analytical_trajectory:
        raise RuntimeError("The thesis example did not produce an analytical trajectory.")

    print(f"Turning radius: {unicycle.max_radius:.3f} m")
    print(f"Initial world pose: {mp.start_pose}")
    print(f"Final world pose: {mp.end_pose}")
    print(f"Motion primitives: {len(analytical_trajectory)}")
    print(f"Traversal time: {sum(p.maneuver_time for p in analytical_trajectory):.3f} s")
    print(f"Computation time: {mp.comp_time_analytical_sol * 1000:.3f} ms")

    figure, ax = plt.subplots(figsize=(12, 6))
    draw_problem(ax, mp, analytical_trajectory)
    ax.set_title("Standing unicycle planner: three corridors")
    figure.tight_layout()
    controls = plot_velocity_profiles(analytical_trajectory, unicycle)

    combined = plt.figure(figsize=(12, 5.5), layout="constrained")
    grid = combined.add_gridspec(2, 2, width_ratios=(1.2, 1), wspace=.06)
    path_ax = combined.add_subplot(grid[:, 0])
    velocity_ax = combined.add_subplot(grid[0, 1])
    angular_ax = combined.add_subplot(grid[1, 1], sharex=velocity_ax)
    draw_problem(path_ax, mp, analytical_trajectory)
    draw_controls(velocity_ax, angular_ax, analytical_trajectory, unicycle)
    path_ax.set_title("Trajectory", fontsize=13)
    velocity_ax.set_title("Control profiles", fontsize=13)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for name, plot in (("trajectory", figure), ("controls", controls), ("combined", combined)):
            for extension in ("pdf", "png"):
                path = args.save_dir / f"unicycle_thesis_example_{name}.{extension}"
                plot.savefig(path, dpi=180, bbox_inches="tight")
                print(path)
    plt.close(figure)
    plt.close(controls)
    if not args.no_show:
        plt.show(block=True)
    else:
        plt.close("all")
    return mp, analytical_trajectory


if __name__ == "__main__":
    main()
