from math import pi, degrees

from matplotlib import pyplot as plt

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import (
    compute_ocp_pose_to_pose_trajectory,
)


def print_ocp_rotation_primitives(ocp_result):
    """
    Print angular displacement information for the OCP rotation primitives.

    Uses:
        ocp_result["primitives_with_info"]

    This is returned by compute_ocp_pose_to_pose_trajectory(...)
    from ocp_pose_to_pose_unicycle.py.
    """

    rotation_labels = {"SpinL", "SpinR", "ArcL", "ArcR"}

    print("\nOCP rotation primitives")
    print("-----------------------")

    found_rotation = False

    for i, primitive in enumerate(ocp_result["primitives_with_info"], start=1):
        label = primitive["label"]

        if label not in rotation_labels:
            continue

        found_rotation = True

        duration = primitive["duration"]
        delta_theta_rad = primitive["delta_theta"]
        delta_theta_deg = degrees(delta_theta_rad)

        print(f"\nPrimitive {i}: {label}")
        print(f"  Duration: {duration:.3f} s")
        print(
            f"  Angular displacement: "
            f"{delta_theta_rad:.3f} rad ({delta_theta_deg:.1f} deg)"
        )

    if not found_rotation:
        print("No rotation primitives found.")


def print_all_ocp_primitives(ocp_result):
    """
    Print all OCP-classified primitives, including Straight and Stop.
    """

    print("\nAll OCP primitives")
    print("------------------")

    for i, primitive in enumerate(ocp_result["primitives_with_info"], start=1):
        label = primitive["label"]
        duration = primitive["duration"]
        delta_theta_rad = primitive["delta_theta"]
        delta_theta_deg = degrees(delta_theta_rad)

        print(f"\nPrimitive {i}: {label}")
        print(f"  Duration: {duration:.3f} s")
        print(
            f"  Angular displacement: "
            f"{delta_theta_rad:.3f} rad ({delta_theta_deg:.1f} deg)"
        )


if __name__ == "__main__":
    # Start and goal poses
    start_pose = Pose(Point(0, 0), 0 * pi / 180)
    end_pose = Pose(Point(0, 1.5), 180 * pi / 180)

    # Define Unicycle vehicle
    vehicle_width = 0.430
    vehicle_length = 0.430
    vehicle_vmax = 1.0
    vehicle_omegamax = 1.0

    unicycle = Unicycle(
        state=[0, 0, 0],
        width=vehicle_width,
        length=vehicle_length,
        v_max=vehicle_vmax,
        v_min=0,
        omega_max=vehicle_omegamax,
        omega_min=-vehicle_omegamax,
    )

    # OCP trajectory
    ocp_result = compute_ocp_pose_to_pose_trajectory(
        start_pose,
        end_pose,
        unicycle,
        analytical_initial_guess=None,
        T_guess=None,
        N=100,
    )

    print(f"OCP success: {ocp_result['success']}")
    print(f"OCP time: {ocp_result['time']:.3f} s")
    print("OCP sequence:", " - ".join(ocp_result["sequence"]))
    print(f"OCP solve time: {ocp_result['solve_time']:.3f} s")

    # Print primitive information
    print_ocp_rotation_primitives(ocp_result)

    # Optional: uncomment this if you also want Straight/Stop primitives printed
    # print_all_ocp_primitives(ocp_result)

    # Plot OCP trajectory
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        ocp_result["xs"],
        ocp_result["ys"],
        "k--",
        linewidth=2.5,
        label=f"OCP, T={ocp_result['time']:.3f} s",
    )

    # Plot start and end positions
    ax.plot(start_pose.position.x, start_pose.position.y, "go", label="Start")
    ax.plot(end_pose.position.x, end_pose.position.y, "ro", label="Goal")

    ax.set_title(f"OCP trajectory\nT = {ocp_result['time']:.3f} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    # Plot OCP velocity profiles only
    fig_vel, ax_vel = plt.subplots(figsize=(10, 5))

    t = ocp_result["ts_ctrl"]

    ax_vel.plot(t, ocp_result["vs"], label="v")
    ax_vel.plot(t, ocp_result["omegas"], label="omega")

    ax_vel.set_title("OCP velocity profiles")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("control")
    ax_vel.legend()
    ax_vel.grid(True)

    plt.show(block=True)