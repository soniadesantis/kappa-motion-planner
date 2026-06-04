from math import pi, degrees
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import (
    compute_ocp_pose_to_pose_trajectory,
)
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory


def print_case_result(
    case_id,
    n_cases,
    theta0,
    thetaf,
    best_name,
    best_time,
    ocp_result,
):
    ocp_time = ocp_result["time"]
    solve_time = ocp_result["solve_time"]
    sequence = " - ".join(ocp_result["sequence"])
    time_difference = ocp_time - best_time

    if abs(time_difference) > 0.2:

        print("  -> Plotting suspicious case")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Analytical trajectory
        plot_analytical_trajectory(
            best_trajectory,
            figure=ax,
            plot_circles=True,
            color="blue",
            linewidth=2.5,
            plot_primitive_arrows=True,
            plot_turn_sectors=True,
        )

        # OCP trajectory
        ax.plot(
            ocp_result["xs"],
            ocp_result["ys"],
            "k--",
            linewidth=2.0,
            label=f"OCP ({ocp_result['time']:.3f} s)",
        )

        # OCP initial guess
        initial_guess = ocp_result["initial_guess"]

        ax.plot(
            initial_guess["x"],
            initial_guess["y"],
            color="gray",
            linestyle=":",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Initial guess",
        )

        # Start pose
        arrow_scale = 0.5

        ax.quiver(
            x0,
            y0,
            arrow_scale * np.cos(theta0),
            arrow_scale * np.sin(theta0),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
        )

        # Goal pose
        ax.quiver(
            xf,
            yf,
            arrow_scale * np.cos(thetaf),
            arrow_scale * np.sin(thetaf),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )

        ax.set_title(
            f"Case {case_id:02d}\n"
            f"{degrees(theta0):.0f}° → {degrees(thetaf):.0f}°\n"
            f"Analytical: {best_time:.3f} s\n"
            f"OCP: {ocp_result['time']:.3f} s\n"
            f"ΔT = {time_difference:.3f} s"
        )

        ax.axis("equal")
        ax.grid(True)
        ax.legend()

        plt.show(block=True)

    print(
        f"\nCase {case_id:02d}/{n_cases}: "
        f"theta0={degrees(theta0):6.1f} deg, "
        f"thetaf={degrees(thetaf):6.1f} deg"
    )

    print(f"  Best analytical : {best_name}")
    print(f"  Analytical time : {best_time:.6f} s")
    print(f"  OCP time        : {ocp_time:.6f} s")
    print(f"  Difference      : {time_difference:.6f} s")
    print(f"  OCP solve time  : {solve_time:.3f} s")
    print(f"  OCP sequence    : {sequence}")


def compute_arc_segment_ratios(trajectory):

    arcs = [
        p for p in trajectory
        if p.label.lower() == "arc"
    ]

    segments = [
        p for p in trajectory
        if p.label.lower() == "segment"
    ]

    if len(arcs) != 2 or len(segments) != 1:
        return {
            "r1": None,
            "r2": None,
        }

    arc1 = arcs[0]
    arc2 = arcs[1]
    segment = segments[0]

    r1 = (
        segment.path_length / arc1.path_length
        if arc1.path_length > 1e-12
        else None
    )

    r2 = (
        segment.path_length / arc2.path_length
        if arc2.path_length > 1e-12
        else None
    )

    return {
        "arc1_length": arc1.path_length,
        "segment_length": segment.path_length,
        "arc2_length": arc2.path_length,
        "r1": r1,
        "r2": r2,
    }


def extract_analytical_primitive_info(trajectory):
    """
    Extract serializable information from an analytical trajectory.

    Parameters
    ----------
    trajectory : list
        List of motion primitive objects.

    Returns
    -------
    list of dict
        One dictionary per primitive.
    """

    primitives_info = []

    for i, primitive in enumerate(trajectory, start=1):

        primitive_info = {
            "index": i,
            "label": primitive.label,
            "maneuver_time": float(primitive.maneuver_time),
            "path_length": float(primitive.path_length),
        }

        # Optional but useful: save turn direction when available
        if hasattr(primitive, "turn_direction"):
            primitive_info["turn_direction"] = int(primitive.turn_direction)

        # Optional: save angular amplitude for arcs and turns
        if hasattr(primitive, "iota"):
            primitive_info["angular_amplitude_rad"] = float(primitive.iota)
            primitive_info["angular_amplitude_deg"] = float(degrees(primitive.iota))

        elif hasattr(primitive, "delta_angle"):
            primitive_info["angular_amplitude_rad"] = float(primitive.delta_angle)
            primitive_info["angular_amplitude_deg"] = float(degrees(primitive.delta_angle))

        # Optional: save start/end poses
        if hasattr(primitive, "start_pose"):
            primitive_info["start_pose"] = [
                float(value) for value in primitive.start_pose
            ]

        if hasattr(primitive, "end_pose"):
            primitive_info["end_pose"] = [
                float(value) for value in primitive.end_pose
            ]

        primitives_info.append(primitive_info)

    return primitives_info


if __name__ == "__main__":

    # Fixed positions
    x0, y0 = 0.0, 0.0
    xf, yf = 0.0, 5.0

    # Vehicle
    vehicle_width = 0.430
    vehicle_length = 0.430
    vehicle_vmax = 1
    vehicle_omegamax = 1

    unicycle = Unicycle(
        state=[0, 0, 0],
        width=vehicle_width,
        length=vehicle_length,
        v_max=vehicle_vmax,
        v_min=0,
        omega_max=vehicle_omegamax,
        omega_min=-vehicle_omegamax,
    )

    # Angle sweep
    n_angles = 4
    start_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)
    final_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)

    n_cases = n_angles * n_angles
    case_id = 0
    N = 30
    M = 4
    analytical_initial_guess = True

    results = []

    print("\n" + "=" * 80)
    print("POSE-TO-POSE UNICYCLE SWEEP")
    print("=" * 80)

    t0_simulation = time.perf_counter()

    for theta0 in start_angles:
        for thetaf in final_angles:

            case_id += 1

            start_pose = Pose(Point(x0, y0), theta0)
            end_pose = Pose(Point(xf, yf), thetaf)

            try:
                t0 = time.perf_counter()
                trajectories = compute_all_pose_to_pose_trajectories(
                    start_pose,
                    end_pose,
                    unicycle,
                )
                best_name, best_data = min(
                    trajectories.items(),
                    key=lambda item: item[1]["time"],
                )

                best_trajectory = best_data["trajectory"]
                best_time = best_data["time"]
                analytical_solve_time = time.perf_counter() - t0
                best_analytical_primitives = extract_analytical_primitive_info(
                    best_trajectory
                )
                ratios = compute_arc_segment_ratios(
                    best_trajectory
                )

            except ValueError as e:
                print(
                    f"\nCase {case_id:02d}/{n_cases}: "
                    f"theta0={degrees(theta0):6.1f} deg, "
                    f"thetaf={degrees(thetaf):6.1f} deg"
                )
                print(f"  Skipped: {e}")

                results.append({
                    "case_id": case_id,
                    "success": False,

                    "x0": float(x0),
                    "y0": float(y0),
                    "xf": float(xf),
                    "yf": float(yf),

                    "theta0_rad": float(theta0),
                    "thetaf_rad": float(thetaf),
                    "theta0_deg": float(degrees(theta0)),
                    "thetaf_deg": float(degrees(thetaf)),

                    "v_max": float(vehicle_vmax),
                    "omega_max": float(vehicle_omegamax),
                    "R": float(unicycle.max_radius),

                    "N": N,
                    "M": M,
                    "analytical_initial_guess": analytical_initial_guess,
                    "analytical_solve_time": None,
                    "best_analytical_primitives": None,

                    "arc_segment_ratio_1": None,
                    "arc_segment_ratio_2": None,
                    "arc1_length": None,
                    "segment_length": None,
                    "arc2_length": None,

                    "ocp_success": False,
                    "error": str(e),
                })
   
                continue

            ocp_result = compute_ocp_pose_to_pose_trajectory(
                start_pose,
                end_pose,
                unicycle,
                analytical_initial_guess=best_trajectory,
                T_guess=best_time,
                N=N,
                M=M,
            )

            print_case_result(
                case_id,
                n_cases,
                theta0,
                thetaf,
                best_name,
                best_time,
                ocp_result,
            )

            case_result = {
                "case_id": case_id,
                "success": True,

                "x0": float(x0),
                "y0": float(y0),
                "xf": float(xf),
                "yf": float(yf),

                "theta0_rad": float(theta0),
                "thetaf_rad": float(thetaf),
                "theta0_deg": float(degrees(theta0)),
                "thetaf_deg": float(degrees(thetaf)),

                "v_max": float(vehicle_vmax),
                "omega_max": float(vehicle_omegamax),
                "R": float(unicycle.max_radius),

                "N": N,
                "M": M,
                "analytical_initial_guess": analytical_initial_guess,

                "best_analytical_name": best_name,
                "best_analytical_time": float(best_time),
                "analytical_solve_time": float(analytical_solve_time),

                "ocp_time": float(ocp_result["time"]),
                "ocp_solve_time": float(ocp_result["solve_time"]),
                "ocp_sequence": ocp_result["sequence"],

                "time_difference": float(ocp_result["time"] - best_time),
                "ocp_success": bool(ocp_result["success"]),
                "best_analytical_primitives": best_analytical_primitives,
                "arc_segment_ratio_1": ratios["r1"],
                "arc_segment_ratio_2": ratios["r2"],
                "arc1_length": ratios["arc1_length"],
                "segment_length": ratios["segment_length"],
                "arc2_length": ratios["arc2_length"],
            }

            results.append(case_result)

    total_simulation_time = time.perf_counter() - t0_simulation

    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)

    print(f"Total simulation time: {total_simulation_time:.3f} s")

    metadata = {
        "sweep_name": "orientation_sweep",
        "description": "Fixed start and final positions. Vary theta0 and thetaf.",
        "x0": x0,
        "y0": y0,
        "xf": xf,
        "yf": yf,
        "v_max": vehicle_vmax,
        "omega_max": vehicle_omegamax,
        "R": unicycle.max_radius,
        "n_angles": n_angles,
        "N": N,
        "analytical_initial_guess": analytical_initial_guess,
        "total_simulation_time": total_simulation_time,
    }

    RESULTS_FILENAME = "test_before_run.json"

    current_dir = Path(__file__).resolve().parent

    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / RESULTS_FILENAME

    output = {
        "metadata": metadata,
        "results": results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to: {save_path}")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)