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

    if len(arcs) != 2 or len(segments) != 1:
        return {
            "arc1_length": None,
            "segment_length": None,
            "arc2_length": None,
            "r1": None,
            "r2": None,
        }

    return {
        "arc1_length": float(arc1.path_length),
        "segment_length": float(segment.path_length),
        "arc2_length": float(arc2.path_length),
        "r1": float(r1) if r1 is not None else None,
        "r2": float(r2) if r2 is not None else None,
    }


def extract_analytical_primitive_info(trajectory):
    primitives_info = []

    for i, primitive in enumerate(trajectory, start=1):

        primitive_info = {
            "index": i,
            "label": primitive.label,
            "maneuver_time": float(primitive.maneuver_time),
            "path_length": float(primitive.path_length),
        }

        if hasattr(primitive, "turn_direction"):
            primitive_info["turn_direction"] = int(primitive.turn_direction)

        if hasattr(primitive, "iota"):
            primitive_info["angular_amplitude_rad"] = float(primitive.iota)
            primitive_info["angular_amplitude_deg"] = float(degrees(primitive.iota))

        elif hasattr(primitive, "delta_angle"):
            primitive_info["angular_amplitude_rad"] = float(primitive.delta_angle)
            primitive_info["angular_amplitude_deg"] = float(degrees(primitive.delta_angle))

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


def generate_sweep_cases(sweep_id):
    """
    Return cases and metadata for the selected sweep.

    Each case contains:
        x0, y0, theta0, xf, yf, thetaf, v_max, omega_max
    """

    cases = []

    if sweep_id == 1:
        # Sweep 1: fixed positions, fixed R, vary theta0/thetaf
        x0, y0 = 0.0, 0.0
        xf, yf = 0.0, 5.0

        v_max = 1.0
        omega_max = 1.0

        n_angles = 4
        start_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)
        final_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)

        for theta0 in start_angles:
            for thetaf in final_angles:
                cases.append({
                    "x0": x0,
                    "y0": y0,
                    "xf": xf,
                    "yf": yf,
                    "theta0": theta0,
                    "thetaf": thetaf,
                    "v_max": v_max,
                    "omega_max": omega_max,
                })

        metadata = {
            "sweep_id": 1,
            "sweep_name": "orientation_sweep",
            "description": "Fixed positions and R. Vary theta0 and thetaf.",
            "n_angles": n_angles,
            "D_over_R": 5.0,
        }

    elif sweep_id == 2:
        # Sweep 2: vary D/R, coarser angle grid
        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0
        R = v_max / omega_max

        d_over_r_values = [5.0, 10.0, 15.0, 20.0]

        n_angles = 30
        start_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)
        final_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)

        for d_over_r in d_over_r_values:
            xf = 0.0
            yf = d_over_r * R

            for theta0 in start_angles:
                for thetaf in final_angles:
                    cases.append({
                        "x0": x0,
                        "y0": y0,
                        "xf": xf,
                        "yf": yf,
                        "theta0": theta0,
                        "thetaf": thetaf,
                        "v_max": v_max,
                        "omega_max": omega_max,
                    })

        metadata = {
            "sweep_id": 2,
            "sweep_name": "distance_over_radius_sweep",
            "description": "Vary D/R and orientations.",
            "d_over_r_values": d_over_r_values,
            "n_angles": n_angles,
        }

    elif sweep_id == 3:
        # Sweep 3: vary final point using polar coordinates,
        # while enforcing D/R > 4.

        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0
        R = v_max / omega_max

        d_over_r_values = [5.0, 10.0, 15.0, 20.0]

        n_goal_angles = 16
        goal_angles = np.linspace(
            0.0,
            2 * pi,
            n_goal_angles,
            endpoint=False,
        )

        n_angles = 12
        start_angles = np.linspace(
            0.0,
            2 * pi,
            n_angles,
            endpoint=False,
        )
        final_angles = np.linspace(
            0.0,
            2 * pi,
            n_angles,
            endpoint=False,
        )

        for d_over_r in d_over_r_values:

            D = d_over_r * R

            for phi in goal_angles:

                xf = x0 + D * np.cos(phi)
                yf = y0 + D * np.sin(phi)

                for theta0 in start_angles:
                    for thetaf in final_angles:
                        cases.append({
                            "x0": x0,
                            "y0": y0,
                            "xf": float(xf),
                            "yf": float(yf),
                            "theta0": theta0,
                            "thetaf": thetaf,
                            "phi": float(phi),
                            "phi_deg": float(np.degrees(phi)),
                            "D": float(D),
                            "D_over_R": float(d_over_r),
                            "v_max": v_max,
                            "omega_max": omega_max,
                        })

        metadata = {
            "sweep_id": 3,
            "sweep_name": "polar_goal_position_sweep",
            "description": (
                "Vary final point using polar coordinates, together with "
                "initial and final orientations."
            ),
            "d_over_r_values": d_over_r_values,
            "n_goal_angles": n_goal_angles,
            "n_angles": n_angles,
            "R": R,
            "distance_assumption": "D/R > 4",
            "n_cases": len(cases),
        }

    elif sweep_id == 4:
        # Sweep 4: Sobol sampling
        from scipy.stats import qmc

        x0, y0 = 0.0, 0.0

        n_samples_power = 12
        n_samples = 2 ** n_samples_power

        sampler = qmc.Sobol(d=5, scramble=True, seed=1)
        samples = sampler.random_base2(m=n_samples_power)

        x_min, x_max = -15.0, 15.0
        y_min, y_max = -15.0, 15.0
        r_min, r_max = 0.5, 2.0

        v_max = 1.0

        for sample in samples:
            sx, sy, stheta0, sthetaf, sr = sample

            xf = x_min + sx * (x_max - x_min)
            yf = y_min + sy * (y_max - y_min)

            theta0 = 2 * pi * stheta0
            thetaf = 2 * pi * sthetaf

            R = r_min + sr * (r_max - r_min)
            omega_max = v_max / R

            D = np.hypot(xf - x0, yf - y0)

            if D <= 4.0 * R:
                continue

            cases.append({
                "x0": x0,
                "y0": y0,
                "xf": float(xf),
                "yf": float(yf),
                "theta0": theta0,
                "thetaf": thetaf,
                "v_max": v_max,
                "omega_max": omega_max,
            })

        metadata = {
            "sweep_id": 4,
            "sweep_name": "sobol_sweep",
            "description": "Sobol sampling over final point, theta0, thetaf, and R.",
            "n_samples_requested": n_samples,
            "n_samples_valid": len(cases),
            "xf_range": [x_min, x_max],
            "yf_range": [y_min, y_max],
            "R_range": [r_min, r_max],
            "distance_assumption": "D > 4R",
        }

    else:
        raise ValueError(f"Unknown sweep_id: {sweep_id}")

    return cases, metadata


if __name__ == "__main__":

    sweep_id = 1   # choose 1, 2, 3, or 4

    cases, metadata = generate_sweep_cases(sweep_id)

    n_cases = len(cases)
    case_id = 0

    N = 30
    M = 4
    analytical_initial_guess = True

    results = []

    print("\n" + "=" * 80)
    print("POSE-TO-POSE UNICYCLE SWEEP")
    print("=" * 80)

    t0_simulation = time.perf_counter()

    for case in cases:

        case_id += 1

        x0 = case["x0"]
        y0 = case["y0"]
        xf = case["xf"]
        yf = case["yf"]
        theta0 = case["theta0"]
        thetaf = case["thetaf"]

        vehicle_vmax = case["v_max"]
        vehicle_omegamax = case["omega_max"]

        unicycle = Unicycle(
            state=[0, 0, 0],
            width=0.430,
            length=0.430,
            v_max=vehicle_vmax,
            v_min=0,
            omega_max=vehicle_omegamax,
            omega_min=-vehicle_omegamax,
        )

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
   
    metadata.update({
        "N": N,
        "M": M,
        "analytical_initial_guess": analytical_initial_guess,
        "total_simulation_time": total_simulation_time,
    })

    RESULTS_FILENAME = f"{metadata['sweep_name']}_test.json"

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