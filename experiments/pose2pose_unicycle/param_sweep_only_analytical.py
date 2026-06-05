from math import pi, degrees
import json
import time
from pathlib import Path

import numpy as np

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)


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

        n_angles = 360
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
        # Sweep 3: vary final point, enforce D > 4R
        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0
        R = v_max / omega_max

        n_positions = 15
        x_values = np.linspace(-10.0, 10.0, n_positions)
        y_values = np.linspace(-10.0, 10.0, n_positions)

        n_angles = 16
        start_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)
        final_angles = np.linspace(0.0, 2 * pi, n_angles, endpoint=False)

        for xf in x_values:
            for yf in y_values:
                D = np.hypot(xf - x0, yf - y0)

                if D <= 4.0 * R:
                    continue

                for theta0 in start_angles:
                    for thetaf in final_angles:
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
            "sweep_id": 3,
            "sweep_name": "final_position_sweep",
            "description": "Vary final point and orientations, enforcing D > 4R.",
            "n_positions_per_axis": n_positions,
            "n_angles": n_angles,
            "distance_assumption": "D > 4R",
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


def trajectory_to_jsonable(best_trajectory):
    """
    Convert the best analytical trajectory to something JSON serializable.

    This intentionally stores only the best trajectory, not the info of every
    candidate primitive/trajectory. Each primitive is represented with the
    fields most likely to be useful later.
    """
    trajectory_data = []

    for primitive in best_trajectory:
        primitive_data = {
            "label": primitive.label,
            "maneuver_time": float(primitive.maneuver_time),
            "path_length": float(primitive.path_length),
        }

        if hasattr(primitive, "turn_direction"):
            primitive_data["turn_direction"] = int(primitive.turn_direction)

        if hasattr(primitive, "iota"):
            primitive_data["angular_amplitude_rad"] = float(primitive.iota)
            primitive_data["angular_amplitude_deg"] = float(degrees(primitive.iota))
        elif hasattr(primitive, "delta_angle"):
            primitive_data["angular_amplitude_rad"] = float(primitive.delta_angle)
            primitive_data["angular_amplitude_deg"] = float(degrees(primitive.delta_angle))

        if hasattr(primitive, "start_pose"):
            primitive_data["start_pose"] = [
                float(value) for value in primitive.start_pose
            ]

        if hasattr(primitive, "end_pose"):
            primitive_data["end_pose"] = [
                float(value) for value in primitive.end_pose
            ]

        trajectory_data.append(primitive_data)

    return trajectory_data


if __name__ == "__main__":

    sweep_id = 1   # choose 1, 2, 3, or 4

    cases, metadata = generate_sweep_cases(sweep_id)

    n_cases = len(cases)
    results = []

    print("\n" + "=" * 80)
    print("POSE-TO-POSE UNICYCLE ANALYTICAL SWEEP")
    print("=" * 80)

    t0_simulation = time.perf_counter()

    for case_id, case in enumerate(cases, start=1):

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
            t0_case = time.perf_counter()

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
            analytical_solve_time = time.perf_counter() - t0_case

            print(
                f"\nCase {case_id:02d}/{n_cases}: "
                f"theta0={degrees(theta0):6.1f} deg, "
                f"thetaf={degrees(thetaf):6.1f} deg"
            )
            print(f"  Best analytical : {best_name}")
            print(f"  Analytical time : {best_time:.6f} s")
            print(f"  Solve time      : {analytical_solve_time:.6f} s")

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

                "best_analytical_name": best_name,
                "best_analytical_time": float(best_time),
                "analytical_solve_time": float(analytical_solve_time),
                "best_trajectory": trajectory_to_jsonable(best_trajectory),
            }

        except ValueError as e:
            print(
                f"\nCase {case_id:02d}/{n_cases}: "
                f"theta0={degrees(theta0):6.1f} deg, "
                f"thetaf={degrees(thetaf):6.1f} deg"
            )
            print(f"  Skipped: {e}")

            case_result = {
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

                "best_analytical_name": None,
                "best_analytical_time": None,
                "analytical_solve_time": None,
                "best_trajectory": None,
                "error": str(e),
            }

        results.append(case_result)

    total_simulation_time = time.perf_counter() - t0_simulation

    print("\n" + "=" * 80)
    print("ANALYTICAL SWEEP COMPLETED")
    print("=" * 80)
    print(f"Total simulation time: {total_simulation_time:.3f} s")

    metadata.update({
        "total_simulation_time": float(total_simulation_time),
    })

    results_filename = f"{metadata['sweep_name']}_analytical.json"

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / results_filename

    output = {
        "metadata": metadata,
        "results": results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to: {save_path}")