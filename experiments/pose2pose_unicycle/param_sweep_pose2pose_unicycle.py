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
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SWEEP_ID = 1

# OCP transcription settings
N = 100
M = 4

# Use the analytical trajectory to initialize the OCP
ANALYTICAL_INITIAL_GUESS = True

# Sweep 1 angular resolution
SWEEP1_N_ANGLES = 90

# Sobol sampling power for Sweep 4:
# number of requested samples = 2**SOBOL_POWER
SOBOL_POWER = 13

# Print the keys returned by the OCP solver for the first successful case.
# This is useful for verifying whether an exact time grid is returned.
PRINT_OCP_KEYS_ON_FIRST_CASE = True


# =============================================================================
# PRINTING AND DIAGNOSTIC FUNCTIONS
# =============================================================================

def print_case_result(
    case_id,
    n_cases,
    theta0,
    thetaf,
    best_name,
    best_time,
    ocp_result,
    best_trajectory,
    x0,
    y0,
    xf,
    yf,
):
    """
    Print the main results for one case.

    Suspicious cases with a large time discrepancy are plotted.
    """
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

        # OCP initial guess, when available
        initial_guess = ocp_result.get("initial_guess")

        if initial_guess is not None:
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


# =============================================================================
# ANALYTICAL TRAJECTORY INFORMATION
# =============================================================================

def compute_arc_segment_ratios(trajectory):
    """
    Compute the ratios between the straight-segment length and the lengths
    of the first and second arcs.

    The ratios are returned only when the trajectory contains exactly
    two arcs and one straight segment.
    """
    arcs = [
        primitive
        for primitive in trajectory
        if primitive.label.lower() == "arc"
    ]

    segments = [
        primitive
        for primitive in trajectory
        if primitive.label.lower() == "segment"
    ]

    empty_result = {
        "arc1_length": None,
        "segment_length": None,
        "arc2_length": None,
        "r1": None,
        "r2": None,
    }

    if len(arcs) != 2 or len(segments) != 1:
        return empty_result

    arc1 = arcs[0]
    arc2 = arcs[1]
    segment = segments[0]

    r1 = (
        segment.path_length / arc1.path_length
        if arc1.path_length > 1.0e-12
        else None
    )

    r2 = (
        segment.path_length / arc2.path_length
        if arc2.path_length > 1.0e-12
        else None
    )

    return {
        "arc1_length": float(arc1.path_length),
        "segment_length": float(segment.path_length),
        "arc2_length": float(arc2.path_length),
        "r1": float(r1) if r1 is not None else None,
        "r2": float(r2) if r2 is not None else None,
    }


def extract_analytical_primitive_info(trajectory):
    """
    Extract the information required to reconstruct the analytical trajectory
    after loading the saved JSON file.
    """
    primitives_info = []

    for index, primitive in enumerate(trajectory, start=1):
        primitive_info = {
            "index": index,
            "label": primitive.label,
            "maneuver_time": float(primitive.maneuver_time),
            "path_length": float(primitive.path_length),
        }

        if hasattr(primitive, "turn_direction"):
            primitive_info["turn_direction"] = int(
                primitive.turn_direction
            )

        if hasattr(primitive, "iota"):
            primitive_info["angular_amplitude_rad"] = float(
                primitive.iota
            )
            primitive_info["angular_amplitude_deg"] = float(
                degrees(primitive.iota)
            )

        elif hasattr(primitive, "delta_angle"):
            primitive_info["angular_amplitude_rad"] = float(
                primitive.delta_angle
            )
            primitive_info["angular_amplitude_deg"] = float(
                degrees(primitive.delta_angle)
            )

        if hasattr(primitive, "start_pose"):
            primitive_info["start_pose"] = [
                float(value)
                for value in primitive.start_pose
            ]

        if hasattr(primitive, "end_pose"):
            primitive_info["end_pose"] = [
                float(value)
                for value in primitive.end_pose
            ]

        primitives_info.append(primitive_info)

    return primitives_info


# =============================================================================
# OCP TRAJECTORY EXTRACTION
# =============================================================================

def extract_ocp_pose_trajectory(ocp_result):
    """
    Extract the OCP time grid and state trajectory.

    The function first looks for an exact time grid returned by the OCP
    solver. If no time grid is present, it assumes that the saved states are
    uniformly distributed over the optimized traversal time.

    Returns
    -------
    ocp_time_grid : numpy.ndarray
        Time samples associated with the OCP states.

    ocp_x : numpy.ndarray
        OCP x-coordinate samples.

    ocp_y : numpy.ndarray
        OCP y-coordinate samples.

    ocp_theta : numpy.ndarray
        OCP heading samples.
    """
    required_state_keys = ("xs", "ys", "thetas")

    missing_keys = [
        key
        for key in required_state_keys
        if key not in ocp_result
    ]

    if missing_keys:
        raise KeyError(
            "The OCP result does not contain the required trajectory "
            f"entries: {missing_keys}. "
            f"Available keys are: {sorted(ocp_result.keys())}"
        )

    ocp_x = np.asarray(
        ocp_result["xs"],
        dtype=float,
    ).reshape(-1)

    ocp_y = np.asarray(
        ocp_result["ys"],
        dtype=float,
    ).reshape(-1)

    ocp_theta = np.asarray(
        ocp_result["thetas"],
        dtype=float,
    ).reshape(-1)

    n_samples = len(ocp_x)

    if n_samples < 2:
        raise ValueError(
            "The OCP trajectory must contain at least two state samples."
        )

    if len(ocp_y) != n_samples or len(ocp_theta) != n_samples:
        raise ValueError(
            "The OCP state arrays xs, ys, and thetas must have "
            "the same length."
        )

    # Search for a time grid returned directly by the solver.
    possible_time_keys = (
        "times",
        "time_grid",
        "ts",
        "t",
    )

    ocp_time_grid = None

    for key in possible_time_keys:
        if key in ocp_result:
            candidate_grid = np.asarray(
                ocp_result[key],
                dtype=float,
            ).reshape(-1)

            if len(candidate_grid) == n_samples:
                ocp_time_grid = candidate_grid
                break

    # If no exact grid is returned, assume uniformly spaced shooting nodes.
    if ocp_time_grid is None:
        ocp_time_grid = np.linspace(
            0.0,
            float(ocp_result["time"]),
            n_samples,
            endpoint=True,
        )

    if not np.all(np.isfinite(ocp_time_grid)):
        raise ValueError(
            "The OCP time grid contains non-finite values."
        )

    if not np.all(np.diff(ocp_time_grid) >= 0.0):
        raise ValueError(
            "The OCP time grid must be monotonically nondecreasing."
        )

    return (
        ocp_time_grid,
        ocp_x,
        ocp_y,
        ocp_theta,
    )


# =============================================================================
# PARAMETER SWEEPS
# =============================================================================

def generate_sweep_cases(sweep_id):
    """
    Return the cases and metadata for the selected sweep.

    Each case contains:
        x0, y0, theta0,
        xf, yf, thetaf,
        v_max, omega_max.
    """
    cases = []

    if sweep_id == 1:
        # Sweep 1:
        # Fixed positions and vehicle parameters.
        # Vary theta0 and thetaf.

        x0, y0 = 0.0, 0.0
        xf, yf = 0.0, 5.0

        v_max = 1.0
        omega_max = 1.0

        n_angles = SWEEP1_N_ANGLES

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

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
            "sweep_name": "orientation_sweep_OCP_path_saved",
            "description": (
                "Fixed positions and R. "
                "Vary theta0 and thetaf."
            ),
            "n_angles": n_angles,
            "D_over_R": 5.0,
            "n_cases": len(cases),
        }

    elif sweep_id == 2:
        # Sweep 2:
        # Vary D/R and the boundary orientations.

        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0
        radius = v_max / omega_max

        d_over_r_values = [
            10.0,
            15.0,
            20.0,
        ]

        n_angles = 50

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        for d_over_r in d_over_r_values:
            xf = 0.0
            yf = d_over_r * radius

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
                        "D_over_R": d_over_r,
                    })

        metadata = {
            "sweep_id": 2,
            "sweep_name": "distance_over_radius_sweep",
            "description": (
                "Vary D/R and the boundary orientations."
            ),
            "d_over_r_values": d_over_r_values,
            "n_angles": n_angles,
            "n_cases": len(cases),
        }

    elif sweep_id == 3:
        # Sweep 3:
        # Vary the final point using polar coordinates,
        # together with theta0 and thetaf.

        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0
        radius = v_max / omega_max

        d_over_r_values = [
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        n_goal_angles = 16

        goal_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_goal_angles,
            endpoint=False,
        )

        n_angles = 12

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        for d_over_r in d_over_r_values:
            distance = d_over_r * radius

            for phi in goal_angles:
                xf = x0 + distance * np.cos(phi)
                yf = y0 + distance * np.sin(phi)

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
                            "phi_deg": float(degrees(phi)),
                            "D": float(distance),
                            "D_over_R": float(d_over_r),
                            "v_max": v_max,
                            "omega_max": omega_max,
                        })

        metadata = {
            "sweep_id": 3,
            "sweep_name": "polar_goal_position_sweep",
            "description": (
                "Vary the final point using polar coordinates, "
                "together with the initial and final orientations."
            ),
            "d_over_r_values": d_over_r_values,
            "n_goal_angles": n_goal_angles,
            "n_angles": n_angles,
            "R": radius,
            "distance_assumption": "D/R > 4",
            "n_cases": len(cases),
        }

    elif sweep_id == 4:
        # Sweep 4:
        # Sobol sampling over xf, yf, theta0, thetaf, and R.

        from scipy.stats import qmc

        x0, y0 = 0.0, 0.0

        n_samples = 2 ** SOBOL_POWER

        sampler = qmc.Sobol(
            d=5,
            scramble=True,
            seed=1,
        )

        samples = sampler.random_base2(
            m=SOBOL_POWER
        )

        x_min, x_max = -15.0, 15.0
        y_min, y_max = -15.0, 15.0
        r_min, r_max = 0.5, 2.0

        v_max = 1.0

        for sample in samples:
            sx, sy, stheta0, sthetaf, sr = sample

            xf = x_min + sx * (x_max - x_min)
            yf = y_min + sy * (y_max - y_min)

            theta0 = 2.0 * pi * stheta0
            thetaf = 2.0 * pi * sthetaf

            radius = r_min + sr * (r_max - r_min)
            omega_max = v_max / radius

            distance = np.hypot(
                xf - x0,
                yf - y0,
            )

            if distance <= 4.0 * radius:
                continue

            cases.append({
                "x0": x0,
                "y0": y0,
                "xf": float(xf),
                "yf": float(yf),
                "theta0": float(theta0),
                "thetaf": float(thetaf),
                "v_max": v_max,
                "omega_max": float(omega_max),
                "D": float(distance),
                "D_over_R": float(distance / radius),
            })

        metadata = {
            "sweep_id": 4,
            "sweep_name": "sobol_sweep",
            "description": (
                "Sobol sampling over the final position, theta0, "
                "thetaf, and R."
            ),
            "sobol_power": SOBOL_POWER,
            "sobol_seed": 1,
            "n_samples_requested": n_samples,
            "n_samples_valid": len(cases),
            "xf_range": [x_min, x_max],
            "yf_range": [y_min, y_max],
            "R_range": [r_min, r_max],
            "distance_assumption": "D > 4R",
        }

    else:
        raise ValueError(
            f"Unknown sweep_id: {sweep_id}"
        )

    return cases, metadata


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cases, metadata = generate_sweep_cases(
        SWEEP_ID
    )

    n_cases = len(cases)
    results = []

    first_ocp_case = True

    print("\n" + "=" * 80)
    print("POSE-TO-POSE UNICYCLE SWEEP")
    print("=" * 80)

    print(f"Sweep ID                 : {SWEEP_ID}")
    print(f"Number of cases          : {n_cases}")
    print(f"OCP control intervals N  : {N}")
    print(f"RK steps per interval M  : {M}")
    print(
        "Analytical initial guess: "
        f"{ANALYTICAL_INITIAL_GUESS}"
    )

    simulation_start_time = time.perf_counter()

    for case_id, case in enumerate(
        cases,
        start=1,
    ):
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

        start_pose = Pose(
            Point(x0, y0),
            theta0,
        )

        end_pose = Pose(
            Point(xf, yf),
            thetaf,
        )

        # ---------------------------------------------------------------------
        # Analytical planner
        # ---------------------------------------------------------------------

        try:
            analytical_start_time = time.perf_counter()

            trajectories = (
                compute_all_pose_to_pose_trajectories(
                    start_pose,
                    end_pose,
                    unicycle,
                )
            )

            best_name, best_data = min(
                trajectories.items(),
                key=lambda item: item[1]["time"],
            )

            best_trajectory = best_data["trajectory"]
            best_time = best_data["time"]

            analytical_solve_time = (
                time.perf_counter()
                - analytical_start_time
            )

            best_analytical_primitives = (
                extract_analytical_primitive_info(
                    best_trajectory
                )
            )

            ratios = compute_arc_segment_ratios(
                best_trajectory
            )

        except (ValueError, RuntimeError) as error:
            print(
                f"\nCase {case_id:02d}/{n_cases}: "
                f"theta0={degrees(theta0):6.1f} deg, "
                f"thetaf={degrees(thetaf):6.1f} deg"
            )

            print(f"  Analytical planner failed: {error}")

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
                "analytical_initial_guess":
                    ANALYTICAL_INITIAL_GUESS,

                "analytical_solve_time": None,
                "best_analytical_primitives": None,

                "arc_segment_ratio_1": None,
                "arc_segment_ratio_2": None,
                "arc1_length": None,
                "segment_length": None,
                "arc2_length": None,

                "ocp_success": False,

                "ocp_time_grid": None,
                "ocp_x": None,
                "ocp_y": None,
                "ocp_theta": None,

                "error": str(error),
            })

            continue

        # ---------------------------------------------------------------------
        # Optimal-control planner
        # ---------------------------------------------------------------------

        try:
            ocp_result = (
                compute_ocp_pose_to_pose_trajectory(
                    start_pose,
                    end_pose,
                    unicycle,
                    analytical_initial_guess=(
                        best_trajectory
                        if ANALYTICAL_INITIAL_GUESS
                        else None
                    ),
                    T_guess=best_time,
                    N=N,
                    M=M,
                )
            )

            if (
                PRINT_OCP_KEYS_ON_FIRST_CASE
                and first_ocp_case
            ):
                print("\n" + "-" * 80)
                print("OCP RESULT KEYS")
                print("-" * 80)

                for key in sorted(ocp_result.keys()):
                    print(key)

                print("-" * 80)

            first_ocp_case = False

            (
                ocp_time_grid,
                ocp_x,
                ocp_y,
                ocp_theta,
            ) = extract_ocp_pose_trajectory(
                ocp_result
            )

        except (
            ValueError,
            RuntimeError,
            KeyError,
        ) as error:
            print(
                f"\nCase {case_id:02d}/{n_cases}: "
                f"theta0={degrees(theta0):6.1f} deg, "
                f"thetaf={degrees(thetaf):6.1f} deg"
            )

            print(f"  OCP solve or extraction failed: {error}")

            results.append({
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
                "analytical_initial_guess":
                    ANALYTICAL_INITIAL_GUESS,

                "best_analytical_name": best_name,
                "best_analytical_time": float(best_time),
                "analytical_solve_time": float(
                    analytical_solve_time
                ),

                "best_analytical_primitives":
                    best_analytical_primitives,

                "arc_segment_ratio_1": ratios["r1"],
                "arc_segment_ratio_2": ratios["r2"],
                "arc1_length": ratios["arc1_length"],
                "segment_length": ratios["segment_length"],
                "arc2_length": ratios["arc2_length"],

                "ocp_success": False,
                "ocp_time": None,
                "ocp_solve_time": None,
                "ocp_sequence": None,
                "time_difference": None,

                "ocp_time_grid": None,
                "ocp_x": None,
                "ocp_y": None,
                "ocp_theta": None,

                "error": str(error),
            })

            continue

        # ---------------------------------------------------------------------
        # Print and save successful case
        # ---------------------------------------------------------------------

        print_case_result(
            case_id=case_id,
            n_cases=n_cases,
            theta0=theta0,
            thetaf=thetaf,
            best_name=best_name,
            best_time=best_time,
            ocp_result=ocp_result,
            best_trajectory=best_trajectory,
            x0=x0,
            y0=y0,
            xf=xf,
            yf=yf,
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
            "analytical_initial_guess":
                ANALYTICAL_INITIAL_GUESS,

            "best_analytical_name": best_name,
            "best_analytical_time": float(best_time),
            "analytical_solve_time": float(
                analytical_solve_time
            ),

            "best_analytical_primitives":
                best_analytical_primitives,

            "ocp_time": float(ocp_result["time"]),
            "ocp_solve_time": float(
                ocp_result["solve_time"]
            ),
            "ocp_sequence": list(
                ocp_result["sequence"]
            ),

            "time_difference": float(
                ocp_result["time"] - best_time
            ),
            "ocp_success": bool(
                ocp_result["success"]
            ),

            # OCP state trajectory saved for later comparison
            "ocp_time_grid": [
                float(value)
                for value in ocp_time_grid
            ],
            "ocp_x": [
                float(value)
                for value in ocp_x
            ],
            "ocp_y": [
                float(value)
                for value in ocp_y
            ],
            "ocp_theta": [
                float(value)
                for value in ocp_theta
            ],

            "arc_segment_ratio_1": ratios["r1"],
            "arc_segment_ratio_2": ratios["r2"],
            "arc1_length": ratios["arc1_length"],
            "segment_length": ratios["segment_length"],
            "arc2_length": ratios["arc2_length"],
        }

        results.append(case_result)

    # =========================================================================
    # SAVE COMPLETE SWEEP
    # =========================================================================

    total_simulation_time = (
        time.perf_counter()
        - simulation_start_time
    )

    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)

    print(
        "Total simulation time: "
        f"{total_simulation_time:.3f} s"
    )

    metadata.update({
        "N": N,
        "M": M,
        "analytical_initial_guess":
            ANALYTICAL_INITIAL_GUESS,
        "ocp_pose_trajectory_saved": True,
        "saved_ocp_fields": [
            "ocp_time_grid",
            "ocp_x",
            "ocp_y",
            "ocp_theta",
        ],
        "total_simulation_time":
            float(total_simulation_time),
    })

    results_filename = (
        f"{metadata['sweep_name']}"
        f"_N{N}_M{M}.json"
    )

    current_dir = Path(__file__).resolve().parent

    results_dir = current_dir / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = results_dir / results_filename

    output = {
        "metadata": metadata,
        "results": results,
    }

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    print(f"Results saved to: {save_path}")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETED")
    print("=" * 80)