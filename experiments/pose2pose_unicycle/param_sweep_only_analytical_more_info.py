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
            "sweep_name": "orientation_sweep_all_candidates",
            "description": (
                "Fixed positions and R. Vary theta0 and thetaf. "
                "Store compact information for all analytical candidates."
            ),
            "n_angles": n_angles,
            "D_over_R": 5.0,
        }

    elif sweep_id == 2:
        # Sweep 2: vary D/R, coarser angle grid
        x0, y0 = 0.0, 0.0

        v_max = 1.0
        omega_max = 1.0

        R = v_max / omega_max

        d_over_r_values = [
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        n_angles = 30

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
            "sweep_name": "distance_over_radius_sweep_all_candidates",
            "description": (
                "Vary D/R and orientations. "
                "Store compact information for all analytical candidates."
            ),
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

        x_values = np.linspace(
            -10.0,
            10.0,
            n_positions,
        )

        y_values = np.linspace(
            -10.0,
            10.0,
            n_positions,
        )

        n_angles = 16

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

        for xf in x_values:
            for yf in y_values:

                D = np.hypot(
                    xf - x0,
                    yf - y0,
                )

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
            "sweep_name": "final_position_sweep_all_candidates",
            "description": (
                "Vary final point and orientations, enforcing D > 4R. "
                "Store compact information for all analytical candidates."
            ),
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

        sampler = qmc.Sobol(
            d=5,
            scramble=True,
            seed=1,
        )

        samples = sampler.random_base2(
            m=n_samples_power,
        )

        x_min, x_max = -15.0, 15.0
        y_min, y_max = -15.0, 15.0
        r_min, r_max = 0.5, 2.0

        v_max = 1.0

        for sample in samples:

            sx, sy, stheta0, sthetaf, sr = sample

            xf = (
                x_min
                + sx * (x_max - x_min)
            )

            yf = (
                y_min
                + sy * (y_max - y_min)
            )

            theta0 = (
                2 * pi * stheta0
            )

            thetaf = (
                2 * pi * sthetaf
            )

            R = (
                r_min
                + sr * (r_max - r_min)
            )

            omega_max = (
                v_max / R
            )

            D = np.hypot(
                xf - x0,
                yf - y0,
            )

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
            "sweep_name": "sobol_sweep_all_candidates",
            "description": (
                "Sobol sampling over final point, theta0, thetaf, and R. "
                "Store compact information for all analytical candidates."
            ),
            "n_samples_requested": n_samples,
            "n_samples_valid": len(cases),
            "xf_range": [
                x_min,
                x_max,
            ],
            "yf_range": [
                y_min,
                y_max,
            ],
            "R_range": [
                r_min,
                r_max,
            ],
            "distance_assumption": "D > 4R",
        }

    else:
        raise ValueError(
            f"Unknown sweep_id: {sweep_id}"
        )

    return cases, metadata


def compute_trajectory_quantities(trajectory):
    """
    Compute the quantities needed for the family-comparison analysis.

    Gamma:
        Sum of the absolute angular amplitudes of all rotational primitives.

    d:
        Sum of the path lengths of the straight primitives.

    For the CSC/TCSC/CSCT/TCSCT families considered here, each trajectory
    contains one straight segment.
    """

    gamma = 0.0
    d = 0.0

    for primitive in trajectory:

        if hasattr(primitive, "iota"):

            gamma += abs(
                float(primitive.iota)
            )

        elif hasattr(
            primitive,
            "delta_angle",
        ):

            gamma += abs(
                float(primitive.delta_angle)
            )

        else:

            d += float(
                primitive.path_length
            )

    return d, gamma


def parse_candidate_name(candidate_name):
    """
    Parse a candidate name such as:

        'TCSC left-right'

    Returns:
        family, tau0, tauf

    Convention:
        tau = +1 -> left
        tau = -1 -> right
    """

    family, directions = (
        candidate_name.split(
            " ",
            1,
        )
    )

    d0, df = directions.split("-")

    direction_to_tau = {
        "left": +1,
        "right": -1,
    }

    tau0 = direction_to_tau[d0]
    tauf = direction_to_tau[df]

    return (
        family,
        tau0,
        tauf,
    )


def candidate_to_jsonable(
    candidate_name,
    candidate_data,
):
    """
    Store only the compact information needed for comparison of candidates.

    The complete primitive trajectory is deliberately not saved here.
    """

    trajectory = candidate_data[
        "trajectory"
    ]

    candidate_time = float(
        candidate_data["time"]
    )

    d, gamma = (
        compute_trajectory_quantities(
            trajectory
        )
    )

    family, tau0, tauf = (
        parse_candidate_name(
            candidate_name
        )
    )

    return {
        "family": family,
        "tau0": tau0,
        "tauf": tauf,
        "time": candidate_time,
        "d": float(d),
        "Gamma_rad": float(gamma),
    }


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Select sweep
    # -------------------------------------------------------------------------

    sweep_id = 1

    cases, metadata = (
        generate_sweep_cases(
            sweep_id
        )
    )

    n_cases = len(cases)

    results = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "POSE-TO-POSE UNICYCLE "
        "ALL-CANDIDATE SWEEP"
    )

    print(
        "=" * 80
    )

    print(
        f"Number of cases: {n_cases}"
    )

    t0_simulation = (
        time.perf_counter()
    )


    # =========================================================================
    # Main sweep
    # =========================================================================

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

        vehicle_vmax = (
            case["v_max"]
        )

        vehicle_omegamax = (
            case["omega_max"]
        )

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
            Point(
                x0,
                y0,
            ),
            theta0,
        )

        end_pose = Pose(
            Point(
                xf,
                yf,
            ),
            thetaf,
        )

        try:

            t0_case = (
                time.perf_counter()
            )

            trajectories = (
                compute_all_pose_to_pose_trajectories(
                    start_pose,
                    end_pose,
                    unicycle,
                )
            )

            analytical_solve_time = (
                time.perf_counter()
                - t0_case
            )

            # -----------------------------------------------------------------
            # Compact representation of every candidate
            # -----------------------------------------------------------------

            all_candidates = {}

            for (
                candidate_name,
                candidate_data,
            ) in trajectories.items():

                all_candidates[
                    candidate_name
                ] = candidate_to_jsonable(
                    candidate_name,
                    candidate_data,
                )

            # -----------------------------------------------------------------
            # Global winner
            # -----------------------------------------------------------------

            best_name, best_data = min(
                trajectories.items(),
                key=lambda item: item[1][
                    "time"
                ],
            )

            best_candidate = (
                all_candidates[
                    best_name
                ]
            )

            # -----------------------------------------------------------------
            # Console output
            # -----------------------------------------------------------------

            # Avoid printing 129600 large blocks.
            # Print the first few cases, then every 1000th case,
            # and finally the last case.
            print_case = (
                case_id <= 5
                or case_id % 1000 == 0
                or case_id == n_cases
            )

            if print_case:

                print(
                    f"\nCase "
                    f"{case_id:06d}/{n_cases}: "
                    f"theta0="
                    f"{degrees(theta0):6.1f} deg, "
                    f"thetaf="
                    f"{degrees(thetaf):6.1f} deg"
                )

                print(
                    f"  Best analytical : "
                    f"{best_name}"
                )

                print(
                    f"  Analytical time : "
                    f"{best_candidate['time']:.6f} s"
                )

                print(
                    f"  d               : "
                    f"{best_candidate['d']:.6f}"
                )

                print(
                    f"  Gamma           : "
                    f"{degrees(best_candidate['Gamma_rad']):.3f} deg"
                )

                print(
                    f"  Candidates      : "
                    f"{len(all_candidates)}"
                )

                print(
                    f"  Solve time      : "
                    f"{analytical_solve_time:.6f} s"
                )

            # -----------------------------------------------------------------
            # Save compact case information
            # -----------------------------------------------------------------

            case_result = {
                "case_id": case_id,
                "success": True,

                "x0": float(x0),
                "y0": float(y0),

                "xf": float(xf),
                "yf": float(yf),

                "theta0_rad": float(
                    theta0
                ),

                "thetaf_rad": float(
                    thetaf
                ),

                "theta0_deg": float(
                    degrees(theta0)
                ),

                "thetaf_deg": float(
                    degrees(thetaf)
                ),

                "v_max": float(
                    vehicle_vmax
                ),

                "omega_max": float(
                    vehicle_omegamax
                ),

                "R": float(
                    unicycle.max_radius
                ),

                "analytical_solve_time": float(
                    analytical_solve_time
                ),

                "best_analytical_name": (
                    best_name
                ),

                "best_analytical_time": float(
                    best_data["time"]
                ),

                "all_candidates": (
                    all_candidates
                ),
            }

        except ValueError as e:

            print(
                f"\nCase "
                f"{case_id:06d}/{n_cases}: "
                f"theta0="
                f"{degrees(theta0):6.1f} deg, "
                f"thetaf="
                f"{degrees(thetaf):6.1f} deg"
            )

            print(
                f"  Skipped: {e}"
            )

            case_result = {
                "case_id": case_id,
                "success": False,

                "x0": float(x0),
                "y0": float(y0),

                "xf": float(xf),
                "yf": float(yf),

                "theta0_rad": float(
                    theta0
                ),

                "thetaf_rad": float(
                    thetaf
                ),

                "theta0_deg": float(
                    degrees(theta0)
                ),

                "thetaf_deg": float(
                    degrees(thetaf)
                ),

                "v_max": float(
                    vehicle_vmax
                ),

                "omega_max": float(
                    vehicle_omegamax
                ),

                "R": float(
                    unicycle.max_radius
                ),

                "analytical_solve_time": None,

                "best_analytical_name": None,
                "best_analytical_time": None,

                "all_candidates": None,

                "error": str(e),
            }

        results.append(
            case_result
        )


    # =========================================================================
    # Save results
    # =========================================================================

    total_simulation_time = (
        time.perf_counter()
        - t0_simulation
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ALL-CANDIDATE SWEEP "
        "COMPUTATION COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"Total simulation time: "
        f"{total_simulation_time:.3f} s"
    )

    metadata.update({
        "total_simulation_time": float(
            total_simulation_time
        ),
        "stored_candidate_fields": [
            "family",
            "tau0",
            "tauf",
            "time",
            "d",
            "Gamma_rad",
        ],
    })

    current_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    results_dir = (
        current_dir
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    results_filename = (
        f"{metadata['sweep_name']}.json"
    )

    save_path = (
        results_dir
        / results_filename
    )

    output = {
        "metadata": metadata,
        "results": results,
    }

    print(
        "\nWriting compact JSON..."
    )

    with open(
        save_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            separators=(",", ":"),
        )

    print(
        "JSON write completed."
    )

    print(
        f"Results saved to: "
        f"{save_path}"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ALL-CANDIDATE SWEEP COMPLETED"
    )

    print(
        "=" * 80
    )