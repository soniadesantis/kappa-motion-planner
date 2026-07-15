# results_filename = "orientation_sweep_N100_OCPpath.json"

# current_dir = Path(__file__).resolve().parent

# results_dir = current_dir / "results"

# results_path = results_dir / results_filename

import json
from pathlib import Path

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = Path(
    "experiments/pose2pose_unicycle/results/"
    "orientation_sweep_N100_OCPpath.json"
)

N_COMPARISON_POINTS = 1001

# Print detailed information for the worst cases
N_WORST_CASES_TO_PRINT = 10


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wrap_to_pi(angle):
    """
    Wrap an angle or array of angles to [-pi, pi).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def interpolate_pose(
    times,
    xs,
    ys,
    thetas,
    normalized_grid,
):
    """
    Interpolate a pose trajectory on a common normalized-time grid.
    """
    times = np.asarray(times, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    if not (
        len(times)
        == len(xs)
        == len(ys)
        == len(thetas)
    ):
        raise ValueError(
            "Time and state arrays must have equal lengths."
        )

    if len(times) < 2:
        raise ValueError(
            "At least two trajectory samples are required."
        )

    duration = times[-1] - times[0]

    if duration <= 0.0:
        raise ValueError(
            "Trajectory duration must be positive."
        )

    tau = (times - times[0]) / duration

    # Remove repeated normalized-time samples.
    tau_unique, unique_indices = np.unique(
        tau,
        return_index=True,
    )

    xs = xs[unique_indices]
    ys = ys[unique_indices]
    thetas = thetas[unique_indices]

    # Remove artificial jumps at +/- pi before interpolation.
    thetas_unwrapped = np.unwrap(thetas)

    x_interp = np.interp(
        normalized_grid,
        tau_unique,
        xs,
    )

    y_interp = np.interp(
        normalized_grid,
        tau_unique,
        ys,
    )

    theta_interp = np.interp(
        normalized_grid,
        tau_unique,
        thetas_unwrapped,
    )

    return x_interp, y_interp, theta_interp


# =============================================================================
# ANALYTICAL TRAJECTORY RECONSTRUCTION
# =============================================================================

def exact_unicycle_step(
    x,
    y,
    theta,
    v,
    omega,
    dt,
):
    """
    Exact state update for constant unicycle controls.
    """
    tolerance = 1.0e-12

    if abs(omega) < tolerance:
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta

    elif abs(v) < tolerance:
        x_new = x
        y_new = y
        theta_new = theta + omega * dt

    else:
        theta_new = theta + omega * dt

        x_new = (
            x
            + (v / omega)
            * (
                np.sin(theta_new)
                - np.sin(theta)
            )
        )

        y_new = (
            y
            - (v / omega)
            * (
                np.cos(theta_new)
                - np.cos(theta)
            )
        )

    return x_new, y_new, theta_new


def primitive_controls(
    primitive,
    v_max,
    omega_max,
):
    """
    Return the constant controls associated with a saved primitive.
    """
    label = primitive["label"].strip().lower()

    if label == "segment":
        return v_max, 0.0

    if label == "arc":
        direction = int(primitive["turn_direction"])

        return (
            v_max,
            direction * omega_max,
        )

    if label == "turn on-the-spot":
        direction = int(primitive["turn_direction"])

        return (
            0.0,
            direction * omega_max,
        )

    raise ValueError(
        f"Unknown analytical primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """
    Reconstruct the analytical pose trajectory from the saved primitive data.

    Each primitive is sampled using the exact constant-control unicycle
    solution.
    """
    primitives = case["best_analytical_primitives"]

    v_max = float(case["v_max"])
    omega_max = float(case["omega_max"])

    initial_pose = primitives[0]["start_pose"]

    x = float(initial_pose[0])
    y = float(initial_pose[1])
    theta = float(initial_pose[2])

    times = [0.0]
    xs = [x]
    ys = [y]
    thetas = [theta]

    current_time = 0.0

    for primitive in primitives:
        duration = float(
            primitive["maneuver_time"]
        )

        # Ignore zero-duration primitives.
        if duration <= 1.0e-12:
            continue

        v, omega = primitive_controls(
            primitive,
            v_max,
            omega_max,
        )

        n_steps = max(
            2,
            samples_per_primitive,
        )

        dt = duration / n_steps

        for _ in range(n_steps):
            x, y, theta = exact_unicycle_step(
                x,
                y,
                theta,
                v,
                omega,
                dt,
            )

            current_time += dt

            times.append(current_time)
            xs.append(x)
            ys.append(y)
            thetas.append(theta)

    return (
        np.asarray(times),
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(thetas),
    )


# =============================================================================
# TRAJECTORY COMPARISON
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """
    Compare one analytical trajectory with its saved OCP trajectory.
    """
    analytical_times, analytical_x, analytical_y, analytical_theta = (
        reconstruct_analytical_trajectory(case)
    )

    ocp_times = np.asarray(
        case["ocp_time_grid"],
        dtype=float,
    )

    ocp_x = np.asarray(
        case["ocp_x"],
        dtype=float,
    )

    ocp_y = np.asarray(
        case["ocp_y"],
        dtype=float,
    )

    ocp_theta = np.asarray(
        case["ocp_theta"],
        dtype=float,
    )

    tau = np.linspace(
        0.0,
        1.0,
        n_comparison_points,
    )

    analytical_x, analytical_y, analytical_theta = (
        interpolate_pose(
            analytical_times,
            analytical_x,
            analytical_y,
            analytical_theta,
            tau,
        )
    )

    ocp_x, ocp_y, ocp_theta = interpolate_pose(
        ocp_times,
        ocp_x,
        ocp_y,
        ocp_theta,
        tau,
    )

    position_error = np.hypot(
        ocp_x - analytical_x,
        ocp_y - analytical_y,
    )

    heading_error = wrap_to_pi(
        ocp_theta - analytical_theta
    )

    absolute_heading_error = np.abs(
        heading_error
    )

    return {
        "case_id": int(case["case_id"]),
        "theta0_deg": float(case["theta0_deg"]),
        "thetaf_deg": float(case["thetaf_deg"]),
        "analytical_name": case[
            "best_analytical_name"
        ],
        "ocp_sequence": case[
            "ocp_sequence"
        ],

        "position_mean_error": float(
            np.mean(position_error)
        ),
        "position_rms_error": float(
            np.sqrt(
                np.mean(position_error**2)
            )
        ),
        "position_max_error": float(
            np.max(position_error)
        ),

        "heading_mean_abs_error_rad": float(
            np.mean(absolute_heading_error)
        ),
        "heading_rms_error_rad": float(
            np.sqrt(
                np.mean(heading_error**2)
            )
        ),
        "heading_max_error_rad": float(
            np.max(absolute_heading_error)
        ),

        "heading_mean_abs_error_deg": float(
            np.degrees(
                np.mean(absolute_heading_error)
            )
        ),
        "heading_rms_error_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(heading_error**2)
                )
            )
        ),
        "heading_max_error_deg": float(
            np.degrees(
                np.max(absolute_heading_error)
            )
        ),
    }


def print_statistics(
    name,
    values,
    unit,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    print(f"\n{name}")
    print("-" * 80)
    print(
        f"Mean    : {np.mean(values):.6e} {unit}"
    )
    print(
        f"Median  : {np.median(values):.6e} {unit}"
    )
    print(
        f"95th pct: {np.percentile(values, 95):.6e} {unit}"
    )
    print(
        f"Maximum : {np.max(values):.6e} {unit}"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    metadata = data["metadata"]
    cases = data["results"]

    print("=" * 80)
    print("ANALYTICAL / OCP POSE-TRAJECTORY COMPARISON")
    print("=" * 80)

    print(f"Results file : {RESULTS_FILE}")
    print(f"Sweep ID     : {metadata.get('sweep_id')}")
    print(f"N            : {metadata.get('N')}")
    print(f"M            : {metadata.get('M')}")
    print(f"Cases loaded : {len(cases)}")

    comparison_results = []
    skipped_cases = []

    for case in cases:
        if not case.get("success", False):
            skipped_cases.append(
                case.get("case_id")
            )
            continue

        if not case.get("ocp_success", False):
            skipped_cases.append(
                case.get("case_id")
            )
            continue

        required_fields = (
            "ocp_time_grid",
            "ocp_x",
            "ocp_y",
            "ocp_theta",
            "best_analytical_primitives",
        )

        if any(
            case.get(field) is None
            for field in required_fields
        ):
            skipped_cases.append(
                case.get("case_id")
            )
            continue

        result = compare_case(
            case,
            N_COMPARISON_POINTS,
        )

        comparison_results.append(result)

    print(
        f"Cases compared: {len(comparison_results)}"
    )

    print(
        f"Cases skipped : {len(skipped_cases)}"
    )

    if skipped_cases:
        print(
            f"Skipped IDs   : {skipped_cases[:20]}"
        )

    position_mean_errors = [
        result["position_mean_error"]
        for result in comparison_results
    ]

    position_rms_errors = [
        result["position_rms_error"]
        for result in comparison_results
    ]

    position_max_errors = [
        result["position_max_error"]
        for result in comparison_results
    ]

    heading_mean_errors = [
        result["heading_mean_abs_error_deg"]
        for result in comparison_results
    ]

    heading_rms_errors = [
        result["heading_rms_error_deg"]
        for result in comparison_results
    ]

    heading_max_errors = [
        result["heading_max_error_deg"]
        for result in comparison_results
    ]

    print_statistics(
        "MEAN PLANAR DEVIATION",
        position_mean_errors,
        "m",
    )

    print_statistics(
        "RMS PLANAR DEVIATION",
        position_rms_errors,
        "m",
    )

    print_statistics(
        "MAXIMUM PLANAR DEVIATION",
        position_max_errors,
        "m",
    )

    print_statistics(
        "MEAN ABSOLUTE HEADING DEVIATION",
        heading_mean_errors,
        "deg",
    )

    print_statistics(
        "RMS HEADING DEVIATION",
        heading_rms_errors,
        "deg",
    )

    print_statistics(
        "MAXIMUM HEADING DEVIATION",
        heading_max_errors,
        "deg",
    )

    # -------------------------------------------------------------------------
    # Worst planar-error cases
    # -------------------------------------------------------------------------

    worst_position_cases = sorted(
        comparison_results,
        key=lambda result: result[
            "position_max_error"
        ],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("WORST CASES BY MAXIMUM PLANAR DEVIATION")
    print("=" * 80)

    for result in worst_position_cases[
        :N_WORST_CASES_TO_PRINT
    ]:
        sequence = " - ".join(
            result["ocp_sequence"]
        )

        print(
            f"\nCase {result['case_id']}: "
            f"theta0={result['theta0_deg']:.1f} deg, "
            f"thetaf={result['thetaf_deg']:.1f} deg"
        )

        print(
            f"  Analytical family : "
            f"{result['analytical_name']}"
        )

        print(
            f"  OCP sequence      : {sequence}"
        )

        print(
            f"  RMS position error: "
            f"{result['position_rms_error']:.6e} m"
        )

        print(
            f"  Max position error: "
            f"{result['position_max_error']:.6e} m"
        )

        print(
            f"  RMS heading error : "
            f"{result['heading_rms_error_deg']:.6e} deg"
        )

        print(
            f"  Max heading error : "
            f"{result['heading_max_error_deg']:.6e} deg"
        )

    # -------------------------------------------------------------------------
    # Worst heading-error cases
    # -------------------------------------------------------------------------

    worst_heading_cases = sorted(
        comparison_results,
        key=lambda result: result[
            "heading_max_error_deg"
        ],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("WORST CASES BY MAXIMUM HEADING DEVIATION")
    print("=" * 80)

    for result in worst_heading_cases[
        :N_WORST_CASES_TO_PRINT
    ]:
        sequence = " - ".join(
            result["ocp_sequence"]
        )

        print(
            f"\nCase {result['case_id']}: "
            f"theta0={result['theta0_deg']:.1f} deg, "
            f"thetaf={result['thetaf_deg']:.1f} deg"
        )

        print(
            f"  Analytical family : "
            f"{result['analytical_name']}"
        )

        print(
            f"  OCP sequence      : {sequence}"
        )

        print(
            f"  RMS position error: "
            f"{result['position_rms_error']:.6e} m"
        )

        print(
            f"  Max position error: "
            f"{result['position_max_error']:.6e} m"
        )

        print(
            f"  RMS heading error : "
            f"{result['heading_rms_error_deg']:.6e} deg"
        )

        print(
            f"  Max heading error : "
            f"{result['heading_max_error_deg']:.6e} deg"
        )