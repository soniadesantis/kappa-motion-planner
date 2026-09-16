"""Compare ECDFs from one to three sweep JSON files. Run --help for options."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# MATPLOTLIB TYPOGRAPHY
# =============================================================================

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "font.size": 18,
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 19,
    "axes.titlesize": 22,
    "axes.titlepad": 14,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.unicode_minus": True,
})


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_FILES = [
    DIRECTORY / "results/sweep4/sobol_sweep_OCP_TST_initial_guess_N50_M4.json",
    DIRECTORY / "results/sweep4/sobol_sweep_OCP_TST_initial_guess_N100_M4.json",
    DIRECTORY / "results/sweep4/sobol_sweep_OCP_TST_initial_guess_N300_M4.json",
]
N_COMPARISON_POINTS = 1001
SAVE_FIGURE = True
SHOW_FIGURE = True
FIGURES_DIRECTORY = DIRECTORY / "figures"
OUTPUT_BASENAME = "sweep_ecdf_comparison"
CURVE_COLORS = ("#D55E00", "#0072B2", "#009E73")
CURVE_LINESTYLES = ("-", "--", "-.")

GRID_COLOR = "0.75"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.7

ZOOM_PERCENTILE = 95

LINEWIDTH = 2.2


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_pose(
    times,
    xs,
    ys,
    thetas,
    normalized_grid,
):
    """Interpolate a pose trajectory on a common normalized-time grid."""

    times = np.asarray(
        times,
        dtype=float,
    ).reshape(-1)

    xs = np.asarray(
        xs,
        dtype=float,
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=float,
    ).reshape(-1)

    thetas = np.asarray(
        thetas,
        dtype=float,
    ).reshape(-1)

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

    duration = (
        times[-1]
        - times[0]
    )

    if duration <= 0.0:
        raise ValueError(
            "Trajectory duration must be positive."
        )

    normalized_times = (
        times - times[0]
    ) / duration

    (
        normalized_times_unique,
        unique_indices,
    ) = np.unique(
        normalized_times,
        return_index=True,
    )

    xs = xs[
        unique_indices
    ]

    ys = ys[
        unique_indices
    ]

    thetas = thetas[
        unique_indices
    ]

    thetas_unwrapped = np.unwrap(
        thetas
    )

    x_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        xs,
    )

    y_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        ys,
    )

    theta_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        thetas_unwrapped,
    )

    return (
        x_interp,
        y_interp,
        theta_interp,
    )


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
    """Exact state update for constant unicycle controls."""

    tolerance = 1.0e-12

    if abs(omega) < tolerance:
        x_new = (
            x
            + v
            * np.cos(theta)
            * dt
        )

        y_new = (
            y
            + v
            * np.sin(theta)
            * dt
        )

        theta_new = theta

    elif abs(v) < tolerance:
        x_new = x
        y_new = y

        theta_new = (
            theta
            + omega
            * dt
        )

    else:
        theta_new = (
            theta
            + omega
            * dt
        )

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

    return (
        x_new,
        y_new,
        theta_new,
    )


def primitive_controls(
    primitive,
    v_max,
    omega_max,
):
    """Return the constant controls associated with one primitive."""

    label = primitive[
        "label"
    ].strip().lower()

    if label == "segment":
        return (
            v_max,
            0.0,
        )

    if label == "arc":
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            v_max,
            direction
            * omega_max,
        )

    if label == "turn on-the-spot":
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            0.0,
            direction
            * omega_max,
        )

    raise ValueError(
        f"Unknown primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """Reconstruct the analytical pose trajectory."""

    primitives = case[
        "best_analytical_primitives"
    ]

    if not primitives:
        raise ValueError(
            f"No analytical primitives saved for case "
            f"{case.get('case_id')}."
        )

    v_max = float(
        case[
            "v_max"
        ]
    )

    omega_max = float(
        case[
            "omega_max"
        ]
    )

    initial_pose = primitives[
        0
    ][
        "start_pose"
    ]

    x = float(
        initial_pose[
            0
        ]
    )

    y = float(
        initial_pose[
            1
        ]
    )

    theta = float(
        initial_pose[
            2
        ]
    )

    times = [
        0.0
    ]

    xs = [
        x
    ]

    ys = [
        y
    ]

    thetas = [
        theta
    ]

    current_time = 0.0

    for primitive in primitives:
        duration = float(
            primitive[
                "maneuver_time"
            ]
        )

        if duration <= 1.0e-12:
            continue

        v, omega = primitive_controls(
            primitive=primitive,
            v_max=v_max,
            omega_max=omega_max,
        )

        n_steps = max(
            2,
            int(
                samples_per_primitive
            ),
        )

        dt = (
            duration
            / n_steps
        )

        for _ in range(
            n_steps
        ):
            (
                x,
                y,
                theta,
            ) = exact_unicycle_step(
                x=x,
                y=y,
                theta=theta,
                v=v,
                omega=omega,
                dt=dt,
            )

            current_time += dt

            times.append(
                current_time
            )

            xs.append(
                x
            )

            ys.append(
                y
            )

            thetas.append(
                theta
            )

    return (
        np.asarray(
            times,
            dtype=float,
        ),
        np.asarray(
            xs,
            dtype=float,
        ),
        np.asarray(
            ys,
            dtype=float,
        ),
        np.asarray(
            thetas,
            dtype=float,
        ),
    )


# =============================================================================
# HAUSDORFF DISTANCE
# =============================================================================

def discrete_symmetric_hausdorff_distance(
    analytical_x,
    analytical_y,
    ocp_x,
    ocp_y,
):
    """Compute the discrete symmetric Hausdorff distance."""

    analytical_points = np.column_stack(
        (
            analytical_x,
            analytical_y,
        )
    )

    ocp_points = np.column_stack(
        (
            ocp_x,
            ocp_y,
        )
    )

    analytical_tree = cKDTree(
        analytical_points
    )

    ocp_tree = cKDTree(
        ocp_points
    )

    analytical_to_ocp, _ = ocp_tree.query(
        analytical_points,
        k=1,
    )

    ocp_to_analytical, _ = analytical_tree.query(
        ocp_points,
        k=1,
    )

    return max(
        float(
            np.max(
                analytical_to_ocp
            )
        ),
        float(
            np.max(
                ocp_to_analytical
            )
        ),
    )


# =============================================================================
# CASE COMPARISON
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """Compute the two metrics used in the Sweep 4 ECDF figure."""

    analytical_time = float(
        case[
            "best_analytical_time"
        ]
    )

    ocp_time = float(
        case[
            "ocp_time"
        ]
    )

    if analytical_time <= 0.0:
        raise ValueError(
            f"Nonpositive analytical time in case "
            f"{case.get('case_id')}."
        )

    relative_time_error_percent = (
        100.0
        * abs(
            ocp_time
            - analytical_time
        )
        / analytical_time
    )

    (
        analytical_times,
        analytical_x,
        analytical_y,
        analytical_theta,
    ) = reconstruct_analytical_trajectory(
        case
    )

    ocp_times = np.asarray(
        case[
            "ocp_time_grid"
        ],
        dtype=float,
    )

    ocp_x = np.asarray(
        case[
            "ocp_x"
        ],
        dtype=float,
    )

    ocp_y = np.asarray(
        case[
            "ocp_y"
        ],
        dtype=float,
    )

    ocp_theta = np.asarray(
        case[
            "ocp_theta"
        ],
        dtype=float,
    )

    normalized_grid = np.linspace(
        0.0,
        1.0,
        n_comparison_points,
    )

    (
        analytical_x_interp,
        analytical_y_interp,
        analytical_theta_interp,
    ) = interpolate_pose(
        analytical_times,
        analytical_x,
        analytical_y,
        analytical_theta,
        normalized_grid,
    )

    (
        ocp_x_interp,
        ocp_y_interp,
        ocp_theta_interp,
    ) = interpolate_pose(
        ocp_times,
        ocp_x,
        ocp_y,
        ocp_theta,
        normalized_grid,
    )

    hausdorff_distance = (
        discrete_symmetric_hausdorff_distance(
            analytical_x=(
                analytical_x_interp
            ),
            analytical_y=(
                analytical_y_interp
            ),
            ocp_x=ocp_x_interp,
            ocp_y=ocp_y_interp,
        )
    )

    return {
        "relative_time_error_percent": float(
            relative_time_error_percent
        ),
        "hausdorff_distance": float(
            hausdorff_distance
        ),
    }


def load_and_process_results(results_file, n_comparison_points=N_COMPARISON_POINTS):
    """Prefer saved Hausdorff measurements; reconstruct only for older files."""
    with Path(results_file).open(encoding="utf-8") as source:
        data = json.load(source)
    metrics = []
    saved_count = 0
    for case in data["results"]:
        if not (case.get("success") and case.get("ocp_success")):
            continue
        analytical_time = case.get("best_analytical_time")
        ocp_time = case.get("ocp_time")
        if analytical_time is None or ocp_time is None or analytical_time <= 0:
            continue
        distance = case.get("hausdorff_distance")
        if distance is not None:
            result = {
                "relative_time_error_percent": 100 * abs(ocp_time - analytical_time) / analytical_time,
                "hausdorff_distance": float(distance),
            }
            saved_count += 1
        else:
            required = ("best_analytical_primitives", "ocp_time_grid", "ocp_x", "ocp_y", "ocp_theta")
            if any(case.get(field) is None for field in required):
                continue
            result = compare_case(case, n_comparison_points)
        if all(np.isfinite(value) and value >= 0 for value in result.values()):
            metrics.append(result)
    if not metrics:
        raise ValueError(f"No valid comparisons in {results_file}")
    print(f"\nInput: {results_file}")
    print(f"Compared: {len(metrics):,}; skipped: {len(data['results']) - len(metrics):,}")
    print(f"Hausdorff distances: {saved_count:,} saved, {len(metrics) - saved_count:,} reconstructed")
    return data.get("metadata", {}), metrics


# =============================================================================
# ECDF HELPERS
# =============================================================================

def compute_ecdf(values):
    """Return sorted values and empirical cumulative probabilities."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot compute an ECDF from an empty array."
        )

    sorted_values = np.sort(
        values
    )

    cumulative_probabilities = (
        np.arange(
            1,
            sorted_values.size + 1,
            dtype=float,
        )
        / sorted_values.size
    )

    return (
        sorted_values,
        cumulative_probabilities,
    )


def print_metric_statistics(
    transcription_resolution,
    relative_errors,
    hausdorff_distances,
):
    """Print the main percentile values used to interpret the ECDF."""

    print(
        f"\nN={transcription_resolution}"
    )

    print(
        "-" * 70
    )

    print(
        "Relative traversal-time discrepancy"
    )

    print(
        f"  Median : "
        f"{np.median(relative_errors):.6e} %"
    )

    print(
        f"  95th   : "
        f"{np.percentile(relative_errors, 95):.6e} %"
    )

    print(
        f"  Maximum: "
        f"{np.max(relative_errors):.6e} %"
    )

    print(
        "Symmetric Hausdorff distance"
    )

    print(
        f"  Median : "
        f"{np.median(hausdorff_distances):.6e} m"
    )

    print(
        f"  95th   : "
        f"{np.percentile(hausdorff_distances, 95):.6e} m"
    )

    print(
        f"  Maximum: "
        f"{np.max(hausdorff_distances):.6e} m"
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sobol_ecdfs(sweeps):
    """Show full and percentile-zoomed ECDFs without renormalizing the zoom."""
    if not 1 <= len(sweeps) <= 3:
        raise ValueError("Provide between one and three sweeps.")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True, layout="constrained")
    metric_specs = (
        ("relative_time_error_percent", "Relative traversal-time discrepancy\n"
         + r"$e_{\mathcal{T}}^{\mathrm{rel}}\,[\%]$"),
        ("hausdorff_distance", "Symmetric Hausdorff distance\n"
         + r"$d_{\mathrm{H}}\,[\mathrm{m}]$"),
    )
    for row, (key, xlabel) in enumerate(metric_specs):
        samples = [np.asarray([result[key] for result in metrics]) for _, metrics in sweeps]
        full_limit = max(float(np.max(values)) for values in samples) * 1.03 or 1e-6
        zoom_cutoff = max(float(np.percentile(values, ZOOM_PERCENTILE)) for values in samples)
        zoom_limit = zoom_cutoff * 1.05 or min(full_limit, 1e-6)
        print(f"{key}: zoom cutoff = {zoom_cutoff:.6e} "
              f"(largest {ZOOM_PERCENTILE}th percentile across sweeps)")
        for (label, _), values, color, linestyle in zip(
            sweeps, samples, CURVE_COLORS, CURVE_LINESTYLES
        ):
            unique_x, counts = np.unique(values, return_counts=True)
            probabilities = np.cumsum(counts) / len(values)
            for axis in axes[row]:
                axis.step(np.r_[0, unique_x, full_limit], np.r_[0, probabilities, 1],
                          where="post", color=color, linestyle=linestyle,
                          linewidth=LINEWIDTH, label=label)
        for column, limit in enumerate((full_limit, zoom_limit)):
            axis = axes[row, column]
            axis.set(xlim=(0, limit), ylim=(0, 1.01), xlabel=xlabel)
            axis.axhline(0.95, color="0.15", linestyle="--", linewidth=2, zorder=2)
            axis.annotate(r"$95\%$", xy=(0.98, 0.95),
                          xycoords=axis.get_yaxis_transform(), xytext=(0, -5),
                          textcoords="offset points", ha="right", va="top",
                          fontsize=18, color="0.15")
            axis.grid(color=GRID_COLOR, linestyle=GRID_LINESTYLE,
                      linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
            axis.legend(loc="lower right", frameon=True)
        axes[row, 0].set_ylabel("Fraction of cases")
    # Column headings distinguish the ranges; axis labels identify the metrics.
    axes[0, 0].set_title("Full range", fontsize=20)
    axes[0, 1].set_title(f"Zoom to the {ZOOM_PERCENTILE}th percentile", fontsize=20)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="One to three sweep JSON paths.")
    parser.add_argument("--labels", nargs="+", help="Optional legend labels, one per input file.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-name", default=OUTPUT_BASENAME)
    args = parser.parse_args()
    files = args.files or RESULTS_FILES
    if not 1 <= len(files) <= 3:
        parser.error("Provide one to three input files.")
    if args.labels and len(args.labels) != len(files):
        parser.error("Provide exactly one label per input file.")
    sweeps = []
    for index, path in enumerate(files):
        path = path.expanduser()
        if not path.is_absolute() and not path.is_file():
            path = DIRECTORY / "results" / path
        metadata, metrics = load_and_process_results(path)
        label = (args.labels[index] if args.labels else
                 rf"$N={metadata['N']}$"
                 if "N" in metadata else path.stem)
        if not args.labels and any(previous_label == label for previous_label, _ in sweeps):
            label = path.stem
        sweeps.append((label, metrics))
        print_metric_statistics(metadata.get("N", "unknown"),
                                [m["relative_time_error_percent"] for m in metrics],
                                [m["hausdorff_distance"] for m in metrics])
    figure = plot_sobol_ecdfs(sweeps)
    if SAVE_FIGURE:
        FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            output = FIGURES_DIRECTORY / f"{args.output_name}.{extension}"
            figure.savefig(output, dpi=300, bbox_inches="tight")
            print(f"Saved {output}")
    if SHOW_FIGURE and not args.no_show:
        plt.show(block=True)
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
