import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def load_abs_difference_grid(results_path):

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]

    successful_cases = [
        case for case in results
        if case.get("success", False) and case.get("ocp_success", False)
    ]

    theta0_values = sorted(
        set(case["theta0_deg"] for case in successful_cases)
    )
    thetaf_values = sorted(
        set(case["thetaf_deg"] for case in successful_cases)
    )

    n_theta0 = len(theta0_values)
    n_thetaf = len(thetaf_values)

    theta0_to_idx = {theta: i for i, theta in enumerate(theta0_values)}
    thetaf_to_idx = {theta: i for i, theta in enumerate(thetaf_values)}

    abs_diff_grid = np.full(
        (n_thetaf, n_theta0),
        np.nan,
    )

    abs_differences = []

    for case in successful_cases:

        theta0 = case["theta0_deg"]
        thetaf = case["thetaf_deg"]

        i = thetaf_to_idx[thetaf]
        j = theta0_to_idx[theta0]

        abs_diff = abs(case["time_difference"])

        abs_diff_grid[i, j] = abs_diff
        abs_differences.append(abs_diff)

    extent = [
        min(theta0_values),
        max(theta0_values) + 360 / n_theta0,
        min(thetaf_values),
        max(thetaf_values) + 360 / n_thetaf,
    ]

    return abs_diff_grid, extent, np.array(abs_differences)


if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent

    results_dir = current_dir / "results"
    figures_dir = current_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Change these filenames if needed
    results_file_n30 = results_dir / "orientation_sweep_test30.json"
    results_file_n100 = results_dir / "orientation_sweep_test.json"

    grid_n30, extent_n30, diffs_n30 = load_abs_difference_grid(
        results_file_n30
    )

    grid_n100, extent_n100, diffs_n100 = load_abs_difference_grid(
        results_file_n100
    )

    # ------------------------------------------------------------------
    # Logarithmic color scale
    # ------------------------------------------------------------------
    # LogNorm cannot display exact zeros, so very small values are clipped
    # to vmin only for visualization.

    vmin = 1e-4
    vmax = max(
        np.nanmax(grid_n30),
        np.nanmax(grid_n100),
    )

    grid_n30_plot = np.maximum(grid_n30, vmin)
    grid_n100_plot = np.maximum(grid_n100, vmin)

    norm = LogNorm(
        vmin=vmin,
        vmax=vmax,
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        grid_n30_plot,
        origin="lower",
        extent=extent_n30,
        aspect="equal",
        interpolation="nearest",
        norm=norm,
    )

    axes[0].set_title(r"$N=30$")
    axes[0].set_xlabel(r"Initial orientation $\theta_0$ [deg]")
    axes[0].set_ylabel(r"Final orientation $\theta_f$ [deg]")

    im1 = axes[1].imshow(
        grid_n100_plot,
        origin="lower",
        extent=extent_n100,
        aspect="equal",
        interpolation="nearest",
        norm=norm,
    )

    axes[1].set_title(r"$N=100$")
    axes[1].set_xlabel(r"Initial orientation $\theta_0$ [deg]")
    axes[1].set_ylabel(r"Final orientation $\theta_f$ [deg]")

    tick_values = np.arange(0, 361, 90)

    for ax in axes:
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)

    cbar = fig.colorbar(
        im1,
        ax=axes,
        shrink=0.95,
        pad=0.02,
    )

    cbar.set_label(
        r"$|T_{\mathrm{OCP}} - T_{\mathrm{analytical}}|$ [s]"
    )

    pdf_path = (
        figures_dir
        / "sweep1_abs_time_difference_heatmap_comparison_log.pdf"
    )

    png_path = (
        figures_dir
        / "sweep1_abs_time_difference_heatmap_comparison_log.png"
    )

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print("\nN = 30")
    print("-" * 60)
    print(f"Mean   |Delta T| : {np.mean(diffs_n30):.6e} s")
    print(f"Median |Delta T| : {np.median(diffs_n30):.6e} s")
    print(f"Maximum|Delta T| : {np.max(diffs_n30):.6e} s")

    print("\nN = 100")
    print("-" * 60)
    print(f"Mean   |Delta T| : {np.mean(diffs_n100):.6e} s")
    print(f"Median |Delta T| : {np.median(diffs_n100):.6e} s")
    print(f"Maximum|Delta T| : {np.max(diffs_n100):.6e} s")

    print("\nColor scale")
    print("-" * 60)
    print(f"vmin: {vmin:.1e} s")
    print(f"vmax: {vmax:.6e} s")

    print("\nSaved figures:")
    print(pdf_path)
    print(png_path)

    plt.show()