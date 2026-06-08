import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


if __name__ == "__main__":

    RESULTS_FILENAME = "orientation_sweep_test.json"

    current_dir = Path(__file__).resolve().parent
    results_path = current_dir / "results" / RESULTS_FILENAME

    figures_dir = current_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

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

    abs_differences = np.array(abs_differences)

    mean_diff = np.mean(abs_differences)
    median_diff = np.median(abs_differences)
    max_diff = np.max(abs_differences)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        abs_diff_grid,
        origin="lower",
        extent=[
            min(theta0_values),
            max(theta0_values) + 360 / n_theta0,
            min(thetaf_values),
            max(thetaf_values) + 360 / n_thetaf,
        ],
        aspect="equal",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"$|T_{\mathrm{OCP}} - T_{\mathrm{analytical}}|$ [s]"
    )

    ax.set_xlabel(r"Initial orientation $\theta_0$ [deg]")
    ax.set_ylabel(r"Final orientation $\theta_f$ [deg]")
    ax.set_title(r"Absolute time difference")

    tick_values = np.arange(0, 361, 45)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    fig.tight_layout()

    heatmap_pdf_path = figures_dir / "sweep1_abs_time_difference_heatmap100.pdf"
    heatmap_png_path = figures_dir / "sweep1_abs_time_difference_heatmap100.png"

    fig.savefig(heatmap_pdf_path, bbox_inches="tight")
    fig.savefig(heatmap_png_path, dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        abs_differences,
        bins=40,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axvline(
        mean_diff,
        linestyle="--",
        linewidth=2,
        label=rf"Mean $= {mean_diff:.2e}\,\mathrm{{s}}$",
    )

    ax.axvline(
        median_diff,
        linestyle=":",
        linewidth=2,
        label=rf"Median $= {median_diff:.2e}\,\mathrm{{s}}$",
    )

    ax.set_xlabel(
        r"$|T_{\mathrm{OCP}} - T_{\mathrm{analytical}}|$ [s]"
    )
    ax.set_ylabel(r"Number of cases")
    ax.set_title(r"Distribution of absolute time differences")
    ax.legend(frameon=True)

    fig.tight_layout()

    hist_pdf_path = figures_dir / "sweep1_abs_time_difference_histogram100.pdf"
    hist_png_path = figures_dir / "sweep1_abs_time_difference_histogram100.png"

    fig.savefig(hist_pdf_path, bbox_inches="tight")
    fig.savefig(hist_png_path, dpi=300, bbox_inches="tight")

    print("\nTime difference statistics")
    print("-" * 60)
    print(f"Mean   |Delta T| : {mean_diff:.6e} s")
    print(f"Median |Delta T| : {median_diff:.6e} s")
    print(f"Maximum|Delta T| : {max_diff:.6e} s")

    print("\nSaved figures:")
    print(heatmap_pdf_path)
    print(heatmap_png_path)
    print(hist_pdf_path)
    print(hist_png_path)

    plt.show()