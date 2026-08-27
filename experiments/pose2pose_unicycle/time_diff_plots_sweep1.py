import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RESULTS_FILENAMES = {
    30: "sweep1/orientation_sweep_OCP_path_saved_N30_M4.json",
    100: "sweep1/orientation_sweep_N100_OCPpath.json",
}

NUMBER_OF_BINS = 40

SAVE_FIGURE = True
SHOW_FIGURE = True

OUTPUT_FILENAME = (
    "sweep1_abs_time_difference_histograms_N30_N100"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_absolute_time_differences(results_path):
    """
    Load the successful cases from a sweep result file and return the
    absolute traversal-time discrepancies.
    """

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    results = data["results"]

    successful_cases = [
        case
        for case in results
        if case.get("success", False)
        and case.get("ocp_success", False)
    ]

    absolute_differences = np.array(
        [
            abs(float(case["time_difference"]))
            for case in successful_cases
        ],
        dtype=float,
    )

    if absolute_differences.size == 0:
        raise ValueError(
            "No successful cases were found in:\n"
            f"{results_path}"
        )

    return absolute_differences


def compute_statistics(absolute_differences):
    """Compute summary statistics for one discrepancy distribution."""

    return {
        "mean": float(
            np.mean(absolute_differences)
        ),
        "median": float(
            np.median(absolute_differences)
        ),
        "maximum": float(
            np.max(absolute_differences)
        ),
    }


def print_statistics(
    transcription_resolution,
    absolute_differences,
    statistics,
):
    """Print summary statistics for one transcription resolution."""

    print(
        f"\nN = {transcription_resolution}"
    )
    print("-" * 60)

    print(
        f"Cases             : "
        f"{absolute_differences.size}"
    )

    print(
        f"Mean   |Delta T|  : "
        f"{statistics['mean']:.6e} s"
    )

    print(
        f"Median |Delta T|  : "
        f"{statistics['median']:.6e} s"
    )

    print(
        f"Maximum|Delta T|  : "
        f"{statistics['maximum']:.6e} s"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    current_directory = (
        Path(__file__).resolve().parent
    )

    results_directory = (
        current_directory
        / "results"
    )

    figures_directory = (
        current_directory
        / "figures"
    )

    if SAVE_FIGURE:
        figures_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------------------
    # Load the two result files
    # -------------------------------------------------------------------------

    absolute_differences = {}

    for transcription_resolution, filename in (
        RESULTS_FILENAMES.items()
    ):
        results_path = (
            results_directory
            / filename
        )

        absolute_differences[
            transcription_resolution
        ] = load_absolute_time_differences(
            results_path=results_path,
        )

    # -------------------------------------------------------------------------
    # Compute and print statistics
    # -------------------------------------------------------------------------

    statistics = {}

    for transcription_resolution in (
        30,
        100,
    ):
        values = absolute_differences[
            transcription_resolution
        ]

        statistics[
            transcription_resolution
        ] = compute_statistics(
            values
        )

        print_statistics(
            transcription_resolution=(
                transcription_resolution
            ),
            absolute_differences=values,
            statistics=statistics[
                transcription_resolution
            ],
        )

    # -------------------------------------------------------------------------
    # Create side-by-side histograms
    #
    # Each panel uses its own horizontal and vertical scale. This keeps both
    # distributions visible despite the substantial reduction in the
    # discrepancies for N = 100.
    # -------------------------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.4),
        sharex=False,
        sharey=False,
    )

    for axis, transcription_resolution in zip(
        axes,
        (
            30,
            100,
        ),
    ):
        values = absolute_differences[
            transcription_resolution
        ]

        mean_difference = statistics[
            transcription_resolution
        ]["mean"]

        median_difference = statistics[
            transcription_resolution
        ]["median"]

        maximum_difference = statistics[
            transcription_resolution
        ]["maximum"]

        # Use the same number of bins in both panels, while adapting the
        # horizontal range to each discrepancy distribution.
        bin_edges = np.linspace(
            0.0,
            maximum_difference,
            NUMBER_OF_BINS + 1,
        )

        axis.hist(
            values,
            bins=bin_edges,
            edgecolor="black",
            linewidth=0.7,
        )

        axis.axvline(
            mean_difference,
            linestyle="--",
            linewidth=1.8,
            label=(
                rf"Mean $="
                rf"{mean_difference:.2e}\,"
                rf"\mathrm{{s}}$"
            ),
        )

        axis.axvline(
            median_difference,
            linestyle=":",
            linewidth=1.8,
            label=(
                rf"Median $="
                rf"{median_difference:.2e}\,"
                rf"\mathrm{{s}}$"
            ),
        )

        axis.set_xlim(
            0.0,
            maximum_difference,
        )

        axis.set_xlabel(
            r"$|T_{\mathrm{OCP}}"
            r"-T_{\mathrm{analytical}}|$ [s]"
        )

        axis.set_ylabel(
            "Number of cases"
        )

        axis.set_title(
            rf"$N={transcription_resolution}$"
        )

        axis.legend(
            frameon=True,
            loc="upper right",
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    # figure.suptitle(
    #     "Distribution of absolute traversal-time discrepancies",
    #     fontsize=13,
    # )

    figure.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.94,
        ],
        w_pad=2.5,
    )

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    histogram_pdf_path = (
        figures_directory
        / f"{OUTPUT_FILENAME}.pdf"
    )

    histogram_png_path = (
        figures_directory
        / f"{OUTPUT_FILENAME}.png"
    )

    if SAVE_FIGURE:
        figure.savefig(
            histogram_pdf_path,
            bbox_inches="tight",
        )

        figure.savefig(
            histogram_png_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("\nSaved figures:")
        print(histogram_pdf_path)
        print(histogram_png_path)

    if SHOW_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            figure
        )