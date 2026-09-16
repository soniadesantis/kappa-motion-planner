"""Plot signed traversal-time differences for one saved sweep.

Run without arguments for the configured Sobol sweep, or pass another JSON:
    python histogram_plot.py /path/to/results.json --no-show
Paths relative to the parent results directory are also accepted.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_FILENAME = "sweep4/sobol_sweep_OCP_TST_initial_guess_N300_M4.json"
NUMBER_OF_BINS = 40
ZOOM_PERCENTILE = 99
SAVE_FIGURE = True
SHOW_FIGURE = True
FIGURES_DIRECTORY = DIRECTORY / "figures"

STYLE = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
}


def load_time_differences(results_path):
    """Return finite signed differences for cases where both methods succeeded."""
    with Path(results_path).open(encoding="utf-8") as source:
        data = json.load(source)
    values = np.array([
        float(case["time_difference"])
        for case in data["results"]
        if case.get("success", False) and case.get("ocp_success", False)
        and case.get("time_difference") is not None
    ], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No successful cases with finite time differences in {results_path}")
    return values


def compute_statistics(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "p99": float(np.percentile(values, ZOOM_PERCENTILE))}


def print_statistics(results_path, values, statistics):
    print(f"\nInput: {results_path}")
    print(f"Cases            : {values.size:,}")
    for name, value in statistics.items():
        print(f"{name.capitalize():7s} Delta T: {value:.6e} s")


def scientific_notation(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def create_figure(values, number_of_bins=NUMBER_OF_BINS):
    if number_of_bins < 1:
        raise ValueError("The number of bins must be positive.")
    statistics = compute_statistics(values)
    # Include zero and retain a usable range when all differences are zero.
    lower_limit = min(0.0, statistics["minimum"])
    upper_limit = max(0.0, statistics["maximum"])
    if lower_limit == upper_limit:
        upper_limit = 1e-6
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        bin_edges = np.linspace(lower_limit, upper_limit, number_of_bins + 1)
        axis.hist(values, bins=bin_edges, edgecolor="black", linewidth=0.7,
                  log=True)
        for name, linestyle in (("mean", "--"), ("median", ":")):
            value = statistics[name]
            axis.axvline(
                value, linestyle=linestyle, linewidth=1.8,
                label=rf"{name.capitalize()} $={scientific_notation(value)}\,\mathrm{{s}}$",
            )
        axis.set(
            xlim=(lower_limit, upper_limit),
            xlabel=r"$\Delta\mathcal{T}=\mathcal{T}_{\mathrm{OCP}}-\mathcal{T}_{\mathrm{analytical}}$ [s]",
            ylabel="Number of cases (log scale)",
        )
        axis.set_ylim(bottom=0.8)
        axis.legend(frameon=True, loc="upper right")
        axis.grid(axis="y", alpha=0.25)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=DIRECTORY / "results" / RESULTS_FILENAME,
                        help="Sweep JSON path (default: configured Sobol N300 sweep).")
    parser.add_argument("--bins", type=int, default=NUMBER_OF_BINS)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIRECTORY)
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window.")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    results_path = args.input.expanduser()
    if not results_path.is_absolute() and not results_path.is_file():
        results_path = DIRECTORY / "results" / results_path
    values = load_time_differences(results_path)
    print_statistics(results_path, values, compute_statistics(values))
    figure = create_figure(values, args.bins)
    if SAVE_FIGURE:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            output = args.output_dir / f"{results_path.stem}_signed_time_difference_histogram.{extension}"
            figure.savefig(output, dpi=300, bbox_inches="tight")
            print(f"Saved {output}")
    if SHOW_FIGURE and not args.no_show:
        plt.show(block=True)
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
