import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def get_best_name(case):
    return (
        case.get("best_analytical_name")
        or case.get("best_trajectory")
        or case.get("best_name")
    )


def short_label(best_name):
    """
    Example:
        'TCSCT left-right' -> 'TCSCT LR'
    """
    family, directions = best_name.split(" ", 1)
    d0, df = directions.split("-")

    direction_code = {
        "left": "L",
        "right": "R",
    }

    return f"{family} {direction_code[d0]}{direction_code[df]}"


if __name__ == "__main__":

    RESULTS_FILENAME = "orientation_sweep_analytical.json"

    current_dir = Path(__file__).resolve().parent
    results_path = current_dir / "results" / RESULTS_FILENAME

    figures_dir = current_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]

    successful_cases = [
        case for case in results
        if case.get("success", False) and get_best_name(case) is not None
    ]

    theta0_values = sorted(
        set(case["theta0_deg"] for case in successful_cases)
    )
    thetaf_values = sorted(
        set(case["thetaf_deg"] for case in successful_cases)
    )

    n_theta0 = len(theta0_values)
    n_thetaf = len(thetaf_values)

    theta0_to_idx = {
        theta: i for i, theta in enumerate(theta0_values)
    }

    thetaf_to_idx = {
        theta: i for i, theta in enumerate(thetaf_values)
    }

    labels = sorted(
        set(short_label(get_best_name(case)) for case in successful_cases)
    )

    label_to_id = {
        label: i for i, label in enumerate(labels)
    }

    sequence_grid = np.full(
        (n_thetaf, n_theta0),
        np.nan,
    )

    label_counts = {
        label: 0 for label in labels
    }

    n_valid_cases = 0

    for case in successful_cases:

        theta0 = case["theta0_deg"]
        thetaf = case["thetaf_deg"]

        i = thetaf_to_idx[thetaf]
        j = theta0_to_idx[theta0]

        best_name = get_best_name(case)
        label = short_label(best_name)

        sequence_grid[i, j] = label_to_id[label]

        label_counts[label] += 1
        n_valid_cases += 1

    n_labels = len(labels)

    sequence_colors = {
        # CSC: blue shades
        "CSC LL": "#08306B",
        "CSC LR": "#2171B5",
        "CSC RL": "#6BAED6",
        "CSC RR": "#C6DBEF",

        # TCSC: orange shades
        "TCSC LL": "#7F2704",
        "TCSC LR": "#D94801",
        "TCSC RL": "#FD8D3C",
        "TCSC RR": "#FDD0A2",

        # CSCT: green shades
        "CSCT LL": "#00441B",
        "CSCT LR": "#238B45",
        "CSCT RL": "#74C476",
        "CSCT RR": "#C7E9C0",

        # TCSCT: purple shades
        "TCSCT LL": "#4A1486",
        "TCSCT LR": "#807DBA",
        "TCSCT RL": "#BCBDDC",
        "TCSCT RR": "#EFEDF5",
    }

    colors = [
        sequence_colors[label]
        for label in labels
    ]

    listed_cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, n_labels + 0.5, 1),
        ncolors=n_labels,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.imshow(
        sequence_grid,
        origin="lower",
        cmap=listed_cmap,
        norm=norm,
        extent=[
            min(theta0_values),
            max(theta0_values) + 360 / n_theta0,
            min(thetaf_values),
            max(thetaf_values) + 360 / n_thetaf,
        ],
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel(r"Initial orientation $\theta_0$ [deg]")
    ax.set_ylabel(r"Final orientation $\theta_f$ [deg]")
    ax.set_title(r"Optimal analytical sequence map")

    tick_values = np.arange(0, 361, 45)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    ax.grid(False)

    legend_patches = [
        mpatches.Patch(
            color=colors[label_to_id[label]],
            label=label,
        )
        for label in labels
    ]

    ax.legend(
        handles=legend_patches,
        title="Sequence",
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
    )

    fig.tight_layout()

    pdf_path = figures_dir / "sweep1_sequence_map.pdf"
    png_path = figures_dir / "sweep1_sequence_map.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print("\nSequence labels")
    print("-" * 60)

    for label, idx in label_to_id.items():
        print(f"{idx:02d}: {label}")

    print("\nSequence occurrence statistics")
    print("-" * 60)
    print(f"Total valid cases: {n_valid_cases}")

    for label in labels:
        count = label_counts[label]
        percentage = 100.0 * count / n_valid_cases

        print(
            f"{label:9s}: "
            f"{count:7d} cases "
            f"({percentage:6.2f} %)"
        )

    print("\nLaTeX table rows")
    print("-" * 60)

    for label in labels:
        count = label_counts[label]
        percentage = 100.0 * count / n_valid_cases

        print(f"{label} & {count} & {percentage:.2f} \\\\")

    print("\nSaved figures:")
    print(pdf_path)
    print(png_path)

    plt.show()