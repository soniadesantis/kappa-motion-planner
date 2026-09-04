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
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def extract_family(best_name):
    """
    Example:
        'TCSCT left-right' -> 'TCSCT'
    """
    return best_name.split(" ", 1)[0]


if __name__ == "__main__":

    RESULTS_FILENAME = "sweep1/orientation_sweep_analytical.json"

    current_dir = Path(__file__).resolve().parent
    results_path = current_dir / "results" / RESULTS_FILENAME

    figures_dir = current_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]

    successful_cases = [
        case for case in results
        if case.get("success", False)
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

    family_order = [
        "CSC",
        "TCSC",
        "CSCT",
        "TCSCT",
    ]

    family_to_id = {
        family: i for i, family in enumerate(family_order)
    }

    family_grid = np.full(
        (n_thetaf, n_theta0),
        np.nan,
    )

    family_counts = {
        family: 0 for family in family_order
    }

    n_valid_cases = 0

    for case in successful_cases:

        theta0 = case["theta0_deg"]
        thetaf = case["thetaf_deg"]

        i = thetaf_to_idx[thetaf]
        j = theta0_to_idx[theta0]

        best_name = (
            case.get("best_analytical_name")
            or case.get("best_trajectory")
            or case.get("best_name")
        )

        if best_name is None:
            continue

        family = extract_family(best_name)

        family_grid[i, j] = family_to_id[family]

        family_counts[family] += 1
        n_valid_cases += 1

    n_families = len(family_order)

    family_colors = {
        "CSC":   "#0072B2",  # blue
        "TCSC":  "#D55E00",  # vermillion
        "CSCT":  "#009E73",  # bluish green
        "TCSCT": "#CC79A7",  # reddish purple
    }

    colors = [
        family_colors["CSC"],
        family_colors["TCSC"],
        family_colors["CSCT"],
        family_colors["TCSCT"],
    ]
    listed_cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, n_families + 0.5, 1),
        ncolors=n_families,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(
        family_grid,
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
    ax.set_title(r"Optimal trajectory family map")

    tick_values = np.arange(0, 361, 45)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    legend_patches = [
        mpatches.Patch(
            color=colors[family_to_id[family]],
            label=family,
        )
        for family in family_order
    ]

    ax.legend(
        handles=legend_patches,
        title="Family",
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()

    pdf_path = figures_dir / "sweep1_family_map.pdf"
    png_path = figures_dir / "sweep1_family_map.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print("\nFamily labels")
    print("-" * 60)
    for family, idx in family_to_id.items():
        print(f"{idx}: {family}")

    print("\nFamily occurrence statistics")
    print("-" * 60)
    print(f"Total valid cases: {n_valid_cases}")

    for family in family_order:
        count = family_counts[family]
        percentage = 100.0 * count / n_valid_cases

        print(
            f"{family:5s}: "
            f"{count:7d} cases "
            f"({percentage:6.2f} %)"
        )

    print("\nLaTeX table rows")
    print("-" * 60)

    for family in family_order:
        count = family_counts[family]
        percentage = 100.0 * count / n_valid_cases

        print(
            f"{family} & {count} & {percentage:.2f} \\\\"
        )

    print("\nSaved figures:")
    print(pdf_path)
    print(png_path)

    plt.show()