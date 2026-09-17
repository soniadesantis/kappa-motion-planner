"""Plot the four-family and sixteen-sequence sweep1 maps side by side.

Run with arena-env. Reads the saved analytical sweep and writes PDF and PNG
to the adjacent figures directory; no trajectory computation is needed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
RESULTS_PATH = DIRECTORY / "results/sweep1/orientation_sweep_analytical.json"
OUTPUT_NAME = "sweep1_family_sequence_maps"
SHOW_FIGURE = True

FAMILIES = ("CSC", "TCSC", "CSCT", "TCSCT")
DIRECTIONS = ("LL", "LR", "RL", "RR")
FAMILY_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
FAMILY_HATCH = "///"
FAMILY_HATCH_COLOR = (0, 0, 0, 0.55)
# Same shades as the standalone sweep1_sequence_map.
SEQUENCE_COLORS = (
    "#08306B", "#2171B5", "#6BAED6", "#C6DBEF",
    "#7F2704", "#D94801", "#FD8D3C", "#FDD0A2",
    "#00441B", "#238B45", "#74C476", "#C7E9C0",
    "#4A1486", "#807DBA", "#BCBDDC", "#EFEDF5",
)
STYLE = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "font.size": 17,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "axes.titlepad": 18,
    "legend.fontsize": 17,
    "legend.title_fontsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "hatch.linewidth": 1.1,
}


def sequence_notation(family, directions):
    """Use the adjacent arc's sign for each turn on the spot."""
    initial, final = ("+" if direction == "L" else "-" for direction in directions)
    label = rf"C^{{{initial}}}SC^{{{final}}}"
    if family.startswith("T"):
        label = rf"T^{{{initial}}}" + label
    if family.endswith("T"):
        label += rf"T^{{{final}}}"
    return f"${label}$"


def load_grids(results_path):
    with Path(results_path).open() as source:
        cases = [case for case in json.load(source)["results"]
                 if case.get("success", False)]
    if not cases:
        raise ValueError("The sweep contains no successful cases.")
    initial_angles = sorted({case["theta0_deg"] for case in cases})
    final_angles = sorted({case["thetaf_deg"] for case in cases})
    initial_indices = {angle: i for i, angle in enumerate(initial_angles)}
    final_indices = {angle: i for i, angle in enumerate(final_angles)}
    shape = (len(final_angles), len(initial_angles))
    family_grid = np.full(shape, np.nan)
    sequence_grid = np.full(shape, np.nan)
    sequence_counts = np.zeros(len(FAMILIES) * len(DIRECTIONS), dtype=int)
    direction_codes = {"left": "L", "right": "R"}
    for case in cases:
        name = (case.get("best_analytical_name") or case.get("best_trajectory")
                or case.get("best_name"))
        if name is None:
            continue
        family, directions = name.split(" ", 1)
        pair = "".join(direction_codes[direction] for direction in directions.split("-"))
        family_id = FAMILIES.index(family)
        index = (final_indices[case["thetaf_deg"]], initial_indices[case["theta0_deg"]])
        family_grid[index] = family_id
        sequence_id = 4 * family_id + DIRECTIONS.index(pair)
        sequence_grid[index] = sequence_id
        sequence_counts[sequence_id] += 1
    # Match the cell extents used by both existing standalone maps.
    extent = (min(initial_angles), max(initial_angles) + 360 / len(initial_angles),
              min(final_angles), max(final_angles) + 360 / len(final_angles))
    return family_grid, sequence_grid, extent, sequence_counts


def print_occurrences(sequence_counts):
    """Report counts and percentages of valid named successful cases."""
    total = int(sequence_counts.sum())
    family_counts = sequence_counts.reshape(len(FAMILIES), len(DIRECTIONS)).sum(axis=1)
    print(f"\nTotal valid cases: {total:,}")
    print("\nFamily occurrences")
    for family, count in zip(FAMILIES, family_counts):
        percentage = 100 * count / total if total else 0
        print(f"{family:6s} {count:8,d}  ({percentage:6.2f}%)")
    print("\nSequence occurrences (L = +, R = -)")
    for index, count in enumerate(sequence_counts):
        family = FAMILIES[index // len(DIRECTIONS)]
        pair = DIRECTIONS[index % len(DIRECTIONS)]
        percentage = 100 * count / total if total else 0
        print(f"{family:6s} {pair}  {count:8,d}  ({percentage:6.2f}%)")


def create_figure(results_path=RESULTS_PATH):
    family_grid, sequence_grid, extent, sequence_counts = load_grids(results_path)
    print_occurrences(sequence_counts)
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(12, 7), layout="constrained")
        layout = figure.add_gridspec(2, 4, height_ratios=(3, 1.1))
        axes = (figure.add_subplot(layout[0, :2]), figure.add_subplot(layout[0, 2:]))
        panel_specs = (
            (family_grid, FAMILY_COLORS, "a) Analytical optimal family map"),
            (sequence_grid, SEQUENCE_COLORS, "b) Analytical optimal sequence map"),
        )
        for ax, (grid, colors, title) in zip(axes, panel_specs):
            ax.imshow(
                grid, origin="lower", extent=extent, aspect="equal",
                interpolation="nearest", cmap=ListedColormap(colors),
                norm=BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors)),
            )
            if grid is family_grid:
                # Hatch the family map only; the sequence map stays solid.
                # One overlay covers the map without altering category boundaries.
                ax.add_patch(Rectangle(
                    (0, 0), 360, 360, facecolor="none",
                    edgecolor=FAMILY_HATCH_COLOR, hatch=FAMILY_HATCH, linewidth=0,
                ))
            ax.set(xlabel=r"$\theta_0$ [deg]",
                   ylabel=r"$\theta_f$ [deg]",
                   title=title, xlim=(0, 360), ylim=(0, 360))
            ax.set_xticks(np.arange(0, 361, 45))
            ax.set_yticks(np.arange(0, 361, 45))
            ax.grid(False)
        # Each parent swatch belongs to panel a; its four children belong to b.
        for family_index, family in enumerate(FAMILIES):
            legend_ax = figure.add_subplot(layout[1, family_index])
            legend_ax.set(xlim=(0, 1), ylim=(0, 1))
            legend_ax.set_axis_off()
            legend_ax.add_patch(Rectangle(
                (0.04, 0.85), 0.13, 0.065,
                facecolor=FAMILY_COLORS[family_index], edgecolor=FAMILY_HATCH_COLOR,
                hatch=FAMILY_HATCH, linewidth=0,
            ))
            legend_ax.text(0.21, 0.8825, f"${family}$", va="center",
                           fontsize=STYLE["legend.title_fontsize"])
            legend_ax.plot([0.105, 0.105], [0.81, 0.12], color="0.65", linewidth=0.9)
            for direction_index, pair in enumerate(DIRECTIONS):
                y = 0.69 - 0.19 * direction_index
                legend_ax.plot([0.105, 0.19], [y, y], color="0.65", linewidth=0.9)
                legend_ax.add_patch(Rectangle(
                    (0.19, y - 0.0325), 0.13, 0.065,
                    facecolor=SEQUENCE_COLORS[4 * family_index + direction_index],
                    edgecolor="none",
                ))
                legend_ax.text(0.36, y, sequence_notation(family, pair), va="center",
                               fontsize=STYLE["legend.fontsize"])
    return figure


if __name__ == "__main__":
    figure = create_figure()
    figures_dir = DIRECTORY / "figures"
    figures_dir.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        output = figures_dir / f"{OUTPUT_NAME}.{extension}"
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)
