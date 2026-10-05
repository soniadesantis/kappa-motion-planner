"""Overlap-based corner-circle placement: four ordered local construction classes.

Run directly in VS Code, or use --no-show for PDF/PNG export. The width map
shows which candidate is selected by the effective overlap dimensions; it does
not apply the planner's A_j acceptance check or certify a complete path.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Patch, Polygon, Rectangle
import numpy as np


# Editable study parameters. All example widths below are multiples of R.
R = 1.0
r = 0.2
w_plot_max = 2.8 * R
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "thesis_figures"
EXAMPLE_WIDTHS = ((1.6, 1.6), (1.0, 1.8), (1.0, 1.0), (0.5, 1.0))

S = R + r
D = R - r
q = D / np.sqrt(2)
w_min = 2 * r
w_45 = S - q
WIDTH_TOL = 32 * np.finfo(float).eps * S
SQUARED_TOL = 32 * np.finfo(float).eps * S**2

CASE_NAMES = {
    1: r"Preferred $45^\circ$ placement",
    2: "Preferred shifted placement",
    3: r"Ordinary $45^\circ$ placement",
    4: "Ordinary shifted placement",
}
CASE_COLORS = {1: "#486A7C", 2: "#416B59", 3: "#946D28", 4: "#80556F"}
CASE_FILLS = {1: "#B7C8D2", 2: "#B7C8B9", 3: "#DFD0A8", 4: "#D0B9C6"}
STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 20,
    "axes.labelsize": 23,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def lower_bounds(w1, w2):
    """Ordinary wall-clearance bounds a, b for the local center (x, y)."""
    return np.maximum(0.0, S - np.asarray(w1)), np.maximum(0.0, S - np.asarray(w2))


def centerline_half_bounds(w1, w2):
    """Preferred-half bounds h_1, h_2 for the local center (x, y)."""
    return (np.maximum(0.0, S - np.asarray(w1) / 2),
            np.maximum(0.0, S - np.asarray(w2) / 2))


def is_feasible_width_pair(w1, w2):
    """Whether a center in the local corner-following family exists."""
    a, b = lower_bounds(w1, w2)
    return a**2 + b**2 <= D**2 + SQUARED_TOL


def is_centerline_half_feasible(w1, w2):
    """Whether some center satisfies both preferred-half bounds."""
    h1, h2 = centerline_half_bounds(w1, w2)
    return h1**2 + h2**2 <= D**2 + SQUARED_TOL


def placement_classes(w1, w2):
    """Return classes 1--4 in priority order; 0 means no local candidate.

    Equality is admitted at every clearance boundary. The roundoff tolerance
    only protects numerical evaluation of those boundary equalities.
    """
    h1, h2 = centerline_half_bounds(w1, w2)
    a, b = lower_bounds(w1, w2)
    conditions = (
        (h1 <= q + WIDTH_TOL) & (h2 <= q + WIDTH_TOL),
        h1**2 + h2**2 <= D**2 + SQUARED_TOL,
        (a <= q + WIDTH_TOL) & (b <= q + WIDTH_TOL),
        a**2 + b**2 <= D**2 + SQUARED_TOL,
    )
    return np.select(conditions, (1, 2, 3, 4), default=0)


def choose_example_center(w1, w2):
    """Select a width-based candidate, without applying the planner's A_j test."""
    case = int(placement_classes(w1, w2))
    if case in (1, 3):
        center = np.array([q, q])
    elif case == 2:
        center = np.array(centerline_half_bounds(w1, w2))
    elif case == 4:
        center = np.array(lower_bounds(w1, w2))
    else:
        return None, "no local placement"
    return center, CASE_NAMES[case]


def minimum_w2_for_w1(w1):
    """Lower boundary a(w_1)^2 + b(w_2)^2 <= D^2."""
    a = np.maximum(0.0, S - np.asarray(w1))
    required = S - np.sqrt(np.maximum(0.0, D**2 - a**2))
    return np.where(a <= D + WIDTH_TOL, np.maximum(w_min, required), np.nan)


def minimum_w2_centerline_half_for_w1(w1):
    """Lower boundary h_1(w_1)^2 + h_2(w_2)^2 <= D^2."""
    h1 = np.maximum(0.0, S - np.asarray(w1) / 2)
    required = 2 * (S - np.sqrt(np.maximum(0.0, D**2 - h1**2)))
    return np.where(h1 <= D + WIDTH_TOL, np.maximum(w_min, required), np.nan)


def example_cases():
    """Return one explicit width pair and selected center for each class."""
    examples = []
    for expected, (w1_ratio, w2_ratio) in enumerate(EXAMPLE_WIDTHS, 1):
        w1, w2 = R * w1_ratio, R * w2_ratio
        actual = int(placement_classes(w1, w2))
        if actual != expected:
            raise ValueError(
                f"Example {expected} belongs to class {actual} for r/R={r/R:g}. "
                "Update EXAMPLE_WIDTHS to illustrate the four classes."
            )
        center, _ = choose_example_center(w1, w2)
        examples.append((expected, w1, w2, center))
    return examples


def create_width_map():
    """Partition effective overlap dimensions into the four placement classes.

    The scalar clearance helpers also apply to corridor widths at a regular
    junction, where these equal the overlap dimensions.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 3.15))
        # Reserve a narrow right column for the legend and keep the square
        # data axes compact; this uses less height than the two-row legend.
        fig.subplots_adjust(left=0.13, right=0.59, bottom=0.20, top=0.98)
        values = np.unique(np.concatenate([
            np.linspace(w_min, w_plot_max, 1200),
            [w_min, 2 * w_min, w_45, 2 * w_45, S, 2 * S],
        ]))
        values = values[(values >= w_min) & (values <= w_plot_max)]
        ordinary = minimum_w2_for_w1(values)
        preferred = minimum_w2_centerline_half_for_w1(values)

        # Paint in reverse priority order, using opaque fills. No mixed colors.
        ax.fill_between(values / R, ordinary / R, w_plot_max / R,
                        color=CASE_FILLS[4], linewidth=0)
        ax.add_patch(Rectangle((w_45 / R, w_45 / R),
                               (w_plot_max - w_45) / R,
                               (w_plot_max - w_45) / R,
                               facecolor=CASE_FILLS[3], edgecolor="none"))
        ax.fill_between(values / R, preferred / R, w_plot_max / R,
                        color=CASE_FILLS[2], linewidth=0)
        ax.add_patch(Rectangle((2 * w_45 / R, 2 * w_45 / R),
                               (w_plot_max - 2 * w_45) / R,
                               (w_plot_max - 2 * w_45) / R,
                               facecolor=CASE_FILLS[1], edgecolor="none"))

        # Only draw boundaries which separate different selected classes.
        ax.plot(values[values <= S] / R, ordinary[values <= S] / R,
                color=CASE_COLORS[4], lw=1.15)
        ax.plot(values / R, preferred / R, color=CASE_COLORS[2], lw=1.05)
        upper = np.linspace(w_45, w_plot_max, 600)
        ordinary_active = ~is_centerline_half_feasible(w_45, upper)
        visible = np.where(ordinary_active, upper / R, np.nan)
        ax.plot(np.full_like(upper, w_45 / R), visible,
                color=CASE_COLORS[3], lw=1.05)
        ax.plot(visible, np.full_like(upper, w_45 / R),
                color=CASE_COLORS[3], lw=1.05)
        threshold = 2 * w_45 / R
        ax.plot([threshold, threshold, w_plot_max / R],
                [w_plot_max / R, threshold, threshold],
                color=CASE_COLORS[2], lw=1.05)
        half_min = 2 * (S - D) / R
        half_saturation = 2 * S / R
        ax.plot([half_min, half_min], [half_saturation, w_plot_max / R],
                color=CASE_COLORS[2], lw=1.05)
        ax.plot([half_saturation, w_plot_max / R], [half_min, half_min],
                color=CASE_COLORS[2], lw=1.05)

        ax.set(xlim=(w_min / R, w_plot_max / R),
               ylim=(w_min / R, w_plot_max / R), aspect="equal")
        ax.set_xlabel(r"$d_{x,j}/R$", fontsize=14)
        ax.set_ylabel(r"$d_{y,j}/R$", fontsize=14)
        ax.tick_params(axis="both", labelsize=9.5)
        fig.text(0.61, 0.83, rf"$r/R={r/R:g}$", fontsize=12,
                 ha="left", va="center")
        ax.tick_params(direction="out", length=4)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(handles=[
            Patch(facecolor=CASE_FILLS[case], edgecolor="none",
                  label=CASE_NAMES[case])
            for case in (1, 2, 3, 4)
        ], loc="center left", bbox_to_anchor=(0.59, 0.49),
            ncol=1, fontsize=12, frameon=False, labelspacing=0.65,
            handlelength=0.85, handleheight=0.8)
        return fig


def draw_width(ax, start, end, symbol, offset, rotation=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->",
                                mutation_scale=11, lw=0.95, color="0.2",
                                shrinkA=0, shrinkB=0))
    midpoint = (np.asarray(start) + np.asarray(end)) / 2 + offset
    ax.text(*midpoint, symbol, ha="center", va="center",
            rotation=rotation, fontsize=17)


def draw_case(ax, case, w1, w2, center):
    """Show the actual L-junction, supporting circle, and swept SW quarter."""
    lower, upper = -2.25 * R, 1.95 * R
    color = CASE_COLORS[case]
    # The two open corridor arms meet at the concave corner (0, 0).
    vertices = [(-w1, -w2), (upper, -w2), (upper, 0),
                (0, 0), (0, upper), (-w1, upper)]
    ax.add_patch(Polygon(vertices, facecolor="0.96", edgecolor="none"))
    for xs, ys in [
        ([-w1, -w1], [-w2, upper]), ([-w1, upper], [-w2, -w2]),
        ([0, 0], [0, upper]), ([0, upper], [0, 0]),
    ]:
        ax.plot(xs, ys, color="black", lw=1.2)
    ax.plot([-w1 / 2] * 2, [-w2, upper], color="0.55", lw=1.0, ls=":")
    ax.plot([-w1, upper], [-w2 / 2] * 2, color="0.55", lw=1.0, ls=":")

    ax.add_patch(Circle((0, 0), r, facecolor="0.88", edgecolor="0.5", lw=0.9))
    ax.plot(0, 0, "o", ms=3, color="0.25")
    ax.add_patch(Circle(center, R, fill=False, edgecolor="0.45", lw=1.0))
    ax.add_patch(Arc(center, 2 * S, 2 * S, theta1=180, theta2=270,
                     color="0.4", lw=1.2, ls="--"))
    ax.add_patch(Arc(center, 2 * R, 2 * R, theta1=180, theta2=270,
                     color=color, lw=3.0))
    if case in (2, 4):
        nominal = np.array([q, q])
        ax.plot(*nominal, "o", ms=5, mfc="white", mec="0.5", zorder=6)
        ax.add_patch(FancyArrowPatch(nominal, center, arrowstyle="->",
                                    color="0.5", lw=0.9, mutation_scale=9,
                                    shrinkA=4, shrinkB=4, zorder=6))
    ax.plot(*center, "o", ms=5, color=color, zorder=7)
    ax.text(*(center + [0.14 * R, 0.13 * R]), r"$\mathbf{o}$",
            color=color, fontsize=20, ha="left", va="center")

    draw_width(ax, (-w1, -w2 - 0.25 * R), (0, -w2 - 0.25 * R),
               rf"$w_1={w1/R:g}R$", (0, -0.15 * R))
    draw_width(ax, (-w1 - 0.25 * R, -w2), (-w1 - 0.25 * R, 0),
               rf"$w_2={w2/R:g}R$", (-0.15 * R, 0), rotation=90)
    ax.set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
    ax.set_title(CASE_NAMES[case], fontsize=18, pad=7)
    ax.axis("off")


def create_geometry_figure(examples):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.1))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.11, top=0.95,
                            wspace=0.08, hspace=0.20)
        for ax, example in zip(axes.flat, examples):
            draw_case(ax, *example)
        fig.legend(handles=[
            Line2D([], [], color="0.45", lw=1.0, label=r"Supporting circle $R$"),
            Line2D([], [], color="0.4", lw=1.2, ls="--",
                   label=r"Outer swept radius $R+r$"),
            Line2D([], [], color="0.55", lw=1.0, ls=":", label="Corridor centerlines"),
            Patch(facecolor="0.88", edgecolor="0.5", label=r"Corner disk $r$"),
        ], loc="lower center", bbox_to_anchor=(0.5, 0.02),
            ncol=2, fontsize=14, frameon=False, columnspacing=1.6,
            handlelength=1.5, labelspacing=0.6)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Export without opening windows.")
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if not 0 < r < R:
        raise ValueError("The parameters must satisfy 0 < r < R.")
    if w_plot_max <= 2 * S:
        raise ValueError("Use w_plot_max > 2(R+r) to show the preferred-half region clearly.")
    examples = example_cases()
    for case, w1, w2, center in examples:
        print(f"Class {case}: w1/R={w1/R:g}, w2/R={w2/R:g}; "
              f"center/R=({center[0]/R:.4f}, {center[1]/R:.4f})")

    figures = {
        "corridor_dimensions_placement_map": create_width_map(),
        "corridor_dimensions_placement_examples": create_geometry_figure(examples),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                output = args.output_directory / f"{name}.{extension}"
                fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.06)
                print(f"Saved {output}")
    if args.no_show:
        for fig in figures.values():
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
