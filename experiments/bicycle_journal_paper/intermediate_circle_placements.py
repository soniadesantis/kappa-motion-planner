"""Four local intermediate-circle placements, styled for the thesis.

Run directly in VS Code or use --no-show to export PDF and PNG.
The cases illustrate the width/overlap clearance rule, independently of the
planner's additional waypoint-region acceptance checks.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as PlotPath

from corridor_dimensions_study import D, R, S, STYLE, example_cases, r


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "thesis_figures"
FIGURE_STYLE = {**STYLE,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"}
TITLES = (
    "(a) Preferred\n$45^\\circ$ placement",
    "(b) Preferred\nshifted placement",
    "(c) Ordinary\n$45^\\circ$ placement",
    "(d) Ordinary\nshifted placement",
)


def draw_placement(ax, case, dx, dy, center):
    """Draw a regular junction with overlap dimensions dx and dy."""
    extent = 5 * R
    safe_color, safe_border = "#EDF5FC", "#486F91"
    arc = PlotPath.arc(180, 270)
    vertices = [(-dx + r, -dy + r), (-dx + r, extent - r),
                (-r, extent - r), (-r, 0)]
    codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * 3
    vertices.extend(r * arc.vertices[1:])
    codes.extend(arc.codes[1:])
    vertices.extend([(extent - r, -r), (extent - r, -dy + r),
                     (-dx + r, -dy + r)])
    codes.extend([PlotPath.LINETO, PlotPath.LINETO, PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=safe_color,
                           edgecolor=safe_border, linewidth=0.65, zorder=1))
    for lower_left, width, height in (((-dx, -dy), dx, extent + dy),
                                      ((-dx, -dy), extent + dx, dy)):
        ax.add_patch(Rectangle(lower_left, width, height, fill=False,
                               edgecolor="0.2", linewidth=0.4, zorder=2))
    # Half-junction boundaries make the distinction between preferred and
    # ordinary clearance visible in all four panels.
    ax.plot([-dx / 2] * 2, [-dy, extent], color="0.65", linewidth=0.5,
            linestyle=":", zorder=2)
    ax.plot([-dx, extent], [-dy / 2] * 2, color="0.65", linewidth=0.5,
            linestyle=":", zorder=2)
    ax.add_patch(Wedge((0, 0), D, 0, 90, facecolor="#F0F0F0",
                       edgecolor="none", zorder=1))
    ax.add_patch(Arc((0, 0), 2 * D, 2 * D, theta1=0, theta2=90,
                     color="0.3", linewidth=0.65, linestyle="--", zorder=3))
    ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#C93434",
                        linewidth=0.5, zorder=4))
    ax.plot(0, 0, "o", color="black", markersize=2.5, zorder=5)
    for end in ((1.25 * R, 0), (0, 1.25 * R)):
        ax.add_patch(FancyArrowPatch((0, 0), end, arrowstyle="-|>",
                                    mutation_scale=9, linewidth=0.7,
                                    color="black", zorder=4))
    ax.text(1.27 * R, 0.08 * R, r"$\bm e_{x,j}$", fontsize=16,
            ha="center", va="bottom")
    ax.text(0.08 * R, 1.27 * R, r"$\bm e_{y,j}$", fontsize=16,
            ha="left", va="center")

    # Only the southwest quarter is the local turning maneuver. The rest of
    # its supporting circle is shown faintly and need not be collision-free.
    ax.add_patch(Circle(center, R, fill=False, edgecolor="0.6", linewidth=0.6,
                        linestyle="--", zorder=3))
    ax.add_patch(Arc(center, 2 * S, 2 * S, theta1=180, theta2=270,
                     color="0.45", linewidth=0.65, linestyle="--", zorder=3))
    ax.add_patch(Arc(center, 2 * R, 2 * R, theta1=180, theta2=270,
                     color="0.25", linewidth=1.15, zorder=4))
    ax.plot(*center, "o", color="0.25", markersize=3.5, zorder=5)
    ax.annotate(r"$\bm o_j$", center, xytext=(4, 5), textcoords="offset points",
                fontsize=17, ha="left", va="bottom")

    bottom, left = -dy - 0.28 * R, -dx - 0.28 * R
    for start, end in (((-dx, bottom), (0, bottom)), ((left, -dy), (left, 0))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->",
                                    mutation_scale=8, linewidth=0.55,
                                    color="0.2", shrinkA=0, shrinkB=0))
    ax.text(-dx / 2, bottom - 0.08 * R, rf"$d_{{x,j}}={dx/R:g}R$", fontsize=17,
            ha="center", va="top")
    ax.text(left - 0.08 * R, -dy / 2, rf"$d_{{y,j}}={dy/R:g}R$", fontsize=17,
            rotation=90, ha="right", va="center")
    ax.set(xlim=(-2.3 * R, 2 * R), ylim=(-2.3 * R, 2 * R), aspect="equal")
    ax.set_title(TITLES[case - 1], fontsize=17, pad=8)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.955,
                            wspace=0.02)
        for ax, example in zip(axes.flat, example_cases()):
            draw_placement(ax, *example)
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            output = OUTPUT_DIRECTORY / f"intermediate_circle_placements.{suffix}"
            fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved {output}")
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
