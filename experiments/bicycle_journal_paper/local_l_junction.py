"""Cropped L-junction and disk erosion of its corridor union for the thesis.

Run directly in VS Code, or use --no-show to export without opening a window.
The shaded safe region is the corridor union eroded by a radius-r disk.
The example has downward incoming heading, rightward outgoing heading and a
counterclockwise turn: the local basis therefore points right and up.
"""

import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath

from corridor_dimensions_study import STYLE


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "thesis_figures"
ROBOT_RADIUS = 0.25
MINIMUM_TURNING_RADIUS = 1.5
VERTICAL_WIDTH = 2.0
HORIZONTAL_WIDTH = 1.6


def create_figure():
    """Place the concave corner at the origin and crop the corridor arms."""
    r = ROBOT_RADIUS
    R = MINIMUM_TURNING_RADIUS
    w1, w2 = VERTICAL_WIDTH, HORIZONTAL_WIDTH
    dx, dy = w1, w2
    extent = 6.0  # Corridor ends lie outside the cropped view.
    if not 0 < 2 * r < min(w1, w2):
        raise ValueError("Both corridors must be wider than the robot diameter.")
    if R <= r:
        raise ValueError("The minimum turning radius must exceed the footprint radius.")

    figure_style = {**STYLE,
                    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"}
    with plt.rc_context(figure_style):
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
        original_color, safe_color = "white", "#EDF5FC"
        safe_border = "#486F91"
        center_color = "white"
        ax.add_patch(Rectangle((0, 0), extent, extent,
                               facecolor=center_color, edgecolor="none", zorder=0))
        original = [(-w1, -w2), (extent, -w2), (extent, 0),
                    (0, 0), (0, extent), (-w1, extent)]
        # The disk erosion of the union offsets the outer walls by r and
        # rounds the concave corner with a radius-r arc centered at the origin.
        corner_arc = PlotPath.arc(180, 270)
        safe_vertices = [(-w1 + r, -w2 + r), (-w1 + r, extent - r),
                         (-r, extent - r), (-r, 0)]
        safe_codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * 3
        safe_vertices.extend(r * corner_arc.vertices[1:])
        safe_codes.extend(corner_arc.codes[1:])
        safe_vertices.extend([(extent - r, -r), (extent - r, -w2 + r)])
        safe_codes.extend([PlotPath.LINETO] * 2)
        safe_vertices.append((-w1 + r, -w2 + r))
        safe_codes.append(PlotPath.CLOSEPOLY)
        ax.add_patch(Polygon(original, facecolor=original_color,
                             edgecolor="none", zorder=1))
        ax.add_patch(PathPatch(PlotPath(safe_vertices, safe_codes),
                               facecolor=safe_color, edgecolor=safe_border,
                               linewidth=0.75, zorder=2))
        # Show both original rectangles, including their overlap boundaries.
        ax.add_patch(Rectangle((-w1, -w2), w1, extent + w2,
                               fill=False, edgecolor="0.2", linewidth=0.4,
                               zorder=3))
        ax.add_patch(Rectangle((-w1, -w2), extent + w1, w2,
                               fill=False, edgecolor="0.2", linewidth=0.4,
                               zorder=3))
        ax.text(-1.4, 2.2, r"$\mathcal C_j$", ha="center",
                va="center", fontsize=18)
        ax.text(2.0, -1.15, r"$\mathcal C_{j+1}$", ha="center",
                va="center", fontsize=18)
        label_background = {"facecolor": "white", "edgecolor": "none", "pad": 1}
        # Dimension the full overlap of the original corridor rectangles.
        dimension_y, dimension_x = -1.95, -2.4
        for x in (-dx, 0):
            ax.plot([x, x], [-dy, dimension_y - 0.08],
                    color="0.5", linewidth=0.4, linestyle=":", zorder=3)
        for y in (-dy, 0):
            ax.plot([-dx, dimension_x - 0.08], [y, y],
                    color="0.5", linewidth=0.4, linestyle=":", zorder=3)
        for start, end in (((-dx, dimension_y), (0, dimension_y)),
                           ((dimension_x, -dy), (dimension_x, 0))):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->",
                                        mutation_scale=10, linewidth=0.65,
                                        color="0.2", shrinkA=0, shrinkB=0, zorder=7))
        ax.text(-w1 / 2, dimension_y, r"$d_{x,j}$", ha="center", va="center",
                fontsize=16, bbox=label_background, zorder=8)
        ax.text(dimension_x, -w2 / 2, r"$d_{y,j}$", ha="center", va="center",
                rotation=90, fontsize=16, bbox=label_background, zorder=8)

        # Both basis vectors point toward the turn side of their own corridor.
        for end in ((1.55, 0), (0, 1.55)):
            ax.add_patch(FancyArrowPatch((0, 0), end, arrowstyle="-|>",
                                        mutation_scale=12, linewidth=1,
                                        color="black", zorder=5))
        ax.text(1.6, 0.13, r"$\bm e_{x,j}$", ha="center", va="bottom", fontsize=16)
        ax.text(0.13, 1.6, r"$\bm e_{y,j}$", ha="left", va="center", fontsize=16)
        ax.plot(0, 0, "o", color="black", markersize=3, zorder=6)
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#C93434",
                            linewidth=0.55, zorder=5))
        ax.text(-0.13, 0.16, r"$\bm p_j^{\mathrm{corn}}$", fontsize=16,
                ha="right", va="bottom")
        ax.add_patch(FancyArrowPatch((0, 0), (-0.25, 0.27),
                                    connectionstyle="arc3,rad=0.4",
                                    arrowstyle="->", mutation_scale=8,
                                    linewidth=0.55, color="black",
                                    shrinkA=4, shrinkB=2, zorder=5))
        radius = R - r
        ax.add_patch(Wedge((0, 0), radius, theta1=0, theta2=90,
                           facecolor="#F0F0F0", edgecolor="none", zorder=2))
        ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta1=0, theta2=90,
                         color="0.2", linewidth=0.9, linestyle="--", zorder=4))
        radial_contact = (radius / sqrt(2), radius / sqrt(2))
        ax.add_patch(FancyArrowPatch((0, 0), radial_contact, arrowstyle="->",
                                    mutation_scale=10, linewidth=0.65,
                                    color="0.2", shrinkA=0, shrinkB=0, zorder=4))
        ax.annotate(r"$R-r$", (radial_contact[0] / 2, radial_contact[1] / 2),
                    xytext=(9, -9), textcoords="offset points", fontsize=16,
                    rotation=45, ha="center", va="center")
        # The longitudinal centerlines meet inside the extended overlap.
        waypoint = (-w1 / 2, -w2 / 2)
        ax.plot([waypoint[0], waypoint[0], extent],
                [extent, waypoint[1], waypoint[1]],
                color="0.3", linewidth=0.7, zorder=5)
        ax.plot(*waypoint, "o", color="0.3", markersize=4, zorder=6)
        ax.annotate(r"$\bm p_j$", waypoint, xytext=(-4, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=16)

        ax.set(xlim=(-2.8, 2.7), ylim=(-2.3, 2.7), aspect="equal")
        ax.set_axis_off()
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    fig = create_figure()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**STYLE,
                         "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"}):
        for suffix in ("pdf", "png"):
            output = OUTPUT_DIRECTORY / f"local_l_junction.{suffix}"
            fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved {output}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
