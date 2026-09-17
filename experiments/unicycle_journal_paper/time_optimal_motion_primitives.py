"""Draw the five motion primitives from the journal reference PDF.

Run with --no-show to export PDF and PNG without opening a window.
The original time_optimal_motion_primitives.pdf is kept as a reference.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
import numpy as np

from tangent_convention_journal import STYLE

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures"
OUTPUT_NAME = "time_optimal_motion_primitives_recreated"
START_COLOR = "green"
END_COLOR = "red"
PATH_COLOR = "#0072B2"
TURN_COLOR = "#B87400"
ARC_COLOR = "#AA4499"
GREY = "0.5"


class PoseLegendHandler(HandlerBase):
    """Show the same circular footprint and heading arrow as a plotted pose."""

    def create_artists(self, legend, original, xdescent, ydescent,
                       width, height, fontsize, transform):
        color = original.get_color()
        center = np.array([width * 0.30 - xdescent, height * 0.40 - ydescent])
        radius = height * 0.40
        circle = Circle(center, radius, fill=False, edgecolor=color,
                        lw=1.8, transform=transform)
        arrow = FancyArrowPatch(center, center + height * np.array([0.95, 0.55]),
                                arrowstyle="-|>", mutation_scale=fontsize * 0.65,
                                color=color, lw=1.8, shrinkA=0, shrinkB=0,
                                transform=transform)
        return [circle, arrow]


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def pose(ax, position, heading, color):
    ax.add_patch(Circle(position, 0.14, fill=False, edgecolor=color, lw=1.8, zorder=6))
    ax.add_patch(FancyArrowPatch(position, position + 0.36 * unit(heading),
                                arrowstyle="-|>", mutation_scale=13, color=color,
                                lw=1.8, shrinkA=0, shrinkB=0, zorder=7))


def angle(ax, center, start, end, radius, color, symbol):
    lower, upper = sorted(np.degrees([start, end]))
    ax.add_patch(Wedge(center, radius, lower, upper, facecolor=color,
                       edgecolor="none", alpha=0.18, zorder=1))
    values = np.linspace(start, end, 100)
    points = center + radius * np.array([np.cos(values), np.sin(values)]).T
    ax.plot(*points.T, color=color, lw=1.1, zorder=3)
    ax.add_patch(FancyArrowPatch(points[-5], points[-1], arrowstyle="-|>",
                                mutation_scale=10, color=color, lw=1,
                                shrinkA=0, shrinkB=0, zorder=4))
    ax.text(*(center + (radius + 0.15) * unit((start + end) / 2)), symbol,
            fontsize=25, color=color, ha="center", va="center")


def turn(ax, sign):
    center = np.array([0., 0.])
    initial = np.pi / 6
    final = -np.pi / 2 if sign == "-" else 2 * np.pi / 3
    angle(ax, center, initial, final, 0.43, TURN_COLOR, r"$\phi$")
    pose(ax, center, initial, START_COLOR)
    pose(ax, center, final, END_COLOR)
    ax.text(-0.36 if sign == "-" else -0.43, 0.10,
            rf"$T^{{{sign}}}$", fontsize=30, ha="center")
    ax.set(xlim=(-1.05, 0.85), ylim=(-0.7, 0.75))


def circular_arc(ax, sign):
    # Preserve the C-/C+ convention used in the supplied reference.
    center = np.array([0., 0.])
    start, end = (0., 1.37) if sign == "-" else (np.pi, np.pi - 1.37)
    values = np.linspace(start, end, 150)
    points = np.array([np.cos(values), np.sin(values)]).T
    ax.plot(*points.T, color=PATH_COLOR, lw=2.5, zorder=3)
    for point in (points[0], points[-1]):
        ax.plot([0, point[0]], [0, point[1]], "--", color=GREY, lw=1)
    angle(ax, center, start, end, 0.39, ARC_COLOR, r"$\iota$")
    direction = 1 if sign == "-" else -1
    pose(ax, points[0], start + direction * np.pi / 2, START_COLOR)
    pose(ax, points[-1], end + direction * np.pi / 2, END_COLOR)
    ax.text(-0.10 if sign == "-" else -0.85, 1.22 if sign == "-" else 0.92,
            rf"$C^{{{sign}}}$", fontsize=30)
    ax.set(xlim=(-0.45, 1.45) if sign == "-" else (-1.45, 0.45),
           ylim=(-0.25, 1.55))


def straight(ax):
    start = np.array([0., 0.])
    end = np.array([1.10, 1.10])
    ax.plot([start[0], end[0]], [start[1], end[1]], color=PATH_COLOR, lw=2.5)
    pose(ax, start, np.pi / 4, START_COLOR)
    pose(ax, end, np.pi / 4, END_COLOR)
    offset = np.array([0.22, -0.22])
    ax.add_patch(FancyArrowPatch(start + offset, end + offset, arrowstyle="<->",
                                color=GREY, lw=1.1, mutation_scale=12,
                                shrinkA=0, shrinkB=0))
    ax.text(*(0.5 * (start + end) + 1.5 * offset), r"$d$", color=GREY,
            fontsize=25, ha="center", va="center")
    ax.text(0.38, 0.78, r"$S$", fontsize=30)
    ax.set(xlim=(-0.35, 1.65), ylim=(-0.45, 1.65))


def create_figure():
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 4.4))
        # Nest the legend and turns above the arcs, as in the reference.
        axes = [fig.add_axes((0.29, 0.57, 0.25, 0.41)),
                fig.add_axes((0.54, 0.57, 0.25, 0.41)),
                fig.add_axes((0.02, 0.01, 0.32, 0.61)),
                fig.add_axes((0.36, 0.01, 0.32, 0.61)),
                fig.add_axes((0.73, 0.07, 0.26, 0.77))]
        turn(axes[0], "-")
        turn(axes[1], "+")
        circular_arc(axes[2], "-")
        circular_arc(axes[3], "+")
        straight(axes[4])
        for ax in axes:
            ax.set_aspect("equal")
            ax.axis("off")
        legend = fig.legend(handles=[
            Line2D([], [], color=START_COLOR, marker="o", markerfacecolor="none",
                   lw=1.8, markersize=9, label=r"Start pose $\mathbf{x}^{s}$"),
            Line2D([], [], color=END_COLOR, marker="o", markerfacecolor="none",
                   lw=1.8, markersize=9, label=r"End pose $\mathbf{x}^{e}$")],
            loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1,
            fontsize=21, frameon=True, fancybox=False, edgecolor="0.8",
            handler_map={Line2D: PoseLegendHandler()}, handleheight=1.4)
        legend.get_frame().set_linewidth(0.6)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    fig = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ("pdf", "png"):
            path = OUTPUT_DIRECTORY / f"{OUTPUT_NAME}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved {path}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
