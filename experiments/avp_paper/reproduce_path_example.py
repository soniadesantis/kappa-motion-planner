"""Redraw path_example.svg with the notation used in the AVP manuscript.

Usage:
    python experiments/avp_paper/reproduce_path_example.py

Requires NumPy and Matplotlib; neither Inkscape nor a LaTeX installation is
needed. All map geometry, paths, and labels are drawn as editable vector
objects. Coordinates follow the reference image (y increases downward).
The layout is reconstructed from the illustration, not from planner output.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

FIGURES = Path(__file__).resolve().parent / "figures"
PATH_COLOR = "#00b958"
CIRCLE_COLOR = "#ff3030"
CENTERLINE_COLOR = "#176887"
PRIMITIVE_COLOR = "#009d00"
RADIUS = 58.0

# The two parking circles have equal radii and are externally tangent.
# Circle 1 is tangent to lane 3; circle 2 is tangent to centerline 4.
NAV_1 = np.array([177.0, 261.0])
NAV_2 = np.array([177.0, 529.0])
PARK_1 = np.array([396.0, 529.0])
PARK_2 = np.array([
    488.0,
    PARK_1[1] + math.sqrt((2 * RADIUS) ** 2 - (488 - PARK_1[0]) ** 2),
])
P0 = np.array([394.0, 182.0])

# P[i] is the start position of primitive i+1; P[8] is the final position.
P = np.array([
    [394, 203], [177, 203], [119, 261], [119, 529], [177, 587],
    [396, 587], (PARK_1 + PARK_2) / 2, [430, PARK_2[1]], [430, 403],
])


def rectangle(ax, x, y, width, height, color, linewidth=1.2,
              linestyle="--", alpha=1.0):
    ax.add_patch(Rectangle(
        (x, y), width, height, fill=False, edgecolor=color,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha,
    ))


def draw_environment(ax):
    """Faint corridor decomposition and the four selected centerlines."""
    for x in (84, 163):
        for y, height in ((27, 320), (170, 451)):
            rectangle(ax, x, y, 74, height, "red", alpha=0.25)
    for y in (170, 249, 472, 551):
        rectangle(ax, 85, y, 392, 74, "green", alpha=0.25)
    for y in (47, 347):
        for x in (270, 335, 400):
            rectangle(ax, x, y, 64, 225, "#bbbb00", alpha=0.28)
        rectangle(ax, 270, y, 194, 123, "black", alpha=0.23)

    for bounds in ((85, 170, 392, 70), (85, 170, 70, 451),
                   (85, 551, 392, 70), (399, 347, 62, 274)):
        rectangle(ax, *bounds, color="#042433", linewidth=2,
                  linestyle="-")

    centerlines = (
        ((473, 203), (88, 203)), ((119, 174), (119, 619)),
        ((88, 587), (475, 587)), ((430, 618), (430, 349)),
    )
    for start, end in centerlines:
        ax.plot(*zip(start, end), color=CENTERLINE_COLOR, linewidth=1.2,
                linestyle=(0, (5, 4)), zorder=2)
        delta = np.array(end) - np.array(start)
        tail = np.array(end) - 18 * delta / np.linalg.norm(delta)
        ax.annotate("", xy=end, xytext=tail, arrowprops=dict(
            arrowstyle="-|>", color=CENTERLINE_COLOR, lw=1.2,
            mutation_scale=17,
        ))


def arc(ax, center, start_angle, end_angle):
    angles = np.deg2rad(np.linspace(start_angle, end_angle, 160))
    ax.plot(center[0] + RADIUS * np.cos(angles),
            center[1] + RADIUS * np.sin(angles),
            color=PATH_COLOR, lw=2.8, zorder=4)


def draw_path(ax):
    for center in (NAV_1, NAV_2, PARK_1, PARK_2):
        ax.add_patch(Circle(
            center, RADIUS, fill=False, edgecolor=CIRCLE_COLOR,
            linewidth=1.3, linestyle=(0, (7, 4)), zorder=3,
        ))
    for first, last in ((0, 1), (2, 3), (4, 5), (7, 8)):
        ax.plot(P[[first, last], 0], P[[first, last], 1],
                color=PATH_COLOR, lw=2.8, zorder=4)
    arc(ax, NAV_1, -90, -180)
    arc(ax, NAV_2, 180, 90)
    tangent_angle = math.degrees(math.atan2(
        PARK_2[1] - PARK_1[1], PARK_2[0] - PARK_1[0]
    ))
    arc(ax, PARK_1, 90, tangent_angle)
    arc(ax, PARK_2, tangent_angle + 180, 180)
    ax.plot([P0[0], P[0, 0]], [P0[1], P[0, 1]],
            color="0.4", linestyle=":", lw=1, zorder=3)
    ax.scatter(P[:, 0], P[:, 1], s=33, color="black", zorder=5)
    ax.scatter(*P0, s=33, color="black", zorder=5)


def label(ax, text, position, color="black", size=30, **kwargs):
    ax.text(*position, "$" + text + "$", color=color, fontsize=size,
            ha="center", va="center", zorder=6, **kwargs)


def leader(ax, text, position, target, color="black", dashed=False, size=29):
    ax.annotate(
        "$" + text + "$", xy=target, xytext=position,
        ha="center", va="center", color=color, fontsize=size, zorder=6,
        arrowprops=dict(
            arrowstyle="-" if dashed else "->", color=color,
            linestyle=(0, (4, 3)) if dashed else "-", lw=1.1,
            shrinkA=5, shrinkB=5,
        ),
    )


def circle_labels(ax, endpoint_panel=False):
    label(ax, r"\mathcal{O}_{1,2}", (266, 296), CIRCLE_COLOR)
    label(ax, r"\mathcal{O}_{2,3}", (230, 455), CIRCLE_COLOR)
    position = (354, 464) if endpoint_panel else (350, 446)
    label(ax, r"\mathcal{O}^{\mathrm{park}}_{1}", position, CIRCLE_COLOR)
    label(ax, r"\mathcal{O}^{\mathrm{park}}_{2}", (539, 674), CIRCLE_COLOR)


def primitive_panel(ax):
    circle_labels(ax)
    for j, position in enumerate(((66, 204), (120, 647),
                                   (497, 587), (428, 324)), start=1):
        label(ax, r"\boldsymbol{\ell}_{%d}" % j, position, CENTERLINE_COLOR)
    leader(ax, r"\boldsymbol{p}_0", (398, 122), P0)
    annotations = (
        (r"S_1", (246, 117), (286, 203)),
        (r"C_2^{+}", (94, 122), (137, 220)),
        (r"S_3", (32, 403), (119, 403)),
        (r"C_4^{+}", (34, 606), (137, 570)),
        (r"S_5", (287, 681), (287, 587)),
        (r"C_6^{+}", (312, 512), (420, 581)),
        (r"C_7^{b-}", (516, 516), (433, 581)),
        (r"S_8", (517, 428), (430, 442)),
    )
    for text, position, target in annotations:
        leader(ax, text, position, target, PRIMITIVE_COLOR, dashed=True)


def endpoint_panel(ax):
    circle_labels(ax, endpoint_panel=True)
    leader(ax, r"\boldsymbol{p}_0", (395, 121), P0)
    positions = (
        (394, 284), (176, 126), (8, 267), (8, 522), (139, 678),
        (313, 660), (549, 489), (343, 721), (503, 403),
    )
    for i, position in enumerate(positions):
        if i == 0:
            text = r"\boldsymbol{p}_1^s"
        elif i == 8:
            text = r"\boldsymbol{p}_8^e"
        else:
            text = r"\boldsymbol{p}_{%d}^e=\boldsymbol{p}_{%d}^s" % (i, i + 1)
        leader(ax, text, position, P[i], size=28)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=FIGURES / "path_example_notation.pdf",
        help="Destination PDF file.",
    )
    parser.add_argument("--preview", action="store_true",
                        help="Also save a PNG preview next to the PDF.")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 8.3))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-65, 620)
        ax.set_ylim(745, 0)
        ax.axis("off")
        draw_environment(ax)
        draw_path(ax)
    primitive_panel(axes[0])
    endpoint_panel(axes[1])
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.995,
                        wspace=0.01)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(args.output.resolve())
    if args.preview:
        preview = args.output.with_suffix(".png")
        fig.savefig(preview, dpi=150, bbox_inches="tight", pad_inches=0.05)
        print(preview.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
