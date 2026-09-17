"""Single-panel common-tangent construction for the journal paper.

Each tangent has its own color. Legends explain the tangent signs and the
point, distance, and angle notation. Run with --no-show for headless export.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Patch, Wedge
import numpy as np

CENTER_1 = np.array([-2.1, -0.35])
CENTER_2 = np.array([2.15, 1.25])
RADIUS = 1.15
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures"
COLORS = {("-", "-"): "#0072B2", ("-", "+"): "#D55E00",
          ("+", "-"): "#009E73", ("+", "+"): "#AA4499"}
ANGLE_FRACTIONS = {("-", "-"): 0.45, ("-", "+"): 0.25,
                   ("+", "-"): 0.61, ("+", "+"): 0.60}
STYLE = {
    "text.usetex": True, "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 16, "legend.fontsize": 14,
    "text.latex.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage[notext]{stix}
\usepackage{DejaVuSerif}
\let\mathtextoriginal\text
\renewcommand{\text}[1]{\mathtextoriginal{\fontfamily{cmr}\selectfont #1}}
""",
}


def common_tangents(center_1, center_2, radius):
    displacement = center_2 - center_1
    squared_distance = np.dot(displacement, displacement)
    if squared_distance <= 4 * radius**2:
        raise ValueError("Four distinct common tangents require c > 2R.")
    perpendicular = np.array([-displacement[1], displacement[0]])
    tangents = {}
    for signed_radius in (radius, -radius):
        difference = radius - signed_radius
        height = np.sqrt(squared_distance - difference**2)
        for branch in (-1, 1):
            normal = (displacement * difference + perpendicular * height * branch) / squared_distance
            q1, q2 = center_1 + radius * normal, center_2 + signed_radius * normal
            signs = tuple("+" if np.dot(perpendicular, q - center) < 0 else "-"
                          for q, center in ((q1, center_1), (q2, center_2)))
            tangents[signs] = (q1, q2)
    return tangents


def draw_angle(axis, vertex, direction, color, radius=0.25):
    angle = np.arctan2(direction[1], direction[0]) % (2 * np.pi)
    axis.add_patch(Wedge(vertex, radius, 0, np.degrees(angle),
                         facecolor=color, edgecolor="none", alpha=0.2, zorder=3))
    axis.add_patch(Arc(vertex, 2 * radius, 2 * radius, theta1=0,
                       theta2=np.degrees(angle), color=color, lw=1.1, zorder=4))
    axis.plot([vertex[0], vertex[0] + radius + 0.08], [vertex[1], vertex[1]],
              color="0.5", lw=0.9, zorder=4)
    start_angle = max(0, angle - min(0.16, angle / 3))
    axis.add_patch(FancyArrowPatch(
        vertex + radius * np.array([np.cos(start_angle), np.sin(start_angle)]),
        vertex + radius * np.array([np.cos(angle), np.sin(angle)]),
        arrowstyle="->", mutation_scale=8, color=color, lw=1,
        shrinkA=0, shrinkB=0, zorder=5))


def create_figure():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.84)
        for index, (center, label_angle) in enumerate(((CENTER_1, 200), (CENTER_2, 25)), 1):
            ax.add_patch(Circle(center, RADIUS, fill=False, color="0.15", lw=1.7, zorder=2))
            ax.plot(*center, "o", color="0.15", ms=5, zorder=7)
            ax.text(center[0], center[1] + 0.34, rf"$\mathbf{{p}}^{{o}}_{index}$",
                    fontsize=21, ha="center", va="center")
            direction = np.array([np.cos(np.radians(label_angle)), np.sin(np.radians(label_angle))])
            ax.text(*(center + 0.62 * RADIUS * direction),
                    rf"$\mathcal{{O}}_{index}$", fontsize=24,
                    ha="center", va="center")
        ax.add_patch(FancyArrowPatch(CENTER_1, CENTER_2, arrowstyle="<|-|>",
                                    color="0.45", lw=1.1, mutation_scale=11, zorder=1))
        center_displacement = CENTER_2 - CENTER_1
        center_normal = np.array([-center_displacement[1], center_displacement[0]])
        center_normal /= np.linalg.norm(center_normal)
        ax.text(*(CENTER_1 + 0.36 * center_displacement - 0.10 * center_normal),
                r"$c$", color="0.45", fontsize=18, ha="center", va="center", zorder=8)
        for signs, (q1, q2) in common_tangents(CENTER_1, CENTER_2, RADIUS).items():
            color = COLORS[signs]
            direction = (q2 - q1) / np.linalg.norm(q2 - q1)
            a, b = q1 - 1.0 * direction, q2 + 1.0 * direction
            ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=color, lw=0.9, alpha=0.6, zorder=0)
            ax.add_patch(FancyArrowPatch(q1, q2, arrowstyle="<|-|>", color=color,
                                        lw=1.7, mutation_scale=12, shrinkA=0, shrinkB=0, zorder=4))
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], linestyle="None", marker="o",
                    ms=7, color=color, markeredgecolor="white", markeredgewidth=0.7, zorder=6)
            angle_vertex = q1 + ANGLE_FRACTIONS[signs] * (q2 - q1)
            sector_radius = 0.20 if signs == ("-", "+") else 0.25
            draw_angle(ax, angle_vertex, direction, color, radius=sector_radius)
            normal = np.array([-direction[1], direction[0]])
            distance_placements = {
                ("-", "-"): (0.74, 0.14), ("+", "+"): (0.30, -0.15),
                ("-", "+"): (0.68, -0.15), ("+", "-"): (0.28, -0.16),
            }
            fraction, offset = distance_placements[signs]
            label_position = q1 + fraction * (q2 - q1) + offset * normal
            superscript = ",".join(signs)
            ax.text(*label_position, rf"$d^{{{superscript}}}$", color=color,
                    fontsize=18, ha="center", va="center", zorder=8,
                    rotation=np.degrees(np.arctan2(direction[1], direction[0])),
                    rotation_mode="anchor")
            sector_angle = np.arctan2(direction[1], direction[0]) % (2 * np.pi)
            label_angle = np.radians(30) if signs == ("-", "+") else sector_angle / 2
            bisector = np.array([np.cos(label_angle), np.sin(label_angle)])
            ax.text(*(angle_vertex + (sector_radius + 0.27) * bisector),
                    rf"$\alpha^{{{superscript}}}$", color=color, fontsize=16,
                    ha="center", va="center", zorder=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
        ax.set(xlim=(-4.0, 4.15), ylim=(-2.10, 2.80), aspect="equal")
        ax.axis("off")
        legend = ax.legend(handles=[Line2D([], [], color=color, lw=1.5, linestyle="--",
                                   label=rf"$t^{{{s1},{s2}}}$")
                            for (s1, s2), color in COLORS.items()],
                   loc="lower right", bbox_to_anchor=(0.98, 0.05), ncol=2,
                   frameon=True, edgecolor="0.8", facecolor="white", framealpha=1,
                   fancybox=False, columnspacing=1.2, handlelength=1.5, fontsize=21)
        legend.get_frame().set_linewidth(0.6)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        output = OUTPUT_DIRECTORY / f"tangent_convention_journal.{extension}"
        with plt.rc_context(STYLE):
            figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
