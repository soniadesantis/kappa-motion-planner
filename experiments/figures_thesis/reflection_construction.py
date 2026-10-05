"""Single-panel reflection geometry, styled to match the thesis tangent figure.

Run: python experiments/figures_thesis/reflection_construction.py --no-show
Requires NumPy, Matplotlib, and a LaTeX installation with amsmath and bm.
Exports a vector PDF and a 300 dpi PNG beside this script by default.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D


# Editable geometry and appearance (all positions are in data coordinates).
FIGURE_SIZE = (6.8, 5.2)
FONT_SIZE = 20
RADIUS = 1.05
POINT = np.array([-1.95, 2.05])
CENTER = np.array([-1.95, -0.65])
AXIS_X = 0.0
# Clockwise tilt from vertical; change this to adjust the reflection axis.
AXIS_TILT_DEG = 30.0
ORIGINAL_COLOR = "black"
REFLECTED_COLOR = "black"
CONSTRUCTION_COLOR = "black"
MIDPOINT_COLOR = "#0072B2"
MIDPOINT_MARKER_SIZE = 7.0
OUTPUT_NAME = "geometric_reflection_clean"
STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": FONT_SIZE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


def reflect(point):
    """Reflect a point across the vertical line x = AXIS_X."""
    return np.array([2 * AXIS_X - point[0], point[1]])


def distance_ticks(ax, start, end, count=1, fraction=0.5):
    """Mark congruent segments with short transverse strokes."""
    tangent = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-tangent[1], tangent[0]])
    midpoint = start + fraction * (end - start)
    for offset in (np.arange(count) - (count - 1) / 2) * 0.075:
        location = midpoint + offset * tangent
        ends = np.array([location - 0.085 * normal,
                         location + 0.085 * normal])
        ax.plot(*ends.T, color=CONSTRUCTION_COLOR, lw=1.0, zorder=3)


def perpendicular_pair(ax, original, count):
    reflected = reflect(original)
    midpoint = (original + reflected) / 2
    ax.plot([original[0], reflected[0]], [original[1], reflected[1]],
            color=CONSTRUCTION_COLOR, lw=1.05, zorder=1)
    fraction = 0.72 if count == 2 else 0.5
    distance_ticks(ax, original, midpoint, count, fraction)
    distance_ticks(ax, midpoint, reflected, count, 1 - fraction)
    # Small square in the upper-right quadrant at the perpendicular foot.
    size = 0.18
    corner = midpoint + np.array([[0, size], [size, size], [size, 0]])
    ax.plot(*corner.T, color=CONSTRUCTION_COLOR, lw=0.85, zorder=2)
    ax.plot(*midpoint, "o", color=MIDPOINT_COLOR,
            ms=MIDPOINT_MARKER_SIZE, zorder=5)
    return reflected, midpoint


def label(ax, xy, text, offset=(0, 0), **kwargs):
    ax.annotate(text, xy, xytext=offset, textcoords="offset points",
                ha="center", va="center", **kwargs)


def create_figure():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_xlim(-3.35, 3.35)
        ax.set_ylim(-2.0, 2.95)

        ax.plot([AXIS_X, AXIS_X], [-1.95, 2.7], color="black",
                lw=1.1, linestyle=(0, (6, 4)))
        label(ax, (AXIS_X, 2.7), r"$t$", offset=(12, 0))

        reflected_point, midpoint = perpendicular_pair(ax, POINT, count=1)
        for point, text, color in (
            (POINT, r"$\bm p$", ORIGINAL_COLOR),
            (reflected_point, r"$\tilde{\bm p}$", REFLECTED_COLOR),
        ):
            ax.plot(*point, "o", color=color, ms=5.8, zorder=5)
            label(ax, point, text, offset=(0, 17))
        label(ax, midpoint, r"$\bm m_p$", offset=(16, -17),
              color=MIDPOINT_COLOR)

        reflected_center, center_midpoint = perpendicular_pair(ax, CENTER, count=2)
        label(ax, center_midpoint, r"$\bm m_o$", offset=(16, -17),
              color=MIDPOINT_COLOR)
        # Radius directions are reflected too, making preservation visible.
        angle = np.deg2rad(52)
        for center, direction, color, circle_text, center_text in (
            (CENTER, np.array([-np.cos(angle), np.sin(angle)]), ORIGINAL_COLOR,
             r"$\mathcal O$", r"$\bm o$"),
            (reflected_center, np.array([np.cos(angle), np.sin(angle)]),
             REFLECTED_COLOR, r"$\tilde{\mathcal O}$", r"$\tilde{\bm o}$"),
        ):
            ax.add_patch(Circle(center, RADIUS, fill=False, ec="black",
                                lw=1.6, zorder=2))
            end = center + RADIUS * direction
            ax.plot([center[0], end[0]], [center[1], end[1]],
                    color=color, lw=1.35, zorder=3)
            ax.plot(*center, "o", color=color, ms=5.8, zorder=5)
            side = np.sign(direction[0])
            label(ax, center + 0.59 * RADIUS * direction, r"$R$",
                  offset=(-side * 10, 5), color=color)
            label(ax, center, center_text, offset=(0, -17))
            label(ax, center + np.array([0, RADIUS]), circle_text,
                  offset=(0, 12))
        # Rotate the entire construction rigidly, preserving reflection,
        # perpendicularity, equal distances, and circular radii. Text stays upright.
        rotation = Affine2D().rotate_deg(-AXIS_TILT_DEG)
        geometry_transform = rotation + ax.transData
        for artist in [*ax.lines, *ax.patches]:
            artist.set_transform(geometry_transform)
        for annotation in ax.texts:
            annotation.xycoords = geometry_transform
            annotation.set_position(rotation.transform(annotation.get_position()))

        # Fit the rotated geometry, including space for labels.
        centers = rotation.transform([CENTER, reflected_center])
        endpoints = rotation.transform([
            POINT, reflected_point, [AXIS_X, -1.95], [AXIS_X, 2.7],
        ])
        lower = np.minimum(centers.min(axis=0) - RADIUS, endpoints.min(axis=0))
        upper = np.maximum(centers.max(axis=0) + RADIUS, endpoints.max(axis=0))
        ax.set_xlim(lower[0] - 0.4, upper[0] + 0.4)
        ax.set_ylim(lower[1] - 0.3, upper[1] + 0.5)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    with plt.rc_context(STYLE):
        for extension in ("pdf", "png"):
            output = args.output_dir / f"{OUTPUT_NAME}.{extension}"
            figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved {output}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
