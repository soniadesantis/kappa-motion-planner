"""Reproduce ctrl_profiles_new.pdf as an editable control-profile schematic.

Run: python experiments/avp_paper/reproduce_ctrl_profiles.py

The reference contains symbolic levels and no numerical time scale. Values below
are normalized schematic values, not recovered experimental measurements. The
primitive sequence, plateaus, signs, and phase transitions follow the reference;
velocity is integrated from acceleration to keep the two panels consistent.
Requires NumPy, Matplotlib, and LaTeX (amsmath and bm).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import blended_transform_factory

FIGURES = Path(__file__).resolve().parent / "figures"
# Acceleration on each interval [TIMES[i], TIMES[i+1]).
TIMES = np.array([0, .8, 1.15, 1.35, 2.35, 2.55, 3.35, 3.55,
                  4.55, 4.75, 5.15, 5.75, 6.6, 8.6, 10.6, 13.1, 14.1])
ACCELERATION = np.array([5, 0, -5, 0, 5, 0, -5, 0, 5, 0, -5, 0,
                         -1, 1, 0, -1])
BOUNDARIES = np.array([0, 1.35, 2.35, 3.55, 4.55, 5.75, 7.6, 9.6, 14.1])
STEERING = np.array([0, 1, 0, 1, 0, 1, -1, 0])
PRIMITIVES = [r"S_1", r"C_2^{+}", r"S_3", r"C_4^{+}", r"S_5",
              r"C_6^{+}", r"C_7^{b-}", r"S_8"]
PHASE_COLOR = "#7293B3"
STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 13,
    "axes.labelsize": 15,
    "ytick.labelsize": 16,
    "axes.linewidth": .65,
    "axes.edgecolor": "0.45",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def brace(ax, left, right, text):
    """Draw an upward brace using time coordinates and axes-relative height."""
    center = (left + right) / 2
    shoulder = min(.28, (right - left) / 8)
    bottom, baseline, peak = 1.31, 1.41, 1.52
    vertices = [
        (left, bottom), (left, baseline), (left + shoulder, baseline),
        (left + shoulder, baseline),
        (center - shoulder, baseline),
        (center - shoulder, baseline), (center, baseline), (center, peak),
        (center, baseline), (center + shoulder, baseline),
        (center + shoulder, baseline),
        (right - shoulder, baseline),
        (right - shoulder, baseline), (right, baseline), (right, bottom),
    ]
    codes = [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * 3
    codes += [DrawingPath.LINETO] + [DrawingPath.CURVE4] * 6
    codes += [DrawingPath.LINETO] + [DrawingPath.CURVE4] * 3
    ax.add_patch(PathPatch(DrawingPath(vertices, codes), fill=False,
                           lw=.9, color="0.2", clip_on=False,
                           transform=ax.get_xaxis_transform()))
    ax.text(center, 1.55, text, ha="center", va="bottom", fontsize=16,
            transform=ax.get_xaxis_transform())


def create_figure():
    velocity = np.r_[0, np.cumsum(np.diff(TIMES) * ACCELERATION)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.6, 7.3))
        fig.subplots_adjust(left=.16, right=.985, bottom=.09, top=.77,
                            hspace=.25)
        levels = [
            ([-1, 0, 1, 3, 4],
             [r"$-v_{\mathrm{park}}$", "$0$", r"$v_{\mathrm{park}}$",
              r"$v_{\mathrm{nav}}^{C}$", r"$v_{\mathrm{nav}}^{S}$"],
             (-1.28, 4.35), "Velocity"),
            ([-5, -1, 0, 1, 5],
             [r"$-a_{\mathrm{nav}}$", r"$-a_{\mathrm{park}}$", "$0$",
              r"$a_{\mathrm{park}}$", r"$a_{\mathrm{nav}}$"],
             (-5.65, 5.65), "Acceleration"),
            ([-1, 0, 1], [r"$-\bar\delta$", "$0$", r"$\bar\delta$"],
             (-1.22, 1.22), "Steering angle"),
        ]
        for ax, (ticks, labels, limits, title) in zip(axes, levels):
            ax.set_ylim(*limits)
            ax.set_yticks(ticks, labels)
            for tick_label in ax.get_yticklabels():
                tick_label.set_verticalalignment("center")
            ax.tick_params(axis="both", which="both", length=0, pad=5)
            ax.set_ylabel(title, labelpad=5)
            for level in ticks:
                ax.axhline(level, color="0.65" if level else "0.45",
                           lw=.65, ls=(0, (4, 4)) if level else (0, (2, 2)),
                           zorder=0)
            for boundary in BOUNDARIES[1:-1]:
                if boundary == BOUNDARIES[4]:
                    continue  # One continuous dashed divider is drawn below.
                ax.axvline(boundary, color=PHASE_COLOR, lw=.85, alpha=.8,
                           zorder=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-.2, TIMES[-1] + .2)
            ax.set_xticks([])

        fig.align_ylabels(axes)

        # Slight offsets keep the central acceleration labels readable in
        # the compact layout without requiring a disproportionately tall panel.
        acceleration_labels = axes[1].get_yticklabels()
        acceleration_labels[1].set_verticalalignment("top")
        acceleration_labels[3].set_verticalalignment("bottom")

        line_style = dict(color="black", lw=2.0, zorder=3,
                          solid_capstyle="butt", solid_joinstyle="miter")
        axes[0].plot(TIMES, velocity, **line_style)
        axes[1].step(TIMES, np.r_[ACCELERATION, 0], where="post", **line_style)
        axes[2].step(BOUNDARIES, np.r_[STEERING, STEERING[-1]],
                     where="post", **line_style)
        axes[2].set_xlabel("Time", labelpad=10)
        for left, right, text in zip(BOUNDARIES[:-1], BOUNDARIES[1:], PRIMITIVES):
            axes[0].text((left + right) / 2, 1.08, "$" + text + "$",
                         ha="center", va="bottom", fontsize=16,
                         transform=axes[0].get_xaxis_transform())
        for boundary in BOUNDARIES[1:-1]:
            if boundary == BOUNDARIES[4]:
                continue
            axes[0].plot([boundary, boundary], [1, 1.28], color=PHASE_COLOR,
                         lw=.85, alpha=.8, clip_on=False,
                         transform=axes[0].get_xaxis_transform())
        # Span all panels and their gaps at the C4 / S5 transition.
        top_panel = axes[0].get_position()
        fig.add_artist(Line2D(
            [BOUNDARIES[4], BOUNDARIES[4]],
            [axes[-1].get_position().y0, top_panel.y1 + .28 * top_panel.height],
            transform=blended_transform_factory(axes[0].transData, fig.transFigure),
            color="#475569", lw=1.1, linestyle=(0, (5, 4)), alpha=.85,
        ))
        brace(axes[0], BOUNDARIES[0], BOUNDARIES[4], r"$\bm{\mathcal S}_{\mathrm{nav}}$")
        brace(axes[0], BOUNDARIES[4], BOUNDARIES[-1], r"$\bm{\mathcal S}_{\mathrm{park}}$")
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=FIGURES / "ctrl_profiles_clean.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = create_figure()
        for output in (args.output, args.output.with_suffix(".png")):
            fig.savefig(output, dpi=250, bbox_inches="tight", pad_inches=.08)
            print(output.resolve())
        plt.close(fig)


if __name__ == "__main__":
    main()
