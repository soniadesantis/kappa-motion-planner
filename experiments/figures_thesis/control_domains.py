from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
SYMBOL_FONT_SIZE = 20
PANEL_LABEL_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 18
PANEL_LABEL_PAD = 12
AXES_LINEWIDTH = 0.8
AXIS_CROSS_LINEWIDTH = 0.85
GRID_LINEWIDTH = 0.45
PATCH_BOUNDARY_LINEWIDTH = 1.2
SPECIAL_LINEWIDTH = 3.0
TICK_LENGTH = 3

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": SYMBOL_FONT_SIZE,
        "axes.titlesize": PANEL_LABEL_FONT_SIZE,
        "axes.labelsize": SYMBOL_FONT_SIZE,
        "legend.fontsize": SYMBOL_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "axes.linewidth": AXES_LINEWIDTH,
    }
)


# ---------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------
fill_color = "0.86"
boundary_color = "0.35"
special_line_color = "black"


# ---------------------------------------------------------------------
# Common axis formatting
# ---------------------------------------------------------------------
def setup_axis(ax, title: str) -> None:
    """Apply common formatting to a control-domain axis."""
    ax.set_title(
        title,
        pad=PANEL_LABEL_PAD,
    )

    # Individual labels with limited separation from the axes
    ax.set_xlabel(
        r"$v$",
        labelpad=1,
    )

    ax.set_ylabel(
        r"$\omega$",
        labelpad=1,
    )
    ax.yaxis.set_label_coords(-0.07, 0.5)

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax.tick_params(
        axis="both",
        which="major",
        pad=2,
        length=TICK_LENGTH,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # Light grid behind the admissible domains
    ax.grid(
        True,
        color="0.75",
        linewidth=GRID_LINEWIDTH,
        alpha=0.40,
        zorder=0,
    )

    # Coordinate axes above the filled domains
    ax.axhline(
        0,
        color="black",
        linewidth=AXIS_CROSS_LINEWIDTH,
        zorder=6,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=AXIS_CROSS_LINEWIDTH,
        zorder=6,
    )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axs = plt.subplots(
    1,
    4,
    figsize=(15.6, 4.8),
    sharex=True,
    sharey=True,
)

fig.subplots_adjust(
    left=0.050,
    right=0.995,
    bottom=0.16,
    top=0.88,
    wspace=0.16,
)

for ax in axs:
    ax.tick_params(labelleft=True)


# ---------------------------------------------------------------------
# 1) Forward-only unicycle
# ---------------------------------------------------------------------
ax = axs[0]

setup_axis(
    ax,
    "(a) Forward-only unicycle",
)

ax.add_patch(
    Rectangle(
        (0, -1),
        width=1,
        height=2,
        facecolor=fill_color,
        edgecolor=boundary_color,
        linewidth=PATCH_BOUNDARY_LINEWIDTH,
        zorder=2,
    )
)


# ---------------------------------------------------------------------
# 2) Forward-only bicycle and Dubins
# ---------------------------------------------------------------------
ax = axs[1]

setup_axis(
    ax,
    "(b) Forward-only bicycle",
)

ax.add_patch(
    Polygon(
        [
            (0, 0),
            (1, 1),
            (1, -1),
        ],
        closed=True,
        facecolor=fill_color,
        edgecolor=boundary_color,
        linewidth=PATCH_BOUNDARY_LINEWIDTH,
        zorder=2,
    )
)

# Dubins domain: v = 1
ax.plot(
    [1, 1],
    [-1, 1],
    color=special_line_color,
    linewidth=SPECIAL_LINEWIDTH,
    label="Dubins",
    zorder=4,
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.0,
    frameon=True,
    framealpha=0.90,
    fontsize=LEGEND_FONT_SIZE,
    borderpad=0.25,
    labelspacing=0.25,
    handlelength=1.45,
    handletextpad=0.45,
)


# ---------------------------------------------------------------------
# 3) Reversible bicycle and Reeds--Shepp
# ---------------------------------------------------------------------
ax = axs[2]

setup_axis(
    ax,
    "(c) Reversible bicycle",
)

ax.add_patch(
    Polygon(
        [
            (-1, -1),
            (0, 0),
            (-1, 1),
        ],
        closed=True,
        facecolor=fill_color,
        edgecolor=boundary_color,
        linewidth=PATCH_BOUNDARY_LINEWIDTH,
        zorder=2,
    )
)

ax.add_patch(
    Polygon(
        [
            (1, -1),
            (0, 0),
            (1, 1),
        ],
        closed=True,
        facecolor=fill_color,
        edgecolor=boundary_color,
        linewidth=PATCH_BOUNDARY_LINEWIDTH,
        zorder=2,
    )
)

# Reeds--Shepp domain: v = -1 and v = 1
rs_line, = ax.plot(
    [-1, -1],
    [-1, 1],
    color=special_line_color,
    linewidth=SPECIAL_LINEWIDTH,
    label="Reeds-Shepp",
    zorder=4,
)

ax.plot(
    [1, 1],
    [-1, 1],
    color=rs_line.get_color(),
    linewidth=SPECIAL_LINEWIDTH,
    zorder=4,
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.50, 0.98),
    borderaxespad=0.0,
    frameon=True,
    framealpha=0.90,
    fontsize=LEGEND_FONT_SIZE,
    borderpad=0.25,
    labelspacing=0.25,
    handlelength=1.45,
    handletextpad=0.45,
)


# ---------------------------------------------------------------------
# 4) Differential drive
# ---------------------------------------------------------------------
ax = axs[3]

setup_axis(
    ax,
    "(d) Differential drive",
)

# With |v_R| <= 1, |v_L| <= 1, and b = 2,
#
#     |v| + (b/2)|omega| <= 1
#
# becomes
#
#     |v| + |omega| <= 1.
omega_max = 1.0

ax.add_patch(
    Polygon(
        [
            (1, 0),
            (0, omega_max),
            (-1, 0),
            (0, -omega_max),
        ],
        closed=True,
        facecolor=fill_color,
        edgecolor=boundary_color,
        linewidth=PATCH_BOUNDARY_LINEWIDTH,
        zorder=2,
    )
)

# Differential-drive angular-velocity extrema
ax.text(
    0.24,
    1.0,
    r"$(0,\,2/b)$",
    ha="left",
    va="center",
    fontsize=SYMBOL_FONT_SIZE,
    zorder=7,
)

ax.plot(
    [0.0, 0.16],
    [omega_max, 1.0],
    color="black",
    linewidth=1.0,
    zorder=6,
)

ax.text(
    0.24,
    -1.0,
    r"$(0,\,-2/b)$",
    ha="left",
    va="center",
    fontsize=SYMBOL_FONT_SIZE,
    zorder=7,
)

ax.plot(
    [0.0, 0.16],
    [-omega_max, -1.0],
    color="black",
    linewidth=1.0,
    zorder=6,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    fig_dir / "control_domains_comparison.pdf",
    bbox_inches="tight",
)

fig.savefig(
    fig_dir / "control_domains_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
