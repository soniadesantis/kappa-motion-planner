from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
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
        pad=5,
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

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax.tick_params(
        axis="both",
        which="major",
        pad=2,
        length=3,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # Light grid behind the admissible domains
    ax.grid(
        True,
        color="0.75",
        linewidth=0.45,
        alpha=0.40,
        zorder=0,
    )

    # Coordinate axes above the filled domains
    ax.axhline(
        0,
        color="black",
        linewidth=0.85,
        zorder=6,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.85,
        zorder=6,
    )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axs = plt.subplots(
    1,
    4,
    figsize=(13.2, 3.65),
    sharex=True,
    sharey=True,
)

fig.subplots_adjust(
    left=0.040,
    right=0.995,
    bottom=0.19,
    top=0.84,
    wspace=0.035,
)


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
        linewidth=1.2,
        zorder=2,
    )
)


# ---------------------------------------------------------------------
# 2) Forward-only bicycle and Dubins
# ---------------------------------------------------------------------
ax = axs[1]

setup_axis(
    ax,
    "(b) Bicycle and Dubins",
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
        linewidth=1.2,
        zorder=2,
    )
)

# Dubins domain: v = 1
ax.plot(
    [1, 1],
    [-1, 1],
    color=special_line_color,
    linewidth=3.0,
    label="Dubins",
    zorder=4,
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.0,
    frameon=True,
    framealpha=0.90,
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
    "(c) Bicycle and Reeds--Shepp",
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
        linewidth=1.2,
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
        linewidth=1.2,
        zorder=2,
    )
)

# Reeds--Shepp domain: v = -1 and v = 1
rs_line, = ax.plot(
    [-1, -1],
    [-1, 1],
    color=special_line_color,
    linewidth=3.0,
    label="Reeds--Shepp",
    zorder=4,
)

ax.plot(
    [1, 1],
    [-1, 1],
    color=rs_line.get_color(),
    linewidth=3.0,
    zorder=4,
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.50, 0.98),
    borderaxespad=0.0,
    frameon=True,
    framealpha=0.90,
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
    r"(d) Differential drive, $b=2$",
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
        linewidth=1.2,
        zorder=2,
    )
)

# Differential-drive angular-velocity extrema
ax.text(
    0.05,
    omega_max - 0.04,
    r"$2/b$",
    ha="left",
    va="top",
    fontsize=11,
    zorder=7,
)

ax.text(
    0.05,
    -omega_max + 0.04,
    r"$-2/b$",
    ha="left",
    va="bottom",
    fontsize=11,
    zorder=7,
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