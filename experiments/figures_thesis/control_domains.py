from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


def setup_axis(ax, title: str) -> None:
    """Apply common formatting to a control-domain axis."""
    ax.set_title(title)
    ax.set_xlabel(r"$v$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.grid(True, linewidth=0.5, alpha=0.5)


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig, axs = plt.subplots(
    2,
    2,
    figsize=(9, 8),
    constrained_layout=True,
)

# ---------------------------------------------------------------------
# 1) Forward-only unicycle
# ---------------------------------------------------------------------
ax = axs[0, 0]
setup_axis(ax, "Forward-only unicycle")

ax.add_patch(
    Rectangle(
        (0, -1),
        width=1,
        height=2,
        alpha=0.3,
        edgecolor="black",
    )
)

ax.text(
    0.5,
    0,
    r"$v\in[0,1]$" "\n" r"$\omega\in[-1,1]$",
    ha="center",
    va="center",
)

# ---------------------------------------------------------------------
# 2) Forward-only bicycle and Dubins
# ---------------------------------------------------------------------
ax = axs[0, 1]
setup_axis(ax, "Forward-only bicycle and Dubins")

ax.add_patch(
    Polygon(
        [(0, 0), (1, 1), (1, -1)],
        closed=True,
        alpha=0.3,
        edgecolor="black",
        label="Forward-only bicycle",
    )
)

ax.plot(
    [1, 1],
    [-1, 1],
    linewidth=4,
    label="Dubins",
)

ax.text(
    0.52,
    0,
    r"$|\omega|\leq v$",
    ha="center",
    va="center",
)

ax.legend(loc="upper left")

# ---------------------------------------------------------------------
# 3) Reversible bicycle and Reeds--Shepp
# ---------------------------------------------------------------------
ax = axs[1, 0]
setup_axis(ax, "Reversible bicycle and Reeds--Shepp")

ax.add_patch(
    Polygon(
        [(-1, -1), (0, 0), (-1, 1)],
        closed=True,
        alpha=0.3,
        edgecolor="black",
        label="Reversible bicycle",
    )
)

ax.add_patch(
    Polygon(
        [(1, -1), (0, 0), (1, 1)],
        closed=True,
        alpha=0.3,
        edgecolor="black",
    )
)

rs_line, = ax.plot(
    [1, 1],
    [-1, 1],
    linewidth=4,
    label="Reeds--Shepp",
)

ax.plot(
    [-1, -1],
    [-1, 1],
    linewidth=4,
    color=rs_line.get_color(),
)

ax.text(
    0,
    -0.2,
    r"$|\omega|\leq |v|$",
    ha="center",
    va="center",
)

ax.legend(loc="upper center")

# ---------------------------------------------------------------------
# 4) Differential drive
# ---------------------------------------------------------------------
ax = axs[1, 1]
setup_axis(ax, "Differential drive")

ax.add_patch(
    Polygon(
        [(1, 0), (0, 1), (-1, 0), (0, -1)],
        closed=True,
        alpha=0.3,
        edgecolor="black",
    )
)

ax.text(
    0,
    0,
    r"$|v|+|\bar{\omega}|\leq 1$",
    ha="center",
    va="center",
)

ax.set_ylabel(r"$\bar{\omega}$")

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
output_dir = Path("saved_figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "control_domains_comparison.pdf",
    bbox_inches="tight",
)
fig.savefig(
    output_dir / "control_domains_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()