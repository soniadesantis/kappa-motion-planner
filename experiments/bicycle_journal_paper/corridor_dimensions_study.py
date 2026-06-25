import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# ============================================================
# Parameters
# ============================================================
R = 1.0
r = 0.2

S = R + r      # inflated / swept radius
D = R - r      # allowed distance of intermediate-circle center from corner
q = D / np.sqrt(2)

w_min = 2 * r
w_45 = S - q

w_plot_max = 1.8

print(f"R = {R}")
print(f"r = {r}")
print(f"S = R + r = {S:.3f}")
print(f"D = R - r = {D:.3f}")
print(f"q = D/sqrt(2) = {q:.3f}")
print(f"45-degree threshold width = S-q = {w_45:.3f}")
print(f"Narrow corridor width = 2r = {w_min:.3f}")
print(f"Companion width in narrow case = S = R+r = {S:.3f}")


# ============================================================
# Basic feasibility functions
# ============================================================
def lower_bounds(w1, w2):
    """
    Feasibility lower bounds for the Intermediate Circle center:
        x >= a = max(0, S - w1)
        y >= b = max(0, S - w2)
    """
    a = max(0.0, S - w1)
    b = max(0.0, S - w2)
    return a, b


def centerline_half_bounds(w1, w2):
    """
    Lower bounds for the swept circle to remain entirely on the preferred
    half-side of both corridor centerlines.

    Corridor 1 centerline: x = -w1/2
    Corridor 2 centerline: y = -w2/2

    Swept circle radius is S.

    Need:
        Cx - S >= -w1/2  ->  Cx >= S - w1/2
        Cy - S >= -w2/2  ->  Cy >= S - w2/2
    """
    h1 = max(0.0, S - w1 / 2.0)
    h2 = max(0.0, S - w2 / 2.0)
    return h1, h2


def is_feasible_width_pair(w1, w2):
    """
    Some collision-free Intermediate Circle placement exists.
    """
    a, b = lower_bounds(w1, w2)
    return a**2 + b**2 <= D**2


def is_centerline_half_feasible(w1, w2):
    """
    A feasible placement exists such that the inflated circle stays completely
    on the preferred half-side of both corridor centerlines.
    """
    h1, h2 = centerline_half_bounds(w1, w2)
    return h1**2 + h2**2 <= D**2


def minimum_w2_for_w1(w1):
    """
    Feasibility boundary:
        max(0,S-w1)^2 + max(0,S-w2)^2 = D^2

    Returns the minimum w2 needed for a given w1.
    """
    a = max(0.0, S - w1)

    if a > D:
        return np.nan

    remaining = D**2 - a**2
    b_max = np.sqrt(max(0.0, remaining))

    # Need max(0,S-w2) <= b_max
    w2_required = S - b_max

    return max(w_min, w2_required)


def minimum_w2_centerline_half_for_w1(w1):
    """
    Boundary for existence of a placement completely on the preferred
    half-side of both centerlines:
        max(0,S-w1/2)^2 + max(0,S-w2/2)^2 = D^2

    Returns the minimum w2 needed for a given w1.
    """
    h1 = max(0.0, S - w1 / 2.0)

    if h1 > D:
        return np.nan

    remaining = D**2 - h1**2
    h2_max = np.sqrt(max(0.0, remaining))

    # Need max(0,S-w2/2) <= h2_max
    # S - w2/2 <= h2_max
    # w2 >= 2(S - h2_max)
    w2_required = 2.0 * (S - h2_max)

    return max(w_min, w2_required)


def choose_example_center(w1, w2):
    """
    Simple placement policy:
    1. use 45-degree point if feasible;
    2. else use both-centerline half-side target if feasible;
    3. else use minimal feasible point.
    """
    a, b = lower_bounds(w1, w2)

    if not is_feasible_width_pair(w1, w2):
        return None, "infeasible"

    # Candidate 1: nominal 45-degree point
    c45 = np.array([q, q])
    if c45[0] >= a and c45[1] >= b and np.dot(c45, c45) <= D**2:
        return c45, "45-degree rule"

    # Candidate 2: tangent to both centerlines
    cstar = np.array([S - w1 / 2.0, S - w2 / 2.0])
    cstar = np.maximum(cstar, 0.0)

    if (
        cstar[0] >= a
        and cstar[1] >= b
        and np.dot(cstar, cstar) <= D**2
    ):
        return cstar, "centerline half-side rule"

    # Candidate 3: minimal feasible shifted point
    cmin = np.array([a, b])
    return cmin, "minimal shifted feasible point"


# ============================================================
# Drawing helpers for local geometry figure
# ============================================================
def draw_double_arrow(ax, start, end, text, text_offset=(0, 0), fontsize=10):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="<->", linewidth=1.4),
    )
    xm = 0.5 * (start[0] + end[0]) + text_offset[0]
    ym = 0.5 * (start[1] + end[1]) + text_offset[1]
    ax.text(xm, ym, text, fontsize=fontsize, ha="center", va="center")


def draw_case(ax, w1, w2, C, title):
    """
    Draw one local geometry case.

    Local convention:
        O = (0,0) is the corner point.
        active wall 1 is x=-w1.
        active wall 2 is y=-w2.
        positive x,y point away from the active walls.
    """
    Cx, Cy = C
    a, b = lower_bounds(w1, w2)

    xmin = -max(w1, S) - 0.25
    ymin = -max(w2, S) - 0.25
    xmax = D + 0.45
    ymax = D + 0.45

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)

    # Region on the allowed side of active walls
    rect = Rectangle(
        (-w1, -w2),
        xmax + w1,
        ymax + w2,
        alpha=0.08,
    )
    ax.add_patch(rect)

    # Active walls
    ax.axvline(-w1, linestyle="-", linewidth=2)
    ax.axhline(-w2, linestyle="-", linewidth=2)
    ax.text(-w1, ymax - 0.08, "wall 1\nx=-w1", fontsize=9, ha="center", va="top")
    ax.text(xmax - 0.04, -w2, "wall 2: y=-w2", fontsize=9, ha="right", va="center")

    # Corridor centerlines
    ax.axvline(-w1 / 2.0, linestyle=":", linewidth=1.8)
    ax.axhline(-w2 / 2.0, linestyle=":", linewidth=1.8)
    ax.text(
        -w1 / 2.0,
        ymin + 0.08,
        "centerline 1",
        fontsize=8,
        ha="center",
        va="bottom",
        rotation=90,
    )
    ax.text(
        xmin + 0.08,
        -w2 / 2.0,
        "centerline 2",
        fontsize=8,
        ha="left",
        va="center",
    )

    # Local axes
    ax.annotate(
        "",
        xy=(xmax * 0.92, 0),
        xytext=(xmin * 0.12, 0),
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )
    ax.annotate(
        "",
        xy=(0, ymax * 0.92),
        xytext=(0, ymin * 0.12),
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )
    ax.text(xmax * 0.94, 0.04, "x", fontsize=12)
    ax.text(0.04, ymax * 0.94, "y", fontsize=12)

    # Corner
    ax.plot(0, 0, marker="o", markersize=6)
    ax.text(0.04, -0.08, "O corner", fontsize=10)

    # Width arrows
    draw_double_arrow(
        ax,
        (-w1, -0.12),
        (0, -0.12),
        r"$w_1$",
        text_offset=(0, -0.08),
        fontsize=10,
    )
    draw_double_arrow(
        ax,
        (-0.12, -w2),
        (-0.12, 0),
        r"$w_2$",
        text_offset=(-0.08, 0),
        fontsize=10,
    )

    # Feasible disk for C
    feasible_disk = Circle((0, 0), D, fill=False, linestyle="--", linewidth=2)
    ax.add_patch(feasible_disk)
    ax.text(
        0.34 * D,
        0.78 * D,
        "center feasible disk\nx^2+y^2 <= D^2",
        fontsize=9,
        ha="center",
    )

    # Small corner circle radius r
    small_circle = Circle((0, 0), r, fill=False, linewidth=2)
    ax.add_patch(small_circle)
    ax.text(-0.02, r + 0.04, "radius r", fontsize=9, ha="right")

    # Intermediate Circle radius R
    intermediate_circle = Circle((Cx, Cy), R, fill=False, linewidth=2)
    ax.add_patch(intermediate_circle)

    # Swept circle radius S
    swept_circle = Circle((Cx, Cy), S, fill=False, linestyle=":", linewidth=2)
    ax.add_patch(swept_circle)

    # Center C
    ax.plot(Cx, Cy, marker="*", markersize=14)
    ax.text(Cx + 0.04, Cy + 0.04, "C", fontsize=12)

    # Radius markers
    ax.plot(
        [Cx, Cx + R / np.sqrt(2)],
        [Cy, Cy + R / np.sqrt(2)],
        linewidth=1.2,
    )
    ax.text(Cx + 0.38 * R, Cy + 0.38 * R + 0.04, "R", fontsize=10)

    ax.plot([Cx, Cx - S], [Cy, Cy], linewidth=1.0)
    ax.text(Cx - 0.5 * S, Cy + 0.04, "S=R+r", fontsize=9, ha="center")

    # Wall-clearance distances
    draw_double_arrow(
        ax,
        (-w1, Cy - 0.12),
        (Cx, Cy - 0.12),
        r"$x+w_1$",
        text_offset=(0, -0.07),
        fontsize=9,
    )
    draw_double_arrow(
        ax,
        (Cx - 0.12, -w2),
        (Cx - 0.12, Cy),
        r"$y+w_2$",
        text_offset=(-0.12, 0),
        fontsize=9,
    )

    ax.text(
        xmin + 0.05,
        ymax - 0.25,
        f"w1={w1:.3f}\nw2={w2:.3f}\n"
        f"a=max(0,S-w1)={a:.3f}\n"
        f"b=max(0,S-w2)={b:.3f}",
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.text(
        xmin + 0.05,
        ymin + 0.08,
        "Wall constraints:\n"
        "x + w1 >= S\n"
        "y + w2 >= S\n\n"
        "Half-side constraints:\n"
        "x + w1/2 >= S\n"
        "y + w2/2 >= S",
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )


# ============================================================
# Figure 1: three local geometry cases
# ============================================================
w1_nominal = w_45
w2_nominal = w_45
C_nominal = (q, q)

w1_narrow1 = w_min
w2_narrow1 = S
C_narrow1 = (D, 0.0)

w1_narrow2 = S
w2_narrow2 = w_min
C_narrow2 = (0.0, D)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

draw_case(
    axs[0],
    w1_nominal,
    w2_nominal,
    C_nominal,
    "Nominal 45-degree case",
)

draw_case(
    axs[1],
    w1_narrow1,
    w2_narrow1,
    C_narrow1,
    "Narrow corridor 1",
)

draw_case(
    axs[2],
    w1_narrow2,
    w2_narrow2,
    C_narrow2,
    "Narrow corridor 2",
)

fig.suptitle(
    "Local convention for Intermediate Circle placement\n"
    "positive x and y point away from the two active limiting walls",
    fontsize=14,
)

plt.tight_layout()


# ============================================================
# Figure 2: width-space feasibility map
# ============================================================
fig, ax = plt.subplots(figsize=(9, 8))

w_values = np.linspace(w_min, w_plot_max, 900)
w2_min_values = np.array([minimum_w2_for_w1(w1) for w1 in w_values])
w2_half_values = np.array([minimum_w2_centerline_half_for_w1(w1) for w1 in w_values])

# General feasible region
ax.fill_between(
    w_values,
    w2_min_values,
    w_plot_max,
    where=~np.isnan(w2_min_values),
    alpha=0.18,
    label="some feasible arc placement exists",
)

# Region where a placement exists completely on preferred half-side of both centerlines
ax.fill_between(
    w_values,
    w2_half_values,
    w_plot_max,
    where=~np.isnan(w2_half_values),
    alpha=0.28,
    label="feasible placement on preferred half-side of both centerlines",
)

# General feasibility boundary
ax.plot(
    w_values,
    w2_min_values,
    linewidth=2.4,
    label="feasibility boundary",
)

# Centerline half-side boundary
ax.plot(
    w_values,
    w2_half_values,
    linewidth=2.4,
    linestyle="--",
    label="both-centerlines tangent / half-side boundary",
)

# Physical lower bounds
ax.axvline(w_min, linestyle="--", linewidth=1.4, label="w1 = 2r")
ax.axhline(w_min, linestyle="--", linewidth=1.4, label="w2 = 2r")

# R+r thresholds
ax.axvline(S, linestyle=":", linewidth=1.5, label="wi = R+r")
ax.axhline(S, linestyle=":", linewidth=1.5)

# 45-degree threshold
ax.axvline(w_45, linestyle="-.", linewidth=1.5, label="wi = R+r-q")
ax.axhline(w_45, linestyle="-.", linewidth=1.5)

# Optional: 2S thresholds, where the swept circle can sit without crossing a centerline
ax.axvline(2 * S, linestyle=(0, (3, 5, 1, 5)), linewidth=1.2, label="wi = 2(R+r)")
ax.axhline(2 * S, linestyle=(0, (3, 5, 1, 5)), linewidth=1.2)

# Special points
ax.plot(w_min, S, marker="o", markersize=8)
ax.text(w_min + 0.02, S + 0.02, "(2r, R+r)", fontsize=10)

ax.plot(S, w_min, marker="o", markersize=8)
ax.text(S + 0.02, w_min + 0.02, "(R+r, 2r)", fontsize=10)

ax.plot(w_45, w_45, marker="o", markersize=8)
ax.text(w_45 + 0.02, w_45 + 0.02, "(R+r-q, R+r-q)", fontsize=10)

# Symmetric point for centerline half-side boundary
# Solve w = 2(S - D/sqrt(2))
w_half_sym = 2.0 * (S - q)
ax.plot(w_half_sym, w_half_sym, marker="s", markersize=7)
ax.text(
    w_half_sym + 0.02,
    w_half_sym + 0.02,
    "symmetric half-side point",
    fontsize=10,
)

# Region where nominal 45-degree placement is feasible:
# w1 >= w_45 and w2 >= w_45
ax.fill_between(
    w_values,
    w_45,
    w_plot_max,
    where=(w_values >= w_45),
    alpha=0.18,
    label="nominal 45-degree placement feasible",
)

# Add annotations explaining regions
ax.text(
    0.46,
    1.45,
    "feasible, but\nrequires shifted placement",
    fontsize=10,
    bbox=dict(boxstyle="round", alpha=0.15),
)

ax.text(
    1.08,
    1.35,
    "preferred half-side\nplacement exists",
    fontsize=10,
    bbox=dict(boxstyle="round", alpha=0.15),
)

ax.text(
    0.72,
    0.50,
    "infeasible for\nfixed R",
    fontsize=10,
    bbox=dict(boxstyle="round", alpha=0.15),
)

ax.set_xlabel("corridor width w1")
ax.set_ylabel("corridor width w2")
ax.set_title(
    "Width-space feasibility and centerline half-side placement\n"
    f"R={R}, r={r}, S=R+r={S:.2f}, D=R-r={D:.2f}"
)

ax.set_xlim(w_min - 0.05, w_plot_max)
ax.set_ylim(w_min - 0.05, w_plot_max)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.show()