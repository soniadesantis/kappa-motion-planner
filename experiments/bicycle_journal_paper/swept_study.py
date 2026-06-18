import numpy as np
import matplotlib.pyplot as plt


from matplotlib.patches import Polygon

from matplotlib.patches import Polygon
import numpy as np

def draw_vehicle(ax, x, y, psi, L, w):
    # vehicle frame: x = forward, y = lateral
    corners = np.array([
        [0, -w/2],   # rear right
        [L, -w/2],   # front right
        [L,  w/2],   # front left
        [0,  w/2]    # rear left
    ])

    Rmat = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    corners_world = corners @ Rmat.T
    corners_world[:, 0] += x
    corners_world[:, 1] += y

    ax.add_patch(
        Polygon(
            corners_world,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=1
        )
    )

# -----------------------
# Parameters
# -----------------------

w = 0.20          # vehicle width [m]
L = 0.40          # wheelbase [m]

R = 1 / w - 0.2   # turning radius of rear axle center [m]

R_inner = R - w/2
R_outer = np.sqrt((R + w/2)**2 + L**2)

B_min = R_outer - R_inner

print(f"w       = {w:.3f} m")
print(f"L       = {L:.3f} m")
print(f"R       = {R:.3f} m")
print(f"R_inner = {R_inner:.3f} m")
print(f"R_outer = {R_outer:.3f} m")
print(f"B_min   = {B_min:.3f} m")

# -----------------------
# Angles
# -----------------------

theta_left = np.linspace(-np.pi/2, 0, 200)
theta_right = np.linspace(np.pi, np.pi/2, 200)

# -----------------------
# Curve centers
# -----------------------

C1 = np.array([0, 0])        # center of first left turn
C2 = np.array([2*R, 0])      # center of second right turn

corner_point1 = np.array([
    R_inner * np.cos(-np.pi/4),
    R_inner * np.sin(-np.pi/4)
    ])

C1_horizontal = np.array([
    corner_point1[0] + R_inner * np.cos(np.pi),
    corner_point1[1] + R_inner * np.sin(np.pi)
])

C1_vertical = np.array([
    corner_point1[0] + R_inner * np.cos(np.pi/2),
    corner_point1[1] + R_inner * np.sin(np.pi/2)
])

corner_point2 = np.array([
    2*R + R_inner * np.cos(3*np.pi/4),
    R_inner * np.sin(3*np.pi/4)
    ])

C2_horizontal = np.array([
    corner_point2[0] + R_inner * np.cos(0),
    corner_point2[1] + R_inner * np.sin(0)
])

C2_vertical = np.array([
    corner_point2[0] + R_inner * np.cos(-np.pi/2),
    corner_point2[1] + R_inner * np.sin(-np.pi/2)
])

x1, y1 = corner_point1
x2, y2 = corner_point2

# -----------------------
# First curve: left 90°
# -----------------------

x1_inner = C1[0] + R_inner * np.cos(theta_left)
y1_inner = C1[1] + R_inner * np.sin(theta_left)

x1_outer = C1[0] + R_outer * np.cos(theta_left)
y1_outer = C1[1] + R_outer * np.sin(theta_left)


# -----------------------
# Second curve: right 90°
# -----------------------

x2_inner = C2[0] + R_inner * np.cos(theta_right)
y2_inner = C2[1] + R_inner * np.sin(theta_right)

x2_outer = C2[0] + R_outer * np.cos(theta_right)
y2_outer = C2[1] + R_outer * np.sin(theta_right)

# -----------------------
# Plot
# -----------------------

fig, ax = plt.subplots(figsize=(7, 5))

# swept areas
ax.fill_between(x1_inner, y1_inner, y1_outer, alpha=0.3)
ax.fill_between(x2_inner, y2_inner, y2_outer, alpha=0.3)

ax.axvline(x1, color='k', linestyle='--')
ax.axhline(y1, color='k', linestyle='--')

ax.axvline(x2, color='k', linestyle='--')
ax.axhline(y2, color='k', linestyle='--')

# boundaries
ax.plot(x1_inner, y1_inner, linestyle="--")
ax.plot(x1_outer, y1_outer, linestyle="--")

ax.plot(x2_inner, y2_inner, linestyle="--")
ax.plot(x2_outer, y2_outer, linestyle="--")

# rear axle centerline
x1_center = C1[0] + R * np.cos(theta_left)
y1_center = C1[1] + R * np.sin(theta_left)

x2_center = C2[0] + R * np.cos(theta_right)
y2_center = C2[1] + R * np.sin(theta_right)

ax.plot(x1_center, y1_center, linewidth=2, label="rear axle path")
ax.plot(x2_center, y2_center, linewidth=2)

angle_array = np.linspace(0, 2*np.pi, 100)

ax.plot(C1[0] + R * np.cos(angle_array),
        C1[1] + R * np.sin(angle_array), color = "k", linestyle="-")

ax.plot(C2[0] + R * np.cos(angle_array),
        C2[1] + R * np.sin(angle_array), color = "k", linestyle="-")

ax.plot(C1_horizontal[0] + R * np.cos(angle_array),
        C1_horizontal[1] + R * np.sin(angle_array), color = "grey", linestyle="--")

ax.plot(C1_vertical[0] + R * np.cos(angle_array),
        C1_vertical[1] + R * np.sin(angle_array), color = "grey", linestyle="--")

ax.plot(C2_horizontal[0] + R * np.cos(angle_array),
        C2_horizontal[1] + R * np.sin(angle_array), color = "grey", linestyle="--")

ax.plot(C2_vertical[0] + R * np.cos(angle_array),
        C2_vertical[1] + R * np.sin(angle_array), color = "grey", linestyle="--")

ax.plot(C1[0], C1[1], "o", label="curve center")
ax.plot(C2[0], C2[1], "o")
ax.plot(corner_point1[0], corner_point1[1], "o", label="corner point")
ax.plot(corner_point2[0], corner_point2[1], "o")

# first curve: left turn
theta1 = np.linspace(-np.pi/2, 0, 20)

x1 = R * np.cos(theta1)
y1 = R * np.sin(theta1)
psi1 = theta1 + np.pi/2


# second curve: right turn
theta2 = np.linspace(np.pi, np.pi/2, 20)

x2 = 2*R + R * np.cos(theta2)
y2 = R * np.sin(theta2)
psi2 = theta2 - np.pi/2

for x, y, psi in zip(x1, y1, psi1):
    draw_vehicle(ax, x, y, psi, L, w)

for x, y, psi in zip(x2, y2, psi2):
    draw_vehicle(ax, x, y, psi, L, w)

ax.set_aspect("equal")
ax.grid(True)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Swept area: left 90° then right 90°")
ax.legend()

plt.show()