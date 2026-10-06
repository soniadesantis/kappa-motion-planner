import matplotlib.pylab as plt

from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory, plot_velocity_profiles

""" Hello World Example: Motion Planning for a Unicycle Robot Within 2 Corridors"""

### Define corridors ###
width1 = 3
height1 = 7.2
center1 = [0, 3]
phi1 = 1.570796  # 90 degrees 
width2 = 3
height2 = 4.8
center2 = [1.73, 7]
phi2 = 0.523598  # 30 degrees 

corridor1 = CorridorWorld(
    width = width1,
    height = height1,
    center = center1,
    tilt = phi1)

corridor2 = CorridorWorld(
    width = width2,
    height = height2,
    center = center2,
    tilt = phi2)

corridor_list = [corridor1, corridor2]

### Define Unicycle vehicle ###
footprint_radius = 0.215
vehicle_vmax = 0.5
vehicle_omegamax = 0.5

unicycle = Unicycle(
    state = [0,0,0],
    footprint_radius = footprint_radius,
    v_max = vehicle_vmax,
    v_min = -vehicle_vmax,
    omega_max = vehicle_omegamax,
    omega_min = -vehicle_omegamax)

### Define initial pose and final pose ###
initial_pose = [0.6425, -0.23, 0.0]

final_pose = [3, 8.3, 2.88]

### Define Motion Planner ###
mp = MotionPlanner(unicycle, corridor_list, initial_pose, final_pose)

### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()
print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")

### Plot results ###
figure, ax = plt.subplots(figsize=(12, 6))
mp.plot_planner_inputs(figure=figure, plot_shrunken_corridors=False,
                       plot_corridor_numbers=True)
plot_analytical_trajectory(analytical_trajectory, figure=ax, color="#2563eb", linewidth=2.5)
ax.set(title="Standing unicycle planner: two corridors", xlabel="x [m]", ylabel="y [m]")
ax.grid(alpha=0.15)
figure.tight_layout()
plot_velocity_profiles(analytical_trajectory, unicycle)

plt.show(block=True)
