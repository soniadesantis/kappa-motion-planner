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
vehicle_width = 0.430
vehicle_length = 0.430
vehicle_vmax = 0.5
vehicle_omegamax = 0.5

unicycle = Unicycle(
    state = [0,0,0],
    width = vehicle_width,
    length = vehicle_length,
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
figure = mp.plot_planner_inputs()
plt.title('Analytical Motion Planner - Unicycle in Two Corridors')
plot_analytical_trajectory(analytical_trajectory, figure)
plot_velocity_profiles(analytical_trajectory, unicycle)

plt.show(block = True)