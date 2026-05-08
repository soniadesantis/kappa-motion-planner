from arena import CorridorWorld, Unicycle, MotionPlanner
from math import pi


c1 = CorridorWorld(0.6, 2, [0.0, 0.0], pi/2)
c2 = CorridorWorld(0.6, 2.1, [0.75, 1], 0)
c3 = CorridorWorld(0.6, 2.1, [1.8, 0.25], -pi/2)
c4 = CorridorWorld(0.6, 2.1, [2.55, -0.8], 0)
c_list = [c1, c2, c3, c4]

veh = Unicycle(model='Rosbot circular')
start = [0.145, -0.617, pi]
end   = [3.295, -0.727, 3*pi/4]

mp = MotionPlanner(veh, c_list, start, end)
traj = mp.compute_trajectory_analytical()

figure = mp.plot_planner_inputs(plot_shrunken_corridors=False)

print(mp.comp_time_analytical_sol)
import matplotlib.pyplot as plt
import arena

arena.plot_analytical_trajectory(traj, figure = figure)
plt.axis('off')
plt.savefig("benelux26_path.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


arena.plot_velocity_profiles(traj, veh)
# plt.savefig("example_multiple_velocity.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show(block = True)