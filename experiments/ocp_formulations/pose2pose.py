
from math import pi
from multi_stage_ocp_pose2pose import ms_ocp_function
import matplotlib.pylab as plt

x0, y0 = 0, 0
xf, yf = 0, 6

v_max, omega_max = 1, 1
v_min, omega_min = 0, 0

theta0 = 0 
thetaf = 100 * pi/180

xs, ys, thetas, ts, vs, omegas, xs_ctrl_grid, ys_ctrl_grid, thetas_ctrl_grid, ts_ctrl_grid, vs_ctrl_grid, omegas_ctrl_grid = ms_ocp_function(x0, y0, theta0, xf, yf, thetaf, v_min, v_max, omega_min, omega_max)
print(thetas_ctrl_grid)
print(xs_ctrl_grid)
delta_theta1 = abs(thetas_ctrl_grid[1] - theta0)* 180/pi
delta_theta2 = abs(thetas_ctrl_grid[3] - thetas_ctrl_grid[1])* 180/pi
delta_theta3 = abs(thetas_ctrl_grid[5] - thetas_ctrl_grid[3])* 180/pi
delta_theta4 = abs(thetas_ctrl_grid[7] - thetas_ctrl_grid[5])* 180/pi
delta_theta5 = abs(thetas_ctrl_grid[9] - thetas_ctrl_grid[7])* 180/pi
total_delta_theta = (delta_theta1 + delta_theta2 + delta_theta3 + delta_theta4 + delta_theta5)
print(f"Total time: {ts_ctrl_grid[-1]}")
print(f'''Delta thetas: \n Tots1: {delta_theta1:.4f}, \n Arc2: {delta_theta2:.4f}, \n Seg3: {delta_theta3:.4f}, 
      \n Arc4: {delta_theta4:.4f}, \n Tots5: {delta_theta5:.4f} \n Total delta theta: {total_delta_theta:.4f}''')

## Plots
x_min = min(x0, xf) - 1
x_max = max(x0, xf) + 1
y_min = min(y0, yf) - 1
y_max = max(y0, yf) + 1
if x_max - x_min > y_max - y_min:
    range = x_max - x_min
    y_max = y_min + range
else:
    range = y_max - y_min
    x_max = x_min + range

plt.figure()
plt.title("Path")
plt.plot(xs, ys, 'b')
plt.plot(xs_ctrl_grid, ys_ctrl_grid, 'ro')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.figure()
plt.title("Velocity")
plt.plot(ts, vs, 'b')
plt.plot(ts_ctrl_grid, vs_ctrl_grid, 'ro')

plt.figure()
plt.title("Angular velocity")
plt.plot(ts, omegas, 'b')
plt.plot(ts_ctrl_grid, omegas_ctrl_grid, 'ro')

plt.show(block=True)
