from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import  MotionPlanner, CurvilinearArcUnicycle, LinearSegmentUnicycle, TurnOnTheSpot, Unicycle
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt


# Define vehicle
vehicle = Unicycle(model = 'Rosbot circular')
vehicle.width = 0.4
r = vehicle.width*0.5
v_max = 0.8
vehicle.update(v_max = v_max )
vehicle.update(omega_max = 1)

## Start poses 
start_pose1 = [0, 0, pi/6]
start_pose2_left= [5, 0, pi/2]
start_pose2_right = [6, 0, pi/2]
start_pose3 = [10, 0, pi/4]

## End poses
end_pose1_left = [0, 5, 4 * pi/6]
end_pose1_right = [0, 4, -3* pi/6]

end_pose3 = [start_pose3[0] + 1.5 * cos(start_pose3[2]),
             start_pose3[1] + 1.5 * sin(start_pose3[2]),
             pi/4]

xc_right = start_pose2_right[0] + vehicle.max_radius * cos(start_pose2_right[2] - pi/2)
yc_right = start_pose2_right[1] + vehicle.max_radius * sin(start_pose2_right[2] - pi/2)
xc_left = start_pose2_left[0] + vehicle.max_radius * cos(start_pose2_left[2] + pi/2)
yc_left = start_pose2_left[1] + vehicle.max_radius * sin(start_pose2_left[2] + pi/2)

end_pose2_left = [xc_left+ vehicle.max_radius * cos(pi/2 - 0.2),
             yc_left+ vehicle.max_radius * sin(pi/2 - 0.2),
             pi - 0.2]

end_pose2_right = [xc_right+ vehicle.max_radius * cos(pi/2 + 0.2),
             yc_right+ vehicle.max_radius * sin(pi/2 + 0.2),
             0 + 0.2]
## Define arc
arc_left = CurvilinearArcUnicycle(
                            xc_left,
                            yc_left,
                            x0 = start_pose2_left[0],
                            y0 = start_pose2_left[1],
                            theta0 = start_pose2_left[2],
                            xf = end_pose2_left[0],
                            yf = end_pose2_left[1],
                            thetaf = end_pose2_left[2],
                            radius = vehicle.max_radius,
                            turn_direction = 1,
                            v = v_max,
                            omega = vehicle.omega_max,
                            unicycle = vehicle,
                            t0 = 0,
                            samples_number = 50)

arc_right = CurvilinearArcUnicycle(
                            xc_right,
                            yc_right,
                            x0 = start_pose2_right[0],
                            y0 = start_pose2_right[1],
                            theta0 = start_pose2_right[2],
                            xf = end_pose2_right[0],
                            yf = end_pose2_right[1],
                            thetaf = end_pose2_right[2],
                            radius = vehicle.max_radius,
                            turn_direction = -1,
                            v = v_max,
                            omega = vehicle.omega_max,
                            unicycle = vehicle,
                            t0 = 0,
                            samples_number = 50)
## Define segment
segment = LinearSegmentUnicycle(start_pose3[0],
                                start_pose3[1],
                                end_pose3[0],
                                end_pose3[1],
                                end_pose3[2],
                                v_max,
                                unicycle=vehicle,
                                t0=0,
                                samples_number=10)



angle_array = np.linspace(0, 2 * np.pi, 100)
r = vehicle.width*0.5
R = vehicle.max_radius
l = r + 0.2

figure = plt.figure()

## Start pose turn on-the-spot
plt.plot(start_pose1[0] + r * np.cos(angle_array), start_pose1[1] + r * np.sin(angle_array), 'g-')
plt.arrow(start_pose1[0], start_pose1[1], l * cos(start_pose1[2]), l * sin(start_pose1[2]), head_width = 0.05, color = 'g')

## End pose turn on-the-spot
plt.plot(end_pose1_left[0] + r * np.cos(angle_array), end_pose1_left[1] + r * np.sin(angle_array), 'r-')
plt.arrow(end_pose1_left[0], end_pose1_left[1], l * cos(end_pose1_left[2]), l * sin(end_pose1_left[2]), head_width = 0.05, color = 'r')

plt.plot(end_pose1_right[0] + r * np.cos(angle_array), end_pose1_right[1] + r * np.sin(angle_array), 'r-')
plt.arrow(end_pose1_right[0], end_pose1_right[1], l * cos(end_pose1_right[2]), l * sin(end_pose1_right[2]), head_width = 0.05, color = 'r')

## Start pose arc
arc_left.plot_path(figure = figure, color = 'k', linewidth = 2)
arc_right.plot_path(figure = figure, color = 'k', linewidth = 2)

plt.plot([xc_left, start_pose2_left[0]], [yc_left, start_pose2_left[1]], color = 'gray', linestyle = 'dashed', linewidth = 1, zorder = 3)
plt.plot([xc_right, start_pose2_right[0]], [yc_right, start_pose2_right[1]], color = 'gray', linestyle = 'dashed', linewidth = 1, zorder = 3)

plt.plot([xc_left, end_pose2_left[0]], [yc_left, end_pose2_left[1]], color = 'gray', linestyle = 'dashed', linewidth = 1, zorder = 3)
plt.plot([xc_right, end_pose2_right[0]], [yc_right, end_pose2_right[1]], color = 'gray', linestyle = 'dashed', linewidth = 1, zorder = 3)

plt.plot(start_pose2_left[0] + r * np.cos(angle_array), start_pose2_left[1] + r * np.sin(angle_array), 'g-', zorder = 3)
plt.arrow(start_pose2_left[0], start_pose2_left[1], l * cos(start_pose2_left[2]), l * sin(start_pose2_left[2]), head_width = 0.05, color = 'g', zorder = 3)

plt.plot(start_pose2_right[0] + r * np.cos(angle_array), start_pose2_right[1] + r * np.sin(angle_array), 'g-', zorder = 3)
plt.arrow(start_pose2_right[0], start_pose2_right[1], l * cos(start_pose2_right[2]), l * sin(start_pose2_right[2]), head_width = 0.05, color = 'g', zorder = 3)

## End pose arc left
plt.plot(end_pose2_left[0] + r * np.cos(angle_array), end_pose2_left[1] + r * np.sin(angle_array), 'r-', zorder = 3)
plt.arrow(end_pose2_left[0], end_pose2_left[1], l * cos(end_pose2_left[2]), l * sin(end_pose2_left[2]), head_width =
    0.05, color = 'r', zorder = 3)  


## End pose arc right
plt.plot(end_pose2_right[0] + r * np.cos(angle_array), end_pose2_right[1] + r * np.sin(angle_array), 'r-', zorder = 3)
plt.arrow(end_pose2_right[0], end_pose2_right[1], l * cos(end_pose2_right[2]), l * sin(end_pose2_right[2]), head_width = 0.05, color = 'r', zorder = 3)

## Start pose segment
segment.plot_path(figure = figure, color = 'k', linewidth = 2)

plt.plot(start_pose3[0] + r * np.cos(angle_array), start_pose3[1] + r * np.sin(angle_array), 'g-', zorder = 3)
plt.arrow(start_pose3[0], start_pose3[1], l * cos(start_pose3[2]), l * sin(start_pose3[2]), head_width = 0.05, color = 'g', zorder = 3)
## End pose segment
plt.plot(end_pose3[0] + r * np.cos(angle_array), end_pose3[1] + r * np.sin(angle_array), 'r-', zorder = 3)
plt.arrow(end_pose3[0], end_pose3[1], l * cos(end_pose3[2]), l * sin(end_pose3[2]), head_width = 0.05, color = 'r', zorder = 3)     

plt.tight_layout()
plt.axis('equal')
plt.axis('off')
# plt.savefig("motion_primitives.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show(block = True)

