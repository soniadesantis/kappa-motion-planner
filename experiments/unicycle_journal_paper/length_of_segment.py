from math import sin, cos, pi, atan2
import numpy as np
from arena import (
    compute_angular_difference,
    compute_angular_difference_with_turn_direction,
    compute_extreme_poses_arc_line,
    wrapPositiveAngle,
    compute_distance_two_points,
    MotionPlanner,
    CorridorWorld,
    Unicycle,
) 
import arena
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt

tau1 = 1
tau0 = tau1
R = 3

x0, y0 = 0.0, 0.0
xc1, yc1 = 0, 10
thetat = pi

a = compute_distance_two_points([x0,y0], [xc1,yc1])  # distance between start point and circumference center
beta = asin(R/a)
alpha0 = wrapPositiveAngle(atan2((yc1 - y0), (xc1 - x0)))

theta_array = np.linspace(alpha0 - tau1*beta, alpha0 - tau1*beta + 2*pi, 1000)
segment_length_array = [0] * len(theta_array)
segment_length_fun_array = [0] * len(theta_array)
iota1_array = [0] * len(theta_array)
iota2_array = [0] * len(theta_array)
motion_time = [0] * len(theta_array)
motion_time_diff_turn_direction = [0] * len(theta_array)

dstar = sqrt(R**2 + a**2 - 2*R*a*cos(0))
iota2star = compute_angular_difference_with_turn_direction(alpha0, thetat, tau0)
d_star_array = [0] * len(theta_array)
actual_d_array = [0] * len(theta_array)
iota1_proof_array = [0] * len(theta_array)
difference_d = [0] * len(theta_array)

bound1 = alpha0 + 3*pi/2
bound2 = alpha0 + 2*pi-beta

for i, theta in enumerate(theta_array):
    chi = theta + tau0 * pi/2
    d_fun = R**2 + a**2 - 2*R*a*cos(chi - alpha0)
    d = sqrt(d_fun)
    xc0 = x0 + R * cos(theta + tau0 * pi/2)
    yc0 = y0 + R * sin(theta + tau0 * pi/2)
    segment_length_array[i] = compute_distance_two_points([xc0, yc0], [xc1, yc1])
    segment_length_fun_array[i] = d

    # Compute the extreme poses for each maneuver
    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc0, yc0, xc1, yc1, tau0, tau1, R)
     
    #Compute orientations for each primitive
    theta0_p1 = theta
    thetaf_p1 = theta1
    theta_p2 = theta1

    iota1 = compute_angular_difference_with_turn_direction(theta0_p1, thetaf_p1, tau0)
    iota2 = compute_angular_difference_with_turn_direction(theta_p2, thetat, tau1)
    iota1_array[i] = iota1 * R
    iota2_array[i] = iota2 * R
    motion_time[i] = iota1 * R + iota2 * R + segment_length_array[i]

    if theta > bound1 and theta <= bound2: #tau0 = 1
        iota1 = compute_angular_difference(theta, alpha0 - tau1*beta + 2*pi - (pi/2 - beta))
        iota11 = pi/2
        motion_time_diff_turn_direction[i] = (abs(iota1) + iota11) * R + abs(iota2star) * R + dstar
        stop = 1

        d_star_array[i] = dstar
        actual_d_array[i] = segment_length_array[i]
        difference_d[i] = segment_length_array[i] - dstar

        iota1_proof_array[i] = abs(iota1) * R 
        

plt.figure()
plt.plot(theta_array, segment_length_array)
plt.plot(theta_array, segment_length_fun_array, linestyle ='--')
plt.axvline(x=alpha0 - tau1*beta + 2*pi - (pi/2 - beta), linestyle='--')
plt.xlabel('Theta (radians)')
plt.ylabel('Length of Segment from Circle to Point')
plt.title('Length of Segment from Circle to Point vs Theta')
plt.grid()

plt.figure()
plt.plot(theta_array, motion_time, label='iota1 (Arc)')
plt.plot(theta_array, motion_time_diff_turn_direction, label='iota1 (Opposite Arc)', linestyle='--')
plt.axvline(x=alpha0 - tau1*beta + 2*pi - (pi/2 - beta), linestyle='--')

plt.figure()
plt.plot(theta_array, actual_d_array, label='Actual d')
plt.plot(theta_array, d_star_array, label='d*', linestyle='--')
plt.plot(theta_array, iota1_proof_array, label='Difference d - d*', linestyle=':')
plt.plot(theta_array, difference_d, label='Difference d - d*', linestyle='-.')
plt.axvline(x=alpha0 - tau1*beta + 2*pi - (pi/2 - beta), linestyle='--')
plt.xlabel('Theta (radians)')
plt.show()
    