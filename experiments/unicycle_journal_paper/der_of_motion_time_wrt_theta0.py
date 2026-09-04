import sympy as sp
import matplotlib.pylab as plt
import numpy as np
from math import atan2, asin, cos, pi, sin, sqrt

from kappa_planner.helpers.geometry_operations import wrapPositiveAngle
'''
In this script we compute the derivative of the total time of executing arc-segment-arc (right-left case) maneuvers wrt to theta0. 
It can be improved by adding the left-left case, the indication of the slope 1/omega, and a plot of the path.
'''
# Define fixed parameters
x0, y0 = 0, 0
xc2, yc2 = 0, 6
tau1 = -1
tau2 = 1 
v_max, omega_max = 1, 1
slope = 1/omega_max
R = abs(v_max/omega_max)
xt, yt, thetaf = xc2 + R * cos(pi/2), yc2 + R * sin(pi/2), pi

# Define symbolic variable
theta0 = sp.symbols('theta0')

# Define expressions
xc1 = x0 + R * sp.cos(theta0 + tau1 * pi/2)
yc1 = y0 + R * sp.sin(theta0 + tau1 * pi/2)

dist = sp.sqrt((xc2 - xc1)**2 + (yc2 - yc1)**2) # distance between the two circles
c = sp.sqrt(dist - 4 * R**2) # length of tangent segment
theta1 = sp.atan2(yc2 - yc1, xc2 - xc1) - tau1 * sp.asin(c/dist) - pi/2  # orientation at the end of the first arc
iota1 = tau1 * (theta1 - theta0)
iota2 = tau2 * (thetaf - theta1)
total_time = iota1/omega_max + c/v_max + iota2/omega_max

# Create the symbolic expression of the derivative of total time with respect to theta0 and print it
der = sp.diff(total_time, theta0)
print("Expression of derivative of total time (iota1/omega_max + c/v_max + iota2/omega_max) wrt theta0:")
print(der)

# Compute angle of separation
a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
beta = asin(R/a)
alfa0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
angle_of_separation = alfa0 - tau2 * beta

# Evaluate the derivative at the angle of separation
N = 1000
derivative_array = np.zeros(N) # Initialize array containing derivative of total time wrt theta0
time_array = np.zeros(N) # Initialize array containing total time 
theta0_array = np.linspace(angle_of_separation, angle_of_separation + 2 * pi, N) # Array of theta0 values, ranging of 360 degrees
ind = 0

for theta0_val in theta0_array:
    time_array[ind] = total_time.subs(theta0, theta0_val)
    derivative_array[ind] = der.subs(theta0, theta0_val)
    if derivative_array[ind] == slope:
        print(f'The angle we are looking for is {time_array[ind]}')
    ind += 1
    
fig1, (subplot11, subplot12) = plt.subplots(2, 1)
subplot11.plot(theta0_array, time_array, color = 'r')
subplot11.set_title('Total motion time')
subplot12.plot(theta0_array, derivative_array, color = 'r')
subplot12.set_title('Derivative of total motion time wrt theta0, right-left case')

plt.show(block = True)


