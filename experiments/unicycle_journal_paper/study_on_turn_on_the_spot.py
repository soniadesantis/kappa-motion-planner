from arena import circle_intersection, CurvilinearArcUnicycle, wrapPositiveAngle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors, compute_turn_direction, get_intersection, check_point_inside_corridor, compute_three_maneuvers_compact, compute_two_maneuvers
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt
import numpy as np

'''
This script compares the motion times of the two trajectories, with and without turn on-the-spot for two cases: left-left and right-left.
'''
### Initial example ###
# Define initial parameters
x0, y0, theta0 = 0, 0, 5*pi/3
xc2, yc2 = 0, 8 
tau2 = 1 
v_max = 1
omega_max = 1
R = abs(v_max/omega_max)
xt, yt = xc2 + R * cos(pi*0.5), yc2 + R * sin(pi*0.5)

# Create a big enough corridor to test the algorithm without collision
corridor1 = CorridorWorld(width = 15, height = 15, center = [0 , 5 ], tilt = pi*0.5) 

# Create a vehicle
unicycle = Unicycle([x0, y0, theta0], 0.430, length = 0.430, v_max = v_max, v_min = 0, omega_max = omega_max, omega_min = -omega_max)

# Compute two maneuvers left-left
C1_no_tots, S2_no_tots = compute_two_maneuvers([x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
C3_no_tots = CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S2_no_tots.xf, y0=S2_no_tots.yf, theta0=S2_no_tots.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)
time_no_tots = C1_no_tots.maneuver_time + S2_no_tots.maneuver_time + C3_no_tots.maneuver_time

# Compute three maneuvers left-left
T1_tots, C2_tots, S3_tots = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
C4_tots = CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S3_tots.xf, y0=S3_tots.yf, theta0=S3_tots.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)
time_tots = T1_tots.maneuver_time + C2_tots.maneuver_time + S3_tots.maneuver_time + C4_tots.maneuver_time

# Plot example of two paths----------------
figure1 = plot_corridors(corridor_list = [corridor1], figure = None, plot_vectors = False)
C1_no_tots.plot_path(figure1, color = 'g', label = 'Without turn on-the-spot')
S2_no_tots.plot_path(figure1, color = 'g')
C3_no_tots.plot_path(figure1, color = 'g')

T1_tots.plot_path(figure1, color = 'r', label = 'With turn on-the-spot')
C2_tots.plot_path(figure1, color = 'r')
S3_tots.plot_path(figure1, color = 'r')
C4_tots.plot_path(figure1, color = 'r')

plt.plot(x0, y0, 'bo')
plt.plot(xt, yt, 'bo')

plt.arrow(x0, y0, cos(theta0), sin(theta0), head_width = 0.1, color = 'r')
plt.plot(xc2 + R * np.cos(np.linspace(0, 2*pi, 50)), yc2 + R * np.sin(np.linspace(0, 2*pi, 50)), 'k--', linewidth = 0.8)
plt.legend()


### Study on the whole range of theta0 ###
# Compute the maneuver time with varying theta0
# Consider initially a left-left maneuver

# Define initial parameters
N = 3000
time_left_array_tots = np.zeros(N) # contains complete motion time with tots
time_left_array = np.zeros(N) # contains complete motion time without tots
time_tots = np.zeros(N) # contains motion time of only tots
ind = 0
theta0_array = np.linspace(0, 2*pi, N)
colors_tots = ['b']*N
colors = ['r']*N

# Compute analytically th trajectories
for theta0 in theta0_array:
    # Compute the three maneuvers turning to the left 
    T1_left, C1_left, S1_left = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
    if T1_left.maneuver_time <= 10e-3:
        colors_tots[ind] = 'g'
    time_tots[ind] = T1_left.maneuver_time
    time_left_array_tots[ind] = T1_left.maneuver_time + C1_left.maneuver_time + S1_left.maneuver_time
    C1_left, S1_left = compute_two_maneuvers([x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
    time_left_array[ind] = C1_left.maneuver_time + S1_left.maneuver_time
    ind += 1

a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
beta = asin(abs(v_max/omega_max)/a)
alfa0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
tau1 = 1
angle = alfa0 - tau1 * beta
#Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Maneuver time with the initial orientation')

ax1.axvline(x = angle, color = 'r', label = 'theta1')
ax1.plot(theta0_array, time_left_array_tots, label = 'With tots')
ax1.plot(theta0_array, time_left_array, label = 'W/o tots')

# ind = 0
# for time in time_left_array_tots:
#     ax1.plot(theta0_array[ind], time, '.', color = colors_tots[ind])
#     ind +=1

# ind = 0
# for time in time_left_array:
#     ax1.plot(theta0_array[ind], time, '.', color = colors[ind])
#     ind +=1
ax1.legend()

ax2.set_xlabel('Initial orientation theta0 [rad]')
ax2.plot(theta0_array, time_tots, label = 'Time of turn-on-the-spot')

##
fig2, (subplot1, subplot2, subplot3) = plt.subplots(3, 1)
N = 1000
time_left_array_tots = np.zeros(N)
time_left_tots = np.zeros(N)
time_right_array_tots = np.zeros(N)
time_right_tots = np.zeros(N)
ind = 0
theta0_array = np.linspace(0, 2*pi, N)
colors_left_tots = ['b']*N
colors_left = ['r']*N
colors_right_tots = ['b']*N
colors_right = ['r']*N
left = 1 
right = -1

angles_to_plot = [0, pi/2, pi, 3*pi/2]
for theta0 in theta0_array:
    # Compute the three maneuvers turning to the left 
    T1_left, C1_left, S1_left = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
    C2_left =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S1_left.xf, y0=S1_left.yf, theta0=S1_left.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

    if T1_left.maneuver_time <= 10e-3:
        colors_left_tots[ind] = 'g'
    time_left_tots[ind] = T1_left.maneuver_time
    time_left_array_tots[ind] = T1_left.maneuver_time + C1_left.maneuver_time + S1_left.maneuver_time + C2_left.maneuver_time
    # Compute the three maneuvers turning to the right 
    T1_right, C1_right, S1_right = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = -1)
    C2_right =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S1_right.xf, y0=S1_right.yf, theta0=S1_right.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

    if T1_right.maneuver_time <= 10e-3:
        colors_right_tots[ind] = 'g'
    time_right_tots[ind] = T1_right.maneuver_time
    time_right_array_tots[ind] = T1_right.maneuver_time + C1_right.maneuver_time + S1_right.maneuver_time + C2_right.maneuver_time
    ind += 1

## Plot the comparison between motion time of Left-Left and Right-Right
subplot1.plot(theta0_array, time_left_array_tots, color = 'r', label = 'Time Left-Left')
subplot1.plot(theta0_array, time_right_array_tots, color = 'b', label = 'Time Right-Left')
for angle in angles_to_plot:
    subplot1.axvline(x = angle, color = 'k')
subplot1.axvline(x = 3 * pi/2 - beta, color = 'g')

for ind in range(N):
    if abs(time_left_array_tots[ind] - time_right_array_tots[ind]) <= 0.01:
        print(f'{theta0_array[ind]}')

## Plot the motion time Left-Left and region where tots is needed
for ind in range(N):
    if time_left_tots[ind] < 10e-3:
        subplot2.axvline(x = theta0_array[ind], color = 'mistyrose')
    else:
        subplot2.axvline(x = theta0_array[ind], color = 'lightcyan')
for angle in angles_to_plot:
    subplot2.axvline(x = angle, color = 'k')

subplot2.axvline(x = alfa0 - tau2 * beta, color = 'g')

subplot2.plot(theta0_array, time_left_array_tots, color = 'r', label = 'Time Left-Left')

## Plot the motion time Right-Left and region where tots is needed
for ind in range(N):
    if time_right_tots[ind] < 10e-3:
        subplot3.axvline(x = theta0_array[ind], color = 'mistyrose')
    else:
        subplot3.axvline(x = theta0_array[ind], color = 'lightcyan')
for angle in angles_to_plot:
    subplot3.axvline(x = angle, color = 'k')

subplot3.axvline(x = alfa0 - tau2 * beta, color = 'g')

subplot3.plot(theta0_array, time_right_array_tots, color = 'b', label = 'Time Right-Left')
plt.legend()

#### 
T_left_beta, C_left_beta, S_left_beta = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, alfa0 - tau2*beta + 1e-6], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
C2_left_beta =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S_left_beta.xf, y0=S_left_beta.yf, theta0=S_left_beta.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

T_right_beta, C_right_beta, S_right_beta = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, alfa0 - tau2*beta], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = -1)
C2_right_beta =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S_right_beta.xf, y0=S_right_beta.yf, theta0=S_right_beta.theta, xf=xt, yf=yt, thetaf=pi, radius=R , turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

time_left_beta= T_left_beta.maneuver_time + C_left_beta.maneuver_time + S_left_beta.maneuver_time + C2_left_beta.maneuver_time
time_right_beta= T_right_beta.maneuver_time + C_right_beta.maneuver_time + S_right_beta.maneuver_time + C2_right_beta.maneuver_time
print(f'Time left beta: {time_left_beta}\nTime right beta: {time_right_beta}\nDifference: {abs(time_right_beta-time_left_beta)}')

plt.show(block = True)
