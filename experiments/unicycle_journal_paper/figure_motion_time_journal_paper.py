import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import compute_angular_difference, compute_three_maneuvers_no_collision_avoidance, CurvilinearArcUnicycle, wrapPositiveAngle, Unicycle, compute_two_maneuvers
from math import sin, cos, pi, sqrt, asin, atan2
# import matplotlib.pylab as plt

import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams['text.usetex'] = True  # Enable LaTeX
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Default LaTeX serif font
# Set global font to serif (LaTeX-style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 10,          # base font size
    'axes.labelsize': 14,     # x- and y-labels
    'axes.titlesize': 18,     # axes titles
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})


find_exact_solution = True
use_sympy = False # Set to False to make the code faster

##1- Define numerical parameters
# Test two cases: left-left and right-left
x0_val, y0_val = 0, 0
xc2_val, yc2_val = 0, 6
right = -1
left = 1
tau2_val = 1 # fixed to the left
v_max_val, omega_max_val = 2, 1
slope = 1/omega_max_val
R_val = abs(v_max_val/omega_max_val)

a_val = sqrt((xc2_val - x0_val)**2 + (yc2_val - y0_val)**2)
beta_val = asin(R_val/a_val)
alpha0 = wrapPositiveAngle(atan2((yc2_val - y0_val), (xc2_val - x0_val)))
angle_of_separation = alpha0 - tau2_val * beta_val
cat1 = 2 * R_val # Radius of enlarged second circle for right-left case
cat2 = sqrt(a_val**2 - cat1**2) # Lenght of tangent to second circle for right-left case
beta_2r = asin(cat1/a_val)
gamma = alpha0 - tau2_val * beta_2r # Direction of center first circle for right-left case
xc2_mir = x0_val + cat2 * cos(gamma) 
yc2_mir = y0_val + cat2 * sin(gamma)
xt_val, yt_val = xc2_val + R_val * cos(alpha0), yc2_val + R_val * sin(alpha0)
thetaf_val = alpha0 +tau2_val * pi/2 # Final orientation of the unicycle

# Compute the theta0 from which to start turn on-the-spot with planner
theta0_start_planner_ll = alpha0 - tau2_val * pi/2
theta0_start_planner_rl = alpha0 - tau2_val * beta_2r + tau2_val * pi/2

# Fix the angles in case they are not defined over the interval [alpha0 - tau2_val * beta, alpha0 - tau2_val * beta + 2pi]
if (theta0_start_planner_ll < angle_of_separation): 
    theta0_start_planner_ll += 2 * pi
elif (theta0_start_planner_ll > angle_of_separation + 2 * pi):
    theta0_start_planner_ll -= 2 * pi
if (theta0_start_planner_rl < angle_of_separation):
    theta0_start_planner_rl += 2 * pi
elif (theta0_start_planner_rl > angle_of_separation + 2 * pi):
    theta0_start_planner_rl -= 2 * pi

##2- Define the symbols
# Leave the sympy part but no need to use it for the plots
theta0, thetaf, tau1, tau2, xc2, yc2, x0, y0, R, omega_max, v_max = sp.symbols('theta0 thetaf tau1 tau2 xc2 yc2 x0 y0 R omega_max v_max')

##3- Compute total time and derivative Left-Left
xc1_ll = x0 + R * sp.cos(theta0 + tau1 * sp.pi/2)
yc1_ll = y0 + R * sp.sin(theta0 + tau1 * sp.pi/2)
c_ll = sp.sqrt((xc2 - xc1_ll)**2 + (yc2 - yc1_ll)**2)

theta1_ll = sp.atan2(yc2 - yc1_ll, xc2 - xc1_ll)
iota1_ll = tau1 * (theta1_ll - theta0)
iota2_ll = tau2 * (thetaf - theta1_ll)

total_time_ll = iota1_ll/omega_max + c_ll/v_max + iota2_ll/omega_max

der_ll = sp.diff(total_time_ll, theta0)

##4- Compute total time and derivative Right-Left
xc1_rl = x0 + R * sp.cos(theta0 + tau1 * sp.pi/2)
yc1_rl = y0 + R * sp.sin(theta0 + tau1 * sp.pi/2)

dist = sp.sqrt((xc2 - xc1_rl)**2 + (yc2 - yc1_rl)**2)
c_rl = sp.sqrt(dist**2 - 4 * R**2)
theta1_rl = sp.atan2(yc2 - yc1_rl, xc2 - xc1_rl) - tau1 * sp.asin(c_rl/dist) - sp.pi/2 

iota1_rl = tau1 * (theta1_rl - theta0)
iota2_rl = tau2 * (thetaf - theta1_rl)

total_time_rl = iota1_rl/omega_max + c_rl/v_max + iota2_rl/omega_max

der_rl = sp.diff(total_time_rl, theta0)

##5- Compute the total time and derivative wrt theta0 for different values of theta0
# Create unicycle object
unicycle = Unicycle([x0_val, y0_val, 0], 0.430, length = 0.430, v_max = v_max_val, v_min = 0, omega_max = omega_max_val, omega_min = -omega_max_val)
# Initialize arrays
N = 1000
total_time_sympy_ll_array = np.zeros(N) # with sympy
total_time_sympy_rl_array = np.zeros(N) # with sympy
der_ll_sympy_array = np.zeros(N) # with sympy
der_rl_sympy_array = np.zeros(N) # with sympy

total_time_planner_ll_array = np.zeros(N) # with planner
total_time_planner_4_ll_array = np.zeros(N) # with planner, 4 maneuvers
total_time_planner_rl_array = np.zeros(N) # with planner
total_time_planner_4_rl_array = np.zeros(N) # with planner, 4 maneuvers

total_angular_disp_ll_array = np.zeros(N) # with planner
total_angular_disp_4_ll_array = np.zeros(N) # with planner, 4 maneuvers
total_angular_disp_rl_array = np.zeros(N) # with planner
total_angular_disp_4_rl_array = np.zeros(N) # with planner, 4 maneuvers

theta0_val = angle_of_separation # start from angle_of_separation = alpha0 - tau2_val * beta_val
theta0_array = np.linspace(0, 2 * pi, N, endpoint=True)
last_theta = theta0_val + 2 * pi

## Compute expressions where all the symbols except for theta0 are substituted
total_time_sympy_ll_theta0 = total_time_ll.subs([(thetaf, thetaf_val), (tau1, left), (tau2, tau2_val), (xc2, xc2_val), (yc2, yc2_val),
                            (x0, x0_val), (y0, y0_val), (R, R_val), (omega_max, omega_max_val), (v_max, v_max_val)])
total_time_sympy_rl_theta0 = total_time_rl.subs([(thetaf, thetaf_val), (tau1, right), (tau2, tau2_val), (xc2, xc2_val), (yc2, yc2_val),
                            (x0, x0_val), (y0, y0_val), (R, R_val), (omega_max, omega_max_val), (v_max, v_max_val)])
der_ll_sympy_theta0 = der_ll.subs([(thetaf, thetaf_val), (tau1, left), (tau2, tau2_val), (xc2, xc2_val), (yc2, yc2_val),
                            (x0, x0_val), (y0, y0_val), (R, R_val), (omega_max, omega_max_val), (v_max, v_max_val)])
der_rl_sympy_theta0 = der_rl.subs([(thetaf, thetaf_val), (tau1, right), (tau2, tau2_val), (xc2, xc2_val), (yc2, yc2_val),
                            (x0, x0_val), (y0, y0_val), (R, R_val), (omega_max, omega_max_val), (v_max, v_max_val)])

# Compute the actual values of the total time ll and rl with 4 and 3 maneuvers
ind = 0
for delta_theta in theta0_array:
    # Store the values for theta0 in [angle_of_separation, angle_of_separation + 2pi]
    if use_sympy:
        total_time_sympy_ll_array[ind] = total_time_sympy_ll_theta0.subs([(theta0, theta0_val + delta_theta)])
        der_ll_sympy_array[ind] = der_ll_sympy_theta0.subs(theta0, theta0_val + delta_theta)

    # Three maneuvers with the planner left-left
    C1_ll, S1_ll = compute_two_maneuvers([x0_val, y0_val, theta0_val + delta_theta],
                                         unicycle, xc2_val, yc2_val, tau2_val, t0 = 0, turn1 = 1)
    C2_ll =  CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val,
                                    x0=S1_ll.xf, y0=S1_ll.yf, theta0=S1_ll.theta,
                                    xf=xt_val, yf=yt_val, thetaf=thetaf_val, radius=R_val,
                                    turn_direction=tau2_val,
                                    v=unicycle.v_max, omega=unicycle.omega_max,
                                    unicycle=unicycle, t0=0, samples_number=50)
    total_time_planner_ll_array[ind] = C1_ll.maneuver_time + S1_ll.maneuver_time + C2_ll.maneuver_time
    total_angular_disp_ll_array[ind] = abs(C1_ll.iota) + abs(C2_ll.iota)

    ## Four maneuvers
    T1_4_ll, C1_4_ll, S1_4_ll = compute_three_maneuvers_no_collision_avoidance([x0_val, y0_val, theta0_val + delta_theta],
                                                                               unicycle, xc2_val, yc2_val,
                                                                               tau2_val, t0 = 0, turn1 = 1)
    C2_4_ll= CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val, x0=S1_4_ll.xf, y0=S1_4_ll.yf, theta0=S1_4_ll.theta, xf=xt_val, yf=yt_val, thetaf=thetaf_val, radius=R_val, turn_direction=tau2_val, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

    total_time_planner_4_ll_array[ind] = T1_4_ll.maneuver_time + C1_4_ll.maneuver_time + S1_4_ll.maneuver_time + C2_4_ll.maneuver_time
    total_angular_disp_4_ll_array[ind] = abs(T1_4_ll.delta_angle) + abs(C1_4_ll.iota) + abs(C2_4_ll.iota)
    
    # Store the values for theta0 in [angle_of_separation, angle_of_separation + 2pi]   
    if use_sympy:               
        total_time_sympy_rl_array[ind] = total_time_sympy_rl_theta0.subs([(theta0, theta0_val + delta_theta)])
        der_rl_sympy_array[ind] = der_rl_sympy_theta0.subs(theta0, theta0_val + delta_theta)

    # Three maneuvers with the planner right-left
    C1_rl, S1_rl = compute_two_maneuvers([x0_val, y0_val, theta0_val + delta_theta],
                                         unicycle, xc2_val, yc2_val, tau2_val, t0 = 0, turn1 = -1)
    C2_rl =  CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val,
                                    x0=S1_rl.xf, y0=S1_rl.yf, theta0=S1_rl.theta,
                                    xf=xt_val, yf=yt_val, thetaf=thetaf_val, radius=R_val,
                                    turn_direction=tau2_val,
                                    v=unicycle.v_max, omega=unicycle.omega_max,
                                    unicycle=unicycle, t0=0, samples_number=50)
    total_time_planner_rl_array[ind] = C1_rl.maneuver_time + S1_rl.maneuver_time + C2_rl.maneuver_time

    ## Four maneuvers
    T1_4_rl, C1_4_rl, S1_4_rl = compute_three_maneuvers_no_collision_avoidance([x0_val, y0_val, theta0_val + delta_theta],
                                                                               unicycle, xc2_val, yc2_val,
                                                                               tau2_val, t0 = 0, turn1 = -1)
    C2_4_rl= CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val, x0=S1_4_rl.xf, y0=S1_4_rl.yf, theta0=S1_4_rl.theta, xf=xt_val, yf=yt_val, thetaf=thetaf_val, radius=R_val, turn_direction=tau2_val, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

    total_time_planner_4_rl_array[ind] = T1_4_rl.maneuver_time + C1_4_rl.maneuver_time + S1_4_rl.maneuver_time + C2_4_rl.maneuver_time
    total_angular_disp_4_rl_array[ind] = abs(T1_4_rl.delta_angle) + abs(C1_4_rl.iota) + abs(C2_4_rl.iota)

    ind += 1

##6- Compute the theta0 start as the theta0 where the derivative = slope
if use_sympy:
    theta0_ll_sympy_solve = sp.solve(der_ll_sympy_theta0 - (-left * slope), theta0)
    print(f'\n\nCompute theta0 with sympy for left-left: derivative of motion time wrt theta0 equals -tau1 * slope')
    # print(f'Theta0 for ll computed with sympy {theta0_ll_solve} rad, {[theta0_ll_solve[0] * 180/pi, theta0_ll_solve[1] * 180/pi]} deg')
    print(f'Theta0 for ll computed with sympy {theta0_ll_sympy_solve} rad')
    
    theta0_rl_sympy_solve = sp.solve(der_rl_sympy_theta0 - (-right * slope), theta0)
    print(f'\nCompute theta0 with sympy for right-left: derivative of motion time wrt theta0 equals -tau1 * slope')
    # print(f'Theta0 for rl computed with sympy {theta0_rl_solve} rad, {[theta0_rl_solve[0] * 180/pi, theta0_rl_solve[1] * 180/pi]} deg')
    print(f'Theta0 for rl computed with sympy {theta0_rl_sympy_solve} rad')

    ## Pick the correct theta0 (there are two solutions)
    x1_ll = theta0_ll_sympy_solve[0] # theta0 start for left-left
    y1_ll = total_time_sympy_ll_theta0.subs(theta0, x1_ll) # total time for left-left at theta0 start

    x1_ll_prev, x1_ll_foll = x1_ll - 0.1, x1_ll + 0.1
    y1_ll_prev, y1_ll_foll = total_time_sympy_ll_theta0.subs(theta0, x1_ll_prev), total_time_sympy_ll_theta0.subs(theta0, x1_ll_foll)

    if y1_ll_prev > - left * slope * (x1_ll_prev - x1_ll) + y1_ll and y1_ll_foll > - left * slope * (x1_ll_foll - x1_ll) + y1_ll:
        x_ll, y_ll = theta0_ll_sympy_solve[0] + 2 * pi, total_time_sympy_ll_theta0.subs(theta0, theta0_ll_sympy_solve[0])
    else:
        x_ll, y_ll = theta0_ll_sympy_solve[1] + 2 * pi, total_time_sympy_ll_theta0.subs(theta0, theta0_ll_sympy_solve[1])

    x1_rl = theta0_rl_sympy_solve[0] # theta0 start for right-left
    y1_rl = total_time_sympy_rl_theta0.subs(theta0, x1_rl) # total time for right-left at theta0 start

    x1_rl_prev, x1_rl_foll = x1_rl - 0.1, x1_rl + 0.1
    y1_rl_prev, y1_rl_foll = total_time_sympy_rl_theta0.subs(theta0, x1_rl_prev), total_time_sympy_rl_theta0.subs(theta0, x1_rl_foll)

    if y1_rl_prev > - left * slope * (x1_rl_prev - x1_rl) + y1_rl and y1_rl_foll > - left * slope * (x1_rl_foll - x1_rl) + y1_rl:
        x_rl, y_rl = x1_rl, y1_rl
    else:
        x_rl, y_rl = theta0_rl_sympy_solve[1], total_time_sympy_rl_theta0.subs(theta0, theta0_rl_sympy_solve[1])
# else:
#     # Solution not computed with sympy
#     N_comp = 100
#     ind = 0
#     der_ll_array_short = np.zeros(N_comp)
#     theta0_array_ll = np.linspace(1, 2, N_comp)
#     for delta_theta in theta0_array_ll:
#         der_ll_array_short[ind] = der_ll_sympy_theta0.subs(theta0, theta0_val - delta_theta)
#         ind += 1

#     ind = 0
#     der_rl_array_short = np.zeros(N_comp)
#     theta0_array_rl = np.linspace(1, 2, N_comp)
#     for delta_theta in theta0_array_rl:
#         der_rl_array_short[ind] = der_rl_sympy_theta0.subs(theta0, theta0_val + delta_theta)
#         ind += 1

#     ## Compute the theta0 start for left-left
#     slope_ll_array = -left * slope * np.ones(N_comp)
#     error_ll_array = np.absolute(der_ll_array_short - slope_ll_array)
#     index_ll = np.argmin(error_ll_array)
#     theta0_ll_start_sympy = theta0_array_ll[index_ll]    

#     slope_rl_array = -right * slope * np.ones(N_comp)
#     error_rl_array = np.absolute(der_rl_array_short - slope_rl_array)
#     index_rl = np.argmin(error_rl_array)
#     theta0_rl_start_sympy = theta0_array_rl[index_rl]   

#     x_ll = theta0_val - theta0_ll_start_sympy + 2 * pi
#     y_ll = total_time_sympy_ll_theta0.subs(theta0, theta0_val - theta0_ll_start_sympy)
#     x_rl = theta0_val + theta0_rl_start_sympy
#     y_rl = total_time_sympy_rl_theta0.subs(theta0, theta0_val + theta0_rl_start_sympy)

# print(f'\n\nSelected Theta0 left-left computed with sympy= {x_ll * 180/pi} deg')
# print(f'Selected Theta0 right-left computed with sympy= {x_rl * 180/pi} deg')

##7- Compute the intersection point with the planner
# Compute the total motion time left-left at theta0_start_planner_ll 
C1_ll, S1_ll = compute_two_maneuvers([x0_val, y0_val, theta0_start_planner_ll], unicycle, xc2_val, yc2_val, tau2_val, t0 = 0, turn1 = 1)
C2_ll =  CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val, x0=S1_ll.xf, y0=S1_ll.yf, theta0=S1_ll.theta, xf=xt_val, yf=yt_val, thetaf=pi, radius=R_val, turn_direction=tau2_val, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

y_start_planner_ll = C1_ll.maneuver_time + S1_ll.maneuver_time + C2_ll.maneuver_time
    
C1_rl, S1_rl = compute_two_maneuvers([x0_val, y0_val, theta0_start_planner_rl], unicycle, xc2_val, yc2_val, tau2_val, t0 = 0, turn1 = -1)
C2_rl =  CurvilinearArcUnicycle(xc=xc2_val, yc=yc2_val, x0=S1_rl.xf, y0=S1_rl.yf, theta0=S1_rl.theta, xf=xt_val, yf=yt_val, thetaf=pi, radius=R_val, turn_direction=tau2_val, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)
y_start_planner_rl = C1_rl.maneuver_time + S1_rl.maneuver_time + C2_rl.maneuver_time

x_int_planner = (slope * (theta0_start_planner_ll + theta0_start_planner_rl) + (y_start_planner_ll - y_start_planner_rl))/(2*slope)
y_int_planner = -slope * (x_int_planner - theta0_start_planner_ll) + y_start_planner_ll

print(f'\n\nTheta0 left-left computed with planner= {theta0_start_planner_ll * 180/pi} deg')
print(f'Theta0 right-left computed with planner= {theta0_start_planner_rl * 180/pi} deg')

if use_sympy:
    x, y = sp.symbols('x y')
    m1 = -left * slope
    m2 = -right * slope
    line1 = y - (m1 * (x - x_ll) + y_ll) # y_ll is computed with sympy
    line2 = y - (m2 * (x - x_rl) + y_rl) # x_ll is computed with sympy
    solutions = sp.solve([line1, line2], [x, y])
    print(f'Intersection: {solutions}')
    x_int = solutions[x]
    y_int = solutions[y]

    expr_num = total_time_sympy_ll_theta0 - total_time_sympy_rl_theta0
    y_at_x_int = total_time_sympy_rl_theta0.subs(theta0, x_int)
    expr_num = total_time_sympy_ll_theta0 - y_at_x_int

    print(expr_num.free_symbols)
    solutions2 = sp.nsolve(expr_num, theta0, x_int)
    y_int_numerical_ll, y_int_numerical_rl = total_time_sympy_ll_theta0.subs(theta0, solutions2+2*pi), total_time_sympy_rl_theta0.subs(theta0, solutions2 + 2*pi)
    x_comp = (slope * (x_ll + x_rl) + (y_ll - y_rl))/(2*slope)
    y_comp = -slope * (x_comp - x_ll) + y_ll

# Important! Compute the angular ranges:
# (1) left-left case: where the turn on-the-spot is not needed, from angle_of_separation to theta0_start_planner_ll
range_planner_ll = abs(compute_angular_difference(theta0_start_planner_ll, angle_of_separation))
range_planner_rl = abs(compute_angular_difference(theta0_start_planner_rl, angle_of_separation))
# rule_ll = abs(compute_angular_difference(x_ll, alpha0))
# rule_rl = abs(compute_angular_difference(x_rl, alpha0))
range_computed_ll = tau2_val * (pi/2 - beta_val)
range_computed_rl = tau2_val * (pi/2 + beta_val - beta_2r)
absolute_int_angle = (2*pi - range_computed_ll - range_computed_rl)/2
x_int_computed = angle_of_separation + absolute_int_angle

x_int_computed = theta0_start_planner_ll - (theta0_start_planner_ll - theta0_start_planner_rl)/2
x_appr = alpha0 - tau2_val * beta_val + pi
############### Make all the figures ######################
fig1, (subplot11, subplot12, subplot13, subplot14) = plt.subplots(4, 1)
if use_sympy:
    fig2, (subplot21, subplot22, subplot23, subplot24) = plt.subplots(4, 1)
fig3, (subplot31) = plt.subplots(1, 1)
# fig4, (subplot41, subplot42) = plt.subplots(1, 2, figsize = (10, 10))
fig6, (subplot61, subplot62) = plt.subplots(2, 1)
fig7, (subplot71, subplot72) = plt.subplots(2, 1)

## Plot 1: Left-left compare sympy with planner (total motion time)
# Tick locations (where you want the labels)
# Theta0_array = np.linspace(0, 2 * pi, N) remember! But you computed 
# total_time_ll_array and total_time_ll_array for theta0 in [angle_of_separation, angle_of_separation - 2pi] and
# total_time_rl_array and total_time_rl_array for theta0 in [angle_of_separation, angle_of_separation + 2pi]

if use_sympy:   
    subplot11.plot(theta0_array, total_time_sympy_ll_array, color = 'r', label = 'Total time Left-Left sympy')
    subplot12.plot(theta0_array, total_time_sympy_rl_array , color = 'b', label = 'Total time Right-Left sympy')
    subplot13.plot(theta0_array, total_time_sympy_ll_array , color = 'r', label = 'Total time Left-Left sympy')
    subplot13.plot(theta0_array, total_time_sympy_rl_array , color = 'b', label = 'Total time Right-Left sympy')

subplot11.plot(theta0_array, total_time_planner_ll_array, color = 'g', label = 'Total time Left-Left planner')
subplot11.title.set_text(f'Left-Left: Sympy vs Planner')

subplot12.plot(theta0_array, total_time_planner_rl_array, color = 'g', label = 'Total time Right-Left planner')
subplot12.title.set_text(f'Right-Left: Sympy vs Planner')

subplot13.title.set_text(f'Overlap of left-left and right-right with sympy')

subplot14.plot(theta0_array, total_time_planner_ll_array , color = 'r', label = 'Total time Left-Left planner')
subplot14.plot(theta0_array, total_time_planner_rl_array , color = 'b', label = 'Total time Right-Left planner')
subplot14.title.set_text(f'Overlap of left-left and right-right with planner')

subplot11.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta - 2\pi$')
subplot12.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta - 2\pi$')
subplot13.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')
subplot14.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')

subplot11.legend()
subplot12.legend()
subplot13.legend()
subplot14.legend()

## Figure 2: plot the derivative wrt theta0
if use_sympy:
    fig2.suptitle(f'Derivative wrt theta0. Slope = {slope}')
    subplot21.plot(theta0_val + theta0_array, total_time_sympy_ll_array, color = 'r', label = 'Total time Left-Left sympy')
    subplot22.plot(theta0_val + theta0_array, der_ll_sympy_array, color = 'r', label = 'Derivative wrt theta0 Left-Left')
    subplot23.plot(theta0_val + theta0_array, total_time_sympy_rl_array, color = 'b', label = 'Total time Right-Left sympy')
    subplot24.plot(theta0_val + theta0_array, der_rl_sympy_array, color = 'b', label = 'Derivative wrt theta0 Right-Left')

    subplot21.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot22.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot23.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot24.set_xlabel(r'$\theta_0 \in [\alpha_0 - \tau_2 \beta, \alpha_0 - \tau_2 \beta + 2\pi$')

    subplot21.legend()
    subplot22.legend()
    subplot23.legend()
    subplot24.legend()

    subplot21.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
    subplot21.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot21.axvline(x = x_ll, color = 'r')

    subplot22.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
    subplot22.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot22.axvline(x = x_ll, color = 'r')

    subplot23.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
    subplot23.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot23.axvline(x = x_rl, color = 'b')

    subplot24.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
    subplot24.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
    subplot24.axvline(x = x_rl, color = 'b')


## Figure 3: plot the derivative wrt theta0 and the slope
fig, subplot31 = plt.subplots(figsize=(8, 3.2))
linewidth = 0.8
color_ll = '#008080'   # teal
color_rl = '#8E44AD'   # muted magenta

subplot31.axvline(x = theta0_start_planner_ll, color = '0.5', linewidth = 0.5, linestyle = '--')
subplot31.axvline(x = theta0_start_planner_rl, color = '0.5', linewidth = 0.5, linestyle = '--')
subplot31.axvline(x = x_int_planner, color = '0.5', linewidth = 0.5, linestyle = '--')

subplot31.axvline(x = theta0_val, color = 'k', linewidth = 0.5, linestyle = '--')
subplot31.axvline(x = theta0_val+ 2*pi, color = 'k', linewidth = 0.5, linestyle = '--')

x_main = theta0_val + theta0_array  # exclude the last point to avoid overlap

subplot31.plot(x_main[:-1], total_time_planner_4_ll_array[:-1], color = color_ll, label = r'$\boldsymbol{T}^{\mathrm{eq}}$')
subplot31.plot(x_main[:-1], total_time_planner_4_rl_array[:-1], color = color_rl, label = r'$\boldsymbol{T}^{\mathrm{neq}}$')

## Plot the important points
subplot31.plot(theta0_start_planner_ll, y_start_planner_ll, color = color_ll, marker = 'o', linestyle='None', markersize = 6,
               label = r'Switch from $\boldsymbol{T}^{\mathrm{eq}}_{\text{no-turn}}$ to $\boldsymbol{T}^{\mathrm{eq}}_{\text{turn}}$')

subplot31.plot(theta0_start_planner_rl, y_start_planner_rl, color = color_rl, marker = 'o', linestyle='None', markersize = 6, 
               label = r'Switch from $\boldsymbol{T}^{\mathrm{neq}}_{\text{no-turn}}$ to $\boldsymbol{T}^{\mathrm{neq}}_{\text{turn}}$')

subplot31.plot(
    theta0_val,
    total_time_planner_4_ll_array[0],
    marker='D',              # diamond
    linestyle='None',
    color='w',
    markersize=8,
    markeredgecolor='black',
    markeredgewidth=0.8,
    label=r'Intersection at $\theta_0 = \alpha_0 - \tau_1 \beta$'
)
subplot31.plot(x_int_planner, y_int_planner,  color = 'k', marker = 'o', linestyle='None', markersize = 6, 
                label = r'Linear branches intersection')

xticks = [
    theta0_val,
    theta0_start_planner_ll,
    x_int_planner,
    theta0_start_planner_rl,
    last_theta,
]

xtick_labels = [
    r'$\alpha_0 - \beta$',
    r'$\alpha_0 + \frac{3\pi}{2}$',
    r'$\theta_0^{\mathrm{int}}$',
    r'$\alpha_0 - \beta^\prime + \frac{\pi}{2}$',
    r'$\alpha_0 - \beta + 2\pi$',
]
subplot31.set_yticks([])

subplot31.set_xticks(xticks)
subplot31.set_xticklabels(xtick_labels)

subplot31.set_xlabel(r'Initial orientation $\theta_0$')
subplot31.set_ylabel(r'Motion time $\boldsymbol{T}$ when $\tau_1 = 1$')

subplot31.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.6
)

# fig.savefig(
#     "motion_time.svg",
#     format='svg',
#     bbox_inches='tight',
#     pad_inches=0,
#     transparent=True
# )
# fig.savefig(
#     "motion_time.pdf",
#     bbox_inches='tight',
#     pad_inches=0,
#     transparent=True
# )


xc1_val_rl = x0_val + R_val * cos(theta0_start_planner_rl + right * pi/2)
yc1_val_rl = y0_val + R_val * sin(theta0_start_planner_rl + right * pi/2)
dist_val = sqrt((xc2_val - xc1_val_rl)**2 + (yc2_val- yc1_val_rl)**2)
c_val_rl = sqrt(dist_val**2 - 4 * R_val**2)

xc1_val_ll = x0_val + R_val * cos(theta0_start_planner_ll + left * pi/2)
yc1_val_ll = y0_val + R_val * sin(theta0_start_planner_ll + left * pi/2)
c_val_ll = sqrt((xc2_val - xc1_val_ll)**2 + (yc2_val - yc1_val_ll)**2)

## Figure 4: plot figure of left-left and right-left cases for visualization
arrow_len = 1.5

x1 = min(x0_val - 3 * R_val, xc2_val - 3 * R_val)
y1 = min(y0_val - 3 * R_val, yc2_val - 3 * R_val)
y2 = max(y0_val + 3 * R_val, yc2_val + 3 * R_val)
range_axis = y2 - y1
x2 = x1 + range_axis

# ## Subplot41 left-left
# subplot41.set_xlim([x1, x2])
# subplot41.set_ylim([y1, y2])
# # Plot initial position
# subplot41.plot(x0_val, y0_val, 'ko', label = 'Initial position')
# # Plot final position
# subplot41.plot(xt_val, yt_val, 'ko', label = 'Final position')
# # Plot center of second circle
# subplot41.plot(xc2_val, yc2_val, 'ko')
# # Plot second circumference
# subplot41.plot(xc2_val + R_val * np.cos(np.linspace(0, 2*pi, 100)), yc2_val + R_val * np.sin(np.linspace(0, 2*pi, 100)), 'k--') # Plot second circle
# # Plot alpha0 direction
# subplot41.plot([x0_val,xc2_val], [y0_val, yc2_val], 'k--', linewidth = 0.8)
# # Plot direction of tots theta0 start ll
# subplot41.arrow(x=x0_val, y=x0_val, dx=arrow_len * cos(theta0_start_planner_ll), dy=arrow_len * sin(theta0_start_planner_ll), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
# subplot41.plot([x0_val,x0_val + 10 * cos(theta0_start_planner_ll)],
#                [y0_val, y0_val + 10 * sin(theta0_start_planner_ll)],
#                'k--', linewidth = 0.8, label = 'Theta0 start for Direction ll planner') # Plot direction of motion

# ## Subplot42 right-left
# subplot42.set_xlim([x1, x2])
# subplot42.set_ylim([y1, y2])
# # Plot initial position
# subplot42.plot(x0_val, y0_val, 'ko', label = 'Initial position')
# # Plot final position 
# subplot42.plot(xt_val, yt_val, 'ko', label = 'Final position')
# # Plot center of second circle
# subplot42.plot(xc2_val, yc2_val, 'ko')
# # Plot second circumference
# subplot42.plot(xc2_val + R_val * np.cos(np.linspace(0, 2*pi, 100)), yc2_val + R_val * np.sin(np.linspace(0, 2*pi, 100)), 'k--') # Plot second circle
# # Plot second circumference with 2R
# subplot42.plot(xc2_val + 2*R_val * np.cos(np.linspace(0, 2*pi, 100)), yc2_val + 2*R_val * np.sin(np.linspace(0, 2*pi, 100)), 'k--') # Plot second circle
# # Plot alpha0 direction
# subplot42.plot([x0_val,xc2_val], [y0_val, yc2_val], 'k--', linewidth = 0.8)
# # Plot direction of tots theta0 start rl
# subplot42.arrow(x=x0_val, y=x0_val, dx=arrow_len * cos(theta0_start_planner_rl), dy=arrow_len * sin(theta0_start_planner_rl), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
# subplot42.plot([x0_val,x0_val + 10 * cos(theta0_start_planner_rl)],
#                [y0_val, y0_val + 10 * sin(theta0_start_planner_rl)],
#                'k--', linewidth = 0.8, label = 'Theta0 start for Direction rl planner')
# # Plot mirrored circumference
# subplot42.plot(xc2_mir, yc2_mir, 'ko') # Plot center of mirrored circle
# subplot42.plot(xc2_mir + R_val * np.cos(np.linspace(0, 2*pi, 100)), yc2_mir + R_val * np.sin(np.linspace(0, 2*pi, 100)), 'k--') # Plot mirrored circle

####### Figure for paper
## Figure 6: plot the derivative wrt theta0
subplot61.title.set_text(r'Total time $\tau_1 = \tau_2 = 1$')
subplot61.plot(theta0_val + theta0_array, total_time_planner_ll_array, color = 'r', label = r'Total time without turn on-the-spot $\boldsymbol{T}_{\text{no-turn}}$')
subplot61.plot(theta0_val + theta0_array, total_time_planner_4_ll_array, color = 'r', linestyle = 'dashed', label = r'Total time with turn on-the-spot $\boldsymbol{T}_{\text{turn}}$')

subplot62.title.set_text(r'Derivative of $\boldsymbol{T}_{\text{no-turn}}$ wrt $\theta_0$, $\tau_1 = \tau_2 = 1$')
if use_sympy:
    subplot62.plot(theta0_val + theta0_array, der_ll_sympy_array, color = 'r', label = r'Derivative of total time without turn on-the-spot wrt $\theta_0$')

subplot61.legend()
subplot62.legend()

subplot61.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
subplot61.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
subplot61.axvline(x = theta0_start_planner_ll, color = 'k')

subplot62.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
subplot62.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
subplot62.axvline(x = theta0_start_planner_ll, color = 'k')

subplot61.set_xticks([])
subplot61.set_yticks([])

subplot62.set_xticks([])
subplot62.set_yticks([])

## Figure 7: plot the derivative wrt theta0
subplot71.title.set_text(r'Total time $\tau_1 = -1, \tau_2 = 1$')
subplot71.plot(theta0_val + theta0_array, total_time_planner_rl_array, color = 'b', label = r'Total time without turn on-the-spot $\boldsymbol{T}_{\text{no-turn}}$')
subplot71.plot(theta0_val + theta0_array, total_time_planner_4_rl_array, color = 'b', linestyle = 'dashed', label = r'Total time with turn on-the-spot $\boldsymbol{T}_{\text{turn}}$')

subplot72.title.set_text(r'Derivative of $\boldsymbol{T}_{\text{no-turn}}$ wrt $\theta_0$, $\tau_1 = -1, \tau_2 = 1$')
if use_sympy:
    subplot72.plot(theta0_val + theta0_array, der_rl_sympy_array, color = 'b', label = r'Derivative of total time without turn on-the-spot wrt $\theta_0$')

subplot71.legend()
subplot72.legend()

subplot71.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
subplot71.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
subplot71.axvline(x = theta0_start_planner_rl, color = 'k')

subplot72.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
subplot72.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
subplot72.axvline(x = theta0_start_planner_rl, color = 'k')

subplot71.set_xticks([])
subplot71.set_yticks([])

subplot72.set_xticks([])
subplot72.set_yticks([])
# subplot23.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
# subplot23.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
# subplot23.axvline(x = x_rl, color = 'b')

# subplot24.axvline(x = theta0_val, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta$')
# subplot24.axvline(x = theta0_val + 2*pi, color = 'k', linewidth = 0.5, label = r'$\theta_0 = \alpha_0 - \tau_2 \beta + 2\pi$')
# subplot24.axvline(x = x_rl, color = 'b')

plt.show(block = True)


