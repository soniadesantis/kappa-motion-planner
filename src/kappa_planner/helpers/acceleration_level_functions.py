
import numpy as np
import time
import matplotlib.pyplot as plt
from math import pi, cos, sin, tan, sqrt, exp, atan2, asin, acos
from scipy.interpolate import interp1d
from .geometry_operations import (
    compute_distance_two_points,
    wrapPositiveAngle,
    compute_angular_difference
)
from .intersections import (
    compute_intersection_two_segments,
)


def line_circle_intersection(m, b, h, k, r, tol=1e-9):
    """
    Compute the intersection points between a line y = m*x + b
    and a circle (x - h)^2 + (y - k)^2 = r^2.

    Returns:
        A list of tuples (x, y) for each intersection point.
        Could be 0, 1, or 2 points depending on geometry.
    """
    # Substitute y = m*x + b into circle equation
    # (x - h)^2 + (m*x + b - k)^2 = r^2
    A = 1 + m**2
    B = 2 * (m*(b - k) - h)
    C = h**2 + (b - k)**2 - r**2

    # Compute discriminant
    D = B**2 - 4*A*C

    if D < -tol:
        # No real intersection
        return []
    elif abs(D) <= tol:
        # One intersection (tangent)
        x = -B / (2*A)
        y = m*x + b
        return [(x, y)]
    else:
        # Two intersections
        sqrtD = math.sqrt(D)
        x1 = (-B + sqrtD) / (2*A)
        x2 = (-B - sqrtD) / (2*A)
        y1 = m*x1 + b
        y2 = m*x2 + b
        return [(x1, y1), (x2, y2)]
    

def compute_overtake_maneuver_with_final_pose(Bicycle, start_pose, end_pose, overtake = 1, t0 = 0, time_interval = 0.05):
    '''
    Compute an overtaking maneuver for a bicycle model using an analytical approach.
    Parameters:
    - Bicycle: Bicycle model parameters (L, delta_max, delta_dot_max, v_max)
    - start_pose: tuple (x0, y0, theta0)
    - end_pose: tuple (xf, yf, thetaf)
    - overtake: 1 for overtaking on the left, -1 for overtaking on the right
    - t0: initial time
    Returns:
    - ts_analytical: time samples of the maneuver
    - xs_analytical: x positions of the maneuver
    - ys_analytical: y positions of the maneuver
    '''
    ## Extract the variables 
    L = Bicycle.wheelbase
    v_max = Bicycle.v_max
    a_max = Bicycle.a_max
    delta_max = Bicycle.delta_max
    delta_dot_max = Bicycle.delta_dot_max
    R = Bicycle.max_radius 
    x0, y0, theta0 = start_pose
    xf, yf, thetaf = end_pose

    start_analytical = time.perf_counter()

    ## Find the orientation that you want to reach at the end of phase A
    if overtake == 1:      
        tau1 = 1 
        tau2 = -1 
    else:
        tau1 = -1 
        tau2 = 1

    # R = L / tan(delta_max)
    xc1, yc1 = x0 + R * cos(theta0 + tau1 * pi * 0.5), y0 + R * sin(theta0 + tau1 * pi * 0.5)
    xc2, yc2 = xf + R * cos(thetaf + tau2 * pi * 0.5), yf + R * sin(thetaf + tau2 * pi * 0.5)

    zeta = - (tau1 + tau2) * pi * 0.25
    eta = (tau1 - tau2) * pi * 0.25
    a1 = sqrt((yc1 - yc2)**2 + (xc1 - xc2)**2)
    c1 = sqrt((a1)**2 - (R - tau1 * tau2 * R)**2)
    alpha1 = wrapPositiveAngle(atan2((yc2 - yc1), (xc2 - xc1)))
    gamma1 = asin((R - R)/a1) if tau1 * tau2 > 0 else asin((c1/a1))
    beta1 = alpha1 - tau1 * gamma1
    x1 = xc1 + R * cos(beta1 + zeta)
    y1 = yc1 + R * sin(beta1 + zeta)
    x2 = x1 + c1 * cos(beta1 + eta)
    y2 = y1 + c1 * sin(beta1 + eta)
    theta_hold = wrapPositiveAngle(atan2((y2 - y1),(x2 - x1)))

    ## Phase A
    # Compute the delta_peak of phase A 
    dtheta_A = wrapPositiveAngle(theta_hold) - wrapPositiveAngle(theta0)
    delta_peak_A = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_A)))
    # Compute motion time of phase A
    T_A_half = delta_peak_A / delta_dot_max
    T_A = 2 * T_A_half
    # Simulate phase A
    t_A, x_A, y_A, theta_A = simulate_bicycle_trapezoid(
        x0, y0, theta0,
        v_max, L,
        0, 2*T_A_half, time_interval,
        T_A_half, delta_dot_max,
        tau1 = tau1, tau2 = tau2
    )
    v_A = np.ones_like(t_A) * v_max
    delta_A = [delta_of_t(ti, 0, T_A_half, delta_dot_max, tau1, tau2) for ti in t_A]
    delta_dot_A = [delta_dot_of_t(ti, 0, T_A_half, delta_dot_max, tau1, tau2) for ti in t_A]
    a_A = 0*np.ones_like(t_A)

    ## Phase C
    # Compute the delta_peak of phase C
    dtheta_C = wrapPositiveAngle(theta_hold + np.pi) - wrapPositiveAngle(thetaf + np.pi) # Remember that the final orientation is reversed
    delta_peak_C = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_C)))
    # Compute motion time of phase C
    T_C_half = delta_peak_C / delta_dot_max
    T_C = 2 * T_C_half
    # Compute the correct turning directions for phase C ## CHECK THIS
    if wrapPositiveAngle(thetaf + np.pi) - wrapPositiveAngle(theta_hold + np.pi) >= 0:
        tau3 = -1
        tau4 = 1
    else:
        tau3 = 1
        tau4 = -1
    # Simulate phase C
    t_C, x_C, y_C, theta_C = simulate_bicycle_trapezoid(
        xf, yf, thetaf + np.pi,
        v_max, L,
        0, 2*T_C_half, time_interval,
        T_C_half, delta_dot_max,
        tau1 = tau3, tau2 = tau4
    )
    x_C = x_C[::-1]
    y_C = y_C[::-1]
    theta_C = theta_C[::-1] - np.pi
    v_C = np.ones_like(t_C) * v_max
    delta_C = [delta_of_t(ti, 0, T_C_half, delta_dot_max, tau4, tau3) for ti in t_C]
    delta_dot_C = [delta_dot_of_t(ti, 0, T_C_half, delta_dot_max, tau4, tau3) for ti in t_C]
    a_C = 0*np.ones_like(t_C)

    ## Phase B
    N_B = 10
    x_B = np.linspace(x_A[-1], x_C[0], N_B)
    y_B = np.linspace(y_A[-1], y_C[0], N_B)
    theta_B = np.ones(N_B) * wrapPositiveAngle(atan2((y_C[0] - y_A[-1]), (x_C[0] - x_A[-1])))
    v_B = np.ones(N_B) * v_max
    delta_B = np.ones(N_B) * 0
    a_B = np.zeros(N_B)
    delta_dot_B = np.zeros(N_B)
    phase_B_length = compute_distance_two_points([x_A[-1], y_A[-1]], [x_C[0], y_C[0]])
    T_B = phase_B_length / v_max

    ## Adjust time vectors of the three phases
    t_A = t_A + t0
    t_B = np.linspace(t0 + T_A, t0 + T_A + T_B, N_B)
    t_C = t_C + t0 + T_A + T_B
    total_motion_time = T_A + T_B + T_C

    ## Interpolate the results
    ts_analytical = np.concatenate((t_A, t_B, t_C))
    xs_analytical = np.concatenate((x_A, x_B, x_C))
    ys_analytical = np.concatenate((y_A, y_B, y_C))
    thetas_analytical = np.concatenate((theta_A, theta_B, theta_C))
    vs_analytical =  np.concatenate((v_A, v_B, v_C))
    deltas_analytical = np.concatenate((delta_A, delta_B, delta_C))
    delta_dots_analytical = np.concatenate((delta_dot_A, delta_dot_B, delta_dot_C))

    f_x_analytical = interp1d(ts_analytical, xs_analytical, kind='linear')
    f_y_analytical = interp1d(ts_analytical, ys_analytical, kind='linear')
    f_theta_analytical = interp1d(ts_analytical, thetas_analytical, kind='linear')
    f_v_analytical = interp1d(ts_analytical, vs_analytical, kind='linear')
    f_delta_analytical = interp1d(ts_analytical, deltas_analytical, kind='linear')
    f_delta_dot_analytical = interp1d(ts_analytical, delta_dots_analytical, kind='linear')

    end_analytical = time.perf_counter()
    comp_time_analytical = end_analytical - start_analytical
    print(f"Computation time analytical: {comp_time_analytical:.6f} s")

    return (
        ts_analytical, f_x_analytical, f_y_analytical, f_theta_analytical,
        f_v_analytical, f_delta_analytical, f_delta_dot_analytical,
        total_motion_time,
        x_A, y_A, theta_A,
        x_B, y_B, theta_B,
        x_C, y_C, theta_C
    )


def compute_overtake_maneuver_shortest_distance(Bicycle, start_pose, overtake = 1, t0 = 0, time_interval = 0.05, lane_width = 3.5, theta_hold = 0.09):
    '''
    Compute an overtaking maneuver for a bicycle model using an analytical approach.
    Parameters:
    - Bicycle: Bicycle model parameters (L, delta_max, delta_dot_max, v_max)
    - start_pose: tuple (x0, y0, theta0)
    - end_pose: tuple (xf, yf, thetaf)
    - overtake: 1 for overtaking on the left, -1 for overtaking on the right
    - t0: initial time
    Returns:
    - ts_analytical: time samples of the maneuver
    - xs_analytical: x positions of the maneuver
    - ys_analytical: y positions of the maneuver
    '''
    ## Extract the variables 
    L = Bicycle.wheelbase
    v_max = Bicycle.v_max
    a_max = Bicycle.a_max
    delta_max = Bicycle.delta_max
    delta_dot_max = Bicycle.delta_dot_max
    R = Bicycle.max_radius 
    x0, y0, theta0 = start_pose
    theta_hold = theta0 + overtake * theta_hold
    start_analytical = time.perf_counter()

    ## Find the orientation that you want to reach at the end of phase A
    if overtake == 1:      
        tau1 = 1 
        tau2 = -1 
    else:
        tau1 = -1 
        tau2 = 1

    ## Phase A
    # Compute the delta_peak of phase A 
    dtheta_A = wrapPositiveAngle(theta_hold) - wrapPositiveAngle(theta0)
    delta_peak_A = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_A)))
    # Compute motion time of phase A
    T_A_half = delta_peak_A / delta_dot_max
    T_A = 2 * T_A_half
    # Simulate phase A
    t_A, x_A, y_A, theta_A = simulate_bicycle_trapezoid(
        x0, y0, theta0,
        v_max, L,
        0, 2*T_A_half, time_interval,
        T_A_half, delta_dot_max,
        tau1 = tau1, tau2 = tau2
    )
    v_A = np.ones_like(t_A) * v_max
    delta_A = [delta_of_t(ti, 0, T_A_half, delta_dot_max, tau1, tau2) for ti in t_A]
    delta_dot_A = [delta_dot_of_t(ti, 0, T_A_half, delta_dot_max, tau1, tau2) for ti in t_A]
    a_A = 0*np.ones_like(t_A)

    ## Phase C
    # Compute the delta_peak of phase C
    dtheta_C = wrapPositiveAngle(theta_hold + np.pi) - wrapPositiveAngle(0 + np.pi) # Remember that the final orientation is reversed
    delta_peak_C = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_C)))
    # Compute motion time of phase C
    T_C_half = delta_peak_C / delta_dot_max
    T_C = 2 * T_C_half
    # Compute the correct turning directions for phase C ## CHECK THIS
    if wrapPositiveAngle(0 + np.pi) - wrapPositiveAngle(theta_hold + np.pi) >= 0:
        tau3 = -1
        tau4 = 1
    else:
        tau3 = 1
        tau4 = -1
    # Simulate phase C
    t_C, x_C, y_C, theta_C = simulate_bicycle_trapezoid(
        0, tau1 * lane_width*0.5, 0 + np.pi,
        v_max, L,
        0, 2*T_C_half, time_interval,
        T_C_half, delta_dot_max,
        tau1 = tau3, tau2 = tau4
    )
    x_C = x_C[::-1]
    y_C = y_C[::-1]
    theta_C = theta_C[::-1] - np.pi
    v_C = np.ones_like(t_C) * v_max
    delta_C = [delta_of_t(ti, 0, T_C_half, delta_dot_max, tau4, tau3) for ti in t_C]
    delta_dot_C = [delta_dot_of_t(ti, 0, T_C_half, delta_dot_max, tau4, tau3) for ti in t_C]
    a_C = 0*np.ones_like(t_C)

    dy_phase_C = abs(y_C[0] - y_C[-1])
    int_point, _ = compute_intersection_two_segments([x_A[-1], y_A[-1]],
                                      [x_A[-1] + 1000 * cos(theta_hold), y_A[-1]+ 1000 * sin(theta_hold)],
                                      [x0, lane_width*0.5 - dy_phase_C], [1000, lane_width*0.5 - dy_phase_C])
    x_shift = int_point[0] - x_C[0]
    # plt.figure()
    # plt.plot([x_A[-1], x_A[-1] + 1000 * cos(theta_hold)],[y_A[-1], y_A[-1]+ 1000 * sin(theta_hold)], 'b-', label='Phase A')
    # plt.plot([x0, 1000 ], [lane_width*0.5 - dy_phase_C, lane_width*0.5 - dy_phase_C], 'r-', label='Intersection line')
    # plt.show()
    x_C = x_C + x_shift

    ## Phase B
    N_B = 10
    x_B = np.linspace(x_A[-1], int_point[0], N_B)
    y_B = np.linspace(y_A[-1], int_point[1], N_B)
    theta_B = np.ones(N_B) * wrapPositiveAngle(atan2((int_point[1] - y_A[-1]), (int_point[0] - x_A[-1])))
    v_B = np.ones(N_B) * v_max
    delta_B = np.ones(N_B) * 0
    a_B = np.zeros(N_B)
    delta_dot_B = np.zeros(N_B)
    phase_B_length = compute_distance_two_points([x_A[-1], y_A[-1]], [int_point[0], int_point[1]])
    T_B = phase_B_length / v_max

    ## Adjust time vectors of the three phases
    t_A = t_A + t0
    t_B = np.linspace(t0 + T_A, t0 + T_A + T_B, N_B)
    t_C = t_C + t0 + T_A + T_B
    total_motion_time = T_A + T_B + T_C

    ## Interpolate the results
    ts_analytical = np.concatenate((t_A, t_B, t_C))
    xs_analytical = np.concatenate((x_A, x_B, x_C))
    ys_analytical = np.concatenate((y_A, y_B, y_C))
    thetas_analytical = np.concatenate((theta_A, theta_B, theta_C))
    vs_analytical =  np.concatenate((v_A, v_B, v_C))
    deltas_analytical = np.concatenate((delta_A, delta_B, delta_C))
    delta_dots_analytical = np.concatenate((delta_dot_A, delta_dot_B, delta_dot_C))

    f_x_analytical = interp1d(ts_analytical, xs_analytical, kind='linear')
    f_y_analytical = interp1d(ts_analytical, ys_analytical, kind='linear')
    f_theta_analytical = interp1d(ts_analytical, thetas_analytical, kind='linear')
    f_v_analytical = interp1d(ts_analytical, vs_analytical, kind='linear')
    f_delta_analytical = interp1d(ts_analytical, deltas_analytical, kind='linear')
    f_delta_dot_analytical = interp1d(ts_analytical, delta_dots_analytical, kind='linear')

    # even_time_grid = np.arange(t0, t0 + total_motion_time, time_interval)
    # x_analytical_on_grid = f_x_analytical(even_time_grid)
    # y_analytical_on_grid = f_y_analytical(even_time_grid)

    end_analytical = time.perf_counter()
    comp_time_analytical = end_analytical - start_analytical
    print(f"Computation time analytical: {comp_time_analytical:.6f} s")

    return (
        ts_analytical, f_x_analytical, f_y_analytical, f_theta_analytical,
        f_v_analytical, f_delta_analytical, f_delta_dot_analytical,
        total_motion_time,
        x_A, y_A, theta_A,
        x_B, y_B, theta_B,
        x_C, y_C, theta_C
    )

def delta_of_t(t, t0, T_half, delta_dot_max, tau1=1, tau2=-1):
    if t <= t0 + T_half:
        return tau1*delta_dot_max * (t - t0)
    else:
        return tau1*delta_dot_max * T_half + tau2*delta_dot_max * (t - (t0 + T_half))
    
def delta_dot_of_t(t, t0, T_half, delta_dot_max, tau1=1, tau2=-1):
    if t <= t0 + T_half:
        return delta_dot_max * tau1
    else:
        return delta_dot_max * tau2


def delta_of_t_trapezoidal(
    t, t0,
    T_acc, T_plateau, T_dec,
    delta_dot_max, delta_max,
    tau1, tau2
):
    """Trapezoidal steering profile δ(t)."""

    # time in profile
    tau_t = t - t0

    # phase boundaries
    t1 = T_acc
    t2 = T_acc + T_plateau
    t3 = T_acc + T_plateau + T_dec

    if tau_t <= 0:
        return 0.0

    # Phase 1: acceleration (0 → delta_max)
    if tau_t <= t1:
        delta = tau1 * delta_dot_max * tau_t

    # Phase 2: plateau (delta_max)
    elif tau_t <= t2:
        delta = tau1 * delta_max

    # Phase 3: deceleration (delta_max → 0)
    elif tau_t <= t3:
        time_in_dec = tau_t - t2
        delta = tau1 * delta_max + tau2 * delta_dot_max * time_in_dec

    else:
        return 0.0  # profile finished

    # enforce saturation (numerical safety)
    return np.clip(delta, -delta_max, delta_max)

def delta_dot_of_t_trapezoidal(
    t, t0,
    T_acc, T_plateau, T_dec,
    delta_dot_max, delta_max,
    tau1, tau2
):
    """Time derivative of trapezoidal steering δ̇(t)."""

    tau_t = t - t0
    
    t1 = T_acc
    t2 = T_acc + T_plateau
    t3 = T_acc + T_plateau + T_dec

    if tau_t <= 0:
        return 0.0

    # Phase 1: ramp up
    if tau_t <= t1:
        return tau1 * delta_dot_max

    # Phase 2: plateau
    elif tau_t <= t2:
        return 0.0

    # Phase 3: ramp down
    elif tau_t <= t3:
        return tau2 * delta_dot_max

    # After profile ends
    return 0.0

def simulate_bicycle_trapezoid(
    x0, y0, theta0,
    v_bar, L,
    t0, T_transition, dt,
    T_acc, T_plateau, T_dec,
    delta_max, delta_dot_max,
    tau1=1, tau2=-1
):
    # Number of time steps
    N = int(np.round((T_transition) / dt)) + 1
    t = np.linspace(t0, t0 + T_transition, N)

    # Initialize arrays
    x = np.zeros(N)
    y = np.zeros(N)
    theta = np.zeros(N)

    # Initial conditions
    x[0], y[0], theta[0] = x0, y0, theta0

    for k in range(N-1):
        tk = t[k]
        tk1 = t[k+1]

        # steering at t and t+dt
        # delta_k = delta_of_t(tk, t0, T_half, delta_dot_max, tau1, tau2)
        delta_k = delta_of_t_trapezoidal(
                            tk, t0,
                            T_acc, T_plateau, T_dec,
                            delta_dot_max, delta_max,
                            tau1, tau2
                        )
        # delta_k1 = delta_of_t(tk1, t0, T_half, delta_dot_max, tau1, tau2)
        delta_k1 = delta_of_t_trapezoidal(
                    tk1, t0,
                    T_acc, T_plateau, T_dec,
                    delta_dot_max, delta_max,
                    tau1, tau2
                )

        # predictor step (Euler)
        fx = v_bar * np.cos(theta[k])
        fy = v_bar * np.sin(theta[k])
        ftheta = (v_bar / L) * np.tan(delta_k)

        # x_pred = x[k] + dt * fx
        # y_pred = y[k] + dt * fy
        theta_pred = theta[k] + dt * ftheta

        # corrector step
        fx1 = v_bar * np.cos(theta_pred)
        fy1 = v_bar * np.sin(theta_pred)
        ftheta1 = (v_bar / L) * np.tan(delta_k1)

        x[k+1] = x[k] + 0.5 * dt * (fx + fx1)
        y[k+1] = y[k] + 0.5 * dt * (fy + fy1)
        theta[k+1] = theta[k] + 0.5 * dt * (ftheta + ftheta1)

    return t, x, y, theta

def project_points_onto_line(x, y, m, intercept, t=None):
    """
    Project points (x, y) onto the line y = m x + intercept.
    x, y can be scalars or NumPy arrays of the same shape.

    Returns
    -------
    x_proj, y_proj : projections onto the line
    d_perp         : signed perpendicular distance (positive/negative sides)
    """
    # Convert y = m x + intercept  -->  a x + b_line y + c = 0
    a = -m
    b_line = 1.0
    c = -intercept

    denom = a*a + b_line*b_line
    factor = (a*x + b_line*y + c) / denom

    x_proj = x - a * factor
    y_proj = y - b_line * factor
    d_perp = (a*x + b_line*y + c) / np.sqrt(denom)

    intersection = None
    if np.ndim(x) > 0:
        sign_change = np.where(d_perp[:-1] * d_perp[1:] < 0)[0]
        if len(sign_change) > 0:
            i = sign_change[0]
            d0, d1 = d_perp[i], d_perp[i+1]
            lam = abs(d0) / (abs(d0) + abs(d1))

            xi = x[i] + lam * (x[i+1] - x[i])
            yi = y[i] + lam * (y[i+1] - y[i])

            if t is not None:
                ti = t[i] + lam * (t[i+1] - t[i])
                intersection = (xi, yi, ti)
            else:
                intersection = (xi, yi)
    return x_proj, y_proj, d_perp, intersection


def first_time_vehicle_above_line(x, y, theta, length_front, length_rear, width, m, intercept, t=None):
    """
    Find the first time when all four corners of the vehicle
    are above the line a*x + b*y + c = 0.
    
    x, y, theta : arrays (trajectory of vehicle center)
    length, width: vehicle dimensions
    a,b,c : line coefficients
    t : optional time array
    """
    a = -m
    b = 1.0
    c = -intercept
    # local corners in body frame (center of vehicle)
    local_corners = np.array([
        [ length_front,  width/2],
        [ length_front, -width/2],
        [-length_rear,  width/2],
        [-length_rear, -width/2]
    ])

    for i in range(len(x)):
        # rotation matrix
        ct, st = np.cos(theta[i]), np.sin(theta[i])
        R = np.array([[ct, -st],
                      [st,  ct]])

        # rotate and translate corners
        corners_global = (R @ local_corners.T).T + np.array([x[i], y[i]])

        # signed distance of each corner from line
        d = a*corners_global[:,0] + b*corners_global[:,1] + c

        # check if all corners are above
        if np.all(d > 0):
            if t is not None:
                return t[i], i, corners_global
            else:
                return None, i, corners_global

    return None  # no time step found where the vehicle is fully above the line


def compute_overtake_maneuver_shortest_distance_initial_part(Bicycle, start_pose, overtake = 1, t0 = 0, time_interval = 0.05, theta_hold = 0.09):
    '''
    Compute an overtaking maneuver for a bicycle model using an analytical approach.
    Parameters:
    - Bicycle: Bicycle model parameters (L, delta_max, delta_dot_max, v_max)
    - start_pose: tuple (x0, y0, theta0)
    - end_pose: tuple (xf, yf, thetaf)
    - overtake: 1 for overtaking on the left, -1 for overtaking on the right
    - t0: initial time
    Returns:
    - ts_analytical: time samples of the maneuver
    - xs_analytical: x positions of the maneuver
    - ys_analytical: y positions of the maneuver
    '''
    ## Extract the variables 
    L = Bicycle.wheelbase
    v_max = Bicycle.v_max
    delta_dot_max = Bicycle.delta_dot_max
    R = Bicycle.max_radius 
    x0, y0, theta0 = start_pose

    theta_hold = theta0 + overtake * theta_hold

    ## Find the orientation that you want to reach at the end of phase A
    if overtake == 1:      
        tau1 = 1 
        tau2 = -1 
    else:
        tau1 = -1 
        tau2 = 1

    ## Phase A
    # Compute the delta_peak of phase A 
    dtheta_A = wrapPositiveAngle(theta_hold) - wrapPositiveAngle(theta0)
    delta_peak_A = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_A)))
    # Compute motion time of phase A
    T_A_half = delta_peak_A / delta_dot_max
    T_A = 2 * T_A_half
    # Simulate phase A
    t_A, x_A, y_A, theta_A = simulate_bicycle_trapezoid(
        x0, y0, theta0,
        v_max, L,
        0, 2*T_A_half, time_interval,
        T_A_half, delta_dot_max,
        tau1 = tau1, tau2 = tau2
    )

    ## Phase B
    N_B = 10
    x_B = np.linspace(x_A[-1], x_A[-1] + 100 * cos(theta_A[-1]), N_B)
    y_B = np.linspace(y_A[-1], y_A[-1] + 100 * sin(theta_A[-1]), N_B)
    theta_B = np.ones(N_B) * wrapPositiveAngle(atan2((y_A[-1] + 100 * sin(theta_A[-1]) - y_A[-1]), (x_A[-1] + 100 * cos(theta_A[-1]) - x_A[-1])))

    ## Adjust time vectors of the three phases
    t_A = t_A + t0

    return (
        t_A, T_A,
        x_A, y_A, theta_A,
        x_B, y_B, theta_B,
    )

def compute_overtake_maneuver_shortest_distance_curved_lane(
    Bicycle,
    start_pose,
    circle_center,
    signed_kappa1,
    overtake=1,
    t0=0,
    time_interval=0.05,
    lane_width=3.5,
    theta_hold=0.09,
):
    
    xc, yc = circle_center
    x0, y0, theta0 = start_pose
    L = Bicycle.wheelbase
    v0 = Bicycle.v_max
    delta_max = Bicycle.delta_max
    delta_dot_max = Bicycle.delta_dot_max
    
    R1 = abs(1 / signed_kappa1)
    if signed_kappa1 > 0: # turn ccw 
        turn = 1
        R2 = R1 + lane_width
    else: # turn cw
        turn = -1
        R2 = R1 - lane_width

    # Compute first part of the overtaking maneuver
    # From initial pose to segment with theta_hold
    (
        t_A, T_A,
        x_A, y_A, theta_A,
        x_B, y_B, theta_B
    ) = compute_overtake_maneuver_shortest_distance_initial_part(
        Bicycle,
        start_pose,
        overtake=overtake,
        t0=t0,
        time_interval = time_interval,
        theta_hold = theta_hold
    )

    ### Iterative procedure to find final pose
    # Intersection between segment and current circle
    m1 = (y_B[-1] - y_B[0]) / (x_B[-1] - x_B[0]) 
    b1 = y_B[0] - m1 * x_B[0]

    points = line_circle_intersection(m1, b1, xc, yc, R1, tol=1e-9)
    # Take the closest point to the starting point
    if len(points) == 2: 
        dist1 = compute_distance_two_points([x0, y0], points[0])
        dist2 = compute_distance_two_points([x0, y0], points[1])
        if dist1 < dist2:
            int_point1 = points[0]
        else:
            int_point1 = points[1]
    else:
        int_point1 = points[0]

    # Angles at each point
    theta1 = np.arctan2(int_point1[1] - yc, int_point1[0] - xc)
    theta2 = np.arctan2(y_B[0] - yc, x_B[0] - xc)

    # Smallest signed angular difference 
    # important to compute a good final pose
    dtheta = np.arctan2(np.sin(theta2 - theta1), np.cos(theta2 - theta1))

    # Intersection between segment and adjacent circle
    points = line_circle_intersection(m1, b1, xc, yc, R2, tol=1e-9)
    # Take the closest point to the starting point
    if len(points) ==2: 
        dist1 = compute_distance_two_points([x0, y0], points[0])
        dist2 = compute_distance_two_points([x0, y0], points[1])
        if dist1 < dist2:
            int_point2 = points[0]
        else:
            int_point2 = points[1]
    else:
        int_point2 = points[0]

    # Compute the new point
    theta1 = np.arctan2(int_point2[1] - yc, int_point2[0] - xc)
    theta2 = theta1 + dtheta
    x_end = xc + R2 * np.cos(theta2)
    y_end = yc + R2 * np.sin(theta2)
    theta_end = atan2(y_end - yc, x_end- xc) + turn * pi/2
    # Compute second part of the overtaking maneuver starting frome the end pose
    (
        t_A2, T_A2,
        x_A2, y_A2, theta_A2,
        x_B2, y_B2, theta_B2
    ) = compute_overtake_maneuver_shortest_distance_initial_part(
        Bicycle,
        [x_end, y_end, theta_end + pi],
        overtake = overtake,
        t0=t0,
        time_interval = time_interval,
        theta_hold = theta_end - theta0 + theta_hold
    )

    # Check steering angle constraint at the first connection point
    theta_segment = np.arctan2(y_B2[0] - y_B[0], x_B2[0] - x_B[0])
    # 1. Compute wrapped heading change
    dtheta = np.arctan2(np.sin(theta_segment - theta_A[-1]), np.cos(theta_segment - theta_A[-1]))
    # 2. Angular rate
    theta_dot = dtheta / time_interval
    # 3. Equivalent steering angle
    delta = np.arctan((L / v0) * theta_dot)
    # 4. Check steering limit
    viol_delta1 = abs(delta) > delta_max
    # Check steering angle constraint at the second connection point
    theta_segment = np.arctan2(y_B[0] - y_B2[0], x_B[0] - x_B2[0])
    dtheta = np.arctan2(np.sin(theta_segment - wrapPositiveAngle(theta_A2[-1])), np.cos(theta_segment - wrapPositiveAngle(theta_A2[-1])))
    # 2. Angular rate
    theta_dot = dtheta / time_interval
    # 3. Equivalent steering angle
    delta = np.arctan((L / v0) * theta_dot)
    # 4. Check steering limit
    viol_delta2 = abs(delta) > delta_max

    if not(viol_delta1) and not(viol_delta2):
        ## Phase A
        v_A = np.ones_like(t_A) * v0
        delta_A = [delta_of_t(ti, 0, T_A/2, delta_dot_max, overtake, -overtake) for ti in t_A]
        delta_dot_A = [delta_dot_of_t(ti, 0, T_A/2, delta_dot_max, overtake, -overtake) for ti in t_A]
        a_A = 0*np.ones_like(t_A)
        ## Phase B 
        N_B = 10
        x_B = np.linspace(x_A[-1], x_A2[-1], N_B) # includes the last point
        y_B = np.linspace(y_A[-1], y_A2[-1], N_B)
        theta_B = np.ones(N_B) * theta_segment
        v_B = np.ones(N_B) * v0
        delta_B = np.ones(N_B) * 0
        a_B = np.zeros(N_B)
        delta_dot_B = np.zeros(N_B)
        phase_B_length = compute_distance_two_points([x_A[-1], y_A[-1]], [x_A2[-1], y_A2[-1]])
        T_B = phase_B_length / v0
        ## Phase C
        x_C = x_A2[::-1]
        y_C = y_A2[::-1]
        theta_C = theta_A2[::-1] - np.pi
        v_C = np.ones_like(t_A2) * v0
        delta_C = [delta_of_t(ti, 0, T_A2/2, delta_dot_max, overtake, -overtake) for ti in t_A2]
        delta_dot_C = [delta_dot_of_t(ti, 0, T_A2/2, delta_dot_max, overtake, -overtake) for ti in t_A2]
        a_C = 0*np.ones_like(t_A2)
        ## Adjust time vectors of the three phases
        t_A = t_A + t0
        t_B = np.linspace(t0 + T_A, t0 + T_A + T_B, N_B)
        t_C = t_A2 + t0 + T_A + T_B
        T_C = T_A2
        total_motion_time = T_A + T_B + T_C

        ## Interpolate the results
        ts_analytical = np.concatenate((t_A, t_B, t_C))
        xs_analytical = np.concatenate((x_A, x_B, x_C))
        ys_analytical = np.concatenate((y_A, y_B, y_C))
        thetas_analytical = np.concatenate((theta_A, theta_B, theta_C))
        vs_analytical =  np.concatenate((v_A, v_B, v_C))
        deltas_analytical = np.concatenate((delta_A, delta_B, delta_C))
        delta_dots_analytical = np.concatenate((delta_dot_A, delta_dot_B, delta_dot_C))

        f_x_analytical = interp1d(ts_analytical, xs_analytical, kind='linear')
        f_y_analytical = interp1d(ts_analytical, ys_analytical, kind='linear')
        f_theta_analytical = interp1d(ts_analytical, thetas_analytical, kind='linear')
        f_v_analytical = interp1d(ts_analytical, vs_analytical, kind='linear')
        f_delta_analytical = interp1d(ts_analytical, deltas_analytical, kind='linear')
        f_delta_dot_analytical = interp1d(ts_analytical, delta_dots_analytical, kind='linear')

        return (
            ts_analytical, f_x_analytical, f_y_analytical, f_theta_analytical,
            f_v_analytical, f_delta_analytical, f_delta_dot_analytical,
            total_motion_time,
            x_A, y_A, theta_A,
            x_B, y_B, theta_B,
            x_C, y_C, theta_C
        )

import numpy as np

def project_points_to_circle(x, y, xc, yc, R, eps=1e-12):
    """
    Radially project 2D points (x,y) onto the circle centered at (xc,yc) with radius R.
    Returns x_proj, y_proj, theta (angle), and radial error (distance to circle minus R).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x - xc
    dy = y - yc
    r = np.hypot(dx, dy)                      # distance to center
    theta = np.arctan2(dy, dx)                # angle of each point
    # Avoid divide-by-zero at the center
    scale = np.where(r > eps, R / r, 0.0)

    x_proj = xc + dx * scale
    y_proj = yc + dy * scale

    radial_error = r - R                       # >0 outside, <0 inside, 0 on circle
    return x_proj, y_proj, theta, radial_error


def compute_smooth_transition(
        bicycle,
        start_pose,
        turn_direction = 1,
        t0 = 0,
        time_interval = 0.05,
        theta_hold = 0.09
    ):
    """
    Compute an overtaking maneuver for a bicycle model using an analytical approach.
    Parameters:
    - Bicycle: Bicycle model parameters (L, delta_max, delta_dot_max, v_max)
    - start_pose: tuple (x0, y0, theta0)
    - end_pose: tuple (xf, yf, thetaf)
    - overtake: 1 for overtaking on the left, -1 for overtaking on the right
    - t0: initial time
    Returns:
    - ts_analytical: time samples of the maneuver
    - xs_analytical: x positions of the maneuver
    - ys_analytical: y positions of the maneuver
    """
    ## Extract the variables 
    # bicycle limits and dimension
    L = bicycle.wheelbase
    v_max = bicycle.v_max
    delta_max = bicycle.delta_max
    delta_dot_max = bicycle.delta_dot_max
    # start pose
    x0, y0, theta0 = start_pose

    ## Compute the target heading
    theta_hold = theta0 + turn_direction * theta_hold

    ## Compute turn directions
    if turn_direction == 1:      
        tau1 = 1 
        tau2 = -1 
    else:
        tau1 = -1 
        tau2 = 1

    ## Compute the delta_peak of the transition
    # compute the theta difference
    dtheta_A = compute_angular_difference(wrapPositiveAngle(theta0), wrapPositiveAngle(theta_hold))
    # dtheta_A = wrapPositiveAngle(theta_hold) - wrapPositiveAngle(theta0) 
    # compute the delta peak with no saturation on delta
    delta_peak_A_unsaturated = acos(exp(-L * delta_dot_max/(2*v_max)*abs(dtheta_A)))
    # check whether delta_peak is achievable by the bicycle
    delta_peak_A = min(delta_peak_A_unsaturated, bicycle.delta_max)
    # Compute the time to reach delta_peak (half because you want to do delta0 -> delta_peak -> delta0)
    T_transition_half = delta_peak_A / delta_dot_max
    T_transition = 2 * T_transition_half
    T_plateau = 0
    # Check total transition time
    # if delta_peak_A_unsaturated > delta_peak_A:
    #     # We saturate steering → trapezoid must include a flat plateau
    #     T_transition_half = delta_peak_A / delta_dot_max
    #     # Solve for the plateau duration
    #     # theta change = integrate(v/L * tan(delta(t)) dt)
    #     # For small delta we can approximate tan(delta) ≈ delta:
    #     plateau_delta = delta_peak_A_unsaturated - delta_peak_A
    #     T_plateau = plateau_delta / (v_max / L * np.tan(delta_peak_A))
    #     T_transition = T_transition_half + T_plateau + T_transition_half

    ### NEW ###
    # Desired heading change
    dtheta_target = abs(dtheta_A)

    # Max heading change with saturated triangular profile (no plateau)
    dtheta_tri_max = (2 * v_max / (L * delta_dot_max)) * np.log(1 / np.cos(delta_max))

    if dtheta_target <= dtheta_tri_max:
        # No plateau needed: unsaturated or just-saturated triangular profile
        delta_peak_A = delta_peak_A_unsaturated  # will be <= delta_max
        T_acc = delta_peak_A / delta_dot_max
        T_plateau = 0.0
        T_dec = T_acc
    else:
        # Steering saturates and we need a plateau at delta_max
        delta_peak_A = delta_max
        T_acc = delta_peak_A / delta_dot_max
        T_dec = T_acc

        # exact formula for required plateau time
        T_plateau = (dtheta_target - dtheta_tri_max) * (L / (v_max * np.tan(delta_peak_A)))

    T_transition = T_acc + T_plateau + T_dec


    # Simulate phase A
    t_A, x_A, y_A, theta_A = simulate_bicycle_trapezoid(
    x0, y0, theta0,
    v_max, L,
    t0, T_transition, time_interval,
    T_transition_half, T_plateau, T_transition_half,
    delta_max, delta_dot_max,
    tau1, tau2
    )

    ## Phase B
    N_B = 10

    x_end = x_A[-1]
    y_end = y_A[-1]
    theta_end = theta_A[-1]

    # 100 m forward direction
    dx = 100 * np.cos(theta_end)
    dy = 100 * np.sin(theta_end)

    x_B = np.linspace(x_end, x_end + dx, N_B)
    y_B = np.linspace(y_end, y_end + dy, N_B)

    theta_B = np.ones(N_B) * wrapPositiveAngle(np.arctan2(dy, dx))

    ## Adjust time vectors of the three phases
    t_A = t_A + t0

    v_A = np.ones_like(t_A) * v_max
    # delta_A = [delta_of_t(t = ti, t0 = 0, T_half = T_A_half, delta_dot_max = delta_dot_max, tau1 = tau1, tau2 = tau2) for ti in t_A]
    delta_A = [delta_of_t_trapezoidal(
    t, t0,
    T_transition_half, T_plateau, T_transition_half,
    delta_dot_max, delta_max,
    tau1, tau2
    ) for t in t_A]
    
    # delta_dot_A = [delta_dot_of_t(t = ti, t0 = 0, T_half = T_A_half, delta_dot_max = delta_dot_max, tau1 = tau1, tau2 = tau2) for ti in t_A]
    
    delta_dot_A = [delta_dot_of_t_trapezoidal(
    t, t0,
    T_transition_half, T_plateau, T_transition_half,
    delta_dot_max, delta_max,
    tau1, tau2
    ) for t in t_A]
    
    a_A = 0*np.ones_like(t_A)

    return (
        t_A, T_transition,
        x_A, y_A, theta_A,
        x_B, y_B, theta_B,
        delta_A, delta_dot_A,
    )

    
