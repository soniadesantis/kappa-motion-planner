"""
Legacy optimal control problem (OCP) utilities for the single-corridor case.

These functions were used for trajectory generation using an OCP formulation.
They are kept for reproducibility but are not part of the main analytical planner.
"""

import casadi as cs

from ..trajectory import LinearSegmentUnicycle, CurvilinearArcUnicycle, TurnOnTheSpot

### Optimization problem to solve intersection case
def set_OCP():
    '''
    Set up the optimization problem to be solved in case of intersection, meaning that there's no need of performing 
    an arc maneuver along an intermediate arc to find the time-optimal trajectory between two corridors.

    :return: casadi function for the optimization problem to be solved in case of intersection
    :rtype: CasADi function
    '''
    opti = cs.Opti()

    x0 = opti.parameter()
    y0 = opti.parameter()
    theta0 = opti.parameter()
    xf = opti.parameter()
    yf = opti.parameter()
    thetaf = opti.parameter()
    nominal_radius = opti.parameter()
    turn1 = opti.parameter()
    turn3 = opti.parameter()

    psi1 = opti.variable()
    iota1 = opti.variable()
    psi3 = opti.variable()
    iota3 = opti.variable()

    gamma1 = theta0 + turn1*psi1 + turn1*0.5*cs.pi
    gamma3 = thetaf - turn3*psi3 - turn3*0.5*cs.pi

    xc1 = x0 + nominal_radius*cs.cos(gamma1)
    yc1 = y0 + nominal_radius*cs.sin(gamma1)
    xc3 = xf + nominal_radius*cs.cos(gamma3)
    yc3 = yf + nominal_radius*cs.sin(gamma3)

    dist_centers_sq = (xc3 - xc1)**2 + (yc3 - yc1)**2
    dl_sq = dist_centers_sq - (nominal_radius - turn1*turn3*nominal_radius)**2 # dl squared
    
    opti.subject_to(psi1 >= 0)
    opti.subject_to(iota1 >= 0)
    opti.subject_to(psi3 >= 0)
    opti.subject_to(iota3 >= 0)
    # opti.subject_to(0 <= (psi1 <= cs.pi))
    # opti.subject_to(0 <= (psi3 <= cs.pi))
    # opti.subject_to(0 <= (iota1 <= cs.pi))
    # opti.subject_to(0 <= (iota3 <= cs.pi))

    
    # opti.subject_to(-cs.pi <= (gamma1 <= cs.pi))
    # opti.subject_to(-cs.pi <= (gamma3 <= cs.pi))

    alpha1 = theta0 + turn1*(psi1 + iota1)
    alpha3 = thetaf - turn3*(psi3 + iota3)
    opti.subject_to(alpha1 == alpha3)

    # opti.subject_to(xc1 + turn1*nominal_radius*cs.sin(alpha1) + cs.sqrt(dl_sq)*cs.cos(alpha1) == xc3 + turn3*nominal_radius*cs.sin(alpha1))
    # opti.subject_to(yc1 - turn1*nominal_radius*cs.cos(alpha1) + cs.sqrt(dl_sq)*cs.sin(alpha1) == yc3 - turn3*nominal_radius*cs.cos(alpha1))
    # opti.subject_to(cs.cos(theta0 + turn1*psi1 + turn1*iota1 + turn3*psi3 + turn3*iota3 - thetaf) == 1)
    # opti.subject_to(cs.sin(theta0 + turn1*psi1 + turn1*iota1 + turn3*psi3 + turn3*iota3 - thetaf) == 0)
    
    opti.subject_to(x0 + nominal_radius*cs.cos(gamma1) + turn1*nominal_radius*cs.sin(alpha1) + cs.sqrt(dl_sq)*cs.cos(alpha1) == xf + nominal_radius * cs.cos(gamma3) + turn3*nominal_radius*cs.sin(alpha1))
    opti.subject_to(y0 + nominal_radius*cs.sin(gamma1) - turn1*nominal_radius*cs.cos(alpha1) + cs.sqrt(dl_sq)*cs.sin(alpha1) == yf + nominal_radius * cs.sin(gamma3) - turn3*nominal_radius*cs.cos(alpha1))
    opti.subject_to(theta0 + turn1* (psi1 + iota1) + turn3*(iota3 + psi3) == thetaf)


    #################
    
    total_time = (psi1*nominal_radius)**2 + (iota1*nominal_radius)**2 + dl_sq + (iota3*nominal_radius)**2 + (psi3*nominal_radius)**2
    
    opti.minimize(total_time)

    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": False, # True
                "error_on_fail": False,
                # "ipopt": {	
                #             "mu_init": 1e-5,
                #             "warm_start_init_point" : 'yes',
                #             'warm_start_bound_push' : 1e-7,
                #             'warm_start_slack_bound_push' : 1e-7,
                #             'warm_start_mult_bound_push' : 1e-7,
                #             },
            }
    tol = 1e-4
    # mu_init = 1e-8
    # bound_push = 1e-8
    options['ipopt'] = {
        "linear_solver": "ma27",
        "print_level": 0, # 3, 5
        "tol": tol,
        'sb': 'yes', # supress IPOPT banner
        # 'hessian_approximation': 'limited-memory',
        'tol' : tol,
        'dual_inf_tol' : tol,
        'compl_inf_tol' : tol,
        'constr_viol_tol' : tol,
        'acceptable_tol' : tol,
        # 'warm_start_init_point' : 'yes',
        # 'warm_start_bound_push' : bound_push,
        # 'warm_start_slack_bound_push' : bound_push,
        # 'warm_start_mult_bound_push' : bound_push,
        # 'mu_init' : mu_init
    }
    # options["print_time"] = False
    # options["ipopt.print_level"] = 0

    # options['jit'] = True
    # options['compiler'] = 'shell'
    # options['jit_temp_suffix'] = False
    # options['jit_options'] = {'flags': ['-O3', '-march=native'], 'verbose': False}

    opti.solver('ipopt', options)

    center_circle1 = cs.vertcat(xc1, yc1)
    center_circle3 = cs.vertcat(xc3, yc3)

    OCP_function = opti.to_function('OCP_fun', 
                            [x0, y0, theta0, xf, yf, thetaf, nominal_radius, turn1, turn3], 
                            [gamma1, gamma3, psi1, iota1, psi3, iota3, center_circle1, center_circle3, opti.f], 
                            ['x0', 'y0', 'theta0', 'xf', 'yf', 'thetaf', 'nominal_radius', 'turn1', 'turn3'],
                            ['gamma1', 'gamma3', 'psi1', 'iota1', 'psi3', 'iota3', 'center_circle1', 'center_circle3', 'total_time'])

    return OCP_function

def solve_OCP_function(start_pose, end_pose, vehicle, turn1, turn3):
    '''
    Solves the geometrical optimization problem set up by the set_OCP function. The returned trajectory is a sequence of five maneuvers:
    [turn on-the-spot (turn1), arc (turn1), segment, arc (turn3), turn on-the-spot (turn3)]. 
    turn1 and turn3 are assumed to be known and are given as an input to the function.

    :param start_pose: initial vehicle's pose
    :type start_pose: list of floats or np.ndarray
    :param end_pose: final vehicle's pose
    :type end_pose: list of floats or np.ndarray
    :param vehicle: considered unicycle vehicle
    :type vehicle: Unicycle
    :param turn1: initial turn direction
    :type turn1: float [-1,1]
    :param turn3: final turn direction
    :type turn3: float [-1,1]

    :return: list of five maneuvers that all connected are the time optimal trajectory
    :rtype: list of trajectory pieces [TurnOnTheSpot, CurvilinearArcUnicycle, LinearSegmentUnicycle, CurvilinearArcUnicycle, TurnOnTheSpot]
    :return: total motion time in seconds
    :rtype: float
    '''
    OCP_function = set_OCP()
    gamma1, gamma3, psi1, iota1, psi3, iota3, center_circle1, center_circle3, T = OCP_function(*start_pose, *end_pose, vehicle.max_radius, turn1, turn3)

    psi1 = float(psi1)
    iota1 = float(iota1)
    psi3 = float(psi3)
    iota3 = float(iota3)
    # print(f'iota1 = {iota1} \niota3={iota3} \npsi1={psi1} \npsi3={psi3}')
    # print(f'gamma1 = {gamma1*180/cs.pi} \ngamma3 = {gamma3*180/cs.pi}')

    center_circle1 = [float(center_circle1[0]), float(center_circle1[1])]
    center_circle3 = [float(center_circle3[0]), float(center_circle3[1])]
    T = float(T)
    
    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(center_circle1[0], center_circle1[1], center_circle3[0], center_circle3[1], turn1, turn3, vehicle.max_radius)

    # Primitive 1: turn-on-the-spot
    primitive1 = TurnOnTheSpot(x=start_pose[0], y=start_pose[1], theta0=start_pose[2], thetaf=start_pose[2] + turn1*psi1, omega=turn1*vehicle.omega_max, unicycle = vehicle, t0 = 0, samples_number=5)
    # Primitive 2: arc
    primitive2 = CurvilinearArcUnicycle(xc=center_circle1[0], yc=center_circle1[1], x0 = start_pose[0], y0 = start_pose[1], theta0 = primitive1.thetaf, xf = x1, yf = y1, thetaf = primitive1.thetaf + turn1*iota1, radius = vehicle.max_radius, turn_direction = turn1, v = vehicle.v_max, omega = turn1*vehicle.omega_max, unicycle = vehicle, t0 = primitive1.tf, samples_number = 10)
    # Primitive 3: segment
    primitive3 = LinearSegmentUnicycle(x0=x1, y0=y1, xf=x2, yf=y2, theta=primitive1.thetaf + turn1*iota1, v=vehicle.v_max, t0 = primitive2.tf, unicycle = vehicle, samples_number=10)
    # Primitive 4: arc
    primitive4 = CurvilinearArcUnicycle(xc=center_circle3[0], yc=center_circle3[1], x0 = x2, y0 = y2, theta0 = primitive3.thetaf, xf = end_pose[0], yf = end_pose[1], thetaf = primitive3.thetaf + turn3*iota3, radius = vehicle.max_radius, turn_direction = turn3, v = vehicle.v_max, omega = turn3*vehicle.omega_max, unicycle = vehicle, t0 = primitive3.tf, samples_number = 10)
    # Primitive 5: turn-on-the-spot
    primitive5 = TurnOnTheSpot(x=end_pose[0], y=end_pose[1], theta0=primitive4.thetaf, thetaf=primitive4.thetaf + turn3*psi3, omega= turn3*vehicle.omega_max, unicycle = vehicle, t0 = primitive4.tf, samples_number=5)

    maneuvers = [primitive1, primitive2, primitive3, primitive4, primitive5]
    return maneuvers, T


def compute_trajectory_intersection_case(start_pose, end_pose, unicycle):
    '''
    Compute the time-optimal trajectory in the case that an intersection is detected. 
    To find the solution, four geometrical optimization problems are solved, featuring all the combinations of initial and final turns. 
    The faster trajectory among the four solution is selected and provided as the time-optimal trajectory. 

    :param start_pose: initial pose 
    :type start_pose: list of floats
    :param end_pose: final pose
    :type end_pose: list of floats
    :param unicycle: unicycle vehicle
    :type unicycle: Unicycle

    :return: list of 5 primitives building the time-optimal solution
    :rtype: list of primitives
    '''
    maneuvers_left_left, total_time_left_left = solve_OCP_function(start_pose, end_pose, unicycle, turn1 = 1, turn3 = 1)
    maneuvers_left_right, total_time_left_right = solve_OCP_function(start_pose, end_pose, unicycle, turn1 = 1, turn3 = 1)
    maneuvers_right_left, total_time_right_left = solve_OCP_function(start_pose, end_pose, unicycle, turn1 = 1, turn3 = 1)
    maneuvers_right_right, total_time_right_right = solve_OCP_function(start_pose, end_pose, unicycle, turn1 = 1, turn3 = 1)
    total_time = min(total_time_left_left, total_time_left_right, total_time_right_left, total_time_right_right) 
    if total_time == total_time_left_left:
        maneuvers = maneuvers_left_left
    elif total_time == total_time_left_right:
        maneuvers = maneuvers_left_right
    elif total_time == total_time_right_left:
        maneuvers = maneuvers_right_left
    elif total_time == total_time_right_right:
        maneuvers = maneuvers_right_right
    return maneuvers


def ocp_function_one_corridor(corridor, start_pose, end_pose, unicycle):
    from rockit import Ocp, FreeTime, FreeGrid, MultipleShooting
    import casadi as cs

    x0, y0, theta0 = start_pose
    xf, yf, thetaf = end_pose
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max

    shrunken_corridor1 = corridor.shrink(unicycle.width * 0.5)
    WT = shrunken_corridor1.W.T

    # Initialize problem parameters
    T = compute_distance_two_points([x0,y0], [xf, yf])/v_max# Time horizon
    N = 10 # number of control intervals
    M = 8

    min_grid = T/N - 2 # minimum time step for the grid
    max_grid = T/N + 2 # maximum time step for the grid

    t0 = 0 # constraint to start at t0 = 0
    tf = FreeTime(T)

    # Create initial guess for states and controls
    time_grid_initial_guess = np.linspace(0, T, N + 1)
    x_initial_guess = np.linspace(x0,xf,N+1)
    y_initial_guess = np.linspace(y0,yf,N+1)
    theta_initial_guess = np.full(N + 1, corridor.tilt)
    v_initial_guess     = np.full(N, v_max)
    omega_initial_guess = np.zeros(N)

    # Define the optimal control problem
    ocp = Ocp(t0 = t0, T = tf)

    # Define the states and controls
    x = ocp.state()
    y = ocp.state()
    theta = ocp.state()
    v = ocp.control()
    omega = ocp.control()

    # Define system's dynamics
    ocp.set_der(x, v*cs.cos(theta))
    ocp.set_der(y, v*cs.sin(theta))
    ocp.set_der(theta, omega)

    # Define boundaries for states and controls
    ocp.subject_to(ocp.at_t0(x) == x0)
    ocp.subject_to(ocp.at_t0(y) == y0)
    ocp.subject_to(ocp.at_t0(np.cos(theta)) == np.cos(theta0))
    ocp.subject_to(ocp.at_t0(np.sin(theta)) == np.sin(theta0))

    ocp.subject_to(ocp.at_tf(x) == xf)
    ocp.subject_to(ocp.at_tf(y) == yf)
    ocp.subject_to(ocp.at_tf(np.cos(theta)) == np.cos(thetaf))
    ocp.subject_to(ocp.at_tf(np.sin(theta)) == np.sin(thetaf))

    ocp.subject_to(0 <= (v <= v_max))
    ocp.subject_to(-omega_max <= (omega <= omega_max))

    # Collision avoidance constraints
    p = cs.vertcat(x, y, 1)
    ocp.subject_to(WT @ p <= 0, grid = 'integrator')

    # Define objective function
    ocp.add_objective(ocp.T)

    # ocp.method(MultipleShooting(N=N, M=M, intg='rk', grid=FreeGrid(min = min_grid, max = max_grid)))
    ocp.method(MultipleShooting(N=N, M=M, intg='rk'))

    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": False, # True
                "error_on_fail": False,
                "ipopt": {	"linear_solver": "mumps", #"ma27",
                            "print_level": 0, # 5
                            "tol": 1e-6,
                            'sb': 'yes', # supress IPOPT banner
                            # 'hessian_approximation': 'limited-memory',
                            },
            }
    # options["print_time"] = False
    # options["ipopt.print_level"] = 0

    # options['jit'] = True
    # options['compiler'] = 'shell'
    # options['jit_temp_suffix'] = False
    # options['jit_options'] = {'flags': ['-O3', '-march=native'], 'verbose': False}

    # Define solver
    ocp.solver('ipopt', options)

    try:
        sol = ocp.solve()
    except:
        sol = ocp.non_converged_solution
        ocp.show_infeasibilities(1e-5)

    # Extract solution
    _, x_sampled = sol.sample(x, grid='integrator')   
    _, y_sampled = sol.sample(y, grid='integrator')
    _, theta_sampled = sol.sample(theta, grid='integrator')
    _, v_sampled = sol.sample(v, grid='integrator')
    time_grid, omega_sampled = sol.sample(omega, grid='integrator')

    _, x_sampled_control = sol.sample(x, grid='control')
    _, y_sampled_control = sol.sample(y, grid='control')
    _, theta_sampled_control = sol.sample(theta, grid='control')
    _, v_sampled_control = sol.sample(v, grid='control')
    control_time_grid, omega_sampled_control = sol.sample(omega, grid='control')

    total_motion_time = sol.value(ocp.T)
    sampler = ocp.sampler([x, y, theta, v, omega])

    gist = sol.value(ocp.gist)

    # Instantiate UnicycleTrajectoryOptimal
    unicycle_trajectory = UnicycleTrajectoryOptimal(time_grid, x_sampled, y_sampled, theta_sampled,
                                                    v_sampled, omega_sampled,
                                                    control_time_grid, x_sampled_control, y_sampled_control,
                                                    theta_sampled_control, v_sampled_control,
                                                    omega_sampled_control, total_motion_time, 
                                                    x_initial_guess, y_initial_guess, theta_initial_guess, 
                                                    v_initial_guess, omega_initial_guess, 
                                                    time_grid_initial_guess, sampler, gist)
    
    

    return unicycle_trajectory


def ocp_function_free_space(start_pose, end_pose, unicycle):
    from rockit import Ocp, FreeTime, FreeGrid, MultipleShooting
    import casadi as cs

    x0, y0, theta0 = start_pose
    xf, yf, thetaf = end_pose
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max

    # shrunken_corridor1 = corridor.shrink(unicycle.width * 0.5)
    # WT = shrunken_corridor1.W.T

    # Initialize problem parameters
    T = compute_distance_two_points([x0,y0], [xf, yf])/v_max# Time horizon
    N = 10 # number of control intervals
    M = 8

    min_grid = T/N - 0.1 # minimum time step for the grid
    max_grid = T/N + 0.1 # maximum time step for the grid

    t0 = 0 # constraint to start at t0 = 0
    tf = FreeTime(T)

    # Create initial guess for states and controls
    time_grid_initial_guess = np.linspace(0, T, N + 1)
    x_initial_guess = np.linspace(x0,xf,N+1)
    y_initial_guess = np.linspace(y0,yf,N+1)
    theta_initial_guess = np.full(N + 1, atan2((yf - y0), (xf - x0)))
    v_initial_guess     = np.full(N, v_max)
    omega_initial_guess = np.zeros(N)

    # Define the optimal control problem
    ocp = Ocp(t0 = t0, T = tf)

    # Define the states and controls
    x = ocp.state()
    y = ocp.state()
    theta = ocp.state()
    v = ocp.control()
    omega = ocp.control()

    # Define system's dynamics
    ocp.set_der(x, v*cs.cos(theta))
    ocp.set_der(y, v*cs.sin(theta))
    ocp.set_der(theta, omega)

    # Define boundaries for states and controls
    ocp.subject_to(ocp.at_t0(x) == x0)
    ocp.subject_to(ocp.at_t0(y) == y0)
    ocp.subject_to(ocp.at_t0(np.cos(theta)) == np.cos(theta0))
    ocp.subject_to(ocp.at_t0(np.sin(theta)) == np.sin(theta0))

    ocp.subject_to(ocp.at_tf(x) == xf)
    ocp.subject_to(ocp.at_tf(y) == yf)
    ocp.subject_to(ocp.at_tf(np.cos(theta)) == np.cos(thetaf))
    ocp.subject_to(ocp.at_tf(np.sin(theta)) == np.sin(thetaf))

    ocp.subject_to(0 <= (v <= v_max))
    ocp.subject_to(-omega_max <= (omega <= omega_max))

    # Collision avoidance constraints
    # p = cs.vertcat(x, y, 1)
    # ocp.subject_to(WT @ p <= 0, grid = 'integrator')

    # Define objective function
    ocp.add_objective(ocp.T)

    ocp.method(MultipleShooting(N=N, M=M, intg='rk', grid=FreeGrid(min = min_grid, max = max_grid)))
    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": False, # True
                "error_on_fail": False,
                "ipopt": {	"linear_solver": "mumps", #"ma27",
                            "print_level": 0, # 5
                            "tol": 1e-6,
                            'sb': 'yes', # supress IPOPT banner
                            # 'hessian_approximation': 'limited-memory',
                            },
            }
    # options["print_time"] = False
    # options["ipopt.print_level"] = 0

    # options['jit'] = True
    # options['compiler'] = 'shell'
    # options['jit_temp_suffix'] = False
    # options['jit_options'] = {'flags': ['-O3', '-march=native'], 'verbose': False}

    # Define solver
    ocp.solver('ipopt', options)

    try:
        sol = ocp.solve()
    except:
        sol = ocp.non_converged_solution
        ocp.show_infeasibilities(1e-5)

    # Extract solution
    _, x_sampled = sol.sample(x, grid='integrator')   
    _, y_sampled = sol.sample(y, grid='integrator')
    _, theta_sampled = sol.sample(theta, grid='integrator')
    _, v_sampled = sol.sample(v, grid='integrator')
    time_grid, omega_sampled = sol.sample(omega, grid='integrator')

    _, x_sampled_control = sol.sample(x, grid='control')
    _, y_sampled_control = sol.sample(y, grid='control')
    _, theta_sampled_control = sol.sample(theta, grid='control')
    _, v_sampled_control = sol.sample(v, grid='control')
    control_time_grid, omega_sampled_control = sol.sample(omega, grid='control')

    total_motion_time = sol.value(ocp.T)
    sampler = ocp.sampler([x, y, theta, v, omega])

    gist = sol.value(ocp.gist)

    # Instantiate UnicycleTrajectoryOptimal
    unicycle_trajectory = UnicycleTrajectoryOptimal(time_grid, x_sampled, y_sampled, theta_sampled,
                                                    v_sampled, omega_sampled,
                                                    control_time_grid, x_sampled_control, y_sampled_control,
                                                    theta_sampled_control, v_sampled_control,
                                                    omega_sampled_control, total_motion_time, 
                                                    x_initial_guess, y_initial_guess, theta_initial_guess, 
                                                    v_initial_guess, omega_initial_guess, 
                                                    time_grid_initial_guess, sampler, gist)
    
    

    return unicycle_trajectory