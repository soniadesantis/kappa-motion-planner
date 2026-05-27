import numpy as np
from arena import get_corner_point, get_vehicle_vertices, compute_turn_direction, measure_elapsed_time, UnicycleTrajectoryOptimal, compute_end_pose, compute_start_pose, Timer
import arena

class OptimalMotionPlanner:
    '''
    OptimalMotionPlanner object to compute time-optimal trajectories within two orridors for unicycle vehicles.
    '''

    def __init__(self, corridor1, corridor2, start_pose, end_pose, unicycle):
        '''
        Constructor.

        :param corridor1: first corridor
        :type corridor1: CorridorWorld

        :param corridor2: second corridor
        :type corridor2: CorridorWorld

        :param start_pose: initial pose of the vehicle
        :type start_pose: np.ndarray

        :param end_pose: final pose of the vehicle
        :type end_pose: np.ndarray

        :param unicycle: unicycle vehicle
        :type unicycle: Unicycle
        '''

        self.corridor1 = corridor1
        self.corridor2 = corridor2
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.unicycle = unicycle

        self.comp_time = 0

        self.set_OCP()
        self.build_OCP_function()
    
    def set_OCP(self):
        '''
        Set the optimal control problem for the motion planning task.
        This method is called to initialize the OptimalMotionPlanner object.
        '''

        # Parameters: start_pose, end_pose, v_max, omega_max, corridor1.W, corridor1.tilt, corridor2.W, corridor2.tilt

        from rockit import Ocp, FreeTime
        from casadi import DM

        #Initialize stages parameters
        T1 = 10
        N1 = 20

        T2 = 10
        N2 = 20

        M1 = 4
        M2 = 4
        self.N1, self.N2, self.M1, self.M2 = N1, N2, M1, M2

        min_grid = 0.001
        max_grid = 0.6

        t0_stage1 = 0
        tf_stage1 = FreeTime(T1)

        t0_stage2 = FreeTime(T1)
        tf_stage2 = FreeTime(T1 + T2)

        #Define optimal control problem
        ocp = Ocp()

        ##########################
        # Stage 1
        ##########################
        stage1, x1, y1, theta1, v1, omega1, W1_T, \
        vehicle_v_max1, vehicle_omega_max1, \
        vehicle_width1, vehicle_length1 = self.create_stage(ocp, t0_stage1, tf_stage1, N1, M1, min_grid, max_grid)

        # Parameters of stage1
        x0 = stage1.parameter()
        y0 = stage1.parameter()
        theta0 = stage1.parameter()

        # Initial conditions (constraints of stage1)
        stage1.subject_to(stage1.at_t0(x1) == x0)
        stage1.subject_to(stage1.at_t0(y1) == y0)
        stage1.subject_to(stage1.at_t0(theta1) == theta0)

        ##########################
        # Stage 2
        ##########################
        stage2, x2, y2, theta2, v2, omega2, W2_T, \
        vehicle_v_max2, vehicle_omega_max2, \
        vehicle_width2, vehicle_length2 = self.create_stage(ocp, t0_stage2, tf_stage2, N2, M2, min_grid, max_grid)
        
        # Stitch stages
        self.stitch_stages(ocp, stage1, stage2)

        # Parameters of stage2
        xf = stage2.parameter()
        yf = stage2.parameter()
        thetaf = stage2.parameter()

        # Final conditions (constraints of stage2)
        stage2.subject_to(stage2.at_tf(x2) == xf)
        stage2.subject_to(stage2.at_tf(y2) == yf)
        stage2.subject_to(stage2.at_tf(theta2) == thetaf)
        
        # Objective function: minimize time
        ocp.add_objective(stage1.T + stage2.T)

        ###########################################
        # Dummies for params
        stage1.set_value(x0, 0)
        stage1.set_value(y0, 0)
        stage1.set_value(theta0, 0)
        stage1.set_value(W1_T, DM.zeros(4,3))
        stage2.set_value(xf, 0)
        stage2.set_value(yf, 0)
        stage2.set_value(thetaf, 0)
        stage2.set_value(W2_T, DM.zeros(4,3))
        
        stage1.set_value(vehicle_v_max1, 0)
        stage1.set_value(vehicle_omega_max1, 0)
        stage1.set_value(vehicle_width1, 0)
        stage1.set_value(vehicle_length1, 0)
        stage2.set_value(vehicle_v_max2, 0)
        stage2.set_value(vehicle_omega_max2, 0)
        stage2.set_value(vehicle_width2, 0)
        stage2.set_value(vehicle_length2, 0)
        ###########################################

        # Pick a solution method
        options = { "expand": True,
                    "verbose": False,
                    "print_time": False, # True
                    "error_on_fail": False,
                    "ipopt": {	"linear_solver": "ma27",
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

        # Add attributes to the object
        self.ocp = ocp
        self.stage1 = stage1
        self.stage2 = stage2
        self.x1 = x1    # x along stage1
        self.x2 = x2    # x along stage2
        self.y1 = y1    # y along stage1
        self.y2 = y2    # y along stage2
        self.theta1 = theta1    # theta along stage1
        self.theta2 = theta2    # theta along stage2
        self.v1 = v1    # v along stage1
        self.v2 = v2    # v along stage2
        self.omega1 = omega1    # omega along stage1
        self.omega2 = omega2    # omega along stage2

        self.x0 = x0   
        self.y0 = y0    
        self.theta0 = theta0
        self.xf = xf
        self.yf = yf
        self.thetaf = thetaf

        self.W1_T = W1_T
        self.W2_T = W2_T

        self.vehicle_v_max1 = vehicle_v_max1
        self.vehicle_omega_max1 = vehicle_omega_max1
        self.vehicle_width1 = vehicle_width1
        self.vehicle_length1 = vehicle_length1

        self.vehicle_v_max2 = vehicle_v_max2
        self.vehicle_omega_max2 = vehicle_omega_max2
        self.vehicle_width2 = vehicle_width2
        self.vehicle_length2 = vehicle_length2

        #########################################
        # Create samplers
        #########################################
        self.sampler1 = stage1.sampler([x1, y1, theta1, v1, omega1])
        self.sampler2 = stage2.sampler([x2, y2, theta2, v2, omega2])

    def build_OCP_function(self):
        '''
        Build the OCP function for the optimal control problem.
        This method is called to initialize the OptimalMotionPlanner object.
        '''
        
        from casadi import vertcat

        ocp = self.ocp
        stage1 = self.stage1
        stage2 = self.stage2

        #######################################
        # Placeholders for initial guess
        #######################################
        _, x1_samp = stage1.sample(self.x1, grid="control-")
        _, y1_samp = stage1.sample(self.y1, grid="control-")
        _, theta1_samp = stage1.sample(self.theta1, grid="control-")
        _, v1_samp = stage1.sample(self.v1, grid="control-")
        _, omega1_samp = stage1.sample(self.omega1, grid="control-")

        _, x2_samp = stage2.sample(self.x2, grid="control-")
        _, y2_samp = stage2.sample(self.y2, grid="control-")
        _, theta2_samp = stage2.sample(self.theta2, grid="control-")
        _, v2_samp = stage2.sample(self.v2, grid="control-")
        _, omega2_samp = stage2.sample(self.omega2, grid="control-")

        initial_guess_list = [x1_samp, y1_samp, theta1_samp, v1_samp, omega1_samp, x2_samp, y2_samp, theta2_samp, v2_samp, omega2_samp]

        #######################################
        # Define inputs
        #######################################
        # Get values of parameters from stages
        x0s = stage1.value(self.x0)
        y0s = stage1.value(self.y0)
        theta0s = stage1.value(self.theta0)
        W1_Ts = stage1.value(self.W1_T)

        xfs = stage2.value(self.xf)
        yfs = stage2.value(self.yf)
        thetafs = stage2.value(self.thetaf)
        W2_Ts = stage2.value(self.W2_T)

        vmax1s = stage1.value(self.vehicle_v_max1)
        omegamax1s = stage1.value(self.vehicle_omega_max1)
        width1s = stage1.value(self.vehicle_width1)
        length1s = stage1.value(self.vehicle_length1)

        vmax2s = stage2.value(self.vehicle_v_max2)
        omegamax2s = stage2.value(self.vehicle_omega_max2)
        width2s = stage2.value(self.vehicle_width2)
        length2s = stage2.value(self.vehicle_length2)

        # Concatenate some parameters
        start_pose_s = vertcat(x0s, y0s, theta0s)
        end_pose_s = vertcat(xfs, yfs, thetafs)
        veh_vars1_s = vertcat(vmax1s, omegamax1s, width1s, length1s)
        veh_vars2_s = vertcat(vmax2s, omegamax2s, width2s, length2s)

        input_vector = [start_pose_s, end_pose_s, W1_Ts, W2_Ts, veh_vars1_s, veh_vars2_s] + initial_guess_list

        #######################################
        # Define outputs
        #######################################
        T1 = ocp.value(stage1.T)
        T2 = ocp.value(stage2.T)

        output_vector = [ocp._method.opti.x, ocp.gist, T1, T2]

        #######################################
        # Set function
        #######################################
        self.OCP_function = ocp.to_function('OCP_fun', input_vector, output_vector, {'record_time': False})

        return self.OCP_function

    def create_stage(self, ocp, t0, T, N, M, min_grid = 0.0001, max_grid = 5):
        '''
        Create a stage for the optimal control problem with a free time grid.

        :param ocp: Optimal control problem
        :type ocp: Ocp

        :param t0: initial time
        :type t0: float

        :param T: final time for the stage
        :type T: float

        :param N: number of control intervals
        :type N: int

        :param M: number of integration steps
        :type M: int

        :param min_grid: minimum grid size
        :type min_grid: float

        :param max_grid: maximum grid size
        :type max_grid: float
        '''

        from rockit import FreeGrid, MultipleShooting
        import casadi as cs

        stage = ocp.stage(t0=t0, T=T)

        # Define states and controls
        x = stage.state()
        y = stage.state()
        theta = stage.state()

        omega = stage.control()
        v = stage.control()

        # Define parameters
        v_max = stage.parameter()
        omega_max = stage.parameter()
        width = stage.parameter()   # width of the vehicle
        length = stage.parameter()  # length of the vehicle

        # Define dynamics
        stage.set_der(x, v*cs.cos(theta))
        stage.set_der(y, v*cs.sin(theta))
        stage.set_der(theta, omega)

        # Define constraints
        stage.subject_to(0 <= (v <= v_max))
        stage.subject_to(-omega_max <= (omega <= omega_max))

        # Create a vector with the vehicle center [x, y, 1]T
        p = cs.vertcat(x, y, 1)

        # Create a matrix describing the corridor
        W_mat = stage.parameter(4,3) # W transposed is a 4x3 matrix

        # Collision avoidance constraint
        stage.subject_to( W_mat @ p <= 0, grid = 'integrator')

        stage.method(MultipleShooting(N=N, M=M, intg='rk', grid=FreeGrid(min = min_grid, max = max_grid)))

        return stage, x, y, theta, v, omega, W_mat, v_max, omega_max, width, length

    def stitch_stages(self, ocp, stage1, stage2):
        '''
        Function to stitch two subseqeunt stages of the optimal control problem.

        :param ocp: Optimal control problem
        :type ocp: Ocp

        :param stage1: first stage
        :type stage1: Stage

        :param stage2: second stage
        :type stage2: Stage
        '''

        # Stitch time
        ocp.subject_to(stage1.tf == stage2.t0)

        # Stitch states
        for i in range(len(stage1.states)):
            ocp.subject_to(stage2.at_t0(stage2.states[i])
                        == stage1.at_tf(stage1.states[i]))

    def compute_initial_guess(self, start_pose, end_pose, corridor1, corridor2, unicycle):
        '''
        Compute the initial guess for stage1 and stage2 of the optimal control problem.
        The initial guess is computed for x1, y1, theta1, v1, omega1, T1, x2, y2, theta2, v2, omega2, T2
        and stored as attributes of the OptimalMotionPlanner object.

        :param start_pose: initial pose of the vehicle
        :type start_pose: np.ndarray

        :param end_pose: final pose of the vehicle
        :type end_pose: np.ndarray

        :param corridor1: first corridor
        :type corridor1: CorridorWorld

        :param corridor2: second corridor
        :type corridor2: CorridorWorld

        :param unicycle: unicycle vehicle
        :type unicycle: Unicycle
        '''

        vector1, vector2 = corridor1.vector, corridor2.vector
        turn_direction = compute_turn_direction(vector1, vector2)
        corner_point = get_corner_point(corridor1, corridor2, turn_direction)

        # Compute the x and y coordinated of the initial guess for the two stages
        self.x1_initial_guess, self.y1_initial_guess = self.get_initial_guess_path(corridor1, self.N1, unicycle, start_pos = start_pose, end_pos = corner_point)
        self.x2_initial_guess, self.y2_initial_guess = self.get_initial_guess_path(corridor2, self.N2, unicycle, start_pos = corner_point, end_pos = end_pose)

        # Initial guess for time
        self.T1_initial_guess = corridor1.height / unicycle.v_max
        self.T2_initial_guess = corridor2.height / unicycle.v_max

        # Initial guess for the other variables
        self.theta1_initial_guess = np.array([corridor1.tilt])
        self.theta2_initial_guess = np.array([corridor2.tilt])
        self.v1_initial_guess = np.array([unicycle.v_max])
        self.v2_initial_guess = np.array([unicycle.v_max])
        self.omega1_initial_guess = np.array([0])
        self.omega2_initial_guess = np.array([0])

    def solve_OCP(self, start_pose, end_pose, v_max, omega_max, vehicle_width, vehicle_length, corridor1_WT, corridor1_tilt, corridor2_WT, corridor2_tilt):
        '''
        Solve the optimal control problem for the motion planning task.

        :param start_pose: initial pose of the vehicle
        :type start_pose: np.ndarray

        :param end_pose: final pose of the vehicle
        :type end_pose: np.ndarray

        :param v_max: maximum linear velocity of the vehicle
        :type v_max: float

        :param omega_max: maximum angular velocity of the vehicle
        :type omega_max: float

        :param vehicle_width: width of the vehicle
        :type vehicle_width: float

        :param vehicle_length: length of the vehicle
        :type vehicle_length: float 

        :param corridor1_WT: matrix W transposed for the first corridor
        :type corridor1_WT: np.ndarray

        :param corridor1_tilt: tilt of the first corridor
        :type corridor1_tilt: float

        :param corridor2_WT: matrix W transposed for the second corridor
        :type corridor2_WT: np.ndarray

        :param corridor2_tilt: tilt of the second corridor
        :type corridor2_tilt: float
        '''

        ########################################
        # Set parameter values
        ########################################
        self.stage1.set_value(self.x0, start_pose[0])
        self.stage1.set_value(self.y0, start_pose[1])
        self.stage1.set_value(self.theta0, start_pose[2])
        self.stage1.set_value(self.W1_T, corridor1_WT)
        self.stage2.set_value(self.xf, end_pose[0])
        self.stage2.set_value(self.yf, end_pose[1])
        self.stage2.set_value(self.thetaf, end_pose[2])
        self.stage2.set_value(self.W2_T, corridor2_WT)
        
        self.stage1.set_value(self.vehicle_v_max1, v_max)
        self.stage1.set_value(self.vehicle_omega_max1, omega_max)
        self.stage1.set_value(self.vehicle_width1, vehicle_width)
        self.stage1.set_value(self.vehicle_length1, vehicle_length)
        self.stage2.set_value(self.vehicle_v_max2, v_max)
        self.stage2.set_value(self.vehicle_omega_max2, omega_max)
        self.stage2.set_value(self.vehicle_width2, vehicle_width)
        self.stage2.set_value(self.vehicle_length2, vehicle_length)

        #########################################
        # Set initial guesses
        #########################################
        self.stage1.set_initial(self.x1, self.x1_initial_guess)
        self.stage1.set_initial(self.y1, self.y1_initial_guess)
        self.stage1.set_initial(self.theta1, self.theta1_initial_guess)
        self.stage1.set_initial(self.v1, self.v1_initial_guess)
        self.stage1.set_initial(self.omega1, self.omega1_initial_guess)

        self.stage2.set_initial(self.x2, self.x2_initial_guess)
        self.stage2.set_initial(self.y2, self.y2_initial_guess)
        self.stage2.set_initial(self.theta2, self.theta2_initial_guess)
        self.stage2.set_initial(self.v2, self.v2_initial_guess)
        self.stage2.set_initial(self.omega2, self.omega2_initial_guess)

        #########################################
        # Solve
        #########################################
        try:
            sol = self.ocp.solve()
        except:
            sol = self.ocp.non_converged_solution
            self.ocp.show_infeasibilities(1e-5)

        #########################################
        # Post-process solution
        #########################################
        _, xs1 = sol(self.stage1).sample(self.x1, grid='integrator')
        _, xs2 = sol(self.stage2).sample(self.x2, grid='integrator')
        x_sampled = np.concatenate((xs1, xs2))

        _, ys1 = sol(self.stage1).sample(self.y1, grid='integrator')
        _, ys2 = sol(self.stage2).sample(self.y2, grid='integrator')
        y_sampled = np.concatenate((ys1, ys2))

        _, thetas1 = sol(self.stage1).sample(self.theta1, grid='integrator')
        _, thetas2 = sol(self.stage2).sample(self.theta2, grid='integrator')
        theta_sampled = np.concatenate((thetas1, thetas2))

        _, vs1 = sol(self.stage1).sample(self.v1, grid='integrator')
        _, vs2 = sol(self.stage2).sample(self.v2, grid='integrator')
        v_sampled = np.concatenate((vs1, vs2))

        ts1, omegas1 = sol(self.stage1).sample(self.omega1, grid='integrator')
        ts2, omegas2 = sol(self.stage2).sample(self.omega2, grid='integrator')
        omega_sampled = np.concatenate((omegas1, omegas2))

        time_grid = np.concatenate((ts1, ts2))

        _, xs1_control = sol(self.stage1).sample(self.x1, grid='control')
        _, xs2_control = sol(self.stage2).sample(self.x2, grid='control')
        x_sampled_control = np.concatenate((xs1_control, xs2_control))

        _, ys1_control = sol(self.stage1).sample(self.y1, grid='control')
        _, ys2_control = sol(self.stage2).sample(self.y2, grid='control')
        y_sampled_control = np.concatenate((ys1_control, ys2_control))

        _, thetas1_control = sol(self.stage1).sample(self.theta1, grid='control')
        _, thetas2_control = sol(self.stage2).sample(self.theta2, grid='control')
        theta_sampled_control = np.concatenate((thetas1_control, thetas2_control))

        _, vs1_control = sol(self.stage1).sample(self.v1, grid='control')
        _, vs2_control = sol(self.stage2).sample(self.v2, grid='control')
        v_sampled_control = np.concatenate((vs1_control, vs2_control))

        ts1_control, omegas1_control = sol(self.stage1).sample(self.omega1, grid='control')
        ts2_control, omegas2_control = sol(self.stage2).sample(self.omega2, grid='control')
        omega_sampled_control = np.concatenate((omegas1_control, omegas2_control))

        control_time_grid = np.concatenate((ts1_control, ts2_control))

        total_time = sol.value(self.stage1.T + self.stage2.T)

        x_initial_guess = np.concatenate((self.x1_initial_guess, self.x2_initial_guess))
        y_initial_guess = np.concatenate((self.y1_initial_guess, self.y2_initial_guess))

        # Create the object UniCycleTrajectoryOptimal
        unicycle_trajectory = UnicycleTrajectoryOptimal(time_grid, x_sampled, y_sampled, theta_sampled,
                                                        v_sampled, omega_sampled, control_time_grid,
                                                        x_sampled_control, y_sampled_control,
                                                        theta_sampled_control, v_sampled_control,
                                                        omega_sampled_control, x_initial_guess,
                                                        y_initial_guess, total_time)

        return unicycle_trajectory

    def solve_OCP_function(self, start_pose, end_pose, vehicle, corridor1, corridor2): 
        '''
        Solve the function of the optimal control problem for the motion planning task.

        :param start_pose: initial pose of the vehicle
        :type start_pose: np.ndarray

        :param end_pose: final pose of the vehicle
        :type end_pose: np.ndarray

        :param vehicle: unicycle vehicle
        :type vehicle: Unicycle

        :param corridor1: first corridor
        :type corridor1: CorridorWorld

        :param corridor2: second corridor
        :type corridor2: CorridorWorld
        '''

        corridor1_margin = arena.CorridorWorld(width = corridor1.width - vehicle.width,
                                               height = corridor1.height,
                                               center = corridor1.center, 
                                               tilt = corridor1.tilt)
        
        corridor2_margin = arena.CorridorWorld(width = corridor2.width - vehicle.width,
                                               height = corridor2.height,
                                               center = corridor2.center,
                                               tilt = corridor2.tilt)

        v_max, omega_max = vehicle.v_max, vehicle.omega_max
        vehicle_width, vehicle_length = vehicle.width, vehicle.length
        corridor1_WT, corridor1_tilt = corridor1_margin.W.T, corridor1_margin.tilt
        corridor2_WT, corridor2_tilt = corridor2_margin.W.T, corridor2_margin.tilt

        #########################################
        # Update initial guess
        #########################################
        self.compute_initial_guess(start_pose, end_pose, corridor1, corridor2, vehicle)

        #########################################
        # Prepare inputs
        #########################################
        veh_vars = [v_max, omega_max, vehicle_width, vehicle_length]

        parameter_values = [start_pose, end_pose, corridor1_WT, corridor2_WT, veh_vars, veh_vars]

        initial_guess = [self.x1_initial_guess, self.y1_initial_guess,
                         corridor1_tilt, v_max, 0, self.x2_initial_guess,
                         self.y2_initial_guess, corridor2_tilt, v_max, 0]
        
        input_vector = parameter_values + initial_guess

        #########################################
        # Solve OCP
        #########################################
        with Timer() as timer:
            opti_x, gist, T1, T2 = self.OCP_function(*input_vector)
        self.comp_time = timer()

        #########################################
        # Post-process solution
        #########################################
        T1, T2 = float(T1), float(T2)
        total_time = T1 + T2

        ts1_control = np.linspace(0, T1, self.N1)
        ts1 = np.linspace(0, T1, self.N1*self.M1)
        ts2_control = np.linspace(T1 + T2/self.N1, total_time, self.N1)
        ts2 = np.linspace(T1 + T2/self.N1, total_time, self.N1*self.M1)

        [xs1_control, ys1_control, thetas1_control, vs1_control, omegas1_control] = self.sampler1(gist, ts1_control)
        [xs2_control, ys2_control, thetas2_control, vs2_control, omegas2_control] = self.sampler2(gist, ts2_control)
        [xs1, ys1, thetas1, vs1, omegas1] = self.sampler1(gist, ts1)
        [xs2, ys2, thetas2, vs2, omegas2] = self.sampler2(gist, ts2)

        x_sampled = np.concatenate((xs1, xs2))
        y_sampled = np.concatenate((ys1, ys2))
        theta_sampled = np.concatenate((thetas1, thetas2))
        v_sampled = np.concatenate((vs1, vs2))
        omega_sampled = np.concatenate((omegas1, omegas2))
        time_grid = np.concatenate((ts1, ts2))

        x_sampled_control = np.concatenate((xs1_control, xs2_control))
        y_sampled_control = np.concatenate((ys1_control, ys2_control))
        theta_sampled_control = np.concatenate((thetas1_control, thetas2_control))
        v_sampled_control = np.concatenate((vs1_control, vs2_control))
        omega_sampled_control = np.concatenate((omegas1_control, omegas2_control))
        control_time_grid = np.concatenate((ts1_control, ts2_control))

        x_initial_guess = np.concatenate((self.x1_initial_guess, self.x2_initial_guess))
        y_initial_guess = np.concatenate((self.y1_initial_guess, self.y2_initial_guess))

        unicycle_trajectory = UnicycleTrajectoryOptimal(time_grid, x_sampled, y_sampled, theta_sampled, v_sampled, omega_sampled, control_time_grid, x_sampled_control, y_sampled_control, theta_sampled_control, v_sampled_control, omega_sampled_control, x_initial_guess, y_initial_guess, total_time)

        self.gist = gist
        self.T1_sol = T1
        self.T2_sol = T2

        return unicycle_trajectory


    def get_initial_guess_path(self, corridor1, N, unicycle, start_pos = None, end_pos = None, corridor2 = None):
        '''
        Compute the initial guess for the path of the vehicle within a corridor.

        :param corridor1: corridor
        :type corridor1: CorridorWorld

        :param N: number of points in the initial guess
        :type N: int

        :param unicycle: unicycle vehicle
        :type unicycle: Unicycle

        :param start_pos: initial pose of the vehicle
        :type start_pos: np.ndarray

        :param end_pos: final pose of the vehicle
        :type end_pos: np.ndarray

        :param corridor2: second corridor
        :type corridor2: CorridorWorld
        '''

        if start_pos is None:
            margin = unicycle.length + unicycle.length * 0.3
            start_pose = compute_start_pose(corridor1, unicycle, margin)
            x_init = start_pose[0]
            y_init = start_pose[1]
        else:
            x_init = start_pos[0]
            y_init = start_pos[1]
        if end_pos is None:
            if corridor2 is None:
                margin = unicycle.length + unicycle.length * 0.3
                end_pose = compute_end_pose(corridor1, unicycle, margin)
                x_final = end_pose[0]
                y_final = end_pose[1]
            else:
                margin = unicycle.length + unicycle.length * 0.3
                end_pose = compute_start_pose(corridor2, unicycle, margin)
                x_final = end_pose[0]
                y_final = end_pose[1]
        else:
            x_final = end_pos[0]
            y_final = end_pos[1]

        x_initial_guess = np.linspace(x_init, x_final, N)
        y_initial_guess = np.linspace(y_init, y_final, N)

        return x_initial_guess, y_initial_guess



