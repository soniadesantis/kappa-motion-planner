from rockit import *
from casadi import *

import matplotlib.pyplot as plt
from numpy import pi, cos, sin, tan, sqrt, linspace

#-----------------
# #Initialization of the two stages
# x1_initial_guess = linspace(0,9,N1+1)
# y1_initial_guess = linspace(0.5,0.5,N1+1)
# v1_initial_guess = linspace(1,1,N1)

# x2_initial_guess = linspace(9,9.5,N2+1)
# y2_initial_guess = linspace(0.5,1,N2+1)
# v2_initial_guess = linspace(1,1,N2)

# x3_initial_guess = linspace(9.5,10,N3+1)
# y3_initial_guess = linspace(0.5,1.5,N3+1)
# v3_initial_guess = linspace(1,1,N3)

# x4_initial_guess = linspace(10,10,N4+1)
# y4_initial_guess = linspace(1.5,10.5,N4+1)
# v4_initial_guess = linspace(1,1,N4)
#------------

def create_stage(ocp, t0, T, N, M, omega_min, omega_max, v_min, v_max, min_grid, max_grid):
    stage = ocp.stage(t0=t0, T=T)

    x = stage.state()
    y = stage.state()
    theta = stage.state()

    v = stage.control()
    omega = stage.control()

    stage.set_der(x,v*cos(theta))
    stage.set_der(y,v*sin(theta))
    stage.set_der(theta, omega)

    stage.subject_to(v_min <= (v <= v_max))
    stage.subject_to(omega_min <= (omega<= omega_max))

    stage.method(MultipleShooting(N=N, M=M, intg='rk', grid=FreeGrid(min = min_grid, max = max_grid)))

    return stage, x, y, theta, v, omega

#Define a function that facilitate the connection between subsequent stages
def stitch_stages(ocp, stage1, stage2):
    # Stitch time
    ocp.subject_to(stage1.tf == stage2.t0)
    # Stitch states
    for i in range(len(stage1.states)):
        ocp.subject_to(stage2.at_t0(stage2.states[i])
                       == stage1.at_tf(stage1.states[i]))

#------------------------------------------
#Define optimal control problem
def ms_ocp_function(x0, y0, theta0, xf, yf, thetaf, v_min, v_max, omega_min, omega_max):
    #Grid settings
    min_grid = 0.00000000001
    max_grid = 15
    #Initialize stages parameters
    T1 = 2
    N1 = 1

    T2 = 2
    N2 = 1

    T3 = 2
    N3 = 1

    T4 = 2
    N4 = 1

    T5 = 2
    N5 = 1

    NM = 100
    M1 = int(ceil(NM/N1))
    M2 = int(ceil(NM/N2))
    M3 = int(ceil(NM/N3))
    M4 = int(ceil(NM/N4))
    M5 = int(ceil(NM/N5))

    # M = 2
    t0_stage1 = 0
    tf_stage1 = FreeTime(T1)

    t0_stage2 = FreeTime(T1)
    tf_stage2 = FreeTime(T1 + T2)

    t0_stage3 = FreeTime(T1 + T2)
    tf_stage3 = FreeTime(T1 + T2 + T3)

    t0_stage4 = FreeTime(T1 + T2 + T3)
    tf_stage4 = FreeTime(T1 + T2 + T3 + T4)

    t0_stage5 = FreeTime(T1 + T2 + T3 + T4)
    tf_stage5 = FreeTime(T1 + T2 + T3 + T4 + T5)

    ocp = Ocp()

    # Stage 1 - turn on-the-spot
    stage1, x1, y1, theta1, v1, omega1, = create_stage(ocp, t0_stage1, tf_stage1, N1, M1, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    ocp.subject_to(stage1.at_t0(x1) == x0)
    ocp.subject_to(stage1.at_t0(y1) == y0)
    ocp.subject_to(stage1.at_t0(theta1) == theta0)

    # stage1.set_initial(x1, x1_initial_guess)
    # stage1.set_initial(y1, y1_initial_guess)
    # stage1.set_initial(v1, v1_initial_guess)

    #ocp.subject_to(stage1.at_t0(v1) == v0)

    stage1.subject_to(v1 == 0)
    # stage1.subject_to(omega1 == omega_max)

    # Stage 2 - arc
    stage2, x2, y2, theta2, v2, omega2, = create_stage(ocp, t0_stage2, tf_stage2, N2, M2, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage1, stage2)
    # stage2.subject_to(v2 == v_max)
    stage2.subject_to(omega2 == omega_max)

    # stage2.subject_to(0.001 <= (delta2<= delta_max), include_last=False)

    # stage2.set_initial(x2, x2_initial_guess)
    # stage2.set_initial(y2, y2_initial_guess)
    # stage2.set_initial(v2, v2_initial_guess)

    # Stage 3 - segment
    stage3, x3, y3, theta3, v3, omega3, = create_stage(ocp, t0_stage3, tf_stage3, N3, M3, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage2, stage3)
    stage3.subject_to(omega3 == 0)
    # stage3.subject_to(v3 == v_max)

    # stage3.set_initial(x3, x3_initial_guess)
    # stage3.set_initial(y3, y3_initial_guess)
    # stage3.set_initial(v3, v3_initial_guess)

    # Stage 4 - arc
    stage4, x4, y4, theta4, v4, omega4, = create_stage(ocp, t0_stage4, tf_stage4, N4, M4, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage3, stage4)
    # stage4.subject_to(v4 == v_max)
    stage4.subject_to(omega4 == omega_max)

    # Stage 5 - turn on-the-spot
    stage5, x5, y5, theta5, v5, omega5, = create_stage(ocp, t0_stage5, tf_stage5, N5, M5, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage4, stage5)
    stage5.subject_to(v5 == 0)
    # stage5.subject_to(omega5 == omega_max)

    # stage1.set_initial(x1, x1_initial_guess)
    # stage1.set_initial(y1, y1_initial_guess)
    # stage1.set_initial(v1, v1_initial_guess)

    #ocp.subject_to(stage1.at_t0(v1) == v0)

    ocp.subject_to(stage5.at_tf(x5) == xf)
    ocp.subject_to(stage5.at_tf(y5) == yf)
    ocp.subject_to(stage5.at_tf(theta5) == thetaf)

    # stage4.set_initial(x4, x4_initial_guess)
    # stage4.set_initial(y4, y4_initial_guess)
    # stage4.set_initial(v4, v4_initial_guess)

    ocp.add_objective(stage1.T + stage2.T + stage3.T + stage4.T + stage5.T)

    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": True,
                "error_on_fail": False,
                "ipopt": {	"linear_solver": "mumps", #"ma57",
                            "print_level": 5,
                            "tol": 1e-12}}
    ocp.solver('ipopt', options)

    # Solve
    try:
        sol = ocp.solve()
    except:
        sol = ocp.non_converged_solution
    ocp.show_infeasibilities(1e-7)

    total_time = sol.value(stage1.T + stage2.T + stage3.T + stage4.T + stage5.T)
    time_stage1 = sol.value(stage1.T)
    time_stage2 = sol.value(stage2.T)
    time_stage3 = sol.value(stage3.T)
    time_stage4 = sol.value(stage4.T)
    time_stage5 = sol.value(stage5.T)

    ts1, xs1 = sol(stage1).sample(x1, grid='integrator',refine=10)
    _, ys1 = sol(stage1).sample(y1, grid='integrator',refine=10)
    _, thetas1 = sol(stage1).sample(theta1, grid='integrator',refine=10)
    
    ts2, xs2 = sol(stage2).sample(x2, grid='integrator',refine=10)
    _, ys2 = sol(stage2).sample(y2, grid='integrator',refine=10)
    _, thetas2 = sol(stage2).sample(theta2, grid='integrator',refine=10)

    ts3, xs3 = sol(stage3).sample(x3, grid='integrator',refine=10)
    _, ys3 = sol(stage3).sample(y3, grid='integrator',refine=10)
    _, thetas3 = sol(stage3).sample(theta3, grid='integrator',refine=10)

    ts4, xs4 = sol(stage4).sample(x4, grid='integrator',refine=10)
    _, ys4 = sol(stage4).sample(y4, grid='integrator',refine=10)
    _, thetas4 = sol(stage4).sample(theta4, grid='integrator',refine=10)

    ts5, xs5 = sol(stage5).sample(x5, grid='integrator',refine=10)
    _, ys5 = sol(stage5).sample(y5, grid='integrator',refine=10)
    _, thetas5 = sol(stage5).sample(theta5, grid='integrator',refine=10)

    ts = np.concatenate((ts1, ts2, ts3, ts4, ts5), axis=None) 
    xs = np.concatenate((xs1, xs2, xs3, xs4, xs5), axis=None) 
    ys = np.concatenate((ys1, ys2, ys3, ys4, ys5), axis=None) 
    thetas = np.concatenate((thetas1, thetas2, thetas3, thetas4, thetas5), axis=None) 

    ts1, xs1 = sol(stage1).sample(x1, grid='control')
    _, ys1 = sol(stage1).sample(y1, grid='control')
    _, thetas1 = sol(stage1).sample(theta1, grid='control')
    ts2, xs2 = sol(stage2).sample(x2, grid='control')
    _, ys2 = sol(stage2).sample(y2, grid='control')
    _, thetas2 = sol(stage2).sample(theta2, grid='control')
    ts3, xs3 = sol(stage3).sample(x3, grid='control')
    _, ys3 = sol(stage3).sample(y3, grid='control')
    _, thetas3 = sol(stage3).sample(theta3, grid='control')
    ts4, xs4 = sol(stage4).sample(x4, grid='control')
    _, ys4 = sol(stage4).sample(y4, grid='control')
    _, thetas4 = sol(stage4).sample(theta4, grid='control')
    ts5, xs5 = sol(stage5).sample(x5, grid='control')
    _, ys5 = sol(stage5).sample(y5, grid='control')
    _, thetas5 = sol(stage5).sample(theta5, grid='control')

    ts_ctrl_grid = np.concatenate((ts1, ts2, ts3, ts4, ts5), axis=None) 
    xs_ctrl_grid = np.concatenate((xs1, xs2, xs3, xs4, xs5), axis=None) 
    ys_ctrl_grid = np.concatenate((ys1, ys2, ys3, ys4, ys5), axis=None) 
    thetas_ctrl_grid = np.concatenate((thetas1, thetas2, thetas3, thetas4, thetas5), axis=None) 

    _, omegas1 = sol(stage1).sample(omega1, grid='integrator',refine=10)
    _, omegas2 = sol(stage2).sample(omega2, grid='integrator',refine=10)
    _, omegas3 = sol(stage3).sample(omega3, grid='integrator',refine=10)
    _, omegas4 = sol(stage4).sample(omega4, grid='integrator',refine=10)
    _, omegas5 = sol(stage5).sample(omega5, grid='integrator',refine=10)

    omegas = np.concatenate((omegas1, omegas2, omegas3, omegas4, omegas5), axis=None) 

    _, omegas1 = sol(stage1).sample(omega1, grid='control')
    _, omegas2 = sol(stage2).sample(omega2, grid='control')
    _, omegas3 = sol(stage3).sample(omega3, grid='control')
    _, omegas4 = sol(stage4).sample(omega4, grid='control')
    _, omegas5 = sol(stage5).sample(omega5, grid='control')

    omegas_ctrl_grid = np.concatenate((omegas1, omegas2, omegas3, omegas4, omegas5), axis=None) 

    _, vs1 = sol(stage1).sample(v1, grid='integrator',refine=10)
    _, vs2 = sol(stage2).sample(v2, grid='integrator',refine=10)
    _, vs3 = sol(stage3).sample(v3, grid='integrator',refine=10)
    _, vs4 = sol(stage4).sample(v4, grid='integrator',refine=10)
    _, vs5 = sol(stage5).sample(v5, grid='integrator',refine=10)

    vs = np.concatenate((vs1, vs2, vs3, vs4, vs5), axis=None) 

    _, vs1 = sol(stage1).sample(v1, grid='control')
    _, vs2 = sol(stage2).sample(v2, grid='control')
    _, vs3 = sol(stage3).sample(v3, grid='control')
    _, vs4 = sol(stage4).sample(v4, grid='control')
    _, vs5 = sol(stage5).sample(v5, grid='control')

    vs_ctrl_grid = np.concatenate((vs1, vs2, vs3, vs4, vs5), axis=None) 

    return xs, ys, thetas, ts, vs, omegas, xs_ctrl_grid, ys_ctrl_grid, thetas_ctrl_grid, ts_ctrl_grid, vs_ctrl_grid, omegas_ctrl_grid
