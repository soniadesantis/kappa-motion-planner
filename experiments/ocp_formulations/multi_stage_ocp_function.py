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
    min_grid = 0.000001
    max_grid = 10
    #Initialize stages parameters
    T1 = 2
    N1 = 1

    T2 = 2
    N2 = 1

    T3 = 2
    N3 = 1

    T4 = 2
    N4 = 1
    NM = 100
    M1 = int(ceil(NM/N1))
    M2 = int(ceil(NM/N2))
    M3 = int(ceil(NM/N3))
    M4 = int(ceil(NM/N4))
    # M = 2
    t0_stage1 = 0
    tf_stage1 = FreeTime(T1)

    t0_stage2 = FreeTime(T1)
    tf_stage2 = FreeTime(T1 + T2)

    t0_stage3 = FreeTime(T1 + T2)
    tf_stage3 = FreeTime(T1 + T2 + T3)

    t0_stage4 = FreeTime(T1 + T2 + T3)
    tf_stage4 = FreeTime(T1 + T2 + T3 + T4)
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

    # Stage 2 - arc
    stage2, x2, y2, theta2, v2, omega2, = create_stage(ocp, t0_stage2, tf_stage2, N2, M2, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage1, stage2)
    # stage2.subject_to(0.001 <= (delta2<= delta_max), include_last=False)

    # stage2.set_initial(x2, x2_initial_guess)
    # stage2.set_initial(y2, y2_initial_guess)
    # stage2.set_initial(v2, v2_initial_guess)

    # Stage 3 - segment
    stage3, x3, y3, theta3, v3, omega3, = create_stage(ocp, t0_stage3, tf_stage3, N3, M3, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage2, stage3)
    stage3.subject_to(omega3 == 0)

    # stage3.set_initial(x3, x3_initial_guess)
    # stage3.set_initial(y3, y3_initial_guess)
    # stage3.set_initial(v3, v3_initial_guess)

    # Stage 4 - arc
    stage4, x4, y4, theta4, v4, omega4, = create_stage(ocp, t0_stage4, tf_stage4, N4, M4, omega_min, omega_max, v_min, v_max, min_grid, max_grid)
    stitch_stages(ocp, stage3, stage4)

    ocp.subject_to(stage4.at_tf(x4) == xf)
    ocp.subject_to(stage4.at_tf(y4) == yf)
    ocp.subject_to(stage4.at_tf(theta4) == thetaf)

    # stage4.set_initial(x4, x4_initial_guess)
    # stage4.set_initial(y4, y4_initial_guess)
    # stage4.set_initial(v4, v4_initial_guess)

    ocp.add_objective(stage1.T + stage2.T + stage3.T + stage4.T)

    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": True,
                "error_on_fail": False,
                "ipopt": {	"linear_solver": "ma57",
                            "print_level": 5,
                            "tol": 1e-12}}
    ocp.solver('ipopt', options)

    # Solve
    try:
        sol = ocp.solve()
    except:
        sol = ocp.non_converged_solution
    ocp.show_infeasibilities(1e-7)

    total_time = sol.value(stage1.T + stage2.T + stage3.T + stage4.T)
    time_stage1 = sol.value(stage1.T)
    time_stage2 = sol.value(stage2.T)
    time_stage3 = sol.value(stage3.T)
    time_stage4 = sol.value(stage4.T)

    ts1, xs1 = sol(stage1).sample(x1, grid='integrator',refine=10)
    _, ys1 = sol(stage1).sample(y1, grid='integrator',refine=10)
    ts2, xs2 = sol(stage2).sample(x2, grid='integrator',refine=10)
    _, ys2 = sol(stage2).sample(y2, grid='integrator',refine=10)
    ts3, xs3 = sol(stage3).sample(x3, grid='integrator',refine=10)
    _, ys3 = sol(stage3).sample(y3, grid='integrator',refine=10)
    ts4, xs4 = sol(stage4).sample(x4, grid='integrator',refine=10)
    _, ys4 = sol(stage4).sample(y4, grid='integrator',refine=10)
    ts = np.concatenate((ts1, ts2, ts3, ts4), axis=None) 
    xs = np.concatenate((xs1, xs2, xs3, xs4), axis=None) 
    ys = np.concatenate((ys1, ys2, ys3, ys4), axis=None) 

    ts1, xs1 = sol(stage1).sample(x1, grid='control')
    _, ys1 = sol(stage1).sample(y1, grid='control')
    ts2, xs2 = sol(stage2).sample(x2, grid='control')
    _, ys2 = sol(stage2).sample(y2, grid='control')
    ts3, xs3 = sol(stage3).sample(x3, grid='control')
    _, ys3 = sol(stage3).sample(y3, grid='control')
    ts4, xs4 = sol(stage4).sample(x4, grid='control')
    _, ys4 = sol(stage4).sample(y4, grid='control')

    ts_ctrl_grid = np.concatenate((ts1, ts2, ts3, ts4), axis=None) 
    xs_ctrl_grid = np.concatenate((xs1, xs2, xs3, xs4), axis=None) 
    ys_ctrl_grid = np.concatenate((ys1, ys2, ys3, ys4), axis=None) 

    _, omegas1 = sol(stage1).sample(omega1, grid='integrator',refine=10)
    _, omegas2 = sol(stage2).sample(omega2, grid='integrator',refine=10)
    _, omegas3 = sol(stage3).sample(omega3, grid='integrator',refine=10)
    _, omegas4 = sol(stage4).sample(omega4, grid='integrator',refine=10)
    omegas = np.concatenate((omegas1, omegas2, omegas3, omegas4), axis=None) 

    _, omegas1 = sol(stage1).sample(omega1, grid='control')
    _, omegas2 = sol(stage2).sample(omega2, grid='control')
    _, omegas3 = sol(stage3).sample(omega3, grid='control')
    _, omegas4 = sol(stage4).sample(omega4, grid='control')
    omegas_ctrl_grid = np.concatenate((omegas1, omegas2, omegas3, omegas4), axis=None) 

    _, vs1 = sol(stage1).sample(v1, grid='integrator',refine=10)
    _, vs2 = sol(stage2).sample(v2, grid='integrator',refine=10)
    _, vs3 = sol(stage3).sample(v3, grid='integrator',refine=10)
    _, vs4 = sol(stage4).sample(v4, grid='integrator',refine=10)
    vs = np.concatenate((vs1, vs2, vs3, vs4), axis=None) 

    _, vs1 = sol(stage1).sample(v1, grid='control')
    _, vs2 = sol(stage2).sample(v2, grid='control')
    _, vs3 = sol(stage3).sample(v3, grid='control')
    _, vs4 = sol(stage4).sample(v4, grid='control')
    vs_ctrl_grid = np.concatenate((vs1, vs2, vs3, vs4), axis=None) 

    return xs, ys, ts, vs, omegas, xs_ctrl_grid, ys_ctrl_grid, ts_ctrl_grid, vs_ctrl_grid, omegas_ctrl_grid

# t1 = np.concatenate((ts1, ts2[0]), axis=None) 
# del1 = np.concatenate((d1, d2[0]), axis=None)
# t2 = np.concatenate((ts2, ts3[0]), axis=None) 
# del2 = np.concatenate((d2, d3[0]), axis=None)
# t3 = np.concatenate((ts3, ts4[0]), axis=None) 
# del3 = np.concatenate((d3, d4[0]), axis=None)
#-------------------------------------------------------------------
# Plots
# #Plot x and y position + tunnel
# figure_tunnel = plt.figure() #Plot x and y position
# plt.axis('square')
# plt.title('x and y position')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.plot(xs, ys, 'y', linewidth=1)

# plt.plot(xs_ctrl_grid, ys_ctrl_grid, 'yo', markersize=2.5)

# plt.legend()
 
# plt.figure()
# plt.title('Steering angle')
# plt.plot([ts1[0],ts4[-1]], [omega_max, omega_min], 'r-')
# ts1, d1 = sol(stage1).sample(omega1, grid='integrator',refine=10)
# ts2, d2 = sol(stage2).sample(omega2, grid='integrator',refine=10)
# ts3, d3 = sol(stage3).sample(omega3, grid='integrator',refine=10)
# ts4, d4 = sol(stage4).sample(omega4, grid='integrator',refine=10)

# t1 = np.concatenate((ts1, ts2[0]), axis=None) 
# del1 = np.concatenate((d1, d2[0]), axis=None)
# t2 = np.concatenate((ts2, ts3[0]), axis=None) 
# del2 = np.concatenate((d2, d3[0]), axis=None)
# t3 = np.concatenate((ts3, ts4[0]), axis=None) 
# del3 = np.concatenate((d3, d4[0]), axis=None)
# plt.plot(t1, del1, 'y-', label = 'Stage 1')
# plt.plot(t2, del2, 'm-', label = 'Stage 2,3')
# plt.plot(t3, del3, 'm-')
# plt.plot(ts4, d4, 'g-', label = 'Stage 4')

# ts1, d1 = sol(stage1).sample(omega1, grid='control')
# ts2, d2 = sol(stage2).sample(omega2, grid='control')
# ts3, d3 = sol(stage3).sample(omega3, grid='control')
# ts4, d4 = sol(stage4).sample(omega4, grid='control')

# plt.plot(ts1, d1, 'yo', markersize=2.5)
# plt.plot(ts2, d2, 'mo', markersize=2.5)
# plt.plot(ts3, d3, 'mo', markersize=2.5)
# plt.plot(ts4, d4, 'go', markersize=2.5)
# plt.legend()

# plt.figure()
# plt.title("Velocity")
# plt.xlim([0 - 0.2, total_time+1])
# plt.ylim([0 - 0.2 ,v_max + 0.2])
# ts1, vs1 = sol(stage1).sample(v1, grid='integrator',refine=10)
# ts2, vs2 = sol(stage2).sample(v2, grid='integrator',refine=10)
# ts3, vs3 = sol(stage3).sample(v3, grid='integrator',refine=10)
# ts4, vs4 = sol(stage4).sample(v4, grid='integrator',refine=10)

# t1 = np.concatenate((ts1, ts2[0]), axis=None) 
# vel1 = np.concatenate((vs1, vs2[0]), axis=None)
# t2 = np.concatenate((ts2, ts3[0]), axis=None) 
# vel2 = np.concatenate((vs2, vs3[0]), axis=None)
# t3 = np.concatenate((ts3, ts4[0]), axis=None) 
# vel3 = np.concatenate((vs3, vs4[0]), axis=None)
# plt.plot(t1, vel1, 'y-', label = 'Stage 1')
# plt.plot(t2, vel2, 'm-', label = 'Stage 2,3')
# plt.plot(t3, vel3, 'm-')
# plt.plot(ts4, vs4, 'g-', label = 'Stage 4')

# ts1, vs1 = sol(stage1).sample(v1, grid='control')
# ts2, vs2 = sol(stage2).sample(v2, grid='control')
# ts3, vs3 = sol(stage3).sample(v3, grid='control')
# ts4, vs4 = sol(stage4).sample(v4, grid='control')

# plt.plot(ts1, vs1, 'yo', markersize=2.5)
# plt.plot(ts2, vs2, 'mo', markersize=2.5)
# plt.plot(ts3, vs3, 'mo', markersize=2.5)
# plt.plot(ts4, vs4, 'go', markersize=2.5)
# plt.legend()

#--------------------------------------------------
#Additional plot to show comparison between optimal solution and analytical solution

# # Plots
# #Plot x and y position + tunnel
# figure_comparison = plt.figure() #Plot x and y position
# plt.title('x and y position')
# plt.axis('square')
# plt.title(f'Comparison')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# ts1, xs1 = sol(stage1).sample(x1, grid='integrator',refine=10)
# ts1, ys1 = sol(stage1).sample(y1, grid='integrator',refine=10)
# ts2, xs2 = sol(stage2).sample(x2, grid='integrator',refine=10)
# ts2, ys2 = sol(stage2).sample(y2, grid='integrator',refine=10)
# ts3, xs3 = sol(stage3).sample(x3, grid='integrator',refine=10)
# ts3, ys3 = sol(stage3).sample(y3, grid='integrator',refine=10)
# ts4, xs4 = sol(stage4).sample(x4, grid='integrator',refine=10)
# ts4, ys4 = sol(stage4).sample(y4, grid='integrator',refine=10)

# plt.plot(xs1, ys1, 'y', linewidth=1.5, label = 'Optimal solution')
# plt.plot(xs2, ys2, 'y', linewidth=1.5)
# plt.plot(xs3, ys3, 'y', linewidth=1.5)
# plt.plot(xs4, ys4, 'y', linewidth=1.5)

# ts1, xs1 = sol(stage1).sample(x1, grid='control')
# ts1, ys1 = sol(stage1).sample(y1, grid='control')
# ts2, xs2 = sol(stage2).sample(x2, grid='control')
# ts2, ys2 = sol(stage2).sample(y2, grid='control')
# ts3, xs3 = sol(stage3).sample(x3, grid='control')
# ts3, ys3 = sol(stage3).sample(y3, grid='control')
# ts4, xs4 = sol(stage4).sample(x4, grid='control')
# ts4, ys4 = sol(stage4).sample(y4, grid='control')

# plt.plot(xs1, ys1, 'yo', markersize=2.5)
# plt.plot(xs2, ys2, 'yo', markersize=2.5)
# plt.plot(xs3, ys3, 'yo', markersize=2.5)
# plt.plot(xs4, ys4, 'yo', markersize=2.5)

# plt.legend()


# plt.show(block=True)
