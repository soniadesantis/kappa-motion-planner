from rockit import *
from casadi import *
import numpy as np
from .helper_functions import Timer


def compute_ocp_pose_to_pose_trajectory(
    start_pose,
    end_pose,
    unicycle,
    analytical_initial_guess=None,
    T_guess=None,
    N=100,
    M=4,
):
    x0 = start_pose.x
    y0 = start_pose.y
    theta0 = start_pose.theta

    xf = end_pose.x
    yf = end_pose.y
    thetaf = end_pose.theta

    v_min = unicycle.v_min
    v_max = unicycle.v_max
    omega_min = unicycle.omega_min
    omega_max = unicycle.omega_max

    if T_guess is None:
        distance = np.hypot(xf - x0, yf - y0)
        T_guess = 1.5 * distance / v_max

    ocp = Ocp(t0=0, T=FreeTime(T_guess))

    x = ocp.state()
    y = ocp.state()
    theta = ocp.state()

    v = ocp.control()
    omega = ocp.control()

    ocp.set_der(x, v * cos(theta))
    ocp.set_der(y, v * sin(theta))
    ocp.set_der(theta, omega)

    ocp.subject_to(ocp.at_t0(x) == x0)
    ocp.subject_to(ocp.at_t0(y) == y0)
    ocp.subject_to(ocp.at_t0(cos(theta)) == np.cos(theta0))
    ocp.subject_to(ocp.at_t0(sin(theta)) == np.sin(theta0))

    ocp.subject_to(ocp.at_tf(x) == xf)
    ocp.subject_to(ocp.at_tf(y) == yf)
    ocp.subject_to(ocp.at_tf(cos(theta)) == np.cos(thetaf))
    ocp.subject_to(ocp.at_tf(sin(theta)) == np.sin(thetaf))

    ocp.subject_to(v_min <= v)
    ocp.subject_to(v <= v_max)
    ocp.subject_to(omega_min <= omega)
    ocp.subject_to(omega <= omega_max)

    ocp.add_objective(ocp.T)

    # Initial guess
    if analytical_initial_guess is not None:
        initial_guess = compute_initial_guess_from_analytical_trajectory(
            analytical_initial_guess,
            start_pose,
            unicycle,
            N,
        )

        ocp.set_initial(x, initial_guess["x"])
        ocp.set_initial(y, initial_guess["y"])
        ocp.set_initial(theta, initial_guess["theta"])
        ocp.set_initial(v, initial_guess["v"])
        ocp.set_initial(omega, initial_guess["omega"])

    else:
        initial_guess = None
   
        x_guess = np.linspace(x0, xf, N + 1)
        y_guess = np.linspace(y0, yf, N + 1)
        theta_guess = np.linspace(theta0, thetaf, N + 1)

        v_init = min(max(distance / T_guess, v_min), v_max)
        v_guess = np.full(N, v_init)
        omega_guess = np.zeros(N)

        ocp.set_initial(x, x_guess)
        ocp.set_initial(y, y_guess)
        ocp.set_initial(theta, theta_guess)
        ocp.set_initial(v, v_guess)
        ocp.set_initial(omega, omega_guess)

    ocp.method(MultipleShooting(N=N, M=M, intg="rk"))

    # print(f"x initial guess: { initial_guess['x']}")
    # print(f"y initial guess: { initial_guess['y']}")
    # print(f"theta initial guess: { initial_guess['theta']}")
    # print(f"v initial guess: { initial_guess['v']}")
    # print(f"omega initial guess: { initial_guess['omega']}")
    # print(f"T initial guess: { T_guess}")

    options = {
        "expand": True,
        "verbose": False,
        "print_time": False,
        "error_on_fail": False,
        "ipopt": {
            "linear_solver": "mumps",
            "print_level": 0,
            "tol": 1e-6,
            "sb": "yes",
        },
    }

    # dump_folder_no_initial_guess = "casadi_dumps"
    # os.makedirs(dump_folder_no_initial_guess, exist_ok=True)

    # options = {
    #     "common_options": {
    #         "final_options": {
    #             "dump_in": True,
    #             "dump_out": True,
    #             "dump_dir": dump_folder_no_initial_guess,
    #         }
    #     }
    # }


    ocp.solver("ipopt", options)
 
    success = True
    with Timer() as timer:
        try:
            sol = ocp.solve()
        except Exception:
            sol = ocp.non_converged_solution
            success = False
    comp_time = timer()

    ts_int, xs = sol.sample(x, grid="integrator")
    _, ys = sol.sample(y, grid="integrator")
    _, thetas = sol.sample(theta, grid="integrator")

    ts_ctrl, vs = sol.sample(v, grid="control")
    _, omegas = sol.sample(omega, grid="control")

    total_time = sol.value(ocp.T)

    sequence, primitives_with_info = extract_ocp_sequence_with_angles(
        vs,
        omegas,
        ts_ctrl,
        v_max,
        omega_max,
    )

    return {
        "success": success,
        "solve_time": comp_time,
        "xs": xs,
        "ys": ys,
        "thetas": thetas,
        "ts": ts_int,
        "vs": vs,
        "omegas": omegas,
        "ts_ctrl": ts_ctrl,
        "time": float(total_time),
        "sequence": sequence,
        "primitives_with_info": primitives_with_info,
        "initial_guess": initial_guess,
    }


## Classification helpers
def classify_ocp_primitive(
    v,
    omega,
    v_max,
    omega_max,
    v_zero_tol=0.05,
    omega_zero_tol=0.20,
):
    v_eps = v_zero_tol * v_max
    omega_eps = omega_zero_tol * omega_max

    if abs(v) <= v_eps:
        if omega > omega_eps:
            return "SpinL"
        elif omega < -omega_eps:
            return "SpinR"
        else:
            return "Stop"

    if abs(omega) <= omega_eps:
        return "Straight"

    if omega > 0:
        return "ArcL"

    return "ArcR"


def extract_ocp_sequence_with_angles(
    vs,
    omegas,
    ts_ctrl,
    v_max,
    omega_max,
    min_duration_fraction=0.01,
):
    labels = [
        classify_ocp_primitive(v, omega, v_max, omega_max)
        for v, omega in zip(vs, omegas)
    ]

    dt = np.diff(ts_ctrl)

    primitives = []

    for k, (label, duration) in enumerate(zip(labels, dt)):
        delta_theta = omegas[k] * duration

        if len(primitives) == 0:
            primitives.append({
                "label": label,
                "duration": duration,
                "delta_theta": delta_theta,
            })
        elif primitives[-1]["label"] == label:
            primitives[-1]["duration"] += duration
            primitives[-1]["delta_theta"] += delta_theta
        else:
            primitives.append({
                "label": label,
                "duration": duration,
                "delta_theta": delta_theta,
            })

    min_duration = min_duration_fraction * ts_ctrl[-1]

    primitives = [
        p for p in primitives
        if p["duration"] >= min_duration
    ]

    merged = []

    for p in primitives:
        if len(merged) == 0:
            merged.append(p.copy())
        elif merged[-1]["label"] == p["label"]:
            merged[-1]["duration"] += p["duration"]
            merged[-1]["delta_theta"] += p["delta_theta"]
        else:
            merged.append(p.copy())

    sequence = [p["label"] for p in merged]

    return sequence, merged


def compute_initial_guess_from_analytical_trajectory(
    analytical_trajectory,
    start_pose,
    unicycle,
    N,
    time_interval=0.01,
):
    T_analytical = analytical_trajectory[-1].tf

    N_sim = max(int(np.ceil(T_analytical / time_interval)), 2)
    dt = T_analytical / N_sim

    time_grid = np.linspace(0.0, T_analytical, N_sim + 1)

    vs = np.empty(N_sim)
    omegas = np.empty(N_sim)

    j = 0

    for i in range(N_sim):
        t = time_grid[i]

        while j < len(analytical_trajectory) - 1 and t >= analytical_trajectory[j].tf:
            j += 1

        vs[i] = analytical_trajectory[j].v
        omegas[i] = analytical_trajectory[j].omega

    state_trajectory = unicycle.simulate_trajectory(
        np.array(start_pose),
        np.vstack((vs, omegas)),
        dt,
    )

    xs = state_trajectory[0, :]
    ys = state_trajectory[1, :]
    thetas = state_trajectory[2, :]

    state_time_grid = np.linspace(0.0, T_analytical, len(xs))

    ocp_state_time_grid = np.linspace(0.0, T_analytical, N + 1)
    ocp_control_time_grid = np.linspace(0.0, T_analytical, N)

    x_guess = np.interp(ocp_state_time_grid, state_time_grid, xs)
    y_guess = np.interp(ocp_state_time_grid, state_time_grid, ys)
    theta_guess = np.interp(ocp_state_time_grid, state_time_grid, thetas)

    v_guess = np.interp(
        ocp_control_time_grid,
        time_grid[:-1],
        vs,
    )

    omega_guess = np.interp(
        ocp_control_time_grid,
        time_grid[:-1],
        omegas,
    )

    return {
        "time": T_analytical,

        # OCP-sized initial guess
        "x": x_guess,
        "y": y_guess,
        "theta": theta_guess,
        "v": v_guess,
        "omega": omega_guess,
        "ts_state": ocp_state_time_grid,
        "ts_ctrl": ocp_control_time_grid,

        # optional: full high-resolution simulated analytical trajectory
        "simulated": {
            "x": xs,
            "y": ys,
            "theta": thetas,
            "v": vs,
            "omega": omegas,
            "ts_state": state_time_grid,
            "ts_ctrl": time_grid[:-1],
            "dt": dt,
        },
    }