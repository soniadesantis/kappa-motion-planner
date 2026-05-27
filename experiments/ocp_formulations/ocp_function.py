from rockit import *
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def ocp_function(x0, y0, theta0, xf, yf, thetaf,
                 v_min, v_max, omega_min, omega_max):

    T_guess = 5.0
    N = 100
    M = 4

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

    x_guess = np.linspace(x0, xf, N + 1)
    y_guess = np.linspace(y0, yf, N + 1)

    line_angle = np.arctan2(yf - y0, xf - x0)
    theta_guess = np.linspace(theta0, thetaf, N + 1)

    distance = np.hypot(xf - x0, yf - y0)
    T_init = max(distance / max(v_max, 1e-8), 1e-3)

    v_guess = np.full(N, min(max(distance / T_init, v_min), v_max))
    omega_guess = np.full(N, 0.0)

    ocp.set_initial(ocp.T, T_init)
    ocp.set_initial(x, x_guess)
    ocp.set_initial(y, y_guess)
    ocp.set_initial(theta, theta_guess)
    ocp.set_initial(v, v_guess)
    ocp.set_initial(omega, omega_guess)

    ocp.method(MultipleShooting(N=N, M=M, intg='rk'))

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

    ocp.solver("ipopt", options)

    try:
        sol = ocp.solve()
    except Exception:
        sol = ocp.non_converged_solution
        ocp.show_infeasibilities(1e-5)

    ts_int, xs = sol.sample(x, grid="integrator")
    _, ys = sol.sample(y, grid="integrator")
    _, thetas = sol.sample(theta, grid="integrator")

    ts_ctrl, vs = sol.sample(v, grid="control")
    _, omegas = sol.sample(omega, grid="control")

    total_time = sol.value(ocp.T)

    return xs, ys, thetas, ts_int, vs, omegas, ts_ctrl, total_time


def classify_primitive_snapped(v, omega, v_max, omega_max,
                               v_zero_tol=0.05,
                               omega_zero_tol=0.20):
    """
    Classify one control sample into a motion primitive.

    v_zero_tol:
        Fraction of v_max below which v is treated as zero.

    omega_zero_tol:
        Fraction of omega_max below which omega is treated as zero.
    """

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


def extract_sequence_snapped(vs, omegas, v_max, omega_max,
                             ts_ctrl=None,
                             min_duration_fraction=0.01):

    labels = [
        classify_primitive_snapped(v, omega, v_max, omega_max)
        for v, omega in zip(vs, omegas)
    ]

    if ts_ctrl is None:
        sequence = []
        for label in labels:
            if len(sequence) == 0 or label != sequence[-1]:
                sequence.append(label)
        return sequence, labels

    dt = np.diff(ts_ctrl)

    primitives = []
    for label, duration in zip(labels, dt):
        if len(primitives) == 0:
            primitives.append([label, duration])
        elif primitives[-1][0] == label:
            primitives[-1][1] += duration
        else:
            primitives.append([label, duration])

    min_duration = min_duration_fraction * ts_ctrl[-1]

    primitives = [
        primitive for primitive in primitives
        if primitive[1] >= min_duration
    ]

    merged = []
    for label, duration in primitives:
        if len(merged) == 0:
            merged.append([label, duration])
        elif merged[-1][0] == label:
            merged[-1][1] += duration
        else:
            merged.append([label, duration])

    sequence = [label for label, _ in merged]

    return sequence, labels


def extract_sequence_snapped_with_angles(
    vs,
    omegas,
    ts_ctrl,
    v_max,
    omega_max,
    min_duration_fraction=0.01,
):

    labels = [
        classify_primitive_snapped(v, omega, v_max, omega_max)
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


def solve_and_plot_random_case():

    x0, y0 = 0.0, 0.0
    xf, yf = 0.0, 10.0

    v_max = 1.0
    omega_max = 1.0
    v_min = 0.0
    omega_min = -omega_max

    n_angles = 8
    angle_grid = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    theta0 = np.random.choice(angle_grid)
    thetaf = np.random.choice(angle_grid)

    print(
        f"Random case: "
        f"theta0 = {np.rad2deg(theta0):.1f} deg, "
        f"thetaf = {np.rad2deg(thetaf):.1f} deg"
    )

    (
        xs,
        ys,
        thetas,
        ts,
        vs,
        omegas,
        ts_ctrl,
        total_time,
    ) = ocp_function(
        x0,
        y0,
        theta0,
        xf,
        yf,
        thetaf,
        v_min,
        v_max,
        omega_min,
        omega_max,
    )

    sequence, labels = extract_sequence_snapped(
        vs,
        omegas,
        v_max,
        omega_max,
        ts_ctrl=ts_ctrl,
        min_duration_fraction=0.01,
    )

    sequence_string = " - ".join(sequence)

    print(f"Optimal time: {total_time:.6f} s")
    print(f"Primitive sequence: {sequence_string}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    ax_path = axes[0]
    ax_v = axes[1]
    ax_omega = axes[2]

    ax_path.plot(xs, ys, "b-", linewidth=2)
    ax_path.plot(x0, y0, "go", markersize=8, label="start")
    ax_path.plot(xf, yf, "ro", markersize=8, label="goal")

    arrow_scale = 0.6

    ax_path.quiver(
        x0,
        y0,
        arrow_scale*np.cos(theta0),
        arrow_scale*np.sin(theta0),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="g",
    )

    ax_path.quiver(
        xf,
        yf,
        arrow_scale*np.cos(thetaf),
        arrow_scale*np.sin(thetaf),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
    )

    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True)
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.legend()

    ax_path.set_title(
        f"Trajectory: {np.rad2deg(theta0):.0f}° → {np.rad2deg(thetaf):.0f}°\n"
        f"T = {total_time:.3f} s\n"
        f"{sequence_string}"
    )

    ax_v.step(ts_ctrl, vs, where="post", color="r", label="v")
    ax_v.axhline(v_max, color="k", linestyle="--", linewidth=0.8)
    ax_v.axhline(v_min, color="k", linestyle="--", linewidth=0.8)
    ax_v.set_ylabel("v")
    ax_v.grid(True)
    ax_v.legend()

    ax_omega.step(ts_ctrl, omegas, where="post", color="b", label="omega")
    ax_omega.axhline(omega_max, color="k", linestyle="--", linewidth=0.8)
    ax_omega.axhline(omega_min, color="k", linestyle="--", linewidth=0.8)
    ax_omega.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
    ax_omega.set_xlabel("time")
    ax_omega.set_ylabel("omega")
    ax_omega.grid(True)
    ax_omega.legend()

    for k, label in enumerate(labels):
        if k >= len(ts_ctrl) - 1:
            break

        t_mid = 0.5 * (ts_ctrl[k] + ts_ctrl[k + 1])

        ax_omega.text(
            t_mid,
            1.08 * omega_max,
            label,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":

    # solve_and_plot_random_case()

    x0, y0 = 0.0, 0.0
    xf, yf = 0.0, 10.0

    v_max = 1.0
    omega_max = 1.0
    v_min = 0.0
    omega_min = -omega_max

    save_path = "/home/sonia/Projects/arena-framework/experiments/ocp_formulations/unicycle_sweep_results.npy"

    n_angles = 8
    start_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    final_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    results = {}

    fig, axes = plt.subplots(
        n_angles,
        n_angles,
        figsize=(18, 18),
        sharex=True,
        sharey=True
    )

    for i, theta0 in enumerate(start_angles):
        for j, thetaf in enumerate(final_angles):

            case_id = i * n_angles + j + 1

            print(
                f"Solving case {case_id:02d}/{n_angles*n_angles}: "
                f"theta0={np.rad2deg(theta0):.0f} deg, "
                f"thetaf={np.rad2deg(thetaf):.0f} deg"
            )

            (
                xs,
                ys,
                thetas,
                ts,
                vs,
                omegas,
                ts_ctrl,
                total_time,
            ) = ocp_function(
                x0,
                y0,
                theta0,
                xf,
                yf,
                thetaf,
                v_min,
                v_max,
                omega_min,
                omega_max,
            )

            # sequence, primitives_with_duration = extract_sequence_snapped(
            #     vs,
            #     omegas,
            #     v_max,
            #     omega_max,
            #     ts_ctrl,
            # )

            sequence, primitives_with_info = extract_sequence_snapped_with_angles(
                vs,
                omegas,
                ts_ctrl,
                v_max,
                omega_max,
            )

            sequence_string = " - ".join(sequence)

            print(f"  T = {total_time:.4f} s")
            print(f"  sequence = {sequence_string}")

            for p in primitives_with_info:
                if p["label"] in ["ArcL", "ArcR"]:
                    print(
                        f"{p['label']}: "
                        f"duration = {p['duration']:.4f} s, "
                        f"delta_theta = {p['delta_theta']:.4f} rad "
                        f"({np.rad2deg(p['delta_theta']):.2f} deg)"
                    )

            results[(float(theta0), float(thetaf))] = {
                "theta0": float(theta0),
                "thetaf": float(thetaf),
                "theta0_deg": float(np.rad2deg(theta0)),
                "thetaf_deg": float(np.rad2deg(thetaf)),
                "xs": xs,
                "ys": ys,
                "thetas": thetas,
                "ts": ts,
                "vs": vs,
                "omegas": omegas,
                "ts_ctrl": ts_ctrl,
                "total_time": float(total_time),
                "sequence": sequence,
                "primitives_with_info": primitives_with_info,
            }

            ax = axes[i, j]

            ax.plot(xs, ys, "b-", linewidth=1.1)
            ax.plot(x0, y0, "go", markersize=3)
            ax.plot(xf, yf, "ro", markersize=3)

            arrow_scale = 0.4

            ax.quiver(
                x0,
                y0,
                arrow_scale * np.cos(theta0),
                arrow_scale * np.sin(theta0),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="g",
                width=0.006,
            )

            ax.quiver(
                xf,
                yf,
                arrow_scale * np.cos(thetaf),
                arrow_scale * np.sin(thetaf),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="r",
                width=0.006,
            )

            ax.set_title(
                f"{np.rad2deg(theta0):.0f}° → {np.rad2deg(thetaf):.0f}°\n"
                f"T={total_time:.2f}s\n"
                f"{sequence_string}",
                fontsize=7,
            )

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.3)

    fig.suptitle(
        "Time-optimal unicycle trajectories and extracted primitive sequences",
        fontsize=16
    )

    plt.tight_layout()

    np.save(save_path, results, allow_pickle=True)
    print(f"\nResults saved to: {save_path}")


    print("\n" + "=" * 80)
    print("EXTRACTED MOTION PRIMITIVE SEQUENCES")
    print("=" * 80)

    for i, theta0 in enumerate(start_angles):
        for j, thetaf in enumerate(final_angles):

            data = results[(float(theta0), float(thetaf))]

            theta0_deg = data["theta0_deg"]
            thetaf_deg = data["thetaf_deg"]

            sequence_string = " - ".join(data["sequence"])

            print(
                f"{theta0_deg:6.1f}° -> {thetaf_deg:6.1f}° : "
                f"{sequence_string}"
            )

    plt.show(block=True)