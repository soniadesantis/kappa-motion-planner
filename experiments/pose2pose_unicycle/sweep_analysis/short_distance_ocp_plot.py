"""Explore D/R = 1, 2, 3 using OCP only, with opposing or equal headings.

Use the same TST initialization strategy for every distance, with theta_0 = 0
and theta_f = pi (default) or 0 (--theta-f 0). Plot each OCP result and its seed; no analytical optimum is used.
"""

import argparse
import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import compute_ocp_pose_to_pose_trajectory
from worst_cases_plot import STYLE, load_tst_builder


RATIOS = (1, 2, 3)
RADIUS = 1.0  # R = v_max / omega_max, not a minimum radius at reduced speed.
V_MAX = 1.0
N = 100
M = 4
FIGURES_DIRECTORY = Path(__file__).resolve().parent.parent / "figures"
COLORS = {"initial": "#009E73", "ocp": "#D55E00"}


def solve_cases(n, m, linear_solver, theta_f=pi):
    vehicle = Unicycle(state=[0, 0, 0], width=0.430, length=0.430,
                       v_max=V_MAX, v_min=0, omega_max=V_MAX / RADIUS,
                       omega_min=-V_MAX / RADIUS)
    build_tst = load_tst_builder()
    runs, summaries = [], []
    for ratio in RATIOS:
        distance = ratio * RADIUS
        start, end = Pose(Point(0, 0), 0), Pose(Point(0, distance), theta_f)
        tst, tst_time, _ = build_tst(start, end, vehicle)
        print(f"D/R={ratio}: TST initialization, {linear_solver}, N={n}, M={m}", flush=True)
        result = compute_ocp_pose_to_pose_trajectory(
            start, end, vehicle, analytical_initial_guess=tst, T_guess=tst_time,
            N=n, M=m, linear_solver=linear_solver,
        )
        trial = {"initial_guess": "TST", "initial_time": tst_time,
                 "success": bool(result["success"]), "time": float(result["time"]),
                 "solve_time": float(result["solve_time"]),
                 "sequence": result["sequence"], "solver_error": result["solver_error"]}
        print(json.dumps(trial, indent=2), flush=True)
        if not result["success"]:
            raise RuntimeError(f"TST-initialized OCP failed for D/R={ratio}.")
        runs.append((ratio, result))
        summaries.append({"D_over_R": ratio, "start_pose": [0, 0, 0],
                          "end_pose": [0, distance, theta_f], "initial_guess": "TST",
                          "time": float(result["time"]), "trials": [trial]})
    return runs, summaries


def create_figure(runs, theta_f=pi):
    heading_label = r"\pi" if theta_f == pi else "0"
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(16, 10), layout="constrained")
        grid = figure.add_gridspec(3, 3, height_ratios=(1.7, 1, 1), hspace=0.06)
        columns = []
        for column, (ratio, result) in enumerate(runs):
            path = figure.add_subplot(grid[0, column])
            velocity = figure.add_subplot(grid[1, column])
            angular = figure.add_subplot(grid[2, column], sharex=velocity)
            velocity.tick_params(labelbottom=False)
            columns.append((path, velocity, angular))
            guess = result["initial_guess"]
            path.plot(guess["x"], guess["y"], color=COLORS["initial"], linestyle=":", linewidth=2.5)
            path.plot(result["xs"], result["ys"], color=COLORS["ocp"], linewidth=2.5, linestyle="--")
            for suffix, y, direction, color in (("0", 0, 1, "g"), ("f", ratio * RADIUS, np.cos(theta_f), "r")):
                path.plot(0, y, "o", color=color, markersize=5)
                path.annotate("", xy=(0.3 * RADIUS * direction, y), xytext=(0, y),
                              arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
                path.annotate(rf"$\boldsymbol{{p}}_{suffix}$", (0, y), xytext=(5, 7),
                              textcoords="offset points", fontsize=15)
            path.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]",
                     title=rf"{chr(97 + column)}) $D/R={ratio}$"
                     + "\n" + rf"$T={result['time']:.4f}\,\mathrm{{s}}$"
                     + "\n" + rf"$\theta_0=0,\quad\theta_f={heading_label}$")
            path.set_title(path.get_title(), pad=16)
            path.set_aspect("equal", adjustable="box")
            path.margins(0.18)
            # Heading annotations do not contribute to Matplotlib's data limits.
            left, right = path.get_xlim()
            horizontal_margin = 0.4 * RADIUS * (ratio if theta_f == 0 else 1)
            path.set_xlim(min(left, -horizontal_margin), max(right, horizontal_margin))
            for source, color, linestyle in ((guess, COLORS["initial"], ":"),
                                              (result, COLORS["ocp"], "--")):
                times = np.asarray(source["ts_ctrl"]).ravel()
                for axis, key in ((velocity, "v" if source is guess else "vs"),
                                  (angular, "omega" if source is guess else "omegas")):
                    values = np.asarray(source[key]).ravel()
                    axis.step(np.r_[times, source["time"]], np.r_[values, values[-1]],
                              where="post", color=color, linestyle=linestyle, linewidth=2.3)
            omega_max = V_MAX / RADIUS
            for axis, limits in ((velocity, (0, V_MAX)), (angular, (-omega_max, omega_max))):
                for bound in limits:
                    axis.axhline(bound, color="0.5", linestyle="-.", linewidth=1, zorder=0)
                margin = (limits[1] - limits[0]) * 0.12
                axis.set_ylim(limits[0] - margin, limits[1] + margin)
                axis.set_xlim(0, max(guess["time"], result["time"]) * 1.02)
            angular.set_xlabel(r"$t$ [s]")
            if column == 0:
                velocity.set_ylabel(r"$v(t)$ [m/s]")
                angular.set_ylabel(r"$\omega(t)$ [rad/s]")
            for axis in (path, velocity, angular):
                axis.grid(linestyle=":", alpha=0.4)
        figure.legend(handles=[
            Line2D([], [], color=COLORS["initial"], linestyle=":", linewidth=2.5,
                   label="TST initial guess"),
            Line2D([], [], color=COLORS["ocp"], linestyle="--", linewidth=2.5,
                   label="OCP solution"),
            Line2D([], [], color="0.5", linestyle="-.", label="Control bounds"),
        ], loc="outside upper center", ncol=3, frameon=False)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = [[axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
                   for axis in column] for column in columns]
        for column in range(2):
            x = (max(b.x1 for b in bounds[column]) + min(b.x0 for b in bounds[column + 1])) / 2
            separator = Line2D([x, x], [min(b.y0 for col in bounds for b in col),
                                       max(b.y1 for col in bounds for b in col)],
                               transform=figure.transFigure, color="0.8", linewidth=0.9)
            separator.set_in_layout(False)
            figure.add_artist(separator)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-solver", default="mumps")
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--theta-f", choices=("0", "pi"), default="pi",
                        help="Final heading; the initial heading is always zero.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.N < 1 or args.M < 1:
        parser.error("N and M must be positive.")
    theta_f = pi if args.theta_f == "pi" else 0.0
    runs, summaries = solve_cases(args.N, args.M, args.linear_solver, theta_f)
    figure = create_figure(runs, theta_f)
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    stem = f"short_distance_ocp_N{args.N}_M{args.M}_{args.linear_solver}"
    if args.theta_f == "0":
        stem += "_thetaf_0"
    for extension in ("pdf", "png"):
        output = FIGURES_DIRECTORY / f"{stem}.{extension}"
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}", flush=True)
    (FIGURES_DIRECTORY / f"{stem}.json").write_text(json.dumps({
        "R": RADIUS, "v_max": V_MAX, "omega_max": V_MAX / RADIUS,
        "N": args.N, "M": args.M, "linear_solver": args.linear_solver,
        "initial_guess": "TST",
        "cases": summaries,
    }, indent=2) + "\n")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
