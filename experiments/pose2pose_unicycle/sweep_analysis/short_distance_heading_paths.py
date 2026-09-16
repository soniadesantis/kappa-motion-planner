"""Plot TST-initialized OCP paths for four start and four final headings.

Positions are (0, 0) and (0, D), with D/R = 1, 2, 3. Results are cached
incrementally so plots can be adjusted without repeating the 48 OCP solves.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import compute_ocp_pose_to_pose_trajectory
from short_distance_ocp_plot import RADIUS, V_MAX, RATIOS, N, M, FIGURES_DIRECTORY
from worst_cases_plot import STYLE, load_tst_builder

HEADINGS = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
LABELS = ("0", r"\pi/2", r"\pi", r"3\pi/2")
COLORS = ("#0072B2", "#D55E00", "#009E73")
STYLES = ("-", "--", "-.")


def solve_cases(args, cache):
    metadata = dict(R=RADIUS, v_max=V_MAX, omega_max=V_MAX / RADIUS,
                    N=args.N, M=args.M, linear_solver=args.linear_solver,
                    initial_guess="TST", headings=list(HEADINGS), ratios=list(RATIOS))
    data = dict(metadata=metadata, cases=[])
    if cache.exists() and not args.recompute:
        data = json.loads(cache.read_text())
        if data["metadata"] != metadata:
            raise ValueError("Cached parameters differ; use --recompute.")
    completed = {(c["row"], c["column"], c["D_over_R"]): c for c in data["cases"]}
    vehicle = Unicycle(state=[0, 0, 0], width=0.430, length=0.430,
                       v_max=V_MAX, v_min=0, omega_max=V_MAX / RADIUS,
                       omega_min=-V_MAX / RADIUS)
    build_tst = load_tst_builder()
    for row, theta0 in enumerate(HEADINGS):
        for column, thetaf in enumerate(HEADINGS):
            for ratio in RATIOS:
                key = row, column, ratio
                if key in completed:
                    continue
                print(f"Solving {len(completed) + 1}/48: D/R={ratio}, theta0={theta0:.4f}, thetaf={thetaf:.4f}", flush=True)
                start, end = Pose(Point(0, 0), theta0), Pose(Point(0, ratio * RADIUS), thetaf)
                seed, duration, _ = build_tst(start, end, vehicle)
                result = compute_ocp_pose_to_pose_trajectory(
                    start, end, vehicle, analytical_initial_guess=seed, T_guess=duration,
                    N=args.N, M=args.M, linear_solver=args.linear_solver)
                record = dict(row=row, column=column, D_over_R=ratio,
                              start_pose=[0, 0, theta0], end_pose=[0, ratio * RADIUS, thetaf],
                              success=bool(result["success"]), time=float(result["time"]),
                              solve_time=float(result["solve_time"]), sequence=result["sequence"],
                              solver_error=result["solver_error"])
                for name in ("xs", "ys", "thetas", "ts", "vs", "omegas", "ts_ctrl"):
                    record[name] = np.asarray(result[name]).ravel().tolist()
                completed[key] = record
                data["cases"].append(record)
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(data, indent=2) + "\n")
                temporary.replace(cache)
                print(f"  success={record['success']}, T={record['time']:.6f} s", flush=True)
    return data


def create_figure(data, ratios):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 4, figsize=(14, 17), layout="compressed")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.01, hspace=0.01)
        records = [c for c in data["cases"] if c["D_over_R"] in ratios and c["success"]]
        xs = [x / RADIUS for c in records for x in c["xs"]] + [-0.35, 0.35]
        ys = [y / RADIUS for c in records for y in c["ys"]] + [-0.35, max(ratios) + 0.35]
        xmin, xmax = min(xs) - 0.15, max(xs) + 0.15
        ymin, ymax = min(ys) - 0.15, max(ys) + 0.15
        # A minimum width keeps ticks readable for nearly vertical trajectories.
        halfwidth = max((xmax - xmin) / 2, 0.38 * (ymax - ymin))
        center = (xmin + xmax) / 2
        for row, theta0 in enumerate(HEADINGS):
            for column, thetaf in enumerate(HEADINGS):
                ax = axes[row, column]
                for ratio in ratios:
                    color, style = COLORS[RATIOS.index(ratio)], STYLES[RATIOS.index(ratio)]
                    case = next(c for c in data["cases"] if (c["row"], c["column"], c["D_over_R"]) == (row, column, ratio))
                    if case["success"]:
                        ax.plot(np.asarray(case["xs"]) / RADIUS, np.asarray(case["ys"]) / RADIUS,
                                color=color, linestyle=style, linewidth=2)
                    else:
                        ax.text(0.03, 0.95 - 0.07 * RATIOS.index(ratio), f"D/R={ratio}: failed",
                                transform=ax.transAxes, color=color, fontsize=11)
                    ax.plot(0, ratio, "o", color=color, markersize=4)
                    ax.annotate("", xy=(0.28 * np.cos(thetaf), ratio + 0.28 * np.sin(thetaf)),
                                xytext=(0, ratio), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
                ax.plot(0, 0, "o", color="0.2", markersize=4)
                ax.annotate("", xy=(0.28 * np.cos(theta0), 0.28 * np.sin(theta0)),
                            xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.5))
                ax.set(xlim=(center - halfwidth, center + halfwidth), ylim=(ymin, ymax), aspect="equal")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.grid(linestyle=":", alpha=0.35)
                ax.tick_params(labelsize=13)
                ax.set_title(rf"$\theta_0={LABELS[row]},\quad\theta_f={LABELS[column]}$", fontsize=17, pad=12)
                if row == 3:
                    ax.set_xlabel(r"$x/R$")
                if column == 0:
                    ax.set_ylabel(r"$y/R$")
        fig.legend(handles=[Line2D([], [], color=COLORS[RATIOS.index(r)],
                                  linestyle=STYLES[RATIOS.index(r)], linewidth=2,
                                  label=rf"$D/R={r}$") for r in ratios],
                   loc="outside upper center", ncol=3, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--linear-solver", default="mumps")
    parser.add_argument("--layout", choices=("overlay", "separate"), default="overlay")
    parser.add_argument("--recompute", action="store_true", help="Replace cached solves.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.N < 1 or args.M < 1:
        parser.error("N and M must be positive.")
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    stem = f"short_distance_heading_paths_N{args.N}_M{args.M}_{args.linear_solver}"
    data = solve_cases(args, FIGURES_DIRECTORY / f"{stem}.json")
    groups = [RATIOS] if args.layout == "overlay" else [(r,) for r in RATIOS]
    for ratios in groups:
        figure = create_figure(data, ratios)
        suffix = "" if len(ratios) > 1 else f"_D_over_R_{ratios[0]}"
        for extension in ("pdf", "png"):
            output = FIGURES_DIRECTORY / f"{stem}{suffix}.{extension}"
            figure.savefig(output, dpi=300, bbox_inches="tight")
            print(f"Saved {output}", flush=True)
        if args.no_show:
            plt.close(figure)
    print(f"Converged: {sum(c['success'] for c in data['cases'])}/{len(data['cases'])}", flush=True)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
