"""Rerun the three largest saved time discrepancies with MUMPS.

Defaults to the Sobol TST N100 sweep. Pass another JSON path to select a sweep;
use --no-show to save the three-panel figure without opening a window.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import compute_ocp_pose_to_pose_trajectory
from kappa_planner.helpers.pose_to_pose_unicycle import compute_all_pose_to_pose_trajectories
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
    plot_primitive_arrows_and_markers,
)


DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_FILENAME = "sweep4/sobol_sweep_OCP_TST_initial_guess_N100_M4.json"
FIGURES_DIRECTORY = DIRECTORY / "figures"
LINEAR_SOLVER = "mumps"
COLORS = {"analytical": "#0072B2", "initial": "#009E73", "ocp": "#D55E00"}
STYLE = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.labelsize": 17,
    "axes.titlesize": 16, "legend.fontsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
}


def load_worst_cases(path):
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    cases = [case for case in data["results"]
             if case.get("success") and case.get("ocp_success")
             and case.get("ocp_time") is not None
             and case.get("best_analytical_time") is not None
             and np.isfinite(case["ocp_time"] - case["best_analytical_time"])]
    cases.sort(key=lambda case: abs(case["ocp_time"] - case["best_analytical_time"]),
               reverse=True)
    if len(cases) < 3:
        raise ValueError("The sweep must contain at least three successful comparisons.")
    return data["metadata"], cases[:3]


def load_tst_builder():
    # Reuse the sweep's exact initialization without executing its main block.
    path = DIRECTORY / "param_sweep_pose2pose_unicycle_simple_initial_guess.py"
    spec = importlib.util.spec_from_file_location("tst_sweep_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_tst_initial_guess


def rerun_case(case, metadata, build_tst, linear_solver=LINEAR_SOLVER):
    if case.get("ocp_initial_guess", metadata.get("ocp_initial_guess")) != "TST":
        raise ValueError("This script reproduces sweeps using TST initialization.")
    start = Pose(Point(case["x0"], case["y0"]), case["theta0_rad"])
    end = Pose(Point(case["xf"], case["yf"]), case["thetaf_rad"])
    vehicle = Unicycle(
        state=[start.x, start.y, start.theta], width=0.430, length=0.430,
        v_max=case["v_max"], v_min=0, omega_max=case["omega_max"],
        omega_min=-case["omega_max"],
    )
    candidates = compute_all_pose_to_pose_trajectories(start, end, vehicle)
    name, best = min(candidates.items(), key=lambda item: item[1]["time"])
    if not np.isclose(float(best["time"]), case["best_analytical_time"], rtol=1e-9, atol=1e-8):
        raise ValueError(f"Analytical time does not reproduce saved case {case['case_id']}.")
    initial, initial_time, _ = build_tst(
        start, end, vehicle,
        samples_per_maneuver=metadata.get("initial_guess_samples_per_maneuver", 20),
    )
    if "tst_T_guess" in case and not np.isclose(initial_time, case["tst_T_guess"], rtol=1e-9):
        raise ValueError(f"TST initial guess does not reproduce case {case['case_id']}.")
    result = compute_ocp_pose_to_pose_trajectory(
        start, end, vehicle, analytical_initial_guess=initial, T_guess=initial_time,
        N=case["N"], M=case["M"], linear_solver=linear_solver,
    )
    if not result["success"]:
        raise RuntimeError(f"Case {case['case_id']} failed: {result['solver_error']}")
    summary = {
        "case_id": case["case_id"], "N": case["N"], "M": case["M"],
        "linear_solver": linear_solver, "initial_guess": "TST",
        "analytical_name": name, "analytical_time": float(best["time"]),
        "initial_guess_time": initial_time, "saved_ocp_time": case["ocp_time"],
        "recomputed_ocp_time": result["time"],
        "saved_discrepancy": abs(case["ocp_time"] - case["best_analytical_time"]),
        "recomputed_discrepancy": abs(result["time"] - float(best["time"])),
        "ocp_sequence": result["sequence"], "solve_time": result["solve_time"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return best["trajectory"], result, summary


def create_figure(runs, cases):
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(17, 11), layout="constrained")
        layout = figure.add_gridspec(2, 3, height_ratios=(1, 1.2), hspace=0.08)
        axes = [figure.add_subplot(layout[0, column]) for column in range(3)]
        control_axes = np.empty((2, 3), dtype=object)
        for column in range(3):
            controls = layout[1, column].subgridspec(2, 1, hspace=0.04)
            control_axes[0, column] = figure.add_subplot(controls[0])
            control_axes[1, column] = figure.add_subplot(
                controls[1], sharex=control_axes[0, column])
            control_axes[0, column].tick_params(labelbottom=False)
        for index, (axis, (analytical, ocp, summary), case) in enumerate(zip(axes, runs, cases)):
            guess = ocp["initial_guess"]
            axis.plot(np.asarray(guess["x"]).ravel(), np.asarray(guess["y"]).ravel(),
                      color=COLORS["initial"], linestyle=":", linewidth=2.4)
            plot_analytical_trajectory(analytical, figure=axis, color=COLORS["analytical"],
                                       linewidth=2.5, plot_turn_sectors=False)
            plot_primitive_arrows_and_markers(
                analytical, axis, intermediate_color=COLORS["analytical"],
                plot_turn_sectors=False,
            )
            axis.plot(ocp["xs"], ocp["ys"], color=COLORS["ocp"],
                      linestyle="--", linewidth=2.5)
            for suffix, x, y in (("0", case["x0"], case["y0"]),
                                 ("f", case["xf"], case["yf"])):
                axis.annotate(rf"$\mathbf{{p}}_{suffix}$", (x, y), xytext=(5, 6),
                              textcoords="offset points", fontsize=15)
            axis.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]",
                     title=f"{chr(97 + index)}) Case {case['case_id']}\n"
                     + rf"$|\Delta T|={summary['recomputed_discrepancy']:.4f}\,\mathrm{{s}}$")
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(axis.get_title(), pad=20)
            axis.margins(0.12)
            axis.grid(linestyle=":", alpha=0.4)
        create_control_figure(runs, cases, figure=figure, axes=control_axes)
        figure.legend(handles=[
            Line2D([], [], color=COLORS["analytical"], linewidth=2.5, label="Analytical"),
            Line2D([], [], color=COLORS["initial"], linestyle=":", linewidth=2.4,
                   label="OCP initial guess (TST)"),
            Line2D([], [], color=COLORS["ocp"], linestyle="--", linewidth=2.5,
                   label="OCP solution"),
            Line2D([], [], color="0.5", linestyle="-.", linewidth=1,
                   label="Control bounds"),
        ], loc="outside upper center", ncol=4, frameon=False)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        column_bounds = []
        for column in range(3):
            bounds = [axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
                      for axis in (axes[column], *control_axes[:, column])]
            column_bounds.append((min(box.x0 for box in bounds),
                                  max(box.x1 for box in bounds),
                                  min(box.y0 for box in bounds),
                                  max(box.y1 for box in bounds)))
        bottom = min(bounds[2] for bounds in column_bounds)
        top = max(bounds[3] for bounds in column_bounds)
        for column in range(2):
            x = (column_bounds[column][1] + column_bounds[column + 1][0]) / 2
            separator = Line2D([x, x], [bottom, top], transform=figure.transFigure,
                               color="0.8", linewidth=0.9)
            separator.set_in_layout(False)
            figure.add_artist(separator)
    return figure


def create_control_figure(runs, cases, figure=None, axes=None):
    """Compare controls in physical time; each curve ends at its own duration."""
    with plt.rc_context(STYLE):
        standalone = figure is None
        if standalone:
            figure, axes = plt.subplots(2, 3, figsize=(17, 8), sharex="col",
                                        layout="constrained")
        for column, ((analytical, ocp, summary), case) in enumerate(zip(runs, cases)):
            guess = ocp["initial_guess"]
            # Exact piecewise-constant analytical controls at primitive boundaries.
            analytical_times = np.r_[[primitive.t0 for primitive in analytical],
                                      analytical[-1].tf]
            profiles = (
                (analytical_times,
                 np.r_[[primitive.v for primitive in analytical], analytical[-1].v],
                 np.r_[[primitive.omega for primitive in analytical], analytical[-1].omega],
                 summary["analytical_time"], "analytical", "-", "Analytical"),
                (guess["ts_ctrl"], guess["v"], guess["omega"], guess["time"],
                 "initial", ":", "OCP initial guess (TST)"),
                (ocp["ts_ctrl"], ocp["vs"], ocp["omegas"], ocp["time"],
                 "ocp", "--", "OCP solution"),
            )
            for times, velocity, omega, duration, color_key, linestyle, label in profiles:
                times = np.asarray(times).ravel()
                for row, values in enumerate((velocity, omega)):
                    values = np.asarray(values).ravel()
                    if len(times) != len(values):
                        raise ValueError(f"Control/time length mismatch in case {case['case_id']}.")
                    # Extend the last held control to the trajectory's endpoint.
                    plot_times = np.r_[times, duration] if times[-1] < duration else times
                    plot_values = np.r_[values, values[-1]] if times[-1] < duration else values
                    axes[row, column].step(plot_times, plot_values, where="post",
                                           color=COLORS[color_key], linestyle=linestyle,
                                           linewidth=2.3, label=label)
            for row, bounds in enumerate(((0, case["v_max"]),
                                           (-case["omega_max"], case["omega_max"]))):
                axis = axes[row, column]
                axis.set_ylabel("")
                for bound in bounds:
                    axis.axhline(bound, color="0.5", linestyle="-.", linewidth=1,
                                 zorder=0)
                margin = 0.12 * (bounds[1] - bounds[0])
                axis.set_ylim(bounds[0] - margin, bounds[1] + margin)
                axis.set_xlim(0, max(profile[3] for profile in profiles) * 1.02)
                axis.grid(linestyle=":", alpha=0.4)
            if standalone:
                axes[0, column].set_title(f"{chr(97 + column)}) Case {case['case_id']}",
                                          pad=20)
            axes[1, column].set_xlabel(r"$t$ [s]")
        axes[0, 0].set_ylabel(r"$v(t)$ [m/s]")
        axes[1, 0].set_ylabel(r"$\omega(t)$ [rad/s]")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], color="0.5", linestyle="-.", linewidth=1))
        labels.append("Control bounds")
        if standalone:
            figure.legend(handles, labels, loc="outside upper center",
                          ncol=4, frameon=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=DIRECTORY / "results" / RESULTS_FILENAME)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--linear-solver", default=LINEAR_SOLVER,
                        help="IPOPT linear solver (default: mumps; use ma27 where installed).")
    args = parser.parse_args()
    path = args.input.expanduser()
    if not path.is_absolute() and not path.is_file():
        path = DIRECTORY / "results" / path
    metadata, cases = load_worst_cases(path)
    build_tst = load_tst_builder()
    runs = []
    for rank, case in enumerate(cases, 1):
        print(f"\nRunning rank {rank}, case {case['case_id']}, N={case['N']}, "
              f"M={case['M']}, {args.linear_solver}", flush=True)
        runs.append(rerun_case(case, metadata, build_tst, args.linear_solver))
    figure = create_figure(runs, cases)
    control_figure = create_control_figure(runs, cases)
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    stem = f"{path.stem}_three_worst_cases_{args.linear_solver}"
    for extension in ("pdf", "png"):
        output = FIGURES_DIRECTORY / f"{stem}.{extension}"
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}", flush=True)
        control_output = FIGURES_DIRECTORY / f"{stem}_controls.{extension}"
        control_figure.savefig(control_output, dpi=300, bbox_inches="tight")
        print(f"Saved {control_output}", flush=True)
    summary_path = FIGURES_DIRECTORY / f"{stem}.json"
    summary_path.write_text(json.dumps({"source": str(path),
                                        "cases": [run[2] for run in runs]}, indent=2) + "\n")
    if not args.no_show:
        plt.show(block=True)
    else:
        plt.close(figure)
        plt.close(control_figure)


if __name__ == "__main__":
    main()
