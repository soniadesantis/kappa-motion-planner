"""Recompute and plot three selected N300 Sobol cases.

The cases represent the largest saved time difference (3395), largest saved
Hausdorff distance (7337), and a case in both top-three lists (4092).
Uses MUMPS by default; pass --linear-solver ma27 on a machine with MA27.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import worst_cases_plot as base


DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_FILENAME = "sweep4/sobol_sweep_OCP_TST_initial_guess_N300_M4.json"
CASE_IDS = (3395, 7337, 4092)
CASE_ROLES = (
    "largest time difference",
    "largest Hausdorff distance",
    "in both top-three lists",
)


def load_selected_cases(path):
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    metadata = data["metadata"]
    if metadata.get("N") != 300:
        raise ValueError("Selected case IDs refer to the N300 sweep.")
    selected = {case["case_id"]: case for case in data["results"]
                if case.get("case_id") in CASE_IDS}
    missing = set(CASE_IDS) - selected.keys()
    if missing:
        raise ValueError(f"Missing case IDs: {sorted(missing)}")
    cases = [selected[case_id] for case_id in CASE_IDS]
    for case in cases:
        if not (case.get("success") and case.get("ocp_success")
                and case.get("best_analytical_time") is not None
                and case.get("ocp_time") is not None
                and case.get("hausdorff_distance") is not None):
            raise ValueError(f"Case {case['case_id']} has incomplete saved metrics.")
    return metadata, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=DIRECTORY / "results" / RESULTS_FILENAME)
    parser.add_argument("--linear-solver", default="mumps",
                        help="IPOPT linear solver (default: mumps).")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    path = args.input.expanduser()
    if not path.is_absolute() and not path.is_file():
        path = DIRECTORY / "results" / path

    metadata, cases = load_selected_cases(path)
    build_tst = base.load_tst_builder()
    runs = []
    for case, role in zip(cases, CASE_ROLES):
        print(f"\nRunning case {case['case_id']} ({role}) with "
              f"{args.linear_solver}", flush=True)
        analytical, ocp, summary = base.rerun_case(
            case, metadata, build_tst, args.linear_solver)
        summary["selection_role"] = role
        summary["saved_signed_time_difference"] = (
            case["ocp_time"] - case["best_analytical_time"])
        summary["saved_hausdorff_distance"] = case["hausdorff_distance"]
        summary["recomputed_signed_time_difference"] = (
            summary["recomputed_ocp_time"] - summary["analytical_time"])
        runs.append((analytical, ocp, summary))

    base.STYLE.update({
        "font.size": 17,
        "axes.labelsize": 19,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })
    figure = base.create_figure(runs, cases)
    for separator in list(figure.artists):
        if isinstance(separator, Line2D) and separator.get_color() == "0.8":
            separator.remove()
    for index, (axis, (_, _, summary), case, role) in enumerate(
            zip(figure.axes[:3], runs, cases, CASE_ROLES)):
        axis.set_anchor("N")
        axis.set_title(
            f"{chr(97 + index)}) Case {case['case_id']}\n{role.capitalize()}\n"
            + rf"$\Delta\mathcal{{T}}="
              rf"{summary['recomputed_signed_time_difference']:.4f}\,\mathrm{{s}}$",
            fontsize=21, pad=20,
        )
        for label in axis.texts:
            if label.get_text() in (r"$\mathbf{p}_0$", r"$\mathbf{p}_f$"):
                label.remove()
            else:
                label.set_fontsize(17)
    figure.legends[0].remove()
    figure.legend(handles=[
        Line2D([], [], color=base.COLORS["analytical"], linewidth=2.5,
               label="Analytical"),
        Line2D([], [], color=base.COLORS["initial"], linestyle=":", linewidth=2.4,
               label="OCP initial guess (TST)"),
        Line2D([], [], color=base.COLORS["ocp"], linestyle="--", linewidth=2.5,
               label="OCP solution"),
        Line2D([], [], color="g", marker="o", linestyle="None", markersize=9),
        Line2D([], [], color="r", marker="o", linestyle="None", markersize=9),
        Line2D([], [], color="0.5", linestyle="-.", linewidth=1,
               label="Control bounds"),
    ], labels=["Analytical", "OCP initial guess (TST)", "OCP solution",
               "Start pose", "End pose", "Control bounds"],
       loc="outside lower center", ncol=3, frameon=False, fontsize=18)
    control_figure = base.create_control_figure(runs, cases)
    for axis in control_figure.axes[:3]:
        axis.set_title(axis.get_title(), fontsize=21)
    control_figure.legends[0].remove()
    control_figure.legend(handles=[
        Line2D([], [], color=base.COLORS["analytical"], linewidth=2.5,
               label="Analytical"),
        Line2D([], [], color=base.COLORS["initial"], linestyle=":", linewidth=2.4,
               label="OCP initial guess (TST)"),
        Line2D([], [], color=base.COLORS["ocp"], linestyle="--", linewidth=2.5,
               label="OCP solution"),
        Line2D([], [], color="0.5", linestyle="-.", linewidth=1,
               label="Control bounds"),
    ], loc="outside lower center", ncol=4, frameon=False, fontsize=18)

    base.FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stem = f"{path.stem}_selected_worst_cases_{args.linear_solver}"
    for extension in ("pdf", "png"):
        output = base.FIGURES_DIRECTORY / f"{stem}.{extension}"
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}", flush=True)
        control_output = base.FIGURES_DIRECTORY / f"{stem}_controls.{extension}"
        control_figure.savefig(control_output, dpi=300, bbox_inches="tight")
        print(f"Saved {control_output}", flush=True)
    summary_path = base.FIGURES_DIRECTORY / f"{stem}.json"
    summary_path.write_text(json.dumps({
        "source": str(path), "linear_solver": args.linear_solver,
        "cases": [run[2] for run in runs],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {summary_path}", flush=True)
    if not args.no_show:
        plt.show(block=True)
    else:
        plt.close(figure)
        plt.close(control_figure)


if __name__ == "__main__":
    main()
