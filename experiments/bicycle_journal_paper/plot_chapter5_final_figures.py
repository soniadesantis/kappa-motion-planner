"""Create publication figures for the final Chapter 5 benchmark."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import AutoMinorLocator, LogLocator
import numpy as np

from benchmark_new_bicycle_baseline import (
    attach_boundary_connections,
    corridor_worlds_from_bounds,
    make_fixed_bicycle,
    plot_maneuver_path,
)
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from kappa_planner.refinement_new import refine_bicycle_baseline


ROOT = Path(__file__).parent
COLORS = {
    "internal": "#2563eb",
    "complete_baseline": "#0891b2",
    "direct_refinement": "#7c3aed",
    "final": "#16a34a",
    "baseline": "#1f2937",
    "refinement": "#7c3aed",
    "corridor_face": "#dbeafe",
    "corridor_edge": "#64748b",
}


def configure_style(latex_font=False):
    style = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if latex_font:
        style.update({
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        })
    plt.rcParams.update(style)


def save_figure(fig, output_stem, aliases=()):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for stem in (output_stem, *aliases):
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_success_rates(report, output):
    counts = [group["corridors"] for group in report["groups"]]
    quantities = {
        "Internal baseline": [sum(r["baseline_success"] for r in g["records"]) for g in report["groups"]],
        "Complete baseline": [sum(r["boundary_success"] for r in g["records"]) for g in report["groups"]],
        "Direct refinement": [sum(r["refinement"].get("solution_source") == "refined" for r in g["records"]) for g in report["groups"]],
        "Final trajectory": [sum(r["refinement"].get("status") == "success" for r in g["records"]) for g in report["groups"]],
    }
    colors = [COLORS[key] for key in ("internal", "complete_baseline", "direct_refinement", "final")]
    x = np.arange(len(counts))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    offsets = (np.arange(len(quantities)) - 1.5) * width
    for offset, (label, values), color in zip(offsets, quantities.items(), colors):
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, padding=2, fontsize=7.5)
    ax.set_xticks(x, counts)
    ax.set_xlabel("Number of corridors")
    ax.set_ylabel("Successful scenarios (out of 100)")
    ax.set_ylim(0, 108)
    ax.set_title("Planning success by processing stage")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        ncol=4, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.16), columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output / "planning_success_rates")


def plot_traversal_reduction(
    report,
    output,
    filename="traversal_time_reduction_violin",
    compatibility_alias=True,
    reference_style=False,
):
    counts, values = [], []
    for group in report["groups"]:
        paired = [
            record["comparison"]["traversal_time_reduction_percent"]
            for record in group["records"]
            if record["comparison"].get("paired_success")
        ]
        counts.append(group["corridors"])
        values.append(paired)
    positions = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(5.82, 4.37) if reference_style else (6.8, 4.3))
    violin = ax.violinplot(
        values, positions=positions, widths=0.78,
        showmeans=False, showmedians=False, showextrema=False,
        points=300, bw_method="scott",
    )
    for body in violin["bodies"]:
        body.set_facecolor("#9ecae1" if reference_style else COLORS["refinement"])
        body.set_edgecolor("#65737e" if reference_style else "#5b21b6")
        body.set_linewidth(1.0 if reference_style else 1.1)
        body.set_alpha(0.95 if reference_style else 0.82)

    quartile_1 = np.asarray([np.quantile(sample, 0.25) for sample in values])
    medians = np.asarray([np.median(sample) for sample in values])
    quartile_3 = np.asarray([np.quantile(sample, 0.75) for sample in values])
    if reference_style:
        minima = np.asarray([min(sample) for sample in values])
        maxima = np.asarray([max(sample) for sample in values])
        ax.vlines(positions, minima, maxima, color="black", linewidth=1.15, zorder=4)
        cap_width = 0.13
        ax.hlines(minima, positions - cap_width, positions + cap_width,
                  color="black", linewidth=1.15, zorder=4)
        ax.hlines(maxima, positions - cap_width, positions + cap_width,
                  color="black", linewidth=1.15, zorder=4)
        ax.hlines(medians, positions - 0.16, positions + 0.16,
                  color="#d00032", linewidth=1.5, zorder=5)
    else:
        ax.vlines(positions, quartile_1, quartile_3, color="#4c1d95", linewidth=5, zorder=4)
        ax.scatter(
            positions, medians, s=34, color="white", edgecolor="#4c1d95",
            linewidth=1.0, zorder=5, label="Median",
        )
    ax.axhline(0, color="#6b7280", linewidth=0.7)
    ax.set_ylim(-1.6, max(max(sample) for sample in values) + 2.0)
    if reference_style:
        labels = [rf"$n={count}$" + "\n" + rf"$N={len(sample)}$"
                  for count, sample in zip(counts, values)]
        ax.set_xticks(positions, labels, rotation=18)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(axis="y", which="major", color="#9ca3af", linewidth=0.55, alpha=0.75)
        ax.grid(axis="y", which="minor", color="#6b7280", linestyle=":",
                linewidth=0.65, alpha=0.9)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    else:
        for position, sample, median in zip(positions, values, medians):
            ax.text(position, median + 0.65, f"{median:.2f}%", ha="center", va="bottom",
                    fontsize=8, color="#4c1d95")
            ax.text(position, -1.25, f"n={len(sample)}", ha="center", va="bottom",
                    fontsize=7.5, color="#475569")
        ax.set_xticks(positions, counts)
        ax.grid(axis="y", alpha=0.2)
    if not reference_style:
        ax.set_xlabel("Number of corridors")
    ax.set_ylabel("Relative Traversal-Time Reduction [%]" if reference_style
                  else "Relative traversal-time reduction (%)")
    if not reference_style:
        ax.set_title("Traversal-time benefit of refinement with baseline fallback")
    fig.tight_layout()
    aliases = (
        (output / "traversal_time_reduction_boxplot",)
        if compatibility_alias else ()
    )
    save_figure(fig, output / filename, aliases=aliases)


def plot_computation_times(report, output):
    counts = np.asarray([group["corridors"] for group in report["groups"]])
    baseline = np.asarray([
        group["successful_baseline_timing_ms"]["median"] for group in report["groups"]
    ])
    complete = np.asarray([
        group["comparison"]["refined_computation_ms"]["median"] for group in report["groups"]
    ])
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(counts, baseline, "o-", color=COLORS["internal"], linewidth=2,
            markersize=5, label="Baseline construction")
    ax.plot(counts, complete, "s-", color=COLORS["refinement"], linewidth=2,
            markersize=5, label="Complete refined planner")
    for x, y in zip(counts, baseline):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COLORS["internal"])
    for x, y in zip(counts, complete):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COLORS["refinement"])
    ax.set_xticks(counts)
    ax.set_xlabel("Number of corridors")
    ax.set_ylabel("Median computation time (ms)")
    ax.set_title("Computation time versus corridor-sequence length")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "computation_times")


def plot_computation_time_violins(report, output, *, linear_scale=False):
    """Compare complete baseline and refined-planner times on paired cases."""
    samples = []
    labels = []
    for group in report["groups"]:
        corridor_count = group["corridors"]
        paired_records = [
            record for record in group["records"]
            if record["comparison"].get("paired_success")
        ]
        complete_baseline = [
            record["comparison"]["baseline_computation_ms"]
            for record in paired_records
        ]
        complete_planner = [
            record["comparison"]["refined_computation_ms"]
            for record in paired_records
        ]
        samples.extend((complete_baseline, complete_planner))
        labels.extend((f"B{corridor_count}", f"R{corridor_count}"))

    positions = np.arange(1, len(samples) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    violin = ax.violinplot(
        samples, positions=positions, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
        points=300, bw_method="scott",
    )
    for index, body in enumerate(violin["bodies"]):
        # Baseline and refined-planner distributions alternate on the axis.
        body.set_facecolor("#9ecae1" if index % 2 == 0 else "#a8ddb5")
        body.set_edgecolor("#65737e")
        body.set_linewidth(1.0)
        body.set_alpha(0.95)

    minima = np.asarray([min(sample) for sample in samples])
    medians = np.asarray([np.median(sample) for sample in samples])
    maxima = np.asarray([max(sample) for sample in samples])
    ax.vlines(positions, minima, maxima, color="black", linewidth=1.1, zorder=4)
    ax.hlines(minima, positions - 0.12, positions + 0.12,
              color="black", linewidth=1.1, zorder=4)
    ax.hlines(maxima, positions - 0.12, positions + 0.12,
              color="black", linewidth=1.1, zorder=4)
    ax.hlines(medians, positions - 0.15, positions + 0.15,
              color="#d00032", linewidth=1.5, zorder=5)

    if linear_scale:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax.set_yscale("log")
        # Both benchmark datasets remain below 500 ms while the upper limit
        # leaves visible headroom above the observed R50 extrema.
        ax.set_ylim(1.0, 500.0)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.grid(axis="y", which="major", color="#9ca3af", linewidth=0.55, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#6b7280", linestyle=":",
            linewidth=0.65, alpha=0.9)
    ax.set_xticks(positions, labels, rotation=0, fontsize=12)
    ax.tick_params(axis="y", which="both", labelsize=12)
    ax.set_ylabel(
        "Computation Time [ms]" if linear_scale
        else "Computation Time [ms] (log scale)",
        fontsize=13,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    fig.tight_layout()
    filename = (
        "computation_times_violin_matched_linear"
        if linear_scale else "computation_times_violin_matched"
    )
    save_figure(fig, output / filename)


def longest_turn_run(types):
    best = current = 0
    for transition in types:
        current = current + 1 if transition == "turn" else 0
        best = max(best, current)
    return best


def select_representatives(report):
    groups = {group["corridors"]: group for group in report["groups"]}
    eligible = lambda group: [
        record for record in group["records"]
        if record["boundary_success"]
        and record["refinement"].get("solution_source") == "refined"
    ]
    short = min(
        eligible(groups[10]),
        key=lambda r: (min(r["sampled"]["widths"][1:-1]),
                       min(r["sampled"]["walk_lengths"][1:-1])),
    )
    clustered = max(
        eligible(groups[30]),
        key=lambda r: (
            longest_turn_run(r["sampled"]["transition_types"]),
            sum(length < 5.0 for length in r["sampled"]["walk_lengths"][1:-1]),
        ),
    )
    long_candidates = eligible(groups[50])
    reductions = np.asarray([
        record["comparison"]["traversal_time_reduction_percent"]
        for record in long_candidates
    ])
    target = float(np.quantile(reductions, 0.75))
    long_case = min(
        long_candidates,
        key=lambda r: abs(r["comparison"]["traversal_time_reduction_percent"] - target),
    )
    return [
        ("Short sequence with a narrow passage", 10, short),
        ("Closely spaced sequence of turns", 30, clustered),
        ("Long sequence with substantial refinement", 50, long_case),
    ]


def replay_complete(record, bicycle):
    corridors = corridor_worlds_from_bounds(record["bounds"], record["sampled"]["headings"])
    start, end = record["initial_pose"]["world"], record["final_pose"]["world"]
    baseline = compute_bicycle_baseline(corridors, bicycle, initial_pose=start, final_pose=end)
    boundary = attach_boundary_connections(corridors, baseline, bicycle, start, end)
    refinement, failure = refine_bicycle_baseline(
        corridors, bicycle, baseline,
        initial_pose=start, final_pose=end, return_failure=True,
    )
    if boundary["status"] != "success" or refinement is None or refinement.trajectory is None:
        raise RuntimeError(f"Representative replay failed: {boundary}, {failure}")
    return corridors, baseline, refinement


def plot_representative_trajectories(report, output):
    bicycle = make_fixed_bicycle(
        arc_radius=report["vehicle"]["arc_radius"],
        footprint_radius=report["vehicle"]["footprint_radius"],
    )
    selected = select_representatives(report)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    metadata = []
    for ax, (description, corridor_count, record) in zip(axes, selected):
        corridors, baseline, refinement = replay_complete(record, bicycle)
        for corridor in corridors:
            ax.add_patch(Polygon(
                corridor.corners, closed=True,
                facecolor=COLORS["corridor_face"], edgecolor=COLORS["corridor_edge"],
                linewidth=0.7, alpha=0.55,
            ))
        plot_maneuver_path(ax, baseline.trajectory, COLORS["baseline"], linewidth=1.4)
        plot_maneuver_path(ax, refinement.trajectory, COLORS["refinement"], linewidth=2.0)
        for key, color in (("initial_pose", "#16a34a"), ("final_pose", "#dc2626")):
            x, y, heading = record[key]["world"]
            ax.plot(x, y, "o", color=color, markersize=3.5, zorder=8)
            ax.annotate("", (x + np.cos(heading), y + np.sin(heading)), xytext=(x, y),
                        arrowprops={"arrowstyle": "->", "color": color}, zorder=8)
        reduction = record["comparison"]["traversal_time_reduction_percent"]
        ax.set_title(
            f"{description}\n$n={corridor_count}$, case {record['case']}, "
            f"$\\Delta T={reduction:.1f}\\%$",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        ax.margins(0.06)
        ax.set_xticks([])
        ax.set_yticks([])
        metadata.append({
            "description": description,
            "corridors": corridor_count,
            "case": record["case"],
            "generated_index": record["generated_index"],
            "minimum_interior_width": min(record["sampled"]["widths"][1:-1]),
            "minimum_interior_length": min(record["sampled"]["walk_lengths"][1:-1]),
            "longest_turn_run": longest_turn_run(record["sampled"]["transition_types"]),
            "traversal_time_reduction_percent": reduction,
        })
    fig.legend(handles=[
        Patch(facecolor=COLORS["corridor_face"], edgecolor=COLORS["corridor_edge"], label="Corridors"),
        Line2D([], [], color=COLORS["baseline"], linewidth=1.5, label="Complete baseline"),
        Line2D([], [], color=COLORS["refinement"], linewidth=2, label="Refined trajectory"),
        Line2D([], [], color="#16a34a", marker="o", linestyle="none", label="Initial pose"),
        Line2D([], [], color="#dc2626", marker="o", linestyle="none", label="Final pose"),
    ], loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, output / "representative_trajectories")
    (output / "representative_trajectories.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "final_chapter5_benchmark_500.json",
    )
    parser.add_argument("--latex-font", action="store_true")
    parser.add_argument("--reference-style", action="store_true")
    parser.add_argument("--linear-scale", action="store_true")
    parser.add_argument(
        "--only", choices=("all", "traversal", "computation-violin"), default="all",
        help="Generate every Chapter 5 figure or only the traversal-time violin.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline" / "chapter5_final",
    )
    args = parser.parse_args()
    configure_style(latex_font=args.latex_font and not args.reference_style)
    if args.reference_style:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        })
    report = json.loads(args.report.read_text())
    if args.only == "traversal":
        filename = (
            "traversal_time_reduction_violin_matched"
            if args.reference_style else
            ("traversal_time_reduction_violin_latex"
             if args.latex_font else "traversal_time_reduction_violin")
        )
        plot_traversal_reduction(
            report, args.output, filename=filename, compatibility_alias=False,
            reference_style=args.reference_style,
        )
        print(f"Saved traversal-time violin to {args.output}")
        return
    if args.only == "computation-violin":
        plot_computation_time_violins(
            report, args.output, linear_scale=args.linear_scale,
        )
        print(f"Saved computation-time violin to {args.output}")
        return
    plot_success_rates(report, args.output)
    plot_traversal_reduction(report, args.output)
    plot_computation_times(report, args.output)
    plot_representative_trajectories(report, args.output)
    print(f"Saved final Chapter 5 figures to {args.output}")


if __name__ == "__main__":
    main()
