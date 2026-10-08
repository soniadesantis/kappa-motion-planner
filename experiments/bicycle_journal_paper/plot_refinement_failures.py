"""Replay saved benchmark failures and plot the rejected refinement geometry."""

from argparse import ArgumentParser
from copy import deepcopy
import inspect
import json
from pathlib import Path
from textwrap import fill
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np

import kappa_planner.refinement_new as refinement
from benchmark_new_bicycle_baseline import (
    attach_boundary_connections, corridor_worlds_from_bounds, make_fixed_bicycle,
)
from bicycle_axis_aligned_corridors import draw_sequence
from kappa_planner.baseline_construction_new import compute_bicycle_baseline

ROOT = Path(__file__).parent
REASONS = {
    "no_tangent_connection_found": "Internal tangent chain could not be repaired",
    "refined_trajectory_assembly_failed": "Refined arc assembly or validation failed",
    "boundary_tangent_chain_could_not_be_simplified": "Intersecting boundary tangent could not be simplified",
    "boundary_connection_failed_after_circle_fallback": "Boundary connection failed, including baseline-circle retry",
}


def replay_failure(record, bicycle):
    corridors = corridor_worlds_from_bounds(record["bounds"], record["sampled"]["headings"])
    start, end = record["initial_pose"]["world"], record["final_pose"]["world"]
    baseline = compute_bicycle_baseline(corridors, bicycle, initial_pose=start, final_pose=end)
    if baseline is None:
        raise RuntimeError(f"Cannot reproduce baseline for case {record['case']}.")
    attach_boundary_connections(corridors, baseline, bicycle, start, end)
    trace = {"arcs": [], "failed_arcs": [], "failed_boundary": None}
    names = ["build_and_repair_tangent_chain", "simplify_tangent_chain",
             "assembled_refinement_arc_is_safe", "assemble_refined_trajectory",
             "try_boundary_connection_with_baseline_fallback", "try_skip_boundary_circle",
             "simplify_boundary_tangent_chain"]
    originals = {name: getattr(refinement, name) for name in names}

    def wrapper(name):
        original = originals[name]

        def observe(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            if name == "build_and_repair_tangent_chain" and "positioned" not in trace:
                groups = deepcopy(inputs["circle_groups"])
                trace["positioned"] = {
                    "groups": groups,
                    "indices": refinement.get_active_circle_indices(groups),
                    "tangents": [],
                }
            result = original(*args, **kwargs)
            if "circle_groups" in inputs:
                groups = deepcopy(inputs["circle_groups"])
                indices = inputs.get("active_circle_indices")
                if indices is None:
                    indices = refinement.get_active_circle_indices(groups)
                state = {"groups": groups, "indices": list(indices),
                         "tangents": deepcopy(inputs.get("tangents") or [])}
                for key in ("initial_maneuvers", "final_maneuvers"):
                    if key in inputs:
                        state[key] = deepcopy(inputs[key])
                if name == "build_and_repair_tangent_chain":
                    state["tangents"] = deepcopy(result or [])
                trace["state"] = state
                if result is None:
                    trace[name] = state
            if name == "assembled_refinement_arc_is_safe":
                trace["arcs"].append(deepcopy(inputs["arc"]))
                if not result:
                    trace["failed_arcs"].append((deepcopy(inputs["arc"]), deepcopy(inputs["circle"])))
            if name == "try_boundary_connection_with_baseline_fallback" and result[0] is None:
                trace["failed_boundary"] = ("initial" if inputs["initial"] else "final",
                                            inputs["transition_index"])
            if name == "try_skip_boundary_circle" and result is None:
                trace["failed_skip_side"] = inputs["side"]
            if name == "assemble_refined_trajectory":
                trace["initial_maneuvers"] = deepcopy(inputs["initial_maneuvers"])
                trace["final_maneuvers"] = deepcopy(inputs["final_maneuvers"])
            return result

        return observe

    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in names:
            stack.enter_context(patch.object(refinement, name, side_effect=wrapper(name)))
        result, failure = refinement.refine_bicycle_baseline(
            corridors, bicycle, baseline, initial_pose=start, final_pose=end, return_failure=True)
    expected = record["refinement"]["failure_reason"]
    if result is not None or failure is None or failure.reason != expected:
        raise RuntimeError(f"Failure no longer matches saved case {record['case']}: {failure}")
    return corridors, baseline, trace


def diagnose(record, corridors, trace, bicycle):
    reason = record["refinement"]["failure_reason"]
    highlights = []
    detail = REASONS.get(reason, reason)
    state = trace.get("state", {})
    if reason == "no_tangent_connection_found":
        state = trace.get("build_and_repair_tangent_chain", state)
        indices, groups = state.get("indices", []), state.get("groups", [])
        for i, k in zip(indices, indices[1:]):
            tangent = refinement.compute_safe_circle_tangent_between_indices(
                groups, corridors, i, k, bicycle.width/2)
            if tangent is not None:
                continue
            first, second = refinement.get_single_circle(groups[i]), refinement.get_single_circle(groups[k])
            candidate = refinement.compute_correct_circle_tangent(first, second, i, k)
            if candidate is None:
                highlights.append(("line", np.array([first.center, second.center])))
                detail = f"Transitions {i+1} to {k+1}: no compatible common tangent, even after baseline repair."
            else:
                highlights.append(("line", np.array([candidate.start_point, candidate.end_point])))
                detail = f"Transitions {i+1} to {k+1}: tangent leaves the safe corridor union, even after baseline repair."
            break
        else:
            detail = "Tangent intersections persist after circle skipping and baseline restoration."
    elif reason == "refined_trajectory_assembly_failed" and trace["failed_arcs"]:
        arc, circle = trace["failed_arcs"][-1]
        highlights.append(("arc", arc))
        groups = state.get("groups", [])
        index = next((j+1 for j, group in enumerate(groups) if group and
                      np.allclose(refinement.get_single_circle(group).center, circle.center)), None)
        detail = f"Arc at transition {index} fails the directed-arc ordering or corridor-clearance check."
    elif reason == "boundary_tangent_chain_could_not_be_simplified":
        state = trace.get("simplify_boundary_tangent_chain", state)
        side = trace.get("failed_skip_side", "boundary")
        detail = f"The {side} boundary tangent intersects the chain; removing its target circle did not yield feasible connections."
        maneuvers = state.get(f"{side}_maneuvers") or []
        if maneuvers:
            segment = maneuvers[-1] if side == "initial" else maneuvers[0]
            highlights.append(("line", np.array([[segment.x0, segment.y0], [segment.xf, segment.yf]])))
    elif reason == "boundary_connection_failed_after_circle_fallback":
        side, index = trace["failed_boundary"] or ("boundary", 0)
        detail = f"The {side} pose cannot connect to transition {index+1} with the supported maneuver construction, including baseline-circle retry."
        pose = record["initial_pose" if side == "initial" else "final_pose"]["world"]
        highlights.append(("point", np.array(pose[:2])))
    return state, detail, highlights


def draw_case(ax, record, corridors, baseline, trace, bicycle, fontsize=10, positioning=False):
    if positioning:
        state, highlights = trace["positioned"], []
        detail = "Initial circle placement before tangent construction, repair, or skipping."
    else:
        state, detail, highlights = diagnose(record, corridors, trace, bicycle)
    for j, corridor in enumerate(corridors, 1):
        ax.add_patch(Polygon(corridor.corners, facecolor="#eef2f6", edgecolor="#64748b", linewidth=.6))
        ax.text(*corridor.center, str(j), fontsize=7, color="#64748b", ha="center", va="center")
    if baseline.trajectory:
        draw_sequence(ax, baseline.trajectory, "#8b929e", 1.2, zorder=2)
    else:
        for fillet in baseline.fillets:
            if fillet is None:
                continue
            angle = np.arctan2(*(fillet.start_point-fillet.center)[::-1])
            angles = angle + fillet.turn_direction*np.linspace(0, np.pi/2, 101)
            points = fillet.center + bicycle.max_radius*np.column_stack([np.cos(angles), np.sin(angles)])
            ax.plot(*points.T, color="#8b929e", linewidth=1.2, zorder=2)
        for first, second in zip(baseline.waypoints, baseline.waypoints[1:]):
            ax.plot(*np.array([first, second]).T, color="#8b929e", linewidth=.8, linestyle="--", zorder=2)
        draw_sequence(ax, baseline.initial_maneuvers or [], "#8b929e", 1.2, zorder=2)
        draw_sequence(ax, baseline.final_maneuvers or [], "#8b929e", 1.2, zorder=2)
    groups = state.get("groups", [])
    for index in state.get("indices", []):
        circle = refinement.get_single_circle(groups[index])
        color = ("#0f766e" if refinement.is_baseline_circle(circle) else "#7c3aed") if positioning else "#2563eb"
        ax.add_patch(Circle(circle.center, circle.radius, fill=False, edgecolor=color, linewidth=.8, zorder=3))
        ax.plot(*circle.center, marker="o", markersize=2.5, color=color, zorder=5)
        if positioning:
            ax.annotate(f"{index+1}", circle.center, xytext=(6, 6), textcoords="offset points",
                        fontsize=fontsize-1, color=color, zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.3))
    for tangent in state.get("tangents", []):
        ax.plot(*np.array([tangent.start_point, tangent.end_point]).T, color="#2563eb", linewidth=1, zorder=4)
    if not positioning:
        draw_sequence(ax, trace["arcs"], "#2563eb", 1, zorder=4)
    for kind, geometry in highlights:
        if kind == "line":
            ax.plot(*geometry.T, color="#dc2626", linewidth=2, zorder=7)
        elif kind == "arc":
            draw_sequence(ax, [geometry], "#dc2626", 2, zorder=7)
        else:
            ax.plot(*geometry, marker="o", markersize=12, markerfacecolor="none",
                    markeredgecolor="#dc2626", markeredgewidth=2, zorder=7)
    for key, color in [("initial_pose", "#16803a"), ("final_pose", "#dc2626")]:
        x, y, heading = record[key]["world"]
        ax.plot(x, y, marker="o", color=color, markersize=4, zorder=8)
        ax.annotate("", (x+np.cos(heading), y+np.sin(heading)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color), zorder=8)
    source = record["refinement"].get("solution_source")
    outcome = "Complete baseline returned" if source == "baseline_fallback" else "No complete baseline or refined trajectory"
    title = "Initial circle positioning" if positioning else REASONS.get(record['refinement']['failure_reason'], 'Refinement failed')
    ax.set_title(f"Case {record['case']} — {title}",
                 fontsize=fontsize, wrap=True)
    caption = detail if positioning else detail+"\n"+outcome
    ax.text(.5, -.035, fill(caption, 65 if fontsize <= 10 else 85),
            transform=ax.transAxes, fontsize=fontsize-1, ha="center", va="top")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(.08)
    ax.set_axis_off()
    return detail


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT/"results"/"short_corridors_R1_500.json")
    parser.add_argument("--corridors", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT/"figures"/"refinement_failures_n10")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    group = next(g for g in report["groups"] if g["corridors"] == args.corridors)
    records = [r for r in group["records"] if r["refinement"].get("attempt_status") in ("failed", "error")]
    bicycle = make_fixed_bicycle()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = (len(records)+2)//3
    figure, axes = plt.subplots(rows, 3, figsize=(19, 6.5*rows), squeeze=False)
    figure.subplots_adjust(left=.025, right=.975, top=.95, bottom=.05, hspace=.55, wspace=.25)
    placement_figure, placement_axes = plt.subplots(rows, 3, figsize=(19, 6.5*rows), squeeze=False)
    placement_figure.subplots_adjust(left=.025, right=.975, top=.95, bottom=.05, hspace=.35, wspace=.25)
    notes = []
    for ax, placement_ax, record in zip(axes.flat, placement_axes.flat, records):
        corridors, baseline, trace = replay_failure(record, bicycle)
        detail = draw_case(ax, record, corridors, baseline, trace, bicycle)
        draw_case(placement_ax, record, corridors, baseline, trace, bicycle, positioning=True)
        individual, individual_axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
        individual.subplots_adjust(bottom=.16, top=.90, wspace=.12)
        draw_case(individual_axes[0], record, corridors, baseline, trace, bicycle, fontsize=12, positioning=True)
        draw_case(individual_axes[1], record, corridors, baseline, trace, bicycle, fontsize=12)
        individual.legend(handles=[Line2D([], [], color="#8b929e", label="Baseline"),
                                   Line2D([], [], color="#7c3aed", marker="o", label="Repositioned circles"),
                                   Line2D([], [], color="#0f766e", marker="o", label="Baseline circles"),
                                   Line2D([], [], color="#2563eb", label="Candidate refinement"),
                                   Line2D([], [], color="#dc2626", label="Rejected geometry")],
                          loc="lower center", ncol=5, frameon=False, fontsize=11)
        for extension in ("pdf", "png"):
            individual.savefig(args.output/f"case_{record['case']:03d}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(individual)
        notes.append({"case": record["case"], "reason": record["refinement"]["failure_reason"],
                      "detail": detail, "solution_source": record["refinement"].get("solution_source")})
        print(f"Case {record['case']}: {detail}", flush=True)
    for ax in list(axes.flat)[len(records):]:
        ax.set_axis_off()
    for ax in list(placement_axes.flat)[len(records):]:
        ax.set_axis_off()
    figure.suptitle(f"Failed refinement attempts — {args.corridors} corridors ({len(records)} cases)", fontsize=18)
    figure.legend(handles=[Line2D([], [], color="#8b929e", label="Baseline"),
                           Line2D([], [], color="#2563eb", label="Candidate refinement geometry"),
                           Line2D([], [], color="#dc2626", linewidth=2, label="Rejected connection / arc / pose")],
                  loc="lower center", ncol=3, frameon=False, fontsize=12)
    for extension in ("pdf", "png"):
        path = args.output/f"overview.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    placement_figure.suptitle(f"Circle placement before tangent construction — {args.corridors} corridors", fontsize=18)
    placement_figure.legend(handles=[Line2D([], [], color="#8b929e", label="Baseline"),
                                     Line2D([], [], color="#7c3aed", marker="o", label="Repositioned circles"),
                                     Line2D([], [], color="#0f766e", marker="o", label="Circles retained at baseline")],
                            loc="lower center", ncol=3, frameon=False, fontsize=12)
    for extension in ("pdf", "png"):
        path = args.output/f"circle_positioning.{extension}"
        placement_figure.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    (args.output/"failure_details.json").write_text(json.dumps(notes, indent=2)+"\n")
    plt.close(figure)
    plt.close(placement_figure)


if __name__ == "__main__":
    main()
