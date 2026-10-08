"""Investigate benchmark shortcuts without changing production repair rules."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import kappa_planner.refinement_new as refinement
from benchmark_new_bicycle_baseline import attach_boundary_connections
from examples_maps_polyline import example_corridor_sequence
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from refinement_failure_examples import REFINEMENT_FAILURE_EXAMPLES


def diagnose(number, extend_right=0):
    record = REFINEMENT_FAILURE_EXAMPLES[number]
    corridors, start, end, bicycle = example_corridor_sequence(number)
    baseline = compute_bicycle_baseline(corridors, bicycle, initial_pose=start, final_pose=end)
    attach_boundary_connections(corridors, baseline, bicycle, start, end)
    original_build = refinement.build_and_repair_tangent_chain
    original_skip = refinement.find_longest_safe_skip_in_block
    original_restore = refinement.restore_active_circle_block_to_baseline
    history, skipped, failed_states = [], [], []

    def observe_skip(*args, **kwargs):
        inputs = inspect.signature(original_skip).bind(*args, **kwargs).arguments
        result = original_skip(*args, **kwargs)
        active = inputs["active_circle_indices"]
        entry = {"operation": "skip_search", "block": [i+1 for i in active[
            inputs["block_start_position"]:inputs["block_end_position"]+1]]}
        if result is not None:
            left, right, _ = result
            i, j = active[left], active[right]
            removed = active[left+1:right]
            centers = [refinement.get_single_circle(inputs["circle_groups"][k]).center.copy()
                       for k in (i, j)]
            skipped.append((i, j, list(removed), centers))
            entry.update(shortcut=[i+1, j+1], removed=[k+1 for k in removed])
        history.append(entry)
        return result

    def observe_restore(*args, **kwargs):
        inputs = inspect.signature(original_restore).bind(*args, **kwargs).arguments
        active = inputs["active_circle_indices"]
        block = active[inputs["block_start_position"]:inputs["block_end_position"]+1]
        result = original_restore(*args, **kwargs)
        history.append({"operation": "baseline_restore", "block": [i+1 for i in block],
                        "success": bool(result[0]), "changed": bool(result[1])})
        return result

    def observe_build(*args, **kwargs):
        inputs = inspect.signature(original_build).bind(*args, **kwargs).arguments
        result = original_build(*args, **kwargs)
        if result is None:
            groups, active = inputs["circle_groups"], inputs["active_circle_indices"]
            for i, j in zip(active, active[1:]):
                if refinement.compute_safe_circle_tangent_between_indices(
                    groups, corridors, i, j, bicycle.width/2,
                ) is None:
                    failed_states.append((deepcopy(groups), list(active), i, j))
                    history.append({"operation": "rebuild_failed", "pair": [i+1, j+1]})
                    break
        return result

    with patch.object(refinement, "find_longest_safe_skip_in_block", side_effect=observe_skip), \
         patch.object(refinement, "restore_active_circle_block_to_baseline", side_effect=observe_restore), \
         patch.object(refinement, "build_and_repair_tangent_chain", side_effect=observe_build):
        result, failure = refinement.refine_bicycle_baseline(
            corridors, bicycle, baseline, start, end, return_failure=True)
    assert result is None and failure.reason == record["refinement"]["failure_reason"]
    groups, active, i, j = failed_states[-1]
    previous = next((skip for skip in reversed(skipped) if skip[:2] == (i, j)), None)
    changed = previous is not None and any(not np.allclose(
        refinement.get_single_circle(groups[k]).center, center, atol=1e-9, rtol=0,
    ) for k, center in zip((i, j), previous[3]))
    summary = {
        "example": number, "case": record["case"], "history": history,
        "failed_pair": [i+1, j+1], "previously_accepted_skip": previous is not None,
        "endpoints_moved_after_skip": bool(changed),
        "baseline_traversal_time": baseline.traversal_time,
    }

    # Restore the omitted genuine turns at the failed state, rather than
    # changing initial circle placement. Reject the trial if crossings remain.
    trial_groups = deepcopy(groups)
    restored = [k for k in range(i, min(j+1+extend_right, len(groups)))
                if baseline.turn_directions[k] != 0]
    for k in restored:
        trial_groups[k] = {baseline.turn_directions[k]: refinement.compute_baseline_turn_circle(
            baseline, k, bicycle.max_radius, placement_rule="baseline_diagnostic_restore")}
    trial_indices = sorted(set(active) | set(restored))
    trial_tangents = original_build(trial_groups, corridors, baseline, bicycle.max_radius,
                                   bicycle.width/2, active_circle_indices=trial_indices)
    intersection = None if trial_tangents is None else \
        refinement.find_first_problematic_tangent_intersection(trial_tangents)
    compatible = trial_tangents is not None and intersection is None
    summary.update(restored_block=[k+1 for k in restored],
                   restored_chain_has_safe_tangents_and_no_intersections=compatible)
    if intersection is not None:
        summary["remaining_intersection"] = [
            [trial_tangents[k].start_circle_index+1, trial_tangents[k].end_circle_index+1]
            for k in intersection]
    if compatible:
        original_simplify = refinement.simplify_tangent_chain
        applied = False

        def trial_simplify(*args, **kwargs):
            nonlocal applied
            value = original_simplify(*args, **kwargs)
            if value is not None or applied:
                return value
            applied = True
            inputs = inspect.signature(original_simplify).bind(*args, **kwargs).arguments
            inputs["circle_groups"][:] = deepcopy(trial_groups)
            return list(trial_indices), deepcopy(trial_tangents)

        # Continue through normal boundary construction and arc validation.
        with patch.object(refinement, "simplify_tangent_chain", side_effect=trial_simplify):
            repaired, repair_failure = refinement.refine_bicycle_baseline(
                corridors, bicycle, baseline, start, end, return_failure=True)
        summary.update(complete_repair_succeeded=repaired is not None,
                       remaining_failure=None if repair_failure is None else repair_failure.reason)
        if repaired is not None:
            summary.update(repaired_traversal_time=repaired.traversal_time,
                           improvement_percent=100*(baseline.traversal_time-repaired.traversal_time)
                           / baseline.traversal_time)
    else:
        summary["complete_repair_succeeded"] = False
    return summary


def main():
    summaries = [diagnose(number) for number in (36, 37, 42, 43, 44)]
    summaries.extend(diagnose(42, extend_right=extra) for extra in (1, 2, 3, 4))
    destination = Path(__file__).parent / "results" / "skipping_failure_diagnostics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summaries, indent=2)+"\n")
    for summary in summaries:
        print(json.dumps(summary), flush=True)
    print(destination)


if __name__ == "__main__":
    main()
