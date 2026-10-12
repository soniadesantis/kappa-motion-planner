from types import SimpleNamespace

import numpy as np

import kappa_planner.helpers.refinement as refinement


def test_invalidated_skip_reinstates_the_omitted_turn(monkeypatch):
    """A shortcut failure must restore turns omitted by an earlier skip."""
    circle_groups = [
        {1: SimpleNamespace(center=np.array([float(index), 0.0]),
                            placement_rule="independent")}
        for index in range(3)
    ]
    baseline = SimpleNamespace(turn_directions=[1, 1, 1])
    intersection_searches = iter([(0, 1), None])

    monkeypatch.setattr(
        refinement,
        "find_first_problematic_tangent_intersection",
        lambda *args, **kwargs: next(intersection_searches),
    )
    monkeypatch.setattr(
        refinement,
        "find_longest_safe_skip_in_block",
        lambda *args, **kwargs: (0, 2, SimpleNamespace()),
    )

    rebuild_calls = []

    def rebuild(*args, **kwargs):
        active = list(kwargs["active_circle_indices"])
        rebuild_calls.append(active)
        if active == [0, 2]:
            kwargs["failure_info"]["pair"] = (0, 2)
            return None
        return [SimpleNamespace(), SimpleNamespace()]

    monkeypatch.setattr(
        refinement,
        "build_and_repair_tangent_chain",
        rebuild,
    )
    monkeypatch.setattr(
        refinement,
        "compute_baseline_turn_circle",
        lambda baseline, transition_index, R, placement_rule, tol: SimpleNamespace(
            center=np.array([float(transition_index), 0.0]),
            placement_rule=placement_rule,
        ),
    )

    active, tangents = refinement.simplify_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=[SimpleNamespace()] * 4,
        baseline=baseline,
        R=1.0,
        r=0.25,
        active_circle_indices=[0, 1, 2],
        tangents=[SimpleNamespace(), SimpleNamespace()],
    )

    assert rebuild_calls == [[0, 2], [0, 1, 2]]
    assert active == [0, 1, 2]
    assert len(tangents) == 2
    assert circle_groups[1][1].placement_rule == "baseline_repair"
