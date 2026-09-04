# ruff: noqa: ANN201, CPY001, D103, S101
"""Tests for standing-assumption width and circle-shift bounds."""

from __future__ import annotations

from math import cos, pi
from types import SimpleNamespace

import pytest

from kappa_planner import CorridorWorld
from kappa_planner.helpers.inputs_check import compute_minimum_widths


def make_planner(widths: list[float], tilts: list[float]):
    corridors = [
        CorridorWorld(width=width, height=5.0, center=[0.0, 0.0], tilt=tilt)
        for width, tilt in zip(widths, tilts)
    ]
    vehicle = SimpleNamespace(max_radius=2.0, width=0.5)
    return SimpleNamespace(corridor_list=corridors, vehicle=vehicle)


def test_width_equal_to_local_requirement_has_zero_shift() -> None:
    beta = pi / 4
    required_width = 2.0 + 0.25 - (2.0 - 0.25) * cos(beta)
    planner = make_planner([required_width, required_width], [0.0, pi / 2])

    minimum_widths, shifts = compute_minimum_widths(planner)

    assert minimum_widths == pytest.approx([required_width, required_width])
    assert shifts == pytest.approx([0.0], abs=1e-12)


def test_corridor_narrower_than_local_requirement_has_no_admissible_shift() -> None:
    beta = pi / 4
    required_width = 2.0 + 0.25 - (2.0 - 0.25) * cos(beta)
    planner = make_planner([required_width - 1e-4, required_width], [0.0, pi / 2])

    minimum_widths, shifts = compute_minimum_widths(planner)

    assert minimum_widths == pytest.approx([required_width, required_width])
    assert shifts == pytest.approx([0.0])


def test_opposite_corridor_directions_are_rejected() -> None:
    planner = make_planner([3.0, 3.0], [0.0, pi])

    with pytest.raises(ValueError, match=r"cos\(beta\) is zero"):
        compute_minimum_widths(planner)


def test_at_least_two_corridors_are_required() -> None:
    planner = make_planner([3.0], [0.0])

    with pytest.raises(ValueError, match="At least two corridors"):
        compute_minimum_widths(planner)
