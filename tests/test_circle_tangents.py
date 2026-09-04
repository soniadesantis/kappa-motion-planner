# ruff: noqa: CPY001, D103, S101
"""Tests for tangent segments between intermediate circles."""

from math import pi

import pytest

from kappa_planner.helpers.primitives import compute_extreme_poses_arc_line


@pytest.mark.parametrize("turn1", [-1, 1])
def test_opposite_turn_touching_circles_have_contact_point_heading(turn1: int) -> None:
    result = compute_extreme_poses_arc_line(
        xc1=0.0,
        yc1=0.0,
        xc2=2.0,
        yc2=0.0,
        turn1=turn1,
        turn2=-turn1,
        R=1.0,
    )

    x1, y1, theta1, x2, y2, theta2 = result
    assert (x1, y1) == pytest.approx((1.0, 0.0))
    assert (x2, y2) == pytest.approx((1.0, 0.0))
    assert theta1 == pytest.approx((turn1 * pi / 2) % (2 * pi))
    assert theta2 == pytest.approx(theta1)


@pytest.mark.parametrize("turn", [-1, 1])
def test_same_turn_circles_at_distance_two_r_keep_external_tangent(turn: int) -> None:
    result = compute_extreme_poses_arc_line(
        xc1=0.0,
        yc1=0.0,
        xc2=2.0,
        yc2=0.0,
        turn1=turn,
        turn2=turn,
        R=1.0,
    )

    x1, y1, theta1, x2, y2, theta2 = result
    assert abs(x2 - x1) == pytest.approx(2.0)
    assert y1 == pytest.approx(-turn)
    assert y2 == pytest.approx(-turn)
    assert theta1 == pytest.approx(0.0)
    assert theta2 == pytest.approx(theta1)


def test_opposite_turn_separated_circles_keep_nonzero_internal_tangent() -> None:
    result = compute_extreme_poses_arc_line(
        xc1=0.0,
        yc1=0.0,
        xc2=2.1,
        yc2=0.0,
        turn1=1,
        turn2=-1,
        R=1.0,
    )

    x1, y1, _, x2, y2, _ = result
    assert (x2 - x1) ** 2 + (y2 - y1) ** 2 > 0.0


def test_opposite_turn_overlapping_circles_use_overlap_contact() -> None:
    result = compute_extreme_poses_arc_line(
        xc1=0.0,
        yc1=0.0,
        xc2=1.9,
        yc2=0.0,
        turn1=1,
        turn2=-1,
        R=1.0,
        overlap=True,
    )

    x1, y1, theta1, x2, y2, theta2 = result
    assert (x2, y2) == pytest.approx((x1, y1))
    assert theta1 == pytest.approx(pi / 2)
    assert theta2 == pytest.approx(theta1)
