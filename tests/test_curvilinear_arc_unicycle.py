# ruff: noqa: ANN001, ANN201, CPY001, D103, S101
"""Tests for geometrically consistent unicycle arc headings."""

from math import pi

import numpy as np
import pytest

from kappa_planner import CurvilinearArcUnicycle, Unicycle


def make_arc(*, end_position, theta0, thetaf, turn_direction):
    vehicle = Unicycle(v_max=1.0, omega_max=1.0)
    return CurvilinearArcUnicycle(
        xc=0.0,
        yc=0.0,
        x0=1.0,
        y0=0.0,
        theta0=theta0,
        xf=end_position[0],
        yf=end_position[1],
        thetaf=thetaf,
        radius=1.0,
        turn_direction=turn_direction,
        v=1.0,
        omega=turn_direction,
        unicycle=vehicle,
    )


@pytest.mark.parametrize(
    ("end_position", "theta0", "thetaf", "turn_direction", "expected_iota"),
    [
        ((0.0, 1.0), pi / 2, -pi, 1, pi / 2),
        ((0.0, -1.0), -pi / 2, pi, -1, pi / 2),
        ((0.0, -1.0), pi / 2, 0.0, 1, 3 * pi / 2),
        ((0.0, 1.0), -pi / 2, 0.0, -1, 3 * pi / 2),
    ],
)
def test_heading_follows_geometric_arc(
    end_position,
    theta0,
    thetaf,
    turn_direction,
    expected_iota,
):
    arc = make_arc(
        end_position=end_position,
        theta0=theta0,
        thetaf=thetaf,
        turn_direction=turn_direction,
    )

    assert arc.iota == pytest.approx(expected_iota)
    assert arc.heading_displacement == pytest.approx(turn_direction * expected_iota)
    assert arc.thetaf == pytest.approx(theta0 + turn_direction * expected_iota)
    assert arc.theta_trajectory[-1] == pytest.approx(arc.thetaf)


def test_equivalent_headings_do_not_create_a_full_rotation():
    arc = make_arc(
        end_position=(1.0, 0.0),
        theta0=0.0,
        thetaf=2 * pi,
        turn_direction=1,
    )

    assert arc.iota == pytest.approx(0.0)
    assert arc.heading_displacement == pytest.approx(0.0)
    assert np.allclose(arc.theta_trajectory, 0.0)


def test_change_theta0_preserves_arc_heading_displacement():
    arc = make_arc(
        end_position=(0.0, -1.0),
        theta0=pi / 2,
        thetaf=0.0,
        turn_direction=1,
    )
    original_displacement = arc.heading_displacement

    arc.change_theta0(-pi / 3)

    assert arc.heading_displacement == pytest.approx(original_displacement)
    assert arc.thetaf == pytest.approx(-pi / 3 + original_displacement)
    assert arc.end_pose[2] == pytest.approx(arc.thetaf)
    assert arc.theta_trajectory[-1] == pytest.approx(arc.thetaf)
