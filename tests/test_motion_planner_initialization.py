# ruff: noqa: CPY001, S101
"""Regression tests for public MotionPlanner construction."""

from kappa_planner import CorridorWorld, MotionPlanner, Unicycle


def test_core_planner_initializes_warning_messages() -> None:
    corridors = [
        CorridorWorld(width=3.0, height=7.2, center=[0.0, 3.0], tilt=1.570796),
        CorridorWorld(width=3.0, height=4.8, center=[1.73, 7.0], tilt=0.523598),
    ]
    planner = MotionPlanner(
        Unicycle(width=0.43, length=0.43),
        corridors,
        start_pose=[0.6425, -0.23, 0.0],
        end_pose=[3.0, 8.3, 2.88],
    )

    assert isinstance(planner.warn_msgs, list)
