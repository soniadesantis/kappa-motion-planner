"""Public planner routing, optional endpoints, and complete-baseline fallback."""
from math import pi
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kappa_planner import Bicycle, CorridorWorld, MotionPlanner, Unicycle
from kappa_planner.helpers.baseline_construction import compute_trajectory_traversal_time
from kappa_planner.helpers.poses import compute_axis_aligned_boundary_pose
from kappa_planner.trajectory import CurvilinearArcBicycle, LinearSegmentBicycle, BackwardArcBicycle


def scene():
    bicycle = Bicycle(width=1, length=1, wheelbase=1, delta_max=pi/4, delta_min=-pi/4)
    corridors = [CorridorWorld(4, 12, [0, 0], 0),
                 CorridorWorld(4, 12, [4, 4], pi/2),
                 CorridorWorld(4, 12, [8, 8], 0)]
    return bicycle, corridors


def assert_complete(trajectory, planner):
    np.testing.assert_allclose([trajectory[0].x0, trajectory[0].y0], planner.start_pose[:2])
    np.testing.assert_allclose([trajectory[-1].xf, trajectory[-1].yf], planner.end_pose[:2])
    assert np.isclose(np.exp(1j*trajectory[0].theta0), np.exp(1j*planner.start_pose[2]))
    assert np.isclose(np.exp(1j*trajectory[-1].thetaf), np.exp(1j*planner.end_pose[2]))
    for first, second in zip(trajectory, trajectory[1:]):
        np.testing.assert_allclose([first.xf, first.yf, first.tf], [second.x0, second.y0, second.t0], atol=1e-7)
    assert planner.comp_time_analytical_sol > 0
    assert planner.traversal_time == compute_trajectory_traversal_time(trajectory)


def test_axis_aligned_uses_own_validation_and_returns_bicycle_primitives():
    bicycle, corridors = scene()
    with patch('kappa_planner.motion_planner.check_core_assumptions', side_effect=AssertionError), \
         patch('kappa_planner.motion_planner.check_standing_assumptions', side_effect=AssertionError):
        planner = MotionPlanner(bicycle, corridors, assumptions='axis-aligned')
        trajectory = planner.compute_trajectory_analytical()
    assert planner.solution_source == 'refined'
    assert all(isinstance(item, (CurvilinearArcBicycle, LinearSegmentBicycle, BackwardArcBicycle)) for item in trajectory)
    assert_complete(trajectory, planner)


@pytest.mark.parametrize('error', [False, True])
def test_failed_refinement_returns_complete_baseline(error):
    bicycle, corridors = scene()
    planner = MotionPlanner(bicycle, corridors, assumptions='axis-aligned')
    options = dict(side_effect=ValueError('refinement geometry failed')) if error else dict(
        return_value=(None, SimpleNamespace(reason='no_tangent_connection_found')))
    with patch('kappa_planner.motion_planner.refine_bicycle_baseline', **options):
        trajectory = planner.compute_trajectory_analytical()
    assert trajectory is planner.baseline.trajectory
    assert planner.solution_source == 'baseline'
    assert planner.refinement_failure_reason
    assert_complete(trajectory, planner)


def test_incomplete_baseline_and_refinement_raise_instead_of_returning_partial_path():
    bicycle, corridors = scene()
    planner = MotionPlanner(bicycle, corridors, assumptions='axis-aligned')
    with patch('kappa_planner.motion_planner.compute_baseline_boundary_connections', return_value=(None, None)), \
         patch('kappa_planner.motion_planner.refine_bicycle_baseline', return_value=(None, SimpleNamespace(reason='boundary_connection_failed'))):
        with pytest.raises(ValueError, match='No complete boundary-connected trajectory'):
            planner.compute_trajectory_analytical()
    assert planner.solution_source is None


@pytest.mark.parametrize('rotation', [0, pi/2, pi, 3*pi/2])
def test_default_endpoints_follow_overlap_direction_in_every_cardinal_frame(rotation):
    bicycle, corridors = scene()
    transform = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    rotated = [CorridorWorld(c.width, c.height, transform @ c.center, c.tilt+rotation+pi) for c in corridors]
    planner = MotionPlanner(bicycle, rotated, assumptions='axis-aligned')
    np.testing.assert_allclose(planner.start_pose[:2], transform @ [-5, 0], atol=1e-9)
    np.testing.assert_allclose(planner.end_pose[:2], transform @ [13, 8], atol=1e-9)
    assert_complete(planner.compute_trajectory_analytical(), planner)


def test_ambiguous_default_direction_requires_explicit_pose():
    bicycle, _ = scene()
    with pytest.raises(ValueError, match='explicit boundary pose'):
        compute_axis_aligned_boundary_pose(CorridorWorld(4, 12, [0, 0], 0),
                                          CorridorWorld(4, 12, [0, 0], pi/2), bicycle)


def test_rejects_unknown_selection_unsupported_models_and_rotated_corridors():
    bicycle, corridors = scene()
    with pytest.raises(ValueError, match='assumptions'):
        MotionPlanner(bicycle, corridors, assumptions='core')
    with pytest.raises(NotImplementedError, match='Unicycle'):
        MotionPlanner(bicycle, corridors)
    with pytest.raises(NotImplementedError, match='Bicycle'):
        MotionPlanner(Unicycle(), corridors, assumptions='axis-aligned')
    corridors[0] = CorridorWorld(4, 12, [0, 0], .1)
    with pytest.raises(ValueError, match='axis-aligned rectangles'):
        MotionPlanner(bicycle, corridors, assumptions='axis-aligned')


def test_invalid_disk_pose_cannot_be_planned():
    bicycle, corridors = scene()
    with pytest.warns(UserWarning, match='footprint'):
        planner = MotionPlanner(bicycle, corridors, start_pose=[-6, 0, 0], assumptions='axis-aligned')
    with pytest.raises(ValueError, match='footprint'):
        planner.compute_trajectory_analytical()


def test_update_preserves_explicit_poses_and_recomputes_defaults():
    bicycle, corridors = scene()
    planner = MotionPlanner(bicycle, corridors, start_pose=[-4, 0, 0], assumptions='axis-aligned')
    planner.compute_trajectory_analytical()
    replacement = [corridors[0], corridors[1], CorridorWorld(4, 16, [10, 8], 0)]
    planner.update(corridor_list=replacement, vehicle=bicycle.copy())
    np.testing.assert_allclose(planner.start_pose, [-4, 0, 0])
    assert planner.end_pose[0] == 17
    assert planner.solution_source is None
    assert_complete(planner.compute_trajectory_analytical(), planner)
    with pytest.raises(NotImplementedError):
        planner.update(assumptions='standing')
    assert planner.assumptions == 'axis-aligned'


def test_relative_pose_survives_vehicle_update():
    bicycle, corridors = scene()
    planner = MotionPlanner(bicycle, corridors, relative_start_pose=[0, -.8, pi/2],
                            relative_end_pose=[0, .8, pi/2], assumptions='axis-aligned')
    poses = np.array([planner.start_pose, planner.end_pose])
    planner.update(vehicle=bicycle.copy())
    np.testing.assert_allclose([planner.start_pose, planner.end_pose], poses)
    assert_complete(planner.compute_trajectory_analytical(), planner)
