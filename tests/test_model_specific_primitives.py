"""Model identity and bicycle controls in analytical motion primitives."""

from math import atan, pi
from types import SimpleNamespace

import numpy as np
import pytest

from kappa_planner import (
    BackwardArc, BackwardArcBicycle, Bicycle, CurvilinearArcBicycle,
    CurvilinearArcUnicycle, LinearSegmentBicycle, LinearSegmentUnicycle,
    TurnOnTheSpot, TurnOnTheSpotUnicycle, Unicycle,
)
from kappa_planner.helpers.primitives import (
    compute_arc_from_two_tangents_objects, compute_segment_between_two_circles_objects,
    correct_angles, invert_maneuvers,
)


def bicycle():
    return Bicycle(wheelbase=1, v_max=1, delta_max=atan(.5), delta_min=-atan(.5))


def arc(turn=1, vehicle=None):
    return CurvilinearArcBicycle(
        xc=0, yc=0, x0=2, y0=0, theta0=turn*pi/2,
        xf=0, yf=turn*2, thetaf=turn*pi, radius=2, turn_direction=turn,
        v=1, omega=turn*.5, bicycle=vehicle or bicycle(),
    )


@pytest.mark.parametrize("turn", [-1, 1])
def test_bicycle_arc_has_separate_model_identity_and_correct_controls(turn):
    maneuver = arc(turn)
    assert not isinstance(maneuver, CurvilinearArcUnicycle)
    assert not hasattr(maneuver, "unicycle")
    np.testing.assert_allclose(maneuver.path_coordinates[[0, -1]], [[2, 0], [0, turn*2]], atol=1e-12)
    assert maneuver.maneuver_time == pytest.approx(pi)
    assert maneuver.delta == pytest.approx(turn*atan(.5))
    np.testing.assert_allclose(
        maneuver.angular_velocity,
        maneuver.forward_velocity * np.tan(maneuver.steering_angle) / maneuver.bicycle.wheelbase,
    )
    maneuver.resample(17)
    assert len(maneuver.steering_angle) == len(maneuver.time_grid) == 17


def test_bicycle_segment_stays_straight_after_resampling_and_reversal():
    maneuver = LinearSegmentBicycle(0, 0, 3, 0, 0, 1, bicycle(), t0=2)
    assert not isinstance(maneuver, LinearSegmentUnicycle)
    assert not hasattr(maneuver, "unicycle")
    assert maneuver.tf == pytest.approx(5)
    maneuver.reverse()
    maneuver.resample(19)
    np.testing.assert_allclose(maneuver.steering_angle, np.zeros(19))
    np.testing.assert_allclose(maneuver.forward_velocity, -1)


def test_model_specific_primitives_reject_the_other_vehicle_model():
    with pytest.raises(TypeError):
        LinearSegmentUnicycle(0, 0, 1, 0, 0, 1, bicycle())
    with pytest.raises(TypeError):
        LinearSegmentBicycle(0, 0, 1, 0, 0, 1, Unicycle())
    with pytest.raises(TypeError):
        arc(vehicle=Unicycle())
    with pytest.raises(TypeError):
        TurnOnTheSpotUnicycle(0, 0, 0, pi/2, 1, bicycle())


def test_backward_arc_resampling_preserves_its_endpoints_and_steering():
    maneuver = BackwardArcBicycle(
        xc=0, yc=0, x0=2, y0=0, theta0=pi/2, xf=0, yf=-2, thetaf=0,
        radius=2, turn_direction=1, v=-1, omega=-.5, bicycle=bicycle(),
    )
    maneuver.resample(31)
    np.testing.assert_allclose(maneuver.path_coordinates[[0, -1]], [[2, 0], [0, -2]], atol=1e-12)
    np.testing.assert_allclose(maneuver.steering_angle, atan(.5))


@pytest.mark.parametrize("vehicle, segment_class, arc_class", [
    (Unicycle(v_max=1, omega_max=.5), LinearSegmentUnicycle, CurvilinearArcUnicycle),
    (bicycle(), LinearSegmentBicycle, CurvilinearArcBicycle),
])
def test_shared_construction_helpers_preserve_vehicle_model(vehicle, segment_class, arc_class):
    first = SimpleNamespace(xc=0, yc=0, turn_direction=1)
    second = SimpleNamespace(xc=4, yc=0, turn_direction=1)
    segment = compute_segment_between_two_circles_objects(first, second, vehicle)
    assert type(segment) is segment_class
    before = SimpleNamespace(end_position=[2, 0], thetaf=pi/2, tf=0)
    after = SimpleNamespace(start_position=[0, 2], theta0=pi)
    maneuver = compute_arc_from_two_tangents_objects(before, after, first, vehicle)
    assert type(maneuver) is arc_class


def test_inversion_and_angle_correction_preserve_bicycle_types_and_timing():
    circular = arc()
    straight = LinearSegmentBicycle(0, 2, -2, 2, pi, 1, circular.bicycle, t0=circular.tf)
    correct_angles([circular, straight])
    inverted = invert_maneuvers([circular, straight], t0=5)
    assert type(inverted[0]) is LinearSegmentBicycle
    assert type(inverted[1]) is CurvilinearArcBicycle
    assert inverted[0].t0 == 5
    assert inverted[1].t0 == pytest.approx(inverted[0].tf)
    np.testing.assert_allclose(inverted[0].end_position, inverted[1].start_position, atol=1e-12)
    assert inverted[0].thetaf == pytest.approx(inverted[1].theta0)


def test_renamed_primitives_keep_compatibility_imports():
    assert TurnOnTheSpot is TurnOnTheSpotUnicycle
    assert BackwardArc is BackwardArcBicycle
