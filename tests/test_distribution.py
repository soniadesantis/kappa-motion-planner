from importlib.resources import files

import pytest

from kappa_planner import Bicycle, Unicycle


@pytest.mark.parametrize(
    ("relative_path", "model_name", "vehicle_type", "expected_width"),
    [
        ("vehicle_library/unicycle_library.yaml", "Rosbot circular", Unicycle, 0.237),
        ("vehicle_library/bicycle_library.yaml", "Bicycle standard", Bicycle, 0.43),
    ],
)
def test_packaged_vehicle_model_library(
    relative_path, model_name, vehicle_type, expected_width
):
    assert files("kappa_planner").joinpath(relative_path).is_file()
    assert vehicle_type(model=model_name).width == pytest.approx(expected_width)
