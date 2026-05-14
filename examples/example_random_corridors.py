from __future__ import annotations

import math as m
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pylab as plt

from kappa_planner.vehicle import Bicycle
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import (
    plot_corridors, 
    plot_analytical_trajectory,
)
from kappa_planner.helpers.intermediate_circles_choice import (
    create_intermediate_circle_choice_sequence,
    plot_intermediate_circle_choices,
)


# ============================================================
# Data containers
# ============================================================

@dataclass
class CorridorSpec:
    index: int
    tail: Tuple[float, float]
    head: Tuple[float, float]
    width: float
    length: float
    theta: float
    delta_theta: Optional[float]
    add_height: float
    start_rel_x_norm: Optional[float] = None
    start_rel_y_norm: Optional[float] = None
    start_rel_x: Optional[float] = None
    start_rel_y: Optional[float] = None


# ============================================================
# Sampling helpers
# ============================================================

def sample_uniform(value_range: Sequence[float], rng: random.Random) -> float:
    vmin, vmax = value_range
    return rng.uniform(vmin, vmax)


def sample_from_choices(choices: Sequence[float], rng: random.Random) -> float:
    if not choices:
        raise ValueError("choices must not be empty")
    return rng.choice(list(choices))


def wrap_to_pi(angle: float) -> float:
    return (angle + m.pi) % (2 * m.pi) - m.pi


def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(value, vmax))


# ============================================================
# Serialization helper
# ============================================================

def corridor_spec_to_dict(spec: CorridorSpec) -> Dict[str, Any]:
    d = asdict(spec)
    d["tail"] = list(d["tail"])
    d["head"] = list(d["head"])
    return d


# ============================================================
# Geometry helpers
# ============================================================

def get_corridor_axis_and_lateral_vectors(corridor: Any) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dx = corridor.head[0] - corridor.tail[0]
    dy = corridor.head[1] - corridor.tail[1]
    length = m.hypot(dx, dy)

    if length <= 1e-12:
        raise ValueError("Corridor has near-zero length")

    axis = (dx / length, dy / length)
    lateral = (axis[1], -axis[0])  # right side

    return axis, lateral


def normalized_to_local_coordinates(
    corridor: Any,
    rel_x_norm: float,
    rel_y_norm: float,
) -> Tuple[float, float]:
    rel_x_norm = clamp(rel_x_norm, -1.0, 1.0)
    rel_y_norm = clamp(rel_y_norm, -1.0, 1.0)

    rel_x = 0.5 * corridor.width * rel_x_norm
    rel_y = 0.5 * corridor.height * rel_y_norm

    return rel_x, rel_y


def local_xy_to_absolute_point(
    corridor: Any,
    rel_x: float,
    rel_y: float,
) -> Tuple[float, float]:
    axis, lateral = get_corridor_axis_and_lateral_vectors(corridor)

    px = corridor.center[0] + rel_x * lateral[0] + rel_y * axis[0]
    py = corridor.center[1] + rel_x * lateral[1] + rel_y * axis[1]

    return (px, py)


def normalized_to_absolute_point(
    corridor: Any,
    rel_x_norm: float,
    rel_y_norm: float,
) -> Tuple[float, float]:
    rel_x, rel_y = normalized_to_local_coordinates(corridor, rel_x_norm, rel_y_norm)
    return local_xy_to_absolute_point(corridor, rel_x, rel_y)


# ============================================================
# Corridor generation
# ============================================================

def generate_corridor_sequence_with_discrete_turns(
    n_corridors: int,
    width_range: Sequence[float],
    length_range: Sequence[float],
    theta0_range: Sequence[float],
    delta_theta_choices: Sequence[float],
    relative_x_norm_range: Sequence[float],
    relative_y_norm_range: Sequence[float],
    add_height: float = 0.0,
    start_tail: Tuple[float, float] = (0.0, 0.0),
    rng_seed: Optional[int] = None,
    wrap_angles: bool = False,
) -> Tuple[List[Any], List[CorridorSpec]]:
    """
    Generate a sequence of corridors where each new corridor orientation differs
    from the previous one by one value sampled from delta_theta_choices.

    Example:
        delta_theta_choices = [-m.pi/2, 0.0, m.pi/2]
    """
    if n_corridors < 1:
        raise ValueError("n_corridors must be at least 1")

    rng = random.Random(rng_seed)

    corridors: List[Any] = []
    specs: List[CorridorSpec] = []

    # --------------------------------------------------------
    # First corridor
    # --------------------------------------------------------
    theta = sample_uniform(theta0_range, rng)
    width = sample_uniform(width_range, rng)
    length = sample_uniform(length_range, rng)

    tail = start_tail
    head = (
        tail[0] + length * m.cos(theta),
        tail[1] + length * m.sin(theta),
    )

    first_corridor = get_corridor_from_vector(
        tail,
        head,
        width,
        add_height=add_height,
    )

    corridors.append(first_corridor)
    specs.append(
        CorridorSpec(
            index=0,
            tail=tail,
            head=head,
            width=width,
            length=length,
            theta=theta,
            delta_theta=None,
            add_height=add_height,
        )
    )

    current_corridor = first_corridor
    current_theta = theta

    # --------------------------------------------------------
    # Remaining corridors
    # --------------------------------------------------------
    for i in range(1, n_corridors):
        new_width = sample_uniform(width_range, rng)
        new_length = sample_uniform(length_range, rng)

        # Only allow -pi/2, 0, +pi/2
        delta_theta = sample_from_choices(delta_theta_choices, rng)

        new_theta = current_theta + delta_theta
        if wrap_angles:
            new_theta = wrap_to_pi(new_theta)

        rel_x_norm = clamp(sample_uniform(relative_x_norm_range, rng), -1.0, 1.0)
        rel_y_norm = clamp(sample_uniform(relative_y_norm_range, rng), -1.0, 1.0)

        rel_x, rel_y = normalized_to_local_coordinates(
            current_corridor,
            rel_x_norm,
            rel_y_norm,
        )

        new_tail = local_xy_to_absolute_point(current_corridor, rel_x, rel_y)

        new_head = (
            new_tail[0] + new_length * m.cos(new_theta),
            new_tail[1] + new_length * m.sin(new_theta),
        )

        new_corridor = get_corridor_from_vector(
            new_tail,
            new_head,
            new_width,
            add_height=add_height,
        )

        corridors.append(new_corridor)
        specs.append(
            CorridorSpec(
                index=i,
                tail=new_tail,
                head=new_head,
                width=new_width,
                length=new_length,
                theta=new_theta,
                delta_theta=delta_theta,
                add_height=add_height,
                start_rel_x_norm=rel_x_norm,
                start_rel_y_norm=rel_y_norm,
                start_rel_x=rel_x,
                start_rel_y=rel_y,
            )
        )

        current_corridor = new_corridor
        current_theta = new_theta

    return corridors, specs


# ============================================================
# Optional plotting helper
# ============================================================

def plot_corridor_sequence_with_start_points(
    corridors: List[Any],
    specs: List[CorridorSpec],
    show_centers: bool = False,
) -> None:
    plot_corridors(corridors)

    start_points_x = []
    start_points_y = []

    for spec in specs[1:]:
        start_points_x.append(spec.tail[0])
        start_points_y.append(spec.tail[1])

    if start_points_x:
        plt.plot(start_points_x, start_points_y, "ro", label="next corridor start points")

    if show_centers:
        cx = [c.center[0] for c in corridors]
        cy = [c.center[1] for c in corridors]
        plt.plot(cx, cy, "kx", label="corridor centers")

    plt.legend()
    plt.title("Generated corridor sequence")
    plt.show()


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    generation_config = {
        "n_corridors": 15,
        "width_range": [1.5, 10],
        "length_range": [10.0, 20.0],
        "theta0_range": [m.pi / 2, m.pi / 2],

        # Only allow left turn, straight, right turn
        "delta_theta_choices": [-m.pi / 2, 0.0, m.pi / 2],

        "relative_x_norm_range": [-0.8, 0.8],
        "relative_y_norm_range": [0.8, 0.8],

        "add_height": 2.0,
        "start_tail": (0.0, 0.0),
        "rng_seed": 15,
        "wrap_angles": True,
    }

    corridors, specs = generate_corridor_sequence_with_discrete_turns(
        n_corridors=generation_config["n_corridors"],
        width_range=generation_config["width_range"],
        length_range=generation_config["length_range"],
        theta0_range=generation_config["theta0_range"],
        delta_theta_choices=generation_config["delta_theta_choices"],
        relative_x_norm_range=generation_config["relative_x_norm_range"],
        relative_y_norm_range=generation_config["relative_y_norm_range"],
        add_height=generation_config["add_height"],
        start_tail=generation_config["start_tail"],
        rng_seed=generation_config["rng_seed"],
        wrap_angles=generation_config["wrap_angles"],
    )

    # corridors.pop(10)
    plot_corridor_sequence_with_start_points(corridors, specs, show_centers=False)

    vehicle_width = 0.430
    vehicle_length = 0.508
    vehicle_wheelbase = 1
    vehicle_vmax = 2
    vehicle_deltamax = 0.5

    bicycle = Bicycle(
        [0, 0, 0],
        width=vehicle_width,
        length=vehicle_length,
        wheelbase=vehicle_wheelbase,
        v_max=vehicle_vmax,
        v_min=-vehicle_vmax,
        delta_max=vehicle_deltamax,
        delta_min=-vehicle_deltamax,
    )

    start_pose = compute_start_pose(corridors[0], bicycle, 0)
    end_pose = compute_end_pose(corridors[-1], bicycle, 0)
    figure = plot_corridors(corridors)
    ax = plt.gca()

    circle_choices_sequence = create_intermediate_circle_choice_sequence(
        corridors,
        bicycle,
        start_pose,
        end_pose,
    )

    plot_intermediate_circle_choices(
        ax,
        circle_choices_sequence,
        bicycle.width/2,
    )

    ### Define Motion Planner ###
    mp = MotionPlanner(bicycle, corridors)

    ### Compute analytical trajectory ###
    analytical_trajectory = mp.compute_trajectory_analytical()
    print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")
    ### Plot results ###
    figure = mp.plot_planner_inputs()
    plt.title('Analytical Motion Planner - Unicycle Within Multiple Corridors')
    plot_analytical_trajectory(analytical_trajectory, figure = figure)

    plt.show()