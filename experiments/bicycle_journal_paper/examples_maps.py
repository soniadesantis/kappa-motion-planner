from __future__ import annotations

import math as m
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pylab as plt

from kappa_planner.corridor import CorridorWorld
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


def example_corridor_sequence(num):

    if num == 1: 
        corridor1 = CorridorWorld(1.5000000223517425, 12.000000178813934, [11.200000166893005, 6.550000097602606], 1.5707963267948966)
        corridor2 = CorridorWorld(3.500000052154064, 30.000000447034836, [15.45000023022294, 10.800000160932541], 0.0)
        corridor3 = CorridorWorld(1.5000000223517431, 20.000000298023224, [15.20000022649765, 10.55000015720725], 1.5707963267948966)
        corridor4 = CorridorWorld(1.5000000223517418, 30.000000447034836, [15.45000023022294, 13.800000205636024], 0.0)
        corridor5 = CorridorWorld(0.5000000074505818, 20.000000298023224, [16.700000248849392, 10.55000015720725], 1.5707963267948966)
        corridor6 = CorridorWorld(1.5000000223517418, 30.000000447034836, [15.45000023022294, 15.800000235438347], 0.0)
        corridor7 = CorridorWorld(1.5000000223517431, 20.000000298023224, [29.700000442564487, 10.55000015720725], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7]
        start_pose = [11.24559211730957, 2.833745002746582, -0.6947391079002443]
        end_pose = [29.358922958374023, 17.969539642333984, 0.887089182465026]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 2:

        corridor1 = CorridorWorld(4.969999888911843, 2.959999933838844, [8.644999806769192, 1.7399999611079693], 1.5707963267948966)
        corridor2 = CorridorWorld(1.7199999615550046, 8.069999819621444, [8.719999805092812, 4.294999903999269], 1.5707963267948966)
        corridor3 = CorridorWorld(1.0199999772012247, 23.449999475851655, [8.469999810680747, 11.984999732114375], 1.5707963267948966)
        corridor4 = CorridorWorld(2.1199999526143083, 15.239999659359455, [8.499999810010195, 16.089999640360475], 1.5707963267948966)
        corridor5 = CorridorWorld(1.029999976977705, 7.269999837502837, [11.044999753125012, 21.004999530501664], 3.141592653589793)
        corridor6 = CorridorWorld(1.0199999772012225, 7.27999983727932, [5.929999867454171, 21.319999523460865], 3.141592653589793)
        corridor7 = CorridorWorld(4.999999888241291, 4.979999888688326, [4.789999892935157, 21.199999526143074], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7]
        start_pose = [10.956110000610352, 1.4690418243408203, 2.699218939309123]
        end_pose = [5.497298240661621, 22.849388122558594, 0.0]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    return corridor_list, start_pose, end_pose, vehicle


if __name__ == "__main__":
    example_num = 2
    corridor_list, start_pose, end_pose, vehicle = example_corridor_sequence(example_num)

    figure = plot_corridors(corridor_list)
    for i in range(len(corridor_list)-1):
        plot_corridors([corridor_list[i], corridor_list[i+1]])

    ax = plt.gca()
    # plt.show(block = True)
    circle_choices_sequence = create_intermediate_circle_choice_sequence(
        corridor_list,
        vehicle,
        start_pose,
        end_pose,
    )

    plot_intermediate_circle_choices(
        ax,
        circle_choices_sequence,
        vehicle.width/2,
    )
    plt.show(block = True)


    ### Define Motion Planner ###
    mp = MotionPlanner(vehicle, corridor_list, start_pose, end_pose)

    ### Compute analytical trajectory ###
    analytical_trajectory = mp.compute_trajectory_analytical()
    print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")
    ### Plot results ###
    figure = mp.plot_planner_inputs()
    plt.title('Analytical Motion Planner - Unicycle Within Multiple Corridors')
    plot_analytical_trajectory(analytical_trajectory, figure = figure)

    plt.show(block = True)

