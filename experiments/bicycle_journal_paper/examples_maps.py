from __future__ import annotations

import math as m
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pylab as plt

from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Bicycle, Unicycle
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import (
    plot_corridors, 
    plot_analytical_trajectory,
    plot_circular_footprint,
    plot_rectangular_footprint,
    plot_turning_front_corner_path,
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

    elif num == 3: # Rule on merging two overlapping circles with same turn direction
        corridor1 = CorridorWorld(5, 10, [0, 0], 0)
        corridor2 = CorridorWorld(3, 6, [3, 3], 1.5707963267948966)
        corridor3 = CorridorWorld(5, 10, [0, 5.5], 3.142592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)

    elif num == 4: # Rule on merging two overlapping circles with same turn direction
        corridor1 = CorridorWorld(5, 10, [0, 0], 0)
        corridor2 = CorridorWorld(0.6, 6, [3, 3], 1.5707963267948966)
        corridor3 = CorridorWorld(5, 10, [5, 5.1], 0)
        corridor_list = [corridor1, corridor2, corridor3]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.1, v_max=1.0, v_min=-1.0, delta_max=0.1, delta_min=-0.5)

        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)


    elif num == 5: 
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor6 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor7 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor8 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor9 = CorridorWorld(0.5399999879300594, 4.3199999034404755, [4.1999999061226845, 6.689999850466847], 0.0)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9]
        start_pose = [2.61918306350708, 2.154087543487549, -0.12029518960373457]
        end_pose = [4.050821781158447, 6.627957344055176, 0.07130739522438935]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 6:

        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor6 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor7 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor8 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor9 = CorridorWorld(0.5399999879300594, 4.3199999034404755, [4.1999999061226845, 6.689999850466847], 0.0)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9]
        start_pose = [2.5406384468078613, 2.26594877243042, -0.0388158088104548]
        end_pose = [5.087650299072266, 6.557488441467285, 0.15702971301306384]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 7: # Example 1
        corridor1 = CorridorWorld(1.0999999754130843, 2.699999939650297, [5.839999869465828, 2.509999943897128], 1.5707963267948966)
        corridor2 = CorridorWorld(0.5399999879300587, 6.149999862536788, [3.314999925903976, 3.519999921321869], 3.141592653589793)
        corridor3 = CorridorWorld(0.4499999899417165, 5.069999886676669, [4.944999889470637, 3.694999917410314], 1.5707963267948966)
        corridor4 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor5 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor6 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor7 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor8 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor9 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor10 = CorridorWorld(0.749999983236194, 4.619999896734953, [2.4349999455735087, 8.149999817833304], 1.5707963267948966)
        corridor11 = CorridorWorld(0.709999984130263, 2.699999939650297, [1.5899999644607306, 10.10499977413565], 3.141592653589793)
        corridor12 = CorridorWorld(0.4399999901652338, 2.5299999434500933, [1.5599999651312828, 11.014999753795564], 1.5707963267948966)
        corridor13 = CorridorWorld(0.4299999903887507, 2.699999939650297, [1.5899999644607306, 11.34499974641949], 3.141592653589793)
        corridor14 = CorridorWorld(0.6899999845772983, 2.5299999434500933, [0.584999986924231, 11.014999753795564], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9, corridor10, corridor11, corridor12, corridor13, corridor14]

        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)

    elif num == 8: # Example 2 To Be Solved
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor4 = CorridorWorld(1.0999999754130838, 1.6999999620020392, [0.4399999745190144, 1.1299999479204417], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3, corridor4]
        start_pose = [-0.1127556711435318, -0.45102250576019287, 0.0]
        end_pose = [0.47704362869262695, 0.7632699012756348, -3.0466413835047237]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 9: # Example 3 
        corridor1 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor2 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor3 = CorridorWorld(1.0999999754130838, 1.6999999620020392, [0.4399999745190144, 1.1299999479204417], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [2.0646212100982666, -0.5284115076065063, 1.8878378130555056]
        end_pose = [0.40652525424957275, 1.492927074432373, 2.9939433356042744]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
        vehicle = Unicycle(model = 'Rosbot circular')
        # Modify vehicle parameters
        vehicle.update(v_max = 0.5)
        vehicle.update(omega_max = 1)

    elif num == 10: # Example 4 To Be Solved
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.49999998882412877, 2.849999936297536, [1.0149999616667629, 2.6099999148398636], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [0.1301027536392212, -0.45102250576019287, 0.05963050567379781]
        start_pose = [1.399, 0.268, 0.05963050567379781]
        end_pose = [0.45969676971435547, 2.532668352127075, 2.9889436231146522]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 11: 
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [0.1301027536392212, -0.45102250576019287, 0.05963050567379781]
        end_pose = [0.7719430327415466, 1.1622517108917236, 3.141592653589793]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)
        
    elif num == 12: 
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [-0.17406433820724487, 0.22661757469177246, -1.239699812246792]
        end_pose = [1.1206774711608887, 0.885758638381958, -2.598403358706264]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 13:
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [-0.15052366256713867, -0.08725953102111816, -0.04626290891359498]
        end_pose = [0.7440255880355835, 1.1368603706359863, -3.1246445435086296]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 14:
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.49999998882412877, 2.849999936297536, [1.0149999616667629, 2.6099999148398636], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [0.022108793258666992, -0.09510636329650879, -0.03224664555597171]
        end_pose = [0.761948823928833, 2.6914305686950684, 2.771099374037871]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 15: # Case with overlapping circles opposite taus
        corridor1 = CorridorWorld(2.47999994456768, 5.129999885335565, [9.199999794363976, 10.404999767430127], -1.5707963267948966)
        corridor2 = CorridorWorld(0.37, 7.58999983035028, [9.134999795816839, 9.869999779388309], 3.141592653589793)
        corridor3 = CorridorWorld(2.47999994456768, 7.639999829232693, [6.599999852478504, 9.149999795481563], -1.5707963267948966)
        corridor4 = CorridorWorld(1.0199999772012234, 5.099999886006117, [7.909999823197722, 5.979999866336584], 0.0)
        corridor5 = CorridorWorld(1.0099999774247408, 4.969999888911843, [10.444999766536057, 5.844999869354069], 0.0)
        corridor6 = CorridorWorld(2.369999947026372, 2.4799999445676804, [11.744999737478793, 6.479999855160713], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6]
        start_pose = [9.173179626464844, 11.993743896484375, -1.7382290420511535]
        end_pose = [11.442094802856445, 7.074225902557373, -0.612152762864258]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 16:
        corridor1 = CorridorWorld(6.989999843761325, 4.899999890476465, [30.694999313913286, 11.069999752566218], -1.5707963267948966)
        corridor2 = CorridorWorld(1.0299999769777053, 10.02999977581203, [32.60499927122146, 8.504999809898436], -1.5707963267948966)
        corridor3 = CorridorWorld(4.979999888688326, 2.969999933615327, [32.67499926965684, 5.979999866336584], 0.0)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [30.759105682373047, 12.061663627624512, 3.141592653589793]
        end_pose = [33.177406311035156, 5.57621955871582, -1.158386219431387]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)


    return corridor_list, start_pose, end_pose, vehicle


if __name__ == "__main__":
    example_num = 16

    corridor_list, start_pose, end_pose, vehicle = example_corridor_sequence(example_num)

    figure = plot_corridors(corridor_list, plot_vectors=True)
    # plt.show(block = True)
    # for i in range(len(corridor_list)-1):
    #     plot_corridors([corridor_list[i], corridor_list[i+1]])

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
    figure = mp.plot_planner_inputs()
    plt.show(block = True)

    # mp.plot_planner_inputs()
    # plt.show(block = True)

    ### Compute analytical trajectory ###
    analytical_trajectory = mp.compute_trajectory_analytical()
    print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")
    ### Plot results ###
    figure = mp.plot_planner_inputs()
    plt.title('Analytical Motion Planner - Unicycle Within Multiple Corridors')

    plot_analytical_trajectory(
        analytical_trajectory,
        figure=plt.gca(),
        plot_circles=True
    )

    ax = plt.gca()
    # plot_rectangular_footprint(
    #     analytical_trajectory,
    #     width=vehicle.width,
    #     front_overhang=0.2,
    #     rear_overhang=0.1,
    #     ax=ax,
    #     step=5,
    #     color="k",
    #     alpha=1,
    # )

    # plot_turning_front_corner_path(
    #     analytical_trajectory,
    #     width=vehicle.width,
    #     front_overhang=0.2,
    #     ax=ax,
    #     color="r",
    #     linewidth=2,
    # )

    plot_circular_footprint(
    analytical_trajectory,
    vehicle.width/2,
    ax=ax,
    step=1,
    color="k",
    linewidth=0.5,
    linestyle="-",
    alpha=0.2,
    )
    
    plt.show(block = True)

