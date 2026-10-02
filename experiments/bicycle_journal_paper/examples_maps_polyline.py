"""Run the nominal internal orthogonal-polyline check on the example maps.

Run directly or with VS Code Run/Debug; choose EXAMPLE_NUM below. Figure 1 shows
union erosion, safe overlaps, and inferred unique H/V directions. Figure 2 uses
waypoints selected by fillet-aware midpoint backtracking, with a joint-solver
fallback when the bounded search cannot find a compatible sequence.
Fillets use the analytic admissible-region conditions. An independent full-arc
audit is optional through VALIDATE_BASELINE_ARCS.
Exact start/end positions are preferred; one-coordinate projections are used
when needed for feasibility. Pose headings are not enforced at this stage.
Figure 3 places legacy intermediate circles using the baseline polyline (bp)
directions and corners, then optionally shifts conflicting same-turn circles.
It does not certify a full connected trajectory or merge circle records.
The previous tangent refinement remains optional via PLOT_TANGENT_REFINEMENT.
"""

from __future__ import annotations

import math as m
from dataclasses import dataclass
from time import perf_counter
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.nominal_polyline import check_orthogonal_polyline
from kappa_planner.helpers.endpoint_polyline import extend_nominal_with_pose_coordinates
from kappa_planner.helpers.fillet_safety import (
    revised_corner_condition, fillet_vertex_region, region_margin, continuous_arc_clearance,
    axis_aligned_concave_corners,
)
from kappa_planner.helpers.smoothed_polyline import solve_fillet_waypoints
from kappa_planner.helpers.corridor_geometry import shrink_corridor_list
from kappa_planner.helpers.plot_helpers import plot_corridors
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose
from kappa_planner.vehicle import Bicycle, Unicycle


EXAMPLE_NUM = 1 # Choose any example from 1 through 35.
EXAMPLE_NUMBERS = range(1, 36)
FILLET_RADIUS = None  # Radius in metres; None uses vehicle.max_radius.
EROSION_PLOT_RESOLUTION = 900  # Grid samples along the longer plot dimension.
PLOT_TANGENT_REFINEMENT = False  # Optional previous refinement experiment.
PLOT_BP_CIRCLES = True  # Legacy placement from bp, with optional same-turn repair below.
SHIFT_SAME_TURN_CIRCLES = True  # Local repair; retain initial centers for comparison.
SHIFT_OPPOSITE_TURN_CIRCLES = True  # Separate opposite-turn circles within A_j.
CONNECT_BP_CIRCLES = True  # Greedy internal tangent chain; no endpoint maneuvers.
VALIDATE_BASELINE_ARCS = False  # Optional independent geometric audit; slower.
PLOT_SPACING_COUPLING = True  # Local 2R sensitivity of the original safe-door pairs.
SPACING_COLORS = {"coupled": "#d97706", "impossible": "#c62828"}


def plot_spacing_coupling(ax, nominal, points, door_offset=0, draw_segments=True):
    """Flag original overlap vertices, excluding any added pose endpoints.

    This depicts potential coupling over the original safe doors, not a
    violation by the selected solution or by its later refined fillet regions.
    """
    points = np.asarray(points)
    handles = []
    for status, color in SPACING_COLORS.items():
        pairs = [pair for pair in nominal.get("spacing_pairs", [])
                 if pair["status"] == status]
        vertices = set()
        for pair in pairs:
            indices = np.array(pair["overlaps"]) + door_offset
            vertices.update(indices)
            if draw_segments:
                ax.plot(*points[indices].T, color=color, ls="--", lw=2.3, zorder=4)
        if vertices:
            ax.scatter(*points[sorted(vertices)].T, s=125, facecolors="none",
                       edgecolors=color, linewidths=2, zorder=8)
            handles.append(Line2D([], [], color=color, ls="--", marker="o",
                                  markerfacecolor="none", markeredgewidth=2,
                                  label=("2R-coupled pair / vertices" if status == "coupled"
                                         else "No aligned pair can reach 2R")))
    return handles


def example_corridor_sequence(num):
    if num not in EXAMPLE_NUMBERS:
        raise ValueError("Example number must be between 1 and 35.")

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
        corridor_list = [corridor1, corridor3, corridor6, corridor7]
        start_pose = [10.956110000610352, 1.4690418243408203, 2.699218939309123]
        end_pose = [5.497298240661621, 22.849388122558594, 0.0]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
        # vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 3: # Rule on merging two overlapping circles with same turn direction
        corridor1 = CorridorWorld(5, 10, [0, 0], 0)
        corridor2 = CorridorWorld(3, 6, [3, 3], 1.5707963267948966)
        corridor3 = CorridorWorld(5, 10, [0, 5.5], m.pi)
        corridor_list = [corridor1, corridor2, corridor3]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)
        start_pose = [-4.25, 2.41, m.pi/2-0.2]
        start_pose = [-4.9, -0.18, m.pi]


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
        corridor_list = [corridor1, corridor2, corridor3, corridor6, corridor7, corridor8, corridor9]
        start_pose = [2.61918306350708, 2.154087543487549, -0.12029518960373457]
        end_pose = [4.050821781158447, 6.627957344055176, 0.07130739522438935]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.32, v_max=1.0, v_min=-1.0, delta_max=m.pi/4, delta_min=-0.5)

    elif num == 6:
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor6 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor7 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor8 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2, 7.74499982688576], 1.5707963267948966)
        corridor9 = CorridorWorld(0.5399999879300594, 4.3199999034404755, [4.1999999061226845, 6.689999850466847], 0.0)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9]
        start_pose = [2.5406384468078613, 2.26594877243042, -0.0388158088104548]
        end_pose = [5.087650299072266, 6.557488441467285, 0.15702971301306384]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.15, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

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
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor10, corridor11, corridor12, corridor13, corridor14]

        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.15, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
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
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [0.1301027536392212, -0.45102250576019287, 0.05963050567379781]
        end_pose = [0.7719430327415466, 1.1622517108917236, 3.141592653589793]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 11:
        corridor1 = CorridorWorld(1.459999967366457, 2.9999999329447746, [0.9399999633431435, -0.31000001989305015], 0.0)
        corridor2 = CorridorWorld(0.9999999776482585, 5.999999865889549, [1.9399999409914017, 1.959999929368496], 1.5707963267948966)
        corridor3 = CorridorWorld(0.5099999886006114, 2.849999936297536, [1.0149999616667629, 1.1249999480322004], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [-0.15052366256713867, -0.08725953102111816, -0.04626290891359498]
        end_pose = [0.7440255880355835, 1.1368603706359863, -3.1246445435086296]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)

    elif num == 12:
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

    elif num == 13:
        corridor1 = CorridorWorld(6.989999843761325, 4.899999890476465, [30.694999313913286, 11.069999752566218], -1.5707963267948966)
        corridor2 = CorridorWorld(1.0299999769777053, 10.02999977581203, [32.60499927122146, 8.504999809898436], -1.5707963267948966)
        corridor3 = CorridorWorld(4.979999888688326, 2.969999933615327, [32.67499926965684, 5.979999866336584], 0.0)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [30.759105682373047, 12.061663627624512, 3.141592653589793]
        end_pose = [33.177406311035156, 5.57621955871582, -1.158386219431387]
        vehicle = Unicycle(width=0.34, length=0.237, v_max=0.5, v_min=0, omega_max=2.0, omega_min=-2.0)
        vehicle = Bicycle([0, 0, 0], width=0.2, length=0.2, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 14: # Side-Side narrow corridors
        corridor1 = CorridorWorld(5, 10, [0, 0], 0)
        corridor2 = CorridorWorld(0.23, 6, [3, 3], 1.5707963267948966)
        corridor3 = CorridorWorld(3, 10, [0, 5.5], m.pi)
        corridor_list = [corridor1, corridor2, corridor3]
        vehicle = Bicycle([0, 0, 0], width=0.2, length=0.2, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)


    ### Examples for Alex!!
    ############
    # Large rooms, map_20.yaml
    elif num == 15:
        corridor1 = CorridorWorld(4.9899998884648085, 4.969999888911843, [2.734999938867986, 2.734999938867986], 1.5707963267948966)
        corridor2 = CorridorWorld(0.9899999778717756, 10.079999774694443, [5.279999881982803, 4.624999896623194], 0.0)
        corridor3 = CorridorWorld(4.979999888688326, 4.969999888911843, [7.829999824985862, 2.734999938867986], 1.5707963267948966)
        corridor4 = CorridorWorld(0.7499999832361943, 10.06999977491796, [5.834999869577587, 5.284999881871045], 1.5707963267948966)
        corridor5 = CorridorWorld(4.989999888464808, 4.979999888688327, [7.829999824985862, 7.8249998250976205], 3.141592653589793)
        corridor6 = CorridorWorld(1.7299999613314854, 10.079999774694443, [5.279999881982803, 9.304999792017043], 3.141592653589793)
        corridor7 = CorridorWorld(4.9899998884648085, 4.9899998884648085, [2.734999938867986, 7.8249998250976205], 1.5707963267948966)
        corridor8 = CorridorWorld(0.7499999832361943, 10.089999774470925, [0.7849999824538827, 10.374999768100679], 1.5707963267948966)
        corridor9 = CorridorWorld(4.9899998884648085, 4.969999888911843, [2.744999938644469, 12.924999711103737], 0.0)
        corridor10 = CorridorWorld(0.7499999832361937, 10.089999774470925, [5.284999881871045, 13.224999704398215], 0.0)
        corridor11 = CorridorWorld(4.9899998884648085, 4.9899998884648085, [7.834999824874103, 12.924999711103737], 1.5707963267948966)
        corridor12 = CorridorWorld(0.8399999812245369, 10.089999774470925, [10.384999767877162, 14.889999667182565], 0.0)
        corridor13 = CorridorWorld(4.999999888241291, 4.969999888911843, [12.929999710991979, 12.93499971088022], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor4,  corridor6, corridor8, corridor9, corridor10, corridor11, corridor12, corridor13]
        corridor_list = [corridor1, corridor2, corridor4, corridor6, corridor8, corridor10, corridor11, corridor12, corridor13]
        start_pose = [2.0562686920166016, 1.892181396484375, 0.1610918028903515]
        end_pose = [11.480597496032715, 12.35735034942627, -0.933557986462627]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 16:
        corridor1 = CorridorWorld(4.9899998884648085, 4.969999888911843, [2.734999938867986, 2.734999938867986], 1.5707963267948966)
        corridor2 = CorridorWorld(0.9899999778717756, 10.079999774694443, [5.279999881982803, 4.624999896623194], 0.0)
        corridor3 = CorridorWorld(4.979999888688326, 4.969999888911843, [7.829999824985862, 2.734999938867986], 1.5707963267948966)
        corridor4 = CorridorWorld(0.7499999832361943, 10.06999977491796, [5.834999869577587, 5.284999881871045], 1.5707963267948966)
        corridor5 = CorridorWorld(4.9899998884648085, 4.979999888688326, [7.829999824985862, 7.8249998250976205], 0.0)
        corridor_list = [corridor1, corridor2, corridor4, corridor5]
        start_pose = [2.0562686920166016, 1.892181396484375, 0.1610918028903515]
        end_pose = [9.140119552612305, 8.82896614074707, 0.37425914860034093]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)

    elif num == 17:
        corridor1 = CorridorWorld(4.9899998884648085, 4.969999888911843, [2.734999938867986, 2.734999938867986], 1.5707963267948966)
        corridor2 = CorridorWorld(0.9899999778717756, 10.079999774694443, [5.279999881982803, 4.624999896623194], 0.0)
        corridor3 = CorridorWorld(4.979999888688326, 4.969999888911843, [7.829999824985862, 2.734999938867986], 1.5707963267948966)
        corridor4 = CorridorWorld(0.7499999832361943, 10.06999977491796, [5.834999869577587, 5.284999881871045], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor4]
        start_pose = [2.0562686920166016, 1.892181396484375, 0.1610918028903515]
        end_pose = [5.787396430969238, 8.004776000976562, 1.59079628393173]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    elif num == 18:
        corridor1 = CorridorWorld(4.979999888688326, 4.969999888911843, [7.829999824985862, 2.734999938867986], 1.5707963267948966)
        corridor2 = CorridorWorld(0.7499999832361943, 10.06999977491796, [5.834999869577587, 5.284999881871045], 1.5707963267948966)
        corridor3 = CorridorWorld(4.989999888464808, 4.979999888688327, [7.829999824985862, 7.8249998250976205], 3.141592653589793)
        corridor4 = CorridorWorld(1.7299999613314854, 10.079999774694443, [5.279999881982803, 9.304999792017043], 3.141592653589793)
        corridor5 = CorridorWorld(4.9899998884648085, 4.9899998884648085, [2.734999938867986, 7.8249998250976205], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor4, corridor5]
        start_pose = [7.488751411437988, 1.4896717071533203, 1.0098192277286622]
        end_pose = [3.0030465126037598, 7.1541008949279785, -2.194297739327803]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    ################################
    # Large rooms gmap_6.yaml
    elif num ==19:
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.4499999899417165, 5.069999886676669, [4.944999889470637, 3.694999917410314], 1.5707963267948966)
        corridor4 = CorridorWorld(0.42999999038875103, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 0.0)
        corridor5 = CorridorWorld(0.4499999899417163, 2.249999949708581, [5.6949998727068305, 5.104999885894358], 1.5707963267948966)
        corridor_list = [corridor1,corridor3, corridor4, corridor5]
        start_pose = [2.1186683177948, 1.9495484828948975, 0.3805063771123649]
        end_pose = [5.830934047698975, 5.696508884429932, 2.0803865078004256]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)

    elif num == 20:
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.4299999903887506, 3.5599999204277992, [4.639999896287918, 4.644999896176159], 3.141592653589793)
        corridor6 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6]
        start_pose = [1.1300705671310425, 2.018000364303589, 0.1896916775775073]
        end_pose = [3.3331446647644043, 5.053731918334961, 1.7382273895680398]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=m.pi/4, delta_min=-0.5)

    elif num == 21:
        corridor1 = CorridorWorld(1.0999999754130843, 2.699999939650297, [5.839999869465828, 2.509999943897128], 1.5707963267948966)
        corridor2 = CorridorWorld(0.5399999879300587, 6.149999862536788, [3.314999925903976, 3.519999921321869], 3.141592653589793)
        corridor3 = CorridorWorld(0.4499999899417165, 5.069999886676669, [4.944999889470637, 3.694999917410314], 1.5707963267948966)
        corridor4 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor5 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor6 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor7 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor8 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor9 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        # Removed from this sequence.
        # corridor10 = CorridorWorld(0.7699999827891588, 2.699999939650297, [1.5899999644607306, 7.044999842531979], 0.0)
        # corridor11 = CorridorWorld(0.6299999859184031, 6.439999856054783, [2.4949999442324042, 9.05999979749322], 1.5707963267948966)
        corridor12 = CorridorWorld(0.6999999843537804, 2.699999939650297, [1.5899999644607306, 8.24999981559813], 3.141592653589793)
        corridor13 = CorridorWorld(1.6799999624490738, 1.7999999597668648, [1.0799999758601189, 8.709999805316329], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9,corridor12, corridor13]
        start_pose = [6.2307658195495605, 1.5503556728363037, 2.118358872661069]
        end_pose = [1.5123708248138428, 9.1657075881958, 2.930501583106648]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 22:
        corridor1 = CorridorWorld(0.689999984577298, 2.5299999434500933, [0.584999986924231, 11.014999753795564], -1.5707963267948966)
        corridor2 = CorridorWorld(0.42999999038875103, 2.699999939650297, [1.5899999644607306, 11.34499974641949], 0.0)
        corridor3 = CorridorWorld(0.43999999016523345, 2.5299999434500933, [1.5599999651312828, 11.014999753795564], -1.5707963267948966)
        corridor4 = CorridorWorld(0.7099999841302633, 2.699999939650297, [1.5899999644607306, 10.10499977413565], 0.0)
        corridor5 = CorridorWorld(0.7499999832361933, 4.619999896734953, [2.4349999455735087, 8.149999817833304], -1.5707963267948966)
        corridor6 = CorridorWorld(0.7699999827891585, 2.699999939650297, [1.5899999644607306, 7.044999842531979], 3.141592653589793)
        corridor7 = CorridorWorld(0.5199999883770943, 3.4799999222159386, [1.9799999557435513, 6.919999845325947], 0.0)
        corridor8 = CorridorWorld(0.6799999848008159, 5.949999867007136, [3.3799999244511127, 9.304999792017043], 1.5707963267948966)
        corridor9 = CorridorWorld(0.5199999883770943, 3.3199999257922173, [4.699999894946814, 8.4399998113513], 0.0)
        corridor10 = CorridorWorld(0.4799999892711642, 4.099999908357859, [4.499999899417162, 10.229999771341681], 1.5707963267948966)
        corridor11 = CorridorWorld(0.4999999888241291, 3.3199999257922173, [4.699999894946814, 9.649999784305692], 0.0)
        corridor12 = CorridorWorld(0.35999999195337334, 5.949999867007136, [5.239999882876873, 9.304999792017043], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor7, corridor8, corridor9, corridor10, corridor11, corridor12]
        start_pose = [0.5405622124671936, 11.975102424621582, -0.03224566090062711]
        end_pose = [5.415079593658447, 11.194485664367676, 1.9108924864768193]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 23:
        corridor1 = CorridorWorld(0.689999984577298, 2.5299999434500933, [0.584999986924231, 11.014999753795564], -1.5707963267948966)
        corridor2 = CorridorWorld(0.42999999038875103, 2.699999939650297, [1.5899999644607306, 11.34499974641949], 0.0)
        corridor3 = CorridorWorld(0.43999999016523345, 2.5299999434500933, [1.5599999651312828, 11.014999753795564], -1.5707963267948966)
        corridor4 = CorridorWorld(0.7099999841302633, 2.699999939650297, [1.5899999644607306, 10.10499977413565], 0.0)
        corridor5 = CorridorWorld(0.7499999832361933, 4.619999896734953, [2.4349999455735087, 8.149999817833304], -1.5707963267948966)
        corridor6 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor7 = CorridorWorld(0.6999999843537804, 2.699999939650297, [1.5899999644607306, 8.24999981559813], 3.141592653589793)
        corridor8 = CorridorWorld(1.6799999624490738, 1.7999999597668648, [1.0799999758601189, 8.709999805316329], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor7, corridor8]
        start_pose = [0.5405622124671936, 11.975102424621582, -0.03224566090062711]
        end_pose = [1.060972809791565, 8.939371109008789, 2.6292034991557545]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)
        front_overhang = 0.4
        rear_overhang = 0.1
        vehicle_width = 0.1
        vehicle_length = front_overhang + rear_overhang

        vehicle = Bicycle(
            state=[0.0, 0.0, 0.0],
            width=vehicle_width,
            length=vehicle_length,
            wheelbase=0.2,
            rear_axle_to_front=front_overhang,
            rectangular_footprint=True,
            v_max=1.0,
            v_min=-1.0,
            delta_max=0.785,
            delta_min=-0.785,
        )

    elif num == 24:

        corridor1 = CorridorWorld(0.7699999827891588, 2.699999939650297, [1.5899999644607306, 7.044999842531979], 0.0)
        corridor2 = CorridorWorld(0.5199999883770943, 3.4799999222159386, [1.9799999557435513, 6.919999845325947], 0.0)
        corridor3 = CorridorWorld(0.6799999848008159, 5.949999867007136, [3.3799999244511127, 9.304999792017043], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [0.5752564072608948, 7.412831783294678, 0.2284963254844108]
        end_pose = [3.3854763507843018, 11.246527671813965, 1.614247351847231]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    #########################
    # Large rooms gmap_1.yaml
    #########################
    elif num == 25:
        corridor1 = CorridorWorld(1.0099999774247401, 5.009999888017774, [5.3349998807534575, 8.624999807216227], 3.141592653589793)
        corridor2 = CorridorWorld(2.379999946802855, 4.969999888911843, [4.039999909698963, 10.464999766089022], 1.5707963267948966)
        corridor3 = CorridorWorld(0.9299999792128795, 4.9899998884648085, [2.734999938867986, 9.884999779053032], 3.141592653589793)
        corridor4 = CorridorWorld(2.489999944344163, 2.4699999447911978, [1.4849999668076634, 9.2149997940287], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4]
        start_pose = [5.6766180992126465, 9.044669151306152, -0.0624198795718705]
        end_pose = [2.1551687717437744, 8.76711654663086, -2.778199852381511]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)
        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)

    #########################
    # Large rooms map_2.yaml
    #########################
    elif num == 26:
        corridor1 = CorridorWorld(2.959999933838844, 4.9899998884648085, [20.499999541789293, 16.174999638460577], -1.5707963267948966)
        corridor2 = CorridorWorld(1.0099999774247401, 10.089999774470925, [20.784999535419047, 13.624999695457518], -1.5707963267948966)
        corridor3 = CorridorWorld(4.989999888464808, 4.979999888688327, [19.499999564141035, 11.07499975245446], 3.141592653589793)
        corridor4 = CorridorWorld(1.1599999740719782, 10.079999774694443, [16.949999621137977, 12.519999720156193], 3.141592653589793)
        corridor5 = CorridorWorld(2.2299999501556154, 4.979999888688326, [14.399999678134918, 12.384999723173678], 3.141592653589793)
        corridor6 = CorridorWorld(2.4599999450147156, 4.899999890476465, [13.149999706074595, 13.719999693334103], 1.5707963267948966)
        corridor7 = CorridorWorld(0.4099999908357843, 12.73999971523881, [8.009999820962548, 13.94499968830496], 3.141592653589793)
        corridor8 = CorridorWorld(2.299999948590994, 4.9899998884648085, [9.274999792687595, 13.229999704286456], 0.0)
        corridor9 = CorridorWorld(1.159999974071979, 7.399999834597111, [9.339999791234732, 10.679999761283398], -1.5707963267948966)
        corridor10 = CorridorWorld(2.3399999476969238, 4.9899998884648085, [9.199999794363976, 9.474999788217247], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor4, corridor6, corridor7, corridor9, corridor10]
        start_pose = [19.40408706665039, 18.369365692138672, -1.2036224182017405]
        end_pose = [8.326583862304688, 9.724310874938965, -1.797594522877994]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    ###########################
    # Large rooms map_14.yaml
    ###########################
    elif num == 27:
        corridor1 = CorridorWorld(4.999999888241291, 4.979999888688326, [4.789999892935157, 21.199999526143074], 1.5707963267948966)
        corridor2 = CorridorWorld(1.0199999772012234, 7.27999983727932, [5.929999867454171, 21.319999523460865], 0.0)
        # Removed from this sequence.
        # corridor3 = CorridorWorld(1.029999976977706, 7.269999837502837, [11.044999753125012, 21.004999530501664], 0.0)
        corridor4 = CorridorWorld(2.1199999526143065, 15.239999659359455, [8.499999810010195, 16.089999640360475], -1.5707963267948966)
        corridor5 = CorridorWorld(1.019999977201222, 23.449999475851655, [8.469999810680747, 11.984999732114375], -1.5707963267948966)
        corridor6 = CorridorWorld(4.979999888688326, 13.759999692440033, [10.249999770894647, 5.829999869689345], 0.0).invert_dimensions(-1)
        corridor7 = CorridorWorld(2.3499999474734046, 16.88999962247908, [8.684999805875123, 4.534999898634851], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor4, corridor5, corridor6, corridor7]
        start_pose = [3.0816783905029297, 19.552017211914062, -0.1543182359159075]
        end_pose = [2.529900550842285, 4.868592262268066, 3.1203191901411564]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    ###########################
    # Large rooms map_18.yaml
    ###########################
    elif num == 28:
        corridor1 = CorridorWorld(3.0299999322742233, 12.629999717697501, [21.384999522008002, 6.594999852590263], 1.5707963267948966)
        corridor2 = CorridorWorld(8.619999807327984, 8.31999981403351, [18.879999577999115, 8.59999980777502], 3.141592653589793)
        corridor3 = CorridorWorld(0.7499999832361945, 13.719999693334103, [16.11499963980168, 11.149999750778079], 1.5707963267948966)
        corridor4 = CorridorWorld(4.989999888464808, 10.64999976195395, [13.264999703504145, 15.514999653212726], 3.141592653589793)
        corridor5 = CorridorWorld(5.589999875053764, 10.469999765977263, [10.734999760054052, 18.254999591968954], 1.5707963267948966)
        corridor6 = CorridorWorld(1.0099999774247417, 14.239999681711197, [12.634999717585742, 20.13999954983592], 1.5707963267948966)
        corridor7 = CorridorWorld(3.649999918416141, 10.679999761283398, [13.279999703168869, 25.434999431483448], 3.141592653589793)
        corridor8 = CorridorWorld(0.7699999827891569, 15.779999647289515, [10.72999976016581, 25.52499942947179], 3.141592653589793)
        corridor9 = CorridorWorld(4.989999888464808, 8.069999819621444, [5.3349998807534575, 23.224999480880797], -1.5707963267948966)
        corridor10 = CorridorWorld(1.7099999617785222, 10.669999761506915, [3.984999910928309, 24.524999451823533], 1.5707963267948966)
        corridor11 = CorridorWorld(4.949999889358878, 2.4899999443441625, [2.734999938867986, 28.61499936040491], 1.5707963267948966)
        corridor12 = CorridorWorld(2.4899999443441625, 4.949999889358878, [1.4849999668076634, 27.364999388344586], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9, corridor10, corridor11, corridor12]
        start_pose = [21.973045349121094, 1.0339956283569336, 1.584682689199424]
        end_pose = [1.617842674255371, 26.465055465698242, -2.4360250174541616]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 29:
        corridor1 = CorridorWorld(0.689999984577298, 2.5299999434500933, [0.584999986924231, 11.014999753795564], -1.5707963267948966)
        corridor2 = CorridorWorld(0.42999999038875103, 2.699999939650297, [1.5899999644607306, 11.34499974641949], 0.0)
        corridor3 = CorridorWorld(0.43999999016523345, 2.5299999434500933, [1.5599999651312828, 11.014999753795564], -1.5707963267948966)
        corridor4 = CorridorWorld(0.7099999841302633, 2.699999939650297, [1.5899999644607306, 10.10499977413565], 0.0)
        corridor5 = CorridorWorld(0.7499999832361933, 4.619999896734953, [2.4349999455735087, 8.149999817833304], -1.5707963267948966)
        corridor6 = CorridorWorld(0.3899999912828207, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 0.0)
        corridor7 = CorridorWorld(0.809999981895089, 2.249999949708581, [3.264999927021563, 5.104999885894358], -1.5707963267948966)
        corridor8 = CorridorWorld(0.42999999038875103, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 0.0)
        corridor9 = CorridorWorld(0.36999999172985515, 5.979999866336584, [4.9049998903647065, 3.2399999275803566], -1.5707963267948966)
        corridor10 = CorridorWorld(0.42999999038875103, 3.5599999204277992, [4.639999896287918, 4.644999896176159], 0.0)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9, corridor10]
        start_pose = [0.7420470714569092, 11.980480194091797, -0.024385715758152154]
        end_pose = [5.326621055603027, 4.623477935791016, 3.0791739633864292]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 30:
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor6 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor7 = CorridorWorld(0.5399999879300594, 4.3199999034404755, [4.1999999061226845, 6.689999850466847], 0.0)
        corridor8 = CorridorWorld(0.44999998994171636, 2.369999947026372, [4.344999902881682, 7.51499983202666], 1.5707963267948966)
        corridor9 = CorridorWorld(0.42999999038875103, 3.3199999257922173, [4.699999894946814, 7.844999824650586], 0.0)
        corridor10 = CorridorWorld(0.44999998994171636, 2.369999947026372, [5.194999883882701, 7.51499983202666], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9, corridor10]
        start_pose = [1.2996304035186768, 3.5857529640197754, -0.3097032871130466]
        end_pose = [5.177893161773682, 8.025226593017578, -1.4711307327774705]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 31:
        corridor1 = CorridorWorld(1.0999999754130843, 2.699999939650297, [5.839999869465828, 2.509999943897128], 1.5707963267948966)
        corridor2 = CorridorWorld(0.5399999879300587, 6.149999862536788, [3.314999925903976, 3.519999921321869], 3.141592653589793)
        corridor3 = CorridorWorld(0.4499999899417165, 5.069999886676669, [4.944999889470637, 3.694999917410314], 1.5707963267948966)
        corridor4 = CorridorWorld(0.4299999903887506, 3.5599999204277992, [4.639999896287918, 4.644999896176159], 3.141592653589793)
        corridor5 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor6 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6]
        start_pose = [5.992624282836914, 1.7426304817199707, 1.6733771147387526]
        end_pose = [4.010105609893799, 5.181060314178467, 2.826377427796413]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 32:
        corridor1 = CorridorWorld(0.4999999888241287, 5.949999867007136, [6.109999863430858, 9.304999792017043], -1.5707963267948966)
        corridor2 = CorridorWorld(0.4999999888241287, 3.3199999257922173, [4.699999894946814, 10.849999757483602], 3.141592653589793)
        corridor3 = CorridorWorld(0.47999998927116366, 4.099999908357859, [5.2999998815357685, 10.229999771341681], -1.5707963267948966)
        corridor4 = CorridorWorld(0.4999999888241287, 3.3199999257922173, [4.699999894946814, 9.649999784305692], 3.141592653589793)
        corridor5 = CorridorWorld(0.47999998927116366, 4.099999908357859, [4.499999899417162, 10.229999771341681], -1.5707963267948966)
        corridor6 = CorridorWorld(0.5199999883770938, 3.3199999257922173, [4.699999894946814, 8.4399998113513], 3.141592653589793)
        corridor7 = CorridorWorld(0.6799999848008152, 5.949999867007136, [3.3799999244511127, 9.304999792017043], -1.5707963267948966)
        corridor8 = CorridorWorld(0.759999983012676, 1.6799999624490738, [2.8799999356269836, 6.799999848008156], 3.141592653589793)
        corridor9 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor10 = CorridorWorld(0.749999983236194, 4.619999896734953, [2.4349999455735087, 8.149999817833304], 1.5707963267948966)
        corridor11 = CorridorWorld(0.709999984130263, 2.699999939650297, [1.5899999644607306, 10.10499977413565], 3.141592653589793)
        corridor12 = CorridorWorld(0.4399999901652338, 2.5299999434500933, [1.5599999651312828, 11.014999753795564], 1.5707963267948966)
        corridor13 = CorridorWorld(0.4299999903887507, 2.699999939650297, [1.5899999644607306, 10.794999758712947], 3.141592653589793)
        corridor14 = CorridorWorld(0.6899999845772983, 2.5299999434500933, [0.584999986924231, 11.014999753795564], 1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6, corridor7, corridor8, corridor9, corridor10, corridor11, corridor12, corridor13, corridor14]
        start_pose = [6.054576873779297, 11.980480194091797, 3.1159584587364484]
        end_pose = [0.8194892406463623, 11.345454216003418, -3.1234125994265534]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.2, v_max=1.0, v_min=-1.0, delta_max=0.785, delta_min=-0.785)

    elif num == 33:
        corridor1 = CorridorWorld(0.8999999798834322, 3.8099999148398638, [2.489999944344163, 7.74499982688576], -1.5707963267948966)
        corridor2 = CorridorWorld(0.3899999912828207, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 0.0)
        corridor3 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], -1.5707963267948966)
        corridor_list = [corridor1, corridor2, corridor3]
        start_pose = [2.550778388977051, 8.500016212463379, -1.6359214275210974]
        end_pose = [3.266406774520874, 5.137430667877197, 1.5707963267948966]
        vehicle = Bicycle([0, 0, 0], width=0.34, length=0.34, wheelbase=0.25, v_max=0.5, v_min=-0.5, delta_max=0.785, delta_min=-0.785)

    elif num == 34:
        corridor1 = CorridorWorld(0.8999999798834322, 3.8099999148398638, [2.489999944344163, 7.74499982688576], -1.5707963267948966)
        corridor2 = CorridorWorld(0.3899999912828207, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 0.0)
        corridor3 = CorridorWorld(0.809999981895089, 2.249999949708581, [3.264999927021563, 5.104999885894358], -1.5707963267948966)
        corridor4 = CorridorWorld(0.42999999038875103, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 0.0)
        corridor5 = CorridorWorld(0.44999998994171586, 5.069999886676669, [4.944999889470637, 3.694999917410314], -1.5707963267948966)
        corridor6 = CorridorWorld(0.4299999903887506, 3.5599999204277992, [4.639999896287918, 4.644999896176159], 3.141592653589793)
        corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5, corridor6]
        start_pose = [2.4522154331207275, 9.091392517089844, -1.5707963267948963]
        end_pose = [4.699445724487305, 4.636357307434082, -3.122727062260215]
        vehicle = Bicycle([0, 0, 0], width=0.34, length=0.34, wheelbase=0.25, v_max=0.5, v_min=-0.5, delta_max=0.785, delta_min=-0.785)
        stop = 1

    elif num == 35:
        # S-bend: feasible bp, but the nominal 45-degree basic placements
        # overlap with opposite signs. R=1, r=.1, q=.9/sqrt(2).
        # Safe-half candidate (.675, .75) has norm > .9, whereas (q,q)
        # satisfies the ordinary clearance bounds (.45, .6).
        corridor_list = [
            CorridorWorld(.65, 6, [-2, 0], 0),
            CorridorWorld(.5, 3, [0, 1], m.pi/2),
            CorridorWorld(.65, 6, [2, 2], 0),
        ]
        start_pose = [-4, -.2, 0]
        end_pose = [4, 2.2, 0]
        vehicle = Unicycle(width=.2, length=.2, v_max=1, v_min=0,
                           omega_max=1, omega_min=-1)

    return corridor_list, start_pose, end_pose, vehicle


def overlap_centroid(first, second):
    """Clip the first rectangle by the second and return its area centroid.

    Half-plane clipping handles arbitrary corridor rotations and containment.
    Disjoint corridors and overlaps with zero area have no area centroid.
    """
    polygon = np.asarray(first.corners, dtype=float)
    for edge in second.W.T:
        if len(polygon) == 0:
            break
        clipped = []
        previous = polygon[-1]
        previous_distance = previous @ edge[:2] + edge[2]
        for current in polygon:
            distance = current @ edge[:2] + edge[2]
            if (distance <= 0) != (previous_distance <= 0):
                fraction = previous_distance / (previous_distance - distance)
                clipped.append(previous + fraction * (current - previous))
            if distance <= 0:
                clipped.append(current)
            previous, previous_distance = current, distance
        polygon = np.asarray(clipped, dtype=float)

    if len(polygon) < 3:
        raise ValueError("Consecutive corridors have no positive-area overlap.")

    # Translate before applying the shoelace formula to reduce cancellation.
    origin = polygon[0]
    vertices = polygon - origin
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    twice_area = cross.sum()
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    if abs(twice_area) <= 32 * np.finfo(float).eps * scale**2:
        raise ValueError("Consecutive corridors have no positive-area overlap.")
    return origin + ((vertices + following) * cross[:, None]).sum(axis=0) / (
        3 * twice_area
    )


def build_orthogonal_corridor_sequence(corridors, tolerance=1e-8):
    """Choose rectangle symmetry axes and the fewest auxiliary corridors.

    States 0/1 select horizontal/vertical axes. Perpendicular axes require a
    centered overlap. Offset parallel axes are joined by one perpendicular
    bridge contained in the union of that pair. Dynamic programming minimizes
    bridge count, then favors the longer rectangle symmetry axis.
    Only axis-aligned rectangles are supported; no geometry is snapped.
    """
    if not corridors:
        raise ValueError("At least one corridor is required.")
    bounds = []
    for index, corridor in enumerate(corridors, start=1):
        if abs(np.sin(2 * corridor.tilt)) > tolerance:
            raise ValueError(
                f"Corridor {index} is not axis-aligned. Exact 90-degree turns "
                "cannot be guaranteed using its symmetry axes. "
                "Auxiliary construction only supports axis-aligned rectangles."
            )
        corners = np.asarray(corridor.corners)
        bounds.append((corners.min(axis=0), corners.max(axis=0)))
    centers = [(low + high) / 2 for low, high in bounds]
    overlaps = []
    for index, ((low_a, high_a), (low_b, high_b)) in enumerate(
        zip(bounds, bounds[1:]), start=1
    ):
        low, high = np.maximum(low_a, low_b), np.minimum(high_a, high_b)
        if np.any(high - low <= tolerance):
            raise ValueError(f"Corridors {index} and {index+1} need positive overlap.")
        overlaps.append((low, high))

    def transition_cost(index, axis_a, axis_b):
        center = sum(overlaps[index]) / 2
        if axis_a != axis_b:
            if (abs(center[1-axis_a] - centers[index][1-axis_a]) > tolerance
                    or abs(center[1-axis_b] - centers[index+1][1-axis_b]) > tolerance):
                return None
            return 0
        return int(abs(centers[index][1-axis_a]
                       - centers[index+1][1-axis_a]) > tolerance)

    def short_axis_cost(index, axis):
        lengths = bounds[index][1] - bounds[index][0]
        return int(lengths[axis] < lengths[1-axis] - tolerance)

    costs = [(0, short_axis_cost(0, axis)) for axis in (0, 1)]
    parents = []
    for index in range(1, len(corridors)):
        next_costs, previous_axes = [], []
        for axis in (0, 1):
            choices = []
            for previous in (0, 1):
                bridge = transition_cost(index-1, previous, axis)
                if bridge is not None:
                    cost = (costs[previous][0] + bridge,
                            costs[previous][1] + short_axis_cost(index, axis))
                    choices.append((cost, previous))
            cost, previous = min(choices)
            next_costs.append(cost)
            previous_axes.append(previous)
        costs = next_costs
        parents.append(previous_axes)
    axis = min((0, 1), key=lambda item: costs[item])
    axes = [axis]
    for parent in reversed(parents):
        axis = parent[axis]
        axes.append(axis)
    axes.reverse()

    def rectangle(low, high, axis):
        lengths = high - low
        return CorridorWorld(
            width=lengths[1-axis], height=lengths[axis],
            center=(low + high) / 2, tilt=axis * np.pi / 2,
        )

    sequence, auxiliary_indices = [], []
    for index, axis in enumerate(axes):
        if index and transition_cost(index-1, axes[index-1], axis) == 1:
            # Common longitudinal span, full transverse span of both originals.
            low = np.minimum(bounds[index-1][0], bounds[index][0])
            high = np.maximum(bounds[index-1][1], bounds[index][1])
            low[axis], high[axis] = overlaps[index-1][0][axis], overlaps[index-1][1][axis]
            auxiliary_indices.append(len(sequence))
            sequence.append(rectangle(low, high, 1-axis))
        sequence.append(rectangle(*bounds[index], axis))
    return sequence, auxiliary_indices


def simplify_orthogonal_polyline(polyline, tolerance=1e-8):
    """Remove repeated/collinear waypoints, including redundant backtracking.

    Return retained raw indices to preserve the originating corridor pair for
    each corner. Shortcutting a collinear run stays within its original trace.
    Start/end coordinates are preserved; first/last turns are unconstrained.
    """
    kept = []
    for index in range(len(polyline)):
        kept.append(index)
        while len(kept) >= 3:
            a, b, c = polyline[kept[-3:]]
            incoming, outgoing = b-a, c-b
            len_in, len_out = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if (min(len_in, len_out) <= tolerance
                    or abs(cross2d(incoming, outgoing))
                    <= tolerance * len_in * len_out):
                kept.pop(-2)
            else:
                break
    return polyline[kept], np.asarray(kept)


def compute_polyline(corridors, start_pose, end_pose):
    """Return ordered XY waypoints, including both endpoints."""
    if not corridors:
        raise ValueError("At least one corridor is required.")
    points = [np.asarray(start_pose, dtype=float)[:2]]
    for index, (first, second) in enumerate(zip(corridors, corridors[1:]), start=1):
        try:
            points.append(overlap_centroid(first, second))
        except ValueError as error:
            raise ValueError(f"Corridors {index} and {index + 1}: {error}") from error
    points.append(np.asarray(end_pose, dtype=float)[:2])
    return np.asarray(points)


def compute_turn_directions(polyline, angle_tolerance=1e-9, length_tolerance=1e-9):
    """Return directions and signed angles (radians) at interior waypoints.

    Each triple is (previous point, overlap centroid, next point), so the
    first/last triples include the start/end position. Directions are +1 for
    left, -1 for right, and 0 for straight. A repeated point gives NaN for
    both values. An exact U-turn has angle pi and direction NaN because the
    three points cannot distinguish left from right.
    """
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Polyline must contain at least two XY points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Polyline points must be finite.")
    directions = np.full(len(points) - 2, np.nan)
    angles = np.full(len(points) - 2, np.nan)
    for index, (previous, current, following) in enumerate(
        zip(points, points[1:], points[2:])
    ):
        incoming = current - previous
        outgoing = following - current
        lengths = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(lengths) <= length_tolerance:
            continue
        incoming, outgoing = incoming / lengths[0], outgoing / lengths[1]
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        angle = np.arctan2(cross, incoming @ outgoing)
        if abs(angle) <= angle_tolerance:
            directions[index], angles[index] = 0, 0
        elif np.pi - abs(angle) <= angle_tolerance:
            angles[index] = np.pi
        else:
            directions[index], angles[index] = np.sign(angle), angle
    return directions, angles


def cross2d(a, b):
    """Scalar cross product of two planar vectors."""
    return a[0] * b[1] - a[1] * b[0]


def boundary_intersections(first, second, tolerance=1e-9):
    """Intersect finite boundary segments, independent of corridor orientation.

    For collinear overlapping edges, include their shared interval endpoints;
    the concavity filter subsequently discards points on straight boundaries.
    """
    points = []

    def add(point):
        if not any(np.linalg.norm(point - old) <= tolerance for old in points):
            points.append(point)

    # Broadcast the 4 x 4 edge pairs rather than doing 16 Python-level solves.
    a = np.asarray(first.corners)
    c = np.asarray(second.corners)
    r = np.roll(a, -1, axis=0) - a
    s = np.roll(c, -1, axis=0) - c
    r_length = np.linalg.norm(r, axis=1)[:, None]
    s_length = np.linalg.norm(s, axis=1)[None, :]
    delta = c[None, :, :] - a[:, None, :]
    denominator = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    nonparallel = abs(denominator) > 1e-12 * r_length * s_length
    numerator_t = delta[:, :, 0] * s[None, :, 1] - delta[:, :, 1] * s[None, :, 0]
    numerator_u = delta[:, :, 0] * r[:, None, 1] - delta[:, :, 1] * r[:, None, 0]
    t = np.divide(numerator_t, denominator, out=np.zeros((4, 4)), where=nonparallel)
    u = np.divide(numerator_u, denominator, out=np.zeros((4, 4)), where=nonparallel)
    valid = (nonparallel & (t >= -tolerance / r_length)
             & (t <= 1 + tolerance / r_length) & (u >= -tolerance / s_length)
             & (u <= 1 + tolerance / s_length))
    collinear = ~nonparallel & (abs(numerator_u) <= tolerance * r_length)
    # Iterate only actual intersections; preserve the original edge-pair order.
    for i, j in np.argwhere(valid | collinear):
        if valid[i, j]:
            add(a[i] + np.clip(t[i, j], 0, 1) * r[i])
        else:
            for point in (a[i], a[i] + r[i], c[j], c[j] + s[j]):
                along_r = (point - a[i]) @ r[i] / (r[i] @ r[i])
                along_s = (point - c[j]) @ s[j] / (s[j] @ s[j])
                if (-tolerance / r_length[i, 0] <= along_r
                        <= 1 + tolerance / r_length[i, 0]
                        and -tolerance / s_length[0, j] <= along_s
                        <= 1 + tolerance / s_length[0, j]):
                    add(point.copy())
    return np.asarray(points, dtype=float).reshape(-1, 2)


def is_union_concave_corner(point, first, second, tolerance=1e-9):
    """Check that the union's local interior angle is strictly between pi and 2pi.

    Active outward normals define each rectangle's local interior directions.
    Partition the unit circle at their tangents and measure the connected
    angular sector occupied by either rectangle. This also handles endpoint
    intersections and rotated rectangles without choosing corridor edge names
    or a finite probe radius.
    """
    active_normals = []
    for corridor in (first, second):
        distances = np.asarray(point) @ corridor.W[:2] + corridor.W[2]
        if np.any(distances > tolerance):
            return False
        normals = corridor.W[:2, np.abs(distances) <= tolerance].T
        if len(normals) == 0:  # Interior to either rectangle is interior to union.
            return False
        active_normals.append(normals)
    if all(len(normals) == 1 for normals in active_normals):
        # At a crossing strictly within both edges, the local union angle is
        # pi plus the angle between outward normals. No angular sweep needed.
        first_normal, second_normal = (normals[0] for normals in active_normals)
        angle = np.arctan2(abs(cross2d(first_normal, second_normal)),
                           first_normal @ second_normal)
        return bool(1e-9 < angle < np.pi - 1e-9)
    normals = np.vstack(active_normals)
    tangent_angles = np.arctan2(normals[:, 1], normals[:, 0]) + np.pi / 2
    boundaries = np.unique(np.mod(
        np.r_[tangent_angles, tangent_angles + np.pi], 2 * np.pi
    ))
    widths = np.diff(np.r_[boundaries, boundaries[0] + 2 * np.pi])
    midpoints = boundaries + widths / 2
    probes = np.column_stack((np.cos(midpoints), np.sin(midpoints)))
    occupied = np.zeros(len(probes), dtype=bool)
    for normals in active_normals:
        occupied |= np.all(probes @ normals.T <= 1e-12, axis=1)
    interior_angle = widths[occupied].sum()
    components = np.count_nonzero(occupied & ~np.roll(occupied, 1))
    return bool(components == 1 and np.pi + 1e-9 < interior_angle < 2*np.pi - 1e-9)


def compute_inside_corners(corridors, polyline, turn_directions, tolerance=1e-9,
                           transition_indices=None):
    """Return all boundary candidates and inside concave matches per transition.

    Multiple matches remain explicit: no arbitrary tie-break is applied.
    Straight, U-turn and degenerate transitions select no corner.
    """
    candidates, selected = [], []
    if transition_indices is None:
        transition_indices = range(len(corridors) - 1)
    for index, pair_index in enumerate(transition_indices):
        first, second = corridors[pair_index:pair_index + 2]
        intersections = boundary_intersections(first, second, tolerance)
        candidates.append(intersections)
        matches = []
        tau = turn_directions[index]
        if np.isfinite(tau) and tau != 0:
            previous, point, following = polyline[index:index + 3]
            incoming = point - previous
            outgoing = following - point
            incoming /= np.linalg.norm(incoming)
            outgoing /= np.linalg.norm(outgoing)
            for corner in intersections:
                offset = corner - point
                if (tau * cross2d(incoming, offset) > tolerance
                        and tau * cross2d(outgoing, offset) > tolerance
                        and is_union_concave_corner(corner, first, second, tolerance)):
                    matches.append(corner)
        selected.append(np.asarray(matches, dtype=float).reshape(-1, 2))
    return candidates, selected


@dataclass
class Fillet:
    center: np.ndarray
    incoming_tangent: np.ndarray
    outgoing_tangent: np.ndarray
    radius: float
    trim: float
    signed_angle: float
    fits_segments: bool


def compute_fillets(polyline, radius):
    """Construct inside-turn tangent circles and check shared segment lengths.

    Returns one Fillet (or None) per interior vertex and one remaining length
    per segment, after trims at BOTH ends. A negative remaining length means
    overlapping trims. NaN means an adjacent degenerate turn cannot be rounded.
    No circle is constructed for straight, repeated-point or U-turn vertices.
    This checks length compatibility only, not containment in eroded free space.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Fillet radius must be finite and positive.")
    points = np.asarray(polyline, dtype=float)
    directions, angles = compute_turn_directions(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    trims = np.zeros(len(points))
    fillets = []
    for j, (tau, angle) in enumerate(zip(directions, angles), start=1):
        if not np.isfinite(tau):
            trims[j] = np.nan
            fillets.append(None)
            continue
        if tau == 0:
            fillets.append(None)
            continue
        incoming = (points[j] - points[j - 1]) / lengths[j - 1]
        outgoing = (points[j + 1] - points[j]) / lengths[j]
        trim = radius * np.tan(abs(angle) / 2)
        trims[j] = trim
        q_minus = points[j] - trim * incoming
        q_plus = points[j] + trim * outgoing
        normal = tau * np.array([-incoming[1], incoming[0]])
        fillets.append(Fillet(
            center=q_minus + radius * normal,
            incoming_tangent=q_minus,
            outgoing_tangent=q_plus,
            radius=radius,
            trim=trim,
            signed_angle=angle,
            fits_segments=False,
        ))
    remaining = lengths - trims[:-1] - trims[1:]
    for index, fillet in enumerate(fillets):
        if fillet is not None:
            fillet.fits_segments = bool(np.all(remaining[index:index + 2] >= -1e-9))
    return fillets, remaining


def sample_fillet_arcs(fillets, samples_per_arc=100):
    """Compute arc coordinates separately from plotting so they can be timed."""
    arcs = []
    for fillet in fillets:
        if fillet is None:
            arcs.append(None)
            continue
        start = fillet.incoming_tangent - fillet.center
        start_angle = np.arctan2(start[1], start[0])
        angles = start_angle + np.linspace(0, fillet.signed_angle, samples_per_arc)
        arcs.append(fillet.center + fillet.radius * np.column_stack((
            np.cos(angles), np.sin(angles)
        )))
    return arcs


def check_smoothed_polyline(corridors, r, R, tol=1e-8, *, nominal=None,
                            validate_arcs=False, _cache=None):
    """Build local regions, then backtrack with a joint-solver fallback.

    The nominal check runs first and remains unchanged. Original overlap points
    are re-selected with equality, signed 2R spacing and local fillet regions.
    Optional endpoint slices participate in the same selection. Acceptance
    relies on full local-region membership. validate_arcs enables an independent
    continuous arc-clearance audit for testing.
    """
    nominal = check_orthogonal_polyline(corridors, r, R) if nominal is None else nominal
    report = dict(feasible=False, status='NO_ORTHOGONAL_POLYLINE', reason=nominal['reason'],
                  points=None, regions=[], corners=[], arc_clearances=[], solver=None,
                  validation_mode='continuous_arc_audit' if validate_arcs else 'region_membership')
    if not nominal['feasible']:
        return report
    initial = nominal['polyline']
    signs, _ = compute_turn_directions(initial)
    indices = np.arange(1, len(initial)-1)
    pair_indices = indices+nominal.get('corridor_offset', 0)
    cache = {} if _cache is None else _cache
    bounds = nominal.get('corridor_bounds')
    if bounds is None:
        from kappa_planner.helpers.nominal_polyline import _bounds
        bounds = [_bounds(c, tol) for c in corridors]
    headings = np.sign(np.diff(initial, axis=0))
    regions = [None]*len(initial)
    corners = []
    report['regions'], report['corners'] = regions, corners
    for j, pair_index in zip(indices, pair_indices):
        if signs[j-1] == 0:
            corners.append(np.empty((0, 2)))
            continue
        if not np.isfinite(signs[j-1]):
            report.update(status='CORNER_UNRESOLVED', reason=f'Vertex {j}: invalid turn.')
            return report
        corner_key = ('corners', int(pair_index))
        if corner_key not in cache:
            cache[corner_key] = axis_aligned_concave_corners(*bounds[pair_index:pair_index+2], tol)
        incoming, outgoing, point = headings[j-1], headings[j], initial[j]
        matches = []
        for corner in cache[corner_key]:
            delta = corner-point
            if (signs[j-1]*cross2d(incoming, delta) > tol
                    and signs[j-1]*cross2d(outgoing, delta) > tol):
                matches.append(corner)
        corners.append(np.asarray(matches).reshape(-1, 2))
        if len(matches) != 1:
            report.update(status='CORNER_UNRESOLVED',
                          reason=f'Vertex {j}: expected one relevant concave corner, found {len(matches)}.')
            return report
        door = nominal['doors'][j]
        key = ('region', int(pair_index), tuple(incoming), tuple(outgoing),
               tuple(door['x']), tuple(door['y']), tuple(matches[0]))
        if key not in cache:
            cache[key] = fillet_vertex_region(door, corridors[pair_index:pair_index+2],
                matches[0], incoming, outgoing, r, R, tol,
                pair_bounds=bounds[pair_index:pair_index+2], validate_inputs=False)
        regions[j] = cache[key]
    empty = [j for j, region in enumerate(regions) if region is not None and region['empty']]
    if empty:
        report.update(status='NO_SAFE_FILLET_REGION', vertices=empty,
                      reason=f'No fillet-admissible point remains at vertices {empty} (zero-based).')
        return report
    solution = solve_fillet_waypoints(nominal, regions, R, tol)
    report['solver'] = solution
    if not solution['feasible']:
        report.update(status=solution['status'], reason=solution['reason'])
        return report
    points = solution['points']
    if validate_arcs:
        for j, pair_index in zip(indices, pair_indices):
            pair, region = corridors[pair_index:pair_index+2], regions[j]
            if region is None:
                continue
            prediction = revised_corner_condition(points[j], region['corner'], headings[j-1],
                                                   headings[j], R, r, tol)
            # Directly check the original OR form as well as the convex search form.
            tangent_ok = all(np.all(q @ corridor.W[:2]+corridor.W[2]
                                   <= -r*np.linalg.norm(corridor.W[:2], axis=0)+tol)
                             for q, corridor in zip((points[j]-R*headings[j-1],
                                                     points[j]+R*headings[j]), pair))
            minimum, _ = continuous_arc_clearance(points[j:j+1], headings[j-1], headings[j], R,
                pair, corridor_union_boundary(pair), union_signed_clearance)
            report['arc_clearances'].append(float(minimum[0]))
            if not prediction['predicted'] or not tangent_ok or minimum[0] < r-tol:
                report.update(status='FILLET_VERIFICATION_FAILED', vertex=int(j),
                              reason=f'Vertex {j}: independent fillet validation failed; no fillets accepted.')
                return report
    report.update(feasible=True, status='FEASIBLE', points=points,
                  reason=('Polyline and fillets pass the independent arc audit.' if validate_arcs else
                          'Polyline satisfies alignment, 2R spacing and fillet-admissible regions.')
                         if any(region is not None for region in regions) else
                         'Polyline passes; no internal turns require fillets.')
    return report


def build_trajectory_geometry(corridors, r, R, *, start_pose=None, end_pose=None,
                              validate_arcs=False):
    """Select and validate waypoints, optionally including pose-coordinate ends.

    Without poses the original internal-only computation remains available.
    With poses there are N+1 vertices for N corridors, and segment j lies in C_j.
    """
    start = perf_counter()
    nominal = check_orthogonal_polyline(corridors, r, R)
    internal = nominal
    if (start_pose is None) != (end_pose is None):
        raise ValueError('Provide both endpoint poses, or neither.')
    nominal_seconds = perf_counter()-start
    smoothing_seconds = 0.
    endpoint_attempts = []
    local_geometry_cache = {}  # Scoped to this build: never stale after geometry/radius changes.
    modes = [(True, True), (True, False), (False, True), (False, False)] if start_pose is not None else [None]
    for mode in modes:
        before = perf_counter()
        nominal = internal if mode is None else extend_nominal_with_pose_coordinates(
            internal, corridors, start_pose, end_pose, r, R, exact_start=mode[0], exact_end=mode[1])
        nominal_seconds += perf_counter()-before
        before = perf_counter()
        smoothed = check_smoothed_polyline(corridors, r, R, nominal=nominal,
                                           validate_arcs=validate_arcs, _cache=local_geometry_cache)
        smoothing_seconds += perf_counter()-before
        if mode is not None:
            endpoint_attempts.append(dict(exact_start=mode[0], exact_end=mode[1],
                feasible=smoothed['feasible'], status=smoothed['status'], reason=smoothed['reason']))
        if smoothed['feasible'] or not internal['feasible']:
            break
    after_smoothing = perf_counter()
    geometry = dict(check=nominal, internal_check=internal,
                    smoothed_check=smoothed, feasible=smoothed['feasible'],
                    polyline=smoothed['points'] if smoothed['feasible'] else nominal['polyline'],
                    fillets=[], arcs=[], inside_corners=smoothed['corners'],
                    remaining_lengths=np.empty(0), endpoint_attempts=endpoint_attempts)
    if nominal['feasible']:
        directions, angles = compute_turn_directions(geometry['polyline'])
        geometry.update(turn_directions=directions, turn_angles=angles,
                        transition_indices=np.arange(1, len(geometry['polyline'])-1)
                        +nominal.get('corridor_offset', 0))
    if smoothed['feasible']:
        fillets, remaining = compute_fillets(smoothed['points'], R)
        geometry.update(fillets=fillets, arcs=sample_fillet_arcs(fillets), remaining_lengths=remaining)
    end = perf_counter()
    geometry['timings'] = {'Nominal check': nominal_seconds,
        'Fillet regions, compatibility and validation': smoothing_seconds,
        'Fillet construction': end-after_smoothing, 'Total': end-start}
    return geometry


def plot_fillets(ax, fillets, arcs):
    """Show complete supporting circles, directed fillet arcs and tangencies."""
    labels_seen = set()
    for fillet, arc in zip(fillets, arcs):
        if fillet is None:
            continue
        color = "tab:purple" if fillet.fits_segments else "darkorange"
        label = ("Fillet (length-compatible)" if fillet.fits_segments
                 else "Fillet (insufficient segment length)")
        ax.add_patch(Circle(
            fillet.center, fillet.radius, fill=False, color=color,
            linestyle=":", linewidth=1, alpha=0.65, zorder=2,
        ))
        ax.plot(*arc.T, color=color, linewidth=2.5, zorder=4,
                label=label if label not in labels_seen else None)
        labels_seen.add(label)
        tangencies = np.array([fillet.incoming_tangent, fillet.outgoing_tangent])
        ax.scatter(*tangencies.T, color=color, marker="x", s=30, zorder=5,
                   label="Tangency points" if "tangencies" not in labels_seen else None)
        labels_seen.add("tangencies")
        ax.scatter(*fillet.center, color=color, marker="+", s=25, zorder=4)


def corridor_union_boundary(corridors, tolerance=1e-9):
    """Return exposed finite edges of the union, including hole boundaries.

    Split rectangle edges at every boundary intersection. Discard pieces whose
    outward side is covered by another rectangle, including shared seams.
    Internal overlap boundaries therefore do not affect erosion clearance.
    """
    intersections = [[] for _ in corridors]
    for i, first in enumerate(corridors):
        for j in range(i + 1, len(corridors)):
            points = boundary_intersections(first, corridors[j], tolerance)
            intersections[i].extend(points)
            intersections[j].extend(points)
    boundary = []
    for index, corridor in enumerate(corridors):
        corners = np.asarray(corridor.corners)
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            edge = b - a
            length = np.linalg.norm(edge)
            normal = np.array([-edge[1], edge[0]]) / length
            if normal @ ((a + b) / 2 - corridor.center) < 0:
                normal = -normal
            parameters = [0., 1.]
            for point in intersections[index]:
                delta = point - a
                t = delta @ edge / length**2
                if (abs(cross2d(delta, edge)) <= tolerance * length
                        and -tolerance / length <= t <= 1 + tolerance / length):
                    parameters.append(float(np.clip(t, 0, 1)))
            parameters = np.sort(parameters)
            for low, high in zip(parameters, parameters[1:]):
                if (high - low) * length <= tolerance:
                    continue
                midpoint = a + ((low + high) / 2) * edge
                covered = False
                for other_index, other in enumerate(corridors):
                    if other_index == index:
                        continue
                    distances = midpoint @ other.W[:2] + other.W[2]
                    if np.all(distances <= tolerance):
                        active = other.W[:2, abs(distances) <= tolerance].T
                        # No active face: strictly inside. Otherwise determine
                        # if the outward ray also lies in the other rectangle.
                        if len(active) == 0 or np.all(active @ normal <= tolerance):
                            covered = True
                            break
                if not covered:
                    boundary.append([a + low * edge, a + high * edge])
    return np.asarray(boundary, dtype=float).reshape(-1, 2, 2)


def union_signed_clearance(points, corridors, boundary=None):
    """Signed distance to the union boundary, positive inside the union.

    Disk erosion by r is precisely the set with clearance >= r. Distances to
    finite exposed edges also give the circular offsets at concave corners.
    This works for rotated rectangles and holes without a polygon dependency.
    """
    if not corridors:
        raise ValueError("At least one corridor is required.")
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    if boundary is None:
        boundary = corridor_union_boundary(corridors)
    inside = np.zeros(len(flat), dtype=bool)
    for corridor in corridors:
        in_rectangle = np.ones(len(flat), dtype=bool)
        for nx, ny, offset in corridor.W.T:
            in_rectangle &= flat[:, 0] * nx + flat[:, 1] * ny + offset <= 1e-10
        inside |= in_rectangle
    minimum_squared = np.full(len(flat), np.inf)
    for a, b in boundary:
        edge = b - a
        dx, dy = flat[:, 0] - a[0], flat[:, 1] - a[1]
        t = np.clip((dx * edge[0] + dy * edge[1]) / (edge @ edge), 0, 1)
        squared = (dx - t * edge[0])**2 + (dy - t * edge[1])**2
        np.minimum(minimum_squared, squared, out=minimum_squared)
    distance = np.sqrt(minimum_squared)
    return np.where(inside, distance, -distance).reshape(shape)


def plot_union_erosion(ax, corridors, radius, resolution=EROSION_PLOT_RESOLUTION):
    """Render disk erosion of the ORIGINAL union as a sampled clearance contour.

    Geometry distances are exact up to floating-point precision; rendered
    contours are grid approximations. This display computation is excluded
    from trajectory timing and does not certify fillet collision clearance.
    """
    corners = np.vstack([corridor.corners for corridor in corridors])
    low, high = corners.min(axis=0), corners.max(axis=0)
    span = high - low
    padding = max(float(span.max()) * 0.025, radius * 0.2)
    low, high = low - padding, high + padding
    span = high - low
    sizes = np.maximum(50, np.ceil(resolution * span / span.max()).astype(int))
    x = np.linspace(low[0], high[0], sizes[0])
    y = np.linspace(low[1], high[1], sizes[1])
    xx, yy = np.meshgrid(x, y)
    boundary = corridor_union_boundary(corridors)
    clearance = union_signed_clearance(np.stack((xx, yy), axis=-1), corridors, boundary)
    maximum = max(float(clearance.max()), radius) + max(radius, 1.0)
    ax.contourf(x, y, clearance, levels=[0, radius, maximum],
                colors=["#f3c7b7", "#a7d9ce"], antialiased=True)
    if clearance.max() > radius:
        ax.contour(x, y, clearance, levels=[radius], colors=["#167365"], linewidths=1.2)
    else:
        ax.text(0.5, 0.5, "Erosion is empty at this plot resolution",
                transform=ax.transAxes, ha="center")
    for corridor in corridors:
        closed = np.vstack([corridor.corners, corridor.corners[0]])
        ax.plot(*closed.T, color="0.5", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.add_collection(LineCollection(boundary, colors="0.2", linewidths=1.1))
    ax.set(title=f"Original corridor union and erosion (r={radius:.3f} m)",
           xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(handles=[
        Patch(facecolor="#a7d9ce", edgecolor="#167365", label="Eroded union"),
        Patch(facecolor="#f3c7b7", label="Removed by erosion"),
    ], loc="upper right")


def plot_nominal_analysis(corridors, result, r, R, example_num):
    """Show erosion, colored overlap doors, and the H/V inference separately."""
    figure, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True,
                                constrained_layout=True)
    figure.suptitle(f"Example {example_num}: nominal orthogonal-polyline check")
    plot_union_erosion(axes[0], corridors, r)
    for ax in axes[1:]:
        for index, corridor in enumerate(corridors, start=1):
            closed = np.vstack([corridor.corners, corridor.corners[0]])
            ax.plot(*closed.T, color="0.6", linestyle="--", linewidth=0.8)
            ax.text(*corridor.center, f"C{index}", color="0.45", fontsize=8)
        ax.set(xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    axes[1].set_title(f"Overlaps and safe doors (r={r:.3f} m)")
    axes[2].set_title("Unique segment directions through internal corridors")
    cmap = plt.get_cmap("tab20")
    handles = []
    for j, (box, door) in enumerate(zip(result["raw_overlaps"], result["doors"])):
        color = cmap(j % 20)
        xmin, xmax, ymin, ymax = box
        if xmin <= xmax and ymin <= ymax:
            axes[1].add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                       fill=False, edgecolor=color, linestyle="--"))
        if door is not None:
            xmin, xmax = door["x"]
            ymin, ymax = door["y"]
            if min(xmax-xmin, ymax-ymin) <= 1e-10:
                axes[1].plot([xmin, xmax], [ymin, ymax], color=color,
                             linewidth=3, marker=".")
            else:
                axes[1].add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           facecolor=color, edgecolor=color, alpha=0.65))
            axes[1].annotate(f"D{j+1}", ((xmin+xmax)/2, (ymin+ymax)/2),
                             fontsize=8, xytext=(3, 3), textcoords="offset points")
        handles.append(Patch(facecolor=color, label=f"D{j+1}: C{j+1}–C{j+2}"))
    if handles:
        handles.append(Patch(facecolor="none", edgecolor="0.4", linestyle="--",
                             label="Dashed: raw overlap"))
        axes[1].legend(handles=handles, fontsize=7, ncol=2 if len(handles)>8 else 1)
    for j, direction in enumerate(result["directions"], start=1):
        first, second = result["doors"][j-1:j+1]
        center = np.asarray(corridors[j].center)
        color = "tab:blue" if direction == "H" else "tab:purple" if direction == "V" else "tab:red"
        spacing = result["spacing_pairs"][j-1]
        flagged = PLOT_SPACING_COUPLING and spacing["status"] in SPACING_COLORS
        if flagged:
            color = SPACING_COLORS[spacing["status"]]
        if direction in ("H", "V"):
            # Schematic undirected arrow; not a selected trajectory segment.
            a = np.array([sum(first["x"])/2, sum(first["y"])/2])
            b = np.array([sum(second["x"])/2, sum(second["y"])/2])
            transverse = "y" if direction == "H" else "x"
            value = (max(first[transverse][0], second[transverse][0])
                     + min(first[transverse][1], second[transverse][1]))/2
            a[1 if direction == "H" else 0] = value
            b[1 if direction == "H" else 0] = value
            center = (a+b)/2
            axes[2].annotate("", xy=b, xytext=a,
                             arrowprops=dict(arrowstyle="<->", color=color, lw=2,
                                             linestyle="--" if flagged else "-"))
        suffix = (" (2R coupled)" if spacing["status"] == "coupled"
                  else " (2R impossible)") if flagged else ""
        axes[2].annotate(f"C{j+1}: {direction}{suffix}", center, xytext=(4, 5),
                         textcoords="offset points", color=color, fontsize=8,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    status = "Accepted" if result["feasible"] else f"Rejected: {result['reason']}"
    axes[2].text(0.02, 0.02,
                 f"{status}\nMinimum internal segment length: 2R = {2*R:.3f} m\n"
                 f"Check time: {result['computation_time']*1000:.3f} ms",
                 transform=axes[2].transAxes, fontsize=8, wrap=True,
                 bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    return figure


def plot_bp_circle_sequence(corridors, geometry, vehicle, R, example_num, *, report=None):
    """Legacy placements and optional local same-turn shifting from bp geometry."""
    from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence

    if report is None:
        report = build_bp_circle_sequence(corridors, vehicle, geometry, radius=R,
                                          shift_same_turn=SHIFT_SAME_TURN_CIRCLES,
                                          shift_opposite_turn=SHIFT_OPPOSITE_TURN_CIRCLES,
                                          connect_tangents=CONNECT_BP_CIRCLES)
    shifting = report.get('shifting')
    opposite_shifting = report.get('opposite_shifting')
    initial_shifting = shifting or opposite_shifting
    chain = report.get('tangent_chain')
    diagnostics = report['diagnostics']
    flags = diagnostics['flags']
    arc_conflicts = {index for pair in flags['same_turn_arc_conflict'] for index in pair}
    opposite_conflicts = {index for pair in flags['opposite_turn_overlap'] for index in pair}
    outside_regions = set(flags['outside_Aj'])
    circle_checks = {item['circle_index']: item for item in diagnostics['circles']}
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(3, 1.35))
    ax = figure.add_subplot(layout[:, 0])
    key_ax = figure.add_subplot(layout[0, 1])
    notes_ax = figure.add_subplot(layout[1, 1])
    key_ax.set_axis_off()
    notes_ax.set_axis_off()
    plot_corridors(corridors, figure=figure, color='0.65', linewidth=.9,
                   plot_corridor_index=True)
    ax.set(xlabel='x [m]', ylabel='y [m]',
           title=f'Example {example_num}: circles from bp' +
                 (' after local shifts' if initial_shifting else ' (initial)'))
    ax.grid(alpha=.15)
    styles = {
        'forty_five_safe_half': ('#0072B2', '45 degrees: preferred half'),
        'safe_half_shifted': ('#009E73', 'Shifted: preferred half'),
        'forty_five_basic': ('#AA4499', '45 degrees: wall clearance'),
        'basic_shifted': ('#D55E00', 'Shifted: wall clearance'),
        'forbidden_tangent_ray': ('#8c564b', 'Blocking-corner fallback'),
    }
    handles = []
    if geometry['feasible']:
        bp = geometry['polyline']
        ax.plot(*bp.T, '.--', color='0.45', lw=1, label='Baseline polyline (bp)')
        if PLOT_SPACING_COUPLING:
            handles.extend(plot_spacing_coupling(
                ax, geometry['internal_check'], bp,
                door_offset=1 if geometry['check'].get('endpoint_constraints') else 0))
    used = set()
    if initial_shifting:
        initial_centers = {c['circle_index']: c['center']
                           for c in initial_shifting['initial_diagnostics']['circles']}
        moved = False
        for c in diagnostics['circles']:
            old, new = initial_centers[c['circle_index']], c['center']
            if np.linalg.norm(new-old) > 1e-8:
                moved = True
                ax.add_patch(Circle(old, c['radius'], fill=False, ec='0.55',
                                    ls='--', lw=.8, alpha=.5))
                ax.annotate('', xy=new, xytext=old,
                            arrowprops=dict(arrowstyle='->', color='0.3', lw=1.1))
        if moved:
            handles.append(Line2D([], [], color='0.55', ls='--',
                                  label='Before shift (arrows show movement)'))
    failures = []
    angles = np.linspace(0, np.pi/2, 60)  # Rendering basis shared by all circles.
    arc_cos, arc_sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    for item in report['placements']:
        if item['status'] == 'straight':
            continue
        corner = item['corner']
        ax.scatter(*corner, color='black', marker='x', s=28, zorder=5)
        if item['status'] == 'placement_failed':
            ax.scatter(*corner, color='red', marker='s', facecolors='none', s=100, zorder=6)
            failures.append(f"C{item['corridor_pair'][0]+1}/C{item['corridor_pair'][1]+1}: "
                            + item['geometry']['reason'])
            continue
        circle = item['circle']
        center = circle_checks[circle.index]['center']
        color, _ = styles[item['rule']]
        used.add(item['rule'])
        ax.add_patch(Circle(center, circle.radius, fill=False, ec=color, lw=.9, alpha=.4))
        if circle.index in opposite_conflicts:
            ax.add_patch(Circle(center, circle.radius, fill=False, ec='#b2188b',
                                lw=2, ls='--', zorder=5))
        # Samples are for display only; the pair diagnostics use analytic intersections.
        quarter = center + circle.radius*(-arc_cos*item['outgoing']
                                         + arc_sin*item['incoming'])
        ax.plot(*quarter.T, color='#c62828' if circle.index in arc_conflicts else color,
                lw=2.5, zorder=4)
        if circle.index in outside_regions:
            ax.scatter(*center, marker='s', facecolors='none', edgecolors='#c62828',
                       s=110, linewidths=1.5, zorder=6)
        ax.plot(*np.array([corner, center]).T, ':', color=color, lw=.9)
        ax.scatter(*center, color=color, marker='+', s=45, zorder=5)
        side = 1 if circle.index % 2 else -1
        ax.annotate(f"O{circle.index+1} ({'L' if item['turn'] > 0 else 'R'})", center,
                    xytext=(side*12, 12), textcoords='offset points', color=color,
                    fontsize=8, ha='left' if side > 0 else 'right',
                    arrowprops=dict(arrowstyle='-', color=color, lw=.6))
    base_handles, _ = ax.get_legend_handles_labels()
    if chain:
        color = '#00695c' if chain['feasible'] else '#ad6800'
        for tangent in chain['tangents']:
            ax.plot(*np.array([tangent['start'], tangent['end']]).T,
                    color=color, lw=2.1, zorder=7)
            if np.linalg.norm(tangent['end']-tangent['start']) > 1e-8:
                middle = (tangent['start']+tangent['end'])/2
                half = .08*(tangent['end']-tangent['start'])
                ax.annotate('', xy=middle+half, xytext=middle-half,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.6), zorder=8)
        for arc in chain['arcs']:
            angles = np.linspace(arc['enter'], arc['leave'], 40)
            points = arc['center']+arc['radius']*(-np.cos(angles)[:, None]*arc['region']['outgoing']
                                                 +np.sin(angles)[:, None]*arc['region']['incoming'])
            ax.plot(*points.T, color=color, lw=3, zorder=7)
        handles.append(Line2D([], [], color=color, lw=2.5,
                              label='Internal tangent chain' if chain['feasible'] else 'Partial chain (unresolved)'))
        for index in chain['skipped']:
            ax.scatter(*circle_checks[index]['center'], marker='x', color='0.3', s=90, zorder=9)
    handles = base_handles + handles
    handles.append(Line2D([], [], color='black', marker='x', ls='none', label='Relevant bp corner'))
    for rule, (color, label) in styles.items():
        if rule in used:
            handles.append(Line2D([], [], color=color, lw=1.8, label=label))
    if arc_conflicts:
        handles.append(Line2D([], [], color='#c62828', lw=2.5,
                              label='Intersecting same-turn quarters'))
    if opposite_conflicts:
        handles.append(Line2D([], [], color='#b2188b', lw=2, ls='--',
                              label='Opposite-turn circle overlap'))
    if outside_regions:
        handles.append(Line2D([], [], color='#c62828', marker='s', markerfacecolor='none',
                              ls='none', label='Outside current bp A_j'))
    key_ax.legend(handles=handles, loc='upper left', borderaxespad=0,
                  fontsize=9, frameon=False, title='Placement rules', labelspacing=1.1)
    notes = [report['reason'], 'O_i corresponds to corridors C_i and C_(i+1).',
             'Radius and turn sign are inherited from the bp. '
             'Circle records remain separate; full trajectory connections are not certified.']
    if shifting:
        notes.append(f"Same-turn repair: {shifting['accepted']} accepted, "
                     f"{shifting['rejected']} rejected. Coincident centers join successive "
                     'quarter-arcs. Rejected proposals retain their original centers.')
    if opposite_shifting:
        notes.append(f"Opposite-turn repair: {opposite_shifting['accepted']} accepted, "
                     f"{opposite_shifting['rejected']} unresolved. "
                     'Separation and local admissibility checked; full tangent chain unchecked.')
    notes.append('Faint outlines: supporting circles. Bold curves: full relevant quarter-arcs.')
    diagnostic_pairs = diagnostics['pairs']
    opposite = [pair for pair in diagnostic_pairs if pair['opposite_turn_overlap']]
    same_conflicts = [pair for pair in diagnostic_pairs
                      if pair['same_turn'] and pair['quarter_conflict']]
    notes.append(f"Consecutive pairs: {len(opposite)} opposite-turn overlaps. "
                 f"Same-turn quarter conflicts: {len(same_conflicts)}. "
                 f"Outside current A_j: {len(outside_regions)}.")
    if any(item['status'] == 'straight' for item in report['placements']):
        notes.append('Straight bp transitions have no turning circle.')
    notes.extend(failures)
    if chain:
        notes = [f"Internal tangent chain: {chain['status']}. " + chain['reason'],
                 f"{len(chain['tangents'])} tangent links; skipped circles: "
                 + (', '.join(f'O{i+1}' for i in chain['skipped']) or 'none') + '.',
                 'Bold path: used arcs and directed tangents. Each used arc turns by at most 90 degrees. '
                 'Zero-length links allow coincident-circle joins.',
                 'Circle positions and bp are unchanged by the connection stage. '
                 'Start/end pose maneuvers are not included.',
                 f"Remaining circle flags: {len(opposite)} opposite overlaps; "
                 f"{len(same_conflicts)} same-turn conflicts; {len(outside_regions)} outside A_j."]
    notes_ax.text(0, 1, '\n\n'.join(fill(note, 38) for note in notes),
                  va='top', fontsize=9, transform=notes_ax.transAxes)
    print(f"  BP circles: {report['status']}: {report['reason']}")
    if shifting:
        print(f"    Same-turn shifting and checks: {shifting['computation_time']*1000:.3f} ms")
    if chain:
        print(f"    Tangent chain: {chain['status']}; {chain['computation_time']*1000:.3f} ms; "
              f"skipped={chain['skipped']}")
    print(f"    Placement: {report['placement_time']*1000:.3f} ms; "
          f"consecutive-pair / A_j checks: {diagnostics['computation_time']*1000:.3f} ms")
    for pair in opposite + same_conflicts:
        label = 'opposite-turn overlap' if pair['opposite_turn_overlap'] else 'same-turn quarter conflict'
        print(f"    O{pair['first']+1}/O{pair['second']+1}: {label} "
              f"({'consecutive' if pair['consecutive'] else 'nonconsecutive'})")
    if outside_regions:
        print('    Outside current bp A_j:', ', '.join(f'O{i+1}' for i in sorted(outside_regions)))
    for item in report['placements']:
        first, second = item['corridor_pair']
        print(f"    C{first+1}/C{second+1}, bp vertex {item['bp_vertex']+1}: "
              f"{item['rule'] or item['status']}")
    return figure, report


def plot_tangent_refinement(corridors, geometry, r, R, example_num):
    """Third figure: a separately certified refinement of figure 2's solution."""
    from kappa_planner.helpers.tangent_refinement import refine_forward_backward

    figure, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    plot_corridors(corridors, figure=figure, colormap=True, plot_corridor_index=True)
    smoothed = geometry['smoothed_check']
    ax.set(xlabel='x [m]', ylabel='y [m]', title=f'Example {example_num}: forward/backward circle refinement')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    if not smoothed['feasible']:
        report = dict(feasible=False, status='NO_FEASIBLE_BASELINE', reason=smoothed['reason'])
        ax.text(.02, .02, 'Refinement needs a certified orthogonal solution.\n'+smoothed['reason'],
                transform=ax.transAxes, fontsize=9, wrap=True)
        return figure, report
    original, regions = smoothed['points'], smoothed['regions']
    report = refine_forward_backward(original, regions, corridors, r, R,
                             corridor_offset=geometry['check'].get('corridor_offset', 0))
    ax.plot(*original.T, '--', color='0.55', lw=1, label='Original orthogonal polyline')
    if report['feasible']:
        for arc in report['arcs']:
            j = arc['index']
            region = regions[j]
            low, high = region['low'], region['high']
            padding = max(.01, float(np.max(high-low))*.03)
            x = np.linspace(low[0]-padding, high[0]+padding, 120)
            y = np.linspace(low[1]-padding, high[1]+padding, 120)
            xx, yy = np.meshgrid(x, y)
            margin = region_margin(np.stack((xx, yy), axis=-1), region)
            if margin.max() > 0:
                ax.contourf(x, y, margin, levels=[0, margin.max()+1], colors=['#2563eb'], alpha=.15)
                ax.contour(x, y, margin, levels=[0], colors=['#2563eb'], linewidths=.6)
            angles = np.linspace(0, np.pi/2, 100)
            certified = arc['center']+R*(-np.cos(angles)[:, None]*region['outgoing']
                                        + np.sin(angles)[:, None]*region['incoming'])
            ax.add_patch(Circle(arc['center'], R, fill=False, edgecolor='tab:purple',
                                linestyle=':', linewidth=.8, alpha=.35))
            ax.plot(*certified.T, color='#c9b4e5', linewidth=4)
            # Rendering only: certification never samples arcs for collision checks.
            used_angles = np.linspace(arc['enter'], arc['leave'], 100)
            used_points = arc['center']+R*(-np.cos(used_angles)[:, None]*region['outgoing']
                                          + np.sin(used_angles)[:, None]*region['incoming'])
            ax.plot(*used_points.T, color='tab:purple', linewidth=2.3)
            ax.annotate(f"C{j} {'L' if arc['turn'] > 0 else 'R'}", arc['center'],
                        xytext=(5, 5), textcoords='offset points', fontsize=7)
        for tangent in report['tangents']:
            points = np.array([tangent['start'], tangent['end']])
            ax.plot(*points.T, color='tab:green', linewidth=2)
            ax.scatter(*points.T, color='tab:green', marker='x', s=20, zorder=5)
        ax.legend(handles=[
            Line2D([], [], color='0.55', linestyle='--', label='Original orthogonal polyline'),
            Patch(facecolor='#2563eb', alpha=.15, label='Admissible vertex regions'),
            Line2D([], [], color='#c9b4e5', lw=4, label='Certified quarter arcs'),
            Line2D([], [], color='tab:purple', lw=2.3, label='Actually traversed arcs'),
            Line2D([], [], color='tab:green', lw=2, label='Certified tangent segments'),
        ], fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.))
    note = report['status']+': '+report['reason']
    note += f"\nRefinement computation (excluding plotting): {report['computation_time']*1000:.2f} ms"
    if report.get('history'):
        counts = {action: sum(step['action'] == action for step in report['history'])
                  for action in ('corner_move', 'neighbor_repair', 'skip', 'keep')}
        note += (f"\nForward + backward: {counts['corner_move']} corner moves; "
                 f"{counts['neighbor_repair']} neighbor repairs;\n{counts['skip']} skips; "
                 f"{counts['keep']} unchanged.")
    note += '\nSame endpoints as figure 2; exact pose-heading connections are not enforced.'
    ax.text(.02, .02, note, transform=ax.transAxes, fontsize=8, wrap=True,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    print(f"  Tangent refinement: {report['status']}: {report['reason']}")
    print(f"  Refinement computation (excluding plotting): {report['computation_time']*1000:.3f} ms")
    return figure, report


def main(example_num=EXAMPLE_NUM, fillet_radius=FILLET_RADIUS):
    corridors, _start_pose, _end_pose, vehicle = example_corridor_sequence(example_num)
    R = vehicle.max_radius if fillet_radius is None else fillet_radius
    r = vehicle.width/2
    geometry = build_trajectory_geometry(corridors, r, R, start_pose=_start_pose, end_pose=_end_pose,
                                         validate_arcs=VALIDATE_BASELINE_ARCS)

    result = geometry["check"]
    smoothed = geometry['smoothed_check']
    print(f"Example {example_num}: polyline stage = {result['status']}")
    print(f"  Fillet stage = {smoothed['status']}: {smoothed['reason']}")
    endpoint_modes = result.get('endpoint_modes', ('unavailable', 'unavailable'))
    print(f"  Endpoint positions: start={endpoint_modes[0]}, end={endpoint_modes[1]}")
    if smoothed['solver']:
        print(f"  Waypoint selection: {smoothed['solver']['selection_method']} "
              f"({smoothed['solver']['backtracking_attempts']} backtracking candidates)")
    if smoothed['arc_clearances']:
        print(f"  Minimum arc clearances: {np.round(smoothed['arc_clearances'], 6)}; required >= {r:g} m")
    if result["reason"]:
        print(f"  {result['reason']} (indices in reasons are zero-based)")
    print(f"  Segment directions: {result['directions']}")
    for item in result["ambiguous_overlaps"]:
        first, second = item["overlaps"]
        kind = "positive area" if item["positive_area"] else "boundary contact"
        print(f"  C{item['corridor']+1}: D{first+1} and D{second+1} intersect "
              f"({kind}); safe bounds (xmin, xmax, ymin, ymax) = "
              f"{np.round(item['safe_intersection'], 4)}")
    for stage, seconds in geometry["timings"].items():
        print(f"  {stage}: {seconds*1000:.3f} ms")
    spacing_pairs = geometry['internal_check']['spacing_pairs']
    spacing_counts = {status: sum(pair['status'] == status for pair in spacing_pairs)
                      for status in ('guaranteed', 'coupled', 'impossible', 'unavailable')}
    print(f"  Original safe-door pairs, local 2R spacing: {spacing_counts}")
    plot_nominal_analysis(corridors, geometry['internal_check'], r, R, example_num)

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(3, 1.35))
    ax = figure.add_subplot(layout[:, 0])
    legend_ax = figure.add_subplot(layout[0, 1])
    notes_ax = figure.add_subplot(layout[1, 1])
    for sidebar_ax in (legend_ax, notes_ax):
        sidebar_ax.set_axis_off()
    plot_corridors(corridors, figure=figure, colormap=True, plot_corridor_index=True)
    ax.set(xlabel="x [m]", ylabel="y [m]")
    if not result["feasible"]:
        ax.set_title(f"Example {example_num}: nominal check rejected")
        notes_ax.text(0, 1, "No polyline or fillets computed.\n\n"
                      + fill(result['reason'], width=38),
                      transform=notes_ax.transAxes, fontsize=9, va="top")
        if PLOT_BP_CIRCLES:
            plot_bp_circle_sequence(corridors, geometry, vehicle, R, example_num)
        if PLOT_TANGENT_REFINEMENT:
            plot_tangent_refinement(corridors, geometry, r, R, example_num)
        plt.show()
        return
    plot_corridors(shrink_corridor_list(corridors, r), figure=figure,
                   color="0.4", linestyle=":", linewidth=1.2, label="Shrunken corridors")
    polyline = geometry["polyline"]
    ax.plot(*polyline.T, color="tab:blue", linewidth=2,
            label="Fillet-compatible polyline" if smoothed['feasible'] else "Straight polyline only")
    for region in smoothed['regions']:
        if region is None or region['empty']:
            continue
        low, high = region['low'], region['high']
        padding = max(float(np.max(high-low))*.03, .01)
        x = np.linspace(low[0]-padding, high[0]+padding, 150)
        y = np.linspace(low[1]-padding, high[1]+padding, 150)
        xx, yy = np.meshgrid(x, y)
        margins = region_margin(np.stack((xx, yy), axis=-1), region)
        if margins.max() > 0:
            ax.contourf(x, y, margins, levels=[0, margins.max()+1], colors=['#2563eb'], alpha=.22)
            ax.contour(x, y, margins, levels=[0], colors=['#1d4ed8'], linewidths=.8)
    ax.scatter(*polyline.T, color="#9ecae1", edgecolors="black", s=35,
               label="Polyline vertices", zorder=5)
    for index, point in enumerate(polyline, start=1):
        suffix = ' (entry)' if index == 1 else ' (exit)' if index == len(polyline) else ''
        ax.annotate(f"p{index}{suffix}", point, xytext=(5, 5), textcoords="offset points", fontsize=8)
    poses = np.array([_start_pose[:2], _end_pose[:2]])
    ax.scatter(*poses.T, marker='*', s=90, color='black', label='Input pose positions', zorder=6)
    plot_fillets(ax, geometry["fillets"], geometry["arcs"])
    spacing_handles = []
    if PLOT_SPACING_COUPLING:
        # Successful endpoint extension adds one point before the original doors.
        door_offset = 1 if result.get('endpoint_constraints') else 0
        spacing_handles = plot_spacing_coupling(
            ax, geometry['internal_check'], polyline, door_offset=door_offset)
    corners = (np.vstack(geometry["inside_corners"]) if geometry["inside_corners"]
               else np.empty((0, 2)))
    for index, corner in enumerate(corners):
        ax.add_patch(Circle(corner, r, fill=False, edgecolor="red", linewidth=1.5,
                            label="Corner circles (radius r)" if index == 0 else None))
    if len(corners):
        ax.scatter(*corners.T, color="red", s=15, zorder=6)
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    print(f"  Segment lengths (including entry/exit): {np.round(lengths, 4)}; required >= {2*R:.4f} m")
    ax.set_title(f"Example {example_num}: " +
                 ("polyline + fillets accepted" if smoothed['feasible'] else "fillet stage not accepted"))
    notes = [
        f"Geometry computation: {geometry['timings']['Total']*1000:.2f} ms",
        smoothed['reason'],
        f"Endpoint positions: start={endpoint_modes[0]}, end={endpoint_modes[1]}.",
        "Headings not enforced.",
    ]
    if PLOT_SPACING_COUPLING:
        notes.append(f"Original safe-door pairs: {spacing_counts['guaranteed']} guaranteed, "
                     f"{spacing_counts['coupled']} 2R-coupled, "
                     f"{spacing_counts['impossible']} impossible, "
                     f"{spacing_counts['unavailable']} unclassified.")
        notes.append("Orange flags mean some aligned choices are too short; "
                     "they do not indicate a violation by this polyline.")
    notes_ax.text(0, 1, "\n\n".join(fill(note, width=38) for note in notes if note),
                  transform=notes_ax.transAxes, fontsize=9, va="top", linespacing=1.35)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='#2563eb', alpha=.22))
    labels.append('Fillet-admissible vertex regions')
    handles.extend(spacing_handles)
    labels.extend(handle.get_label() for handle in spacing_handles)
    legend_ax.legend(handles, [fill(label, width=30) for label in labels],
                     loc="upper left", borderaxespad=0, fontsize=9,
                     frameon=False, labelspacing=1.1, handlelength=2.5,
                     title="Plot key", title_fontsize=10)
    if PLOT_BP_CIRCLES:
        plot_bp_circle_sequence(corridors, geometry, vehicle, R, example_num)
    if PLOT_TANGENT_REFINEMENT:
        plot_tangent_refinement(corridors, geometry, r, R, example_num)
    plt.show()


if __name__ == "__main__":
    main()
