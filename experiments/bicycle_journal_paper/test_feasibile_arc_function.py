import sys
from math import pi, tan
from types import SimpleNamespace

# Adjust these imports to match your package structure
from kappa_planner.corridor import CorridorWorld
from kappa_planner.geometry import Point
from kappa_planner.vehicle import Bicycle, Unicycle
from kappa_planner.helpers.arc_feasibility import (
    compute_intermediate_circle_geometry,
)
from kappa_planner.helpers.plot_helpers import plot_corridors
from kappa_planner.helpers.corridor_geometry import get_corner_point
from kappa_planner.helpers.axis_aligned_int_circle_sequence import (
    build_intermediate_circles_sequence
)
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


import numpy as np


import numpy as np
from matplotlib.patches import Circle


def plot_intermediate_circles_sequence_debug(
    ax,
    intermediate_circles_sequence,
    footprint_radius,
    plot_swept_circle=True,
    plot_shifted_circle=True,
    plot_corner_small_circles=True,
    legend_outside=True,
    plot_forbidden_points=True,
):
    """
    Plot nominal, swept, shifted, and corner-footprint circles from an
    IntermediateCirclesSequence.

    This function does not require circle.geometry_result.

    :param ax: Matplotlib axis.
    :param intermediate_circles_sequence: Sequence of IntermediateCircle objects.
    :param footprint_radius: Robot footprint radius, usually vehicle.width / 2.
    :param plot_swept_circle: Whether to plot swept circles of radius R + r.
    :param plot_shifted_circle: Whether to plot shifted circles at s_max.
    :param plot_corner_small_circles: Whether to plot footprint circles at corner points.
    :param legend_outside: Whether to place legend outside the axis.
    """

    for i, circle in enumerate(intermediate_circles_sequence):

        center = circle.center
        R = circle.radius
        r = footprint_radius
        S = R + r

        is_merged = getattr(circle, "is_merged", False) or getattr(circle, "merged", False)

        rule = getattr(circle, "construction_rule", None)

        if is_merged:
            label_suffix = f"{i}: merged"
        elif rule is not None:
            label_suffix = f"{i}: {rule}"
        else:
            label_suffix = f"{i}"

        # --------------------------------------------------------
        # Nominal Intermediate Circle
        # --------------------------------------------------------
        ax.add_patch(
            Circle(
                (center.x, center.y),
                R,
                fill=False,
                linewidth=2.0,
                label=f"circle {label_suffix}",
            )
        )

        ax.plot(center.x, center.y, "o", markersize=5)
        ax.text(center.x, center.y, f"  C{i}", fontsize=9)

        # --------------------------------------------------------
        # Nominal swept circle
        # --------------------------------------------------------
        if plot_swept_circle:
            ax.add_patch(
                Circle(
                    (center.x, center.y),
                    S,
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"swept {i}",
                )
            )

        # --------------------------------------------------------
        # Small footprint circles centered at corner points
        # --------------------------------------------------------
        if plot_corner_small_circles:
            corner_points = []

            if hasattr(circle, "merged_corner_points"):
                corner_points = list(circle.merged_corner_points)
            elif hasattr(circle, "corner_point"):
                corner_points = [circle.corner_point]

            for k, corner_point in enumerate(corner_points):
                if corner_point is None:
                    continue

                ax.add_patch(
                    Circle(
                        (corner_point.x, corner_point.y),
                        r,
                        fill=False,
                        edgecolor="green",
                        linestyle="-.",
                        linewidth=1.5,
                        label=f"corner small circle {i}.{k}",
                    )
                )

                ax.plot(
                    corner_point.x,
                    corner_point.y,
                    "gx",
                    markersize=6,
                )

                ax.text(
                    corner_point.x,
                    corner_point.y,
                    f"  P{i}.{k}",
                    fontsize=8,
                    color="green",
                )

        # --------------------------------------------------------
        # Forbidden points
        # --------------------------------------------------------
        if plot_forbidden_points:
            forbidden_points = getattr(circle, "forbidden_points", [])

            for k, forbidden_point in enumerate(forbidden_points):
                if forbidden_point is None:
                    continue

                ax.plot(
                    forbidden_point.x,
                    forbidden_point.y,
                    marker="x",
                    color="purple",
                    markersize=8,
                    markeredgewidth=2.0,
                    linestyle="None",
                    label=f"forbidden point {i}.{k}",
                )

                ax.text(
                    forbidden_point.x,
                    forbidden_point.y,
                    f"  F{i}.{k}",
                    fontsize=8,
                    color="purple",
                )

        # --------------------------------------------------------
        # Shifted circle at s_max
        # --------------------------------------------------------
        if plot_shifted_circle:
            s_max = getattr(circle, "s_max", None)
            d_world = getattr(circle, "shift_direction_world", None)
            s_max_reason = getattr(circle, "s_max_reason", "")

            if (
                s_max is not None
                and d_world is not None
                and np.isfinite(s_max)
                and s_max > 0.0
            ):
                shifted_x = center.x + s_max * d_world[0]
                shifted_y = center.y + s_max * d_world[1]

                ax.plot(
                    [center.x, shifted_x],
                    [center.y, shifted_y],
                    color="red",
                    linestyle=":",
                    linewidth=1.5,
                )

                ax.add_patch(
                    Circle(
                        (shifted_x, shifted_y),
                        R,
                        fill=False,
                        edgecolor="red",
                        linewidth=2.0,
                        label=f"shifted circle {i}",
                    )
                )

                if plot_swept_circle:
                    ax.add_patch(
                        Circle(
                            (shifted_x, shifted_y),
                            S,
                            fill=False,
                            edgecolor="red",
                            linestyle="--",
                            linewidth=1.2,
                            label=f"shifted swept {i}",
                        )
                    )

                ax.plot(shifted_x, shifted_y, "ro", markersize=5)

                ax.text(
                    shifted_x,
                    shifted_y,
                    f"  C{i} shifted\n  {s_max_reason}",
                    fontsize=8,
                    color="red",
                )

    ax.set_aspect("equal", adjustable="box")

    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        ax.figure.subplots_adjust(right=0.75)
    else:
        ax.legend()


def plot_intermediate_circle_geometry(
    ax,
    result,
    corner_point=None,
    axis_scale=0.4,
    plot_swept_circle=True,
    plot_shifted_circle=True,
):
    """
    Plot the Intermediate Circle result returned by
    compute_intermediate_circle_geometry(...).

    It plots:
        - canonical Intermediate Circle of radius R,
        - optional canonical swept circle of radius S,
        - optional shifted Intermediate Circle at s = s_max,
        - optional shifted swept circle at s = s_max.
    """
    if not result["feasible"]:
        print("No circle to plot.")
        print(f"reason: {result['reason']}")
        return

    center = result["center"]
    R = result["R"]
    S = result["S"]
    rule = result["rule"]
    ex = result["ex"]
    ey = result["ey"]

    # ------------------------------------------------------------
    # Canonical Intermediate Circle: radius R
    # ------------------------------------------------------------
    circle_R = Circle(
        (center.x, center.y),
        R,
        fill=False,
        linewidth=2.0,
        label=f"Intermediate Circle R ({rule})",
    )
    ax.add_patch(circle_R)

    # Canonical swept circle: radius S = R + r
    if plot_swept_circle:
        circle_S = Circle(
            (center.x, center.y),
            S,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            label="swept circle S = R + r",
        )
        ax.add_patch(circle_S)

    # Canonical center point
    ax.plot(center.x, center.y, "o", markersize=6)
    ax.text(center.x, center.y, f"  C ({rule})", fontsize=9)

    # ------------------------------------------------------------
    # Shifted circle at s = s_max
    # ------------------------------------------------------------
    if plot_shifted_circle:
        s_max = result.get("s_max", None)
        shift_direction_world = result.get("shift_direction_world", None)

        if s_max is not None and shift_direction_world is not None:
            if s_max != float("inf"):
                shifted_x = center.x + s_max * shift_direction_world[0]
                shifted_y = center.y + s_max * shift_direction_world[1]

                # Shift path from canonical center to shifted center
                ax.plot(
                    [center.x, shifted_x],
                    [center.y, shifted_y],
                    color="red",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"shift path, s_max={s_max:.3f}",
                )

                # Shifted Intermediate Circle: radius R
                shifted_circle_R = Circle(
                    (shifted_x, shifted_y),
                    R,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.0,
                    label="shifted Intermediate Circle R",
                )
                ax.add_patch(shifted_circle_R)

                # Shifted swept circle: radius S
                if plot_swept_circle:
                    shifted_circle_S = Circle(
                        (shifted_x, shifted_y),
                        S,
                        fill=False,
                        edgecolor="red",
                        linestyle="--",
                        linewidth=1.5,
                        label="shifted swept circle S",
                    )
                    ax.add_patch(shifted_circle_S)

                # Shifted center
                ax.plot(shifted_x, shifted_y, "ro", markersize=6)
                ax.text(
                    shifted_x,
                    shifted_y,
                    f"  C shifted\n  {result.get('s_max_reason', '')}",
                    fontsize=9,
                    color="red",
                )
            else:
                print("s_max is infinite; shifted circle not plotted.")
        else:
            print("Shift data missing; shifted circle not plotted.")

    # ------------------------------------------------------------
    # Corner and local frame
    # ------------------------------------------------------------
    if corner_point is not None:
        ax.plot(corner_point[0], corner_point[1], "ko", markersize=5)
        ax.text(corner_point[0], corner_point[1], "  corner", fontsize=9)

        ax.arrow(
            corner_point[0],
            corner_point[1],
            axis_scale * ex[0],
            axis_scale * ex[1],
            head_width=0.04,
            length_includes_head=True,
        )
        ax.text(
            corner_point[0] + axis_scale * ex[0],
            corner_point[1] + axis_scale * ex[1],
            "  ex",
            fontsize=9,
        )

        ax.arrow(
            corner_point[0],
            corner_point[1],
            axis_scale * ey[0],
            axis_scale * ey[1],
            head_width=0.04,
            length_includes_head=True,
        )
        ax.text(
            corner_point[0] + axis_scale * ey[0],
            corner_point[1] + axis_scale * ey[1],
            "  ey",
            fontsize=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()


def print_result(name, result):
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"feasible     : {result['feasible']}")
    print(f"reason       : {result['reason']}")
    print(f"rule         : {result['rule']}")
    print(f"center_local : {result['center_local']}")
    print(f"center_world : {result['center']}")
    print(f"R            : {result['R']:.6f}")
    print(f"r            : {result['r']:.6f}")
    print(f"S            : {result['S']:.6f}")
    print(f"D            : {result['D']:.6f}")
    print(f"q            : {result['q']:.6f}")
    print(f"a, b         : {result['a']:.6f}, {result['b']:.6f}")
    print(f"h1, h2       : {result['h1']:.6f}, {result['h2']:.6f}")
    print(f"ex           : {result['ex']}")
    print(f"ey           : {result['ey']}")
    print()


def make_test_vehicle():
    vehicle = SimpleNamespace(
        width=0.4,
        wheelbase=0.5,
        delta_max=0.5,
    )

    vehicle.max_radius = abs(vehicle.wheelbase / tan(vehicle.delta_max))

    return vehicle


def run_manual_test(
    corridor1,
    corridor2,
    vehicle,
    name="Manual intermediate circle test",
    other_intersection_point=None,
    plot_swept_circle=True,
):
    """
    Run one manual test for a pair of corridors.

    This function:
        1. computes the turn direction,
        2. computes the relevant corner point,
        3. computes the Intermediate Circle geometry,
        4. prints debug information,
        5. plots the corridors and the selected circle.
    """
    turn_direction = corridor1.compute_relative_turn_direction(corridor2)

    if turn_direction == 0:
        raise ValueError(
            "The two corridors are aligned, so turn_direction is 0. "
            "For this manual test, provide a perpendicular pair or manually "
            "extend the script to assign a fictitious turn direction."
        )

    turn_direction = int(turn_direction)

    corner_point = get_corner_point(
        corridor1,
        corridor2,
        turn_direction,
    )

    result = compute_intermediate_circle_geometry(
        corridor1=corridor1,
        corridor2=corridor2,
        corner_point=Point(*corner_point),
        turn_direction=turn_direction,
        vehicle=vehicle,
        other_intersection_point=other_intersection_point,
    )

    print_result(name, result)

    plot_corridors([corridor1, corridor2], plot_vectors=True)
    ax = plt.gca()

    plot_intermediate_circle_geometry(
        ax=ax,
        result=result,
        corner_point=corner_point,
        plot_swept_circle=plot_swept_circle,
    )

    if other_intersection_point is not None:
        ax.plot(
            other_intersection_point.x,
            other_intersection_point.y,
            "rx",
            markersize=8,
            label="other intersection point",
        )
        ax.text(
            other_intersection_point.x,
            other_intersection_point.y,
            "  other P",
            fontsize=9,
        )

    ax.set_title(name)
    ax.legend()
    plt.show(block=True)

    return result



def main1():
    vehicle = make_test_vehicle()

    # ------------------------------------------------------------
    # USER-EDITABLE CORRIDORS
    # ------------------------------------------------------------
    corridor1 = CorridorWorld(1, 15.239999659359455, [8.499999810010195, 16.089999640360475], 1.5707963267948966)
    corridor2 = CorridorWorld(0.6, 7.269999837502837, [11.044999753125012, 21.004999530501664], 0)

    # corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
    # corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)

    # corridor1 = CorridorWorld(5, 5.129999885335565, [9.199999794363976, 10.404999767430127], -1.5707963267948966)
    # corridor2 = CorridorWorld(0.8, 7.58999983035028, [9.134999795816839, 9.869999779388309], 3.141592653589793)

    corridor1 = CorridorWorld(0.8, 6, [3, 3], 1.5707963267948966)
    corridor2 = CorridorWorld(0.9, 10, [0, 5.5], 3.142592653589793)

    other_intersection_point = None
    # Example:
    # other_intersection_point = Point(1.4, 1.2)

    run_manual_test(
        corridor1=corridor1,
        corridor2=corridor2,
        vehicle=vehicle,
        name="Manual test: corridor1 to corridor2",
        other_intersection_point=other_intersection_point,
        plot_swept_circle=True,
    )

def main2():
    case_id = 2

    if case_id == 1: 
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

    elif case_id == 2:
        corridor1 = CorridorWorld(2.699999939650297, 4.949999889358878, [2.714999939315021, 2.509999943897128], 0.0)
        corridor2 = CorridorWorld(0.7499999832361939, 3.1499999295920134, [4.814999892376363, 2.734999938867986], 1.5707963267948966)
        corridor3 = CorridorWorld(0.32999999262392476, 3.5599999204277992, [4.639999896287918, 4.14499990735203], 3.141592653589793)
        corridor4 = CorridorWorld(0.4499999899417163, 2.249999949708581, [4.194999906234443, 5.104999885894358], 1.5707963267948966)
        corridor5 = CorridorWorld(0.4299999903887506, 3.5499999206513166, [4.634999896399677, 5.194999883882701], 3.141592653589793)
        corridor6 = CorridorWorld(0.8099999818950893, 2.249999949708581, [3.264999927021563, 5.104999885894358], 1.5707963267948966)
        corridor7 = CorridorWorld(0.38999999128282026, 3.4299999233335257, [1.9549999563023448, 6.034999865107238], 3.141592653589793)
        corridor8 = CorridorWorld(0.8999999798834326, 3.8099999148398638, [2.489999944344163, 7.74499982688576], 1.5707963267948966)
        corridor9 = CorridorWorld(0.5399999879300594, 4.3199999034404755, [4.1999999061226845, 6.689999850466847], 0.0)
        corridor_list = [corridor1, corridor2, corridor3, corridor6,  corridor8, corridor9]
        start_pose = [2.61918306350708, 2.154087543487549, -0.12029518960373457]
        end_pose = [4.050821781158447, 6.627957344055176, 0.07130739522438935]
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.575, v_max=1.0, v_min=-1.0, delta_max=pi/4, delta_min=-0.5)


    elif case_id == 3: 
        corridor1 = CorridorWorld(5, 10, [0, 0], 0)
        corridor2 = CorridorWorld(0.23, 6, [3, 3], 1.5707963267948966)
        corridor3 = CorridorWorld(3, 10, [0, 5.5], 3.142592653589793)
        corridor_list = [corridor1, corridor2, corridor3]
        vehicle = Bicycle([0, 0, 0], width=0.2, length=0.2, wheelbase=0.25, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

        start_pose = compute_start_pose(corridor_list[0], vehicle, 0)
        end_pose = compute_end_pose(corridor_list[-1], vehicle, 0)

    elif case_id == 4:
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
        vehicle = Bicycle([0, 0, 0], width=0.1, length=0.1, wheelbase=0.4, v_max=1.0, v_min=-1.0, delta_max=0.5, delta_min=-0.5)

    circle_sequence = build_intermediate_circles_sequence(
        corridor_list=corridor_list,
        vehicle=vehicle,
        start_pose=start_pose,
        end_pose=end_pose,
        build_only_resolved_turns=True,
    )

    plot_corridors(corridor_list, plot_vectors=True)
    ax = plt.gca()

    plot_intermediate_circles_sequence_debug(
        ax=ax,
        intermediate_circles_sequence=circle_sequence,
        footprint_radius=vehicle.width / 2.0,
        plot_swept_circle=True,
        plot_shifted_circle=True,
        plot_corner_small_circles=True,
    )

    plt.show(block=True)


if __name__ == "__main__":
    # main1()
    main2()