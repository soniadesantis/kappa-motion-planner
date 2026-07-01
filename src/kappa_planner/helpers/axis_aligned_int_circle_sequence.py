from ..geometry import IntermediateCirclesSequence, Point, IntermediateCircle, Circle
from .corridor_sequence_validity import validate_corridor_sequence
from .corridor_geometry import get_corner_point, get_corner_point_and_intersecting_edges, point_matches_any_corner
from .intersections import compute_intersection_two_segments
from .arc_feasibility import world_to_circle_local, compute_nominal_same_turn_merged_center, compute_intermediate_circle_geometry, build_intermediate_circle_from_geometry_result
from .geometry_operations import check_point_inside_segment, project_point_onto_segment, compute_distance_two_points, compute_turn_direction_from_three_points, select_tangency_point_from_point_circle
from .plot_helpers import plot_corridors, plot_intermediate_circles_sequence_debug
from .primitives import compute_extreme_poses_arc_line

import matplotlib.pyplot as plt
import warnings 
from math import pi
import numpy as np


from math import sqrt
import numpy as np


def update_turn_direction_circles(
    failed_shifts,
    intermediate_circles,
    corridor_list,
    vehicle,
    tol=1e-9,
):
    """
    Structurally repair failed tangent shifts by rebuilding the failed circles
    with the opposite turn direction.

    The indices in failed_shifts are extended-sequence indices:
        0 -> virtual initial circle
        1 -> intermediate_circles[0]
        ...
        len(intermediate_circles) -> intermediate_circles[-1]
        len(intermediate_circles) + 1 -> virtual final circle

    :param failed_shifts: Dictionary of failed shifts.
    :param intermediate_circles: IntermediateCirclesSequence to modify.
    :param corridor_list: List of corridors.
    :param vehicle: Vehicle object.
    :param tol: Numerical tolerance.
    :return: True if at least one circle was structurally updated.
    :rtype: bool
    """

    updated_any = False

    # Work from right to left, so replacing one circle by two circles
    # does not invalidate the remaining indices.
    for extended_index in sorted(failed_shifts.keys(), reverse=True):

        # Skip virtual initial circle.
        if extended_index == 0:
            continue

        # Skip virtual final circle.
        if extended_index == len(intermediate_circles) + 1:
            continue

        intermediate_index = extended_index - 1
        circle = intermediate_circles[intermediate_index]

        is_merged = (
            getattr(circle, "merged", False)
            or getattr(circle, "is_merged", False)
        )

        # ============================================================
        # Case 1: failed circle is merged
        # ============================================================
        if is_merged:
            print(f"Updating failed merged circle at extended index {extended_index}.")

            merged_from = getattr(circle, "merged_from", None)

            if merged_from is None:
                print("Cannot update merged circle: missing merged_from.")
                continue

            shift_result = failed_shifts[extended_index]["shift_result"]
            attempted_center = shift_result.get("center", None)

            if attempted_center is None:
                print("Cannot update merged circle: missing attempted shifted center.")
                continue

            new_source_circles = []

            for source_index, source_circle in enumerate(merged_from):
                ok, reason, center_local = shifted_center_is_admissible_for_single_circle(
                    source_circle,
                    attempted_center,
                    tol=tol,
                )

                if ok:
                    print(
                        f"  merged source {source_index} was admissible; "
                        "keeping original source circle."
                    )
                    new_source_circles.append(source_circle)
                    continue

                print(
                    f"  merged source {source_index} failed with reason "
                    f"{reason}; rebuilding with opposite turn."
                )

                corridor_index = source_circle.corridor_index_start
                opposite_turn = -source_circle.turn_direction

                rebuilt_circle = build_circle_from_two_corridors(
                    corridor1=corridor_list[corridor_index],
                    corridor2=corridor_list[corridor_index + 1],
                    tau=opposite_turn,
                    vehicle=vehicle,
                    i=corridor_index,
                )

                new_source_circles.append(rebuilt_circle)

            # Try to merge the two repaired/source circles if possible.
            replacement_circles = new_source_circles

            if len(new_source_circles) == 2:
                circle1 = new_source_circles[0]
                circle2 = new_source_circles[1]

                same_turn = circle1.turn_direction == circle2.turn_direction
                center_distance = compute_distance_two_points(
                    circle1.center,
                    circle2.center,
                )

                nominal_overlap = center_distance < (
                    circle1.radius + circle2.radius - tol
                )

                if same_turn and nominal_overlap:
                    try:
                        merged_circle = build_merged_circle_for_same_turn_pair(
                            circle1=circle1,
                            circle2=circle2,
                            corridor_list=corridor_list,
                            vehicle=vehicle,
                        )

                        replacement_circles = [merged_circle]

                        print(
                            "  rebuilt source circles were merged again "
                            "after opposite-turn update."
                        )

                    except Exception as error:
                        print(
                            "  merge attempt failed; keeping the two "
                            f"separate source circles. Reason: {error}"
                        )

            # Replace the merged circle by either:
            #   - one merged circle, if merge succeeded
            #   - two separate source circles otherwise
            intermediate_circles.remove_at(intermediate_index)

            for replacement_circle in reversed(replacement_circles):
                intermediate_circles.insert(
                    intermediate_index,
                    replacement_circle,
                )

            updated_any = True
            continue

        # ============================================================
        # Case 2: failed circle is not merged
        # ============================================================
        print(f"Updating failed circle at extended index {extended_index}.")

        corridor_index = circle.corridor_index_start
        opposite_turn = -circle.turn_direction

        rebuilt_circle = build_circle_from_two_corridors(
            corridor1=corridor_list[corridor_index],
            corridor2=corridor_list[corridor_index + 1],
            tau=opposite_turn,
            vehicle=vehicle,
            i=corridor_index,
        )

        intermediate_circles.remove_at(intermediate_index)
        intermediate_circles.insert(intermediate_index, rebuilt_circle)

        updated_any = True

    return updated_any


def update_intermediate_circle_centers_from_extended_sequence(
    intermediate_circles,
    new_extended_centers,
):
    """
    Update only the real intermediate circles.

    :param intermediate_circles: IntermediateCirclesSequence or list of IntermediateCircle.
    :param new_extended_centers: centers computed on
        [initial_circle] + intermediate_circles + [final_circle].
    """
    if len(new_extended_centers) != len(intermediate_circles) + 2:
        raise ValueError(
            "new_extended_centers must have length len(intermediate_circles) + 2"
        )

    for i, circle in enumerate(intermediate_circles):
        new_center = new_extended_centers[i + 1]

        circle.center = Point(new_center.x, new_center.y)

        if hasattr(circle, "xc"):
            circle.xc = new_center.x
        if hasattr(circle, "yc"):
            circle.yc = new_center.y
            

def update_circle_sequence_centers(circle_sequence, new_centers):
    """
    Update the centers of the circles in a sequence.

    :param circle_sequence: Sequence of IntermediateCircle objects.
    :param new_centers: List of Point objects, one for each circle.
    :return: None.
    """
    if len(circle_sequence) != len(new_centers):
        raise ValueError("circle_sequence and new_centers must have the same length")

    for circle, new_center in zip(circle_sequence, new_centers):
        circle.center = Point(new_center.x, new_center.y)

        # Keep these only if your Circle class uses them.
        if hasattr(circle, "xc"):
            circle.xc = new_center.x
        if hasattr(circle, "yc"):
            circle.yc = new_center.y


def shifted_center_is_admissible(circle, shifted_center, tol=1e-9):
    if getattr(circle, "is_merged", False) or getattr(circle, "merged", False):
        merged_from = getattr(circle, "merged_from", None)

        if merged_from is None:
            return False, "merged_circle_missing_merged_from", None
        
        failed_sources = []

        for source_index, source_circle in enumerate(merged_from):
            ok, reason, center_local = shifted_center_is_admissible_for_single_circle(
                source_circle,
                shifted_center,
                tol=tol,
            )

            if not ok:
                failed_sources.append({
                    "source_index": source_index,
                    "reason": reason,
                    "center_local": center_local,
                    "source_circle": source_circle,
                })

        if failed_sources:
            return (
                False,
                "merged_circle_source_constraint_failed",
                failed_sources,
            )
        return True, "ok", None

    return shifted_center_is_admissible_for_single_circle(
        circle,
        shifted_center,
        tol=tol,
    )

def print_failed_tangent_shift_summary(failed_shifts, tol=1e-9):
    """
    Print a compact summary of failed tangent shifts.

    :param failed_shifts: Dictionary returned by solve_tangent_intersections_blocks_centers.
    :param tol: Numerical tolerance.
    :return: None.
    """
    for failed_index, failure in failed_shifts.items():
        circle = failure["circle"]
        shift_result = failure["shift_result"]
        attempted_center = shift_result.get("center", None)

        is_merged = (
            getattr(circle, "merged", False)
            or getattr(circle, "is_merged", False)
        )

        print("\nFAILED TANGENT SHIFT")
        print("extended index:", failed_index)
        print("block:", failure.get("block", None))
        print("reason:", failure.get("reason", None))
        print("is merged:", is_merged)

        if not is_merged:
            continue

        merged_from = getattr(circle, "merged_from", None)

        if merged_from is None:
            print("merged_from: missing")
            continue

        if attempted_center is None:
            print("attempted shifted center: missing")
            continue

        for source_index, source_circle in enumerate(merged_from):
            ok, reason, center_local = shifted_center_is_admissible_for_single_circle(
                source_circle,
                attempted_center,
                tol=tol,
            )

            print(
                f"merged source {source_index}: "
                f"ok={ok}, reason={reason}, center_local={center_local}"
            )
            

def shifted_center_is_admissible_for_single_circle(circle, shifted_center, tol=1e-9):
    center_local = world_to_circle_local(circle, shifted_center)
    x, y = center_local

    if x < circle.lower_bound_x_unclamped - tol:
        return False, "x_below_lower_bound", center_local

    if y < circle.lower_bound_y_unclamped - tol:
        return False, "y_below_lower_bound", center_local

    # if circle.admissible_radius is not None:
    #     if x * x + y * y > circle.admissible_radius**2 + tol:
    #         return False, "outside_admissible_disk", center_local

    if circle.swept_radius is not None:
        for forbidden_point in circle.forbidden_points:
            distance = compute_distance_two_points(shifted_center, forbidden_point)

            if distance < circle.swept_radius - tol:
                return False, "too_close_to_forbidden_point", center_local

    return True, "ok", center_local


def circle_halfplane_clearance(circle, x1, y1, x2, y2, tol=1e-9):
    signed_distance, n_right = signed_distance_to_directed_line(
        circle.center,
        x1,
        y1,
        x2,
        y2,
        tol=tol,
    )

    # Keep this convention consistent with the one that works in your plots.
    side_clearance = signed_distance + circle.turn_direction * circle.radius
    is_bad = circle.turn_direction * side_clearance < -tol

    return is_bad, signed_distance, n_right, side_clearance


# def shifted_center_to_reference_tangent(circle, x1, y1, x2, y2, tol=1e-9):
#     signed_distance, n_right = signed_distance_to_directed_line(
#         circle.center,
#         x1,
#         y1,
#         x2,
#         y2,
#         tol=tol,
#     )

#     target_distance = -circle.turn_direction * circle.radius
#     delta = target_distance - signed_distance

#     center_np = np.array([circle.center.x, circle.center.y], dtype=float)
#     new_center_np = center_np + delta * n_right
#     new_center = Point(new_center_np[0], new_center_np[1])

#     is_admissible, reason, center_local = shifted_center_is_admissible(
#         circle,
#         new_center,
#         tol=tol,
#     )

#     return {
#         "feasible": is_admissible,
#         "reason": reason,
#         "center": new_center,
#         "center_local": center_local,
#         "signed_distance": signed_distance,
#         "target_distance": target_distance,
#         "delta": delta,
#     }


def shifted_center_to_reference_tangent(circle, x1, y1, x2, y2, tol=1e-9):
    signed_distance, n_right = signed_distance_to_directed_line(
        circle.center,
        x1,
        y1,
        x2,
        y2,
        tol=tol,
    )

    target_distance = -circle.turn_direction * circle.radius
    delta = target_distance - signed_distance

    center_np = np.array([circle.center.x, circle.center.y], dtype=float)

    # First candidate: tangent to the infinite reference line.
    new_center_np = center_np + delta * n_right

    # Tangency point on the infinite line.
    tangent_point_np = new_center_np - target_distance * n_right
    tangent_point = Point(tangent_point_np[0], tangent_point_np[1])

    segment_start = Point(x1, y1)
    segment_end = Point(x2, y2)

    tangent_point_on_segment = check_point_inside_segment(
        A=(x1, y1),
        B=(x2, y2),
        P=(tangent_point.x, tangent_point.y),
        tol=tol,
    )

    used_projected_tangent_point = False

    if not tangent_point_on_segment:
        # Clamp/project the tangency point to the finite segment.
        tangent_point = project_point_onto_segment(
            point=tangent_point,
            segment_start=segment_start,
            segment_end=segment_end,
        )

        tangent_point_np = np.array(
            [tangent_point.x, tangent_point.y],
            dtype=float,
        )

        # Rebuild center from the projected tangent point.
        new_center_np = tangent_point_np + target_distance * n_right
        used_projected_tangent_point = True

    new_center = Point(new_center_np[0], new_center_np[1])

    is_admissible, reason, center_local = shifted_center_is_admissible(
        circle,
        new_center,
        tol=tol,
    )

    print(
        f"Circle {circle.index} shifted to tangent: {new_center.x}, {new_center.y}"
    )

    return {
        "feasible": is_admissible,
        "reason": reason,
        "center": new_center,
        "center_local": center_local,
        "signed_distance": signed_distance,
        "target_distance": target_distance,
        "delta": delta,
        "tangent_point": tangent_point,
        "tangent_point_on_segment": tangent_point_on_segment,
        "used_projected_tangent_point": used_projected_tangent_point,
    }


def solve_tangent_block_centers(
    first_index,
    last_index,
    circle_sequence,
    corrected_centers,
    failed_shifts,
    tol=1e-9,
):
    
    # The block is circle_sequence[first_index : last_index + 1] 
    # If the block has only two circles, there are no interior circles to check
    if last_index - first_index <= 1:
        return

    # If the block has more than two circles, build the tangent between
    # the first and last circles, and check the intermediate circles.
    first_circle = circle_sequence[first_index]
    last_circle = circle_sequence[last_index]

    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
        first_circle.center.x,
        first_circle.center.y,
        last_circle.center.x,
        last_circle.center.y,
        first_circle.turn_direction,
        last_circle.turn_direction,
        first_circle.radius,
    )

    good_indices = []

    for circle_index in range(first_index + 1, last_index):
        circle = circle_sequence[circle_index]

        # A circle is good if it actively pushes the tangent along the correct direction
        is_bad, signed_distance, n_right, side_clearance = circle_halfplane_clearance(
            circle,
            x1,
            y1,
            x2,
            y2,
            tol=tol,
        )

        if not is_bad:
            good_indices.append(circle_index)

    # Split the block in case one circle that actively pushes the tangent is found
    if good_indices:
        split_index = good_indices[0]

        solve_tangent_block_centers(
            first_index,
            split_index,
            circle_sequence,
            corrected_centers,
            failed_shifts,
            tol=tol,
        )

        solve_tangent_block_centers(
            split_index,
            last_index,
            circle_sequence,
            corrected_centers,
            failed_shifts,
            tol=tol,
        )

        return

    # If no good circles are found, it is time to resolve the intersection by shiftin
    # If no good circles are found, try to resolve the intersection by shifting.
    # The block correction is accepted only if all required shifts are feasible.
    block_corrected_centers = {} # tentative successful shifts for this block
    # temporary because the block is accepted only if all required shifts are feasible.
    block_failed_shifts = {} # failed shifts for this block

    for circle_index in range(first_index + 1, last_index):
        circle = circle_sequence[circle_index]

        # Shift the circle until it is tangent to the reference line
        # and check if the shifted center is admissible.
        shift_result = shifted_center_to_reference_tangent(
            circle,
            x1,
            y1,
            x2,
            y2,
            tol=tol,
        )

        if shift_result["feasible"]:
            block_corrected_centers[circle_index] = shift_result["center"]
        else:
            block_failed_shifts[circle_index] = {
                "reason": shift_result["reason"],
                "shift_result": shift_result,
                "block": (first_index, last_index),
                "circle": circle,
            }

    # If at least one circle failed, reject all shifts in this block.
    # The failed circles will be structurally repaired later, and the whole
    # tangent-intersection check will be run again.
    if block_failed_shifts:
        # save the failure information from local to global failed_shifts
        for circle_index, failure in block_failed_shifts.items():
            failed_shifts[circle_index] = failure

            print(
                f"Circle {circle_index} cannot be shifted: "
                f"{failure['reason']}"
            )

        print(
            f"Block ({first_index}, {last_index}) rejected. "
            "All tentative shifts in this block are discarded."
        )

        return

    # If no failures occurred, accept all shifts in this block.
    for circle_index, new_center in block_corrected_centers.items():
        # save the successful shift information from local to global corrected_centers
        corrected_centers[circle_index] = new_center
        print(f"Circle {circle_index} shifted successfully.")


def solve_tangent_intersections_blocks_centers(blocks, circle_sequence, tol=1e-9):
    corrected_centers = {}
    failed_shifts = {}

    # For each block, solve the intersection for each circle by
    # 1- Shifting
    # 2- Not updating anything 
    # 3- Reporting failure
    for first_index, last_index in blocks:
        solve_tangent_block_centers(
            first_index,
            last_index,
            circle_sequence,
            corrected_centers,
            failed_shifts,
            tol=tol,
        )

    new_centers = []

    for index, circle in enumerate(circle_sequence):
        if index in corrected_centers:
            new_centers.append(corrected_centers[index])
        else:
            new_centers.append(circle.center)

    return new_centers, corrected_centers, failed_shifts
     

def signed_distance_to_directed_line(point, x1, y1, x2, y2, tol=1e-9):
    vx = x2 - x1
    vy = y2 - y1

    wx = point.x - x1
    wy = point.y - y1

    line_length = np.hypot(vx, vy)
    if line_length <= tol:
        raise ValueError("Degenerate reference tangent line.")
    
    # Unit normal pointing to the right of the directed line.
    n_right = np.array([vy, -vx], dtype=float) / line_length

    # Positive means point is to the right of the directed line.
    return -(vx * wy - vy * wx) / line_length, n_right


# def solve_tangent_intersections_blocks(blocks, circle_sequence, corridor_list, tol=1e-9):
#     for first_index, last_index in blocks:
#         first_circle = circle_sequence[first_index]
#         last_circle = circle_sequence[last_index]

#         x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
#             first_circle.center.x,
#             first_circle.center.y,
#             last_circle.center.x,
#             last_circle.center.y,
#             first_circle.turn_direction,
#             last_circle.turn_direction,
#             first_circle.radius,
#         )

#         plot_corridors(corridor_list)
#         plt.plot([x1, x2], [y1, y2], 'k--', linewidth=1.5)
#         plt.plot(x1, y1, 'go', markersize=5, label=f'start tangent')
#         plt.plot(x2, y2, 'yo', markersize=5, label=f'end tangent')
#         plt.plot(first_circle.center.x, first_circle.center.y, 'ro', markersize=5, label=f'first circle')
#         plt.plot(last_circle.center.x, last_circle.center.y, 'bo', markersize=5, label=f'last circle')
#         angle_array = np.linspace(0, 2*np.pi, 100)
#         plt.plot(first_circle.center.x + first_circle.radius * np.cos(angle_array),
#                 first_circle.center.y + first_circle.radius * np.sin(angle_array), 'r--', linewidth=1.5, label=f'first circle')
#         plt.plot(last_circle.center.x + last_circle.radius * np.cos(angle_array),
#                 last_circle.center.y + last_circle.radius * np.sin(angle_array), 'b--', linewidth=1.5, label=f'last circle')

#         for circle_index in range(first_index + 1, last_index):
#             circle = circle_sequence[circle_index]

#             signed_distance, n_right= signed_distance_to_directed_line(
#                 circle.center,
#                 x1,
#                 y1,
#                 x2,
#                 y2,
#                 tol=tol,
#             )

#             side_clearance =  signed_distance + circle.turn_direction * circle.radius

#             is_bad = circle.turn_direction * side_clearance < -tol

#             if is_bad:
#                 print(
#                     f"Circle {circle_index} violates block tangent: "
#                     f"signed_distance={signed_distance:.4f}, "
#                     f"side_clearance={side_clearance:.4f}"
#                 )
#                 target_distance = circle.turn_direction * circle.radius
#                 delta = target_distance - signed_distance

#                 new_center_np = np.array([circle.center.x, circle.center.y]) + delta * n_right

#                 plt.plot(new_center_np[0], new_center_np[1], 'mo', markersize=5, label=f'corrected circle {circle_index}')
#                 plt.plot(new_center_np[0] + circle.radius * np.cos(angle_array),
#                         new_center_np[1] + circle.radius * np.sin(angle_array), 'm--', linewidth=1.5, label=f'corrected circle {circle_index}')
                
#             plt.plot(circle.center.x, circle.center.y, 'ro', markersize=5)
#             plt.plot(circle.center.x + circle.radius * np.cos(angle_array),
#                     circle.center.y + circle.radius * np.sin(angle_array), 'r--', linewidth=1.5)
#     plt.legend()
#     plt.show(block=True)

            
def detect_tangent_intersections_blocks(circle_sequence, tol=1e-9):
    flags = []

    for index in range(len(circle_sequence) - 2):
        circle1 = circle_sequence[index]
        circle2 = circle_sequence[index + 1]
        circle3 = circle_sequence[index + 2]

        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
            circle1.center.x,
            circle1.center.y,
            circle3.center.x,
            circle3.center.y,
            circle1.turn_direction,
            circle3.turn_direction,
            circle1.radius,
        )

        try:
            is_bad, signed_distance, n_right, side_clearance = circle_halfplane_clearance(
                circle2,
                x1,
                y1,
                x2,
                y2,
                tol=tol,
            )
        except ValueError:
            is_bad = True

        flags.append(1 if is_bad else 0)

    blocks = []
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            blocks.append((start, i + 1))
            start = None

    if start is not None:
        blocks.append((start, len(flags) + 1))

    return flags, blocks


def compute_merged_circle_corner_point(
    merged_center,
    circle1,
    circle2,
    corridor_list,
):
    """
    Compute the corner/reference point for a merged intermediate circle.

    The point is the orthogonal projection of the merged circle center onto
    the edge of the middle corridor shared by the two original corner points.

    :param merged_center: Center of the merged circle.
    :type merged_center: Point

    :param circle1: First circle in the merge.
    :type circle1: IntermediateCircle

    :param circle2: Second circle in the merge.
    :type circle2: IntermediateCircle

    :param corridor_list: List of corridors.
    :type corridor_list: list[CorridorWorld]

    :return: Projected corner/reference point.
    :rtype: Point
    """
    middle_corridor_index = circle1.corridor_index_end
    middle_corridor = corridor_list[middle_corridor_index]

    middle_edge_from_circle1 = circle1.edge_pair[1]
    middle_edge_from_circle2 = circle2.edge_pair[0]

    if middle_edge_from_circle1 != middle_edge_from_circle2:
        warnings.warn(
            "Cannot compute merged circle corner point: "
            "the two circles do not use the same middle-corridor edge.",
            RuntimeWarning
        )

    segment_start, segment_end = middle_corridor.get_edge_segment(
        middle_edge_from_circle1
    )

    return project_point_onto_segment(
        point=merged_center,
        segment_start=Point(segment_start[0], segment_start[1]),
        segment_end=Point(segment_end[0], segment_end[1]),
    )


def circle_contains_small_circle(big_circle, small_circle, tol=1e-9):
    """
    Check whether big_circle fully contains small_circle.

    :param big_circle: Circle that should contain the other circle.
    :type big_circle: Circle

    :param small_circle: Circle that should be contained.
    :type small_circle: Circle

    :param tol: Numerical tolerance.
    :type tol: float

    :return: True if small_circle is fully inside big_circle.
    :rtype: bool
    """
    center_distance = compute_distance_two_points(
        big_circle.center,
        small_circle.center,
    )

    return center_distance + small_circle.radius <= big_circle.radius + tol


def compute_big_circle_tangent_to_small_circle_and_centerline(
    active_small_circle,
    other_small_circle,
    centerline,
    radius,
    vehicle,
    reference_center=None,
    tol=1e-6,
):
    """
    Compute candidate big circles of fixed radius R that are:

        - internally tangent to active_small_circle;
        - clearance-tangent to the given centerline;
        - containing other_small_circle.

    The centerline is represented as a normalized line equation:

        a*x + b*y + c = 0

    The big circle has radius R.
    The small circles have radius r = vehicle.width / 2.

    Clearance tangency to the centerline means the swept circle of radius
    R + r is tangent to the centerline.

    :param active_small_circle: Small circle that must be internally tangent.
    :type active_small_circle: Circle

    :param other_small_circle: Other small circle that must be contained.
    :type other_small_circle: Circle

    :param centerline: Normalized centerline equation (a, b, c).
    :type centerline: tuple[float, float, float]

    :param radius: Big circle radius R.
    :type radius: float

    :param vehicle: Vehicle object.
    :type vehicle: Vehicle

    :param reference_center: Optional reference center used to select the closest candidate.
    :type reference_center: Point or None

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Best candidate big Circle, or None.
    :rtype: Circle or None
    """
    R = radius
    r = vehicle.width / 2.0

    R_inner = R - r      # distance big center -> active small center
    R_check = R + r      # swept radius tangent to centerline

    if R_inner <= tol:
        return None

    p = active_small_circle.center

    a, b, c = centerline

    norm = sqrt(a * a + b * b)

    if norm <= tol:
        return None

    # Normalize defensively.
    a = a / norm
    b = b / norm
    c = c / norm

    # Unit normal to centerline.
    nx = a
    ny = b

    # Unit tangent to centerline.
    tx = -b
    ty = a

    candidates = []

    # The big center must lie on one of the two parallel lines:
    #
    #     a*x + b*y + c = +R_check
    #     a*x + b*y + c = -R_check
    #
    # Equivalently:
    #
    #     a*x + b*y + (c - sigma * R_check) = 0
    #
    for sigma in (+1, -1):
        target_signed_distance = sigma * R_check

        # Project active small-circle center p onto the target parallel line.
        #
        # Current signed distance of p from original centerline:
        #     d_p = a*p.x + b*p.y + c
        #
        # Need q0 such that:
        #     a*q0.x + b*q0.y + c = target_signed_distance
        #
        d_p = a * p.x + b * p.y + c
        shift_to_parallel = target_signed_distance - d_p

        q0x = p.x + shift_to_parallel * nx
        q0y = p.y + shift_to_parallel * ny

        # Candidate center o(alpha) lies on the parallel line:
        #
        #     o(alpha) = q0 + alpha * t
        #
        # and must also satisfy:
        #
        #     ||o(alpha) - p|| = R_inner
        #
        wx = q0x - p.x
        wy = q0y - p.y

        # Since t is unit:
        #
        # alpha^2 + 2 alpha (w dot t) + ||w||^2 - R_inner^2 = 0
        B = 2.0 * (wx * tx + wy * ty)
        C = wx * wx + wy * wy - R_inner * R_inner

        disc = B * B - 4.0 * C

        if disc < -tol:
            continue

        if disc < 0.0:
            disc = 0.0

        sqrt_disc = sqrt(disc)

        for alpha in (
            (-B + sqrt_disc) / 2.0,
            (-B - sqrt_disc) / 2.0,
        ):
            ox = q0x + alpha * tx
            oy = q0y + alpha * ty

            candidate = Circle(
                center=Point(ox, oy),
                radius=R,
            )

            if not circle_contains_small_circle(
                candidate,
                other_small_circle,
                tol=tol,
            ):
                continue

            candidates.append(candidate)

    if not candidates:
        return None

    if reference_center is None:
        return candidates[0]

    candidates.sort(
        key=lambda circle: compute_distance_two_points(
            circle.center,
            reference_center,
        )
    )

    return candidates[0]


def resolve_same_turn_nominal_overlap_blocks(
    intermediate_circles_sequence,
    corridor_list,
    vehicle,
    overlap_blocks,
):
    """
    Resolve same-turn nominal overlap blocks.

    For now:
        - blocks of length 2 are solved by trying to build one merged circle;
        - blocks of length 3 are detected but skipped.

    :param intermediate_circles_sequence: Current sequence of IntermediateCircle objects.
    :type intermediate_circles_sequence: IntermediateCirclesSequence

    :param corridor_list: Full corridor list.
    :type corridor_list: list[CorridorWorld]

    :param vehicle: Vehicle object.
    :type vehicle: Vehicle

    :param overlap_blocks: Blocks returned by detect_same_turn_nominal_overlap_blocks.
    :type overlap_blocks: list[list[int]]

    :return: Updated IntermediateCirclesSequence.
    :rtype: IntermediateCirclesSequence
    """

    # Process from right to left so that replacing [i, i+1]
    # does not invalidate the indices of earlier blocks.
    for block in reversed(overlap_blocks):

        if len(block) == 3:
            print(
                f"Skipping same-turn overlap triplet {block}. "
                "Triplet merge is not implemented yet."
            )
            continue

        if len(block) != 2:
            print(f"Skipping unsupported overlap block {block}.")
            continue

        index1, index2 = block

        circle1 = intermediate_circles_sequence[index1]
        circle2 = intermediate_circles_sequence[index2]

        if circle1.turn_direction != circle2.turn_direction:
            print(
                f"Skipping block {block}: circles do not have the same turn direction."
            )
            continue

        merged_circle = build_merged_circle_for_same_turn_pair(
            circle1=circle1,
            circle2=circle2,
            corridor_list=corridor_list,
            vehicle=vehicle,
        )

        if merged_circle is None:
            print(f"Could not merge same-turn overlap pair {block}.")
            continue

        print(f"Merged same-turn overlap pair {block}.")

        replace_two_circles_with_one(
            intermediate_circles_sequence=intermediate_circles_sequence,
            index=index1,
            merged_circle=merged_circle,
        )

    reindex_intermediate_circles_sequence(intermediate_circles_sequence)

    return intermediate_circles_sequence


def replace_two_circles_with_one(
    intermediate_circles_sequence,
    index,
    merged_circle,
):
    """
    Replace circles[index] and circles[index + 1] with merged_circle.

    :param intermediate_circles_sequence: Circle sequence.
    :type intermediate_circles_sequence: IntermediateCirclesSequence

    :param index: Index of the first circle to replace.
    :type index: int

    :param merged_circle: New merged IntermediateCircle.
    :type merged_circle: IntermediateCircle
    """
    intermediate_circles_sequence[index] = merged_circle
    intermediate_circles_sequence.remove_at(index + 1)


def reindex_intermediate_circles_sequence(intermediate_circles_sequence):
    """
    Reassign sequence indices after merging/removing circles.
    """
    for i, circle in enumerate(intermediate_circles_sequence):
        circle.index = i


def build_merged_circle_for_same_turn_pair(
    circle1,
    circle2,
    corridor_list,
    vehicle,
    tol=1e-9,
):
    """
    Try to merge two same-turn overlapping IntermediateCircles into one.

    Current version:
        - builds the nominal merged candidate;
        - checks whether it intersects the first/third corridor centerlines;
        - accepts only if the nominal candidate is valid.

    Later:
        - if the nominal candidate crosses one or both centerlines,
          try to shift it or make it tangent to the crossed centerline(s).
    """

    if circle1.turn_direction != circle2.turn_direction:
        return None

    R = vehicle.max_radius
    r = vehicle.width / 2.0
    S = R + r

    # ------------------------------------------------------------
    # 1. Get outer centerlines
    # ------------------------------------------------------------
    first_line, third_line = get_outer_centerlines_for_two_circle_merge(
        circle1=circle1,
        circle2=circle2,
        corridor_list=corridor_list,
    )

    # ------------------------------------------------------------
    # 2. Compute nominal merged candidate
    # ------------------------------------------------------------
    candidate = compute_nominal_same_turn_merged_center(
        corner_point1=circle1.corner_point,
        corner_point2=circle2.corner_point,
        R=R,
        r=r,
        turn_direction=circle1.turn_direction,
        tol=tol,
    )

    if not candidate["feasible"]:
        print(f"Nominal merge infeasible: {candidate['reason']}")
        return None

    center = candidate["center"]

    # ------------------------------------------------------------
    # 3. Check outer centerline intersections
    # ------------------------------------------------------------
    centerline_status = merged_candidate_centerline_intersection_status(
        center=center,
        radius=R,
        first_line=first_line,
        third_line=third_line,
        tol=tol,
    )

    if centerline_status["intersects"]:
        crossed = centerline_status["crossed_centerlines"]

        if len(crossed) == 1:
            crossed_name = crossed[0]

            small_circle1 = Circle(
                center=circle1.corner_point,
                radius=r,
            )

            small_circle2 = Circle(
                center=circle2.corner_point,
                radius=r,
            )

            if crossed_name == "first":
                active_small_circle = small_circle1
                other_small_circle = small_circle2
                violated_line = first_line

            elif crossed_name == "third":
                active_small_circle = small_circle2
                other_small_circle = small_circle1
                violated_line = third_line

            else:
                raise ValueError(f"Unexpected crossed centerline: {crossed_name}")

            corrected_circle = compute_big_circle_tangent_to_small_circle_and_centerline(
                active_small_circle=active_small_circle,
                other_small_circle=other_small_circle,
                centerline=violated_line,
                radius=R,
                vehicle=vehicle,
                reference_center=center,
                tol=tol,
            )

            if corrected_circle is None:
                return None

            center = corrected_circle.center

            centerline_status = merged_candidate_centerline_intersection_status(
                center=center,
                radius=R + r,   # IMPORTANT if this status checks swept-circle line crossing
                first_line=first_line,
                third_line=third_line,
                tol=tol,
            )

    # ------------------------------------------------------------
    # 4. Build IntermediateCircle from accepted candidate
    # ------------------------------------------------------------
    corner_point = compute_merged_circle_corner_point(
        merged_center=center,
        circle1=circle1,
        circle2=circle2,
        corridor_list=corridor_list,
    )
    middle_corridor = corridor_list[circle1.corridor_index_end]
    distance_to_wall = compute_distance_two_points(
        center,
        corner_point,
    )

    s_max = max(0,distance_to_wall + middle_corridor.width - S)

    merged_circle = build_merged_intermediate_circle_from_pair_candidate(
        circle1=circle1,
        circle2=circle2,
        center=center,
        radius=R,
        corner_point=corner_point,
        s_max=s_max,
        candidate=candidate,
        centerline_status=centerline_status,
    )

    return merged_circle


def build_merged_intermediate_circle_from_pair_candidate(
    circle1,
    circle2,
    center,
    radius,
    corner_point,
    s_max,
    candidate=None,
    centerline_status=None,
):
    """
    Build a merged IntermediateCircle from a same-turn pair candidate.
    """
    
    forbidden_points = (
        circle1.forbidden_points + circle2.forbidden_points
    )

    
    merged_circle = IntermediateCircle(
        center=center,
        radius=radius,
        corner_point=corner_point,
        turn_direction=circle1.turn_direction,
        index=circle1.index,
        s_max=s_max,
        edge_pair=None,
        door_point=None,
        door_type=None,
        start_angle_arc=None,
        rho=None,
        merged=True,
        forbidden_points=forbidden_points
    )
    merged_circle.shift_direction_world = compute_shift_direction_from_center_to_point(
        center=merged_circle.center,
        reference_point=merged_circle.corner_point,
    )

    print("MERGE DEBUG")
    print("circle1 corner:", circle1.corner_point.x, circle1.corner_point.y)
    print("circle2 corner:", circle2.corner_point.x, circle2.corner_point.y)
    print("merged center:", center.x, center.y)
    print("merged projected corner:", corner_point.x, corner_point.y)
    print("middle edge c1:", circle1.edge_pair[1])
    print("middle edge c2:", circle2.edge_pair[0])
    print("shift direction:", merged_circle.shift_direction_world)
    print("s_max:", merged_circle.s_max)

    merged_circle.s_max_reason = "middle_corridor_width_bound"

    merged_circle.is_merged = True
    merged_circle.merged_from = (circle1, circle2)
    merged_circle.merged_corner_points = (circle1.corner_point, circle2.corner_point)

    merged_circle.corridor_index_start = circle1.corridor_index_start
    merged_circle.corridor_index_end = circle2.corridor_index_end

    merged_circle.nominal_merge_candidate = candidate
    merged_circle.centerline_status = centerline_status

    return merged_circle


def compute_shift_direction_from_center_to_point(center, reference_point, tol=1e-9):
    """
    Compute the unit shift direction from a center point toward a reference point.

    :param center: Initial circle center.
    :type center: Point

    :param reference_point: Point toward which the circle should shift.
    :type reference_point: Point

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Unit shift direction as [dx, dy].
    :rtype: numpy.ndarray
    """
    direction = np.array(
        [
            reference_point.x - center.x,
            reference_point.y - center.y,
        ],
        dtype=float,
    )

    norm = np.linalg.norm(direction)

    if norm <= tol:
        raise ValueError(
            "Cannot compute shift direction: center and reference point coincide."
        )

    return direction / norm


def compute_merged_circle_s_max(
    merged_circle,
    middle_corridor,
    vehicle,
):
    """
    Compute s_max for a merged circle.

    :param merged_circle: Merged intermediate circle.
    :param middle_corridor: Corridor between the two merged turns.
    :param vehicle: Vehicle object.

    :return: s_max.
    """
    S = vehicle.max_radius + vehicle.width / 2.0

    distance_to_wall = compute_distance_two_points(
        merged_circle.center,
        merged_circle.corner_point,
    )

    s_max = distance_to_wall + middle_corridor.width - S

    return max(0.0, s_max)


def corridor_centerline_equation(corridor, transverse=False):
    """
    Return the centerline of a corridor as a normalized line equation:

        a*x + b*y + c = 0

    If transverse=False, the line is the normal longitudinal centerline.
    If transverse=True, the line is the transverse centerline through the
    corridor center, perpendicular to the corridor direction.

    :param corridor: CorridorWorld object.
    :type corridor: CorridorWorld

    :param transverse: Whether to use the transverse centerline.
    :type transverse: bool

    :return: Line coefficients (a, b, c), with sqrt(a^2 + b^2) = 1.
    :rtype: tuple[float, float, float]
    """
    cx, cy = corridor.center[0], corridor.center[1]

    if not transverse:
        # Longitudinal centerline direction:
        # d = (cos(theta), sin(theta)).
        # A normal to this line is:
        # n = (-sin(theta), cos(theta)).
        a = -np.sin(corridor.tilt)
        b = np.cos(corridor.tilt)
    else:
        # Transverse centerline direction:
        # d = (-sin(theta), cos(theta)).
        # A normal to this line is:
        # n = (cos(theta), sin(theta)).
        a = np.cos(corridor.tilt)
        b = np.sin(corridor.tilt)

    c = -(a * cx + b * cy)

    return a, b, c


def relevant_centerline_equation_from_edge(corridor, edge_index):
    """
    Return the relevant centerline equation for a corridor and edge.

    Side edges use the longitudinal centerline.
    End edges use the transverse centerline.

    :param corridor: CorridorWorld object.
    :type corridor: CorridorWorld

    :param edge_index: Edge index involved in the transition.
    :type edge_index: int

    :return: Line coefficients (a, b, c).
    :rtype: tuple[float, float, float]
    """
    if edge_index in (corridor.RGT, corridor.LFT):
        return corridor_centerline_equation(
            corridor,
            transverse=False,
        )

    if edge_index in (corridor.FWD, corridor.BCK):
        return corridor_centerline_equation(
            corridor,
            transverse=True,
        )

    raise ValueError(f"Unknown corridor edge index: {edge_index}")


def get_outer_centerlines_for_two_circle_merge(
    circle1,
    circle2,
    corridor_list,
):
    """
    Return the relevant centerlines of the first and third corridors for a
    two-circle merge.

    The returned lines are normalized equations:

        a*x + b*y + c = 0

    :param circle1: First IntermediateCircle.
    :type circle1: IntermediateCircle

    :param circle2: Second IntermediateCircle.
    :type circle2: IntermediateCircle

    :param corridor_list: Full corridor list.
    :type corridor_list: list[CorridorWorld]

    :return: first_line, third_line
    :rtype: tuple[tuple[float, float, float], tuple[float, float, float]]
    """
    if circle1.corridor_index_end != circle2.corridor_index_start:
        raise ValueError(
            "Cannot extract merge centerlines for non-consecutive circles"
        )

    corridor1 = corridor_list[circle1.corridor_index_start]
    corridor3 = corridor_list[circle2.corridor_index_end]

    edge1 = circle1.edge_pair[0]
    edge3 = circle2.edge_pair[1]

    first_line = relevant_centerline_equation_from_edge(
        corridor1,
        edge1,
    )

    third_line = relevant_centerline_equation_from_edge(
        corridor3,
        edge3,
    )

    return first_line, third_line


def point_line_distance(point, line):
    """
    Distance from Point to normalized line a*x + b*y + c = 0.
    """
    a, b, c = line
    return abs(a * point.x + b * point.y + c)


def circle_intersects_centerline(center, radius, line, tol=1e-9):
    """
    Return True if the circle intersects the centerline.
    """
    return point_line_distance(center, line) < radius - tol


def merged_candidate_centerline_intersection_status(
    center,
    radius,
    first_line,
    third_line,
    tol=1e-9,
):
    """
    Check whether a merged circle intersects the outer centerlines.

    :param center: Center of the merged circle.
    :type center: Point

    :param radius: Radius of the merged circle.
    :type radius: float

    :param first_line: First outer corridor centerline equation (a, b, c).
    :type first_line: tuple[float, float, float]

    :param third_line: Third outer corridor centerline equation (a, b, c).
    :type third_line: tuple[float, float, float]

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Status dictionary.
    :rtype: dict
    """
    first_distance = point_line_distance(center, first_line)
    third_distance = point_line_distance(center, third_line)

    intersects_first = first_distance < radius - tol
    intersects_third = third_distance < radius - tol

    crossed_centerlines = []

    if intersects_first:
        crossed_centerlines.append("first")

    if intersects_third:
        crossed_centerlines.append("third")

    return {
        "intersects": intersects_first or intersects_third,
        "intersects_first": intersects_first,
        "intersects_third": intersects_third,
        "crossed_centerlines": crossed_centerlines,
        "first_distance": first_distance,
        "third_distance": third_distance,
        "first_clearance": first_distance - radius,
        "third_clearance": third_distance - radius,
    }


def nominal_overlap_status_for_intermediate_circles(circle1, circle2, vehicle, tol=1e-9):
    """
    Check whether two IntermediateCircles overlap in their nominal
    configuration.

    This uses the current nominal centers, not shifted centers.

    :param circle1: First intermediate circle.
    :type circle1: IntermediateCircle

    :param circle2: Second intermediate circle.
    :type circle2: IntermediateCircle

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Dictionary with overlap information.
    :rtype: dict
    """
    distance = compute_distance_two_points(circle1.center, circle2.center)
    required_distance = circle1.radius + circle2.radius

    nominal_overlap = distance < required_distance - tol

    r = vehicle.width / 2.0

    can_be_merged = (
        compute_distance_two_points(circle1.corner_point, circle2.center) + r <= circle2.radius + tol
        or
        compute_distance_two_points(circle2.corner_point, circle1.center) + r <= circle1.radius + tol
    )

    return {
        "nominal_overlap": nominal_overlap,
        "distance": distance,
        "required_distance": required_distance,
        "overlap_amount": required_distance - distance,
        "can_be_merged": can_be_merged,
    }


def detect_same_turn_nominal_overlap_pairs(
    intermediate_circles_sequence,
    tol=1e-9,
):
    """
    Detect consecutive same-turn IntermediateCircles that overlap nominally.

    :param intermediate_circles_sequence: Ordered sequence of IntermediateCircle objects.
    :type intermediate_circles_sequence: IntermediateCirclesSequence

    :param tol: Numerical tolerance.
    :type tol: float

    :return: List of dictionaries describing overlapping pairs.
    :rtype: list[dict]
    """
    overlap_pairs = []

    for i in range(len(intermediate_circles_sequence) - 1):
        circle1 = intermediate_circles_sequence[i]
        circle2 = intermediate_circles_sequence[i + 1]

        if circle1.turn_direction != circle2.turn_direction:
            continue

        status = nominal_overlap_status_for_intermediate_circles(
            circle1,
            circle2,
            vehicle,
            tol=tol,
        )

        if not status["nominal_overlap"]:
            continue

        overlap_pairs.append(
            {
                "indices": [i, i + 1],
                "turn_direction": circle1.turn_direction,
                "circle1": circle1,
                "circle2": circle2,
                "status": status,
            }
        )

    return overlap_pairs


def detect_same_turn_nominal_overlap_blocks(
    intermediate_circles_sequence,
    vehicle,
    max_block_size=3,
    tol=1e-9,
):
    """
    Detect blocks of consecutive IntermediateCircles that:
        - have the same turn direction;
        - have nominal overlap between consecutive circles.

    The returned blocks contain circle indices.

    Example:
        If circle 0 overlaps circle 1, and circle 1 overlaps circle 2,
        the function returns [[0, 1, 2]].

    :param intermediate_circles_sequence: Ordered sequence of IntermediateCircle objects.
    :type intermediate_circles_sequence: IntermediateCirclesSequence

    :param max_block_size: Maximum allowed merge block size.
    :type max_block_size: int

    :param tol: Numerical tolerance.
    :type tol: float

    :return: List of blocks. Each block is a list of circle indices.
    :rtype: list[list[int]]
    """
    blocks = []
    current_block = []

    n_circles = len(intermediate_circles_sequence)

    if n_circles < 2:
        return blocks

    for i in range(n_circles - 1):
        circle1 = intermediate_circles_sequence[i]
        circle2 = intermediate_circles_sequence[i + 1]

        same_turn = circle1.turn_direction == circle2.turn_direction

        status = nominal_overlap_status_for_intermediate_circles(
            circle1,
            circle2,
            vehicle,
            tol=tol,
        )

        overlaps = status["can_be_merged"]

        if same_turn and overlaps:
            if len(current_block) == 0:
                current_block = [i, i + 1]
            else:
                # Continue only if this pair touches the previous block.
                if current_block[-1] == i:
                    current_block.append(i + 1)
                else:
                    blocks.append(current_block)
                    current_block = [i, i + 1]
        else:
            if len(current_block) > 0:
                blocks.append(current_block)
                current_block = []

    if len(current_block) > 0:
        blocks.append(current_block)

    # Optional safety: reject blocks that are too large.
    for block in blocks:
        if len(block) > max_block_size:
            raise ValueError(
                f"Detected same-turn overlap block of size {len(block)}: {block}. "
                f"Current merge logic only supports blocks up to size {max_block_size}."
            )

    return blocks


def assign_preferred_turn_directions_to_ambiguous_turns(
    corridor_list,
    turn_direction_sequence,
    circle_slots,
    start_pose,
    end_pose,
):
    """
    Assign preferred turn directions to ambiguous turns in a corridor sequence.

    This function only updates the turn direction sequence. It does not build
    IntermediateCircle objects for ambiguous transitions.

    For each ambiguous transition, it computes two possible door/corner points:
        - one assuming a right turn, tau = -1;
        - one assuming a left turn, tau = +1.

    The midpoint of these two door points is used as the representative point
    of the ambiguous transition.

    Then each ambiguous transition is resolved by looking at three points:

        previous reference point -> current midpoint -> next reference point

    The turn direction of these three points becomes the preferred turn
    direction for that ambiguous transition.

    :param corridor_list: Ordered list of corridors.
    :type corridor_list: list[CorridorWorld]

    :param turn_direction_sequence: Current sequence of turn directions.
                                    Values are -1, 0, +1.
    :type turn_direction_sequence: list[int]

    :param circle_slots: List of already-built circles or None.
                         Non-ambiguous circles may already be available.
    :type circle_slots: list[IntermediateCircle or None]

    :param start_pose: Initial vehicle pose [x, y, theta].
    :type start_pose: list[float] or numpy.ndarray

    :param end_pose: Final vehicle pose [x, y, theta].
    :type end_pose: list[float] or numpy.ndarray

    :return: Updated turn direction sequence.
    :rtype: list[int]
    """
    updated_turn_direction_sequence = list(turn_direction_sequence)

    if len(updated_turn_direction_sequence) != len(corridor_list) - 1:
        raise ValueError(
            "turn_direction_sequence must have length len(corridor_list) - 1"
        )

    if len(circle_slots) != len(updated_turn_direction_sequence):
        raise ValueError(
            "circle_slots must have the same length as turn_direction_sequence"
        )

    ambiguous_blocks = detect_ambiguous_blocks_from_turn_sequence(
        updated_turn_direction_sequence
    )

    start_point = Point(start_pose[0], start_pose[1])
    end_point = Point(end_pose[0], end_pose[1])

    n_transitions = len(updated_turn_direction_sequence)

    for block in ambiguous_blocks:

        # --------------------------------------------------------
        # Step 1.
        # Compute representative mid-door points for the block.
        #
        # For an ambiguous transition i, we compute:
        #   door_right = corner point assuming tau = -1
        #   door_left  = corner point assuming tau = +1
        #
        # Then:
        #   mid_door = midpoint(door_right, door_left)
        # --------------------------------------------------------
        mid_door_sequence = []

        for transition_index in block:
            corridor1 = corridor_list[transition_index]
            corridor2 = corridor_list[transition_index + 1]

            # Right-turn hypothetical door/corner
            door_right = get_corner_point(
                corridor1,
                corridor2,
                -1,
            )

            # Left-turn hypothetical door/corner
            door_left = get_corner_point(
                corridor1,
                corridor2,
                1,
            )

            if door_right is None or door_left is None:
                plot_corridors([corridor_list[transition_index], corridor_list[transition_index + 1]])
                plt.show(block=True)
            door_right = Point(*door_right)
            door_left = Point(*door_left)

            mid_door = Point(
                0.5 * (door_right.x + door_left.x),
                0.5 * (door_right.y + door_left.y),
            )

            mid_door_sequence.append(mid_door)

        # --------------------------------------------------------
        # Step 2.
        # Resolve each ambiguous transition inside the block.
        # --------------------------------------------------------
        for local_index, transition_index in enumerate(block):

            p_curr = mid_door_sequence[local_index]

            # ----------------------------------------------------
            # Previous reference point
            # ----------------------------------------------------
            if local_index > 0:
                # Previous ambiguous transition in the same block.
                p_prev = mid_door_sequence[local_index - 1]

            else:
                # This is the first ambiguous transition in the block.
                previous_transition_index = transition_index - 1

                if previous_transition_index >= 0:
                    previous_circle = circle_slots[previous_transition_index]
                else:
                    previous_circle = None

                if previous_circle is not None:
                    if (
                        compute_distance_two_points(previous_circle.center, p_curr)
                        < previous_circle.radius
                    ):
                        p_prev = previous_circle.center
                    else:
                        p_prev = select_tangency_point_from_point_circle(
                            p_curr,
                            previous_circle,
                            turn_direction=-previous_circle.turn_direction,
                        )
                else:
                    p_prev = start_point

            # ----------------------------------------------------
            # Next reference point
            # ----------------------------------------------------
            if local_index < len(block) - 1:
                # Next ambiguous transition in the same block.
                p_next = mid_door_sequence[local_index + 1]

            else:
                # This is the last ambiguous transition in the block.
                next_transition_index = transition_index + 1

                if next_transition_index < n_transitions:
                    next_circle = circle_slots[next_transition_index]
                else:
                    next_circle = None

                if next_circle is not None:
                    if (
                        compute_distance_two_points(next_circle.center, p_curr)
                        < next_circle.radius
                    ):
                        p_next = next_circle.center
                    else:
                        p_next = select_tangency_point_from_point_circle(
                            p_curr,
                            next_circle,
                        )
                else:
                    p_next = end_point

            # ----------------------------------------------------
            # Step 3.
            # Compute preferred turn direction from three points.
            # ----------------------------------------------------
            preferred_turn = compute_turn_direction_from_three_points(
                p_prev,
                p_curr,
                p_next,
            )

            # plot_corridors(corridor_list)
            # plt.plot([p_prev.x, p_curr.x, p_next.x], [p_prev.y, p_curr.y, p_next.y], 'ro-')
            # plt.show(block=True)

            # ----------------------------------------------------
            # Step 4.
            # Fallback if perfectly aligned.
            # ----------------------------------------------------
            if preferred_turn == 0:
                # TODO:
                # Replace this fallback later with something smarter.
                # Possible options:
                #   - use previous nonzero turn;
                #   - use next nonzero turn;
                #   - use +1 by convention;
                #   - use old planner heuristic.
                preferred_turn = 1

            updated_turn_direction_sequence[transition_index] = int(preferred_turn)

    return updated_turn_direction_sequence


def detect_ambiguous_blocks_from_turn_sequence(turn_direction_sequence):
    """
    Detect consecutive blocks of ambiguous turns.

    :param turn_direction_sequence: Sequence of turn directions.
                                    Values are -1, 0, +1.
    :type turn_direction_sequence: list[int]

    :return: List of blocks. Each block is a list of consecutive indices
             where turn_direction_sequence[index] == 0.
    :rtype: list[list[int]]
    """
    ambiguous_blocks = []
    current_block = []

    for i, tau in enumerate(turn_direction_sequence):
        if tau == 0:
            current_block.append(i)
        else:
            if len(current_block) > 0:
                ambiguous_blocks.append(current_block)
                current_block = []

    if len(current_block) > 0:
        ambiguous_blocks.append(current_block)

    return ambiguous_blocks


def get_other_intersection_point_if_present(corridor1, corridor2):
    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    # Front edge of corridor1
    A1 = corners1[3]  # top left
    A2 = corners1[0]  # top right
    # Back edge of corridor2
    B1 = corners2[2]  # bottom left
    B2 = corners2[1]  # bottom right

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        is_corner_intersection = (
            point_matches_any_corner(int_point, corners1, tol=1e-9)
            or point_matches_any_corner(int_point, corners2, tol=1e-9)
        )

        if is_corner_intersection:
            intersects = False
            int_point = None

    return int_point, intersects


def build_intermediate_circles_sequence(
    corridor_list,
    vehicle,
    start_pose,
    end_pose,
    build_only_resolved_turns=True,
):
    """
    Build an ordered sequence of IntermediateCircle objects for a sequence
    of corridors.

    :param corridor_list: Ordered list of corridors followed by the planner.
    :type corridor_list: list[CorridorWorld]

    :param vehicle: Vehicle object used to retrieve footprint and turning radius.
    :type vehicle: Vehicle

    :param start_pose: Initial pose of the vehicle.
    :type start_pose: list[float] or numpy.ndarray

    :param end_pose: Final pose of the vehicle.
    :type end_pose: list[float] or numpy.ndarray

    :param build_only_resolved_turns: If True, build only circles for tau in {-1, +1}
                                      and skip ambiguous tau = 0 turns.
                                      This is useful for debugging/visualization.
    :type build_only_resolved_turns: bool

    :return: Ordered sequence of available IntermediateCircle objects.
    :rtype: IntermediateCirclesSequence
    """

    # ------------------------------------------------------------
    # Basic input checks
    # ------------------------------------------------------------
    if corridor_list is None:
        raise ValueError("corridor_list cannot be None")

    if len(corridor_list) < 2:
        raise ValueError(
            "At least two corridors are required to build intermediate circles"
        )

    # ------------------------------------------------------------
    # 1. Corridor sequence feasibility check
    # ------------------------------------------------------------
    validate_corridor_sequence(
        corridor_list=corridor_list,
        vehicle=vehicle,
        min_centerline_distance=2.0 * vehicle.max_radius,
        plot_invalid=True,
    )

    # ------------------------------------------------------------
    # 2. Build turn direction sequence based on corridor tilts
    # ------------------------------------------------------------
    turn_direction_sequence = []

    for i in range(len(corridor_list) - 1):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        tau = corridor1.compute_relative_turn_direction(corridor2)
        tau = int(tau)

        turn_direction_sequence.append(tau)

    # ------------------------------------------------------------
    # 3. Build intermediate circles for tau in {-1, +1}
    # ------------------------------------------------------------
    circle_slots = [None] * (len(corridor_list) - 1)

    for i, tau in enumerate(turn_direction_sequence):
        if tau not in (-1, 1):
            continue

        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        corner_point, intersecting_edges = get_corner_point_and_intersecting_edges(
            corridor1,
            corridor2,
            tau,
        )

        print(f"Building intermediate circle for corridors {i} and {i+1}: {tau}")
        print(f"Corner point: {corner_point}, intersecting edges: {intersecting_edges}")

        # figure = plot_corridors(corridor_list, plot_vectors=True)

        # plot_corridors([corridor1, corridor2], plot_vectors=False, figure=figure, color = "red")
        # plt.plot(corner_point[0], corner_point[1], "ko", label="corner point")
        # plt.legend()
        # plt.show(block = True)

        # --------------------------------------------------------
        # Optional additional intersection point
        # --------------------------------------------------------
        other_intersection_point = None

        if (
            (intersecting_edges[0] == 1 and intersecting_edges[1] == 1)
            or
            (intersecting_edges[0] == 3 and intersecting_edges[1] == 3)
        ):
            candidate_point, intersection_exists = (
                get_other_intersection_point_if_present(corridor1, corridor2)
            )

            if intersection_exists:
                # Make sure we pass a Point object to compute_intermediate_circle_geometry.
                if isinstance(candidate_point, Point):
                    other_intersection_point = candidate_point
                else:
                    other_intersection_point = Point(*candidate_point)

        else:
            raise ValueError(
                f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                f"{intersecting_edges}"
            )

        geometry_result = compute_intermediate_circle_geometry(
            corridor1=corridor1,
            corridor2=corridor2,
            corner_point=Point(*corner_point),
            turn_direction=tau,
            vehicle=vehicle,
            other_intersection_point=other_intersection_point,
        )

        if not geometry_result["feasible"]:
            raise ValueError(
                f"Unfeasible intermediate circle geometry for corridors "
                f"{i} and {i+1}: {geometry_result}"
            )

        circle = build_intermediate_circle_from_geometry_result(
            geometry_result=geometry_result,
            index=i,
            edge_pair=intersecting_edges
        )
        circle.corridor_index_start = i
        circle.corridor_index_end = i + 1
        circle_slots[i] = circle

    # ------------------------------------------------------------
    # 4. Assign preferred turn directions to ambiguous turns tau = 0
    # ------------------------------------------------------------
    print("Turn direction sequence before ambiguous assignment:")
    print(turn_direction_sequence)
    turn_direction_sequence = assign_preferred_turn_directions_to_ambiguous_turns(
        corridor_list=corridor_list,
        turn_direction_sequence=turn_direction_sequence,
        circle_slots=circle_slots,
        start_pose=start_pose,
        end_pose=end_pose,
    )
    print("Turn direction sequence after ambiguous assignment:")
    print(turn_direction_sequence)
    # ------------------------------------------------------------
    # 5. Build all the remaining circles
    # ------------------------------------------------------------
    for i, circle in enumerate(circle_slots):
        if circle is not None:
            continue

        tau = turn_direction_sequence[i]

        if tau not in (-1, 1):
            raise ValueError(
                f"Turn direction at index {i} was not resolved: tau={tau}"
            )

        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        corner_point, intersecting_edges_nominal = get_corner_point_and_intersecting_edges(
            corridor1,
            corridor2,
            tau,
        )

        edge1, edge2 = intersecting_edges_nominal

        print(f"Building intermediate circle for corridors {i} and {i+1}: {tau}")
        print(f"Corner point: {corner_point}, intersecting edges: {intersecting_edges_nominal}")
        
        # figure = plot_corridors(corridor_list, plot_vectors=True)

        # plot_corridors([corridor1, corridor2], plot_vectors=False, figure=figure, color = "red")
        # plt.plot(corner_point[0], corner_point[1], "ko", label="corner point")
        # plt.legend()
        # plt.show(block = True)

        if edge1 == 0: 
            corridor2_rotated = corridor2 
            if edge2 == 1: 
                corridor1_rotated = corridor1.invert_dimensions(1)
            elif edge2 == 3: 
                corridor1_rotated = corridor1.invert_dimensions(-1)
            else:
                raise ValueError(
                    f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                    f"{intersecting_edges_nominal}"
                )
            
        elif edge2 == 2: 
            corridor1_rotated = corridor1
            if edge1 == 1: 
                corridor2_rotated = corridor2.invert_dimensions(-1)
            elif edge1 == 3: 
                corridor2_rotated = corridor2.invert_dimensions(1)
            else:
                raise ValueError(
                    f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                    f"{intersecting_edges_nominal}"
                )

        corner_point, intersecting_edges = get_corner_point_and_intersecting_edges(
            corridor1_rotated,
            corridor2_rotated,
            tau,
        )

        # plot_corridors(corridor_list, plot_vectors=True)
        # plot_corridors([corridor1_rotated, corridor2_rotated], color = "red", plot_vectors=True)
        # plt.plot(corner_point_rotated[0], corner_point_rotated[1], "ko", label="corner point")
        # plt.legend()
        # plt.show(block = True)

        # --------------------------------------------------------
        # Additional intersection point
        # --------------------------------------------------------
        other_intersection_point = None

        if (
            (intersecting_edges[0] == 1 and intersecting_edges[1] == 1)
            or
            (intersecting_edges[0] == 3 and intersecting_edges[1] == 3)
        ):
            candidate_point, intersection_exists = (
                get_other_intersection_point_if_present(corridor1_rotated, corridor2_rotated)
            )

            if intersection_exists:
                # Make sure we pass a Point object to compute_intermediate_circle_geometry.
                if isinstance(candidate_point, Point):
                    other_intersection_point = candidate_point
                else:
                    other_intersection_point = Point(*candidate_point)

        else:
            raise ValueError(
                f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                f"{intersecting_edges}"
            )

        geometry_result = compute_intermediate_circle_geometry(
            corridor1=corridor1_rotated,
            corridor2=corridor2_rotated,
            corner_point=Point(*corner_point),
            turn_direction=tau,
            vehicle=vehicle,
            other_intersection_point=other_intersection_point,
        )

        if not geometry_result["feasible"]:
            raise ValueError(
                f"Unfeasible intermediate circle geometry for corridors "
                f"{i} and {i+1}: {geometry_result}"
            )

        circle = build_intermediate_circle_from_geometry_result(
            geometry_result=geometry_result,
            index=i,
            edge_pair=intersecting_edges_nominal,
        )
        circle.corridor_index_start = i
        circle.corridor_index_end = i + 1
        circle_slots[i] = circle

    # ------------------------------------------------------------
    # Convert available slots into IntermediateCirclesSequence
    # ------------------------------------------------------------
    intermediate_circles_sequence = IntermediateCirclesSequence()

    for circle in circle_slots:
        if circle is None:
            continue

        intermediate_circles_sequence.append(circle)

    #-------------------------------------------------------------
    # Plot for debugging
    #-------------------------------------------------------------
    figure = plot_corridors(corridor_list, plot_vectors=True)
    ax = plt.gca()
    plot_intermediate_circles_sequence_debug(
        ax=ax,
        intermediate_circles_sequence=intermediate_circles_sequence,
        footprint_radius=vehicle.width / 2.0,
        plot_swept_circle=False,
        plot_shifted_circle=False,
        plot_corner_small_circles=True,
    )
    plt.show(block=True)
    # plt.show(block=True)

    # ------------------------------------------------------------
    # 6. Merge or shift overlapping circles
    # ------------------------------------------------------------
    overlap_blocks = detect_same_turn_nominal_overlap_blocks(
        intermediate_circles_sequence,
        vehicle,
        max_block_size=3,
    )

    print("Same-turn nominal overlap blocks:")
    print(overlap_blocks)
    # For now skipped. We only return available circles for visualization.
    intermediate_circles_sequence = resolve_same_turn_nominal_overlap_blocks(
        intermediate_circles_sequence=intermediate_circles_sequence,
        corridor_list=corridor_list,
        vehicle=vehicle,
        overlap_blocks=overlap_blocks,
    )

    return intermediate_circles_sequence




def build_circle_from_two_corridors(corridor1, corridor2, tau, vehicle, i):

    tau_from_corridors = corridor1.compute_relative_turn_direction(corridor2)
    if tau in (1,-1) and tau_from_corridors in (1,-1) and tau != tau_from_corridors:
        corridor2 = corridor2.rotate_corridor(angle=np.pi)

    corner_point, intersecting_edges_nominal = get_corner_point_and_intersecting_edges(
            corridor1,
            corridor2,
            tau,
        )

    edge1, edge2 = intersecting_edges_nominal

    print(f"Building intermediate circle for corridors {i} and {i+1}: {tau}")
    print(f"Corner point: {corner_point}, intersecting edges: {intersecting_edges_nominal}")
    
    # figure = plot_corridors(corridor_list, plot_vectors=True)

    # plot_corridors([corridor1, corridor2], plot_vectors=False, figure=figure, color = "red")
    # plt.plot(corner_point[0], corner_point[1], "ko", label="corner point")
    # plt.legend()
    # plt.show(block = True)

    if tau == 1 and edge1 == 1 and edge2 == 1: 
        corridor1_rotated = corridor1
        corridor2_rotated = corridor2
    elif tau == -1 and edge1 == 1 and edge2 == 1:
        corridor1_rotated = corridor1
        corridor2_rotated = corridor2
    else:

        if edge1 == 0: 
            corridor2_rotated = corridor2 
            if edge2 == 1: 
                corridor1_rotated = corridor1.invert_dimensions(1)
            elif edge2 == 3: 
                corridor1_rotated = corridor1.invert_dimensions(-1)
            else:
                raise ValueError(
                    f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                    f"{intersecting_edges_nominal}"
                )
            
        elif edge2 == 2: 
            corridor1_rotated = corridor1
            if edge1 == 1: 
                corridor2_rotated = corridor2.invert_dimensions(-1)
            elif edge1 == 3: 
                corridor2_rotated = corridor2.invert_dimensions(1)
            else:
                raise ValueError(
                    f"Unexpected intersecting edges for corridors {i}, {i+1}: "
                    f"{intersecting_edges_nominal}"
                )

    corner_point, intersecting_edges = get_corner_point_and_intersecting_edges(
        corridor1_rotated,
        corridor2_rotated,
        tau,
    )

    # plot_corridors([corridor1_rotated, corridor2_rotated], plot_vectors=True)
    # # plot_corridors([corridor1_rotated, corridor2_rotated], color = "red", plot_vectors=True)
    # plt.plot(corner_point[0], corner_point[1], "ko", label="corner point")
    # plt.legend()
    # plt.show(block = True)

    # --------------------------------------------------------
    # Additional intersection point
    # --------------------------------------------------------
    other_intersection_point = None

    if (
        (intersecting_edges[0] == 1 and intersecting_edges[1] == 1)
        or
        (intersecting_edges[0] == 3 and intersecting_edges[1] == 3)
    ):
        candidate_point, intersection_exists = (
            get_other_intersection_point_if_present(corridor1_rotated, corridor2_rotated)
        )

        if intersection_exists:
            # Make sure we pass a Point object to compute_intermediate_circle_geometry.
            if isinstance(candidate_point, Point):
                other_intersection_point = candidate_point
            else:
                other_intersection_point = Point(*candidate_point)

    else:
        raise ValueError(
            f"Unexpected intersecting edges for corridors {i}, {i+1}: "
            f"{intersecting_edges}"
        )

    geometry_result = compute_intermediate_circle_geometry(
        corridor1=corridor1_rotated,
        corridor2=corridor2_rotated,
        corner_point=Point(*corner_point),
        turn_direction=tau,
        vehicle=vehicle,
        other_intersection_point=other_intersection_point,
    )

    if not geometry_result["feasible"]:
        raise ValueError(
            f"Unfeasible intermediate circle geometry for corridors "
            f"{i} and {i+1}: {geometry_result}"
        )

    circle = build_intermediate_circle_from_geometry_result(
        geometry_result=geometry_result,
        index=i,
        edge_pair=intersecting_edges_nominal,
    )
    circle.corridor_index_start = i
    circle.corridor_index_end = i + 1

    return circle



    