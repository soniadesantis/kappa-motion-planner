from ..geometry import Point, Circle, IntermediateCircle, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence
from ..corridor import CorridorWorld
from ..vehicle import Unicycle, Bicycle
from .intersections import compute_intersection_two_segments, compute_line_corridor_intersections
from .corridor_geometry import get_corner_point_and_intersecting_edges
from .intermediate_circles_geometry import compute_center_coordinates_second_circle_according_to_edges, compute_center_coordinates_second_circle_given_two_points, compute_circle_internally_tangent_to_two_circles
from .intermediate_circle_solve_overlap import (
    select_preferred_circle,
    overlap_status_for_intermediate_circles,
    compute_circle_through_two_points_with_radius,
    build_merged_intermediate_circle,
    intermediate_circle_center_at_s,
    build_circle_overlap_opposite_turn_direction,
    compute_consecutive_overlap_pairs,
    extract_overlap_clusters,
    classify_overlap_cluster,
)
from .corridor_sequence_validity import validate_corridor_sequence
from .inputs_check import compute_min_width_s_max_corridor_pair
from .plot_helpers import plot_corridors
from .geometry_operations import (
    select_tangency_point_from_point_circle,
    compute_turn_direction,
    compute_turn_direction_from_three_points,
    efficient_sign,
    compute_distance_two_points
)
from math import sqrt, atan2, cos, sin, pi
from matplotlib import pyplot as plt
import numpy as np


import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# def solve_circles_overlap(
#     circle_choices_sequence,
#     corridor_sequence,
#     vehicle,
#     start_pose,
#     end_pose,
# ):
#     """
#     Resolve overlaps between consecutive selected intermediate circles.

#     Strategy:
#     1. First resolve overlaps between circles with the same turn direction.
#        These cases are treated by direct replacement, when possible, or by
#        merging the two circles.
#     2. Then re-check the resulting sequence and resolve remaining overlaps
#        between circles with opposite turn directions.

#     For ambiguous choices, the preferred candidate is selected.
#     """
#     overlapping_pairs = compute_consecutive_overlap_pairs(circle_choices_sequence)

#     clusters = extract_overlap_clusters(overlapping_pairs)

#     print(clusters)

#     circle_choices_sequence = solve_same_turn_overlaps(
#         circle_choices_sequence,
#         corridor_sequence,
#         vehicle,
#         start_pose,
#         end_pose,
#     )

#     circle_choices_sequence = solve_opposite_turn_overlaps(
#         circle_choices_sequence,
#         corridor_sequence,
#         vehicle,
#         start_pose,
#         end_pose,
#     )

#     return circle_choices_sequence


# def solve_circles_overlap(
#     circle_choices_sequence,
#     corridor_sequence,
#     vehicle,
#     start_pose,
#     end_pose,
# ):
#     while True:
#         overlapping_pairs = compute_consecutive_overlap_pairs(
#             circle_choices_sequence
#         )

#         clusters = extract_overlap_clusters(overlapping_pairs)

#         if not clusters:
#             return circle_choices_sequence

#         # Solve one cluster at a time, then recompute everything.
#         cluster = clusters[0]
#         cluster_type = classify_overlap_cluster(cluster)

#         print(
#             f"Solving cluster {cluster['indices']} "
#             f"of type {cluster_type}"
#         )

#         if cluster_type == "pair_same_turn":
#             circle_choices_sequence = solve_pair_same_turn_cluster(
#                 circle_choices_sequence,
#                 corridor_sequence,
#                 vehicle,
#                 start_pose,
#                 end_pose,
#                 cluster,
#             )

#         # elif cluster_type == "pair_opposite_turn":
#         #     circle_choices_sequence = solve_pair_opposite_turn_cluster(
#         #         circle_choices_sequence,
#         #         corridor_sequence,
#         #         vehicle,
#         #         start_pose,
#         #         end_pose,
#         #         cluster,
#         #     )

#         # elif cluster_type == "multi_same_turn":
#         #     circle_choices_sequence = solve_multi_same_turn_cluster(
#         #         circle_choices_sequence,
#         #         corridor_sequence,
#         #         vehicle,
#         #         start_pose,
#         #         end_pose,
#         #         cluster,
#         #     )

#         # elif cluster_type == "multi_mixed_turn":
#         #     circle_choices_sequence = solve_multi_mixed_turn_cluster(
#         #         circle_choices_sequence,
#         #         corridor_sequence,
#         #         vehicle,
#         #         start_pose,
#         #         end_pose,
#         #         cluster,
#         #     )

#         # else:
#         #     raise ValueError(f"Unknown cluster type: {cluster_type}")
#         else:
#             print(
#                 f"Stopping debug solver: cluster {cluster['indices']} "
#                 f"has type {cluster_type}, which is not implemented yet."
#             )
#             return circle_choices_sequence


def solve_circles_overlap(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
    max_iterations=50,
):
    implemented_cluster_types = {
        "pair_same_turn",
        "pair_opposite_turn",
    }

    for iteration in range(max_iterations):
        overlapping_pairs = compute_consecutive_overlap_pairs(
            circle_choices_sequence
        )

        clusters = extract_overlap_clusters(overlapping_pairs)

        if not clusters:
            return circle_choices_sequence

        print(f"\nIteration {iteration}")
        print("Detected clusters:")

        for cluster in clusters:
            cluster_type = classify_overlap_cluster(cluster)
            print(f"  {cluster['indices']} -> {cluster_type}")

        solvable_cluster = None
        solvable_cluster_type = None

        for cluster in clusters:
            cluster_type = classify_overlap_cluster(cluster)

            if cluster_type in implemented_cluster_types:
                solvable_cluster = cluster
                solvable_cluster_type = cluster_type
                break

        if solvable_cluster is None:
            print(
                "No currently solvable clusters found. "
                "Stopping debug solver."
            )
            return circle_choices_sequence

        print(
            f"Solving cluster {solvable_cluster['indices']} "
            f"of type {solvable_cluster_type}"
        )

        if solvable_cluster_type == "pair_same_turn":
            circle_choices_sequence = solve_pair_same_turn_cluster(
                circle_choices_sequence,
                corridor_sequence,
                vehicle,
                start_pose,
                end_pose,
                solvable_cluster,
            )

        elif solvable_cluster_type == "pair_opposite_turn":
            circle_choices_sequence = solve_pair_opposite_turn_cluster(
                circle_choices_sequence,
                corridor_sequence,
                vehicle,
                start_pose,
                end_pose,
                solvable_cluster,
            )

    raise RuntimeError(
        "Overlap solver reached max_iterations. "
        "A solver may not be modifying the sequence."
    )



def solve_same_turn_overlaps(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
):
    """
    Resolve overlaps between consecutive selected circles with the same
    turn direction.

    The function first tries to replace the two transitions with one direct
    transition between the outer corridors. If this is not possible, it tries
    to merge the two circles.
    """

    index = 0

    while index < len(circle_choices_sequence) - 1:
        choice1 = circle_choices_sequence[index]
        choice2 = circle_choices_sequence[index + 1]

        circle1 = select_preferred_circle(choice1)
        circle2 = select_preferred_circle(choice2)

        status = overlap_status_for_intermediate_circles(circle1, circle2)
        nominal_overlap = status["nominal_overlap"]

        if not nominal_overlap:
            index += 1
            continue

        # ------------------------------------------------------------
        # Step 1: try direct replacement between the outer corridors
        # ------------------------------------------------------------
        corridor1_index = choice1.corridor_index_start
        corridor3_index = choice2.corridor_index_end

        corridor1 = corridor_sequence[corridor1_index]
        corridor3 = corridor_sequence[corridor3_index]

        replacement_choice = try_build_replacement_choice_between_corridors(
            corridor1,
            corridor3,
            index,
            vehicle,
        )

        if replacement_choice is not None:
            replacement_choice.corridor_index_start = choice1.corridor_index_start
            replacement_choice.corridor_index_end = choice2.corridor_index_end

            circle_choices_sequence.replace_two_with_one(
                index,
                replacement_choice,
            )

            reindex_intermediate_circle_choices_sequence(
                circle_choices_sequence
            )

            circle_choices_sequence = assign_preferred_candidates(
                circle_choices_sequence,
                start_pose,
                end_pose,
            )

            # Re-check local neighborhood because replacement may create
            # a new overlap with the previous circle.
            index = max(index - 1, 0)
            continue

        # ------------------------------------------------------------
        # Step 2: if direct replacement fails, try same-turn merge
        # ------------------------------------------------------------

        # In this first pass, ignore opposite-turn overlaps.
        # They will be handled in the second pass.
        if circle1.turn_direction != circle2.turn_direction:
            index += 1
            continue

        print(
            f"Detected same-turn overlap between circles at indices "
            f"{index} and {index + 1}"
        )

        middle_corridor_index = choice1.corridor_index_end
        corridor2 = corridor_sequence[middle_corridor_index]

        R = vehicle.max_radius

        small_circle1 = Circle(
            circle1.corner_point,
            vehicle.width / 2,
        )

        small_circle2 = Circle(
            circle2.corner_point,
            vehicle.width / 2,
        )

        can_merge = (
            compute_distance_two_points(
                small_circle1.center,
                small_circle2.center,
            )
            < 2 * (R - vehicle.width / 2)
        )

        if not can_merge:
            index += 1
            continue

        merged_circle = compute_circle_internally_tangent_to_two_circles(
            small_circle1,
            small_circle2,
            radius=vehicle.max_radius,
            turn_direction=circle1.turn_direction,
        )

        merged_intermediate_circle = build_merged_intermediate_circle(
            circle1,
            circle2,
            corridor1,
            corridor2,
            corridor3,
            merged_circle,
            vehicle,
            index=None,
            s_max=0.0,
        )

        # Optional but useful for debugging
        merged_intermediate_circle.is_merged = True
        merged_intermediate_circle.merged_from = (
            choice1.corridor_index_start,
            choice1.corridor_index_end,
            choice2.corridor_index_start,
            choice2.corridor_index_end,
        )

        replacement_choice = IntermediateCircleChoice(
            candidates=[merged_intermediate_circle],
            index=index,
            corridor_index_start=choice1.corridor_index_start,
            corridor_index_end=choice2.corridor_index_end,
        )

        circle_choices_sequence.replace_two_with_one(
            index,
            replacement_choice,
        )

        reindex_intermediate_circle_choices_sequence(
            circle_choices_sequence
        )

        circle_choices_sequence = assign_preferred_candidates(
            circle_choices_sequence,
            start_pose,
            end_pose,
        )

        # Re-check from previous local neighborhood.
        index = max(index - 1, 0)

        figure = plot_corridors(corridor_sequence, plot_vectors=True)
        ax = plt.gca()
        plot_intermediate_circle_choices(
            ax,
            circle_choices_sequence,
            vehicle.width/2,
        )
        plt.plot(small_circle1.center.x, small_circle1.center.y, "x", label="small_circle1 center")
        plt.plot(small_circle1.center.x + small_circle1.radius * np.cos(np.linspace(0, 2*np.pi, 200)), small_circle1.center.y + small_circle1.radius * np.sin(np.linspace(0, 2*np.pi, 200)), "-", label="small_circle1")
        plt.plot(small_circle2.center.x, small_circle2.center.y, "x", label="small_circle2 center")
        plt.plot(small_circle2.center.x + small_circle2.radius * np.cos(np.linspace(0, 2*np.pi, 200)), small_circle2.center.y + small_circle2.radius * np.sin(np.linspace(0, 2*np.pi, 200)), "-", label="small_circle2")
        plt.show(block = True)

    return circle_choices_sequence

def solve_pair_same_turn_cluster(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
    cluster,
    debug_plot=False,
):
    """
    Resolve one same-turn overlap cluster containing exactly two consecutive
    selected intermediate circles.

    Strategy:
    1. Try to replace the two transitions with one direct transition between
       the outer corridors.
    2. If direct replacement is not possible, try to merge the two circles.
    """

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    indices = cluster["indices"]

    if len(indices) != 2:
        raise ValueError(
            f"Expected a pair cluster, but got indices {indices}."
        )

    index = indices[0]

    if indices[1] != index + 1:
        raise ValueError(
            f"Expected consecutive indices, but got {indices}."
        )

    choice1 = circle_choices_sequence[index]
    choice2 = circle_choices_sequence[index + 1]

    circle1 = select_preferred_circle(choice1)
    circle2 = select_preferred_circle(choice2)

    if circle1.turn_direction != circle2.turn_direction:
        raise ValueError(
            "solve_pair_same_turn_cluster was called on circles "
            "with different turn directions."
        )

    status = overlap_status_for_intermediate_circles(circle1, circle2)

    if not status["nominal_overlap"]:
        # Nothing to solve. Return unchanged sequence.
        return circle_choices_sequence

    print(
        f"Solving same-turn pair overlap between circles "
        f"{index} and {index + 1}"
    )

    # ------------------------------------------------------------
    # Retrieve corresponding corridors
    # ------------------------------------------------------------
    corridor1_index = choice1.corridor_index_start
    corridor2_index = choice1.corridor_index_end
    corridor3_index = choice2.corridor_index_end

    corridor1 = corridor_sequence[corridor1_index]
    corridor2 = corridor_sequence[corridor2_index]
    corridor3 = corridor_sequence[corridor3_index]

    # ------------------------------------------------------------
    # Step 1: try direct replacement between outer corridors
    # ------------------------------------------------------------
    replacement_choice = try_build_replacement_choice_between_corridors(
        corridor1,
        corridor3,
        index,
        vehicle,
    )

    if replacement_choice is not None:
        replacement_choice.corridor_index_start = choice1.corridor_index_start
        replacement_choice.corridor_index_end = choice2.corridor_index_end

        circle_choices_sequence.replace_two_with_one(
            index,
            replacement_choice,
        )

        reindex_intermediate_circle_choices_sequence(
            circle_choices_sequence
        )

        circle_choices_sequence = assign_preferred_candidates(
            circle_choices_sequence,
            start_pose,
            end_pose,
        )

        return circle_choices_sequence

    # ------------------------------------------------------------
    # Step 2: try same-turn merge
    # ------------------------------------------------------------
    R = vehicle.max_radius
    r = vehicle.width / 2

    small_circle1 = Circle(
        circle1.corner_point,
        r,
    )

    small_circle2 = Circle(
        circle2.corner_point,
        r,
    )

    distance_small_centers = compute_distance_two_points(
        small_circle1.center,
        small_circle2.center,
    )

    can_merge = distance_small_centers < 2 * (R - r)

    if not can_merge:
        print(
            "Same-turn pair cannot be merged: small clearance circles "
            "are too far apart."
        )
        return circle_choices_sequence

    merged_circle = compute_circle_internally_tangent_to_two_circles(
        small_circle1,
        small_circle2,
        radius=R,
        turn_direction=circle1.turn_direction,
    )

    merged_intermediate_circle = build_merged_intermediate_circle(
        circle1,
        circle2,
        corridor1,
        corridor2,
        corridor3,
        merged_circle,
        vehicle,
        index=None,
        s_max=0.0,
    )

    # Optional but useful for debugging
    merged_intermediate_circle.is_merged = True
    merged_intermediate_circle.merged_from = (
        choice1.corridor_index_start,
        choice1.corridor_index_end,
        choice2.corridor_index_start,
        choice2.corridor_index_end,
    )

    replacement_choice = IntermediateCircleChoice(
        candidates=[merged_intermediate_circle],
        index=index,
        corridor_index_start=choice1.corridor_index_start,
        corridor_index_end=choice2.corridor_index_end,
    )

    circle_choices_sequence.replace_two_with_one(
        index,
        replacement_choice,
    )

    reindex_intermediate_circle_choices_sequence(
        circle_choices_sequence
    )

    circle_choices_sequence = assign_preferred_candidates(
        circle_choices_sequence,
        start_pose,
        end_pose,
    )

    if debug_plot:
        figure = plot_corridors(corridor_sequence, plot_vectors=True)
        ax = plt.gca()

        plot_intermediate_circle_choices(
            ax,
            circle_choices_sequence,
            vehicle.width / 2,
        )

        angle_array = np.linspace(0, 2 * np.pi, 200)

        plt.plot(
            small_circle1.center.x,
            small_circle1.center.y,
            "x",
            label="small_circle1 center",
        )

        plt.plot(
            small_circle1.center.x + small_circle1.radius * np.cos(angle_array),
            small_circle1.center.y + small_circle1.radius * np.sin(angle_array),
            "-",
            label="small_circle1",
        )

        plt.plot(
            small_circle2.center.x,
            small_circle2.center.y,
            "x",
            label="small_circle2 center",
        )

        plt.plot(
            small_circle2.center.x + small_circle2.radius * np.cos(angle_array),
            small_circle2.center.y + small_circle2.radius * np.sin(angle_array),
            "-",
            label="small_circle2",
        )

        plt.axis("equal")
        plt.legend()
        plt.show(block=True)

    return circle_choices_sequence


def solve_opposite_turn_overlaps(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
):
    """
    Resolve remaining overlaps between consecutive selected circles with
    opposite turn directions.

    This pass is executed after all same-turn merge opportunities have been
    processed.
    """

    index = 0

    while index < len(circle_choices_sequence) - 1:
        choice1 = circle_choices_sequence[index]
        choice2 = circle_choices_sequence[index + 1]

        circle1 = select_preferred_circle(choice1)
        circle2 = select_preferred_circle(choice2)

        status = overlap_status_for_intermediate_circles(circle1, circle2)
        nominal_overlap = status["nominal_overlap"]

        if not nominal_overlap:
            index += 1
            continue

        # In this second pass, ignore same-turn overlaps.
        # They were already handled in the first pass.
        if circle1.turn_direction == circle2.turn_direction:
            index += 1
            continue

        print(
            f"Detected opposite-turn overlap between circles at indices "
            f"{index} and {index + 1}"
        )

        corridor1 = corridor_sequence[choice1.corridor_index_start]
        corridor2 = corridor_sequence[choice1.corridor_index_end]
        corridor3 = corridor_sequence[choice2.corridor_index_end]

        # Check opposite edges
        edge1 = circle1.edge_pair[1]
        edge2 = circle2.edge_pair[0]

        allowed_opposite_pairs = {(0, 2), (2, 0), (1, 3), (3, 1)}

        if (edge1, edge2) not in allowed_opposite_pairs:
            raise ValueError(
                "Expected the two circles to be connected through opposite edges of the middle corridor."
            )

        new_circle1 = build_circle_overlap_opposite_turn_direction(
            circle1,
            corridor2,
            vehicle,
        )

        new_circle2 = build_circle_overlap_opposite_turn_direction(
            circle2,
            corridor2,
            vehicle,
        )

        new_choice1 = IntermediateCircleChoice(
            candidates=[new_circle1],
            index=choice1.index,
            corridor_index_start=choice1.corridor_index_start,
            corridor_index_end=choice1.corridor_index_end,
        )

        new_choice2 = IntermediateCircleChoice(
            candidates=[new_circle2],
            index=choice2.index,
            corridor_index_start=choice2.corridor_index_start,
            corridor_index_end=choice2.corridor_index_end,
        )

        circle_choices_sequence.choices[index] = new_choice1
        circle_choices_sequence.choices[index + 1] = new_choice2

        reindex_intermediate_circle_choices_sequence(
            circle_choices_sequence
        )

        circle_choices_sequence = assign_preferred_candidates(
            circle_choices_sequence,
            start_pose,
            end_pose,
        )

        # Re-check from previous pair, because changing these circles may
        # affect overlap with the previous circle too.
        index = max(index - 1, 0)

    return circle_choices_sequence

# def solve_circles_overlap(
#     circle_choices_sequence,
#     corridor_sequence,
#     vehicle,
#     start_pose,
#     end_pose,
# ):
#     """
#     Resolve overlaps between consecutive selected intermediate circles.

#     For ambiguous choices, the preferred candidate is selected.
#     """

#     index = 0

#     while index < len(circle_choices_sequence) - 1:
#         # for each pair of consecutive choices, check for overlap

#         choice1 = circle_choices_sequence[index]
#         choice2 = circle_choices_sequence[index + 1]

#         circle1 = select_preferred_circle(choice1)
#         circle2 = select_preferred_circle(choice2)

#         status = overlap_status_for_intermediate_circles(
#             circle1,
#             circle2,
#         )

#         nominal_overlap = status["nominal_overlap"]
#         max_shift_overlap = status["max_shift_overlap"] 
#         can_overlap = status["can_overlap"]

#         if not nominal_overlap:
#             index += 1
#             continue

#         print(
#             f"Detected overlap between circles at indices "
#             f"{index} and {index + 1}"
#         )

#         # try replacing the two intermediate circles with one direct circle
#         # Check if the middle corridor is unnecessary
#         corridor1_index = choice1.corridor_index_start
#         corridor3_index = choice2.corridor_index_end

#         corridor1 = corridor_sequence[corridor1_index]
#         corridor3 = corridor_sequence[corridor3_index]

#         # figure = plot_corridors(corridor_sequence)
#         # angle_array = np.linspace(0, 2*np.pi, 200)
#         # plot_corridors([corridor1, corridor3], color="orange", linewidth=3, figure=figure)
#         # plt.plot(circle1.center.x, circle1.center.y, "x", label="circle1 center"
#         #          , figure=figure)
#         # plt.plot(circle1.center.x + circle1.radius * np.cos(angle_array), circle1.center.y + circle1.radius * np.sin(angle_array), "-", label="circle1", figure=figure)
#         # plt.plot(circle2.center.x, circle2.center.y, "x", label="circle2 center", figure=figure)
#         # plt.plot(circle2.center.x + circle2.radius * np.cos(angle_array), circle2.center.y + circle2.radius * np.sin(angle_array), "-", label="circle2", figure=figure)
#         # plt.show(block = True)

#         replacement_choice = try_build_replacement_choice_between_corridors(
#             corridor1,
#             corridor3,
#             index,
#             vehicle,
#         )

#         if replacement_choice is not None:
#             replacement_choice.corridor_index_start = choice1.corridor_index_start
#             replacement_choice.corridor_index_end = choice2.corridor_index_end

#             circle_choices_sequence.replace_two_with_one(
#                 index,
#                 replacement_choice,
#             )

#             reindex_intermediate_circle_choices_sequence(
#                 circle_choices_sequence
#             )

#             circle_choices_sequence = assign_preferred_candidates(
#                 circle_choices_sequence,
#                 start_pose,
#                 end_pose,
#             )

#             # Re-check from previous local neighborhood
#             index = max(index - 1, 0)
#             continue

#         # If replacement is not possible, shift circles with rule based on turn directions
#         else:
#             middle_corridor_index = choice1.corridor_index_end
#             corridor2 = corridor_sequence[middle_corridor_index]
#             if circle1.turn_direction == circle2.turn_direction:
#                 R = vehicle.max_radius
#                 # if (circle1.s != 0 and circle2.s == 0) or (circle1.s == 0 and circle2.s != 0):
#                 #     s1, s2 = 0, 0
#                 # else: 
#                 #     s1 = circle1.s
#                 #     s2 = circle2.s

#                 # center1 = intermediate_circle_center_at_s(circle1, s1)
#                 # center2 = intermediate_circle_center_at_s(circle2, s2)

#                 small_circle1 = Circle(
#                     circle1.corner_point,
#                     vehicle.width / 2,
#                 )

#                 small_circle2 = Circle(
#                     circle2.corner_point,
#                     vehicle.width / 2,
#                 )

#                 # p_critical1 = Point(
#                 #     center1.x + R * cos(circle1.bisector_direction),
#                 #     center1.y + R * sin(circle1.bisector_direction),
#                 # )

#                 # p_critical2 = Point(
#                 #     center2.x + R * cos(circle2.bisector_direction),
#                 #     center2.y + R * sin(circle2.bisector_direction),
#                 # )

#                 if (
#                     compute_distance_two_points(small_circle1.center, small_circle2.center) < 2* (R-vehicle.width/2)
#                     # and compute_distance_two_points(p_critical2, circle1.center) < R
#                     # and compute_distance_two_points(p_critical1, p_critical2) < 2 * R
#                 ):
#                     # merged_circle = compute_circle_through_two_points_with_radius(
#                     #     p_critical1,
#                     #     p_critical2,
#                     #     R,
#                     #     circle1.turn_direction,
#                     # )


#                     merged_circle = compute_circle_internally_tangent_to_two_circles(
#                         small_circle1,
#                         small_circle2,
#                         radius=vehicle.max_radius,
#                         turn_direction=circle1.turn_direction,
#                     )

#                     figure = plot_corridors(corridor_sequence)
#                     angle_array = np.linspace(0, 2*np.pi, 200)
#                     ax = figure.gca()
#                     plt.plot(circle1.center.x, circle1.center.y, "x", label="circle1 center")
#                     plt.plot(circle1.center.x + circle1.radius * np.cos(angle_array), circle1.center.y + circle1.radius * np.sin(angle_array), "-", label="circle1")
#                     plt.plot(circle2.center.x, circle2.center.y, "x", label="circle2 center")
#                     plt.plot(circle2.center.x + circle2.radius * np.cos(angle_array), circle2.center.y + circle2.radius * np.sin(angle_array), "-", label="circle2")
#                     plt.plot(circle1.corner_point.x + vehicle.width/2 * np.cos(angle_array), circle1.corner_point.y + vehicle.width/2 * np.sin(angle_array), "-", label="circle1 corner point")
#                     plt.plot(circle2.corner_point.x + vehicle.width/2 * np.cos(angle_array), circle2.corner_point.y + vehicle.width/2 * np.sin(angle_array), "-", label="circle2 corner point")

#                     plt.plot(merged_circle.center.x, merged_circle.center.y, "x", label="merged circle center")
#                     plt.plot(merged_circle.center.x + merged_circle.radius * np.cos(angle_array), merged_circle.center.y + merged_circle.radius * np.sin(angle_array), "-", label="merged circle")
#                     # plt.plot(p_critical1.x, p_critical1.y, "o", label="p_critical1")
#                     # plt.plot(p_critical2.x, p_critical2.y, "o", label="p_critical2")

#                     plt.legend()
#                     plt.show(block = True)
#                     print(f"center of merged circle: {merged_circle.center}, radius: {merged_circle.radius}")

#                     merged_intermediate_circle = build_merged_intermediate_circle(
#                             circle1,
#                             circle2,
#                             corridor2,
#                             merged_circle,
#                             vehicle,
#                             index=None,
#                             s_max=0.0,
#                         )

#                     print(f"center of merged intermediate circle: {merged_intermediate_circle.center}, radius: {merged_intermediate_circle.radius}")
#                     replacement_choice = IntermediateCircleChoice(
#                         candidates=[merged_intermediate_circle],
#                         index=index,
#                         corridor_index_start=choice1.corridor_index_start,
#                         corridor_index_end=choice2.corridor_index_end,
#                     )
                    
#                     circle_choices_sequence.replace_two_with_one(
#                         index,
#                         replacement_choice,
#                     )

#                     reindex_intermediate_circle_choices_sequence(
#                         circle_choices_sequence
#                     )

#                     circle_choices_sequence = assign_preferred_candidates(
#                         circle_choices_sequence,
#                         start_pose,
#                         end_pose,
#                     )
#                     print(f"center of merged intermediate circle after assign_preferred_candidates: {select_preferred_circle(circle_choices_sequence[index]).center}, radius: {select_preferred_circle(circle_choices_sequence[index]).radius}")
#                     index = max(index - 1, 0)

#                     figure = plot_corridors(corridor_sequence, plot_vectors=True)
#                     ax = plt.gca()
#                     plot_intermediate_circle_choices(
#                         ax,
#                         circle_choices_sequence,
#                         vehicle.width/2,
#                     )
#                     plt.show(block = True)
#                     continue

#             # overlap with opposite turn directions
#             else:
#                 corridor2 = corridor_sequence[choice1.corridor_index_end]
#                 orientation_middle_corridor = corridor2.tilt

#                 new_circle1 = build_circle_overlap_opposite_turn_direction(
#                     circle1,
#                     corridor2,
#                     vehicle,
#                 )

#                 new_circle2 = build_circle_overlap_opposite_turn_direction(
#                     circle2,
#                     corridor2,
#                     vehicle,
#                 )

#                 figure = plot_corridors(corridor_sequence)
#                 angle_array = np.linspace(0, 2*np.pi, 200)
#                 ax = figure.gca()
#                 plt.plot(circle1.center.x, circle1.center.y, "x", label="circle1 center")
#                 plt.plot(circle1.center.x + circle1.radius * np.cos(angle_array), circle1.center.y + circle1.radius * np.sin(angle_array), "-", label="circle1")
#                 plt.plot(circle2.center.x, circle2.center.y, "x", label="circle2 center")
#                 plt.plot(circle2.center.x + circle2.radius * np.cos(angle_array), circle2.center.y + circle2.radius * np.sin(angle_array), "-", label="circle2")
#                 plt.plot(new_circle1.center.x, new_circle1.center.y, "x", label="circle1 center")
#                 plt.plot(new_circle1.center.x + new_circle1.radius * np.cos(angle_array), new_circle1.center.y + new_circle1.radius * np.sin(angle_array), "-", label="circle1")
#                 plt.plot(new_circle2.center.x, new_circle2.center.y, "x", label="circle2 center")
#                 plt.plot(new_circle2.center.x + new_circle2.radius * np.cos(angle_array), new_circle2.center.y + new_circle2.radius * np.sin(angle_array), "-", label="circle2")

#                 new_choice1 = IntermediateCircleChoice(
#                     candidates=[new_circle1],
#                     index=choice1.index,
#                     corridor_index_start=choice1.corridor_index_start,
#                     corridor_index_end=choice1.corridor_index_end,
#                 )

#                 new_choice2 = IntermediateCircleChoice(
#                     candidates=[new_circle2],
#                     index=choice2.index,
#                     corridor_index_start=choice2.corridor_index_start,
#                     corridor_index_end=choice2.corridor_index_end,
#                 )

#                 circle_choices_sequence.choices[index] = new_choice1
#                 circle_choices_sequence.choices[index + 1] = new_choice2

#                 # Re-check from the previous pair, because changing these circles
#                 # may affect overlap with the previous circle too.
#                 index = max(index - 1, 0)
#                 continue

#         index += 1

#     figure = plot_corridors(corridor_sequence, plot_vectors=True)
#     ax = plt.gca()
#     plot_intermediate_circle_choices(
#         ax,
#         circle_choices_sequence,
#         vehicle.width/2,
#     )
#     plt.show(block = True)

#     return circle_choices_sequence


def reindex_intermediate_circle_choices_sequence(circle_choices_sequence):
    """
    Reassign consistent indices to choices and their candidate circles.
    """

    for i, choice in enumerate(circle_choices_sequence):
        choice.index = i

        for circle in choice:
            circle.index = i


def try_build_replacement_choice_between_corridors(
    corridor1,
    corridor3,
    index,
    vehicle,
):
    """
    Try to build one IntermediateCircleChoice directly between corridor1 and corridor3.

    Returns None if no valid direct transition exists.
    """

    try:
        tau = corridor1.compute_relative_turn_direction(corridor3)

        if tau != 0:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau,
                    index,
                    vehicle,
                )
            ]

        else:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau=1,
                    index=index,
                    vehicle=vehicle,
                ),
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau=-1,
                    index=index,
                    vehicle=vehicle,
                ),
            ]

        return IntermediateCircleChoice(
            candidates=candidates,
            index=index,
        )

    except ValueError:
        return None
    
    
def plot_line_from_w(ax, w, xlim, ylim, label=None, linestyle="-"):
    """
    Plot implicit line w[0]*x + w[1]*y + w[2] = 0.
    """
    wa, wb, wc = w

    if abs(wb) > 1e-9:
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = -(wa * xs + wc) / wb
        ax.plot(xs, ys, linestyle=linestyle, label=label)

    elif abs(wa) > 1e-9:
        x = -wc / wa
        ax.plot([x, x], ylim, linestyle=linestyle, label=label)


def plot_circle(ax, center, radius, label=None, linestyle="-"):
    theta = np.linspace(0, 2 * np.pi, 200)
    center = np.asarray(center, dtype=float)

    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)

    ax.plot(xs, ys, linestyle=linestyle, label=label)


def plot_shift_debug(
    corridor1,
    corridor2,
    corner_point,
    int_point,
    xc2,
    yc2,
    u,
    s_max_endpoint,
    s_max,
    rho,
    relevant_side_lines,
    plot_corridors,
):
    """
    Debug plot for the intermediate-circle shift limit.
    """

    figure = plot_corridors([corridor1, corridor2])
    ax = figure.gca()

    p0 = np.array([xc2, yc2], dtype=float)

    p_endpoint = p0 + s_max_endpoint * u
    p_final = p0 + s_max * u

    # Current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot original geometric segment corner -> int_point
    corner_point = np.asarray(corner_point, dtype=float)
    int_point = np.asarray(int_point, dtype=float)

    ax.plot(
        [corner_point[0], int_point[0]],
        [corner_point[1], int_point[1]],
        "--",
        label="corner_point to int_point",
    )

    # Plot center path up to endpoint limit
    ax.plot(
        [p0[0], p_endpoint[0]],
        [p0[1], p_endpoint[1]],
        "-",
        linewidth=2,
        label="circle-center path to s_max_endpoint",
    )

    # Plot accepted center path up to final s_max
    ax.plot(
        [p0[0], p_final[0]],
        [p0[1], p_final[1]],
        "-",
        linewidth=4,
        label="accepted center path to final s_max",
    )

    # Key points
    ax.scatter(p0[0], p0[1], marker="o", label="p0 = initial circle center")
    ax.scatter(p_endpoint[0], p_endpoint[1], marker="x", label="endpoint limit center")
    ax.scatter(p_final[0], p_final[1], marker="*", label="final s_max center")
    ax.scatter(corner_point[0], corner_point[1], marker="s", label="corner_point")
    ax.scatter(int_point[0], int_point[1], marker="s", label="int_point")

    # Plot circle of radius rho at start, endpoint, and final selected point
    plot_circle(ax, p0, rho, label="rho-circle at s=0", linestyle=":")
    plot_circle(ax, p_endpoint, rho, label="rho-circle at s_max_endpoint", linestyle=":")
    plot_circle(ax, p_final, rho, label="rho-circle at final s_max", linestyle="--")

    # Plot relevant side lines and their rho-offset lines
    for i, side_line in enumerate(relevant_side_lines):
        w = np.asarray(side_line, dtype=float)
        wa, wb, wc = w
        normal_norm = np.hypot(wa, wb)

        plot_line_from_w(
            ax,
            w,
            xlim,
            ylim,
            label=f"checked side line {i}",
            linestyle="-",
        )

        # Offset lines at distance rho:
        # wa*x + wb*y + wc = +/- rho * ||normal||
        offset = rho * normal_norm

        w_plus = np.array([wa, wb, wc - offset])
        w_minus = np.array([wa, wb, wc + offset])

        plot_line_from_w(
            ax,
            w_plus,
            xlim,
            ylim,
            label=f"rho-offset + side {i}",
            linestyle=":",
        )

        plot_line_from_w(
            ax,
            w_minus,
            xlim,
            ylim,
            label=f"rho-offset - side {i}",
            linestyle=":",
        )

        safe, points_at_rho, min_dist = segment_line_rho_intersections(
            w,
            p0=p0,
            u=u,
            s_max=s_max_endpoint,
            rho=rho,
        )

        for s_hit, p_hit in points_at_rho:
            ax.scatter(
                p_hit[0],
                p_hit[1],
                marker="D",
                label=f"side {i}: distance rho at s={s_hit:.3f}",
            )

        print(
            f"side {i}: safe={safe}, min_dist={min_dist:.4f}, "
            f"points_at_rho={[float(s) for s, _ in points_at_rho]}"
        )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True)

    return figure


def segment_line_rho_intersections(
    w,
    p0,
    u,
    s_max,
    rho,
    eps=1e-9,
):
    """
    Check whether the segment

        p(s) = p0 + s*u,    s in [0, s_max]

    stays at least distance rho away from the line

        w[0]*x + w[1]*y + w[2] = 0

    :param w:
        Line parameters [wa, wb, wc].
    :type w: array-like, shape (3,)

    :param p0:
        Initial point of the segment.
    :type p0: array-like, shape (2,)

    :param u:
        Segment direction vector.
        Preferably unit norm so that s corresponds to distance.
    :type u: array-like, shape (2,)

    :param s_max:
        Maximum parameter value along the segment.
    :type s_max: float

    :param rho:
        Safety distance.
    :type rho: float

    :param eps:
        Numerical tolerance.
    :type eps: float

    :return:
        Tuple containing:

        - safe:
            True if the whole segment stays at least rho away.

        - intersections:
            List containing tuples (s, point) where the segment
            reaches exactly distance rho from the line.

        - min_distance:
            Minimum distance between the segment and the line.
    :rtype:
        tuple(bool, list[tuple[float, numpy.ndarray]], float)
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    u = np.asarray(u, dtype=float)

    wa, wb, wc = w

    # Norm of the line normal vector
    normal_norm = np.hypot(wa, wb)

    if normal_norm < eps:
        raise ValueError(
            "Invalid line: normal vector has near-zero length."
        )

    # Signed line function along the segment:
    #
    # g(s) = wa*x(s) + wb*y(s) + wc
    #
    # with:
    #
    # p(s) = p0 + s*u
    #
    # Therefore:
    #
    # g(s) = g0 + s*g1
    #
    g0 = wa * p0[0] + wb * p0[1] + wc
    g1 = wa * u[0] + wb * u[1]

    # Distance threshold in implicit-line coordinates
    threshold = rho * normal_norm

    # Candidate points where minimum distance may occur
    candidates = [0.0, s_max]

    # If not parallel, check whether the segment crosses the line
    if abs(g1) > eps:
        s_cross = -g0 / g1

        if 0.0 <= s_cross <= s_max:
            candidates.append(s_cross)

    # Compute minimum distance
    min_abs_g = min(abs(g0 + s * g1) for s in candidates)
    min_distance = min_abs_g / normal_norm

    # Safety check
    safe = min_distance >= rho - eps

    intersections = []

    # Parallel case
    if abs(g1) < eps:

        # Entire segment lies exactly at distance rho
        if abs(abs(g0) - threshold) <= eps:

            intersections = [
                (0.0, p0.copy()),
                (s_max, p0 + s_max * u),
            ]

    # General case
    else:

        # Solve:
        #
        # g(s) = +threshold
        # g(s) = -threshold
        #
        for target in [threshold, -threshold]:

            s = (target - g0) / g1

            if -eps <= s <= s_max + eps:

                s_clamped = min(max(s, 0.0), s_max)

                point = p0 + s_clamped * u

                intersections.append((s_clamped, point))

    return safe, intersections, min_distance


def get_relevant_side_edges_side_side(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    if tau == 1:
        edge_idx = CorridorWorld.RGT
    else:
        edge_idx = CorridorWorld.LFT

    return corridor1.W[:, edge_idx], corridor2.W[:, edge_idx]


def get_relevant_side_edges_front_side(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    edge_idx1 = CorridorWorld.BCK
    if tau == 1:
        edge_idx2 = CorridorWorld.RGT
    else:
        edge_idx2 = CorridorWorld.LFT

    return corridor1.W[:, edge_idx1], corridor2.W[:, edge_idx2]


def get_relevant_side_edges_side_back(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    edge_idx2 = CorridorWorld.FWD
    if tau == 1:
        edge_idx1 = CorridorWorld.RGT
    else:
        edge_idx1 = CorridorWorld.LFT

    return corridor1.W[:, edge_idx1], corridor2.W[:, edge_idx2]


def not_ambiguous_circle_choices(intermediate_circles):
    """
    Convert an IntermediateCirclesSequence into an
    IntermediateCircleChoicesSequence with no ambiguity.

    Each circle is wrapped into a single-candidate choice.

    :param intermediate_circles: sequence of IntermediateCircle
    :type intermediate_circles: IntermediateCirclesSequence or iterable

    :return: sequence of choices (each with exactly one candidate)
    :rtype: IntermediateCircleChoicesSequence
    """
    choices = []

    for i, circle in enumerate(intermediate_circles):
        choice = IntermediateCircleChoice(
            candidates=[circle],
            index=i,
        )
        choices.append(choice)

    return IntermediateCircleChoicesSequence(choices)


def point_matches_any_corner(point, corners, tol=1e-9):
    for corner in corners:
        dx = point[0] - corner[0]
        dy = point[1] - corner[1]

        if dx * dx + dy * dy <= tol * tol:
            return True

    return False


def side_side_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    """
    Compute the intermediate circle for a side-side corridor intersection.

    This function handles the case where two consecutive corridors intersect
    through their side edges. If the front edge of corridor1 and the back edge
    of corridor2 intersect, then
    the circle center is computed from the corner point and that intersection
    point. Otherwise, the circle center is computed using the edge-based rule.

    :param corridor1: first corridor
    :param corridor2: second corridor
    :param tau: turn direction (+1 for left, -1 for right)
    :param edge_pair: pair of intersecting edge indices
    :param corner_point: selected corridor corner point [x, y]
    :param vehicle: vehicle model
    :param circle_index: optional index of the intermediate circle

    :return: intermediate circle
    :rtype: IntermediateCircle
    """
    R = vehicle.max_radius
    vehicle_width = vehicle.width

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

    # Side-Side open case: no intersection between front edge of corridor1 and back edge of corridor2
    if not intersects:
        xc2, yc2 = compute_center_coordinates_second_circle_according_to_edges(
            corner_point,
            tau,
            R,
            vehicle_width,
            0,
            edge_pair,
            corridor1,
            corridor2,
        )

        min_width, s_max = compute_min_width_s_max_corridor_pair(
            corridor1,
            corridor2,
            vehicle,
        )

        # Check for narrow corridor case. 
        # If it can be solved, return shifted circle, otherwise raise error about infeasibility of the turn.
        if corridor1.width < min_width:
            if corridor2.width < R + vehicle_width/2:
                raise ValueError(
                    "Corridor widths are too small for the vehicle to make the turn. "
                    f"Minimum required width: {min_width}"
                )
            else: 
                orientation_narrow_corridor = corridor1.tilt
                xc2 = corner_point[0] + (R - vehicle_width/2) * cos(orientation_narrow_corridor + tau * pi/2)
                yc2 = corner_point[1] + (R - vehicle_width/2) * sin(orientation_narrow_corridor + tau * pi/2)
                s_max = corridor1.width - vehicle_width
                door_point = Point(corner_point[0] + corridor1.width * cos(orientation_narrow_corridor - tau * pi/2),
                                   corner_point[1] + corridor1.width * sin(orientation_narrow_corridor - tau * pi/2))
                
                normal = corridor2.outward_normals[edge_pair[1]]
                theta0 = np.arctan2(normal[1], normal[0])
                
                return IntermediateCircle(
                    Point(xc2, yc2),
                    R,
                    Point(corner_point[0], corner_point[1]),
                    tau,
                    index=circle_index,
                    s_max=s_max,
                    edge_pair=edge_pair,
                    door_point=door_point,
                    door_type="45-degree modified narrow corridor1",
                    start_angle_arc = theta0,
                    rho = R - vehicle_width/2,
                )

        if corridor2.width < min_width:
            if corridor1.width < R + vehicle_width/2:
                raise ValueError(
                    "Corridor widths are too small for the vehicle to make the turn. "
                    f"Minimum required width: {min_width}"
                )
            else: 
                orientation_narrow_corridor = corridor2.tilt
                xc2 = corner_point[0] + (R - vehicle_width/2) * cos(orientation_narrow_corridor + tau * pi/2)
                yc2 = corner_point[1] + (R - vehicle_width/2) * sin(orientation_narrow_corridor + tau * pi/2)
                s_max = corridor2.width - vehicle_width
                door_point = Point(corner_point[0] + corridor2.width * cos(orientation_narrow_corridor - tau * pi/2),
                                   corner_point[1] + corridor2.width * sin(orientation_narrow_corridor - tau * pi/2))
                
                normal = corridor2.outward_normals[edge_pair[1]]
                theta0 = np.arctan2(normal[1], normal[0])

                return IntermediateCircle(
                    Point(xc2, yc2),
                    R,
                    Point(corner_point[0], corner_point[1]),
                    tau,
                    index=circle_index,
                    s_max=s_max,
                    edge_pair=edge_pair,
                    door_point=door_point,
                    door_type="45-degree modified narrow corridor2",
                    start_angle_arc = theta0,
                    rho = R - vehicle_width/2,
                )

        # Return regular side-side open case circle
        door_direction = atan2(corner_point[1] - yc2, corner_point[0] - xc2)
        door_point = Point(xc2 + (s_max + vehicle_width/2) * cos(door_direction),
                           yc2 + (s_max + vehicle_width/2) * sin(door_direction)
                           )
        normal = corridor2.outward_normals[edge_pair[1]]
        theta0 = np.arctan2(normal[1], normal[0])
        return IntermediateCircle(
            Point(xc2, yc2),
            R,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
            edge_pair=edge_pair,
            door_point=door_point,
            door_type="45-degree side-side",
            start_angle_arc = theta0,
            rho = R - vehicle_width/2,
        )

    # Side-Side closed case: front edge of corridor1 and back edge of corridor2 intersect
    xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
        corner_point,
        int_point,
        tau,
        R,
        vehicle_width,
    )

    length_section = compute_distance_two_points(corner_point, int_point)
    r = vehicle_width / 2
    rho = R + r

    if length_section < vehicle_width:
        raise ValueError(
            "The distance between the corner point and the intersection point "
            "is too small for the vehicle to make the turn. "
            f"Minimum required distance: {vehicle_width}"
        )

    s_max_endpoint = length_section - rho + (R-r)

    s_max_candidates = [s_max_endpoint]

    # Direction from corner_point to int_point
    u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
    u = u / np.linalg.norm(u)

    w1, w2 = get_relevant_side_edges_side_side(corridor1, corridor2, tau)
    relevant_side_lines = [w1, w2]
    p0 = np.array([xc2, yc2], dtype=float)

    for side_line in relevant_side_lines:
        safe, points_at_rho, min_dist = segment_line_rho_intersections(
            side_line,
            p0=p0,
            u=u,
            s_max=s_max_endpoint,
            rho=rho,
        )

        if not safe:
            valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_endpoint]

            if not valid_s:
                raise ValueError(
                    "The circle-center path violates a side constraint, "
                    "but no valid rho-boundary point was found."
                )

            s_limit = min(valid_s)
            s_max_candidates.append(s_limit)

    s_max = min(s_max_candidates)

    # fig = plot_shift_debug(
    #     corridor1=corridor1,
    #     corridor2=corridor2,
    #     corner_point=corner_point,
    #     int_point=int_point,
    #     xc2=xc2,
    #     yc2=yc2,
    #     u=u,
    #     s_max_endpoint=s_max_endpoint,
    #     s_max=s_max,
    #     rho=rho,
    #     relevant_side_lines=relevant_side_lines,
    #     plot_corridors=plot_corridors,
    # )

    # plt.show()

    normal = corridor2.outward_normals[edge_pair[1]]
    theta0 = np.arctan2(normal[1], normal[0])

    return IntermediateCircle(
        Point(xc2, yc2),
        R,
        Point(corner_point[0], corner_point[1]),
        tau,
        index=circle_index,
        s_max=s_max,
        edge_pair=edge_pair,
        door_point = int_point,
        door_type="diagonal side-side",
        start_angle_arc = theta0,
        rho = R - vehicle_width/2,
    )


def front_side_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    def build_circle_from_intersection(int_point, edge_pair, door_type = None, add_s_check = False):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        r = vehicle.width / 2
        R = vehicle.max_radius
        rho = R + r

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        s_max_endpoint = length_section - vehicle.width

        s_max_candidates = [s_max_endpoint]

        if add_s_check:
            # Direction from corner_point to int_point
            u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
            u = u / np.linalg.norm(u)

            w1, w2 = get_relevant_side_edges_front_side(corridor1, corridor2, tau)

            relevant_side_lines = [w1, w2]
            p0 = np.array([xc2, yc2], dtype=float)

            for side_line in relevant_side_lines:
                safe, points_at_rho, min_dist = segment_line_rho_intersections(
                    side_line,
                    p0=p0,
                    u=u,
                    s_max=s_max_endpoint,
                    rho=rho,
                )

                if not safe:
                    valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_endpoint]

                    if not valid_s:
                        raise ValueError(
                            "The circle-center path violates a side constraint, "
                            "but no valid rho-boundary point was found."
                        )

                    s_limit = min(valid_s)
                    s_max_candidates.append(s_limit)

        s_max = min(s_max_candidates)

        normal = corridor2.outward_normals[edge_pair[1]]
        theta0 = np.arctan2(normal[1], normal[0])

        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
            edge_pair=edge_pair,
            door_point=int_point,
            door_type=door_type,
            start_angle_arc = theta0,
            rho = R - vehicle.width/2,
        )

    # First attempt: front edge of corridor1 with other side edge of corridor2
    A1 = corners1[3]
    A2 = corners1[0]

    if edge_pair[1] == 1:
        B1 = corners2[2]
        B2 = corners2[3]
    else:
        B1 = corners2[1]
        B2 = corners2[0]

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        return build_circle_from_intersection(int_point, edge_pair, door_type="axis-aligned front-side")

    # Second attempt: side edge of corridor1 with back edge of corridor2
    # Here we need to check for s_max
    if edge_pair[1] == 1:
        A1 = corners1[2]
        A2 = corners1[3]
    else:
        A1 = corners1[1]
        A2 = corners1[0]

    B1 = corners2[2]
    B2 = corners2[1]

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        return build_circle_from_intersection(int_point, edge_pair, door_type="diagonal front-side", add_s_check=True)

    # Fallback: compute circle from edge-based rule
    xc2, yc2 = compute_center_coordinates_second_circle_according_to_edges(
        corner_point,
        tau,
        vehicle.max_radius,
        vehicle.width,
        0,
        edge_pair,
        corridor1,
        corridor2,
    )

    angle = atan2(
        yc2 - corner_point[1],
        xc2 - corner_point[0],
    )

    intersections = compute_line_corridor_intersections(
        corner_point,
        angle,
        corridor2,
        tol=1e-9,
    )

    if len(intersections) != 2:
        raise ValueError(
            "Unexpected number of intersections between the line from the "
            f"corner point to the circle center and the corridor: {len(intersections)}"
        )

    length = compute_distance_two_points(
        intersections[0],
        intersections[1],
    )

    if length < vehicle.width:
        raise ValueError(
            "The distance between the two intersections of the line from the "
            "corner point to the circle center with the corridor is too small "
            "for the vehicle to make the turn. "
            f"Minimum required distance: {vehicle.width}"
        )
    normal = corridor2.outward_normals[edge_pair[1]]
    theta0 = np.arctan2(normal[1], normal[0])
    return IntermediateCircle(
        Point(xc2, yc2),
        vehicle.max_radius,
        Point(corner_point[0], corner_point[1]),
        tau,
        index=circle_index,
        s_max=length - 2 * vehicle.max_radius,
        edge_pair=edge_pair,
        door_type="edge-based front-side fallback",
        start_angle_arc = theta0,
        rho = vehicle.max_radius - vehicle.width/2,
    )


def side_back_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    def build_circle_from_intersection(int_point, edge_pair, door_type=None, add_s_check = False):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        r = vehicle.width / 2
        R = vehicle.max_radius
        rho = R + r

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        s_max_end_point = length_section - vehicle.width
        s_max_candidates = [s_max_end_point]

        if add_s_check:
            # Direction from corner_point to int_point
            u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
            u = u / np.linalg.norm(u)

            w1, w2 = get_relevant_side_edges_side_back(corridor1, corridor2, tau)

            relevant_side_lines = [w1, w2]
            p0 = np.array([xc2, yc2], dtype=float)

            for side_line in relevant_side_lines:
                safe, points_at_rho, min_dist = segment_line_rho_intersections(
                    side_line,
                    p0=p0,
                    u=u,
                    s_max=s_max_end_point,
                    rho=rho,
                )

                if not safe:
                    valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_end_point]

                    if not valid_s:
                        raise ValueError(
                            "The circle-center path violates a side constraint, "
                            "but no valid rho-boundary point was found."
                        )

                    s_limit = min(valid_s)
                    s_max_candidates.append(s_limit)

        s_max = min(s_max_candidates)

        normal = corridor2.outward_normals[edge_pair[1]]
        theta0 = np.arctan2(normal[1], normal[0])
        
        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
            edge_pair=edge_pair,
            door_point=int_point,
            door_type=door_type,
            start_angle_arc = theta0,
            rho = R - vehicle.width/2,
        )

    # First attempt: other side edge of corridor1 with back edge of corridor2
    if edge_pair[0] == 1:
        A1 = corners1[2]  # bottom left
        A2 = corners1[3]  # top left
    else:
        A1 = corners1[1]  # bottom right
        A2 = corners1[0]  # top right

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
        return build_circle_from_intersection(int_point, edge_pair, door_type="axis-aligned side-back")

    # Second attempt: front edge of corridor1 with side edge of corridor2
    A1 = corners1[3]  # top left
    A2 = corners1[0]  # top right

    if edge_pair[0] == 1:
        B1 = corners2[2]  # bottom left
        B2 = corners2[3]  # top left
    else:
        B1 = corners2[1]  # bottom right
        B2 = corners2[0]  # top right

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if not intersects:
        raise ValueError(
            "Could not find a valid intersection point for the side-back case. "
            "This may indicate invalid corridor geometry or an unexpected edge pair."
        )

    return build_circle_from_intersection(int_point, edge_pair, door_type="diagonal side-back")
        

def assign_preferred_turn_directions(circle_choices_sequence):
    """
    Assign preferred turn directions to ambiguous circle choices.

    For each ambiguous block, look at the nearest nonzero turn direction
    before and after the block. If they are equal, use that direction as
    the preferred candidate direction for all choices in the block.
    Otherwise, leave the preference as None.
    """
    turn_directions = []

    for choice in circle_choices_sequence:
        if choice.is_ambiguous:
            turn_directions.append(0)
        else:
            turn_directions.append(choice.first.turn_direction)

    n = len(turn_directions)
    i = 0

    while i < n:
        if turn_directions[i] != 0:
            i += 1
            continue

        # Found start of ambiguous block
        start = i
        while i < n and turn_directions[i] == 0:
            i += 1
        end = i - 1

        previous_turn = turn_directions[start - 1] if start > 0 else None
        next_turn = turn_directions[i] if i < n else None

        if previous_turn is not None and previous_turn == next_turn:
            preferred_turn = previous_turn
        else:
            preferred_turn = None

        for j in range(start, end + 1):
            circle_choices_sequence[j].preferred_turn_direction = preferred_turn

    return circle_choices_sequence


def build_intermediate_circle_candidate(
    corridor1,
    corridor2,
    tau,
    index,
    vehicle,
):
    corner_point, edge_pair = get_corner_point_and_intersecting_edges(
        corridor1,
        corridor2,
        tau,
    )

    if corner_point is None or edge_pair == []:
        raise ValueError(
            f"No valid corner point found for turn direction {tau}."
        )
    
    # Side-side case
    if edge_pair in ((1, 1), (3, 3)):
        return side_side_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    # Front-side case
    if edge_pair in ((0, 1), (0, 3)):
        return front_side_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    # Side-back case
    if edge_pair in ((3, 2), (1, 2)):
        return side_back_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    # Unexpected case
    # plot_corridors([corridor1, corridor2], plot_vectors=True)
    # plt.plot(corner_point[0], corner_point[1], marker="o", label="corner point")
    # plt.legend()
    # plt.title(f"Unexpected edge pair: {edge_pair}, tau={tau}")
    # plt.show()

    raise ValueError(f"Invalid edge pair: {edge_pair}")


def create_intermediate_circle_choice_sequence(
    corridor_list,
    vehicle,
    start_pose,
    end_pose,
):
    
    validate_corridor_sequence(
        corridor_list=corridor_list,
        vehicle=vehicle,
        min_centerline_distance=2.0 * vehicle.max_radius,
        plot_invalid = True,
    )
    choices = []
    left_turn = 1
    right_turn = -1

    for i in range(len(corridor_list) - 1):
        # For each pair of consecutive corridors,
        # compute the turn direction tau and build
        # the corresponding intermediate circle candidates.
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        tau = corridor1.compute_relative_turn_direction(corridor2)

        if tau != 0:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    tau,
                    i,
                    vehicle,
                )
            ]
        else:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    left_turn,
                    i,
                    vehicle,
                ),
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    right_turn,
                    i,
                    vehicle,
                ),
            ]

        # choices.append(
        #     IntermediateCircleChoice(
        #         candidates=candidates,
        #         index=i,
        #     )
        # )

        choices.append(
            IntermediateCircleChoice(
                candidates=candidates,
                index=i,
                corridor_index_start=i,
                corridor_index_end=i + 1,
            )
        )

    choices_sequence = IntermediateCircleChoicesSequence(choices)

    choices_sequence = assign_preferred_candidates(
        choices_sequence,
        start_pose,
        end_pose,
    )

    # figure = plot_corridors(corridor_list, plot_vectors=True)
    # plt.show(block = True)
    # for i in range(len(corridor_list)-1):
    #     plot_corridors([corridor_list[i], corridor_list[i+1]])

    # ax = plt.gca()

    # plot_intermediate_circle_choices(
    #     ax,
    #     choices_sequence,
    #     vehicle.width/2,
    # )
    # plt.show(block = True)

    choices_sequence = solve_circles_overlap(
        choices_sequence,
        corridor_list,
        vehicle,
        start_pose,
        end_pose,
        )

    return choices_sequence


def detect_ambiguous_blocks(circle_choices_sequence):
    ambiguous_blocks = []
    block = []

    for choice in circle_choices_sequence:
        if choice.is_ambiguous:
            block.append(choice.index)
        # Close current block when a non-ambiguous choice is found
        else:
            if block:
                ambiguous_blocks.append(block)
                block = []

    # Add final block if the sequence ends with ambiguous choices
    if block:
        ambiguous_blocks.append(block)

    return ambiguous_blocks


def assign_preferred_candidates(circle_choices_sequence, start_pose, end_pose):
    """
    Assign a preferred turn direction (±1) to all ambiguous circle choices.

    The decision is based on local geometry:
    previous point → door midpoint → next point.

    :param circle_choices_sequence: sequence of IntermediateCircleChoice
    :param start_pose: [x, y, theta]
    :param end_pose: [x, y, theta]
    :return: updated circle_choices_sequence
    """

    n_circles = len(circle_choices_sequence)
    ambiguous_blocks = detect_ambiguous_blocks(circle_choices_sequence)

    start_point = Point(start_pose[0], start_pose[1])
    end_point = Point(end_pose[0], end_pose[1])

    for block in ambiguous_blocks:

        # -----------------------------------------
        # Step 1 — compute mid-door points
        # -----------------------------------------
        mid_door_sequence = []

        for index in block:
            choice = circle_choices_sequence[index]

            door_A = choice[0].corner_point
            door_B = choice[1].corner_point

            mid_door = Point(
                (door_A.x + door_B.x) / 2,
                (door_A.y + door_B.y) / 2,
            )

            mid_door_sequence.append(mid_door)

        # -----------------------------------------
        # Step 2 — assign preferred turn
        # -----------------------------------------
        for i, choice_index in enumerate(block):

            p_curr = mid_door_sequence[i]

            # ---- previous reference ----
            if i > 0:
                p_prev = mid_door_sequence[i - 1]
            else:
                if choice_index > 0:
                    previous_circle = circle_choices_sequence[
                        choice_index - 1
                    ].first

                    if compute_distance_two_points(
                        previous_circle.center,
                        p_curr,
                    ) < previous_circle.radius:
                        p_prev = previous_circle.center
                        
                    else:
                      
                        p_prev = select_tangency_point_from_point_circle(
                            p_curr,
                            previous_circle,
                            turn_direction=-previous_circle.turn_direction,
                        )
                else:
                    p_prev = start_point

            # ---- next reference ----
            if i < len(block) - 1:
                p_next = mid_door_sequence[i + 1]
            else:
                if choice_index < n_circles - 1:
                    next_circle = circle_choices_sequence[choice_index + 1].first

                    if compute_distance_two_points(
                        next_circle.center,
                        p_curr,
                    ) < next_circle.radius:
                        p_next = next_circle.center
                    else:

                        p_next = select_tangency_point_from_point_circle(
                            p_curr,
                            next_circle,
                        )
                else:
                    p_next = end_point

            # ---- compute turn ----
            # plt.plot([p_prev.x, p_curr.x, p_next.x], [p_prev.y, p_curr.y, p_next.y], "ro-"
            #          )
            # plt.show(block = True)
            preferred_turn = compute_turn_direction_from_three_points(
                p_prev,
                p_curr,
                p_next,
            )

            choice = circle_choices_sequence[choice_index]

            # fallback if perfectly aligned
            if preferred_turn == 0:
                preferred_turn = choice[0].turn_direction

            choice.preferred_turn_direction = preferred_turn

        # -----------------------------------------
        # Step 3 — update circles (apply heuristic)
        # -----------------------------------------
        for choice_index in block:
            choice = circle_choices_sequence[choice_index]

            if choice.preferred_turn_direction is None:
                continue

            for circle in choice:
                if circle.turn_direction == choice.preferred_turn_direction:
                    s_new = circle.s_max / 2
                    circle.update_s(s=s_new)

    return circle_choices_sequence
 

def plot_intermediate_circle_choices(
    ax,
    circle_choices_sequence,
    r=0,
    plot_arcs=False,
):
    """
    Plot all intermediate circle candidates at their nominal position s=0.

    The red segment shows the maximum shift from s=0 to s=s_max.
    The door associated with each intermediate circle is also plotted when
    both door endpoints are available.

    Parameters
    ----------
    ax : matplotlib axis
        Axis where the circles are plotted.

    circle_choices_sequence : IntermediateCircleChoicesSequence
        Sequence of intermediate circle choices.

    r : float, optional
        Extra radius used to plot the enlarged circle of radius R + r.

    plot_arcs : bool, optional
        If True, plot the stored arc_coordinates of each IntermediateCircle,
        when available. Default is False.
    """

    def _point_xy(p):
        """
        Return x, y from either a Point object with .x/.y
        or an indexable point like [x, y].
        """
        if hasattr(p, "x") and hasattr(p, "y"):
            return p.x, p.y
        return p[0], p[1]

    for i, choice in enumerate(circle_choices_sequence):
        for j, circle in enumerate(choice):
            # Plot nominal circle center, not current shifted center
            if hasattr(circle, "canonical_center"):
                xc = circle.canonical_center.x
                yc = circle.canonical_center.y
            else:
                xc = circle.center.x
                yc = circle.center.y

            R = circle.radius
            rho = R + r

            is_preferred = (
                choice.is_ambiguous
                and choice.preferred_turn_direction is not None
                and circle.turn_direction == choice.preferred_turn_direction
            )

            if choice.is_ambiguous:
                if is_preferred:
                    color = "green"
                    linestyle = "-"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    color = "orange"
                    linestyle = "--"
                    linewidth = 1.0
                    alpha = 0.6
            else:
                color = "blue"
                linestyle = "-"
                linewidth = 1.5
                alpha = 0.8

            circ = plt.Circle(
                (xc, yc),
                rho,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

            circ_path = plt.Circle(
                (xc, yc),
                R,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

            ax.add_patch(circ)
            ax.add_patch(circ_path)

            ax.plot(xc, yc, "o", color=color, alpha=alpha)

            ax.text(
                xc,
                yc,
                f"{i}:{j}",
                fontsize=8,
                color=color,
            )

            # ------------------------------------------------------------
            # Plot stored arc coordinates, if requested and available
            # ------------------------------------------------------------
            if plot_arcs and getattr(circle, "arc_coordinates", None) is not None:
                arc_coordinates = circle.arc_coordinates

                if len(arc_coordinates) > 0:
                    xs = []
                    ys = []

                    for p in arc_coordinates:
                        px, py = _point_xy(p)
                        xs.append(px)
                        ys.append(py)

                    ax.plot(
                        xs,
                        ys,
                        color="magenta",
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.95,
                    )

                    # Optional: plot arc start and end points
                    ax.plot(
                        [xs[0], xs[-1]],
                        [ys[0], ys[-1]],
                        "o",
                        color="magenta",
                        markersize=4,
                        alpha=0.95,
                    )

            # ------------------------------------------------------------
            # Plot door segment, if both door endpoints are available
            # ------------------------------------------------------------
            door_left = getattr(circle, "door_point_left", None)
            door_right = getattr(circle, "door_point_right", None)

            if door_left is not None and door_right is not None:
                door_left_x, door_left_y = _point_xy(door_left)
                door_right_x, door_right_y = _point_xy(door_right)

                ax.plot(
                    [door_left_x, door_right_x],
                    [door_left_y, door_right_y],
                    color="red",
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                )

                ax.plot(
                    [door_left_x, door_right_x],
                    [door_left_y, door_right_y],
                    "o",
                    color="red",
                    markersize=4,
                    alpha=0.9,
                )

                # Optional: label the door type at the midpoint
                if getattr(circle, "door_type", None) is not None:
                    mx = 0.5 * (door_left_x + door_right_x)
                    my = 0.5 * (door_left_y + door_right_y)

                    ax.text(
                        mx,
                        my,
                        str(circle.door_type),
                        fontsize=7,
                        color="red",
                        ha="center",
                        va="center",
                    )

            if hasattr(circle, "s_max") and hasattr(circle, "bisector_direction"):
                x_shift_end = xc + circle.s_max * cos(circle.bisector_direction)
                y_shift_end = yc + circle.s_max * sin(circle.bisector_direction)

                ax.plot(
                    [xc, x_shift_end],
                    [yc, y_shift_end],
                    "r-",
                    linewidth=1.5,
                    alpha=0.8,
                )

                circ_extreme = plt.Circle(
                    (x_shift_end, y_shift_end),
                    rho,
                    fill=False,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                circ_extreme_path = plt.Circle(
                    (x_shift_end, y_shift_end),
                    R,
                    fill=False,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                ax.add_patch(circ_extreme)
                ax.add_patch(circ_extreme_path)

                ax.plot(
                    x_shift_end,
                    y_shift_end,
                    "x",
                    color="red",
                    alpha=0.7,
                )

        if choice.is_ambiguous and len(choice) == 2:
            c1 = choice[0]
            c2 = choice[1]

            c1_center = c1.canonical_center if hasattr(c1, "canonical_center") else c1.center
            c2_center = c2.canonical_center if hasattr(c2, "canonical_center") else c2.center

            ax.plot(
                [c1_center.x, c2_center.x],
                [c1_center.y, c2_center.y],
                "k--",
                linewidth=1,
                alpha=0.6,
            )


def selected_sequence_from_preferences(circle_choices_sequence):
    selected_circles = []

    for choice in circle_choices_sequence:
        if not choice.is_ambiguous:
            selected_circles.append(choice.first)
            continue

        selected_circle = None

        if choice.preferred_turn_direction is not None:
            for circle in choice:
                if circle.turn_direction == choice.preferred_turn_direction:
                    selected_circle = circle
                    break

        if selected_circle is None:
            selected_circle = choice.first

        selected_circles.append(selected_circle)

    return IntermediateCirclesSequence(selected_circles)


def solve_pair_opposite_turn_cluster(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
    cluster,
    debug_plot=False,
):
    """
    Resolve one opposite-turn overlap cluster containing exactly two
    consecutive selected intermediate circles.

    Strategy:
    Rebuild both circles using the outward normals of the middle corridor.
    This separates the opposite-turn pair by moving each circle center to the
    side associated with its middle-corridor edge.
    """

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    indices = cluster["indices"]

    if len(indices) != 2:
        raise ValueError(
            f"Expected a pair cluster, but got indices {indices}."
        )

    index = indices[0]

    if indices[1] != index + 1:
        raise ValueError(
            f"Expected consecutive indices, but got {indices}."
        )

    choice1 = circle_choices_sequence[index]
    choice2 = circle_choices_sequence[index + 1]

    circle1 = select_preferred_circle(choice1)
    circle2 = select_preferred_circle(choice2)

    if circle1.turn_direction == circle2.turn_direction:
        raise ValueError(
            "solve_pair_opposite_turn_cluster was called on circles "
            "with the same turn direction."
        )

    status = overlap_status_for_intermediate_circles(circle1, circle2)

    if not status["nominal_overlap"]:
        return circle_choices_sequence

    print(
        f"Solving opposite-turn pair overlap between circles "
        f"{index} and {index + 1}"
    )

    # ------------------------------------------------------------
    # Retrieve corresponding corridors
    # ------------------------------------------------------------
    corridor1_index = choice1.corridor_index_start
    corridor2_index = choice1.corridor_index_end
    corridor3_index = choice2.corridor_index_end

    corridor1 = corridor_sequence[corridor1_index]
    corridor2 = corridor_sequence[corridor2_index]
    corridor3 = corridor_sequence[corridor3_index]

    # ------------------------------------------------------------
    # Check that the two circles are connected through opposite
    # edges of the middle corridor
    # ------------------------------------------------------------
    if circle1.edge_pair is None or circle2.edge_pair is None:
        raise ValueError(
            "Both circles must have edge_pair defined for opposite-turn repair."
        )

    edge1 = circle1.edge_pair[1]
    edge2 = circle2.edge_pair[0]

    allowed_opposite_pairs = {
        (0, 2),
        (2, 0),
        (1, 3),
        (3, 1),
    }

    if (edge1, edge2) not in allowed_opposite_pairs:
        raise ValueError(
            "Expected the two circles to be connected through opposite "
            "edges of the middle corridor."
        )

    # ------------------------------------------------------------
    # Rebuild both circles
    # ------------------------------------------------------------
    new_circle1 = build_circle_overlap_opposite_turn_direction(
        circle1,
        corridor2,
        vehicle,
    )

    new_circle2 = build_circle_overlap_opposite_turn_direction(
        circle2,
        corridor2,
        vehicle,
    )

    new_choice1 = IntermediateCircleChoice(
        candidates=[new_circle1],
        index=choice1.index,
        corridor_index_start=choice1.corridor_index_start,
        corridor_index_end=choice1.corridor_index_end,
    )

    new_choice2 = IntermediateCircleChoice(
        candidates=[new_circle2],
        index=choice2.index,
        corridor_index_start=choice2.corridor_index_start,
        corridor_index_end=choice2.corridor_index_end,
    )

    circle_choices_sequence.choices[index] = new_choice1
    circle_choices_sequence.choices[index + 1] = new_choice2

    reindex_intermediate_circle_choices_sequence(
        circle_choices_sequence
    )

    circle_choices_sequence = assign_preferred_candidates(
        circle_choices_sequence,
        start_pose,
        end_pose,
    )

    if debug_plot:
        figure = plot_corridors(corridor_sequence, plot_vectors=True)
        ax = plt.gca()

        plot_intermediate_circle_choices(
            ax,
            circle_choices_sequence,
            vehicle.width / 2,
        )

        plt.axis("equal")
        plt.legend()
        plt.show(block=True)

    return circle_choices_sequence


