import warnings

from .plot_helpers import plot_corridors, plot_analytical_trajectory
from .intermediate_circles_choice import selected_sequence_from_preferences
from .invert_inputs import invert_inputs_all
from .recovery_maneuvers_bicycle import (
    _compute_alignment_state,
    compute_segment_to_next_corridor,
    compute_segment_deeper_into_corridor,
    )
from .geometry_operations import (
    compute_angular_difference,
    wrapPositiveAngle,
    compute_angular_difference_with_turn_direction,
    compute_distance_two_points
)
from .poses import (
    relative_to_absolute_pose,
    absolute_to_relative_pose,
)
from .intersections import (
    circle_intersection,
    check_intersection_case,
)

from .primitives import (
    correct_angles,
    invert_maneuvers,
    compute_segment_between_two_circles_objects,
    compute_arc_from_two_tangents_objects,
    build_segment_with_reversal,
    compute_reversal_maneuver_on_segment,
)

from .corridor_geometry import (
    shrink_corridor_list,
)
from .axis_aligned_int_circle_sequence import (
    detect_tangent_intersections_blocks,
    update_intermediate_circle_centers_from_extended_sequence,
    solve_tangent_intersections_blocks_centers,
    print_failed_tangent_shift_summary,
    update_turn_direction_circles,
)

from .corridor_sequence_validity import check_bicycle_boundary_pose_separation
from .pose_to_circle_bicycle import compute_initial_turn_direction, compute_traj_to_circle_bicycle, compute_traj_to_circle_bicycle_with_fixed_forward_circle, compute_full_traj_bicycle_with_two_fixed_circles


from ..geometry import IntermediateCircle, Point, Pose, Circle, IntermediateCirclesSequence
from ..trajectory import BackwardArc, CurvilinearArcUnicycle, LinearSegmentUnicycle
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi, sqrt, atan2, asin


def get_active_intermediate_circles(intermediate_circles):
    """
    Return intermediate circles that are active in final trajectory construction.

    :param intermediate_circles: Intermediate circles.
    :type intermediate_circles: IntermediateCirclesSequence | list

    :return: Active intermediate circles.
    :rtype: list
    """
    active_circles = []

    for circle in intermediate_circles:
        if getattr(circle, "skip", False):
            continue

        active_circles.append(circle)

    return active_circles


def compute_P_mid(intermediate_circles, bicycle, return_active_circles=False):
    """
    Compute tangent segments between consecutive active intermediate circles.
    """
    active_circles = get_active_intermediate_circles(intermediate_circles)

    n_segments = len(active_circles) - 1

    if n_segments <= 0:
        if return_active_circles:
            return [], active_circles
        return []

    segments = []

    for index in range(n_segments):
        segment = compute_segment_between_two_circles_objects(
            active_circles[index],
            active_circles[index + 1],
            bicycle,
            start_circle_index=active_circles[index].index,
            end_circle_index=active_circles[index + 1].index,
        )

        segments.append(segment)

    if return_active_circles:
        return segments, active_circles

    return segments, active_circles



 


    
















def shift_circles_bicycle(
    intermediate_circles,
    # intermediate_circles_choices,
    segments,
    bicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
):
    """
    Adjust intermediate circle positions to resolve intersections
    between consecutive segments.

    For N corridors:
      - there are (N - 1) intermediate circles
      - there are N connecting segments

    If an intersection is resolved at circle i, the checking process
    restarts from circle max(i-1, 0), since the update may affect the
    neighboring intersection on the left.
    """
    initial_step = 0.1
    step = 0.05
    i = 0
    tried_other_side = [False] * len(intermediate_circles)

    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure=figure)
    # for circle in intermediate_circles:
    #     plt.plot(circle.center.x, circle.center.y, 'ro')    
    #     plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
    # plt.show(block = True)
    while i < len(intermediate_circles):
        if not check_intersection_case(segments[i], segments[i + 1]):
            i += 1
            continue

        # Resolve the intersection at circle i completely
        while check_intersection_case(segments[i], segments[i + 1]):

            # First intermediate circle
            if i == 0:
                circle = intermediate_circles.first
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                first_int_circ = circle
                circ2 = intermediate_circles[1]

                start_maneuvers = compute_traj_to_circle_bicycle(
                    corridor_list[0],
                    corridor_list[1],
                    start_pose,
                    bicycle,
                    first_int_circ,
                )

                new_segment =  compute_segment_between_two_circles_objects(
                    first_int_circ,
                    intermediate_circles[1],
                    bicycle,
                )

                segments[0] = start_maneuvers[-1]
                segments[1] = new_segment

            # Interior intermediate circle
            elif 0 < i < len(intermediate_circles) - 1:
                circle = intermediate_circles[i]
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    # figure = plot_corridors(corridor_list)
                    # plot_analytical_trajectory(segments, figure=figure)
                    # plt.plot(circle.center.x, circle.center.y, 'ro')    
                    # plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                    # plt.show(block = True)

                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                
                circ1 = intermediate_circles[i - 1]
                circ2 = intermediate_circles[i]
                circ3 = intermediate_circles[i + 1]

                new_segment1 =  compute_segment_between_two_circles_objects(
                    circ1,
                    circ2,
                    bicycle,
                )

                new_segment2 =  compute_segment_between_two_circles_objects(
                    circ2,
                    circ3,
                    bicycle,
                )

                segments[i] = new_segment1
                segments[i + 1] = new_segment2

                # figure = plot_corridors(corridor_list)
                # plot_analytical_trajectory(segments, figure=figure)
                # plt.plot(circle.center.x, circle.center.y, 'ro')    
                # plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                # plt.plot(circ3.center.x, circ3.center.y, 'ro')
                # plt.plot(circ3.xc + circ3.radius * np.cos(np.linspace(0, 2*pi, 100)), circ3.yc + circ3.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                # plt.show(block = True)

            # Last intermediate circle
            else:
                circle = intermediate_circles.last
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                last_int_circ = circle
                circ2 = intermediate_circles[-2]

                last_corridor = corridor_list[-1]
                inv_last_corridor, inv_penultimate_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(last_corridor, corridor_list[-2], last_int_circ, end_pose)

                inv_end_maneuvers = compute_traj_to_circle_bicycle(
                    inv_last_corridor,
                    inv_penultimate_corridor,
                    inv_end_pose,
                    bicycle,
                    inv_last_int_circ,
                )
                
                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

                new_segment =  compute_segment_between_two_circles_objects(
                    circ2,
                    last_int_circ,
                    bicycle,
                )

                segments[-2] = new_segment
                segments[-1] = end_maneuvers[0]

            # Optional safety stop if no more shift is possible
            # and the intersection still persists
            if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i + 1]):
                break

        # After changing circle i, restart from the previous circle
        i = max(i - 1, 0)

    return intermediate_circles, segments, start_maneuvers, end_maneuvers


def extract_intermediate_circle_from_maneuver_list(maneuver_list, traj_portion):
    """
    Extract the intermediate circle from a list of maneuvers.

    :param maneuver_list: list of maneuvers
    :type maneuver_list: list of primitives

    :return: intermediate circle
    :rtype: IntermediateCircle object
    """
    if traj_portion == "start": 
        if isinstance(maneuver_list[-2], CurvilinearArcUnicycle) or isinstance(maneuver_list[-2], BackwardArc):
            maneuver = maneuver_list[-2]
            intermediate_circle = IntermediateCircle(
                center=Point(x=maneuver.xc, y=maneuver.yc),
                radius=maneuver.radius,
                corner_point=Point(x=maneuver.xc, y=maneuver.yc),
                turn_direction=maneuver.turn_direction,
                index=None,
            )
            return intermediate_circle
            
    elif traj_portion == "end":
        if isinstance(maneuver_list[1], CurvilinearArcUnicycle) or isinstance(maneuver_list[1], BackwardArc):
            maneuver = maneuver_list[1]
            intermediate_circle = IntermediateCircle(
                center=Point(x=maneuver.xc, y=maneuver.yc),
                radius=maneuver.radius,
                corner_point=Point(x=maneuver.xc, y=maneuver.yc),
                turn_direction=maneuver.turn_direction,
                index=None,
            )
            return intermediate_circle
    print(f"No intermediate circle could be extracted from {traj_portion} trajectory.")
    return None


def all_intermediate_circles_are_skipped(intermediate_circles):
    """
    Check whether all intermediate circles are skipped.

    :param intermediate_circles: Intermediate circles.
    :type intermediate_circles: IntermediateCirclesSequence | list

    :return: True if all circles are skipped.
    :rtype: bool
    """
    return all(
        getattr(circle, "skip", False)
        for circle in intermediate_circles
    )


def prepare_initial_bicycle_trajectory(
    corridor_list,
    start_pose,
    bicycle,
    intermediate_circles,
):
    """
    Prepare the initial part of a trajectory through multiple corridors.
    """
    updated_corridor_list = list(corridor_list)

    updated_intermediate_circles = (
        IntermediateCirclesSequence(
            intermediate_circles
        )
    )

    planning_start_pose = list(start_pose)

    recovery_maneuvers = []
    start_trajectory = None
    start_requires_reversal = False

    # ---------------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------------
    if len(updated_corridor_list) < 2:
        raise ValueError(
            "At least two corridors are required to construct the "
            "initial multiple-corridor connection."
        )

    if len(updated_intermediate_circles) == 0:
        raise ValueError(
            "At least one intermediate circle is required."
        )

    first_corridor = updated_corridor_list[0]
    second_corridor = updated_corridor_list[1]
    first_circle = updated_intermediate_circles.first

    # ---------------------------------------------------------------
    # 1. Inspect the nominal initial problem
    # ---------------------------------------------------------------
    recovery_cause = determine_initial_recovery_cause(
        corridor_list=updated_corridor_list,
        start_pose=planning_start_pose,
        bicycle=bicycle,
        intermediate_circles=updated_intermediate_circles,
    )

    # ---------------------------------------------------------------
    # 2. Try the nominal connection
    # ---------------------------------------------------------------
    if recovery_cause is None:
        start_trajectory = compute_traj_to_circle_bicycle(
            first_corridor,
            second_corridor,
            planning_start_pose,
            bicycle,
            first_circle,
            figure=None,
        )

        if start_trajectory is not None:
            return (
                updated_corridor_list,
                planning_start_pose,
                bicycle,
                updated_intermediate_circles,
                start_trajectory,
                recovery_maneuvers,
                start_requires_reversal,
            )

        recovery_cause = "wall_collision"

    # ---------------------------------------------------------------
    # 3. Try the recovery hierarchy
    # ---------------------------------------------------------------
    recovery_result = attempt_initial_recovery(
        corridor_list=updated_corridor_list,
        start_pose=planning_start_pose,
        bicycle=bicycle,
        intermediate_circles=updated_intermediate_circles,
        recovery_cause=recovery_cause,
    )

    if recovery_result is None:
        raise ValueError(
            "No feasible recovery and subsequent circle connection "
            "could be constructed for the initial trajectory. "
            f"Cause: {recovery_cause}."
        )

    (
        updated_corridor_list,
        planning_start_pose,
        updated_intermediate_circles,
        start_trajectory,
        recovery_maneuvers,
        start_requires_reversal,
    ) = recovery_result

    # ---------------------------------------------------------------
    # 4. Sanity checks
    # ---------------------------------------------------------------
    if start_trajectory is None:
        raise RuntimeError(
            "The recovery function returned successfully without a "
            "valid start trajectory."
        )

    if len(updated_intermediate_circles) == 0:
        raise RuntimeError(
            "The recovery function removed all intermediate circles."
        )

    # ---------------------------------------------------------------
    # 5. Return the prepared initial problem
    # ---------------------------------------------------------------
    return (
        updated_corridor_list,
        planning_start_pose,
        bicycle,
        updated_intermediate_circles,
        start_trajectory,
        recovery_maneuvers,
        start_requires_reversal,
    )


def prepare_final_bicycle_trajectory(
    corridor_list,
    end_pose,
    bicycle,
    intermediate_circles,
):
    """
    Prepare the final trajectory by solving an inverted initial problem.

    The inverted geometry is used only temporarily. Structural changes
    are transferred back by removing the final corridor and circle from
    the original sequences when necessary.
    """
    if len(corridor_list) < 2:
        raise ValueError(
            "At least two corridors are required."
        )

    if len(intermediate_circles) == 0:
        raise ValueError(
            "At least one intermediate circle is required."
        )

    original_corridor_count = len(corridor_list)
    original_circle_count = len(intermediate_circles)

    inverted_corridor_list = (
        invert_and_reverse_corridor_sequence(
            corridor_list
        )
    )

    inverted_intermediate_circles = (
        invert_and_reverse_intermediate_circle_sequence(
            intermediate_circles=intermediate_circles,
            number_of_corridors=len(corridor_list),
        )
    )

    inverted_end_pose, = invert_inputs_all(
        end_pose
    )

    (
        updated_inverted_corridors,
        inverted_planning_end_pose,
        _,
        updated_inverted_circles,
        inverted_end_trajectory,
        inverted_end_recovery_maneuvers,
        end_requires_reversal,
    ) = prepare_initial_bicycle_trajectory(
        corridor_list=inverted_corridor_list,
        start_pose=inverted_end_pose,
        bicycle=bicycle,
        intermediate_circles=inverted_intermediate_circles,
    )

    removed_final_corridor = (
        len(updated_inverted_corridors)
        == original_corridor_count - 1
    )

    removed_final_circle = (
        len(updated_inverted_circles)
        == original_circle_count - 1
    )

    if removed_final_corridor != removed_final_circle:
        raise RuntimeError(
            "The inverted final recovery removed an inconsistent "
            "number of corridors and intermediate circles."
        )

    updated_corridor_list = list(
        corridor_list
    )

    updated_intermediate_circles = (
        IntermediateCirclesSequence(
            intermediate_circles
        )
    )

    if removed_final_corridor:
        updated_corridor_list.pop(-1)
        updated_intermediate_circles.remove_at(-1)

    planning_end_pose, = invert_inputs_all(
        inverted_planning_end_pose
    )

    end_trajectory = invert_maneuvers(
        inverted_end_trajectory,
        t0=0.0,
    )

    end_recovery_maneuvers = invert_maneuvers(
        inverted_end_recovery_maneuvers,
        t0=0.0,
    )

    return (
        updated_corridor_list,
        planning_end_pose,
        bicycle,
        updated_intermediate_circles,
        end_trajectory,
        end_recovery_maneuvers,
        end_requires_reversal,
    )


def invert_and_reverse_corridor_sequence(
    corridor_list,
):
    """
    Invert every corridor and reverse the sequence order.
    """
    inverted_corridors = []

    for corridor in reversed(corridor_list):
        inverted_corridor, = invert_inputs_all(
            corridor
        )

        inverted_corridors.append(
            inverted_corridor
        )

    return inverted_corridors


def invert_and_reverse_intermediate_circle_sequence(
    intermediate_circles,
    number_of_corridors,
):
    """
    Invert every intermediate circle and reverse the sequence order.

    Corridor indices are remapped consistently with reversal of the
    corridor sequence.

    For an original circle associated with corridors

        [old_start, old_end],

    the inverted circle is associated with

        [
            number_of_corridors - 1 - old_end,
            number_of_corridors - 1 - old_start,
        ].

    The same remapping is recursively applied to the source circles stored
    in `merged_from`.

    Parameters
    ----------
    intermediate_circles:
        Original ordered intermediate-circle sequence.

    number_of_corridors:
        Number of corridors in the corresponding corridor sequence.

    Returns
    -------
    IntermediateCirclesSequence
        Inverted and reversed intermediate-circle sequence.
    """
    if number_of_corridors < 2:
        raise ValueError(
            "number_of_corridors must be at least two."
        )

    def _remap_circle_indices(circle):
        """
        Remap the corridor indices of an ordinary or merged circle.
        """
        if not hasattr(
            circle,
            "corridor_index_start",
        ):
            raise AttributeError(
                "IntermediateCircle does not contain "
                "'corridor_index_start'."
            )

        if not hasattr(
            circle,
            "corridor_index_end",
        ):
            raise AttributeError(
                "IntermediateCircle does not contain "
                "'corridor_index_end'."
            )

        old_start = circle.corridor_index_start
        old_end = circle.corridor_index_end

        circle.corridor_index_start = (
            number_of_corridors - 1 - old_end
        )
        circle.corridor_index_end = (
            number_of_corridors - 1 - old_start
        )

        if getattr(circle, "is_merged", False):
            merged_from = getattr(
                circle,
                "merged_from",
                None,
            )

            if merged_from is None:
                raise ValueError(
                    "Merged intermediate circle does not contain "
                    "'merged_from' metadata."
                )

            for source_circle in merged_from:
                _remap_circle_indices(
                    source_circle
                )

        return circle

    inverted_circles = []

    for circle in reversed(
        list(intermediate_circles)
    ):
        inverted_circle, = invert_inputs_all(
            circle
        )

        inverted_circle = _remap_circle_indices(
            inverted_circle
        )

        inverted_circles.append(
            inverted_circle
        )

    return type(intermediate_circles)(
        inverted_circles
    )


def attempt_initial_recovery(
    corridor_list,
    start_pose,
    bicycle,
    intermediate_circles,
    distance_inside_second=1e-3,
    tol=1e-9,
    recovery_cause=None,
):
    """
    Attempt initial recovery strategies in priority order.

    The function stops at the first recovery sequence that admits a valid
    connection to an intermediate circle.

    The effective corridor parametrization stored in the relevant source
    circle is used for:

        - computing the alignment maneuver;
        - measuring whether the vehicle is sufficiently behind the circle;
        - constructing a deeper straight segment;
        - computing the pose-to-circle connection.

    For an ordinary intermediate circle, the circle itself provides the
    inversion metadata.

    For a merged circle, the constituent circle associated with the initial
    boundary is selected. Specifically, the selected constituent satisfies

        constituent.corridor_index_start
            == merged_circle.corridor_index_start.

    Recovery hierarchy
    ------------------
    Ordinary first circle:

        1. Align with preferred direction +1, enter the second corridor,
           remove the first corridor and first intermediate circle, and
           connect to the new first intermediate circle.

        2. Repeat with preferred direction -1.

        3. Align with preferred direction +1, move sufficiently far behind
           the first circle, and connect to it.

        4. Align with preferred direction -1, move sufficiently far behind
           the first circle, and construct a backward connection.

    Merged first circle:

        1. Skip the strategies that enter the second corridor.

        2. Align with preferred direction +1, move sufficiently far behind
           the merged circle, and connect to it.

        3. Repeat with preferred direction -1 if the previous strategy
           fails.

    Parameters
    ----------
    corridor_list:
        Ordered corridor sequence.

    start_pose:
        Initial pose [x, y, theta].

    bicycle:
        Bicycle model.

    intermediate_circles:
        Ordered IntermediateCirclesSequence.

    distance_inside_second:
        Distance travelled after entering the second shrunken corridor.

    tol:
        Numerical tolerance.

    recovery_cause:
        Cause that triggered recovery. Currently used only for diagnostic
        purposes.

    Returns
    -------
    tuple or None
        On success:

        (
            updated_corridor_list,
            recovered_start_pose,
            updated_intermediate_circles,
            start_trajectory,
            recovery_maneuvers,
            start_requires_reversal,
        )

        None is returned when no tested strategy succeeds.
    """

    # ---------------------------------------------------------------
    # Local helper: select the relevant source circle
    # ---------------------------------------------------------------
    def _get_boundary_source_circle(
        circle,
        boundary_side,
    ):
        """
        Return the ordinary constituent circle associated with a boundary.

        For an ordinary circle, return the circle itself.

        For a merged circle:

            - boundary_side == "start":
              select the source circle whose corridor_index_start matches
              the merged circle's corridor_index_start;

            - boundary_side == "end":
              select the source circle whose corridor_index_end matches
              the merged circle's corridor_index_end.
        """
        if boundary_side not in ("start", "end"):
            raise ValueError(
                "boundary_side must be either 'start' or 'end'."
            )

        if not getattr(circle, "is_merged", False):
            return circle

        merged_from = getattr(
            circle,
            "merged_from",
            None,
        )

        if merged_from is None or len(merged_from) == 0:
            raise ValueError(
                "Merged intermediate circle does not contain a valid "
                "'merged_from' sequence."
            )

        if boundary_side == "start":
            target_index = circle.corridor_index_start

            for source_circle in merged_from:
                if (
                    source_circle.corridor_index_start
                    == target_index
                ):
                    return source_circle

            # Defensive fallback.
            return min(
                merged_from,
                key=lambda source_circle: (
                    source_circle.corridor_index_start
                ),
            )

        target_index = circle.corridor_index_end

        for source_circle in merged_from:
            if (
                source_circle.corridor_index_end
                == target_index
            ):
                return source_circle

        # Defensive fallback.
        return max(
            merged_from,
            key=lambda source_circle: (
                source_circle.corridor_index_end
            ),
        )

    # ---------------------------------------------------------------
    # Local helper: reconstruct the effective corridor pair
    # ---------------------------------------------------------------
    def _get_effective_corridor_pair(
        corridors,
        circle,
        corridor_index_offset,
        boundary_side,
    ):
        """
        Reconstruct the effective corridor pair used to construct the
        relevant ordinary source circle.

        Parameters
        ----------
        corridors:
            Current corridor list. This may be the complete original list
            or a suffix obtained after removing one or more corridors.

        circle:
            Ordinary or merged intermediate circle.

        corridor_index_offset:
            Global index represented by corridors[0].

            Examples:

                0 for the complete original corridor list;
                1 after removing the original first corridor.

        boundary_side:
            Either "start" or "end".

        Returns
        -------
        tuple
            (
                effective_corridor1,
                effective_corridor2,
                source_circle,
            )
        """
        source_circle = _get_boundary_source_circle(
            circle=circle,
            boundary_side=boundary_side,
        )

        global_index_1 = (
            source_circle.corridor_index_start
        )
        global_index_2 = (
            source_circle.corridor_index_end
        )

        local_index_1 = (
            global_index_1
            - corridor_index_offset
        )
        local_index_2 = (
            global_index_2
            - corridor_index_offset
        )

        if (
            local_index_1 < 0
            or local_index_2 < 0
            or local_index_1 >= len(corridors)
            or local_index_2 >= len(corridors)
        ):
            raise ValueError(
                "The corridor indices stored in the intermediate circle "
                "are incompatible with the current corridor list. "
                f"Stored global indices: "
                f"({global_index_1}, {global_index_2}); "
                f"corridor-index offset: {corridor_index_offset}; "
                f"current corridor count: {len(corridors)}."
            )

        if local_index_2 != local_index_1 + 1:
            raise ValueError(
                "An ordinary intermediate circle must be associated "
                "with two consecutive corridors. "
                f"Received local indices "
                f"({local_index_1}, {local_index_2})."
            )

        corridor1 = corridors[local_index_1]
        corridor2 = corridors[local_index_2]

        corridor1_inversion = getattr(
            source_circle,
            "corridor1_inversion",
            0,
        )
        corridor2_inversion = getattr(
            source_circle,
            "corridor2_inversion",
            0,
        )

        if corridor1_inversion not in (-1, 0, 1):
            raise ValueError(
                "Invalid corridor1_inversion stored in intermediate "
                f"circle: {corridor1_inversion}."
            )

        if corridor2_inversion not in (-1, 0, 1):
            raise ValueError(
                "Invalid corridor2_inversion stored in intermediate "
                f"circle: {corridor2_inversion}."
            )

        if corridor1_inversion == 0:
            effective_corridor1 = corridor1
        else:
            effective_corridor1 = (
                corridor1.invert_dimensions(
                    corridor1_inversion
                )
            )

        if corridor2_inversion == 0:
            effective_corridor2 = corridor2
        else:
            effective_corridor2 = (
                corridor2.invert_dimensions(
                    corridor2_inversion
                )
            )

        return (
            effective_corridor1,
            effective_corridor2,
            source_circle,
        )

    # ---------------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------------
    if len(corridor_list) < 2:
        return None

    if len(intermediate_circles) == 0:
        return None

    first_circle = intermediate_circles.first

    first_circle_is_merged = bool(
        getattr(
            first_circle,
            "is_merged",
            False,
        )
    )

    # Reconstruct the effective corridor pair associated with the
    # initial boundary of the first target circle.
    (
        effective_first_corridor,
        effective_second_corridor,
        first_source_circle,
    ) = _get_effective_corridor_pair(
        corridors=corridor_list,
        circle=first_circle,
        corridor_index_offset=0,
        boundary_side="start",
    )

    # Retain these values for diagnostics and possible future policies.
    _ = recovery_cause
    _ = first_source_circle

    # ---------------------------------------------------------------
    # Cache both possible alignment attempts
    # ---------------------------------------------------------------
    alignment_results = {}

    for preferred_direction in (1, -1):
        alignment_results[preferred_direction] = (
            _compute_alignment_state(
                start_pose=start_pose,
                corridor=effective_first_corridor,
                bicycle=bicycle,
                preferred_direction=preferred_direction,
                tol=tol,
            )
        )

    # ===============================================================
    # Strategies 1-2:
    # align and enter the second corridor
    #
    # These strategies are disabled for a merged first circle.
    # ===============================================================
    if (
        not first_circle_is_merged
        and len(intermediate_circles) >= 2
    ):
        for preferred_direction in (1, -1):
            alignment_result = alignment_results[
                preferred_direction
            ]

            if alignment_result is None:
                continue

            (
                alignment_maneuvers,
                aligned_pose,
                aligned_time,
            ) = alignment_result

            transition_segment = (
                compute_segment_to_next_corridor(
                    start_pose=aligned_pose,
                    first_corridor=effective_first_corridor,
                    second_corridor=effective_second_corridor,
                    bicycle=bicycle,
                    distance_inside_second=(
                        distance_inside_second
                    ),
                    t0=aligned_time,
                    tol=tol,
                )
            )

            if transition_segment is None:
                continue

            recovered_pose = (
                transition_segment.end_pose
            )

            # Remove the original first corridor.
            candidate_corridors = list(
                corridor_list[1:]
            )

            # Remove the original first intermediate circle.
            candidate_circles = (
                IntermediateCirclesSequence(
                    intermediate_circles
                )
            )
            candidate_circles.remove_at(0)

            if (
                len(candidate_corridors) < 2
                or len(candidate_circles) == 0
            ):
                continue

            candidate_first_circle = (
                candidate_circles.first
            )

            # candidate_corridors[0] corresponds to original global
            # corridor index 1.
            (
                candidate_effective_first_corridor,
                candidate_effective_second_corridor,
                _,
            ) = _get_effective_corridor_pair(
                corridors=candidate_corridors,
                circle=candidate_first_circle,
                corridor_index_offset=1,
                boundary_side="start",
            )

            start_trajectory = (
                compute_traj_to_circle_bicycle(
                    candidate_effective_first_corridor,
                    candidate_effective_second_corridor,
                    recovered_pose,
                    bicycle,
                    candidate_first_circle,
                    figure=None,
                )
            )

            if start_trajectory is None:
                continue

            recovery_maneuvers = list(
                alignment_maneuvers
            )
            recovery_maneuvers.append(
                transition_segment
            )

            return (
                candidate_corridors,
                recovered_pose,
                candidate_circles,
                start_trajectory,
                recovery_maneuvers,
                False,
            )

    # ===============================================================
    # Strategies 3-4:
    # align and move sufficiently far behind the first circle
    #
    # The order remains unchanged for ordinary and merged circles.
    # ===============================================================
    for preferred_direction in (1, -1):
        alignment_result = alignment_results[
            preferred_direction
        ]

        if alignment_result is None:
            continue

        (
            alignment_maneuvers,
            aligned_pose,
            aligned_time,
        ) = alignment_result

        # Alignment changes the vehicle heading, so the initial turn
        # direction must be recomputed.
        tau0_aligned = compute_initial_turn_direction(
            first_circle.xc,
            first_circle.yc,
            first_circle.radius,
            aligned_pose[0],
            aligned_pose[1],
            aligned_pose[2],
            first_circle.turn_direction,
        )

        # Compute the distance needed to place the aligned pose
        # sufficiently far behind the actual target circle. For a merged
        # circle, this uses the merged-circle center and radius.
        required_deeper_distance = (
            compute_required_deeper_distance(
                start_pose=aligned_pose,
                reference_corridor=(
                    effective_first_corridor
                ),
                first_circle=first_circle,
                tau0=tau0_aligned,
                tol=tol,
            )
        )

        deeper_segment = None
        recovered_pose = aligned_pose

        if required_deeper_distance > tol:
            deeper_segment = (
                compute_segment_deeper_into_corridor(
                    start_pose=aligned_pose,
                    corridor=effective_first_corridor,
                    bicycle=bicycle,
                    distance=required_deeper_distance,
                    t0=aligned_time,
                    tol=tol,
                )
            )

            if deeper_segment is None:
                continue

            recovered_pose = (
                deeper_segment.end_pose
            )

        # preferred_direction == +1:
        #     construct a forward connection.
        #
        # preferred_direction == -1:
        #     invert the recovered pose so that the forward analytical
        #     construction represents a backward trajectory in the
        #     original problem.
        if preferred_direction == 1:
            connection_pose = recovered_pose
            start_requires_reversal = False

        else:
            connection_pose, = invert_inputs_all(
                recovered_pose
            )
            start_requires_reversal = True

        start_trajectory = (
            compute_traj_to_circle_bicycle(
                effective_first_corridor,
                effective_second_corridor,
                connection_pose,
                bicycle,
                first_circle,
                figure=None,
            )
        )

        if start_trajectory is None:
            continue

        recovery_maneuvers = list(
            alignment_maneuvers
        )

        if deeper_segment is not None:
            recovery_maneuvers.append(
                deeper_segment
            )

        returned_circles = (
            IntermediateCirclesSequence(
                intermediate_circles
            )
        )

        return (
            list(corridor_list),
            connection_pose,
            returned_circles,
            start_trajectory,
            recovery_maneuvers,
            start_requires_reversal,
        )

    return None


def compute_required_deeper_distance(
    start_pose,
    reference_corridor,
    first_circle,
    tau0,
    tol=1e-9,
    safety_margin=1e-3,
):
    """
    Compute the longitudinal distance required to place the vehicle
    sufficiently behind the first intermediate-circle center.

    The calculation is performed in the effective corridor frame used to
    construct the relevant source circle. Its longitudinal axis must point
    toward the intermediate-circle transition.

    Moving deeper into the corridor is therefore motion along the negative
    longitudinal axis.

    The required longitudinal separation is:

        R
            when tau0 == tau1;

        2R + safety_margin
            when tau0 != tau1.

    Parameters
    ----------
    start_pose:
        Current pose [x, y, theta].

    reference_corridor:
        Effective first-corridor representation associated with the
        relevant source circle. Its tilt defines the positive longitudinal
        direction toward the intermediate circle.

    first_circle:
        Target intermediate circle. This may also be a merged circle.

    tau0:
        Initial turn direction computed from the current pose.

    tol:
        Numerical tolerance.

    safety_margin:
        Positive margin used to avoid the degenerate opposite-turn
        condition at exactly 2R.

    Returns
    -------
    float
        Nonnegative distance to move along the negative longitudinal axis
        of `reference_corridor`.
    """
    if tau0 not in (-1, 1):
        raise ValueError(
            f"tau0 must be either -1 or +1, received {tau0}."
        )

    tau1 = first_circle.turn_direction

    if tau1 not in (-1, 1):
        raise ValueError(
            "first_circle.turn_direction must be either -1 or +1, "
            f"received {tau1}."
        )

    if safety_margin < 0.0:
        raise ValueError(
            "safety_margin must be nonnegative."
        )

    radius = float(first_circle.radius)

    if radius <= 0.0:
        raise ValueError(
            f"Circle radius must be positive, received {radius}."
        )

    longitudinal_axis = np.array(
        [
            cos(reference_corridor.tilt),
            sin(reference_corridor.tilt),
        ],
        dtype=float,
    )

    start_position = np.asarray(
        start_pose[:2],
        dtype=float,
    )

    circle_center = np.array(
        [
            first_circle.xc,
            first_circle.yc,
        ],
        dtype=float,
    )

    # Signed longitudinal position of the vehicle relative to the circle:
    #
    #   > 0: vehicle is beyond/in front of the circle center;
    #   = 0: vehicle is level with the circle center;
    #   < 0: vehicle is behind the circle center.
    longitudinal_coordinate = float(
        np.dot(
            start_position - circle_center,
            longitudinal_axis,
        )
    )

    if tau0 == tau1:
        required_separation = radius
    else:
        required_separation = (
            2.0 * radius
            + safety_margin
        )

    # Desired condition:
    #
    #   longitudinal_coordinate <= -required_separation
    #
    # A displacement L along the negative longitudinal axis gives:
    #
    #   new_coordinate = longitudinal_coordinate - L
    required_distance = (
        longitudinal_coordinate
        + required_separation
    )

    if required_distance <= tol:
        return 0.0

    return float(required_distance)


def determine_initial_recovery_cause(
    corridor_list,
    start_pose,
    bicycle,
    intermediate_circles,
    tol=1e-9,
):
    """
    Determine whether the nominal initial connection requires recovery.

    The boundary-depth check is performed using the effective
    parametrization of the first corridor associated with the first
    intermediate circle.

    For a merged first circle, the constituent circle corresponding to
    the initial boundary is used to recover the relevant corridor
    inversion metadata.

    Returns
    -------
    None
        The nominal geometric conditions are satisfied.

    str
        Reason for recovery. Possible values are:

        - ``"insufficient_depth"``
        - ``"opposite_turn_insufficient_separation"``
    """

    # ---------------------------------------------------------------
    # Local helper: select the source circle at the initial boundary
    # ---------------------------------------------------------------
    def _get_initial_source_circle(circle):
        """
        Return the ordinary circle associated with the initial boundary.

        For an ordinary circle, return the circle itself.

        For a merged circle, select the constituent whose
        corridor_index_start matches that of the merged circle.
        """
        if not getattr(circle, "is_merged", False):
            return circle

        merged_from = getattr(
            circle,
            "merged_from",
            None,
        )

        if merged_from is None or len(merged_from) == 0:
            raise ValueError(
                "Merged intermediate circle does not contain a valid "
                "'merged_from' sequence."
            )

        target_index = circle.corridor_index_start

        for source_circle in merged_from:
            if (
                source_circle.corridor_index_start
                == target_index
            ):
                return source_circle

        # Defensive fallback.
        return min(
            merged_from,
            key=lambda source_circle: (
                source_circle.corridor_index_start
            ),
        )

    # ---------------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------------
    if len(corridor_list) < 2:
        raise ValueError(
            "At least two corridors are required to determine the "
            "initial recovery cause."
        )

    if len(intermediate_circles) == 0:
        raise ValueError(
            "At least one intermediate circle is required to determine "
            "the initial recovery cause."
        )

    first_circle = intermediate_circles.first
    first_source_circle = _get_initial_source_circle(
        first_circle
    )

    # ---------------------------------------------------------------
    # Reconstruct the effective first corridor
    # ---------------------------------------------------------------
    first_corridor = corridor_list[
        first_source_circle.corridor_index_start
    ]

    first_corridor_inversion = getattr(
        first_source_circle,
        "corridor1_inversion",
        0,
    )

    if first_corridor_inversion not in (-1, 0, 1):
        raise ValueError(
            "Invalid corridor1_inversion stored in the first source "
            f"circle: {first_corridor_inversion}."
        )

    if first_corridor_inversion == 0:
        effective_first_corridor = first_corridor
    else:
        effective_first_corridor = (
            first_corridor.invert_dimensions(
                first_corridor_inversion
            )
        )

    # ---------------------------------------------------------------
    # Initial turn-direction and geometric separation
    # ---------------------------------------------------------------
    tau1 = first_circle.turn_direction

    tau0 = compute_initial_turn_direction(
        first_circle.xc,
        first_circle.yc,
        first_circle.radius,
        start_pose[0],
        start_pose[1],
        start_pose[2],
        tau1,
    )

    distance_to_first_circle = (
        compute_distance_two_points(
            [start_pose[0], start_pose[1]],
            [first_circle.xc, first_circle.yc],
        )
    )

    # ---------------------------------------------------------------
    # Boundary-depth check
    # ---------------------------------------------------------------
    (
        start_is_sufficiently_inside,
        _,
        start_missing_distance,
        _,
    ) = check_bicycle_boundary_pose_separation(
        start_pose=start_pose,
        end_pose=start_pose,
        effective_first_corridor=effective_first_corridor,
        effective_last_corridor=effective_first_corridor,
        first_circle=first_circle,
        last_circle=first_circle,
    )

    _ = bicycle
    _ = start_missing_distance

    # The initial position is not sufficiently far behind the first
    # intermediate-circle center in the effective corridor frame.
    if not start_is_sufficiently_inside:
        return "insufficient_depth"

    # For opposite turn directions, the Euclidean distance must be
    # strictly larger than 2R.
    if (
        tau0 != tau1
        and distance_to_first_circle
        <= 2.0 * first_circle.radius + tol
    ):
        return "opposite_turn_insufficient_separation"

    return None


# def compute_trajectory_bicycle_multiple_corridors_optimal(corridor_list,
#     start_pose,
#     end_pose,
#     bicycle,
#     intermediate_circles,
#     ):


#     first_int_circ = intermediate_circles.first
#     d0 = compute_distance_two_points([start_pose[0], start_pose[1]], [first_int_circ.xc, first_int_circ.yc])
#     tau1 = first_int_circ.turn_direction
#     last_int_circ = intermediate_circles.last
#     df = compute_distance_two_points([end_pose[0], end_pose[1]], [last_int_circ.xc, last_int_circ.yc])

#     # 1-Verify whether the pose is sufficiently inside the corridor boundaries
#     (
#         start_is_sufficiently_inside,
#         end_is_sufficiently_inside,
#         start_missing_distance,
#         end_missing_distance,
#     ) = check_bicycle_boundary_pose_separation(
#         start_pose=start_pose,
#         end_pose=end_pose,
#         first_corridor=corridor_list[0],
#         last_corridor=corridor_list[-1],
#         first_circle=first_int_circ,
#         last_circle=last_int_circ,
#     )

#     if not start_is_sufficiently_inside:
#         warnings.warn(
#             "The initial position is less than one turning radius "
#             "before the center of the first intermediate circle."
#         )

#     if not end_is_sufficiently_inside:
#         warnings.warn(
#             "The final position is less than one turning radius "
#             "after the center of the last intermediate circle."
#         )

#     ### Start of the trajectory
#     # 2- compute tau0
#     tau0 = compute_initial_turn_direction(
#                 first_int_circ.xc,
#                 first_int_circ.yc,
#                 first_int_circ.radius,
#                 start_pose[0],
#                 start_pose[1],
#                 start_pose[2],
#                 tau1,
#             )

#     # Recovery condition 1:
#     # the position is not sufficiently far inside the first corridor.
#     start_requires_depth_recovery = (
#         not start_is_sufficiently_inside
#     )

#     # Recovery condition 2:
#     # the opposite-turn free-space construction is not defined because
#     # the start position lies on or inside the auxiliary circle of radius 2R.
#     start_requires_separation_recovery = (
#         tau0 != tau1
#         and d0 <= 2.0 * first_int_circ.radius
#     )

#     start_requires_recovery = (
#         start_requires_depth_recovery
#         or start_requires_separation_recovery
#     )


#     # figure = plot_corridors(corridor_list+shrink_corridor_list(corridor_list, margin=bicycle.width/2))
#     start_requires_reversal = False
#     end_requires_reversal = False

#     corridor1 = corridor_list[0]
#     corridor2 = corridor_list[1]

#     start_maneuvers = compute_traj_to_circle_bicycle(
#         corridor1,
#         corridor2,
#         start_pose,
#         bicycle,
#         first_int_circ,
#         figure=None,
#     )

#     if start_maneuvers is None:
#         start_requires_reversal = True
#         rev_start_pose = invert_inputs_all(start_pose)
#         start_maneuvers = compute_traj_to_circle_bicycle(
#                 corridor1,
#                 corridor2,
#                 rev_start_pose,
#                 bicycle,
#                 first_int_circ,
#                 figure=None,
#             )
#         if start_maneuvers is None:
#             raise ValueError("Start maneuvers are infeasible. No trajectory can be computed.")

#     # 2 — Compute P^final
#     last_corridor = corridor_list[-1]
#     inv_last_corridor, inv_penultimate_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(last_corridor, corridor_list[-2], last_int_circ, end_pose)

#     # figure = plot_corridors(corridor_list+shrink_corridor_list(corridor_list, margin=bicycle.width/2))
#     # figure = None
#     inv_end_maneuvers = compute_traj_to_circle_bicycle(
#         inv_last_corridor,
#         inv_penultimate_corridor,
#         inv_end_pose,
#         bicycle,
#         inv_last_int_circ,
#         figure = None,
#     )

#     if inv_end_maneuvers is None:
#         end_requires_reversal = True
#         inv_end_maneuvers = compute_traj_to_circle_bicycle(
#                 inv_last_corridor,
#                 inv_penultimate_corridor,
#                 inv_end_pose,
#                 bicycle,
#                 inv_last_int_circ,
#                 figure = None
#             )
#         # angle_array = np.linspace(0, 2 * np.pi, 100)
#         # plot_corridors(corridor_list)
#         # plt.plot(inv_last_int_circ.xc + inv_last_int_circ.radius * np.cos(angle_array), inv_last_int_circ.yc + inv_last_int_circ.radius * np.sin(angle_array), 'r--')
#         # plt.arrow(end_pose[0], end_pose[1], 0.5 * cos(end_pose[2]), 0.5 * sin(end_pose[2]), head_width=0.1, head_length=0.1, fc='g', ec='g')
#         # plt.plot(end_pose[0] + bicycle.width/2 * np.cos(angle_array), end_pose[1] + bicycle.width/2 * np.sin(angle_array), 'go', markersize=5)
#         # plt.plot(inv_end_pose[0] + bicycle.width/2 * np.cos(angle_array), inv_end_pose[1] + bicycle.width/2 * np.sin(angle_array), 'bo', markersize=5)
#         # plt.arrow(inv_end_pose[0], inv_end_pose[1], 0.5 * cos(inv_end_pose[2]), 0.5 * sin(inv_end_pose[2]), head_width=0.1, head_length=0.1, fc='b', ec='b')
#         # plt.show(block = True)
#         if inv_end_maneuvers is None:
#             raise ValueError("End maneuvers are infeasible. No trajectory can be computed.")
#     end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)


#     return compute_trajectory_bicycle_multiple_corridors_feasible(
#         corridor_list,
#         start_pose,
#         end_pose,
#         bicycle,
#         intermediate_circles,
#         start_maneuvers, 
#         end_maneuvers,
#         start_requires_reversal, 
#         end_requires_reversal,
#     )

def compute_trajectory_bicycle_multiple_corridors_optimal(
    corridor_list,
    start_pose,
    end_pose,
    bicycle,
    intermediate_circles,
):
    # ---------------------------------------------------------------
    # 1. Prepare the initial part
    # ---------------------------------------------------------------
    (
        corridor_list,
        planning_start_pose,
        bicycle,
        intermediate_circles,
        start_trajectory,
        start_recovery_maneuvers,
        start_requires_reversal,
    ) = prepare_initial_bicycle_trajectory(
        corridor_list=corridor_list,
        start_pose=start_pose,
        bicycle=bicycle,
        intermediate_circles=intermediate_circles,
    )

    # ---------------------------------------------------------------
    # 2. Prepare the final part
    # ---------------------------------------------------------------
    (
        corridor_list,
        planning_end_pose,
        bicycle,
        intermediate_circles,
        end_trajectory,
        end_recovery_maneuvers,
        end_requires_reversal,
    ) = prepare_final_bicycle_trajectory(
        corridor_list=corridor_list,
        end_pose=end_pose,
        bicycle=bicycle,
        intermediate_circles=intermediate_circles,
    )

    # ---------------------------------------------------------------
    # 3. Compute the trajectory for the reduced planning problem
    # ---------------------------------------------------------------
    main_trajectory = (
        compute_trajectory_bicycle_multiple_corridors_feasible(
            corridor_list=corridor_list,
            start_pose=planning_start_pose,
            end_pose=planning_end_pose,
            bicycle=bicycle,
            intermediate_circles=intermediate_circles,
            start_maneuvers=start_trajectory,
            end_maneuvers=end_trajectory,
        )
    )

    # ---------------------------------------------------------------
    # 4. Insert the required cusp reversals
    # ---------------------------------------------------------------
    main_trajectory = insert_required_reversals(
        trajectory=main_trajectory,
        bicycle=bicycle,
        start_requires_reversal=(
            start_requires_reversal
        ),
        end_requires_reversal=(
            end_requires_reversal
        ),
        offset=0.0,
    )

    # ---------------------------------------------------------------
    # 5. Add the physical boundary-recovery sequences
    # ---------------------------------------------------------------
    trajectory = (
        start_recovery_maneuvers
        + main_trajectory
        + end_recovery_maneuvers
    )

    # ---------------------------------------------------------------
    # 6. Restore time continuity
    # ---------------------------------------------------------------
    for index in range(
        len(trajectory) - 1
    ):
        current_final_time = (
            trajectory[index].time_grid[-1]
        )

        next_initial_time = (
            trajectory[index + 1].time_grid[0]
        )

        time_offset = (
            current_final_time
            - next_initial_time
        )

        if abs(time_offset) > 1e-9:
            trajectory[
                index + 1
            ].add_time_offset(
                time_offset
            )

    # ---------------------------------------------------------------
    # 7. Restore angular continuity
    # ---------------------------------------------------------------
    correct_angles(
        trajectory
    )

    return trajectory
     

def compute_trajectory_bicycle_multiple_corridors_feasible(
    corridor_list,
    start_pose,
    end_pose,
    bicycle,
    intermediate_circles,
    start_maneuvers, 
    end_maneuvers,
):
    """
    Compute the sequence of primitives that build the time-optimal trajectory
    for a bicycle vehicle within multiple corridors.
    Backward maneuver both for collision avoidance and time-optimality.

    :param corridor_list: list of corridors
    :type corridor_list: list of CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the last corridor
    :type end_pose: list of floats
    :param bicycle: bicycle vehicle
    :type Bicycle: Bicycle
    :param intermediate_circles_choices: list of choices for intermediate circles
    :type intermediate_circles_choices: list of IntermediateCircleChoice objects

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    """

    # Extract the sequence of intermediate circles from the choices
    # intermediate_circles = selected_sequence_from_preferences(
    #     intermediate_circles_choices
    # )
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure)
    # plt.show(block = True)
    # 1 — Compute P^init
    # first_int_circ = intermediate_circles.first
    # corridor1 = corridor_list[0]

    # start_maneuvers = compute_traj_to_circle_bicycle(
    #     corridor1,
    #     start_pose,
    #     bicycle,
    #     first_int_circ,
    # )

    # # 2 — Compute P^final
    # last_int_circ = intermediate_circles.last
    # last_corridor = corridor_list[-1]
    # inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(last_corridor, last_int_circ, end_pose)

    # inv_end_maneuvers = compute_traj_to_circle_bicycle(
    #     inv_last_corridor,
    #     inv_end_pose,
    #     bicycle,
    #     inv_last_int_circ,
    # )

    # end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

    # Update Intermediate Circles depending on intersections
    intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(start_maneuvers, "start")
    intermediate_circleN = extract_intermediate_circle_from_maneuver_list(end_maneuvers, "end")

    extended_circle_sequence = [intermediate_circle0] + list(intermediate_circles) + [intermediate_circleN]

    flags, blocks = detect_tangent_intersections_blocks(
        circle_sequence=extended_circle_sequence,
        bicycle=bicycle,
    )

    intermediate_circles_before_update = [circle.center for circle in intermediate_circles]

    max_iterations = 100
    iteration = 0
    all_skipped = False

    while any(flags) and iteration < max_iterations:
        # figure = plot_corridors(corridor_list)
        # angle_array = np.linspace(0, 2 * np.pi, 100)
        # segments, _ = compute_P_mid(intermediate_circles, bicycle)
        # radius = bicycle.max_radius
        # for center in intermediate_circles_before_update:
        #     plt.plot(center.x, center.y, 'ro', markersize=5)
        #     plt.plot(center.x + radius * np.cos(angle_array), center.y + radius * np.sin(angle_array), 'g-', markersize=5)
        # for circle in extended_circle_sequence:
        #     plt.plot(
        #         circle.xc + circle.radius * np.cos(angle_array),
        #         circle.yc + circle.radius * np.sin(angle_array),
        #         "r--",
        #     ) 
        # plot_analytical_trajectory(start_maneuvers+segments+end_maneuvers, figure)
        # plt.show(block = True)
        

        iteration += 1

        (
            new_extended_centers,
            corrected_centers,
            failed_shifts,
            boundary_updates,
        ) = solve_tangent_intersections_blocks_centers(
            blocks,
            extended_circle_sequence,
            start_pose=start_pose,
            end_pose=end_pose,
            corridor_list=corridor_list,
            bicycle=bicycle,
            tol=1e-9,
        )
        print(f"corrected_centers at iteration {iteration}: {corrected_centers}")
        # ------------------------------------------------------------
        # 1. Apply all successful center shifts, even if some other
        #    circles failed in other blocks.
        # ------------------------------------------------------------
        intermediate_circles_before_update = [circle.center for circle in intermediate_circles]
        if corrected_centers:
            update_intermediate_circle_centers_from_extended_sequence(
                intermediate_circles,
                new_extended_centers,
                corrected_centers=corrected_centers,
                mark_corrected_as_skip=True,
            )

        for circle in intermediate_circles:
            print(f"Center circle {circle.index}: {circle.center.x}, {circle.center.y}, s: {circle.s}")

        # ------------------------------------------------------------
        # 2. If there are failed shifts, structurally repair them.
        #    Important: this happens AFTER applying successful shifts.
        # ------------------------------------------------------------
        if failed_shifts:
            old_number_of_intermediate_circles = len(intermediate_circles)

            touches_start = (
                1 in failed_shifts
                or 1 in corrected_centers
            )

            touches_end = (
                old_number_of_intermediate_circles in failed_shifts
                or old_number_of_intermediate_circles in corrected_centers
            )

            updated_any = update_turn_direction_circles(
                failed_shifts=failed_shifts,
                intermediate_circles=intermediate_circles,
                corridor_list=corridor_list,
                vehicle=bicycle,
                tol=1e-9,
            )

            if not updated_any:
                print("No failed circle could be updated.")
                break

            # Recompute the start maneuver only if the first actual
            # intermediate circle was shifted or structurally repaired.
            if touches_start:
                skip = True
                ind = 0
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind += 1
                    else:
                        first_int_circ = intermediate_circles[ind]
                        skip = False
                corridor1 = corridor_list[0]

                fixed_circle = extended_circle_sequence[0]
                start_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, first_int_circ, fixed_circle)
                first_int_circ = intermediate_circles.first


                intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
                    start_maneuvers,
                    "start",
                )

            # Recompute the end maneuver only if the last actual
            # intermediate circle was shifted or structurally repaired.
            if touches_end:
                skip = True
                ind = -1
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind -= 1
                    else:
                        last_int_circ = intermediate_circles[ind]
                        skip = False
                
                

                inv_last_int_circ, inv_end_pose, inv_fixed_circle, = invert_inputs_all(
                    last_int_circ,
                    end_pose,
                    extended_circle_sequence[-1]
                )
                inv_end_maneuvers = invert_maneuvers(end_maneuvers, t0=0)
                inv_end_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(inv_end_maneuvers, bicycle, inv_last_int_circ, inv_fixed_circle)
                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0=0)

                intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
                    end_maneuvers,
                    "end",
                )

            # Always rebuild the extended sequence and rerun tangent detection
            # globally, because internal shifts/structural repairs can affect
            # neighboring tangent triples.
            extended_circle_sequence = (
                [intermediate_circle0]
                + list(intermediate_circles)
                + [intermediate_circleN]
            )

            flags, blocks = detect_tangent_intersections_blocks(
                circle_sequence=extended_circle_sequence,
                bicycle=bicycle,
            )

            continue

        # ------------------------------------------------------------
        # 3. If there are no failed shifts but also no successful shifts,
        #    then the solver did not make progress.
        # ------------------------------------------------------------
        if not corrected_centers:
            print("Tangent intersections remain, but no centers were corrected.")
            break

        # ------------------------------------------------------------
        # 4. If successful shifts touched the first or last actual
        #    intermediate circle, recompute boundary maneuvers.
        # ------------------------------------------------------------
        # if "start_maneuvers" in boundary_updates:
        #     start_maneuvers = boundary_updates["start_maneuvers"]
        #     intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
        #         start_maneuvers
        #     )
        #     intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
        #         start_maneuvers
        #     )

        # if "end_maneuvers" in boundary_updates:
        #     end_maneuvers = boundary_updates["end_maneuvers"]
        #     intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
        #         end_maneuvers
        #     )

        #     intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
        #         end_maneuvers
        #     )

        if all_intermediate_circles_are_skipped(intermediate_circles):
            start_fixed_circle = extended_circle_sequence[0]
            end_fixed_circle = extended_circle_sequence[-1]
            start_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, end_fixed_circle, start_fixed_circle)

            inv_start_fixed_circle, inv_end_fixed_circle, = invert_inputs_all(
                    start_fixed_circle,
                    end_fixed_circle
                )
            
            inv_end_maneuvers = invert_maneuvers(end_maneuvers, t0=0)
            inv_end_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(inv_end_maneuvers, bicycle, inv_start_fixed_circle, inv_end_fixed_circle)
            end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0=0)
            end_maneuvers = end_maneuvers[1:]
           
            all_skipped = True
            
        else: 
            if 1 in corrected_centers:
                skip = True
                ind = 0
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind += 1
                    else:
                        first_int_circ = intermediate_circles[ind]
                        skip = False
                # first_int_circ = intermediate_circles.first
                corridor1 = corridor_list[0]

                # start_maneuvers = compute_traj_to_circle_bicycle(
                #     corridor1,
                #     start_pose,
                #     bicycle,
                #     first_int_circ,
                # )
                fixed_circle = extended_circle_sequence[0]
                start_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, first_int_circ, fixed_circle)

                intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
                    start_maneuvers,
                    "start",
                )

            if len(intermediate_circles) in corrected_centers:
                skip = True
                ind = -1
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind -= 1
                    else:
                        last_int_circ = intermediate_circles[ind]
                        skip = False
                # last_int_circ = intermediate_circles.last
                last_corridor = corridor_list[-1]

                inv_last_int_circ, inv_end_pose, inv_fixed_circle, = invert_inputs_all(
                    last_int_circ,
                    end_pose,
                    extended_circle_sequence[-1]
                )

                # inv_end_maneuvers = compute_traj_to_circle_bicycle(
                #     inv_last_corridor,
                #     inv_end_pose,
                #     bicycle,
                #     inv_last_int_circ,
                # )

                inv_end_maneuvers = invert_maneuvers(end_maneuvers, t0=0)
                inv_end_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(inv_end_maneuvers, bicycle, inv_last_int_circ, inv_fixed_circle)

                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0=0)

                intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
                    end_maneuvers,
                    "end",
                )

        extended_circle_sequence = (
            [intermediate_circle0]
            + list(intermediate_circles)
            + [intermediate_circleN]
        )

        flags, blocks = detect_tangent_intersections_blocks(
            circle_sequence=extended_circle_sequence,
            bicycle=bicycle,
        )
        print(f"Iteration {iteration}: flags = {flags}, blocks = {blocks}")

    if any(flags):
        print("Warning: tangent intersections remain after correction loop.")


    # Compute the segments and prune the intermediate circles if necessary
    if all_skipped:
        trajectory = start_maneuvers + end_maneuvers
            # Adjust the time grid and angles
        for i in range(len(trajectory)-1):
            if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
                trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
        correct_angles(trajectory)
        return trajectory

    segments, active_intermediate_circles = compute_P_mid(
        intermediate_circles,
        bicycle,
        return_active_circles=True,
    )

    segments.insert(0, start_maneuvers[-1])
    segments.append(end_maneuvers[0])

    # Compute the arcs.
    arcs = []

    for i in range(len(segments) - 1):
        arcs.append(
            compute_arc_from_two_tangents_objects(
                segments[i],
                segments[i + 1],
                active_intermediate_circles[i],
                bicycle,
            )
        )

    ## Attach everything together
    middle_sequence = [0] * (len(segments)-2 + len(arcs))
    middle_segments = segments[1:-1]
    ind_seg, ind_arc = 0, 0
    for index in range(0, len(middle_sequence), 2):
        middle_sequence[index] = arcs[ind_arc]
        ind_arc += 1
    
    for index in range(1, len(middle_sequence), 2):
        middle_sequence[index] = middle_segments[ind_seg]
        ind_seg += 1
    trajectory = start_maneuvers + middle_sequence + end_maneuvers

    # Adjust the time grid and angles
    for i in range(len(trajectory)-1):
        if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
    correct_angles(trajectory)
    
    return trajectory


def compute_trajectory_bicycle_two_corridors_optimal(corridor1, corridor2, start_pose, end_pose, bicycle, intermediate_circles):
    '''
    Compute the sequence of primitives that build the time-optimal trajectory for a bicycle vehicle within two corridors.
    Backward maneuver both for collision avoidance and time-optimality.

    :param corridor1: first corridor
    :type corridor1: CorridorWorld
    :param corridor2: second corridor
    :type corridor2: CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the second corridor
    :type end_pose: list of floats
    :param bicycle: bicycle vehicle
    :type Bicycle: Bicycle

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    '''
    # Extract the sequence of intermediate circles from the choices
    # intermediate_circles = selected_sequence_from_preferences(
    #     intermediate_circles_choices
    # )
    # 1 — Compute P^init
    intermediate_circle = intermediate_circles.first

    start_maneuvers = compute_traj_to_circle_bicycle(
        corridor1,
        start_pose,
        bicycle,
        intermediate_circle,
    )

    # 2 — Compute P^final
    inv_last_corridor, inv_last_int_circ, inv_end_pose, = invert_inputs_all(corridor2, intermediate_circle, end_pose)

    inv_end_maneuvers = compute_traj_to_circle_bicycle(
        inv_last_corridor,
        inv_end_pose,
        bicycle,
        inv_last_int_circ,
    )

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)


    intermediate_circle, start_maneuvers, end_maneuvers = shift_circles_two_corridors_bicycle(
    intermediate_circle,
    start_maneuvers,
    end_maneuvers,
    start_pose,
    end_pose,
    bicycle,
    corridor1,
    corridor2,
    )
    
    intermediate_arc = compute_arc_from_two_tangents_objects(start_maneuvers[-1], end_maneuvers[0], intermediate_circle, bicycle)
    trajectory = start_maneuvers + [intermediate_arc] + end_maneuvers

    for i in range(len(trajectory)-1):
        if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
    correct_angles(trajectory)

    return trajectory


def shift_circles_two_corridors_bicycle(
    intermediate_circle,
    start_maneuvers,
    end_maneuvers,
    start_pose, 
    end_pose,
    bicycle,
    corridor1,
    corridor2
):
    """
    Shift intermediate circle to resolve possible intersection.
    S: segment
    C: arc
    T: turn on-the-spot
    """
    processed_finished = False
    step = 0.1 

    while not processed_finished:
        S3 = start_maneuvers[-1]
        S5 = end_maneuvers[0]
        changed = False

        if check_intersection_case(S3, S5): 
            if intermediate_circle.s == 0:
                s_new = 0.5 * intermediate_circle.s_max
            else:
                s_new = min(
                    intermediate_circle.s + step * intermediate_circle.s_max,
                    intermediate_circle.s_max
                    )

            intermediate_circle.update_s(s=s_new)
            start_maneuvers = compute_traj_to_circle_bicycle(
                corridor1,
                start_pose,
                bicycle,
                intermediate_circle,
            )

            # 2 — Compute P^final
            inv_last_corridor, inv_last_int_circ, inv_end_pose, = invert_inputs_all(corridor2, intermediate_circle, end_pose)

            inv_end_maneuvers = compute_traj_to_circle_bicycle(
                inv_last_corridor,
                inv_end_pose,
                bicycle,
                inv_last_int_circ,
            )

            end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)
            changed = True

        processed_finished = not changed
            
    return intermediate_circle, start_maneuvers, end_maneuvers


from copy import deepcopy


def insert_required_reversals(
    trajectory,
    bicycle,
    start_requires_reversal=False,
    end_requires_reversal=False,
    offset=0.0,
    tol=1e-9,
):
    """
    Insert the cusp maneuvers required to connect the reduced analytical
    trajectory to the physical boundary-recovery trajectories.

    The input trajectory is assumed to have been constructed using the
    forward analytical representation.

    Start reversal
    --------------
    When ``start_requires_reversal`` is True:

        - search for the first sufficiently long linear segment;
        - replace it with a backward-to-forward cusp maneuver;
        - convert every maneuver preceding that segment to its physical
          backward representation.

    End reversal
    ------------
    When ``end_requires_reversal`` is True:

        - search for the last sufficiently long linear segment;
        - replace it with a forward-to-backward cusp maneuver;
        - convert every maneuver following that segment to its physical
          backward representation.

    Collision checking for the two cusp arcs is not yet performed.

    Parameters
    ----------
    trajectory:
        Complete reduced trajectory, excluding the boundary recovery
        maneuvers.

    bicycle:
        Bicycle vehicle model.

    start_requires_reversal:
        Whether a backward-to-forward cusp is required near the beginning.

    end_requires_reversal:
        Whether a forward-to-backward cusp is required near the end.

    offset:
        Distance from the beginning of a selected segment at which the
        reversal maneuver starts.

    tol:
        Numerical tolerance.

    Returns
    -------
    list
        Trajectory with the required reversal maneuvers inserted.

    Raises
    ------
    ValueError
        If a required reversal cannot be placed on any linear segment.
    """
    updated_trajectory = deepcopy(
        list(trajectory)
    )

    # ---------------------------------------------------------------
    # Local helper: try constructing a reversal on one segment
    # ---------------------------------------------------------------
    def _try_build_reversal(
        segment,
        first_arc_backward,
    ):
        """
        Try both lateral sides for the reversal maneuver.

        Returns
        -------
        list or None
            Replacement sequence if one side is geometrically feasible.
        """
        for side in (1, -1):
            reversal_geometry = (
                compute_reversal_maneuver_on_segment(
                    segment=segment,
                    bicycle=bicycle,
                    side=side,
                    offset=offset,
                )
            )

            if reversal_geometry is None:
                continue

            replacement_maneuvers = (
                build_segment_with_reversal(
                    segment=segment,
                    reversal_geometry=reversal_geometry,
                    bicycle=bicycle,
                    first_arc_backward=(
                        first_arc_backward
                    ),
                    tol=tol,
                )
            )

            if replacement_maneuvers is None:
                continue

            # TODO:
            # Check that both cusp arcs are collision-free.
            #
            # TODO:
            # Check that the offset parallel geometry lies in a valid
            # corridor with the required footprint clearance.

            return replacement_maneuvers

        return None

    # ===============================================================
    # 1. Start reversal:
    #    backward prefix -> cusp -> forward remainder
    # ===============================================================
    if start_requires_reversal:
        start_reversal_found = False

        for segment_index, maneuver in enumerate(
            updated_trajectory
        ):
            if not isinstance(
                maneuver,
                LinearSegmentUnicycle,
            ):
                continue

            replacement_maneuvers = (
                _try_build_reversal(
                    segment=maneuver,
                    first_arc_backward=True,
                )
            )

            if replacement_maneuvers is None:
                continue

            # All primitives preceding the selected segment were created
            # in the inverted forward representation. Convert them into
            # physical backward primitives without changing their order
            # or geometric paths.
            for prefix_maneuver in updated_trajectory[
                :segment_index
            ]:
                prefix_maneuver.reverse()

            # Replace the selected segment with:
            #
            #     backward segment before
            #     backward quarter-circle
            #     forward quarter-circle
            #     forward segment after
            updated_trajectory[
                segment_index : segment_index + 1
            ] = replacement_maneuvers

            start_reversal_found = True
            break

        if not start_reversal_found:
            raise ValueError(
                "The initial connection requires a reversal, but no "
                "linear segment is long enough to accommodate the "
                "backward-to-forward cusp maneuver."
            )

    # ===============================================================
    # 2. End reversal:
    #    forward prefix -> cusp -> backward suffix
    # ===============================================================
    if end_requires_reversal:
        end_reversal_found = False

        for segment_index in range(
            len(updated_trajectory) - 1,
            -1,
            -1,
        ):
            maneuver = updated_trajectory[
                segment_index
            ]

            if not isinstance(
                maneuver,
                LinearSegmentUnicycle,
            ):
                continue

            replacement_maneuvers = (
                _try_build_reversal(
                    segment=maneuver,
                    first_arc_backward=False,
                )
            )

            if replacement_maneuvers is None:
                continue

            # All primitives following the selected segment must be
            # converted from the inverted forward representation into
            # physical backward primitives.
            for suffix_maneuver in updated_trajectory[
                segment_index + 1 :
            ]:
                suffix_maneuver.reverse()

            # Replace the selected segment with:
            #
            #     forward segment before
            #     forward quarter-circle
            #     backward quarter-circle
            #     backward segment after
            updated_trajectory[
                segment_index : segment_index + 1
            ] = replacement_maneuvers

            end_reversal_found = True
            break

        if not end_reversal_found:
            raise ValueError(
                "The final connection requires a reversal, but no "
                "linear segment is long enough to accommodate the "
                "forward-to-backward cusp maneuver."
            )

    return updated_trajectory