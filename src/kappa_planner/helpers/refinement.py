from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace

import numpy as np

from shapely.geometry import LineString, Point

from kappa_planner.geometry import Point as KappaPoint, IntermediateCircle
from kappa_planner.helpers.collision_avoidance import (
    check_arc_collision,
    corridor_to_axis_aligned_polygon,
    compute_safe_corridor_union,
    segment_inside_safe_corridor_union,
)
from kappa_planner.helpers.invert_inputs import invert_inputs_all
from kappa_planner.helpers.pose_to_circle_bicycle import (
    compute_traj_to_circle_bicycle,
)
from kappa_planner.helpers.primitives import (
    compute_arc_from_two_tangents_objects,
    correct_angles,
    invert_maneuvers,
)
from kappa_planner.trajectory import LinearSegmentBicycle





# ===================================================================

# Data structures

# ===================================================================



@dataclass

class RefinementCircle:

    center: np.ndarray

    radius: float

    turn_direction: int



    corner_point: np.ndarray

    local_x_axis: np.ndarray

    local_y_axis: np.ndarray



    local_coordinates: tuple

    effective_dimensions: tuple | None



    placement_rule: str





@dataclass

class CircleTangent:

    start_circle_index: int

    end_circle_index: int



    start_point: np.ndarray

    end_point: np.ndarray



    start_heading: float

    end_heading: float





@dataclass

class StraightPassageGroup:

    """

    Group associated with one maximal run of aligned transitions.



    The group contains the zero run itself and, when available, the

    genuine-turn transition immediately before and immediately after it.



    The two adjacent genuine-turn circles are kept at their validated

    baseline positions.

    """



    start_index: int

    end_index: int



    zero_start_index: int

    zero_end_index: int



    left_circle_index: int | None

    right_circle_index: int | None


@dataclass
class BicycleRefinementResult:
    """Complete result of the bicycle-baseline refinement.

    Iteration deliberately yields the original four return values so that
    existing test code of the form

        circle_groups, straight_passage_groups, active_circle_indices, tangents = result

    remains valid.
    """

    circle_groups: list
    straight_passage_groups: list
    active_circle_indices: list
    tangents: list

    initial_maneuvers: list | None = None
    final_maneuvers: list | None = None

    trajectory: list | None = None
    traversal_time: float | None = None

    def __iter__(self):
        yield self.circle_groups
        yield self.straight_passage_groups
        yield self.active_circle_indices
        yield self.tangents

    def __len__(self):
        return 4


@dataclass(frozen=True)
class BicycleRefinementFailure:
    """Stage-specific reason why no complete refinement was produced."""

    reason: str
    transition_index: int | None = None





# ===================================================================

# Main refinement function

# ===================================================================



def refine_bicycle_baseline(
    corridor_list,
    bicycle,
    baseline,
    initial_pose=None,
    final_pose=None,
    return_failure=False,
):
    """
    Refine a validated bicycle baseline and, when boundary poses are available,
    construct the complete boundary-connected trajectory.

    The refinement applies the following stages:

        1. Find maximal straight-passage groups.

        2. Keep genuine-turn circles adjacent to those groups at their
           validated baseline positions.

        3. Independently position the remaining genuine-turn circles, with the
           validated baseline circle as fallback.

        4. Build and repair the internal circle-to-circle tangent chain.

        5. Simplify internal tangent self-intersections through safe circle
           skipping.

        6. Connect the prescribed initial and final poses to the first and last
           active circles.

           If a boundary connection to a refined circle fails, the corresponding
           validated baseline circle is tried. If that succeeds, the refined
           circle is replaced by the baseline circle and the internal tangent
           chain is rebuilt.

        7. Check the boundary tangent segments against the remaining tangent
           chain. If the initial boundary tangent intersects the chain, try to
           remove the first active circle and rebuild the initial connection to
           the next active circle. The final side is handled symmetrically.

           Every speculative boundary skip is transactional: it is committed
           only after the new boundary connection, possible baseline-circle
           fallback, and rebuilt internal tangent chain are all feasible.

        8. Construct the circular arcs between consecutive supporting tangent
           segments and assemble one complete primitive trajectory.

    Circle removal stops when only one active circle remains. Removing the last
    active circle would require a separate general pose-to-pose construction and
    is outside the present refinement.

    Notes
    -----
    A boundary connection to a circle farther inside the sequence requires the
    pose-to-circle helper to accept the optional keyword argument
    ``admissible_corridors``. The supplied corridor subsequence is used only for
    validating the straight tangent portion; the corrective circular maneuver
    remains constrained to the boundary corridor.

    :return:
        BicycleRefinementResult, or None if the refined chain cannot be made
        feasible. The validated baseline remains the caller's fallback.
    """
    def failed(reason, transition_index=None):
        failure = BicycleRefinementFailure(reason, transition_index)
        return (None, failure) if return_failure else None

    def succeeded(result):
        return (result, None) if return_failure else result

    n = len(corridor_list)

    if n < 2:
        return failed("invalid_corridor_sequence")

    R = bicycle.max_radius
    r = bicycle.width / 2

    # ---------------------------------------------------------------
    # 0. Resolve boundary poses.
    # ---------------------------------------------------------------
    initial_pose, final_pose = resolve_refinement_boundary_poses(
        baseline=baseline,
        initial_pose=initial_pose,
        final_pose=final_pose,
    )

    # ---------------------------------------------------------------
    # 1. Find straight-passage groups
    # ---------------------------------------------------------------
    straight_passage_groups = compute_straight_passage_groups(
        turn_directions=baseline.turn_directions,
    )

    # ---------------------------------------------------------------
    # 2. Genuine-turn circles adjacent to a straight passage are kept
    #    at their validated baseline positions.
    # ---------------------------------------------------------------
    baseline_circle_indices = set()

    for group in straight_passage_groups:
        if group.left_circle_index is not None:
            baseline_circle_indices.add(
                group.left_circle_index
            )

        if group.right_circle_index is not None:
            baseline_circle_indices.add(
                group.right_circle_index
            )

    # ---------------------------------------------------------------
    # 3. Position all genuine-turn circles
    # ---------------------------------------------------------------
    circle_groups = [
        {}
        for _ in range(n - 1)
    ]

    for j in range(n - 1):
        tau = baseline.turn_directions[j]

        # No refinement circle at an aligned transition.
        if tau == 0:
            continue

        if j in baseline_circle_indices:
            circle = compute_baseline_turn_circle(
                baseline=baseline,
                transition_index=j,
                R=R,
                placement_rule="baseline_straight_passage",
            )
        else:
            circle = compute_independent_turn_circle(
                baseline=baseline,
                transition_index=j,
                R=R,
                r=r,
            )

        if circle is None:
            return failed("circle_placement_failed", j)

        circle_groups[j][tau] = circle

    active_circle_indices = get_active_circle_indices(
        circle_groups
    )

    # At least one circle is required by the current complete-trajectory
    # construction.
    if len(active_circle_indices) == 0:
        return failed("no_active_refinement_circles")

    # The safe-region cache depends only on corridor indices, not on circle
    # positions, and can therefore be reused after circle repairs.
    safe_union_cache = {}

    # ---------------------------------------------------------------
    # 4-5. Build and simplify the internal tangent chain.
    # ---------------------------------------------------------------
    internal_state = rebuild_and_simplify_internal_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        safe_union_cache=safe_union_cache,
    )

    if internal_state is None:
        return failed("no_tangent_connection_found")

    (
        active_circle_indices,
        tangents,
    ) = internal_state

    # ---------------------------------------------------------------
    # 6. Attach the two boundary poses, allowing a boundary target
    #    circle to fall back to its validated baseline position.
    # ---------------------------------------------------------------
    if initial_pose is None or final_pose is None:
        # Preserve the useful internal refinement result when the caller did
        # not provide/construct boundary poses, but do not claim a complete
        # trajectory.
        return succeeded(BicycleRefinementResult(
            circle_groups=circle_groups,
            straight_passage_groups=straight_passage_groups,
            active_circle_indices=active_circle_indices,
            tangents=tangents,
        ))

    boundary_state = build_consistent_boundary_state(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        bicycle=bicycle,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        initial_pose=initial_pose,
        final_pose=final_pose,
        safe_union_cache=safe_union_cache,
    )

    if boundary_state is None:
        return failed("boundary_connection_failed_after_circle_fallback")

    (
        active_circle_indices,
        tangents,
        initial_maneuvers,
        final_maneuvers,
    ) = boundary_state

    # ---------------------------------------------------------------
    # 7. Include the boundary tangent segments in the intersection
    #    logic. The first/last active circle may now be removed.
    # ---------------------------------------------------------------
    boundary_simplification = simplify_boundary_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        bicycle=bicycle,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        initial_pose=initial_pose,
        final_pose=final_pose,
        initial_maneuvers=initial_maneuvers,
        final_maneuvers=final_maneuvers,
        safe_union_cache=safe_union_cache,
    )

    if boundary_simplification is None:
        return failed("boundary_tangent_chain_could_not_be_simplified")

    (
        active_circle_indices,
        tangents,
        initial_maneuvers,
        final_maneuvers,
    ) = boundary_simplification

    # ---------------------------------------------------------------
    # 8. Assemble the complete primitive trajectory.
    # ---------------------------------------------------------------
    trajectory = assemble_refined_trajectory(
        corridor_list=corridor_list,
        circle_groups=circle_groups,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        initial_maneuvers=initial_maneuvers,
        final_maneuvers=final_maneuvers,
        bicycle=bicycle,
        r=r,
        safe_union_cache=safe_union_cache,
    )

    if trajectory is None:
        return failed("refined_trajectory_assembly_failed")

    traversal_time = float(
        sum(
            maneuver.maneuver_time
            for maneuver in trajectory
        )
    )

    return succeeded(BicycleRefinementResult(
        circle_groups=circle_groups,
        straight_passage_groups=straight_passage_groups,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        initial_maneuvers=initial_maneuvers,
        final_maneuvers=final_maneuvers,
        trajectory=trajectory,
        traversal_time=traversal_time,
    ))





# ===================================================================
# Boundary connections and complete refined trajectory
# ===================================================================


def resolve_refinement_boundary_poses(
    baseline,
    initial_pose,
    final_pose,
):
    """
    Resolve boundary poses from explicit arguments or from a previously
    boundary-connected baseline.
    """
    if (
        initial_pose is None
        and baseline.initial_maneuvers is not None
        and len(baseline.initial_maneuvers) > 0
    ):
        first = baseline.initial_maneuvers[0]
        initial_pose = [
            first.x0,
            first.y0,
            first.theta0,
        ]

    if (
        final_pose is None
        and baseline.final_maneuvers is not None
        and len(baseline.final_maneuvers) > 0
    ):
        last = baseline.final_maneuvers[-1]
        final_pose = [
            last.xf,
            last.yf,
            last.thetaf,
        ]

    return initial_pose, final_pose


def refinement_circle_to_intermediate_circle(
    circle,
    transition_index,
):
    """
    Adapt a RefinementCircle to the existing IntermediateCircle interface
    used by the pose-to-circle and arc-construction routines.
    """
    center = np.asarray(
        circle.center,
        dtype=float,
    )

    corner = np.asarray(
        circle.corner_point,
        dtype=float,
    )

    return IntermediateCircle(
        center=KappaPoint(
            float(center[0]),
            float(center[1]),
        ),
        radius=float(circle.radius),
        corner_point=KappaPoint(
            float(corner[0]),
            float(corner[1]),
        ),
        turn_direction=int(circle.turn_direction),
        index=int(transition_index),
    )


def compute_refinement_boundary_connection(
    corridor_list,
    circle_groups,
    transition_index,
    bicycle,
    boundary_pose,
    initial,
):
    """
    Connect one prescribed boundary pose to an arbitrary active refinement
    circle.

    For an initial connection to transition j, the admissible corridor union is

        C_0 union ... union C_{j+1}.

    For a final connection from transition j, the reversed problem uses

        C_j union ... union C_{n-1}.

    The pose-to-circle helper must support the optional
    ``admissible_corridors`` keyword so that a boundary circle can be skipped.
    """
    circle = get_single_circle(
        circle_groups[transition_index]
    )

    target_circle = refinement_circle_to_intermediate_circle(
        circle=circle,
        transition_index=transition_index,
    )

    if initial:
        admissible_corridors = corridor_list[
            : transition_index + 2
        ]

        if len(admissible_corridors) < 2:
            return None

        maneuvers = compute_traj_to_circle_bicycle(
            corridor1=corridor_list[0],
            corridor2=corridor_list[1],
            start_pose=boundary_pose,
            bicycle=bicycle,
            circ1=target_circle,
            tau0=0,
            figure=None,
            admissible_corridors=admissible_corridors,
        )

        if maneuvers is None or len(maneuvers) == 0:
            return None

        # Preserve circle-index metadata for the final arc construction.
        if getattr(
            maneuvers[-1],
            "label",
            None,
        ) == "segment":
            maneuvers[-1].end_circle_index = (
                transition_index
            )

        return maneuvers

    # ---------------------------------------------------------------
    # Final connection: solve the reversed boundary problem and invert
    # the primitive sequence back.
    # ---------------------------------------------------------------
    if len(corridor_list) < 2:
        return None

    (
        inverted_last_corridor,
        inverted_penultimate_corridor,
        inverted_target_circle,
        inverted_final_pose,
    ) = invert_inputs_all(
        corridor_list[-1],
        corridor_list[-2],
        target_circle,
        boundary_pose,
    )

    # Only the first reversed corridor needs the inverted wall labels for the
    # corrective-boundary construction. The remaining rectangles have the
    # same occupied set after reversal and are used only in the union test.
    admissible_corridors = (
        [inverted_last_corridor]
        + list(
            reversed(
                corridor_list[
                    transition_index : -1
                ]
            )
        )
    )

    inverted_maneuvers = compute_traj_to_circle_bicycle(
        corridor1=inverted_last_corridor,
        corridor2=inverted_penultimate_corridor,
        start_pose=inverted_final_pose,
        bicycle=bicycle,
        circ1=inverted_target_circle,
        tau0=0,
        figure=None,
        admissible_corridors=admissible_corridors,
    )

    if (
        inverted_maneuvers is None
        or len(inverted_maneuvers) == 0
    ):
        return None

    maneuvers = invert_maneuvers(
        inverted_maneuvers,
        t0=0.0,
    )

    if len(maneuvers) == 0:
        return None

    if getattr(
        maneuvers[0],
        "label",
        None,
    ) == "segment":
        maneuvers[0].start_circle_index = (
            transition_index
        )

    return maneuvers


def try_boundary_connection_with_baseline_fallback(
    corridor_list,
    circle_groups,
    baseline,
    bicycle,
    R,
    transition_index,
    boundary_pose,
    initial,
    tol=1e-9,
):
    """
    Try a boundary connection to the current refined circle.

    If it fails and the target circle is not already a baseline circle, replace
    that target by the validated baseline circle and retry.

    Returns
    -------
    (maneuvers, circle_changed)

    ``None`` maneuvers means that both the refined and baseline-circle attempts
    failed.
    """
    maneuvers = compute_refinement_boundary_connection(
        corridor_list=corridor_list,
        circle_groups=circle_groups,
        transition_index=transition_index,
        bicycle=bicycle,
        boundary_pose=boundary_pose,
        initial=initial,
    )

    if maneuvers is not None:
        return maneuvers, False

    circle = get_single_circle(
        circle_groups[transition_index]
    )

    # The requested fallback has already been exhausted.
    if is_baseline_circle(circle):
        return None, False

    original_group = circle_groups[
        transition_index
    ]

    baseline_circle = compute_baseline_turn_circle(
        baseline=baseline,
        transition_index=transition_index,
        R=R,
        placement_rule="baseline_boundary_repair",
        tol=tol,
    )

    if baseline_circle is None:
        return None, False

    tau = baseline.turn_directions[
        transition_index
    ]

    circle_groups[transition_index] = {
        tau: baseline_circle
    }

    maneuvers = compute_refinement_boundary_connection(
        corridor_list=corridor_list,
        circle_groups=circle_groups,
        transition_index=transition_index,
        bicycle=bicycle,
        boundary_pose=boundary_pose,
        initial=initial,
    )

    if maneuvers is None:
        # Transactional repair: do not destroy the previous circle merely
        # because the fallback boundary connection also failed.
        circle_groups[transition_index] = (
            original_group
        )
        return None, False

    return maneuvers, True


def rebuild_and_simplify_internal_chain(
    circle_groups,
    corridor_list,
    baseline,
    R,
    r,
    active_circle_indices,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Rebuild the complete internal tangent chain after any circle replacement or
    active-circle removal, then reapply the existing internal simplification.
    """
    active_circle_indices = list(
        active_circle_indices
    )

    if len(active_circle_indices) == 0:
        return None

    tangents = build_and_repair_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        safe_union_cache=safe_union_cache,
        tol=tol,
    )

    if tangents is None:
        return None

    simplification_result = simplify_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        safe_union_cache=safe_union_cache,
        tol=tol,
    )

    if simplification_result is None:
        return None

    return simplification_result


def build_consistent_boundary_state(
    circle_groups,
    corridor_list,
    baseline,
    bicycle,
    R,
    r,
    active_circle_indices,
    tangents,
    initial_pose,
    final_pose,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Attach both boundary poses to the current active circle chain.

    If either boundary target falls back from a refined circle to its baseline
    position, the internal tangent chain is rebuilt before boundary attachment
    is attempted again.
    """
    active_circle_indices = list(
        active_circle_indices
    )

    tangents = list(
        tangents
    )

    while True:
        if len(active_circle_indices) == 0:
            return None

        first_index = active_circle_indices[0]

        (
            initial_maneuvers,
            initial_circle_changed,
        ) = try_boundary_connection_with_baseline_fallback(
            corridor_list=corridor_list,
            circle_groups=circle_groups,
            baseline=baseline,
            bicycle=bicycle,
            R=R,
            transition_index=first_index,
            boundary_pose=initial_pose,
            initial=True,
            tol=tol,
        )

        if initial_maneuvers is None:
            return None

        if initial_circle_changed:
            internal_state = (
                rebuild_and_simplify_internal_chain(
                    circle_groups=circle_groups,
                    corridor_list=corridor_list,
                    baseline=baseline,
                    R=R,
                    r=r,
                    active_circle_indices=active_circle_indices,
                    safe_union_cache=safe_union_cache,
                    tol=tol,
                )
            )

            if internal_state is None:
                return None

            (
                active_circle_indices,
                tangents,
            ) = internal_state

            continue

        last_index = active_circle_indices[-1]

        (
            final_maneuvers,
            final_circle_changed,
        ) = try_boundary_connection_with_baseline_fallback(
            corridor_list=corridor_list,
            circle_groups=circle_groups,
            baseline=baseline,
            bicycle=bicycle,
            R=R,
            transition_index=last_index,
            boundary_pose=final_pose,
            initial=False,
            tol=tol,
        )

        if final_maneuvers is None:
            return None

        if final_circle_changed:
            internal_state = (
                rebuild_and_simplify_internal_chain(
                    circle_groups=circle_groups,
                    corridor_list=corridor_list,
                    baseline=baseline,
                    R=R,
                    r=r,
                    active_circle_indices=active_circle_indices,
                    safe_union_cache=safe_union_cache,
                    tol=tol,
                )
            )

            if internal_state is None:
                return None

            (
                active_circle_indices,
                tangents,
            ) = internal_state

            continue

        return (
            active_circle_indices,
            tangents,
            initial_maneuvers,
            final_maneuvers,
        )


def maneuver_segment_to_circle_tangent(
    segment,
    start_circle_index,
    end_circle_index,
):
    """
    Convert an existing straight primitive to the lightweight CircleTangent
    representation used by the intersection logic.
    """
    return CircleTangent(
        start_circle_index=start_circle_index,
        end_circle_index=end_circle_index,
        start_point=np.array(
            [
                segment.x0,
                segment.y0,
            ],
            dtype=float,
        ),
        end_point=np.array(
            [
                segment.xf,
                segment.yf,
            ],
            dtype=float,
        ),
        start_heading=float(
            segment.theta0
        ),
        end_heading=float(
            segment.thetaf
        ),
    )


def boundary_tangent_problem_side(
    active_circle_indices,
    tangents,
    initial_maneuvers,
    final_maneuvers,
    tol=1e-9,
):
    """
    Check the two boundary tangent segments against the entire current tangent
    chain.

    Returns
    -------
    "initial", "final", "both", or None.

    Internal/internal tangent intersections have already been resolved by
    ``simplify_tangent_chain``.
    """
    if (
        initial_maneuvers is None
        or final_maneuvers is None
        or len(initial_maneuvers) == 0
        or len(final_maneuvers) == 0
    ):
        return "both"

    initial_segment = initial_maneuvers[-1]
    final_segment = final_maneuvers[0]

    if (
        getattr(initial_segment, "label", None)
        != "segment"
        or getattr(final_segment, "label", None)
        != "segment"
    ):
        return "both"

    initial_tangent = maneuver_segment_to_circle_tangent(
        segment=initial_segment,
        start_circle_index=-1,
        end_circle_index=active_circle_indices[0],
    )

    final_tangent = maneuver_segment_to_circle_tangent(
        segment=final_segment,
        start_circle_index=active_circle_indices[-1],
        end_circle_index=len(active_circle_indices),
    )

    initial_problem = False
    final_problem = False

    for tangent in tangents:
        if tangent_segments_have_problematic_intersection(
            tangent_1=initial_tangent,
            tangent_2=tangent,
            adjacent=False,
            tol=tol,
        ):
            initial_problem = True
            break

    for tangent in tangents:
        if tangent_segments_have_problematic_intersection(
            tangent_1=tangent,
            tangent_2=final_tangent,
            adjacent=False,
            tol=tol,
        ):
            final_problem = True
            break

    # The two boundary segments can also intersect directly, especially after
    # aggressive internal skipping.
    if tangent_segments_have_problematic_intersection(
        tangent_1=initial_tangent,
        tangent_2=final_tangent,
        adjacent=False,
        tol=tol,
    ):
        initial_problem = True
        final_problem = True

    if initial_problem and final_problem:
        return "both"

    if initial_problem:
        return "initial"

    if final_problem:
        return "final"

    return None


def try_skip_boundary_circle(
    side,
    circle_groups,
    corridor_list,
    baseline,
    bicycle,
    R,
    r,
    active_circle_indices,
    initial_pose,
    final_pose,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Try to remove exactly one active circle at the selected boundary.

    The new boundary target is first attempted at its refined position. If that
    boundary connection fails, the target circle is tried at its validated
    baseline position. The candidate state is committed only if the complete
    internal tangent chain and both boundary connections are feasible.
    """
    if side not in {
        "initial",
        "final",
    }:
        raise ValueError(
            "side must be 'initial' or 'final'."
        )

    if len(active_circle_indices) <= 1:
        return None

    candidate_circle_groups = deepcopy(
        circle_groups
    )

    candidate_active_indices = list(
        active_circle_indices
    )

    if side == "initial":
        del candidate_active_indices[0]
    else:
        del candidate_active_indices[-1]

    internal_state = rebuild_and_simplify_internal_chain(
        circle_groups=candidate_circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=candidate_active_indices,
        safe_union_cache=safe_union_cache,
        tol=tol,
    )

    if internal_state is None:
        return None

    (
        candidate_active_indices,
        candidate_tangents,
    ) = internal_state

    boundary_state = build_consistent_boundary_state(
        circle_groups=candidate_circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        bicycle=bicycle,
        R=R,
        r=r,
        active_circle_indices=candidate_active_indices,
        tangents=candidate_tangents,
        initial_pose=initial_pose,
        final_pose=final_pose,
        safe_union_cache=safe_union_cache,
        tol=tol,
    )

    if boundary_state is None:
        return None

    (
        candidate_active_indices,
        candidate_tangents,
        candidate_initial_maneuvers,
        candidate_final_maneuvers,
    ) = boundary_state

    return (
        candidate_circle_groups,
        candidate_active_indices,
        candidate_tangents,
        candidate_initial_maneuvers,
        candidate_final_maneuvers,
    )


def simplify_boundary_tangent_chain(
    circle_groups,
    corridor_list,
    baseline,
    bicycle,
    R,
    r,
    active_circle_indices,
    tangents,
    initial_pose,
    final_pose,
    initial_maneuvers,
    final_maneuvers,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Extend tangent-chain simplification to the trajectory boundaries.

    A problematic initial boundary tangent removes the current first active
    circle and rebuilds the pose-to-circle connection to the next one. The final
    side is symmetric. A failed speculative skip is not committed.

    If both sides are problematic, the initial side is attempted first and the
    entire chain is then rechecked.
    """
    active_circle_indices = list(
        active_circle_indices
    )

    tangents = list(
        tangents
    )

    while True:
        problem_side = boundary_tangent_problem_side(
            active_circle_indices=active_circle_indices,
            tangents=tangents,
            initial_maneuvers=initial_maneuvers,
            final_maneuvers=final_maneuvers,
            tol=tol,
        )

        if problem_side is None:
            return (
                active_circle_indices,
                tangents,
                initial_maneuvers,
                final_maneuvers,
            )

        if len(active_circle_indices) <= 1:
            # The last circle is deliberately retained. A direct boundary
            # pose-to-pose repair is outside the present planner.
            return None

        side_to_try = (
            "initial"
            if problem_side in {
                "initial",
                "both",
            }
            else "final"
        )

        candidate = try_skip_boundary_circle(
            side=side_to_try,
            circle_groups=circle_groups,
            corridor_list=corridor_list,
            baseline=baseline,
            bicycle=bicycle,
            R=R,
            r=r,
            active_circle_indices=active_circle_indices,
            initial_pose=initial_pose,
            final_pose=final_pose,
            safe_union_cache=safe_union_cache,
            tol=tol,
        )

        if candidate is None:
            # The candidate target has already gone through the requested
            # refined-circle -> baseline-circle fallback. No additional repair
            # is available for this boundary intersection.
            return None

        (
            candidate_circle_groups,
            active_circle_indices,
            tangents,
            initial_maneuvers,
            final_maneuvers,
        ) = candidate

        # Commit the transactional candidate.
        circle_groups[:] = (
            candidate_circle_groups
        )


def circle_tangent_to_linear_segment(
    tangent,
    bicycle,
):
    """
    Convert a stored circle-to-circle tangent to the existing straight primitive.
    """
    return LinearSegmentBicycle(
        x0=float(
            tangent.start_point[0]
        ),
        y0=float(
            tangent.start_point[1]
        ),
        xf=float(
            tangent.end_point[0]
        ),
        yf=float(
            tangent.end_point[1]
        ),
        theta=float(
            tangent.start_heading
        ),
        v=bicycle.v_max,
        bicycle=bicycle,
        t0=0.0,
        start_circle_index=(
            tangent.start_circle_index
        ),
        end_circle_index=(
            tangent.end_circle_index
        ),
    )


def directed_circle_amplitude(
    center,
    start_point,
    end_point,
    turn_direction,
):
    """
    Positive angular amplitude from ``start_point`` to ``end_point`` around
    ``center`` while following ``turn_direction``.
    """
    center = np.asarray(
        center,
        dtype=float,
    )

    start_point = np.asarray(
        start_point,
        dtype=float,
    )

    end_point = np.asarray(
        end_point,
        dtype=float,
    )

    start_angle = np.arctan2(
        start_point[1] - center[1],
        start_point[0] - center[0],
    )

    end_angle = np.arctan2(
        end_point[1] - center[1],
        end_point[0] - center[0],
    )

    if turn_direction == 1:
        return float(
            (
                end_angle
                - start_angle
            )
            % (2.0 * np.pi)
        )

    if turn_direction == -1:
        return float(
            (
                start_angle
                - end_angle
            )
            % (2.0 * np.pi)
        )

    raise ValueError(
        "turn_direction must be +1 or -1."
    )


def make_forward_arc_collision_view(
    reference_arc,
    start_point,
    amplitude,
    bicycle,
):
    """
    Build the minimal arc-like object required by ``check_arc_collision``.

    This avoids introducing another motion-primitive class merely for a local
    geometric collision test.
    """
    start_point = np.asarray(
        start_point,
        dtype=float,
    )

    return SimpleNamespace(
        x0=float(start_point[0]),
        y0=float(start_point[1]),
        xc=float(reference_arc.xc),
        yc=float(reference_arc.yc),
        radius=float(reference_arc.radius),
        turn_direction=int(
            reference_arc.turn_direction
        ),
        iota=float(amplitude),
        label="arc",
        bicycle=bicycle,
    )


def refinement_circle_canonical_tangency_points(
    circle,
):
    """
    Return the endpoints of the locally certified quarter-circle arc.

    The local refinement frame satisfies

        local_x_axis = - direction_in,
        local_y_axis =   direction_out.

    Hence the canonical incoming and outgoing corridor directions are recovered
    directly from the stored local axes.
    """
    center = np.asarray(
        circle.center,
        dtype=float,
    )

    direction_in = -np.asarray(
        circle.local_x_axis,
        dtype=float,
    )

    direction_out = np.asarray(
        circle.local_y_axis,
        dtype=float,
    )

    tau = circle.turn_direction
    R = circle.radius

    left_normal_in = np.array(
        [
            -direction_in[1],
            direction_in[0],
        ],
        dtype=float,
    )

    left_normal_out = np.array(
        [
            -direction_out[1],
            direction_out[0],
        ],
        dtype=float,
    )

    canonical_start = (
        center
        - tau * R * left_normal_in
    )

    canonical_end = (
        center
        - tau * R * left_normal_out
    )

    return (
        canonical_start,
        canonical_end,
    )


def assembled_refinement_arc_is_safe(
    arc,
    circle,
    corridor_in,
    corridor_out,
    bicycle,
    tol=1e-7,
):
    """
    Validate an assembled arc without trajectory/footprint sampling.

    The circle-placement rules certify the quarter-circle portion tangent to the
    incoming and outgoing baseline corridor directions. If the assembled arc is
    a subarc of that certified portion, no further collision test is required.

    If the assembled arc extends before the certified quarter arc, only that
    prefix is new and it is checked geometrically in ``corridor_in`` using
    ``check_arc_collision``. Likewise, a suffix extending beyond the certified
    quarter arc is checked in ``corridor_out``.

    The amplitude consistency check prevents acceptance of a long wrap around
    the circle.
    """
    center = np.asarray(
        circle.center,
        dtype=float,
    )

    (
        canonical_start,
        canonical_end,
    ) = refinement_circle_canonical_tangency_points(
        circle
    )

    assembled_start = np.array(
        [
            arc.x0,
            arc.y0,
        ],
        dtype=float,
    )

    assembled_end = np.array(
        [
            arc.xf,
            arc.yf,
        ],
        dtype=float,
    )

    tau = circle.turn_direction

    canonical_amplitude = (
        directed_circle_amplitude(
            center=center,
            start_point=canonical_start,
            end_point=canonical_end,
            turn_direction=tau,
        )
    )

    # For the orthogonal baseline geometry this should be pi/2. Keep the
    # computed value so that the check follows the actual stored directions.
    start_coordinate = (
        directed_circle_amplitude(
            center=center,
            start_point=canonical_start,
            end_point=assembled_start,
            turn_direction=tau,
        )
    )

    end_coordinate = (
        directed_circle_amplitude(
            center=center,
            start_point=canonical_start,
            end_point=assembled_end,
            turn_direction=tau,
        )
    )

    start_on_certified_arc = (
        start_coordinate
        <= canonical_amplitude + tol
    )

    end_on_certified_arc = (
        end_coordinate
        <= canonical_amplitude + tol
    )

    prefix_amplitude = 0.0

    if not start_on_certified_arc:
        prefix_amplitude = (
            directed_circle_amplitude(
                center=center,
                start_point=assembled_start,
                end_point=canonical_start,
                turn_direction=tau,
            )
        )

    suffix_amplitude = 0.0

    if not end_on_certified_arc:
        suffix_amplitude = (
            directed_circle_amplitude(
                center=center,
                start_point=canonical_end,
                end_point=assembled_end,
                turn_direction=tau,
            )
        )

    # Reconstruct the only accepted ordering:
    #
    #     optional prefix
    #       -> certified local quarter arc (or a subarc of it)
    #       -> optional suffix
    #
    # This rejects trajectories that wrap around the opposite side of the
    # supporting circle.
    if (
        start_on_certified_arc
        and end_on_certified_arc
    ):
        if (
            end_coordinate
            < start_coordinate - tol
        ):
            return False

        expected_amplitude = (
            end_coordinate
            - start_coordinate
        )

    elif start_on_certified_arc:
        expected_amplitude = (
            canonical_amplitude
            - start_coordinate
            + suffix_amplitude
        )

    elif end_on_certified_arc:
        expected_amplitude = (
            prefix_amplitude
            + end_coordinate
        )

    else:
        expected_amplitude = (
            prefix_amplitude
            + canonical_amplitude
            + suffix_amplitude
        )

    if abs(
        expected_amplitude
        - float(arc.iota)
    ) > tol:
        return False

    # Only genuinely new portions require collision checks.
    if prefix_amplitude > tol:
        prefix_arc = make_forward_arc_collision_view(
            reference_arc=arc,
            start_point=assembled_start,
            amplitude=prefix_amplitude,
            bicycle=bicycle,
        )

        collision, _ = check_arc_collision(
            prefix_arc,
            corridor_in,
        )

        if collision:
            return False

    if suffix_amplitude > tol:
        suffix_arc = make_forward_arc_collision_view(
            reference_arc=arc,
            start_point=canonical_end,
            amplitude=suffix_amplitude,
            bicycle=bicycle,
        )

        collision, _ = check_arc_collision(
            suffix_arc,
            corridor_out,
        )

        if collision:
            return False

    return True


def finalize_refined_maneuver_sequence(
    maneuvers,
    tol=1e-9,
):
    """
    Make primitive timing and angle representations continuous.
    """
    for index in range(
        1,
        len(maneuvers),
    ):
        previous_tf = maneuvers[
            index - 1
        ].time_grid[-1]

        current_t0 = maneuvers[
            index
        ].time_grid[0]

        offset = (
            previous_tf
            - current_t0
        )

        if abs(offset) > tol:
            maneuvers[
                index
            ].add_time_offset(
                offset
            )

    correct_angles(
        maneuvers
    )


def refined_maneuver_sequence_is_continuous(
    maneuvers,
    position_tol=1e-7,
    heading_tol=1e-7,
):
    """
    Check C0 position and heading continuity of the assembled primitive chain.
    """
    for previous, current in zip(
        maneuvers[:-1],
        maneuvers[1:],
    ):
        position_error = np.hypot(
            previous.xf - current.x0,
            previous.yf - current.y0,
        )

        if position_error > position_tol:
            return False

        heading_error = abs(
            np.arctan2(
                np.sin(
                    current.theta0
                    - previous.thetaf
                ),
                np.cos(
                    current.theta0
                    - previous.thetaf
                ),
            )
        )

        if heading_error > heading_tol:
            return False

    return True


def assemble_refined_trajectory(
    corridor_list,
    circle_groups,
    active_circle_indices,
    tangents,
    initial_maneuvers,
    final_maneuvers,
    bicycle,
    r,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Assemble the final boundary-connected refined primitive sequence.

    Supporting straight segments are

        initial boundary tangent,
        retained circle-to-circle tangents,
        final boundary tangent.

    One circular arc is then constructed on every retained circle between its
    incoming and outgoing supporting segments.
    """
    if (
        initial_maneuvers is None
        or final_maneuvers is None
        or len(initial_maneuvers) == 0
        or len(final_maneuvers) == 0
        or len(active_circle_indices) == 0
    ):
        return None

    if len(tangents) != max(
        0,
        len(active_circle_indices) - 1,
    ):
        return None

    initial_segment = initial_maneuvers[-1]
    final_segment = final_maneuvers[0]

    if (
        getattr(initial_segment, "label", None)
        != "segment"
        or getattr(final_segment, "label", None)
        != "segment"
    ):
        return None

    internal_segments = [
        circle_tangent_to_linear_segment(
            tangent=tangent,
            bicycle=bicycle,
        )
        for tangent in tangents
    ]

    supporting_segments = (
        [initial_segment]
        + internal_segments
        + [final_segment]
    )

    if len(supporting_segments) != (
        len(active_circle_indices) + 1
    ):
        return None

    arcs = []

    for local_position, transition_index in enumerate(
        active_circle_indices
    ):
        circle = get_single_circle(
            circle_groups[transition_index]
        )

        intermediate_circle = (
            refinement_circle_to_intermediate_circle(
                circle=circle,
                transition_index=transition_index,
            )
        )

        arc = compute_arc_from_two_tangents_objects(
            supporting_segments[
                local_position
            ],
            supporting_segments[
                local_position + 1
            ],
            intermediate_circle,
            bicycle,
        )

        if arc is None:
            return None

        if not assembled_refinement_arc_is_safe(
            arc=arc,
            circle=circle,
            corridor_in=corridor_list[
                transition_index
            ],
            corridor_out=corridor_list[
                transition_index + 1
            ],
            bicycle=bicycle,
            tol=max(
                tol,
                1e-7,
            ),
        ):
            return None

        arcs.append(
            arc
        )

    middle_sequence = []

    for index, arc in enumerate(
        arcs
    ):
        middle_sequence.append(
            arc
        )

        if index < len(
            internal_segments
        ):
            middle_sequence.append(
                internal_segments[index]
            )

    trajectory = (
        list(initial_maneuvers)
        + middle_sequence
        + list(final_maneuvers)
    )

    finalize_refined_maneuver_sequence(
        trajectory
    )

    if not refined_maneuver_sequence_is_continuous(
        trajectory
    ):
        return None

    return trajectory





# ===================================================================

# Straight-passage grouping

# ===================================================================



def compute_straight_passage_groups(

    turn_directions,

):

    """

    Find maximal runs of aligned transitions (tau = 0).



    Each zero run is extended by the genuine turn immediately before it

    and the genuine turn immediately after it, whenever those turns exist.



    Examples

    --------



    [1, 1, 0, 1]



        zero run: [2]

        group:    [1, 2, 3]



    [1, 0, 0, -1]



        zero run: [1, 2]

        group:    [0, 1, 2, 3]



    [0, 0, 1]



        zero run: [0, 1]

        group:    [0, 1, 2]

        left_circle_index = None



    [1, 0, 0]



        zero run: [1, 2]

        group:    [0, 1, 2]

        right_circle_index = None



    Notes

    -----

    Two consecutive straight-passage groups may share one genuine-turn

    transition. This is intentional: that shared turn is adjacent to both

    zero runs and is therefore kept at its baseline position.

    """

    turn_directions = list(turn_directions)



    groups = []



    m = len(turn_directions)



    j = 0



    while j < m:



        # Not the beginning of a zero run.

        if turn_directions[j] != 0:

            j += 1

            continue



        # -----------------------------------------------------------

        # Maximal zero run

        # -----------------------------------------------------------

        zero_start = j



        while (

            j + 1 < m

            and turn_directions[j + 1] == 0

        ):

            j += 1



        zero_end = j



        # -----------------------------------------------------------

        # Genuine turn immediately before the zero run

        # -----------------------------------------------------------

        left_circle_index = None



        if zero_start > 0:



            candidate = zero_start - 1



            if turn_directions[candidate] != 0:

                left_circle_index = candidate



        # -----------------------------------------------------------

        # Genuine turn immediately after the zero run

        # -----------------------------------------------------------

        right_circle_index = None



        if zero_end + 1 < m:



            candidate = zero_end + 1



            if turn_directions[candidate] != 0:

                right_circle_index = candidate



        # -----------------------------------------------------------

        # Full group limits

        # -----------------------------------------------------------

        if left_circle_index is not None:

            start_index = left_circle_index

        else:

            start_index = zero_start



        if right_circle_index is not None:

            end_index = right_circle_index

        else:

            end_index = zero_end



        groups.append(

            StraightPassageGroup(

                start_index=start_index,

                end_index=end_index,

                zero_start_index=zero_start,

                zero_end_index=zero_end,

                left_circle_index=left_circle_index,

                right_circle_index=right_circle_index,

            )

        )



        j += 1



    return groups





# ===================================================================

# Baseline genuine-turn circle

# ===================================================================



def compute_baseline_turn_circle(

    baseline,

    transition_index,

    R,

    placement_rule="baseline",

    tol=1e-9,

):

    """

    Construct a RefinementCircle directly from the validated baseline

    fillet at a genuine turn.

    """

    j = transition_index



    tau = baseline.turn_directions[j]



    if tau not in (-1, 1):

        raise ValueError(

            "A baseline turning circle exists only at a genuine turn."

        )



    baseline_fillet = baseline.fillets[j]



    if baseline_fillet is None:

        return None



    corner_point = np.asarray(

        baseline.candidate_corner_points[j][tau],

        dtype=float,

    )



    admissible_region = baseline.admissible_regions[j]



    local_x_axis = np.asarray(

        admissible_region.local_x_axis,

        dtype=float,

    )



    local_y_axis = np.asarray(

        admissible_region.local_y_axis,

        dtype=float,

    )



    center = np.asarray(

        baseline_fillet.center,

        dtype=float,

    )



    delta = center - corner_point



    x_local = np.dot(

        delta,

        local_x_axis,

    )



    y_local = np.dot(

        delta,

        local_y_axis,

    )



    dimensions = compute_effective_junction_dimensions(

        corridor_overlap=baseline.corridor_overlaps[j],

        corner_point=corner_point,

        local_x_axis=local_x_axis,

        local_y_axis=local_y_axis,

        tol=tol,

    )



    return RefinementCircle(

        center=center,

        radius=R,

        turn_direction=tau,

        corner_point=corner_point,

        local_x_axis=local_x_axis,

        local_y_axis=local_y_axis,

        local_coordinates=(

            x_local,

            y_local,

        ),

        effective_dimensions=dimensions,

        placement_rule=placement_rule,

    )





# ===================================================================

# Independent genuine-turn circle positioning

# ===================================================================



def compute_independent_turn_circle(

    baseline,

    transition_index,

    R,

    r,

    tol=1e-9,

):

    """

    Position the refinement circle associated with a genuine baseline turn.



    The four local heuristic placement rules are attempted first. If none

    can be applied, the validated baseline fillet circle is retained.

    """

    j = transition_index



    tau = baseline.turn_directions[j]



    if tau not in (-1, 1):

        raise ValueError(

            "This helper is defined only for genuine turns."

        )



    # ---------------------------------------------------------------

    # Geometry already determined by the baseline

    # ---------------------------------------------------------------

    corner_point = np.asarray(

        baseline.candidate_corner_points[j][tau],

        dtype=float,

    )



    admissible_region = baseline.admissible_regions[j]



    local_x_axis = np.asarray(

        admissible_region.local_x_axis,

        dtype=float,

    )



    local_y_axis = np.asarray(

        admissible_region.local_y_axis,

        dtype=float,

    )



    # ---------------------------------------------------------------

    # Effective junction dimensions

    # ---------------------------------------------------------------

    dimensions = compute_effective_junction_dimensions(

        corridor_overlap=baseline.corridor_overlaps[j],

        corner_point=corner_point,

        local_x_axis=local_x_axis,

        local_y_axis=local_y_axis,

        tol=tol,

    )



    local_position = None



    if dimensions is not None:



        d_x, d_y = dimensions



        local_position = compute_independent_circle_local_position(

            d_x=d_x,

            d_y=d_y,

            R=R,

            r=r,

            tol=tol,

        )



    # ---------------------------------------------------------------

    # Four local rules succeeded

    # ---------------------------------------------------------------

    if local_position is not None:



        x_local, y_local, placement_rule = local_position



        center = (

            corner_point

            + x_local * local_x_axis

            + y_local * local_y_axis

        )



        return RefinementCircle(

            center=center,

            radius=R,

            turn_direction=tau,

            corner_point=corner_point,

            local_x_axis=local_x_axis,

            local_y_axis=local_y_axis,

            local_coordinates=(

                x_local,

                y_local,

            ),

            effective_dimensions=dimensions,

            placement_rule=placement_rule,

        )



    # ---------------------------------------------------------------

    # Fallback: validated baseline fillet circle

    # ---------------------------------------------------------------

    return compute_baseline_turn_circle(

        baseline=baseline,

        transition_index=j,

        R=R,

        placement_rule="baseline",

        tol=tol,

    )





# ===================================================================

# Collision-free corridor union for tangent validation

# ===================================================================





def get_connection_safe_union(

    corridor_list,

    start_circle_index,

    end_circle_index,

    r,

    cache=None,

):

    """

    Return W_free for the local corridor subsequence traversed by a

    circle-to-circle tangent.



    A circle at transition j is associated with the pair (C_j, C_{j+1}).

    Therefore a tangent from transition i to transition k is checked in

    the union C_i union ... union C_{k+1}.

    """

    key = (

        start_circle_index,

        end_circle_index,

    )



    if cache is not None and key in cache:

        return cache[key]



    local_corridors = corridor_list[

        start_circle_index : end_circle_index + 2

    ]



    safe_union = compute_safe_corridor_union(

        corridor_list=local_corridors,

        r=r,

    )



    if cache is not None:

        cache[key] = safe_union



    return safe_union





# ===================================================================

# Tangent construction and local baseline repair

# ===================================================================



def get_active_circle_indices(

    circle_groups,

):

    """

    Return the transition indices containing genuine-turn circles,

    in trajectory order.

    """

    return [

        j

        for j, group in enumerate(circle_groups)

        if len(group) > 0

    ]





def get_single_circle(

    circle_group,

):

    """Return the single genuine-turn circle stored in a group."""

    if len(circle_group) != 1:

        raise ValueError(

            "Each active circle group must contain exactly one circle."

        )



    return next(

        iter(circle_group.values())

    )





def is_baseline_circle(

    circle,

):

    """Return True if the circle already uses a baseline position."""

    return circle.placement_rule in {

        "baseline",

        "baseline_straight_passage",

        "baseline_repair",

        "baseline_boundary_repair",

    }





def tangent_length(

    tangent,

):

    """Return the length of the stored finite tangent segment."""

    return np.linalg.norm(

        tangent.end_point

        - tangent.start_point

    )





def compute_correct_circle_tangent(

    circle_1,

    circle_2,

    start_circle_index,

    end_circle_index,

    tol=1e-9,

):

    """

    Compute the tangent compatible with an ordered pair of equal-radius

    turning circles.



    Equal turn directions use the compatible external common tangent.

    Opposite turn directions use the compatible internal common tangent.



    The tangent is oriented from circle_1 toward circle_2.



    :return:

        CircleTangent, or None if the required tangent does not exist

        uniquely in the considered geometry.

    """

    center_1 = np.asarray(

        circle_1.center,

        dtype=float,

    )



    center_2 = np.asarray(

        circle_2.center,

        dtype=float,

    )



    R_1 = float(circle_1.radius)

    R_2 = float(circle_2.radius)



    tau_1 = circle_1.turn_direction

    tau_2 = circle_2.turn_direction



    if tau_1 not in (-1, 1):

        raise ValueError(

            "circle_1 must have turn direction +/-1."

        )



    if tau_2 not in (-1, 1):

        raise ValueError(

            "circle_2 must have turn direction +/-1."

        )



    if abs(R_1 - R_2) > tol:

        raise ValueError(

            "This helper assumes equal-radius circles."

        )



    R = 0.5 * (R_1 + R_2)



    delta = (

        center_2

        - center_1

    )



    center_distance = np.linalg.norm(

        delta

    )



    # Coincident equal-radius circles do not define a unique common

    # tangent. In the present refinement they are therefore repaired

    # through the validated baseline.

    if center_distance <= tol:

        return None



    center_direction = (

        delta

        / center_distance

    )



    # ---------------------------------------------------------------

    # Equal turns: compatible external tangent

    # ---------------------------------------------------------------

    if tau_1 == tau_2:



        tangent_direction = (

            center_direction

        )



    # ---------------------------------------------------------------

    # Opposite turns: compatible internal tangent

    # ---------------------------------------------------------------

    else:



        # A nondegenerate internal tangent requires c >= 2R.

        if center_distance < 2.0 * R - tol:

            return None



        ratio = (

            2.0 * R

            / center_distance

        )



        ratio = np.clip(

            ratio,

            -1.0,

            1.0,

        )



        beta = np.arcsin(

            ratio

        )



        center_angle = np.arctan2(

            center_direction[1],

            center_direction[0],

        )



        # This branch is the one compatible with both the ordering

        # circle_1 -> circle_2 and the turn direction of circle_1.

        tangent_angle = (

            center_angle

            + tau_1 * beta

        )



        tangent_direction = np.array([

            np.cos(tangent_angle),

            np.sin(tangent_angle),

        ])



    # ---------------------------------------------------------------

    # Tangency points

    # ---------------------------------------------------------------

    left_normal = np.array([

        -tangent_direction[1],

        tangent_direction[0],

    ])



    radial_direction_1 = (

        -tau_1

        * left_normal

    )



    radial_direction_2 = (

        -tau_2

        * left_normal

    )



    start_point = (

        center_1

        + R * radial_direction_1

    )



    end_point = (

        center_2

        + R * radial_direction_2

    )



    tangent_vector = (

        end_point

        - start_point

    )



    tangent_segment_length = np.linalg.norm(

        tangent_vector

    )



    # At c = 2R for opposite turns, the internal tangent degenerates

    # to one contact point. Keep the limiting tangent heading.

    if tangent_segment_length > tol:



        actual_direction = (

            tangent_vector

            / tangent_segment_length

        )



        if (

            np.dot(

                actual_direction,

                tangent_direction,

            )

            < 1.0 - 1e-7

        ):

            return None



        tangent_heading = np.arctan2(

            actual_direction[1],

            actual_direction[0],

        )



    else:



        tangent_heading = np.arctan2(

            tangent_direction[1],

            tangent_direction[0],

        )



    return CircleTangent(

        start_circle_index=start_circle_index,

        end_circle_index=end_circle_index,

        start_point=start_point,

        end_point=end_point,

        start_heading=tangent_heading,

        end_heading=tangent_heading,

    )





def compute_safe_circle_tangent_between_indices(

    circle_groups,

    corridor_list,

    start_circle_index,

    end_circle_index,

    r,

    safe_union_cache=None,

    tol=1e-9,

):

    """

    Compute the ordered common tangent between two retained circles and

    accept it only if the complete finite tangent segment lies inside the

    eroded union of the corridor subsequence between the two transitions.



    The circles between the two endpoint circles, if any, are deliberately

    irrelevant to this test: when this helper is used for a skip, those

    intermediate circles are being removed from the active chain.

    """

    if start_circle_index >= end_circle_index:

        raise ValueError(

            "start_circle_index must precede end_circle_index."

        )



    circle_1 = get_single_circle(

        circle_groups[start_circle_index]

    )



    circle_2 = get_single_circle(

        circle_groups[end_circle_index]

    )



    tangent = compute_correct_circle_tangent(

        circle_1=circle_1,

        circle_2=circle_2,

        start_circle_index=start_circle_index,

        end_circle_index=end_circle_index,

        tol=tol,

    )



    if tangent is None:

        return None



    safe_union = get_connection_safe_union(

        corridor_list=corridor_list,

        start_circle_index=start_circle_index,

        end_circle_index=end_circle_index,

        r=r,

        cache=safe_union_cache,

    )



    if not segment_inside_safe_corridor_union(

        start_point=tangent.start_point,

        end_point=tangent.end_point,

        safe_union=safe_union,

        tol=tol,

    ):

        return None



    return tangent





def build_and_repair_tangent_chain(

    circle_groups,

    corridor_list,

    baseline,

    R,

    r,

    active_circle_indices=None,

    safe_union_cache=None,

    tol=1e-9,

):

    """

    Build tangents between consecutive active genuine-turn circles.



    A tangent is accepted only when the correct ordered common tangent

    exists and its complete finite segment is contained in the locally

    relevant eroded corridor union.



    If the connection fails, both endpoint circles are restored to their

    validated baseline positions. The preceding connection is then checked

    again because the first endpoint circle may have moved.

    """

    if active_circle_indices is None:

        circle_indices = get_active_circle_indices(

            circle_groups

        )

    else:

        circle_indices = list(

            active_circle_indices

        )



    if len(circle_indices) <= 1:

        return []



    if safe_union_cache is None:

        safe_union_cache = {}



    tangents = [

        None

        for _ in range(len(circle_indices) - 1)

    ]



    k = 0



    while k < len(circle_indices) - 1:



        start_index = circle_indices[k]

        end_index = circle_indices[k + 1]



        tangent = compute_safe_circle_tangent_between_indices(

            circle_groups=circle_groups,

            corridor_list=corridor_list,

            start_circle_index=start_index,

            end_circle_index=end_index,

            r=r,

            safe_union_cache=safe_union_cache,

            tol=tol,

        )



        if tangent is not None:

            tangents[k] = tangent

            k += 1

            continue



        circle_1 = get_single_circle(

            circle_groups[start_index]

        )



        circle_2 = get_single_circle(

            circle_groups[end_index]

        )



        # Both circles are already at baseline: this local repair has

        # no additional action available.

        if (

            is_baseline_circle(circle_1)

            and is_baseline_circle(circle_2)

        ):

            return None



        repaired_circle_1 = compute_baseline_turn_circle(

            baseline=baseline,

            transition_index=start_index,

            R=R,

            placement_rule="baseline_repair",

            tol=tol,

        )



        repaired_circle_2 = compute_baseline_turn_circle(

            baseline=baseline,

            transition_index=end_index,

            R=R,

            placement_rule="baseline_repair",

            tol=tol,

        )



        if (

            repaired_circle_1 is None

            or repaired_circle_2 is None

        ):

            return None



        tau_1 = baseline.turn_directions[start_index]

        tau_2 = baseline.turn_directions[end_index]



        circle_groups[start_index] = {

            tau_1: repaired_circle_1

        }



        circle_groups[end_index] = {

            tau_2: repaired_circle_2

        }



        tangents[k] = None



        if k > 0:

            tangents[k - 1] = None



        k = max(

            0,

            k - 1,

        )



    return tangents





def tangent_segments_have_problematic_intersection(

    tangent_1,

    tangent_2,

    adjacent=False,

    tol=1e-9,

):

    """

    Return True when two finite tangent segments intersect in a way that

    creates a self-intersection of the tangent chain.



    For adjacent tangents, one common endpoint is allowed when the outgoing

    tangency point of the first tangent coincides with the incoming tangency

    point of the second tangent. Any other point intersection or any finite

    overlap is considered problematic.

    """

    def geometry_from_tangent(tangent):

        p0 = np.asarray(

            tangent.start_point,

            dtype=float,

        )

        p1 = np.asarray(

            tangent.end_point,

            dtype=float,

        )



        if np.linalg.norm(p1 - p0) <= tol:

            return Point(

                p0[0],

                p0[1],

            )



        return LineString([

            tuple(p0),

            tuple(p1),

        ])



    geometry_1 = geometry_from_tangent(

        tangent_1

    )

    geometry_2 = geometry_from_tangent(

        tangent_2

    )



    intersection = geometry_1.intersection(

        geometry_2

    )



    if intersection.is_empty:

        return False



    if adjacent:



        expected_1 = np.asarray(

            tangent_1.end_point,

            dtype=float,

        )

        expected_2 = np.asarray(

            tangent_2.start_point,

            dtype=float,

        )



        if np.linalg.norm(

            expected_1 - expected_2

        ) <= tol:



            expected_point = 0.5 * (

                expected_1 + expected_2

            )



            allowed_region = Point(

                expected_point[0],

                expected_point[1],

            ).buffer(

                max(tol, 1e-12)

            )



            if allowed_region.covers(

                intersection

            ):

                return False



    return True





def find_first_problematic_tangent_intersection(

    tangents,

    tol=1e-9,

):

    """

    Find the first pair of tangent segments that produces a self-intersection.



    The pair need not be consecutive. If tangent i intersects tangent j, the

    affected circle block runs from active circle position i through active

    circle position j + 1.



    :return:

        (i, j), or None when the tangent chain has no segment intersections.

    """

    for i in range(len(tangents)):



        for j in range(i + 1, len(tangents)):



            if tangent_segments_have_problematic_intersection(

                tangent_1=tangents[i],

                tangent_2=tangents[j],

                adjacent=(j == i + 1),

                tol=tol,

            ):

                return (

                    i,

                    j,

                )



    return None





def find_longest_safe_skip_in_block(

    active_circle_indices,

    circle_groups,

    corridor_list,

    r,

    block_start_position,

    block_end_position,

    safe_union_cache=None,

    tol=1e-9,

):

    """

    Search a problematic active-circle block for the longest safe direct

    tangent that skips at least one intermediate circle.



    Candidate skips are tested from largest to smallest number of removed

    circles. The first collision-free direct connection is returned.



    :return:

        (start_position, end_position, tangent), or None.

    """

    if (

        block_start_position < 0

        or block_end_position >= len(active_circle_indices)

        or block_start_position >= block_end_position

    ):

        raise ValueError(

            "Invalid active-circle block."

        )



    maximum_span = (

        block_end_position

        - block_start_position

    )



    # span >= 2 means that at least one intermediate circle is skipped.

    for span in range(

        maximum_span,

        1,

        -1,

    ):



        last_start = (

            block_end_position

            - span

        )



        for start_position in range(

            block_start_position,

            last_start + 1,

        ):



            end_position = (

                start_position + span

            )



            start_index = active_circle_indices[

                start_position

            ]

            end_index = active_circle_indices[

                end_position

            ]



            tangent = compute_safe_circle_tangent_between_indices(

                circle_groups=circle_groups,

                corridor_list=corridor_list,

                start_circle_index=start_index,

                end_circle_index=end_index,

                r=r,

                safe_union_cache=safe_union_cache,

                tol=tol,

            )



            if tangent is not None:

                return (

                    start_position,

                    end_position,

                    tangent,

                )



    return None





def restore_active_circle_block_to_baseline(

    active_circle_indices,

    circle_groups,

    baseline,

    R,

    block_start_position,

    block_end_position,

    tol=1e-9,

):

    """

    Restore every non-baseline circle in an active-circle block to its

    validated baseline center.



    :return:

        (success, changed).

    """

    changed = False



    for position in range(

        block_start_position,

        block_end_position + 1,

    ):



        transition_index = active_circle_indices[

            position

        ]



        circle = get_single_circle(

            circle_groups[transition_index]

        )



        if is_baseline_circle(circle):

            continue



        baseline_circle = compute_baseline_turn_circle(

            baseline=baseline,

            transition_index=transition_index,

            R=R,

            placement_rule="baseline_repair",

            tol=tol,

        )



        if baseline_circle is None:

            return (

                False,

                changed,

            )



        tau = baseline.turn_directions[

            transition_index

        ]



        circle_groups[transition_index] = {

            tau: baseline_circle

        }



        changed = True



    return (

        True,

        changed,

    )





def simplify_tangent_chain(

    circle_groups,

    corridor_list,

    baseline,

    R,

    r,

    active_circle_indices,

    tangents,

    safe_union_cache=None,

    tol=1e-9,

):

    """

    Remove self-intersections from the tangent chain through local circle

    skipping.



    For every detected tangent intersection:



        1. identify the active-circle block involved in the crossing;

        2. try the longest safe direct tangent inside that block, allowing

           one or several intermediate circles to be skipped;

        3. if no safe skip exists, restore all active circles in the block

           to their baseline positions and rebuild the local tangent chain;

        4. repeat until no tangent segments intersect.



    The intermediate circles are not used as an additional "push direction"

    criterion. Once they are skipped, the relevant requirement for the new

    straight connection is directly tested: the ordered tangent must exist and

    its complete finite segment must lie inside the eroded corridor union.



    Note that final circular-arc validation between the retained tangency

    points is a separate step and is not performed here.

    """

    active_circle_indices = list(

        active_circle_indices

    )

    tangents = list(

        tangents

    )



    if safe_union_cache is None:

        safe_union_cache = {}



    while True:



        intersection_pair = (

            find_first_problematic_tangent_intersection(

                tangents=tangents,

                tol=tol,

            )

        )



        if intersection_pair is None:

            return (

                active_circle_indices,

                tangents,

            )



        first_tangent, second_tangent = (

            intersection_pair

        )



        # Tangent i connects active circles i -> i+1. Therefore the

        # crossing between tangent i and tangent j affects circle

        # positions i through j+1.

        block_start_position = first_tangent

        block_end_position = second_tangent + 1



        skip_result = find_longest_safe_skip_in_block(

            active_circle_indices=active_circle_indices,

            circle_groups=circle_groups,

            corridor_list=corridor_list,

            r=r,

            block_start_position=block_start_position,

            block_end_position=block_end_position,

            safe_union_cache=safe_union_cache,

            tol=tol,

        )



        # -----------------------------------------------------------

        # A safe direct tangent exists: remove every active circle

        # strictly between its two endpoint circles.

        # -----------------------------------------------------------

        if skip_result is not None:



            (

                skip_start_position,

                skip_end_position,

                _,

            ) = skip_result



            del active_circle_indices[

                skip_start_position + 1 : skip_end_position

            ]



            tangents = build_and_repair_tangent_chain(

                circle_groups=circle_groups,

                corridor_list=corridor_list,

                baseline=baseline,

                R=R,

                r=r,

                active_circle_indices=active_circle_indices,

                safe_union_cache=safe_union_cache,

                tol=tol,

            )



            if tangents is None:

                return None



            continue



        # -----------------------------------------------------------

        # No safe skip exists. Restore the affected active circles to

        # their baseline positions and rebuild the tangent chain.

        # -----------------------------------------------------------

        success, changed = restore_active_circle_block_to_baseline(

            active_circle_indices=active_circle_indices,

            circle_groups=circle_groups,

            baseline=baseline,

            R=R,

            block_start_position=block_start_position,

            block_end_position=block_end_position,

            tol=tol,

        )



        if not success:

            return None



        # The same intersection persists even though every circle in

        # the affected block is already at its baseline position. At

        # this point the refinement is abandoned in favour of the full

        # validated baseline.

        if not changed:

            return None



        tangents = build_and_repair_tangent_chain(

            circle_groups=circle_groups,

            corridor_list=corridor_list,

            baseline=baseline,

            R=R,

            r=r,

            active_circle_indices=active_circle_indices,

            safe_union_cache=safe_union_cache,

            tol=tol,

        )



        if tangents is None:

            return None





# ===================================================================

# Four standard independent-placement rules

# ===================================================================



def compute_independent_circle_local_position(

    d_x,

    d_y,

    R,

    r,

    tol=1e-9,

):

    """

    Apply the four local intermediate-circle placement rules.



    The cases are evaluated in the following order:



        1. preferred 45-degree placement;

        2. preferred shifted placement;

        3. ordinary 45-degree placement;

        4. ordinary shifted placement.



    :return:

        (x_local, y_local, placement_rule), or None if none

        of the four rules is feasible.

    """

    D = R - r

    S = R + r



    if D < -tol:

        return None



    D = max(

        0.0,

        D,

    )



    q = D / sqrt(2.0)



    # ---------------------------------------------------------------

    # 1. Preferred 45-degree placement

    # ---------------------------------------------------------------

    if (

        d_x >= 2.0 * (S - q) - tol

        and d_y >= 2.0 * (S - q) - tol

    ):

        return (

            q,

            q,

            "preferred_45",

        )



    # ---------------------------------------------------------------

    # 2. Preferred shifted placement

    # ---------------------------------------------------------------

    a_hat = max(

        0.0,

        S - d_x / 2.0,

    )



    b_hat = max(

        0.0,

        S - d_y / 2.0,

    )



    if (

        a_hat**2 + b_hat**2

        <= D**2 + tol

    ):

        return (

            a_hat,

            b_hat,

            "preferred_shifted",

        )



    # ---------------------------------------------------------------

    # 3. Ordinary 45-degree placement

    # ---------------------------------------------------------------

    if (

        d_x >= S - q - tol

        and d_y >= S - q - tol

    ):

        return (

            q,

            q,

            "ordinary_45",

        )



    # ---------------------------------------------------------------

    # 4. Ordinary shifted placement

    # ---------------------------------------------------------------

    a = max(

        0.0,

        S - d_x,

    )



    b = max(

        0.0,

        S - d_y,

    )



    if (

        a**2 + b**2

        <= D**2 + tol

    ):

        return (

            a,

            b,

            "ordinary_shifted",

        )



    return None





# ===================================================================

# Effective local overlap dimensions

# ===================================================================



def compute_effective_junction_dimensions(

    corridor_overlap,

    corner_point,

    local_x_axis,

    local_y_axis,

    tol=1e-9,

):

    """

    Compute the effective local overlap dimensions d_x and d_y measured

    from the selected concave corner.



    The overlap must extend from corner_point along the negative local axes.

    """

    (x_min, x_max), (y_min, y_max) = corridor_overlap



    overlap_corners = [

        np.array([x_min, y_min]),

        np.array([x_min, y_max]),

        np.array([x_max, y_min]),

        np.array([x_max, y_max]),

    ]



    corner_point = np.asarray(

        corner_point,

        dtype=float,

    )



    local_x_axis = np.asarray(

        local_x_axis,

        dtype=float,

    )



    local_y_axis = np.asarray(

        local_y_axis,

        dtype=float,

    )



    local_coordinates = []



    for point in overlap_corners:



        delta = point - corner_point



        x_local = np.dot(

            delta,

            local_x_axis,

        )



        y_local = np.dot(

            delta,

            local_y_axis,

        )



        local_coordinates.append(

            (

                x_local,

                y_local,

            )

        )



    # The selected corner must be the positive local corner

    # of the overlap rectangle.

    if any(

        x_local > tol

        or y_local > tol

        for x_local, y_local

        in local_coordinates

    ):

        return None



    d_x = max(

        -x_local

        for x_local, _

        in local_coordinates

    )



    d_y = max(

        -y_local

        for _, y_local

        in local_coordinates

    )



    return (

        d_x,

        d_y,

    )
