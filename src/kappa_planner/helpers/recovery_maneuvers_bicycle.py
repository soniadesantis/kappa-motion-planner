"""
Recovery maneuvers for a bicycle vehicle inside a rectangular corridor.
"""

from math import atan2, cos, pi, sin

from .collision_avoidance import allowed_maneuvers
from ..trajectory import CurvilinearArcUnicycle, BackwardArc, LinearSegmentUnicycle


def _compute_alignment_state(
    start_pose,
    corridor,
    bicycle,
    preferred_direction,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Compute an alignment maneuver sequence and extract the resulting
    pose and final time.

    Returns
    -------
    tuple or None
        On success:

        (
            alignment_maneuvers,
            aligned_pose,
            aligned_time,
        )

        ``alignment_maneuvers`` may be an empty list when the vehicle is
        already aligned.

        ``None`` is returned when no feasible alignment sequence is found.
    """
    alignment_maneuvers = compute_corridor_alignment_maneuvers(
        start_pose=start_pose,
        corridor=corridor,
        bicycle=bicycle,
        preferred_direction=preferred_direction,
        tol=tol,
        t0=t0,
        samples_number=samples_number,
    )

    if alignment_maneuvers is None:
        return None

    if alignment_maneuvers:
        aligned_pose = alignment_maneuvers[-1].end_pose
        aligned_time = alignment_maneuvers[-1].tf
    else:
        # The vehicle was already aligned.
        aligned_pose = start_pose
        aligned_time = t0

    return (
        alignment_maneuvers,
        aligned_pose,
        aligned_time,
    )


def _wrap_to_pi(angle):
    """
    Wrap an angle to [-pi, pi).
    """
    return atan2(
        sin(angle),
        cos(angle),
    )


def _instantiate_arc(
    start_pose,
    amplitude,
    turn_direction,
    is_backward,
    bicycle,
    t0=0.0,
    samples_number=10,
    tol=1e-9,
):
    """
    Instantiate a forward or backward minimum-radius circular arc.

    Parameters
    ----------
    start_pose:
        Initial pose [x, y, theta].

    amplitude:
        Positive angular amplitude in radians.

    turn_direction:
        +1 for left steering.
        -1 for right steering.

    is_backward:
        True for a backward arc, False for a forward arc.

    Returns
    -------
    CurvilinearArcUnicycle or BackwardArc or None
        None is returned when the requested amplitude is effectively zero.
    """
    if amplitude < -tol:
        raise ValueError("The arc amplitude must be nonnegative")

    if amplitude <= tol:
        return None

    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be either -1 or +1")

    x0, y0, theta0 = start_pose

    radius = float(bicycle.max_radius)
    speed = abs(float(bicycle.v_max))

    if radius <= 0.0:
        raise ValueError("bicycle.max_radius must be positive")

    if speed <= 0.0:
        raise ValueError("bicycle.v_max must be positive")

    # Supporting-circle center.
    xc = x0 + radius * cos(
        theta0 + turn_direction * pi / 2.0
    )

    yc = y0 + radius * sin(
        theta0 + turn_direction * pi / 2.0
    )

    start_circle_angle = atan2(
        y0 - yc,
        x0 - xc,
    )

    if is_backward:
        # Backward motion follows the circle in the direction opposite
        # to the steering direction.
        final_circle_angle = (
            start_circle_angle
            - turn_direction * amplitude
        )

        thetaf = (
            theta0
            - turn_direction * amplitude
        )

        v = -speed
        omega = turn_direction * v / radius

    else:
        final_circle_angle = (
            start_circle_angle
            + turn_direction * amplitude
        )

        thetaf = (
            theta0
            + turn_direction * amplitude
        )

        v = speed
        omega = turn_direction * v / radius

    xf = xc + radius * cos(final_circle_angle)
    yf = yc + radius * sin(final_circle_angle)

    if is_backward:
        return BackwardArc(
            xc=xc,
            yc=yc,
            x0=x0,
            y0=y0,
            theta0=theta0,
            xf=xf,
            yf=yf,
            thetaf=thetaf,
            radius=radius,
            turn_direction=turn_direction,
            v=v,
            omega=omega,
            bicycle=bicycle,
            t0=t0,
            samples_number=samples_number,
        )

    return CurvilinearArcUnicycle(
        xc=xc,
        yc=yc,
        x0=x0,
        y0=y0,
        theta0=theta0,
        xf=xf,
        yf=yf,
        thetaf=thetaf,
        radius=radius,
        turn_direction=turn_direction,
        v=v,
        omega=omega,
        unicycle=bicycle,
        t0=t0,
        samples_number=samples_number,
    )


def _alignment_candidates(signed_angle):
    """
    Return the two arc types that change the heading in the required
    direction.

    Each candidate is represented by:

        (
            allowed-maneuver key,
            turn direction,
            is backward,
        )

    Forward motion is listed first for the one-arc case.
    """
    if signed_angle > 0.0:
        # Heading must increase:
        # forward-left or backward-right.
        return (
            ("forward_left", 1, False),
            ("backward_right", -1, True),
        )

    # Heading must decrease:
    # forward-right or backward-left.
    return (
        ("forward_right", -1, False),
        ("backward_left", 1, True),
    )


def _compute_single_alignment_arc(
    start_pose,
    signed_angle,
    corridor,
    bicycle,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Try to align the vehicle with one circular maneuver.
    """
    required_amplitude = abs(signed_angle)

    allowed_amplitudes = allowed_maneuvers(
        start_pose=start_pose,
        corridor=corridor,
        vehicle=bicycle,
        tol=tol,
    )["amplitudes"]

    for (
        maneuver_key,
        turn_direction,
        is_backward,
    ) in _alignment_candidates(signed_angle):

        if (
            required_amplitude
            <= allowed_amplitudes[maneuver_key] + tol
        ):
            return _instantiate_arc(
                start_pose=start_pose,
                amplitude=required_amplitude,
                turn_direction=turn_direction,
                is_backward=is_backward,
                bicycle=bicycle,
                t0=t0,
                samples_number=samples_number,
                tol=tol,
            )

    return None


def _compute_two_arc_alignment(
    start_pose,
    signed_angle,
    corridor,
    bicycle,
    split_fractions,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Try to align the vehicle using two arcs.

    Both arcs change the vehicle heading in the same direction, and their
    amplitudes satisfy

        amplitude_1 + amplitude_2 = abs(signed_angle).

    A backward-forward sequence is tested first so that, when possible,
    the vehicle finishes the recovery moving forward.
    """
    total_amplitude = abs(signed_angle)

    if signed_angle > 0.0:
        # Both maneuvers increase the heading.
        #
        # Preferred:
        #   backward-right -> forward-left
        #
        # Alternative:
        #   forward-left -> backward-right
        sequences = (
            (
                ("backward_right", -1, True),
                ("forward_left", 1, False),
            ),
            (
                ("forward_left", 1, False),
                ("backward_right", -1, True),
            ),
        )

    else:
        # Both maneuvers decrease the heading.
        #
        # Preferred:
        #   backward-left -> forward-right
        #
        # Alternative:
        #   forward-right -> backward-left
        sequences = (
            (
                ("backward_left", 1, True),
                ("forward_right", -1, False),
            ),
            (
                ("forward_right", -1, False),
                ("backward_left", 1, True),
            ),
        )

    initial_allowed = allowed_maneuvers(
        start_pose=start_pose,
        corridor=corridor,
        vehicle=bicycle,
        tol=tol,
    )["amplitudes"]

    for first_data, second_data in sequences:
        (
            first_key,
            first_turn_direction,
            first_is_backward,
        ) = first_data

        (
            second_key,
            second_turn_direction,
            second_is_backward,
        ) = second_data

        maximum_first_amplitude = initial_allowed[first_key]

        for fraction in split_fractions:
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    "Every split fraction must lie strictly between "
                    "zero and one"
                )

            first_amplitude = (
                fraction * total_amplitude
            )

            second_amplitude = (
                total_amplitude - first_amplitude
            )

            # The first maneuver must remain inside the corridor.
            if (
                first_amplitude
                > maximum_first_amplitude + tol
            ):
                continue

            first_arc = _instantiate_arc(
                start_pose=start_pose,
                amplitude=first_amplitude,
                turn_direction=first_turn_direction,
                is_backward=first_is_backward,
                bicycle=bicycle,
                t0=t0,
                samples_number=samples_number,
                tol=tol,
            )

            if first_arc is None:
                continue

            cusp_pose = first_arc.end_pose

            # Recompute the admissible amplitudes from the cusp pose.
            cusp_allowed = allowed_maneuvers(
                start_pose=cusp_pose,
                corridor=corridor,
                vehicle=bicycle,
                tol=tol,
            )["amplitudes"]

            if (
                second_amplitude
                > cusp_allowed[second_key] + tol
            ):
                continue

            second_arc = _instantiate_arc(
                start_pose=cusp_pose,
                amplitude=second_amplitude,
                turn_direction=second_turn_direction,
                is_backward=second_is_backward,
                bicycle=bicycle,
                t0=first_arc.tf,
                samples_number=samples_number,
                tol=tol,
            )

            if second_arc is None:
                continue

            return [
                first_arc,
                second_arc,
            ]

    return None


def compute_corridor_alignment_maneuvers(
    start_pose,
    corridor,
    bicycle,
    preferred_direction=1,
    split_fractions=(
        0.5,
        0.25,
        0.75,
        0.125,
        0.875,
    ),
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Compute zero, one, or two circular maneuvers that align the vehicle
    with a selected longitudinal direction of the corridor.

    The function first attempts a single circular maneuver. If no single
    maneuver is feasible, it attempts a two-arc sequence by splitting the
    required heading displacement between two compatible arcs.

    Parameters
    ----------
    start_pose:
        Initial vehicle pose [x, y, theta].

    corridor:
        Corridor in which the alignment maneuver must remain.

    bicycle:
        Bicycle vehicle model.

    preferred_direction:
        Desired final orientation along the corridor axis:

            +1:
                align with ``corridor.tilt``;

            -1:
                align with ``corridor.tilt + pi``.

        Therefore, after alignment, forward motion follows the selected
        corridor direction.

    split_fractions:
        Fractions of the required angular displacement assigned to the
        first arc during the two-arc search.

        The default order first attempts an equal split and then
        progressively asymmetric splits.

    tol:
        Numerical tolerance.

    t0:
        Initial trajectory time.

    samples_number:
        Number of samples for each returned arc.

    Returns
    -------
    list
        Empty list when the vehicle is already aligned.

        A list containing one arc when a single maneuver is feasible.

        A list containing two arcs when a two-maneuver alignment is
        feasible.

    None
        Returned when alignment cannot be achieved with at most two arcs
        and the supplied split fractions.

    Notes
    -----
    A return value of ``None`` does not prove that no alignment sequence
    exists. It only means that no feasible sequence was found within the
    one- and two-arc families tested here.
    """
    if preferred_direction not in (-1, 1):
        raise ValueError(
            "preferred_direction must be either -1 or +1"
        )

    if preferred_direction == 1:
        target_heading = corridor.tilt
    else:
        target_heading = corridor.tilt + pi

    signed_angle = _wrap_to_pi(
        target_heading - start_pose[2]
    )

    if abs(signed_angle) <= tol:
        return []

    # ---------------------------------------------------------------
    # First attempt: complete the alignment with one arc.
    # ---------------------------------------------------------------
    single_arc = _compute_single_alignment_arc(
        start_pose=start_pose,
        signed_angle=signed_angle,
        corridor=corridor,
        bicycle=bicycle,
        tol=tol,
        t0=t0,
        samples_number=samples_number,
    )

    if single_arc is not None:
        return [single_arc]

    # ---------------------------------------------------------------
    # Second attempt: split the required heading change over two arcs.
    # ---------------------------------------------------------------
    return _compute_two_arc_alignment(
        start_pose=start_pose,
        signed_angle=signed_angle,
        corridor=corridor,
        bicycle=bicycle,
        split_fractions=split_fractions,
        tol=tol,
        t0=t0,
        samples_number=samples_number,
    )


def compute_corridor_alignment_arc(
    start_pose,
    corridor,
    bicycle,
    preferred_direction=1,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Backward-compatible helper that returns a single alignment arc.

    Unlike :func:`compute_corridor_alignment_maneuvers`, this function
    does not attempt a two-arc sequence.

    Returns
    -------
    CurvilinearArcUnicycle or BackwardArc or None
        None is returned when the vehicle is already aligned or when no
        feasible single alignment arc exists.
    """
    if preferred_direction not in (-1, 1):
        raise ValueError(
            "preferred_direction must be either -1 or +1"
        )

    target_heading = (
        corridor.tilt
        if preferred_direction == 1
        else corridor.tilt + pi
    )

    signed_angle = _wrap_to_pi(
        target_heading - start_pose[2]
    )

    if abs(signed_angle) <= tol:
        return None

    return _compute_single_alignment_arc(
        start_pose=start_pose,
        signed_angle=signed_angle,
        corridor=corridor,
        bicycle=bicycle,
        tol=tol,
        t0=t0,
        samples_number=samples_number,
    )


from math import cos, pi, sin

import numpy as np


def _ray_interval_inside_corridor(
    start_position,
    ray_direction,
    corridor,
    tol=1e-9,
):
    """
    Compute the interval of nonnegative ray parameters for which

        p(s) = start_position + s * ray_direction

    lies inside a rectangular corridor.

    Returns
    -------
    tuple[float, float] or None
        ``(s_enter, s_exit)`` for the portion of the ray inside the
        corridor.

        ``None`` when the ray does not intersect the corridor for s >= 0.
    """
    position = np.asarray(
        start_position,
        dtype=float,
    )

    direction = np.asarray(
        ray_direction,
        dtype=float,
    )

    corridor_center = np.asarray(
        corridor.center,
        dtype=float,
    )

    longitudinal_axis = np.array(
        [
            cos(corridor.tilt),
            sin(corridor.tilt),
        ],
        dtype=float,
    )

    lateral_axis = np.array(
        [
            -sin(corridor.tilt),
            cos(corridor.tilt),
        ],
        dtype=float,
    )

    relative_position = position - corridor_center

    local_position = np.array(
        [
            np.dot(relative_position, lateral_axis),
            np.dot(relative_position, longitudinal_axis),
        ],
        dtype=float,
    )

    local_direction = np.array(
        [
            np.dot(direction, lateral_axis),
            np.dot(direction, longitudinal_axis),
        ],
        dtype=float,
    )

    half_extents = np.array(
        [
            corridor.width / 2.0,
            corridor.height / 2.0,
        ],
        dtype=float,
    )

    s_enter = 0.0
    s_exit = float("inf")

    for coordinate in range(2):
        p = local_position[coordinate]
        d = local_direction[coordinate]
        bound = half_extents[coordinate]

        if abs(d) <= tol:
            # The ray is parallel to these two rectangle boundaries.
            if p < -bound - tol or p > bound + tol:
                return None

            continue

        s_1 = (-bound - p) / d
        s_2 = (bound - p) / d

        axis_enter = min(s_1, s_2)
        axis_exit = max(s_1, s_2)

        s_enter = max(
            s_enter,
            axis_enter,
        )

        s_exit = min(
            s_exit,
            axis_exit,
        )

        if s_enter > s_exit + tol:
            return None

    if s_exit < -tol:
        return None

    return (
        max(0.0, s_enter),
        s_exit,
    )


def _distance_to_corridor_front_edge(
    start_position,
    corridor,
):
    """
    Return the signed distance from a position to the front edge of a
    corridor, measured along the corridor's positive longitudinal axis.
    """
    longitudinal_axis = np.array(
        [
            cos(corridor.tilt),
            sin(corridor.tilt),
        ],
        dtype=float,
    )

    relative_position = (
        np.asarray(start_position, dtype=float)
        - np.asarray(corridor.center, dtype=float)
    )

    longitudinal_coordinate = np.dot(
        relative_position,
        longitudinal_axis,
    )

    return (
        corridor.height / 2.0
        - longitudinal_coordinate
    )


def compute_segment_to_next_corridor(
    start_pose,
    first_corridor,
    second_corridor,
    bicycle,
    distance_inside_second=0.0,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Build a straight segment toward the front edge of the first corridor,
    stopping when the vehicle enters the second shrunken corridor or a
    specified distance after entry.

    The vehicle must already be aligned with the longitudinal axis of the
    first corridor. It may face either longitudinal direction:

        - if it faces the front, the segment is travelled forward;
        - if it faces the back, the segment is travelled backward.

    The segment is accepted only if its position path remains inside the
    union of the two shrunken corridors.

    Parameters
    ----------
    start_pose:
        Initial aligned pose [x, y, theta].

    first_corridor:
        Corridor containing the initial pose.

    second_corridor:
        Next corridor to be entered.

    bicycle:
        Bicycle vehicle model.

    distance_inside_second:
        Distance travelled after first entering the second shrunken
        corridor.

        A value of zero stops at the entry point.

    Returns
    -------
    LinearSegmentUnicycle or None
        Straight segment when the transition is feasible.

        ``None`` when:

        - the pose is not aligned with the first corridor;
        - the forward ray does not intersect the second corridor;
        - the ray exits the first corridor before entering the second;
        - the requested additional distance leaves the second corridor.
    """
    if distance_inside_second < -tol:
        raise ValueError(
            "distance_inside_second must be nonnegative"
        )

    margin = bicycle.width / 2.0

    first_shrunken = first_corridor.shrink(margin)
    second_shrunken = second_corridor.shrink(margin)

    x0, y0, theta0 = start_pose

    front_heading = first_corridor.tilt

    heading_error_front = _wrap_to_pi(
        theta0 - front_heading
    )

    heading_error_back = _wrap_to_pi(
        theta0 - (front_heading + pi)
    )

    # The position must always move toward the front edge.
    ray_direction = np.array(
        [
            cos(front_heading),
            sin(front_heading),
        ],
        dtype=float,
    )

    if abs(heading_error_front) <= tol:
        # Vehicle points toward the front.
        velocity = abs(float(bicycle.v_max))

    elif abs(heading_error_back) <= tol:
        # Vehicle points toward the back, so backward motion moves its
        # position toward the front.
        velocity = -abs(float(bicycle.v_max))

    else:
        return None

    start_position = np.array(
        [x0, y0],
        dtype=float,
    )

    distance_to_front = _distance_to_corridor_front_edge(
        start_position=start_position,
        corridor=first_shrunken,
    )

    if distance_to_front < -tol:
        # The start is already beyond the front edge of corridor 1.
        return None

    second_interval = _ray_interval_inside_corridor(
        start_position=start_position,
        ray_direction=ray_direction,
        corridor=second_shrunken,
        tol=tol,
    )

    if second_interval is None:
        return None

    (
        distance_to_second_entry,
        distance_to_second_exit,
    ) = second_interval

    # There must be no uncovered gap between leaving corridor 1 and
    # entering corridor 2.
    if (
        distance_to_second_entry
        > distance_to_front + tol
    ):
        return None

    target_distance = (
        distance_to_second_entry
        + max(0.0, distance_inside_second)
    )

    # The requested endpoint must remain inside corridor 2.
    if (
        target_distance
        > distance_to_second_exit + tol
    ):
        return None

    if target_distance <= tol:
        # The start position is already at the requested depth.
        return None

    xf = (
        x0
        + target_distance * ray_direction[0]
    )

    yf = (
        y0
        + target_distance * ray_direction[1]
    )

    return LinearSegmentUnicycle(
        x0=x0,
        y0=y0,
        xf=xf,
        yf=yf,
        theta=theta0,
        v=velocity,
        unicycle=bicycle,
        t0=t0,
        samples_number=samples_number,
    )


def compute_segment_deeper_into_corridor(
    start_pose,
    corridor,
    bicycle,
    distance,
    tol=1e-9,
    t0=0.0,
    samples_number=10,
):
    """
    Build a straight segment that moves the vehicle deeper into a
    rectangular corridor, toward its back edge.

    The vehicle must already be aligned with the longitudinal axis of the
    corridor. It may face either longitudinal direction:

        - if it faces the back edge, the segment is travelled forward;
        - if it faces the front edge, the segment is travelled backward.

    The corridor is shrunk by half the vehicle width before checking the
    available longitudinal distance.

    Parameters
    ----------
    start_pose:
        Initial aligned pose [x, y, theta].

    corridor:
        Corridor in which the complete segment must remain.

    bicycle:
        Bicycle vehicle model.

    distance:
        Requested nonnegative displacement toward the back edge.

    tol:
        Numerical tolerance.

    t0:
        Initial trajectory time.

    samples_number:
        Number of trajectory samples.

    Returns
    -------
    LinearSegmentUnicycle or None
        The requested segment when feasible.

        ``None`` when:

        - the pose is not aligned with the corridor axis;
        - the start position is outside the shrunken corridor;
        - the requested distance exceeds the available distance;
        - the requested distance is effectively zero.
    """
    if distance < -tol:
        raise ValueError(
            "distance must be nonnegative"
        )

    if distance <= tol:
        return None

    effective_corridor = corridor.shrink(
        bicycle.width / 2.0
    )

    x0, y0, theta0 = start_pose

    longitudinal_axis = np.array(
        [
            cos(corridor.tilt),
            sin(corridor.tilt),
        ],
        dtype=float,
    )

    lateral_axis = np.array(
        [
            -sin(corridor.tilt),
            cos(corridor.tilt),
        ],
        dtype=float,
    )

    corridor_center = np.asarray(
        effective_corridor.center,
        dtype=float,
    )

    start_position = np.array(
        [x0, y0],
        dtype=float,
    )

    relative_position = (
        start_position - corridor_center
    )

    longitudinal_coordinate = np.dot(
        relative_position,
        longitudinal_axis,
    )

    lateral_coordinate = np.dot(
        relative_position,
        lateral_axis,
    )

    half_height = (
        effective_corridor.height / 2.0
    )

    half_width = (
        effective_corridor.width / 2.0
    )

    # Verify that the initial position lies inside the effective corridor.
    if (
        abs(lateral_coordinate) > half_width + tol
        or longitudinal_coordinate < -half_height - tol
        or longitudinal_coordinate > half_height + tol
    ):
        return None

    # The back edge has local longitudinal coordinate -half_height.
    available_distance = (
        longitudinal_coordinate + half_height
    )

    if distance > available_distance + tol:
        return None

    front_heading = corridor.tilt
    back_heading = corridor.tilt + pi

    heading_error_front = _wrap_to_pi(
        theta0 - front_heading
    )

    heading_error_back = _wrap_to_pi(
        theta0 - back_heading
    )

    if abs(heading_error_back) <= tol:
        # The vehicle already faces the back edge.
        velocity = abs(float(bicycle.v_max))

    elif abs(heading_error_front) <= tol:
        # The vehicle faces the front edge, so backward motion moves
        # toward the back edge.
        velocity = float(bicycle.v_min)

        if velocity >= 0.0:
            velocity = -abs(float(bicycle.v_max))

    else:
        # The vehicle is not aligned with the corridor.
        return None

    # Geometric motion is always toward the back edge.
    displacement = (
        -distance * longitudinal_axis
    )

    xf = x0 + displacement[0]
    yf = y0 + displacement[1]

    return LinearSegmentUnicycle(
        x0=x0,
        y0=y0,
        xf=xf,
        yf=yf,
        theta=theta0,
        v=velocity,
        unicycle=bicycle,
        t0=t0,
        samples_number=samples_number,
    )