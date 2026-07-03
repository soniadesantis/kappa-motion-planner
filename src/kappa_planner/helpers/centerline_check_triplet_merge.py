def point_to_array(point):
    return np.array([point.x, point.y], dtype=float)


def array_to_point(array):
    return Point(array[0], array[1])


def signed_distance_to_normalized_line(point, line):
    """
    Signed distance from a point to a normalized line a*x + b*y + c = 0.

    :param point: Point to evaluate.
    :type point: Point

    :param line: Normalized line coefficients.
    :type line: tuple[float, float, float]

    :return: Signed distance.
    :rtype: float
    """
    a, b, c = line
    return a * point.x + b * point.y + c


def choose_line_side(center, line, reference_point=None, tol=1e-9):
    """
    Choose on which side of a line the shifted center should remain.

    :param center: Current center.
    :type center: Point

    :param line: Normalized line coefficients.
    :type line: tuple[float, float, float]

    :param reference_point: Optional point used when center lies on the line.
    :type reference_point: Point or None

    :param tol: Numerical tolerance.
    :type tol: float

    :return: +1, -1, or None if ambiguous.
    :rtype: int or None
    """
    signed_distance = signed_distance_to_normalized_line(center, line)

    if abs(signed_distance) > tol:
        return efficient_sign(signed_distance)

    if reference_point is None:
        return None

    reference_signed_distance = signed_distance_to_normalized_line(
        reference_point,
        line,
    )

    if abs(reference_signed_distance) <= tol:
        return None

    return efficient_sign(reference_signed_distance)


def center_contains_triplet_corner_disks(
    center,
    circle1,
    circle2,
    circle3,
    D,
    tol=1e-9,
):
    """
    Check whether a center contains the three footprint disks.

    This means distance(center, corner_point_i) <= D = R - r.

    :param center: Candidate merged center.
    :type center: Point

    :param circle1: First source circle.
    :type circle1: IntermediateCircle

    :param circle2: Second source circle.
    :type circle2: IntermediateCircle

    :param circle3: Third source circle.
    :type circle3: IntermediateCircle

    :param D: Admissible radius R - r.
    :type D: float

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Feasibility flag and diagnostic dictionary.
    :rtype: tuple[bool, dict]
    """
    d1 = compute_distance_two_points(center, circle1.corner_point)
    d2 = compute_distance_two_points(center, circle2.corner_point)
    d3 = compute_distance_two_points(center, circle3.corner_point)

    feasible = (
        d1 <= D + tol
        and d2 <= D + tol
        and d3 <= D + tol
    )

    return feasible, {
        "distance_to_first_corner": d1,
        "distance_to_second_corner": d2,
        "distance_to_third_corner": d3,
        "max_distance_to_corner": max(d1, d2, d3),
    }


def center_clears_outer_centerlines(
    center,
    first_line,
    fourth_line,
    S,
    tol=1e-9,
):
    """
    Check whether a swept circle of radius S clears both outer centerlines.

    :param center: Candidate merged center.
    :type center: Point

    :param first_line: First outer centerline.
    :type first_line: tuple[float, float, float]

    :param fourth_line: Fourth outer centerline.
    :type fourth_line: tuple[float, float, float]

    :param S: Swept radius R + r.
    :type S: float

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Feasibility flag and diagnostic dictionary.
    :rtype: tuple[bool, dict]
    """
    d_first = signed_distance_to_normalized_line(center, first_line)
    d_fourth = signed_distance_to_normalized_line(center, fourth_line)

    clears_first = abs(d_first) >= S - tol
    clears_fourth = abs(d_fourth) >= S - tol

    return clears_first and clears_fourth, {
        "signed_distance_first": d_first,
        "signed_distance_fourth": d_fourth,
        "clears_first": clears_first,
        "clears_fourth": clears_fourth,
    }


def generate_triplet_centerline_shift_candidates(
    nominal_center,
    first_line,
    fourth_line,
    S,
    circle1,
    circle3,
    tol=1e-9,
):
    """
    Generate candidate centers obtained by shifting the nominal center
    to be tangent to the first and/or fourth outer centerline.

    Candidate order:
        1. nominal center;
        2. tangent to first centerline;
        3. tangent to fourth centerline;
        4. tangent to both centerlines.

    :param nominal_center: Nominal merged circle center.
    :type nominal_center: Point

    :param first_line: First outer centerline.
    :type first_line: tuple[float, float, float]

    :param fourth_line: Fourth outer centerline.
    :type fourth_line: tuple[float, float, float]

    :param S: Swept radius R + r.
    :type S: float

    :param circle1: First source circle, used as side reference for first line.
    :type circle1: IntermediateCircle

    :param circle3: Third source circle, used as side reference for fourth line.
    :type circle3: IntermediateCircle

    :param tol: Numerical tolerance.
    :type tol: float

    :return: List of candidate dictionaries.
    :rtype: list[dict]
    """
    candidates = []

    center_np = point_to_array(nominal_center)

    first_side = choose_line_side(
        center=nominal_center,
        line=first_line,
        reference_point=circle1.center,
        tol=tol,
    )

    fourth_side = choose_line_side(
        center=nominal_center,
        line=fourth_line,
        reference_point=circle3.center,
        tol=tol,
    )

    d_first = signed_distance_to_normalized_line(nominal_center, first_line)
    d_fourth = signed_distance_to_normalized_line(nominal_center, fourth_line)

    first_crossed = abs(d_first) < S - tol
    fourth_crossed = abs(d_fourth) < S - tol

    candidates.append(
        {
            "center": nominal_center,
            "rule": "nominal",
            "shift": np.zeros(2),
            "first_crossed_before": first_crossed,
            "fourth_crossed_before": fourth_crossed,
        }
    )

    if first_crossed and first_side is not None:
        a1, b1, _ = first_line
        target_first = first_side * S
        shift_amount_first = target_first - d_first

        shifted_np = center_np + shift_amount_first * np.array([a1, b1])

        candidates.append(
            {
                "center": array_to_point(shifted_np),
                "rule": "tangent_to_first_centerline",
                "shift": shifted_np - center_np,
                "target_first": target_first,
            }
        )

    if fourth_crossed and fourth_side is not None:
        a4, b4, _ = fourth_line
        target_fourth = fourth_side * S
        shift_amount_fourth = target_fourth - d_fourth

        shifted_np = center_np + shift_amount_fourth * np.array([a4, b4])

        candidates.append(
            {
                "center": array_to_point(shifted_np),
                "rule": "tangent_to_fourth_centerline",
                "shift": shifted_np - center_np,
                "target_fourth": target_fourth,
            }
        )

    if (
        first_crossed
        and fourth_crossed
        and first_side is not None
        and fourth_side is not None
    ):
        a1, b1, c1 = first_line
        a4, b4, c4 = fourth_line

        target_first = first_side * S
        target_fourth = fourth_side * S

        A = np.array(
            [
                [a1, b1],
                [a4, b4],
            ],
            dtype=float,
        )

        rhs = np.array(
            [
                target_first - c1,
                target_fourth - c4,
            ],
            dtype=float,
        )

        determinant = np.linalg.det(A)

        if abs(determinant) > tol:
            shifted_np = np.linalg.solve(A, rhs)

            candidates.append(
                {
                    "center": array_to_point(shifted_np),
                    "rule": "tangent_to_both_centerlines",
                    "shift": shifted_np - center_np,
                    "target_first": target_first,
                    "target_fourth": target_fourth,
                }
            )

    return candidates


def select_valid_triplet_merged_center(
    nominal_center,
    circle1,
    circle2,
    circle3,
    first_line,
    fourth_line,
    R,
    r,
    tol=1e-9,
):
    """
    Select a valid triplet merged center.

    The selected center must:
        - contain the three footprint disks;
        - clear the two outer centerlines.

    Candidate order:
        1. nominal center;
        2. tangent to first centerline;
        3. tangent to fourth centerline;
        4. tangent to both centerlines.

    :param nominal_center: Nominal merged center.
    :type nominal_center: Point

    :param circle1: First source circle.
    :type circle1: IntermediateCircle

    :param circle2: Second source circle.
    :type circle2: IntermediateCircle

    :param circle3: Third source circle.
    :type circle3: IntermediateCircle

    :param first_line: First outer centerline.
    :type first_line: tuple[float, float, float]

    :param fourth_line: Fourth outer centerline.
    :type fourth_line: tuple[float, float, float]

    :param R: Merged circle radius.
    :type R: float

    :param r: Footprint radius.
    :type r: float

    :param tol: Numerical tolerance.
    :type tol: float

    :return: Dictionary with selected center or failure information.
    :rtype: dict
    """
    S = R + r
    D = R - r

    candidates = generate_triplet_centerline_shift_candidates(
        nominal_center=nominal_center,
        first_line=first_line,
        fourth_line=fourth_line,
        S=S,
        circle1=circle1,
        circle3=circle3,
        tol=tol,
    )

    rejected_candidates = []

    for candidate in candidates:
        center = candidate["center"]

        contains_corners, containment_status = center_contains_triplet_corner_disks(
            center=center,
            circle1=circle1,
            circle2=circle2,
            circle3=circle3,
            D=D,
            tol=tol,
        )

        if not contains_corners:
            candidate["rejection_reason"] = "does_not_contain_triplet_corner_disks"
            candidate["containment_status"] = containment_status
            rejected_candidates.append(candidate)
            continue

        clears_centerlines, centerline_status = center_clears_outer_centerlines(
            center=center,
            first_line=first_line,
            fourth_line=fourth_line,
            S=S,
            tol=tol,
        )

        if not clears_centerlines:
            candidate["rejection_reason"] = "does_not_clear_outer_centerlines"
            candidate["containment_status"] = containment_status
            candidate["centerline_status"] = centerline_status
            rejected_candidates.append(candidate)
            continue

        return {
            "feasible": True,
            "reason": "ok",
            "center": center,
            "rule": candidate["rule"],
            "selected_candidate": candidate,
            "containment_status": containment_status,
            "centerline_status": centerline_status,
            "rejected_candidates": rejected_candidates,
        }

    return {
        "feasible": False,
        "reason": "no_valid_triplet_shift_candidate",
        "center": None,
        "rejected_candidates": rejected_candidates,
    }