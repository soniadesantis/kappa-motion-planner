import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, pi

from .plot_helpers import plot_corridors


def compute_distance_between_parallel_centerlines(corridor1, corridor2, tol=1e-9):
    """
    Compute the perpendicular distance between the centerlines of two
    parallel corridors.

    The centerline of each corridor is defined by:
        center + lambda * unit_vector
    """

    c1 = np.asarray(corridor1.center[:2], dtype=float)
    c2 = np.asarray(corridor2.center[:2], dtype=float)

    v1 = np.asarray(corridor1.unit_vector, dtype=float)
    v2 = np.asarray(corridor2.unit_vector, dtype=float)

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < tol or v2_norm < tol:
        raise ValueError("Cannot compute centerline distance with zero unit vector.")

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Check that the two centerlines are parallel.
    if abs(abs(np.dot(v1, v2)) - 1.0) > tol:
        raise ValueError("Centerlines are not parallel.")

    # Normal to the first centerline.
    n1 = np.array([-v1[1], v1[0]])

    # Perpendicular distance between the two infinite parallel lines.
    distance = abs(np.dot(c2 - c1, n1))

    return distance


def get_corridor_centerline_segment(corridor):
    """
    Return the two endpoints of the corridor centerline segment.

    This uses corridor.tail and corridor.head.
    """

    P = np.asarray(corridor.tail[:2], dtype=float)
    Q = np.asarray(corridor.head[:2], dtype=float)

    return P, Q


def plot_invalid_centerline_check(
    corridor_list,
    centerline_specs,
    triple_indices,
    case_id,
    required_distance,
    title="Invalid corridor sequence",
    plot_function=None,
):
    """
    Plot the full corridor sequence and overlay invalid centerlines.

    Parameters
    ----------
    corridor_list : list
        Full corridor sequence.

    centerline_specs : list of dict
        Each dict should contain:
            {
                "corridor": corridor_object,
                "label": str,
                "distance": float or None,
                "linestyle": str,
            }

    triple_indices : tuple
        The checked triple indices, e.g. (i, i+1, i+2).

    case_id : str
        Axis case identifier, e.g. HVH, HHV, HHH.

    required_distance : float
        Minimum required centerline distance.

    plot_function : callable or None
        Function used to plot corridors. If None, assumes plot_corridors
        is available in the current namespace.
    """

    if plot_function is None:
        figure = plot_corridors(corridor_list)
    else:
        figure = plot_function(corridor_list)

    if figure is not None and hasattr(figure, "gca"):
        ax = figure.gca()
    else:
        ax = plt.gca()

    for spec in centerline_specs:
        corridor = spec["corridor"]
        label = spec.get("label", "centerline")
        linestyle = spec.get("linestyle", "-")

        P, Q = get_corridor_centerline_segment(corridor)

        ax.plot(
            [P[0], Q[0]],
            [P[1], Q[1]],
            linewidth=3.0,
            linestyle=linestyle,
            label=label,
        )

        center = np.asarray(corridor.center[:2], dtype=float)
        ax.plot(center[0], center[1], "o")

    distance_text_parts = []

    for spec in centerline_specs:
        if spec.get("distance", None) is not None:
            distance_text_parts.append(
                f"{spec['label']}: {spec['distance']:.3f}"
            )

    distance_text = " | ".join(distance_text_parts)

    ax.set_title(
        f"{title}: triple {triple_indices}, case {case_id}\n"
        f"{distance_text} | required: {required_distance:.3f}"
    )

    ax.axis("equal")
    ax.legend()
    plt.show(block=True)


def validate_corridor_sequence(
    corridor_list,
    vehicle,
    min_centerline_distance=None,
    tol=1e-6,
    plot_invalid=False,
    plot_function=None,
):
    """
    Validate the corridor sequence using local three-corridor checks.

    Corridors are classified only by axis:
        H = horizontal, tilt 0 or pi
        V = vertical, tilt pi/2 or 3pi/2

    For each triple (corridor_i, corridor_{i+1}, corridor_{i+2}),
    the function checks whether the relevant parallel centerlines are
    separated by at least min_centerline_distance.

    By default:
        min_centerline_distance = 2 * vehicle.max_radius

    If plot_invalid=True, the full corridor sequence is plotted and the
    invalid centerlines are highlighted before raising ValueError.
    """

    if min_centerline_distance is None:
        min_centerline_distance = 2.0 * vehicle.max_radius

    for i in range(len(corridor_list) - 2):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]
        corridor3 = corridor_list[i + 2]

        case_id = ""

        # ------------------------------------------------------------
        # Classify corridor1
        # ------------------------------------------------------------
        c = abs(cos(corridor1.tilt))
        s = abs(sin(corridor1.tilt))

        if c >= 1.0 - tol:
            case_id += "H"
        elif s >= 1.0 - tol:
            case_id += "V"
        else:
            raise ValueError(
                f"Corridor {i} is not axis-aligned. "
                f"tilt = {corridor1.tilt}"
            )

        # ------------------------------------------------------------
        # Classify corridor2
        # ------------------------------------------------------------
        c = abs(cos(corridor2.tilt))
        s = abs(sin(corridor2.tilt))

        if c >= 1.0 - tol:
            case_id += "H"
        elif s >= 1.0 - tol:
            case_id += "V"
        else:
            raise ValueError(
                f"Corridor {i + 1} is not axis-aligned. "
                f"tilt = {corridor2.tilt}"
            )

        # ------------------------------------------------------------
        # Classify corridor3
        # ------------------------------------------------------------
        c = abs(cos(corridor3.tilt))
        s = abs(sin(corridor3.tilt))

        if c >= 1.0 - tol:
            case_id += "H"
        elif s >= 1.0 - tol:
            case_id += "V"
        else:
            raise ValueError(
                f"Corridor {i + 2} is not axis-aligned. "
                f"tilt = {corridor3.tilt}"
            )

        # ------------------------------------------------------------
        # Case 1: alternating axes
        # HVH or VHV
        # Directly compare corridor1 and corridor3.
        # ------------------------------------------------------------
        if case_id in {"HVH", "VHV"}:
            centerline_distance = compute_distance_between_parallel_centerlines(
                corridor1,
                corridor3,
                tol=tol,
            )

            print(
                f"Triple {i}, {i+1}, {i+2}: case {case_id}, "
                f"distance = {centerline_distance:.3f}, "
                f"required = {min_centerline_distance:.3f}"
            )

            if centerline_distance < min_centerline_distance - tol:
                if plot_invalid:
                    plot_invalid_centerline_check(
                        corridor_list=corridor_list,
                        centerline_specs=[
                            {
                                "corridor": corridor1,
                                "label": f"centerline corridor {i}",
                                "distance": centerline_distance,
                                "linestyle": "-",
                            },
                            {
                                "corridor": corridor3,
                                "label": f"centerline corridor {i+2}",
                                "distance": None,
                                "linestyle": "-",
                            },
                        ],
                        triple_indices=(i, i + 1, i + 2),
                        case_id=case_id,
                        required_distance=min_centerline_distance,
                        plot_function=plot_function,
                    )

                raise ValueError(
                    f"Invalid corridor sequence at triple {i}, {i+1}, {i+2}. "
                    f"Case {case_id}: centerline distance between corridor "
                    f"{i} and corridor {i+2} is {centerline_distance:.3f}, "
                    f"but at least {min_centerline_distance:.3f} is required."
                )

        # ------------------------------------------------------------
        # Case 2: HHV or VVH
        # Rotate corridor1 by pi/2, then compare with corridor3.
        # ------------------------------------------------------------
        elif case_id in {"HHV", "VVH"}:
            rotated_corridor1 = corridor1.invert_dimensions()

            centerline_distance = compute_distance_between_parallel_centerlines(
                rotated_corridor1,
                corridor3,
                tol=tol,
            )

            print(
                f"Triple {i}, {i+1}, {i+2}: case {case_id}, "
                f"rotating corridor {i}, "
                f"distance = {centerline_distance:.3f}, "
                f"required = {min_centerline_distance:.3f}"
            )

            if centerline_distance < min_centerline_distance - tol:
                if plot_invalid:
                    plot_invalid_centerline_check(
                        corridor_list=corridor_list,
                        centerline_specs=[
                            {
                                "corridor": rotated_corridor1,
                                "label": f"rotated centerline corridor {i}",
                                "distance": centerline_distance,
                                "linestyle": "--",
                            },
                            {
                                "corridor": corridor3,
                                "label": f"centerline corridor {i+2}",
                                "distance": None,
                                "linestyle": "-",
                            },
                        ],
                        triple_indices=(i, i + 1, i + 2),
                        case_id=case_id,
                        required_distance=min_centerline_distance,
                        plot_function=plot_function,
                    )

                raise ValueError(
                    f"Invalid corridor sequence at triple {i}, {i+1}, {i+2}. "
                    f"Case {case_id}: after rotating corridor {i}, "
                    f"centerline distance is {centerline_distance:.3f}, "
                    f"but at least {min_centerline_distance:.3f} is required."
                )

        # ------------------------------------------------------------
        # Case 3: HVV or VHH
        # Rotate corridor3 by pi/2, then compare with corridor1.
        # ------------------------------------------------------------
        elif case_id in {"HVV", "VHH"}:
            rotated_corridor3 = corridor3.invert_dimensions()

            centerline_distance = compute_distance_between_parallel_centerlines(
                corridor1,
                rotated_corridor3,
                tol=tol,
            )

            print(
                f"Triple {i}, {i+1}, {i+2}: case {case_id}, "
                f"rotating corridor {i+2}, "
                f"distance = {centerline_distance:.3f}, "
                f"required = {min_centerline_distance:.3f}"
            )

            if centerline_distance < min_centerline_distance - tol:
                if plot_invalid:
                    plot_invalid_centerline_check(
                        corridor_list=corridor_list,
                        centerline_specs=[
                            {
                                "corridor": corridor1,
                                "label": f"centerline corridor {i}",
                                "distance": centerline_distance,
                                "linestyle": "-",
                            },
                            {
                                "corridor": rotated_corridor3,
                                "label": f"rotated centerline corridor {i+2}",
                                "distance": None,
                                "linestyle": "--",
                            },
                        ],
                        triple_indices=(i, i + 1, i + 2),
                        case_id=case_id,
                        required_distance=min_centerline_distance,
                        plot_function=plot_function,
                    )

                raise ValueError(
                    f"Invalid corridor sequence at triple {i}, {i+1}, {i+2}. "
                    f"Case {case_id}: after rotating corridor {i+2}, "
                    f"centerline distance is {centerline_distance:.3f}, "
                    f"but at least {min_centerline_distance:.3f} is required."
                )

        # ------------------------------------------------------------
        # Case 4: HHH or VVV
        # Check both current and rotated versions.
        # The triple is valid if at least one check is valid.
        # ------------------------------------------------------------
        elif case_id in {"HHH", "VVV"}:
            current_centerline_distance = compute_distance_between_parallel_centerlines(
                corridor1,
                corridor3,
                tol=tol,
            )

            rotated_corridor1 = corridor1.invert_dimensions()
            rotated_corridor3 = corridor3.invert_dimensions()

            rotated_centerline_distance = compute_distance_between_parallel_centerlines(
                rotated_corridor1,
                rotated_corridor3,
                tol=tol,
            )

            current_check_ok = (
                current_centerline_distance >= min_centerline_distance - tol
            )

            rotated_check_ok = (
                rotated_centerline_distance >= min_centerline_distance - tol
            )

            print(
                f"Triple {i}, {i+1}, {i+2}: case {case_id}, "
                f"current distance = {current_centerline_distance:.3f}, "
                f"rotated distance = {rotated_centerline_distance:.3f}, "
                f"required = {min_centerline_distance:.3f}"
            )

            if not (current_check_ok or rotated_check_ok):
                if plot_invalid:
                    plot_invalid_centerline_check(
                        corridor_list=corridor_list,
                        centerline_specs=[
                            {
                                "corridor": corridor1,
                                "label": f"centerline corridor {i}",
                                "distance": current_centerline_distance,
                                "linestyle": "-",
                            },
                            {
                                "corridor": corridor3,
                                "label": f"centerline corridor {i+2}",
                                "distance": None,
                                "linestyle": "-",
                            },
                            {
                                "corridor": rotated_corridor1,
                                "label": f"rotated centerline corridor {i}",
                                "distance": rotated_centerline_distance,
                                "linestyle": "--",
                            },
                            {
                                "corridor": rotated_corridor3,
                                "label": f"rotated centerline corridor {i+2}",
                                "distance": None,
                                "linestyle": "--",
                            },
                        ],
                        triple_indices=(i, i + 1, i + 2),
                        case_id=case_id,
                        required_distance=min_centerline_distance,
                        plot_function=plot_function,
                    )

                raise ValueError(
                    f"Invalid corridor sequence at triple {i}, {i+1}, {i+2}. "
                    f"Case {case_id}: neither current nor rotated centerline "
                    f"separation is sufficient. "
                    f"Current distance = {current_centerline_distance:.3f}, "
                    f"rotated distance = {rotated_centerline_distance:.3f}, "
                    f"required = {min_centerline_distance:.3f}."
                )

        # ------------------------------------------------------------
        # Unexpected case
        # ------------------------------------------------------------
        else:
            raise ValueError(
                f"Unexpected corridor axis case at triple {i}, {i+1}, {i+2}: "
                f"{case_id}"
            )

    return True