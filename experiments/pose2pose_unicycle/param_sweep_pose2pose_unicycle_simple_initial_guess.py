from math import pi, degrees
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.spatial import cKDTree

from kappa_planner import (
    LinearSegmentUnicycle,
    TurnOnTheSpot,
)

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle

from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)

from kappa_planner.helpers.ocp_pose_to_pose_unicycle import (
    compute_ocp_pose_to_pose_trajectory,
)

from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1 = orientation sweep
# 2 = D/R sweep
# 3 = polar final-position sweep
# 4 = Sobol sweep
SWEEP_ID = 4

# OCP transcription settings
N = 30
M = 4


# -----------------------------------------------------------------------------
# TST INITIAL GUESS
# -----------------------------------------------------------------------------

# Number of samples used inside each primitive of the TST initial guess.
INITIAL_GUESS_SAMPLES_PER_MANEUVER = 20


# -----------------------------------------------------------------------------
# HAUSDORFF DISTANCE
# -----------------------------------------------------------------------------

# Number of samples used to represent each spatial analytical primitive
# before the common arclength resampling.
ANALYTICAL_GEOMETRY_SAMPLES_PER_PRIMITIVE = 1001

# Number of approximately uniform-arclength points used for BOTH analytical
# and OCP planar paths when computing the discrete Hausdorff distance.
HAUSDORFF_N_SAMPLES = 2001


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

QUICK_TEST = True
QUICK_TEST_N_CASES = 10


# -----------------------------------------------------------------------------
# DIAGNOSTICS
# -----------------------------------------------------------------------------

# Disable for production runs, especially on a remote machine.
PLOT_SUSPICIOUS_CASES = False
SUSPICIOUS_TIME_ERROR_THRESHOLD = 0.2

# Print the keys returned by the OCP solver for the first OCP call.
PRINT_OCP_KEYS_ON_FIRST_CASE = True


# -----------------------------------------------------------------------------
# SWEEP SETTINGS
# -----------------------------------------------------------------------------

SWEEP1_N_ANGLES = 90

# Number of requested Sobol samples = 2**SOBOL_POWER
SOBOL_POWER = 13

SOBOL_SEED = 1
SOBOL_SCRAMBLE = True


# =============================================================================
# TST INITIAL GUESS
# =============================================================================

def signed_shortest_angle(
    theta_from,
    theta_to,
):
    """
    Signed shortest angular displacement from theta_from to theta_to.

    The result lies in [-pi, pi].
    """

    return float(
        np.arctan2(
            np.sin(theta_to - theta_from),
            np.cos(theta_to - theta_from),
        )
    )


def build_turn_on_spot(
    x,
    y,
    theta_from,
    theta_to,
    vehicle,
    t0,
    samples_number,
):
    """
    Construct a maximum-angular-speed turn on the spot following the
    shortest angular displacement.
    """

    delta_theta = signed_shortest_angle(
        theta_from,
        theta_to,
    )

    if abs(delta_theta) < 1.0e-12:
        omega = 0.0

    else:
        omega = float(
            np.copysign(
                vehicle.omega_max,
                delta_theta,
            )
        )

    theta_end = (
        theta_from
        + delta_theta
    )

    return TurnOnTheSpot(
        x,
        y,
        theta_from,
        theta_end,
        omega,
        unicycle=vehicle,
        t0=t0,
        samples_number=samples_number,
    )


def build_tst_initial_guess(
    start_pose,
    end_pose,
    vehicle,
    samples_per_maneuver=20,
):
    """
    Construct the direct T-S-T initial guess:

        turn on the spot
        -> straight segment
        -> turn on the spot

    The straight segment joins the two boundary positions directly.

    Returns
    -------
    trajectory : list
        [initial_turn, straight_segment, final_turn]

    T_guess : float
        Total traversal time of the TST initial guess.

    information : dict
        Diagnostic information about the TST construction.
    """

    x0 = float(
        start_pose.position.x
    )

    y0 = float(
        start_pose.position.y
    )

    xf = float(
        end_pose.position.x
    )

    yf = float(
        end_pose.position.y
    )

    theta0 = float(
        start_pose.theta
    )

    thetaf = float(
        end_pose.theta
    )

    dx = xf - x0
    dy = yf - y0

    distance = float(
        np.hypot(
            dx,
            dy,
        )
    )

    if distance < 1.0e-12:
        raise ValueError(
            "The TST initial guess requires distinct "
            "initial and final positions."
        )

    segment_heading = float(
        np.arctan2(
            dy,
            dx,
        )
    )

    # -------------------------------------------------------------------------
    # Initial turn
    # -------------------------------------------------------------------------

    initial_turn = build_turn_on_spot(
        x=x0,
        y=y0,
        theta_from=theta0,
        theta_to=segment_heading,
        vehicle=vehicle,
        t0=0.0,
        samples_number=samples_per_maneuver,
    )

    first_heading = float(
        initial_turn.end_pose[2]
    )

    # -------------------------------------------------------------------------
    # Straight segment
    # -------------------------------------------------------------------------

    straight_segment = (
        LinearSegmentUnicycle(
            x0,
            y0,
            xf,
            yf,
            first_heading,
            vehicle.v_max,
            unicycle=vehicle,
            t0=initial_turn.tf,
            samples_number=samples_per_maneuver,
        )
    )

    # -------------------------------------------------------------------------
    # Final turn
    # -------------------------------------------------------------------------

    final_turn = build_turn_on_spot(
        x=xf,
        y=yf,
        theta_from=first_heading,
        theta_to=thetaf,
        vehicle=vehicle,
        t0=straight_segment.tf,
        samples_number=samples_per_maneuver,
    )

    trajectory = [
        initial_turn,
        straight_segment,
        final_turn,
    ]

    T_guess = float(
        sum(
            primitive.maneuver_time
            for primitive
            in trajectory
        )
    )

    information = {
        "segment_heading_rad":
            float(
                segment_heading
            ),

        "segment_heading_deg":
            float(
                degrees(
                    segment_heading
                )
            ),

        "distance":
            distance,

        "initial_turn_angle_rad":
            float(
                signed_shortest_angle(
                    theta0,
                    segment_heading,
                )
            ),

        "final_turn_angle_rad":
            float(
                signed_shortest_angle(
                    segment_heading,
                    thetaf,
                )
            ),

        "T_guess":
            T_guess,
    }

    return (
        trajectory,
        T_guess,
        information,
    )


# =============================================================================
# HAUSDORFF-DISTANCE HELPERS
# =============================================================================

def extract_dense_analytical_planar_path(
    trajectory,
    samples_per_spatial_primitive=1001,
):
    """
    Build a dense planar representation of an analytical trajectory.

    Circular arcs are sampled directly from their circle geometry.
    Straight segments are sampled directly along the segment.
    Turns on the spot contribute only their fixed Cartesian position.

    This prevents the Hausdorff computation from depending on the plotting
    resolution stored in primitive.path_coordinates.
    """

    path_parts = []

    for primitive in trajectory:

        label = primitive.label.lower()

        # ---------------------------------------------------------------------
        # Circular arc
        # ---------------------------------------------------------------------

        if label == "arc":

            required_attributes = (
                "xc",
                "yc",
                "radius",
                "epsilon",
                "turn_direction",
                "iota",
            )

            missing_attributes = [
                attribute
                for attribute
                in required_attributes
                if not hasattr(
                    primitive,
                    attribute,
                )
            ]

            if missing_attributes:
                raise AttributeError(
                    "Arc primitive is missing required "
                    "geometric attributes: "
                    f"{missing_attributes}"
                )

            angles = np.linspace(
                primitive.epsilon,
                (
                    primitive.epsilon
                    + primitive.turn_direction
                    * primitive.iota
                ),
                samples_per_spatial_primitive,
            )

            x = (
                primitive.xc
                + primitive.radius
                * np.cos(
                    angles
                )
            )

            y = (
                primitive.yc
                + primitive.radius
                * np.sin(
                    angles
                )
            )

            coordinates = np.column_stack(
                (
                    x,
                    y,
                )
            )

        # ---------------------------------------------------------------------
        # Straight segment
        # ---------------------------------------------------------------------

        elif label == "segment":

            coordinates = np.column_stack(
                (
                    np.linspace(
                        primitive.x0,
                        primitive.xf,
                        samples_per_spatial_primitive,
                    ),
                    np.linspace(
                        primitive.y0,
                        primitive.yf,
                        samples_per_spatial_primitive,
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Turn on the spot
        # ---------------------------------------------------------------------

        elif label == "turn on-the-spot":

            coordinates = np.array(
                [
                    [
                        float(
                            primitive.x0
                        ),
                        float(
                            primitive.y0
                        ),
                    ]
                ],
                dtype=float,
            )

        # ---------------------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------------------

        else:

            if not hasattr(
                primitive,
                "path_coordinates",
            ):
                raise AttributeError(
                    "Analytical primitive does not expose "
                    "path_coordinates."
                )

            coordinates = np.asarray(
                primitive.path_coordinates,
                dtype=float,
            )

        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
        ):
            raise ValueError(
                "Analytical primitive path coordinates "
                "must have shape (n, 2)."
            )

        # Avoid storing the common boundary point twice.
        if (
            len(path_parts) > 0
            and len(coordinates) > 0
        ):
            coordinates = coordinates[
                1:
            ]

        if len(coordinates) > 0:
            path_parts.append(
                coordinates
            )

    if not path_parts:
        raise ValueError(
            "The analytical trajectory contains no planar path points."
        )

    path = np.vstack(
        path_parts
    )

    return (
        path[:, 0],
        path[:, 1],
    )


def resample_planar_path_by_arclength(
    xs,
    ys,
    n_samples,
):
    """
    Resample a planar path onto an approximately uniform arclength grid.
    """

    xs = np.asarray(
        xs,
        dtype=float,
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=float,
    ).reshape(-1)

    if len(xs) != len(ys):
        raise ValueError(
            "x and y arrays must have equal lengths."
        )

    if len(xs) < 2:
        raise ValueError(
            "A planar path must contain at least two points."
        )

    if not (
        np.all(
            np.isfinite(xs)
        )
        and np.all(
            np.isfinite(ys)
        )
    ):
        raise ValueError(
            "Planar path contains non-finite values."
        )

    segment_lengths = np.hypot(
        np.diff(xs),
        np.diff(ys),
    )

    cumulative_length = np.concatenate(
        (
            [0.0],
            np.cumsum(
                segment_lengths
            ),
        )
    )

    total_length = float(
        cumulative_length[-1]
    )

    if total_length <= 0.0:
        raise ValueError(
            "The planar path has zero geometric length."
        )

    # Repeated Cartesian points occur naturally during turns on the spot.
    # Keep only one point for each repeated cumulative-arclength value.
    unique_length, unique_indices = np.unique(
        cumulative_length,
        return_index=True,
    )

    xs_unique = xs[
        unique_indices
    ]

    ys_unique = ys[
        unique_indices
    ]

    target_length = np.linspace(
        0.0,
        total_length,
        n_samples,
    )

    x_resampled = np.interp(
        target_length,
        unique_length,
        xs_unique,
    )

    y_resampled = np.interp(
        target_length,
        unique_length,
        ys_unique,
    )

    return (
        x_resampled,
        y_resampled,
    )


def symmetric_hausdorff_distance(
    x_a,
    y_a,
    x_b,
    y_b,
):
    """
    Compute the discrete symmetric Hausdorff distance between two planar
    point sets.

    Returns both directed distances in addition to the symmetric value.
    """

    points_a = np.column_stack(
        (
            np.asarray(
                x_a,
                dtype=float,
            ),
            np.asarray(
                y_a,
                dtype=float,
            ),
        )
    )

    points_b = np.column_stack(
        (
            np.asarray(
                x_b,
                dtype=float,
            ),
            np.asarray(
                y_b,
                dtype=float,
            ),
        )
    )

    tree_a = cKDTree(
        points_a
    )

    tree_b = cKDTree(
        points_b
    )

    distances_a_to_b, _ = (
        tree_b.query(
            points_a,
            k=1,
        )
    )

    distances_b_to_a, _ = (
        tree_a.query(
            points_b,
            k=1,
        )
    )

    directed_a_to_b = float(
        np.max(
            distances_a_to_b
        )
    )

    directed_b_to_a = float(
        np.max(
            distances_b_to_a
        )
    )

    return {
        "symmetric":
            max(
                directed_a_to_b,
                directed_b_to_a,
            ),

        "a_to_b":
            directed_a_to_b,

        "b_to_a":
            directed_b_to_a,
    }


# =============================================================================
# PRINTING AND DIAGNOSTIC FUNCTIONS
# =============================================================================

def print_case_result(
    case_id,
    n_cases,
    theta0,
    thetaf,
    best_name,
    best_time,
    ocp_result,
    best_trajectory,
    tst_information,
    absolute_time_error,
    relative_time_error_percent,
    hausdorff_distance,
    x0,
    y0,
    xf,
    yf,
):
    """
    Print the main results for one converged case.

    Optionally plot cases whose traversal-time discrepancy is suspiciously
    large.
    """

    ocp_time = float(
        ocp_result["time"]
    )

    solve_time = float(
        ocp_result["solve_time"]
    )

    sequence = " - ".join(
        ocp_result["sequence"]
    )

    signed_time_error = (
        ocp_time
        - best_time
    )

    if (
        PLOT_SUSPICIOUS_CASES
        and absolute_time_error
        > SUSPICIOUS_TIME_ERROR_THRESHOLD
    ):
        print(
            "  -> Plotting suspicious case"
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        plot_analytical_trajectory(
            best_trajectory,
            figure=ax,
            plot_circles=True,
            color="blue",
            linewidth=2.5,
            plot_primitive_arrows=True,
            plot_turn_sectors=True,
        )

        ax.plot(
            ocp_result["xs"],
            ocp_result["ys"],
            "k--",
            linewidth=2.0,
            label=(
                f"OCP "
                f"({ocp_time:.3f} s)"
            ),
        )

        initial_guess = ocp_result.get(
            "initial_guess"
        )

        if initial_guess is not None:
            ax.plot(
                initial_guess["x"],
                initial_guess["y"],
                color="gray",
                linestyle=":",
                linewidth=2,
                marker="o",
                markersize=3,
                label="TST initial guess",
            )

        arrow_scale = 0.5

        ax.quiver(
            x0,
            y0,
            arrow_scale
            * np.cos(
                theta0
            ),
            arrow_scale
            * np.sin(
                theta0
            ),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
        )

        ax.quiver(
            xf,
            yf,
            arrow_scale
            * np.cos(
                thetaf
            ),
            arrow_scale
            * np.sin(
                thetaf
            ),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )

        ax.set_title(
            f"Case {case_id}\n"
            f"Analytical: {best_time:.3f} s\n"
            f"TST guess: "
            f"{tst_information['T_guess']:.3f} s\n"
            f"OCP: {ocp_time:.3f} s\n"
            f"|ΔT| = {absolute_time_error:.3f} s"
        )

        ax.axis(
            "equal"
        )

        ax.grid(
            True
        )

        ax.legend()

        plt.show(
            block=True
        )

    print(
        f"\nCase {case_id:04d}/{n_cases}: "
        f"theta0={degrees(theta0):7.2f} deg, "
        f"thetaf={degrees(thetaf):7.2f} deg"
    )

    print(
        f"  Best analytical : "
        f"{best_name}"
    )

    print(
        f"  Analytical time : "
        f"{best_time:.9f} s"
    )

    print(
        f"  TST guess time  : "
        f"{tst_information['T_guess']:.9f} s"
    )

    print(
        f"  OCP time        : "
        f"{ocp_time:.9f} s"
    )

    print(
        f"  Signed ΔT       : "
        f"{signed_time_error:.9e} s"
    )

    print(
        f"  Absolute |ΔT|   : "
        f"{absolute_time_error:.9e} s"
    )

    print(
        f"  Relative error  : "
        f"{relative_time_error_percent:.6f} %"
    )

    print(
        f"  Hausdorff       : "
        f"{hausdorff_distance:.9e} m"
    )

    print(
        f"  OCP solve time  : "
        f"{solve_time:.3f} s"
    )

    print(
        f"  OCP sequence    : "
        f"{sequence}"
    )


# =============================================================================
# ANALYTICAL TRAJECTORY INFORMATION
# =============================================================================

def compute_arc_segment_ratios(
    trajectory,
):
    """
    Compute the ratios between the straight-segment length and the lengths
    of the first and second arcs.

    Returned only when the trajectory contains exactly two arcs and one
    straight segment.
    """

    arcs = [
        primitive
        for primitive
        in trajectory
        if primitive.label.lower()
        == "arc"
    ]

    segments = [
        primitive
        for primitive
        in trajectory
        if primitive.label.lower()
        == "segment"
    ]

    empty_result = {
        "arc1_length": None,
        "segment_length": None,
        "arc2_length": None,
        "r1": None,
        "r2": None,
    }

    if (
        len(arcs) != 2
        or len(segments) != 1
    ):
        return empty_result

    arc1 = arcs[0]
    arc2 = arcs[1]
    segment = segments[0]

    r1 = (
        segment.path_length
        / arc1.path_length
        if arc1.path_length
        > 1.0e-12
        else None
    )

    r2 = (
        segment.path_length
        / arc2.path_length
        if arc2.path_length
        > 1.0e-12
        else None
    )

    return {
        "arc1_length":
            float(
                arc1.path_length
            ),

        "segment_length":
            float(
                segment.path_length
            ),

        "arc2_length":
            float(
                arc2.path_length
            ),

        "r1":
            (
                float(r1)
                if r1 is not None
                else None
            ),

        "r2":
            (
                float(r2)
                if r2 is not None
                else None
            ),
    }


def extract_analytical_primitive_info(
    trajectory,
):
    """
    Extract information required to reconstruct the analytical trajectory
    after loading the saved JSON file.
    """

    primitives_info = []

    for index, primitive in enumerate(
        trajectory,
        start=1,
    ):

        primitive_info = {
            "index":
                index,

            "label":
                primitive.label,

            "maneuver_time":
                float(
                    primitive.maneuver_time
                ),

            "path_length":
                float(
                    primitive.path_length
                ),
        }

        if hasattr(
            primitive,
            "turn_direction",
        ):
            primitive_info[
                "turn_direction"
            ] = int(
                primitive.turn_direction
            )

        if hasattr(
            primitive,
            "iota",
        ):
            primitive_info[
                "angular_amplitude_rad"
            ] = float(
                primitive.iota
            )

            primitive_info[
                "angular_amplitude_deg"
            ] = float(
                degrees(
                    primitive.iota
                )
            )

        elif hasattr(
            primitive,
            "delta_angle",
        ):
            primitive_info[
                "angular_amplitude_rad"
            ] = float(
                primitive.delta_angle
            )

            primitive_info[
                "angular_amplitude_deg"
            ] = float(
                degrees(
                    primitive.delta_angle
                )
            )

        if hasattr(
            primitive,
            "start_pose",
        ):
            primitive_info[
                "start_pose"
            ] = [
                float(value)
                for value
                in primitive.start_pose
            ]

        if hasattr(
            primitive,
            "end_pose",
        ):
            primitive_info[
                "end_pose"
            ] = [
                float(value)
                for value
                in primitive.end_pose
            ]

        primitives_info.append(
            primitive_info
        )

    return primitives_info


# =============================================================================
# OCP TRAJECTORY EXTRACTION
# =============================================================================

def extract_ocp_pose_trajectory(
    ocp_result,
):
    """
    Extract the OCP time grid and state trajectory.
    """

    required_state_keys = (
        "xs",
        "ys",
        "thetas",
    )

    missing_keys = [
        key
        for key
        in required_state_keys
        if key not in ocp_result
    ]

    if missing_keys:
        raise KeyError(
            "The OCP result does not contain "
            "the required trajectory entries: "
            f"{missing_keys}. "
            "Available keys are: "
            f"{sorted(ocp_result.keys())}"
        )

    ocp_x = np.asarray(
        ocp_result["xs"],
        dtype=float,
    ).reshape(-1)

    ocp_y = np.asarray(
        ocp_result["ys"],
        dtype=float,
    ).reshape(-1)

    ocp_theta = np.asarray(
        ocp_result["thetas"],
        dtype=float,
    ).reshape(-1)

    n_samples = len(
        ocp_x
    )

    if n_samples < 2:
        raise ValueError(
            "The OCP trajectory must contain "
            "at least two state samples."
        )

    if (
        len(ocp_y) != n_samples
        or len(ocp_theta)
        != n_samples
    ):
        raise ValueError(
            "The OCP state arrays xs, ys, and "
            "thetas must have the same length."
        )

    possible_time_keys = (
        "times",
        "time_grid",
        "ts",
        "t",
    )

    ocp_time_grid = None

    for key in possible_time_keys:

        if key in ocp_result:

            candidate_grid = np.asarray(
                ocp_result[key],
                dtype=float,
            ).reshape(-1)

            if (
                len(candidate_grid)
                == n_samples
            ):
                ocp_time_grid = (
                    candidate_grid
                )
                break

    if ocp_time_grid is None:

        ocp_time_grid = np.linspace(
            0.0,
            float(
                ocp_result["time"]
            ),
            n_samples,
            endpoint=True,
        )

    if not np.all(
        np.isfinite(
            ocp_time_grid
        )
    ):
        raise ValueError(
            "The OCP time grid contains "
            "non-finite values."
        )

    if not np.all(
        np.diff(
            ocp_time_grid
        )
        >= 0.0
    ):
        raise ValueError(
            "The OCP time grid must be "
            "monotonically nondecreasing."
        )

    return (
        ocp_time_grid,
        ocp_x,
        ocp_y,
        ocp_theta,
    )


# =============================================================================
# PARAMETER SWEEPS
# =============================================================================

def generate_sweep_cases(
    sweep_id,
):
    """
    Return the cases and metadata for the selected sweep.
    """

    cases = []

    # =========================================================================
    # SWEEP 1
    # =========================================================================

    if sweep_id == 1:

        x0, y0 = (
            0.0,
            0.0,
        )

        xf, yf = (
            0.0,
            5.0,
        )

        v_max = 1.0
        omega_max = 1.0

        radius = (
            v_max
            / omega_max
        )

        n_angles = (
            SWEEP1_N_ANGLES
        )

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        for theta0 in start_angles:
            for thetaf in final_angles:

                distance = float(
                    np.hypot(
                        xf - x0,
                        yf - y0,
                    )
                )

                cases.append({
                    "x0":
                        x0,

                    "y0":
                        y0,

                    "xf":
                        xf,

                    "yf":
                        yf,

                    "theta0":
                        float(
                            theta0
                        ),

                    "thetaf":
                        float(
                            thetaf
                        ),

                    "v_max":
                        v_max,

                    "omega_max":
                        omega_max,

                    "R":
                        float(
                            radius
                        ),

                    "D":
                        distance,

                    "D_over_R":
                        float(
                            distance
                            / radius
                        ),
                })

        metadata = {
            "sweep_id":
                1,

            "sweep_name":
                "orientation_sweep_OCP_TST_initial_guess",

            "description":
                (
                    "Fixed positions and R. "
                    "Vary theta0 and thetaf. "
                    "Initialize the OCP with a direct "
                    "turn-segment-turn trajectory."
                ),

            "n_angles":
                n_angles,

            "D_over_R":
                5.0,

            "n_cases_full":
                len(
                    cases
                ),
        }

    # =========================================================================
    # SWEEP 2
    # =========================================================================

    elif sweep_id == 2:

        x0, y0 = (
            0.0,
            0.0,
        )

        v_max = 1.0
        omega_max = 1.0

        radius = (
            v_max
            / omega_max
        )

        d_over_r_values = [
            10.0,
            15.0,
            20.0,
        ]

        n_angles = 50

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        for d_over_r in d_over_r_values:

            xf = 0.0
            yf = (
                d_over_r
                * radius
            )

            distance = float(
                np.hypot(
                    xf - x0,
                    yf - y0,
                )
            )

            for theta0 in start_angles:
                for thetaf in final_angles:

                    cases.append({
                        "x0":
                            x0,

                        "y0":
                            y0,

                        "xf":
                            float(
                                xf
                            ),

                        "yf":
                            float(
                                yf
                            ),

                        "theta0":
                            float(
                                theta0
                            ),

                        "thetaf":
                            float(
                                thetaf
                            ),

                        "v_max":
                            v_max,

                        "omega_max":
                            omega_max,

                        "R":
                            float(
                                radius
                            ),

                        "D":
                            distance,

                        "D_over_R":
                            float(
                                d_over_r
                            ),
                    })

        metadata = {
            "sweep_id":
                2,

            "sweep_name":
                "distance_over_radius_sweep_OCP_TST_initial_guess",

            "description":
                (
                    "Vary D/R and boundary orientations. "
                    "Initialize the OCP with a direct "
                    "turn-segment-turn trajectory."
                ),

            "d_over_r_values":
                d_over_r_values,

            "n_angles":
                n_angles,

            "n_cases_full":
                len(
                    cases
                ),
        }

    # =========================================================================
    # SWEEP 3
    # =========================================================================

    elif sweep_id == 3:

        x0, y0 = (
            0.0,
            0.0,
        )

        v_max = 1.0
        omega_max = 1.0

        radius = (
            v_max
            / omega_max
        )

        d_over_r_values = [
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        n_goal_angles = 16

        goal_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_goal_angles,
            endpoint=False,
        )

        n_angles = 12

        start_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        final_angles = np.linspace(
            0.0,
            2.0 * pi,
            n_angles,
            endpoint=False,
        )

        for d_over_r in d_over_r_values:

            distance = (
                d_over_r
                * radius
            )

            for phi in goal_angles:

                xf = (
                    x0
                    + distance
                    * np.cos(
                        phi
                    )
                )

                yf = (
                    y0
                    + distance
                    * np.sin(
                        phi
                    )
                )

                for theta0 in start_angles:
                    for thetaf in final_angles:

                        cases.append({
                            "x0":
                                x0,

                            "y0":
                                y0,

                            "xf":
                                float(
                                    xf
                                ),

                            "yf":
                                float(
                                    yf
                                ),

                            "theta0":
                                float(
                                    theta0
                                ),

                            "thetaf":
                                float(
                                    thetaf
                                ),

                            "phi":
                                float(
                                    phi
                                ),

                            "phi_deg":
                                float(
                                    degrees(
                                        phi
                                    )
                                ),

                            "D":
                                float(
                                    distance
                                ),

                            "D_over_R":
                                float(
                                    d_over_r
                                ),

                            "R":
                                float(
                                    radius
                                ),

                            "v_max":
                                v_max,

                            "omega_max":
                                omega_max,
                        })

        metadata = {
            "sweep_id":
                3,

            "sweep_name":
                "polar_goal_position_sweep_OCP_TST_initial_guess",

            "description":
                (
                    "Vary final position and boundary orientations. "
                    "Initialize the OCP with a direct "
                    "turn-segment-turn trajectory."
                ),

            "d_over_r_values":
                d_over_r_values,

            "n_goal_angles":
                n_goal_angles,

            "n_angles":
                n_angles,

            "R":
                radius,

            "distance_assumption":
                "D/R > 4",

            "n_cases_full":
                len(
                    cases
                ),
        }

    # =========================================================================
    # SWEEP 4 — SOBOL
    # =========================================================================

    elif sweep_id == 4:

        from scipy.stats import qmc

        x0, y0 = (
            0.0,
            0.0,
        )

        n_samples = (
            2 ** SOBOL_POWER
        )

        sampler = qmc.Sobol(
            d=5,
            scramble=SOBOL_SCRAMBLE,
            seed=SOBOL_SEED,
        )

        samples = (
            sampler.random_base2(
                m=SOBOL_POWER
            )
        )

        x_min, x_max = (
            -15.0,
            15.0,
        )

        y_min, y_max = (
            -15.0,
            15.0,
        )

        r_min, r_max = (
            0.5,
            2.0,
        )

        v_max = 1.0

        for sample in samples:

            (
                sx,
                sy,
                stheta0,
                sthetaf,
                sr,
            ) = sample

            xf = (
                x_min
                + sx
                * (
                    x_max
                    - x_min
                )
            )

            yf = (
                y_min
                + sy
                * (
                    y_max
                    - y_min
                )
            )

            theta0 = (
                2.0
                * pi
                * stheta0
            )

            thetaf = (
                2.0
                * pi
                * sthetaf
            )

            radius = (
                r_min
                + sr
                * (
                    r_max
                    - r_min
                )
            )

            omega_max = (
                v_max
                / radius
            )

            distance = float(
                np.hypot(
                    xf - x0,
                    yf - y0,
                )
            )

            if (
                distance
                <= 4.0
                * radius
            ):
                continue

            cases.append({
                "x0":
                    x0,

                "y0":
                    y0,

                "xf":
                    float(
                        xf
                    ),

                "yf":
                    float(
                        yf
                    ),

                "theta0":
                    float(
                        theta0
                    ),

                "thetaf":
                    float(
                        thetaf
                    ),

                "v_max":
                    float(
                        v_max
                    ),

                "omega_max":
                    float(
                        omega_max
                    ),

                "R":
                    float(
                        radius
                    ),

                "D":
                    float(
                        distance
                    ),

                "D_over_R":
                    float(
                        distance
                        / radius
                    ),
            })

        metadata = {
            "sweep_id":
                4,

            "sweep_name":
                "sobol_sweep_OCP_TST_initial_guess",

            "description":
                (
                    "Sobol sampling over final position, "
                    "theta0, thetaf, and R. "
                    "Initialize the OCP with a direct "
                    "turn-segment-turn trajectory."
                ),

            "sobol_dimension":
                5,

            "sobol_power":
                SOBOL_POWER,

            "sobol_seed":
                SOBOL_SEED,

            "sobol_scramble":
                SOBOL_SCRAMBLE,

            "n_samples_requested":
                n_samples,

            "n_samples_valid":
                len(
                    cases
                ),

            "xf_range": [
                x_min,
                x_max,
            ],

            "yf_range": [
                y_min,
                y_max,
            ],

            "R_range": [
                r_min,
                r_max,
            ],

            "v_max":
                v_max,

            "distance_assumption":
                "D > 4R",
        }

    else:

        raise ValueError(
            f"Unknown sweep_id: "
            f"{sweep_id}"
        )

    return (
        cases,
        metadata,
    )


# =============================================================================
# COMMON CASE INFORMATION
# =============================================================================

def build_common_case_information(
    case_id,
    x0,
    y0,
    xf,
    yf,
    theta0,
    thetaf,
    vehicle_vmax,
    vehicle_omegamax,
    unicycle,
    N,
    M,
):
    """
    Build information shared by successful and failed OCP result records.
    """

    distance = float(
        np.hypot(
            xf - x0,
            yf - y0,
        )
    )

    radius = float(
        unicycle.max_radius
    )

    return {
        "case_id":
            int(
                case_id
            ),

        "x0":
            float(
                x0
            ),

        "y0":
            float(
                y0
            ),

        "xf":
            float(
                xf
            ),

        "yf":
            float(
                yf
            ),

        "theta0_rad":
            float(
                theta0
            ),

        "thetaf_rad":
            float(
                thetaf
            ),

        "theta0_deg":
            float(
                degrees(
                    theta0
                )
            ),

        "thetaf_deg":
            float(
                degrees(
                    thetaf
                )
            ),

        "v_max":
            float(
                vehicle_vmax
            ),

        "omega_max":
            float(
                vehicle_omegamax
            ),

        "R":
            radius,

        "D":
            distance,

        "D_over_R":
            float(
                distance
                / radius
            ),

        "N":
            int(
                N
            ),

        "M":
            int(
                M
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    cases, metadata = (
        generate_sweep_cases(
            SWEEP_ID
        )
    )

    full_n_cases = len(
        cases
    )

    # -------------------------------------------------------------------------
    # Quick test
    # -------------------------------------------------------------------------

    if QUICK_TEST:
        cases = cases[
            :QUICK_TEST_N_CASES
        ]

    n_cases = len(
        cases
    )

    results = []

    first_ocp_case = True

    print(
        "\n"
        + "=" * 80
    )

    print(
        "POSE-TO-POSE UNICYCLE SWEEP "
        "— TST OCP INITIAL GUESS"
    )

    print(
        "=" * 80
    )

    print(
        f"Sweep ID                 : "
        f"{SWEEP_ID}"
    )

    print(
        f"Full sweep cases          : "
        f"{full_n_cases}"
    )

    print(
        f"Cases executed            : "
        f"{n_cases}"
    )

    print(
        f"Quick test                : "
        f"{QUICK_TEST}"
    )

    print(
        f"OCP control intervals N   : "
        f"{N}"
    )

    print(
        f"RK steps per interval M   : "
        f"{M}"
    )

    print(
        "OCP initial guess         : "
        "TST"
    )

    print(
        "Hausdorff samples         : "
        f"{HAUSDORFF_N_SAMPLES}"
    )

    simulation_start_time = (
        time.perf_counter()
    )

    # =========================================================================
    # CASE LOOP
    # =========================================================================

    for case_id, case in enumerate(
        cases,
        start=1,
    ):

        x0 = case[
            "x0"
        ]

        y0 = case[
            "y0"
        ]

        xf = case[
            "xf"
        ]

        yf = case[
            "yf"
        ]

        theta0 = case[
            "theta0"
        ]

        thetaf = case[
            "thetaf"
        ]

        vehicle_vmax = case[
            "v_max"
        ]

        vehicle_omegamax = case[
            "omega_max"
        ]

        unicycle = Unicycle(
            state=[
                0,
                0,
                0,
            ],
            width=0.430,
            length=0.430,
            v_max=vehicle_vmax,
            v_min=0,
            omega_max=vehicle_omegamax,
            omega_min=-vehicle_omegamax,
        )

        start_pose = Pose(
            Point(
                x0,
                y0,
            ),
            theta0,
        )

        end_pose = Pose(
            Point(
                xf,
                yf,
            ),
            thetaf,
        )

        common_information = (
            build_common_case_information(
                case_id=
                    case_id,

                x0=
                    x0,

                y0=
                    y0,

                xf=
                    xf,

                yf=
                    yf,

                theta0=
                    theta0,

                thetaf=
                    thetaf,

                vehicle_vmax=
                    vehicle_vmax,

                vehicle_omegamax=
                    vehicle_omegamax,

                unicycle=
                    unicycle,

                N=
                    N,

                M=
                    M,
            )
        )

        # =====================================================================
        # ANALYTICAL PLANNER
        #
        # The analytical solution is used ONLY as the benchmark.
        # It is NOT used to initialize the OCP.
        # =====================================================================

        try:

            analytical_start_time = (
                time.perf_counter()
            )

            trajectories = (
                compute_all_pose_to_pose_trajectories(
                    start_pose,
                    end_pose,
                    unicycle,
                )
            )

            (
                best_name,
                best_data,
            ) = min(
                trajectories.items(),
                key=lambda item:
                    item[1]["time"],
            )

            best_trajectory = (
                best_data[
                    "trajectory"
                ]
            )

            best_time = float(
                best_data[
                    "time"
                ]
            )

            analytical_solve_time = (
                time.perf_counter()
                - analytical_start_time
            )

            best_analytical_primitives = (
                extract_analytical_primitive_info(
                    best_trajectory
                )
            )

            ratios = (
                compute_arc_segment_ratios(
                    best_trajectory
                )
            )

        except Exception as error:

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                f"analytical planner failed: "
                f"{error}"
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    False,

                "ocp_success":
                    False,

                "error_stage":
                    "analytical_planner",

                "error":
                    str(
                        error
                    ),
            })

            results.append(
                failed_result
            )

            continue

        # =====================================================================
        # TST INITIAL GUESS
        # =====================================================================

        try:

            (
                tst_initial_guess,
                tst_T_guess,
                tst_information,
            ) = build_tst_initial_guess(
                start_pose,
                end_pose,
                unicycle,
                samples_per_maneuver=(
                    INITIAL_GUESS_SAMPLES_PER_MANEUVER
                ),
            )

        except Exception as error:

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                f"TST initial guess failed: "
                f"{error}"
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    True,

                "ocp_success":
                    False,

                "error_stage":
                    "TST_initial_guess",

                "error":
                    str(
                        error
                    ),

                "best_analytical_name":
                    best_name,

                "best_analytical_time":
                    float(
                        best_time
                    ),

                "analytical_solve_time":
                    float(
                        analytical_solve_time
                    ),
            })

            results.append(
                failed_result
            )

            continue

        # =====================================================================
        # OCP
        # =====================================================================

        try:

            ocp_result = (
                compute_ocp_pose_to_pose_trajectory(
                    start_pose,
                    end_pose,
                    unicycle,

                    # Historical argument name.
                    # The actual object passed here is the TST trajectory.
                    analytical_initial_guess=(
                        tst_initial_guess
                    ),

                    T_guess=(
                        tst_T_guess
                    ),

                    N=N,
                    M=M,
                )
            )

        except Exception as error:

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                f"OCP helper failed: "
                f"{error}"
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    True,

                "ocp_success":
                    False,

                "error_stage":
                    "ocp_helper",

                "error":
                    str(
                        error
                    ),

                "ocp_initial_guess":
                    "TST",

                "tst_T_guess":
                    float(
                        tst_T_guess
                    ),

                "best_analytical_name":
                    best_name,

                "best_analytical_time":
                    float(
                        best_time
                    ),
            })

            results.append(
                failed_result
            )

            continue

        # ---------------------------------------------------------------------
        # Print returned OCP keys once
        # ---------------------------------------------------------------------

        if (
            PRINT_OCP_KEYS_ON_FIRST_CASE
            and first_ocp_case
        ):

            print(
                "\n"
                + "-" * 80
            )

            print(
                "OCP RESULT KEYS"
            )

            print(
                "-" * 80
            )

            for key in sorted(
                ocp_result.keys()
            ):
                print(
                    key
                )

            print(
                "-" * 80
            )

        first_ocp_case = False

        # =====================================================================
        # IMPORTANT:
        # DO NOT COMPUTE THESIS METRICS FROM A NON-CONVERGED OCP ITERATE
        # =====================================================================

        if not bool(
            ocp_result.get(
                "success",
                False,
            )
        ):

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                "OCP solver did not converge."
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    True,

                "ocp_success":
                    False,

                "error_stage":
                    "ocp_solver",

                "solver_error":
                    ocp_result.get(
                        "solver_error"
                    ),

                "ocp_initial_guess":
                    "TST",

                "tst_T_guess":
                    float(
                        tst_T_guess
                    ),

                "tst_segment_heading_rad":
                    float(
                        tst_information[
                            "segment_heading_rad"
                        ]
                    ),

                "tst_segment_heading_deg":
                    float(
                        tst_information[
                            "segment_heading_deg"
                        ]
                    ),

                "best_analytical_name":
                    best_name,

                "best_analytical_time":
                    float(
                        best_time
                    ),

                "analytical_solve_time":
                    float(
                        analytical_solve_time
                    ),

                "ocp_solve_time":
                    float(
                        ocp_result.get(
                            "solve_time",
                            np.nan,
                        )
                    ),
            })

            results.append(
                failed_result
            )

            continue

        # =====================================================================
        # EXTRACT CONVERGED OCP TRAJECTORY
        # =====================================================================

        try:

            (
                ocp_time_grid,
                ocp_x,
                ocp_y,
                ocp_theta,
            ) = (
                extract_ocp_pose_trajectory(
                    ocp_result
                )
            )

        except Exception as error:

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                f"OCP trajectory extraction failed: "
                f"{error}"
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    True,

                "ocp_success":
                    True,

                "metrics_success":
                    False,

                "error_stage":
                    "ocp_trajectory_extraction",

                "error":
                    str(
                        error
                    ),

                "ocp_time":
                    float(
                        ocp_result[
                            "time"
                        ]
                    ),

                "ocp_solve_time":
                    float(
                        ocp_result[
                            "solve_time"
                        ]
                    ),
            })

            results.append(
                failed_result
            )

            continue

        # =====================================================================
        # TIME-DISCREPANCY METRICS
        # =====================================================================

        signed_time_error = float(
            ocp_result[
                "time"
            ]
            - best_time
        )

        absolute_time_error = float(
            abs(
                signed_time_error
            )
        )

        relative_time_error = float(
            absolute_time_error
            / best_time
        )

        relative_time_error_percent = float(
            100.0
            * relative_time_error
        )

        # =====================================================================
        # HAUSDORFF DISTANCE
        # =====================================================================

        try:

            (
                analytical_x_dense,
                analytical_y_dense,
            ) = (
                extract_dense_analytical_planar_path(
                    best_trajectory,
                    samples_per_spatial_primitive=(
                        ANALYTICAL_GEOMETRY_SAMPLES_PER_PRIMITIVE
                    ),
                )
            )

            (
                analytical_x_hausdorff,
                analytical_y_hausdorff,
            ) = (
                resample_planar_path_by_arclength(
                    analytical_x_dense,
                    analytical_y_dense,
                    HAUSDORFF_N_SAMPLES,
                )
            )

            (
                ocp_x_hausdorff,
                ocp_y_hausdorff,
            ) = (
                resample_planar_path_by_arclength(
                    ocp_x,
                    ocp_y,
                    HAUSDORFF_N_SAMPLES,
                )
            )

            hausdorff = (
                symmetric_hausdorff_distance(
                    analytical_x_hausdorff,
                    analytical_y_hausdorff,
                    ocp_x_hausdorff,
                    ocp_y_hausdorff,
                )
            )

        except Exception as error:

            print(
                f"\nCase "
                f"{case_id:04d}/{n_cases}: "
                f"Hausdorff computation failed: "
                f"{error}"
            )

            failed_result = dict(
                common_information
            )

            failed_result.update({
                "success":
                    False,

                "analytical_success":
                    True,

                "ocp_success":
                    True,

                "metrics_success":
                    False,

                "error_stage":
                    "hausdorff",

                "error":
                    str(
                        error
                    ),

                "best_analytical_name":
                    best_name,

                "best_analytical_time":
                    float(
                        best_time
                    ),

                "ocp_time":
                    float(
                        ocp_result[
                            "time"
                        ]
                    ),

                "time_difference_signed":
                    signed_time_error,

                "time_difference_absolute":
                    absolute_time_error,

                "time_difference_relative":
                    relative_time_error,

                "time_difference_relative_percent":
                    relative_time_error_percent,
            })

            results.append(
                failed_result
            )

            continue

        # =====================================================================
        # SUCCESSFUL CASE
        # =====================================================================

        print_case_result(
            case_id=
                case_id,

            n_cases=
                n_cases,

            theta0=
                theta0,

            thetaf=
                thetaf,

            best_name=
                best_name,

            best_time=
                best_time,

            ocp_result=
                ocp_result,

            best_trajectory=
                best_trajectory,

            tst_information=
                tst_information,

            absolute_time_error=
                absolute_time_error,

            relative_time_error_percent=
                relative_time_error_percent,

            hausdorff_distance=
                hausdorff[
                    "symmetric"
                ],

            x0=
                x0,

            y0=
                y0,

            xf=
                xf,

            yf=
                yf,
        )

        case_result = dict(
            common_information
        )

        case_result.update({
            "success":
                True,

            "analytical_success":
                True,

            "ocp_success":
                True,

            "metrics_success":
                True,

            # -------------------------------------------------------------
            # TST INITIALIZATION
            # -------------------------------------------------------------

            "ocp_initial_guess":
                "TST",

            "tst_T_guess":
                float(
                    tst_T_guess
                ),

            "tst_segment_heading_rad":
                float(
                    tst_information[
                        "segment_heading_rad"
                    ]
                ),

            "tst_segment_heading_deg":
                float(
                    tst_information[
                        "segment_heading_deg"
                    ]
                ),

            "tst_distance":
                float(
                    tst_information[
                        "distance"
                    ]
                ),

            "tst_initial_turn_angle_rad":
                float(
                    tst_information[
                        "initial_turn_angle_rad"
                    ]
                ),

            "tst_final_turn_angle_rad":
                float(
                    tst_information[
                        "final_turn_angle_rad"
                    ]
                ),

            # -------------------------------------------------------------
            # ANALYTICAL BENCHMARK
            # -------------------------------------------------------------

            "best_analytical_name":
                best_name,

            "best_analytical_time":
                float(
                    best_time
                ),

            "analytical_solve_time":
                float(
                    analytical_solve_time
                ),

            "best_analytical_primitives":
                best_analytical_primitives,

            "arc_segment_ratio_1":
                ratios[
                    "r1"
                ],

            "arc_segment_ratio_2":
                ratios[
                    "r2"
                ],

            "arc1_length":
                ratios[
                    "arc1_length"
                ],

            "segment_length":
                ratios[
                    "segment_length"
                ],

            "arc2_length":
                ratios[
                    "arc2_length"
                ],

            # -------------------------------------------------------------
            # OCP RESULT
            # -------------------------------------------------------------

            "ocp_time":
                float(
                    ocp_result[
                        "time"
                    ]
                ),

            "ocp_solve_time":
                float(
                    ocp_result[
                        "solve_time"
                    ]
                ),

            "ocp_sequence":
                list(
                    ocp_result[
                        "sequence"
                    ]
                ),

            "solver_error":
                ocp_result.get(
                    "solver_error"
                ),

            # -------------------------------------------------------------
            # TIME-DISCREPANCY METRICS
            # -------------------------------------------------------------

            # Kept for compatibility with older analysis scripts.
            # This is the SIGNED discrepancy T_ocp - T_analytical.
            "time_difference":
                signed_time_error,

            "time_difference_signed":
                signed_time_error,

            "time_difference_absolute":
                absolute_time_error,

            "time_difference_relative":
                relative_time_error,

            "time_difference_relative_percent":
                relative_time_error_percent,

            # -------------------------------------------------------------
            # GEOMETRIC DISCREPANCY
            # -------------------------------------------------------------

            "hausdorff_distance":
                float(
                    hausdorff[
                        "symmetric"
                    ]
                ),

            "hausdorff_analytical_to_ocp":
                float(
                    hausdorff[
                        "a_to_b"
                    ]
                ),

            "hausdorff_ocp_to_analytical":
                float(
                    hausdorff[
                        "b_to_a"
                    ]
                ),

            # -------------------------------------------------------------
            # SAVED OCP TRAJECTORY
            # -------------------------------------------------------------

            "ocp_time_grid": [
                float(
                    value
                )
                for value
                in ocp_time_grid
            ],

            "ocp_x": [
                float(
                    value
                )
                for value
                in ocp_x
            ],

            "ocp_y": [
                float(
                    value
                )
                for value
                in ocp_y
            ],

            "ocp_theta": [
                float(
                    value
                )
                for value
                in ocp_theta
            ],
        })

        results.append(
            case_result
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    total_simulation_time = (
        time.perf_counter()
        - simulation_start_time
    )

    n_successful_cases = sum(
        bool(
            result.get(
                "success",
                False,
            )
        )
        for result
        in results
    )

    n_ocp_successful_cases = sum(
        bool(
            result.get(
                "ocp_success",
                False,
            )
        )
        for result
        in results
    )

    n_metrics_successful_cases = sum(
        bool(
            result.get(
                "metrics_success",
                False,
            )
        )
        for result
        in results
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TST-INITIALIZED SWEEP COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"Total cases executed       : "
        f"{n_cases}"
    )

    print(
        f"OCP converged cases        : "
        f"{n_ocp_successful_cases}"
    )

    print(
        f"Cases with thesis metrics  : "
        f"{n_metrics_successful_cases}"
    )

    print(
        f"Total simulation time      : "
        f"{total_simulation_time:.3f} s"
    )

    metadata.update({
        "N":
            int(
                N
            ),

        "M":
            int(
                M
            ),

        "experiment_purpose":
            (
                "OCP discretization study with "
                "independent TST initialization"
            ),

        "analytical_role":
            (
                "benchmark only; not used for "
                "OCP initialization"
            ),

        "ocp_initial_guess":
            "TST",

        "tst_initial_guess_definition":
            (
                "shortest turn to direct "
                "position-to-position segment heading, "
                "straight segment at v_max, "
                "shortest turn to final heading"
            ),

        "initial_guess_samples_per_maneuver":
            int(
                INITIAL_GUESS_SAMPLES_PER_MANEUVER
            ),

        "quick_test":
            bool(
                QUICK_TEST
            ),

        "n_cases_executed":
            int(
                n_cases
            ),

        "n_successful_cases":
            int(
                n_successful_cases
            ),

        "n_ocp_successful_cases":
            int(
                n_ocp_successful_cases
            ),

        "n_metrics_successful_cases":
            int(
                n_metrics_successful_cases
            ),

        "time_error_reference":
            "analytical minimum traversal time",

        "time_error_definition":
            (
                "signed error = T_ocp - T_analytical; "
                "absolute and relative errors are also saved"
            ),

        "hausdorff_definition":
            (
                "symmetric discrete Hausdorff distance "
                "between densely sampled planar analytical "
                "and OCP paths"
            ),

        "hausdorff_coordinates":
            "x-y",

        "hausdorff_units":
            "m",

        "hausdorff_resampling":
            "uniform planar arclength",

        "hausdorff_n_samples":
            int(
                HAUSDORFF_N_SAMPLES
            ),

        "analytical_geometry_samples_per_spatial_primitive":
            int(
                ANALYTICAL_GEOMETRY_SAMPLES_PER_PRIMITIVE
            ),

        "ocp_pose_trajectory_saved":
            True,

        "saved_ocp_fields": [
            "ocp_time_grid",
            "ocp_x",
            "ocp_y",
            "ocp_theta",
        ],

        "total_simulation_time":
            float(
                total_simulation_time
            ),
    })

    results_filename = (
        f"{metadata['sweep_name']}"
        f"_N{N}_M{M}"
    )

    if QUICK_TEST:

        results_filename += (
            f"_TEST{n_cases}"
        )

    results_filename += (
        ".json"
    )

    current_dir = (
        Path(
            __file__
        )
        .resolve()
        .parent
    )

    results_dir = (
        current_dir
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = (
        results_dir
        / results_filename
    )

    output = {
        "metadata":
            metadata,

        "results":
            results,
    }

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        f"Results saved to: "
        f"{save_path}"
    )