import warnings
from .vehicle import Unicycle, Bicycle, Bicycle_Acceleration
from .geometry import IntermediateCircle, IntermediateCirclesSequence
from .helpers.poses import compute_end_pose, compute_start_pose, pose_from_shrunken_corridor_relative_frame
from .helpers.inputs_check import check_standing_assumptions, check_core_assumptions, check_inputs_analytical_planner, compute_minimum_widths, check_position_out_of_circles_assumption
from .helpers.helper_functions import Timer
from .helpers.plot_helpers import plot_planner_inputs
from .helpers.corridor_geometry import shrink_corridor_list
from .helpers.trajectory_unicycle import compute_trajectory_unicycle_two_corridors, compute_trajectory_unicycle_multiple_corridors_optimal
from .helpers.axis_aligned_int_circle_sequence import build_intermediate_circles_sequence

from .helpers.trajectory_bicycle import compute_trajectory_bicycle_multiple_corridors_optimal, compute_trajectory_bicycle_two_corridors_optimal
from .helpers.trajectory_unicycle_core import compute_trajectory_unicycle_multiple_corridors_core, compute_trajectory_unicycle_two_corridors_core

from .helpers.intermediate_circles_choice import (
    not_ambiguous_circle_choices,
    create_intermediate_circle_choice_sequence,
)


class MotionPlanner:
    """Analytically compute trajectories within corridors for different kinds of vehicles.
    """

    def __init__(self,
                 vehicle,
                 corridor_list,
                 start_pose = None,
                 end_pose = None,
                 relative_start_pose = None,
                 relative_end_pose = None,
                 waypoints = None,
                 assumptions = "core"):
        """Constructor.

        :param vehicle: Vehicle model used for planning.
        :type vehicle: Vehicle
        :param corridor_list: Ordered sequence of corridors.
        :type corridor_list: list[CorridorWorld]
        :param start_pose: Initial pose included in the first corridor. If not provided, it is computed automatically
            at a distance of 1.3 times the vehicle length from the midpoint of the back edge of the first corridor.
        :type start_pose: list[float] | numpy.ndarray
        :param end_pose: Goal pose included in the last corridor. If not provided, it is computed automatically
            at a distance of 1.3 times the vehicle length from the midpoint of the front edge of the last corridor.
        :type end_pose: list[float] | numpy.ndarray
        :param waypoints: List of waypoints to make the trajectory pass through (used only if provided).
        :type waypoints: list[list[float]] | numpy.ndarray
        :param assumptions: Which set of assumptions to check for the analytical planner. Options are "core" (default) and "standing".
        :type assumptions: str
        """
        # Core inputs
        self.vehicle = vehicle
        self.corridor_list = corridor_list
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.relative_start_pose = relative_start_pose
        self.relative_end_pose = relative_end_pose
        self.waypoints = waypoints
        self.assumptions = assumptions
        # Validate mutually exclusive inputs
        self._validate_pose_inputs()

        # Precompute shrunken corridors used by the analytical planner
        safety_margin = 0.5 * self.vehicle.width
        self.shrunken_corridor_list = shrink_corridor_list(
            self.corridor_list,
            safety_margin,
        )

        # Resolve absolute start and end poses
        self._resolve_start_and_end_poses()

        # Build derived analytical-planner state
        self._refresh_analytical_planner_state()

        # Initialize computation stats
        self.comp_time_analytical_sol = 0.0
  
    def update(self, **kwargs):
        """Update planner attributes and refresh derived state.

        Keyword arguments correspond to existing attributes of the object.
        Unknown keys are ignored with a warning.
        """
        valid_updates = {}
        changed = set()
        print("MEGA PROVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # Collect valid attribute updates, but do not apply them yet
        for key, value in kwargs.items():
            if hasattr(self, key):
                valid_updates[key] = value
                changed.add(key)
            else:
                warnings.warn(f"Ignoring unknown attribute '{key}' in update().")

        if not changed:
            return

        new_start_pose = valid_updates.get("start_pose", self.start_pose)
        new_end_pose = valid_updates.get("end_pose", self.end_pose)
        new_relative_start_pose = valid_updates.get(
            "relative_start_pose", self.relative_start_pose
        )
        new_relative_end_pose = valid_updates.get(
            "relative_end_pose", self.relative_end_pose
        )

        # If one pose representation is updated, it overrides the other
        if "start_pose" in valid_updates and "relative_start_pose" not in valid_updates:
            new_relative_start_pose = None
        elif "relative_start_pose" in valid_updates and "start_pose" not in valid_updates:
            new_start_pose = None

        if "end_pose" in valid_updates and "relative_end_pose" not in valid_updates:
            new_relative_end_pose = None
        elif "relative_end_pose" in valid_updates and "end_pose" not in valid_updates:
            new_end_pose = None

        if new_start_pose is not None and new_relative_start_pose is not None:
            raise ValueError("Provide either start_pose or relative_start_pose, not both.")

        if new_end_pose is not None and new_relative_end_pose is not None:
            raise ValueError("Provide either end_pose or relative_end_pose, not both.")

        # Apply validated updates
        for key, value in valid_updates.items():
            setattr(self, key, value)

        # Apply pose-representation override to the stored state
        if "start_pose" in valid_updates and "relative_start_pose" not in valid_updates:
            self.relative_start_pose = None
        elif "relative_start_pose" in valid_updates and "start_pose" not in valid_updates:
            self.start_pose = None

        if "end_pose" in valid_updates and "relative_end_pose" not in valid_updates:
            self.relative_end_pose = None
        elif "relative_end_pose" in valid_updates and "end_pose" not in valid_updates:
            self.end_pose = None

        corridors_changed = "corridor_list" in changed
        vehicle_changed = "vehicle" in changed
        relative_start_changed = "relative_start_pose" in changed
        relative_end_changed = "relative_end_pose" in changed
        start_changed = "start_pose" in changed
        end_changed = "end_pose" in changed
        waypoints_changed = "waypoints" in changed

        # Recompute shrunken corridors when corridor geometry or vehicle geometry changed
        if corridors_changed or vehicle_changed:
            safety_margin = 0.5 * self.vehicle.width
            self.shrunken_corridor_list = shrink_corridor_list(
                self.corridor_list,
                safety_margin,
            )

        # If relative poses are the source of truth, force recomputation of absolute poses
        if relative_start_changed or (corridors_changed and self.relative_start_pose is not None):
            self.start_pose = None

        if relative_end_changed or (corridors_changed and self.relative_end_pose is not None):
            self.end_pose = None

        # If a vehicle change affects default pose placement, allow recomputation
        if vehicle_changed and self.relative_start_pose is None and "start_pose" not in changed:
            self.start_pose = None

        if vehicle_changed and self.relative_end_pose is None and "end_pose" not in changed:
            self.end_pose = None

        # Resolve absolute poses again when needed
        if (
            start_changed
            or end_changed
            or relative_start_changed
            or relative_end_changed
            or corridors_changed
            or vehicle_changed
        ):
            self._resolve_start_and_end_poses()

        self.print_planner_inputs()

        # Refresh analytical planner state when relevant inputs changed
        if (
            vehicle_changed
            or corridors_changed
            or start_changed
            or end_changed
            or relative_start_changed
            or relative_end_changed
            or waypoints_changed
        ):
            self._refresh_analytical_planner_state()

    def plot_planner_inputs(
        self,
        figure=None,
        plot_intermediate_circles=False,
        plot_shrunken_corridors=True,
    ):
        """
        Plot the corridors and start/end poses of the motion planner.

        :param figure: Matplotlib figure to plot on. If ``None``, a new figure is created.
        :type figure: matplotlib.figure.Figure, optional
        :returns: The resulting matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        return plot_planner_inputs(
            self,
            figure,
            plot_intermediate_circles=plot_intermediate_circles,
            plot_shrunken_corridors=plot_shrunken_corridors,
        )
    
    def print_planner_inputs(self):
        """
        Return a formatted string representation of the planner inputs.
        """
        # Build corridor definitions
        corridor_defs = [
            (
                f"corridor{i} = Corridor.CorridorWorld("
                f"{c.width}, {c.height}, {c.center}, {c.tilt})"
            )
            for i, c in enumerate(self.corridor_list, start=1)
        ]
        corridors_str = "\n".join(corridor_defs)

        # Build corridor list line
        corridor_names = ", ".join(
            f"corridor{i}"
            for i in range(1, len(self.corridor_list) + 1)
        )
        corridor_list_str = f"corridor_list = [{corridor_names}]"

        # Format start/end poses
        start_pose_str = f"start_pose = [{', '.join(map(str, self.start_pose))}]"
        end_pose_str = f"end_pose = [{', '.join(map(str, self.end_pose))}]"

        # Format vehicle definition
        v = self.vehicle

        if isinstance(v, Unicycle):
            vehicle_str = (
                "vehicle = Unicycle("
                f"width={v.width}, length={v.length}, v_max={v.v_max}, "
                f"v_min=0, omega_max={v.omega_max}, omega_min={-v.omega_max})"
            )

        elif isinstance(v, Bicycle):
            vehicle_str = (
                "vehicle = Bicycle([0, 0, 0], "
                f"width={v.width}, length={v.length}, wheelbase={v.wheelbase}, "
                f"v_max={v.v_max}, v_min={v.v_min}, "
                f"delta_max={v.delta_max}, delta_min={v.delta_min})"
            )

        # Join all parts with newlines
        return "\n".join(
            [
                corridors_str,
                corridor_list_str,
                start_pose_str,
                end_pose_str,
                vehicle_str,
            ]
        )

    def compute_trajectory_analytical(self, obstacle_center=None, obstacle_radius=None):
        """
        Compute the time-optimal trajectory from the start pose to the end pose.

        Uses the analytical planner to generate a trajectory constrained within
        the specified corridors or waypoints.
        """
        corridors = self.corridor_list.copy()
        vehicle = self.vehicle.copy()
        start_pose = self.start_pose.copy()
        end_pose = self.end_pose.copy()
        waypoints = (
            self.waypoints.copy()
            if self.waypoints is not None
            else None
        )

        if isinstance(vehicle, Unicycle) and waypoints is None:
            return self._compute_unicycle_trajectory(
                corridors,
                vehicle,
                start_pose,
                end_pose,
            )

        if isinstance(vehicle, Bicycle) and waypoints is None:
            return self._compute_bicycle_trajectory(
                corridors,
                vehicle,
                start_pose,
                end_pose,
            )

        raise NotImplementedError(
            "Analytical trajectory computation is not implemented for waypoints."
        )
    
    def __str__(self):
        return 'MotionPlanner object'
    
    def _validate_pose_inputs(self):
        if self.start_pose is not None and self.relative_start_pose is not None:
            raise ValueError("Provide either start_pose or relative_start_pose, not both.")

        if self.end_pose is not None and self.relative_end_pose is not None:
            raise ValueError("Provide either end_pose or relative_end_pose, not both.")
        
    def _resolve_start_and_end_poses(self):
        default_pose_margin = 1.3 * self.vehicle.length
        if self.start_pose is None:
            if self.relative_start_pose is None:
                margin = default_pose_margin
                self.start_pose = compute_start_pose(
                    self.corridor_list[0],
                    self.vehicle,
                    margin,
                )
            else:
                self.start_pose = pose_from_shrunken_corridor_relative_frame(
                    self.relative_start_pose,
                    self.shrunken_corridor_list[0],
                )

        if self.end_pose is None:
            if self.relative_end_pose is None:
                margin = default_pose_margin
                self.end_pose = compute_end_pose(
                    self.corridor_list[-1],
                    self.vehicle,
                    margin,
                )
            else:
                self.end_pose = pose_from_shrunken_corridor_relative_frame(
                    self.relative_end_pose,
                    self.shrunken_corridor_list[-1],
                )

    def _refresh_analytical_planner_state(self):
        if not isinstance(self.vehicle, (Unicycle, Bicycle, Bicycle_Acceleration)):
            return

        (
            self.inputs_check,
            self.warn_msgs_core_assumptions,   
        ) = check_core_assumptions(self)

        # Journal paper version
        if self.assumptions == "standing":
            (
                self.min_corridor_widths,
                self.s_max_circles
            ) = compute_minimum_widths(self)

            (
                self.inputs_check,
                wrn_msgs_standing_assumptions,   
                self.intermediate_circles
            ) = check_standing_assumptions(self)

            self.intermediate_circles_choice_sequence = not_ambiguous_circle_choices(
                self.intermediate_circles
            )

            self.warn_msgs = self.warn_msgs_core_assumptions + wrn_msgs_standing_assumptions

        # Extension version
        elif self.assumptions == "core":
            # self.intermediate_circles_choice_sequence = create_intermediate_circle_choice_sequence(
            # self.corridor_list,
            # self.vehicle,
            # self.start_pose,
            # self.end_pose,
            #     )
            
            self.intermediate_circles_sequence =build_intermediate_circles_sequence(
                self.corridor_list,
                self.vehicle,
                self.start_pose,
                self.end_pose,
            )
            
        (
            self.position_out_of_circles_assumption_check,
            warn_msgs_position_out_of_circles_assumption,   
            self.inside_first_circle,
            self.inside_last_circle
        ) = check_position_out_of_circles_assumption(self)

        self.exit_trajectory_start = []
        self.exit_trajectory_end = []

        # if self.position_out_of_circles_assumption_check is False:
        #     (
        #         self.exit_trajectory_start,
        #         self.exit_trajectory_end
        #     ) = compute_circle_exit_trajectory(
        #         self.inside_first_circle,
        #         self.inside_last_circle,
        #         self.vehicle,
        #         self.start_pose, 
        #         self.end_pose,
        #         self.intermediate_circles_choice_sequence)
            
            # if self.inside_first_circle and self.inside_last_circle:
            #     new_start_pose = self.exit_trajectory_start[-1].end_pose 
            #     new_end_pose = self.exit_trajectory_end[0].start_pose
            #     # self.update(start_pose = new_start_pose, end_pose = new_end_pose)
            #     self.start_pose = new_start_pose
            #     self.end_pose = new_end_pose
            # elif self.inside_first_circle:
            #     new_start_pose = self.exit_trajectory_start[-1].end_pose 
            #     self.start_pose = new_start_pose
            #     # self.update(start_pose = new_start_pose)
            # elif self.inside_last_circle:
            #     new_end_pose = self.exit_trajectory_end[0].start_pose
            #     self.end_pose = new_end_pose
            #     # self.update(end_pose = new_end_pose)

        if isinstance(self.vehicle, Bicycle):
            self.inputs_check = self.inputs_check and self.position_out_of_circles_assumption_check

        self.warn_msgs = self.warn_msgs_core_assumptions + warn_msgs_position_out_of_circles_assumption

        if not self.inputs_check:
            warnings.warn(
                "Invalid inputs for analytical motion planner.\n"
                "Check warning messages for details:\n"
                + "\n".join(self.warn_msgs)
            )

        self.vehicle.update(state=self.start_pose)


    def _compute_unicycle_trajectory(
        self,
        corridors,
        vehicle,
        start_pose,
        end_pose,
    ):
        """
        Compute an analytical trajectory for a unicycle model.
        """
        if not self.inputs_check:
            raise ValueError(
                "Invalid inputs for analytical motion planner.\n"
                "Check warning messages for details:\n"
                + "\n".join(self.warn_msgs)
            )

        if self.assumptions == "standing":
            if len(corridors) == 2:
                with Timer() as timer:
                    trajectory, intermediate_circles = compute_trajectory_unicycle_two_corridors(
                        corridors[0],
                        corridors[1],
                        start_pose,
                        end_pose,
                        vehicle,
                        self.intermediate_circles,
                    )
            elif len(corridors) > 2:
                with Timer() as timer:
                    trajectory, intermediate_circles = compute_trajectory_unicycle_multiple_corridors_optimal(
                        corridors,
                        self.shrunken_corridor_list,
                        start_pose,
                        end_pose,
                        vehicle,
                        self.intermediate_circles,
                    )
            elif len(corridors) == 1:
                raise NotImplementedError(
                    "Analytical unicycle trajectory computation is not implemented for one corridor."
                )
            self.comp_time_analytical_sol = timer()
            if isinstance(intermediate_circles, IntermediateCirclesSequence):
                self.intermediate_circles = intermediate_circles
            else:
                self.intermediate_circles = IntermediateCirclesSequence([intermediate_circles])

        elif self.assumptions == "core":
            if len(corridors) == 2:
                with Timer() as timer:
                    trajectory = compute_trajectory_unicycle_two_corridors_core(
                        corridors[0],
                        corridors[1],
                        start_pose,
                        end_pose,
                        vehicle,
                        self.intermediate_circles_sequence,
                        self.exit_trajectory_start,
                        self.exit_trajectory_end
                    )
            elif len(corridors) > 2:
                with Timer() as timer:
                    trajectory = compute_trajectory_unicycle_multiple_corridors_core(
                    corridors,
                    start_pose,
                    end_pose,
                    vehicle,
                    self.intermediate_circles_sequence,
                    self.inside_first_circle,
                    self.inside_last_circle,
                )
            elif len(corridors) == 1:
                raise NotImplementedError(
                    "Analytical unicycle trajectory computation is not implemented for one corridor."
                )
            self.comp_time_analytical_sol = timer()


        
        return trajectory
    

    def _compute_bicycle_trajectory(
        self,
        corridors,
        vehicle,
        start_pose,
        end_pose,
    ):
        """
        Compute an analytical trajectory for a bicycle model.
        """
        if not self.inputs_check:
            raise ValueError(
                "Invalid inputs for analytical motion planner.\n"
                "Check warning messages for details:\n"
                + "\n".join(self.warn_msgs)
            )

        if len(corridors) == 2:
            with Timer() as timer:
                trajectory = compute_trajectory_bicycle_two_corridors_optimal(
                    corridors[0],
                    corridors[1],
                    start_pose,
                    end_pose,
                    vehicle,
                    self.intermediate_circles_sequence,
                )
        elif len(corridors) > 2:
            with Timer() as timer:
                trajectory = compute_trajectory_bicycle_multiple_corridors_optimal(
                    corridors,
                    start_pose,
                    end_pose,
                    vehicle,
                    self.intermediate_circles_sequence,
                )
        elif len(corridors) == 1:
            raise NotImplementedError(
                "Analytical bicycle trajectory computation is not implemented for one corridor."
            )
        
        self.comp_time_analytical_sol = timer()
        return trajectory