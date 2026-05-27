import numpy as np
import warnings
from math import atan2, sqrt, asin, cos, sin, pi
from .helpers.geometry_operations import (
    wrapPositiveAngle,
    efficient_sign,
    compute_angular_difference_with_turn_direction,
    compute_distance_two_points,
)
from .helpers.plot_helpers import (
    #plot_vehicle, # TO-DO
    plot_path_all_trajectories,
    plot_circle,
)


class Trajectory: 
    """Base class for different types of trajectories.
    """
    def __init__(self):
        """Constructor. Trajectory class is the parent of all types of trajectories.
        """
        pass

    def __str__(self):
        return 'Trajectory object'

    def plot_path(
        self,
        figure=None,
        color="k",
        linestyle="solid",
        linewidth=2.5,
        label=None,
    ):
        """
        Plot the path of the current trajectory.

        This is a wrapper for :meth:`plot_path_all_trajectories` that
        plots a single path using the provided visual style parameters.

        :param figure: Figure to plot on. If ``None``, a new one is created.
        :type figure: matplotlib.figure.Figure or None
        :param color: Line color (default: ``'k'`` for black).
        :type color: str
        :param linestyle: Line style (default: ``'solid'``).
        :type linestyle: str
        :param linewidth: Line width (default: ``2.5``).
        :type linewidth: float
        :param label: Label for the plotted line.
        :type label: str or None
        :return: The matplotlib figure used for plotting.
        :rtype: matplotlib.figure.Figure
        """
        return plot_path_all_trajectories(
            self,
            figure=figure,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    
    # def plot_vehicle(self, figure, x, y, theta, w_left, w_right, l_front, l_back, color='b'): ## TO_DO
        # veh = vehic_to_plot(figure, x, y, theta, w_left, w_right, l_front, l_back, color='b')
        # return veh


class UnicycleTrajectory(Trajectory):
    """
    Trajectory segment for a unicycle-type vehicle.

    Represents the evolution of position, heading, and control inputs
    (forward and angular velocity) along a given time grid.
    """

    def __init__(
        self,
        time_grid,
        x,
        y,
        theta,
        forward_velocity,
        angular_velocity,
    ):
        """
        Initialize a unicycle trajectory instance.

        :param time_grid: Time grid of the trajectory.
        :type time_grid: list[float] or numpy.ndarray
        :param x: x-coordinates along the trajectory.
        :type x: list[float] or numpy.ndarray
        :param y: y-coordinates along the trajectory.
        :type y: list[float] or numpy.ndarray
        :param theta: Heading angles along the trajectory.
        :type theta: list[float] or numpy.ndarray
        :param forward_velocity: Forward velocity profile over the time grid.
        :type forward_velocity: list[float] or numpy.ndarray
        :param angular_velocity: Angular velocity profile over the time grid.
        :type angular_velocity: list[float] or numpy.ndarray
        """
        # Time attributes
        self.time_grid = time_grid
        self.t0 = time_grid[0]
        self.tf = time_grid[-1]

        # State attributes
        self.x0, self.y0, self.theta0 = x[0], y[0], theta[0]
        self.xf, self.yf, self.thetaf = x[-1], y[-1], theta[-1]
        self.x_coordinates = x
        self.y_coordinates = y
        self.theta = theta
        self.path_coordinates = np.column_stack([x, y])

        # Control attributes
        self.forward_velocity = forward_velocity
        self.angular_velocity = angular_velocity

    def __str__(self):
        return 'UnicycleTrajectory object'


class UnicycleTrajectoryOptimal(UnicycleTrajectory):
    """
    Optimal trajectory for a unicycle-type vehicle.

    Extends :class:`UnicycleTrajectory` with additional data from
    an optimal control problem (OCP) solution, including control
    profiles, initial guesses, and solver metadata.
    """

    def __init__(
        self,
        time_grid,
        x,
        y,
        theta,
        v,
        omega,
        control_time_grid,
        control_x,
        control_y,
        control_theta,
        control_forward_velocity,
        control_angular_velocity,
        total_time,
        x_initial_guess=None,
        y_initial_guess=None,
        theta_initial_guess=None,
        v_initial_guess=None,
        omega_initial_guess=None,
        time_grid_initial_guess=None,
        sampler=None,
        gist=None,
    ):
        """
        Initialize an optimal unicycle trajectory object.

        :param time_grid: Time grid of the state trajectory.
        :type time_grid: list[float] or numpy.ndarray
        :param x: x-coordinates of the trajectory.
        :type x: list[float] or numpy.ndarray
        :param y: y-coordinates of the trajectory.
        :type y: list[float] or numpy.ndarray
        :param theta: Heading angles along the trajectory.
        :type theta: list[float] or numpy.ndarray
        :param v: Forward velocity along the trajectory.
        :type v: list[float] or numpy.ndarray
        :param omega: Angular velocity along the trajectory.
        :type omega: list[float] or numpy.ndarray
        :param control_time_grid: Time grid for control inputs.
        :type control_time_grid: list[float] or numpy.ndarray
        :param control_x: x-coordinates of control points.
        :type control_x: list[float] or numpy.ndarray
        :param control_y: y-coordinates of control points.
        :type control_y: list[float] or numpy.ndarray
        :param control_theta: Heading angles at control points.
        :type control_theta: list[float] or numpy.ndarray
        :param control_forward_velocity: Forward velocity controls.
        :type control_forward_velocity: list[float] or numpy.ndarray
        :param control_angular_velocity: Angular velocity controls.
        :type control_angular_velocity: list[float] or numpy.ndarray
        :param total_time: Total duration of the trajectory (s).
        :type total_time: float
        :param x_initial_guess: Initial guess for x positions (optional).
        :type x_initial_guess: list[float] or numpy.ndarray
        :param y_initial_guess: Initial guess for y positions (optional).
        :type y_initial_guess: list[float] or numpy.ndarray
        :param theta_initial_guess: Initial guess for heading angles.
        :type theta_initial_guess: list[float] or numpy.ndarray
        :param v_initial_guess: Initial guess for forward velocity.
        :type v_initial_guess: list[float] or numpy.ndarray
        :param omega_initial_guess: Initial guess for angular velocity.
        :type omega_initial_guess: list[float] or numpy.ndarray
        :param time_grid_initial_guess: Initial guess for the time grid.
        :type time_grid_initial_guess: list[float] or numpy.ndarray
        :param sampler: Sampling or interpolation object (optional).
        :type sampler: object or None
        :param gist: Additional solver or metadata information.
        :type gist: object or None
        """
        # Initialize base UnicycleTrajectory
        super().__init__(time_grid, x, y, theta, v, omega)

        # Control-related attributes
        self.control_time_grid = control_time_grid
        self.control_x = control_x
        self.control_y = control_y
        self.control_theta = control_theta
        self.control_v = control_forward_velocity
        self.control_omega = control_angular_velocity

        # Total time
        self.total_time = total_time

        # Initial guess attributes
        self.x_initial_guess = x_initial_guess
        self.y_initial_guess = y_initial_guess
        self.theta_initial_guess = theta_initial_guess
        self.v_initial_guess = v_initial_guess
        self.omega_initial_guess = omega_initial_guess
        self.time_grid_initial_guess = time_grid_initial_guess

        # Additional solver / metadata
        self.sampler = sampler
        self.gist = gist

    def __str__(self):
        return "UnicycleTrajectoryOptimal object"

    # def plot_control_points(self, figure):
    #     import matplotlib.pyplot as plt
    #     if figure is None: figure = plt.figure()
    #     plt.step(self.control_x, self.control_y, 'r.',  label = 'control points')
    #     plt.legend()
    #     return figure
        
    # def plot_initial_guess(self, figure):
    #     import matplotlib.pyplot as plt

    #     if figure is None: figure = plt.figure()
    #     plt.plot(self.x_initial_guess, self.y_initial_guess, 'b--', label = 'initial guess')
    #     plt.legend()
    #     return figure

class CurvilinearArcUnicycle(Trajectory):
    """ Curvilinear arc trajectory segment for a unicycle-type vehicle. 
    """

    def __init__(self, xc, yc,
                 x0, y0, theta0,
                 xf, yf, thetaf,
                 radius, turn_direction,
                 v, omega, unicycle,
                 t0 = 0, samples_number = 50):
        # Initialize attributes
        self.xc     = xc
        self.yc     = yc
        self.x0     = x0
        self.y0     = y0
        self.theta0 = theta0
        self.thetaf = thetaf
        self.xf     = xf
        self.yf     = yf
        self.radius = radius
        self.curvature = 1/radius
        self.turn_direction = turn_direction
        self.v  = v
        self.omega  = omega
        self.unicycle   = unicycle
        self.t0     = t0
        self.samples_number = samples_number
        self.label = 'arc'

        # Compute derived attributes
        self.start_position = [self.x0, self.y0]
        self.end_position   = [self.xf, self.yf]
        self.start_pose     = [self.x0, self.y0, self.theta0]
        self.end_pose       = [self.xf, self.yf, self.thetaf]
        self.circle_center  = [self.xc, self.yc]

        self.wrapped_theta0 = wrapPositiveAngle(theta0)
        self.wrapped_thetaf = wrapPositiveAngle(thetaf)
        angle = self.wrapped_thetaf - self.wrapped_theta0
        self.delta_angle    = atan2(sin(angle), cos(angle))
        self.chord  = compute_distance_two_points(self.start_position,
                                                  self.end_position)
        if abs(self.delta_angle) < 1e-6:
            self.iota = 0

        else:
            asin_arg = (self.chord * 0.5) / self.radius
            asin_arg = max(-1.0, min(1.0, asin_arg))

            if efficient_sign(self.delta_angle) == self.turn_direction:
                self.iota = 2 * asin(asin_arg)
            else:
                self.iota = 2 * pi - 2 * asin(asin_arg)

        self.epsilon    = wrapPositiveAngle(atan2((self.y0 - self.yc),
                                                  (self.x0 - self.xc)))
        angles = np.linspace(
        self.epsilon,
        self.epsilon + self.turn_direction * self.iota,
        self.samples_number,
        )
        x = self.xc + self.radius * np.cos(angles)
        y = self.yc + self.radius * np.sin(angles)

        self.path_coordinates = np.column_stack((x, y))

        # Constant profiles for velocity and angular velocity
        self.forward_velocity = np.full(self.samples_number, self.v)
        self.angular_velocity = np.full(self.samples_number, self.omega)

        # Heading trajectory
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number,
        )

        # Path geometry and timing
        self.path_length = self.radius * abs(self.iota)
        self.maneuver_time = abs(self.path_length / self.v)
        self.tf = self.t0 + self.maneuver_time

        # Time grid (linspace automatically includes the final point)
        self.time_grid = np.linspace(
            self.t0,
            self.tf,
            self.samples_number,
        )

    def __str__(self): 
        return 'Arc object'

    def add_time_offset(self, offset):
        """ Add a time offset to the trajectory.

        :param offset: time offset to add
        :type offset: float
        """
        self.t0 += offset
        self.tf = self.t0 + self.maneuver_time
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            self.samples_number
        ) #linspace because includes automatically the last point
    
    def change_theta0(self, new_theta0):
        """
        Update the initial heading of the trajectory.

        Keeps the same internal rotation angle while updating
        the trajectory orientation and poses accordingly.

        :param new_theta0: New initial heading angle (radians).
        :type new_theta0: float
        """
        self.theta0 = new_theta0

        # Compute new final heading while preserving the internal angle
        if efficient_sign(self.delta_angle) == self.turn_direction:
            self.thetaf = self.theta0 + self.delta_angle
        else:
            self.thetaf = (
                self.theta0
                + self.turn_direction * abs(2 * pi - abs(self.delta_angle))
            )

        # Update start/end poses
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose = [self.xf, self.yf, self.thetaf]

        # Recompute heading trajectory
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number,
        )

    def resample(self, new_samples_number):
        """
        Resample the trajectory with a new number of samples.

        Updates the path coordinates, heading, velocity, and time grid.

        :param new_samples_number: Desired number of trajectory samples.
        :type new_samples_number: int
        """
        self.samples_number = new_samples_number

        # Recompute angular samples
        angles = np.linspace(
            self.epsilon,
            self.epsilon + self.turn_direction * self.iota,
            new_samples_number,
        )

        # Recompute path coordinates
        x = self.xc + self.radius * np.cos(angles)
        y = self.yc + self.radius * np.sin(angles)
        self.path_coordinates = np.column_stack((x, y))

        # Constant velocity profiles
        self.forward_velocity = np.full(new_samples_number, self.v)
        self.angular_velocity = np.full(new_samples_number, self.omega)

        # Heading and time grids
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            new_samples_number,
        )
        self.time_grid = np.linspace(
            self.t0,
            self.tf,
            new_samples_number,
        )

    def reverse(self):
        """
        Reverse the trajectory direction.

        Flips the start and end orientations, inverts the motion direction,
        and updates angular quantities and trajectory profiles accordingly.
        """
        # Reverse direction and angular orientation
        self.theta0 += pi
        self.thetaf += pi
        self.turn_direction *= -1
        self.v *= -1
        self.omega *= -1

        # Wrap angles to [0, 2π)
        self.wrapped_theta0 = wrapPositiveAngle(self.theta0)
        self.wrapped_thetaf = wrapPositiveAngle(self.thetaf)

        # Recompute relative angle
        self.delta_angle = atan2(
            sin(self.wrapped_thetaf - self.wrapped_theta0),
            cos(self.wrapped_thetaf - self.wrapped_theta0),
        )

        # Update poses
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose = [self.xf, self.yf, self.thetaf]

        # Update kinematic profiles
        self.forward_velocity = np.full(self.samples_number, self.v)
        self.angular_velocity = np.full(self.samples_number, self.omega)
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number,
        )

    def plot_circle(self, figure=None, color="b", linestyle="dashed", linewidth=0.5):
        """
        Plot the circle associated with this trajectory/vehicle.
        """
        return plot_circle(
            self,
            figure=figure,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


class LinearSegmentUnicycle(Trajectory):
    """ Linear segment trajectory segment for a unicycle-type vehicle. 
    """

    def __init__(self, x0, y0, xf, yf, theta, v, unicycle=None, t0=0, samples_number=10):
        """ Initialize attributes of the linear segment.
        :param x0: initial x position
        :type x0: float
        :param y0: initial y position
        :type y0: float
        :param xf: final x position
        :type xf: float
        :param yf: final y position
        :type yf: float
        :param theta: heading angle
        :type theta: float
        :param v: forward velocity
        :type v: float
        :param unicycle: unicycle vehicle object
        :type unicycle: Unicycle
        :param t0: initial time
        :type t0: float
        :param samples_number: number of samples
        :type samples_number: int
        """
        # Initialize attributes
        self.x0 = x0
        self.y0 = y0
        self.xf = xf
        self.yf = yf
        self.theta  = theta
        self.v  = v
        self.unicycle   = unicycle
        self.t0 = t0
        self.samples_number = samples_number
        self.radius = -100000 # For consistency with other trajectory types
        self.curvature = 0

        # Compute derived attributes
        self.label = "segment"
        self.delta_theta = 0

        self.path_coordinates = np.column_stack((
            np.linspace(self.x0, self.xf, self.samples_number),
            np.linspace(self.y0, self.yf, self.samples_number),
        ))
        self.start_position = [self.x0, self.y0]
        self.end_position = [self.xf, self.yf]
        self.path_length = sqrt((self.xf - self.x0) ** 2 + (self.yf - self.y0) ** 2)
        self.path_heading = wrapPositiveAngle(atan2(self.yf - self.y0, self.xf - self.x0))

        self.theta0 = self.theta
        self.thetaf = self.theta
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose = [self.xf, self.yf, self.thetaf]

        self.omega = 0.0
        self.forward_velocity = np.full(self.samples_number, self.v)
        self.angular_velocity = np.zeros(self.samples_number)
        self.theta_trajectory = np.full(self.samples_number, self.theta)

        self.maneuver_time = abs(self.path_length / self.v)
        self.tf = self.t0 + self.maneuver_time
        self.time_grid = np.linspace(
            self.t0,
            self.tf,
            self.samples_number,
        )  # linspace includes the last point automatically

    def __str__(self):
        return 'Segment object'

    def add_time_offset(self, offset):
        """ Add a time offset to the trajectory.

        :param offset: time offset to add
        :type offset: float
        """
        self.t0 += offset
        self.tf = self.t0 + self.maneuver_time
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            self.samples_number
        )  # linspace because includes automatically the last point

    def change_theta(self, new_theta):
        """ Update the heading of the trajectory.
        :param new_theta: New heading angle (radians).
        :type new_theta: float
        """
        self.theta = new_theta
        self.theta0 = self.theta
        self.thetaf = self.theta
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose   = [self.xf, self.yf, self.thetaf]
        self.theta_trajectory = (
            self.theta * np.ones((self.samples_number))
        )

    def resample(self, new_samples_number):
        """
        Resample the trajectory with a new number of samples.

        Updates the path coordinates, heading, velocity, and time grid.

        :param new_samples_number: Desired number of trajectory samples.
        :type new_samples_number: int
        """
        self.samples_number = new_samples_number

        # Recompute path coordinates
        self.path_coordinates = np.column_stack((
            np.linspace(self.x0, self.xf, new_samples_number),
            np.linspace(self.y0, self.yf, new_samples_number),
        ))

        # Constant velocity and angular velocity profiles
        self.forward_velocity = np.full(new_samples_number, self.v)
        self.angular_velocity = np.zeros(new_samples_number)
        self.theta_trajectory = np.full(new_samples_number, self.theta)

        # Updated time grid (linspace includes the final point automatically)
        self.time_grid = np.linspace(self.t0, self.tf, new_samples_number)

    def reverse(self):
        """
        Reverse the trajectory direction.

        Flips the start and end orientations, inverts the motion direction,
        and updates angular quantities and trajectory profiles accordingly.
        """
        self.theta += pi
        self.v  = - self.v
        self.theta0 = self.theta
        self.thetaf = self.theta
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose   = [self.xf, self.yf, self.thetaf]
        self.forward_velocity   = self.v * np.ones((self.samples_number))
        self.theta_trajectory = self.theta * np.ones((self.samples_number))


class TurnOnTheSpot(Trajectory):
    """ Turn on-the-spot trajectory segment for a unicycle-type vehicle.
    """
    def __init__(self, x, y, theta0, thetaf, omega,
                 unicycle = None, t0 = 0, samples_number = 10):
        """
        Initialize attributes of the turn on-the-spot segment.
        :param x: x position
        :type x: float
        :param y: y position
        :type y: float
        :param theta0: initial heading angle
        :type theta0: float
        :param thetaf: final heading angle
        :type thetaf: float
        :param omega: angular velocity
        :type omega: float
        :param unicycle: unicycle vehicle object
        :type unicycle: Unicycle
        :param t0: initial time
        :type t0: float
        :param samples_number: number of samples
        :type samples_number: int
        """
        self.x0 = x
        self.y0 = y
        self.xf = x
        self.yf = y
        self.theta0 = theta0
        self.omega  = omega
        self.unicycle   = unicycle
        self.t0 = t0
        self.samples_number = samples_number

        # Compute derived attributes
        self.wrapped_theta0 = wrapPositiveAngle(theta0)
        self.wrapped_thetaf = wrapPositiveAngle(thetaf)
        self.turn_direction = efficient_sign(self.omega)

        self.delta_angle = compute_angular_difference_with_turn_direction(
            self.wrapped_theta0,
            self.wrapped_thetaf,
            self.turn_direction
        )
        if efficient_sign(self.delta_angle) == self.turn_direction:
            self.thetaf = self.theta0 + self.delta_angle
        else:
            self.thetaf = (
                self.theta0
                + self.turn_direction * abs(2 * pi - abs(self.delta_angle))
            )

        self.label = 'turn on-the-spot'
        self.start_position = [self.x0, self.y0]
        self.end_position   = [self.xf, self.yf]
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose   = [self.xf, self.yf, self.thetaf]
        self.path_coordinates = np.tile(self.start_position,
                                        (self.samples_number,1)
        )

        self.v = 0.0
        self.forward_velocity = np.zeros(self.samples_number)
        self.angular_velocity = np.full(self.samples_number, self.omega)

        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number,
        )

        # Maneuver time is zero if angular velocity is zero; always positive otherwise
        self.maneuver_time = 0.0 if self.omega == 0 else abs(self.delta_angle / self.omega)
        self.tf = self.t0 + self.maneuver_time
        self.time_grid = np.linspace(self.t0, self.tf, self.samples_number)

        self.path_length = 0.0  # TODO: verify if this should include rotational arc length

    def __str__(self):
        return 'TurnOnTheSpot object'

    def add_time_offset(self, offset):
        """ Add a time offset to the trajectory.

        :param offset: time offset to add
        :type offset: float
        """
        self.t0 += offset
        self.tf = self.t0 + self.maneuver_time
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            self.samples_number
        ) #linspace because includes automatically the last point

    def change_theta0(self, new_theta0):
        """
        Update the initial heading of the trajectory.

        Keeps the same internal rotation angle while updating
        the overall trajectory orientation and poses accordingly.

        :param new_theta0: New initial heading angle (radians).
        :type new_theta0: float
        """
        self.theta0 = new_theta0

        if efficient_sign(self.delta_angle) == self.turn_direction:
            self.thetaf = self.theta0 + self.delta_angle
        else:
            self.thetaf = (
                self.theta0
                + self.turn_direction * abs(2 * pi - abs(self.delta_angle))
            )

        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose = [self.xf, self.yf, self.thetaf]
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number,
        )

        self.delta_theta = self.thetaf - self.theta0

    def resample(self, new_samples_number):
        """
        Resample the trajectory with a new number of samples.

        Updates the path coordinates, heading, velocity, and time grid.

        :param new_samples_number: Desired number of trajectory samples.
        :type new_samples_number: int
        """
        self.samples_number = new_samples_number
        self.path_coordinates = np.tile(
            self.start_position,
            (new_samples_number,1)
        )
        self.angular_velocity   = self.omega * np.ones((new_samples_number))
        self.forward_velocity   = np.zeros((new_samples_number))
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            new_samples_number
        )
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            new_samples_number
        ) #linspace because includes automatically the last point
    
    def reverse(self):
        """
        Reverse the trajectory direction.

        Flips the start and end orientations, inverts the motion direction,
        and updates angular quantities and trajectory profiles accordingly.
        """
        self.theta0 += pi
        self.thetaf += pi
        self.wrapped_theta0 = wrapPositiveAngle(self.theta0)
        self.wrapped_thetaf = wrapPositiveAngle(self.thetaf)
        self.delta_angle = compute_angular_difference_with_turn_direction(
            self.wrapped_theta0,
            self.wrapped_thetaf,
            self.turn_direction
        )

        if efficient_sign(self.delta_angle) == self.turn_direction:
            self.thetaf = self.theta0 + self.delta_angle
        else:
            self.thetaf = (
                self.theta0
                + self.turn_direction * abs( 2 * pi - abs(self.delta_angle))
            )

        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose   = [self.xf, self.yf, self.thetaf]
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number
        )

    
class BackwardArc(Trajectory):
    """ Backward curvilinear arc trajectory segment for a unicycle-type vehicle.
    """ 

    def __init__(self, xc, yc, 
                 x0, y0, theta0,
                 xf, yf, thetaf,
                 radius, turn_direction,
                 v, omega, bicycle,
                 t0 = 0, samples_number = 10):
        """ Initialize attributes of the backward arc.
        :param xc: x coordinate of the circle center
        :type xc: float
        :param yc: y coordinate of the circle center
        :type yc: float
        :param x0: initial x position
        :type x0: float
        :param y0: initial y position
        :type y0: float
        :param theta0: initial heading angle
        :type theta0: float
        :param xf: final x position
        :type xf: float
        :param yf: final y position
        :type yf: float
        :param thetaf: final heading angle
        :type thetaf: float
        :param radius: radius of the arc
        :type radius: float
        :param turn_direction: turn direction (-1 for right, +1 for left)
        :type turn_direction: int
        :param v: forward velocity
        :type v: float
        :param omega: angular velocity
        :type omega: float
        :param bicycle: bicycle vehicle object
        :type bicycle: Bicycle
        :param t0: initial time
        :type t0: float
        :param samples_number: number of samples
        :type samples_number: int
        """
        self.xc     = xc
        self.yc     = yc
        self.x0     = x0
        self.y0     = y0
        self.theta0 = theta0
        self.thetaf = thetaf
        self.xf     = xf
        self.yf     = yf
        self.radius = radius
        self.turn_direction = turn_direction
        self.v  = v
        self.omega  = omega
        self.bicycle   = bicycle
        self.t0     = t0
        self.samples_number = samples_number

        # Compute derived attributes
        self.label = 'backward arc'
        self.wrapped_theta0 = wrapPositiveAngle(theta0)
        self.wrapped_thetaf = wrapPositiveAngle(thetaf)
        self.start_position = [self.x0, self.y0]
        self.end_position   = [self.xf, self.yf]
        self.chord = compute_distance_two_points(self.start_position,
                                                 self.end_position)
        self.start_pose     = [self.x0, self.y0, self.theta0]
        self.end_pose       = [self.xf, self.yf, self.thetaf]
        self.circle_center  = [self.xc, self.yc]
        
        self.delta_angle    = atan2(
            sin(self.wrapped_thetaf - self.wrapped_theta0),
            cos(self.wrapped_thetaf - self.wrapped_theta0)
        )

        if efficient_sign(self.delta_angle) == - self.turn_direction:
            self.iota   = 2 * asin((self.chord * 0.5)/self.radius)
        else:
            self.iota   = 2 * pi - (2 * asin((self.chord * 0.5)/self.radius))
            
        self.epsilon    = wrapPositiveAngle(
            atan2(
                (self.y0 - self.yc),
                (self.x0 - self.xc)
            )
        )
        self.epsilon = wrapPositiveAngle(atan2(self.y0 - self.yc, self.x0 - self.xc))
        angles = np.linspace(self.epsilon,
                            self.epsilon - self.turn_direction * self.iota,
                            self.samples_number)
        self.path_coordinates = np.column_stack((
            self.xc + self.radius * np.cos(angles),
            self.yc + self.radius * np.sin(angles),
        ))



        self.forward_velocity = self.v * np.ones((self.samples_number))
        self.angular_velocity = self.omega * np.ones((self.samples_number))
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number
        )
        self.path_length    = self.radius * abs(self.iota)
        self.maneuver_time  = abs(self.path_length / self.v)
        self.tf = self.t0 + self.maneuver_time
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            self.samples_number
        ) #linspace because includes automatically the last point

    def __str__(self): 
        return 'BackwardArc object'

    def add_time_offset(self, offset):
        """ Add a time offset to the trajectory.

        :param offset: time offset to add
        :type offset: float
        """
        self.t0 += offset
        self.tf = self.t0 + self.maneuver_time
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            self.samples_number
        ) #linspace because includes automatically the last point
    
    def change_theta0(self, new_theta0):
        """Change the initial heading angle of the trajectory.
        Update the initial heading of the trajectory.
        Keeps the same internal rotation angle while updating
        the overall trajectory orientation and poses accordingly.
        :param new_theta0: New initial heading angle (radians).
        :type new_theta0: float
        """
        self.theta0 = new_theta0
        if efficient_sign(self.delta_angle) == -(self.turn_direction):
            self.thetaf = self.theta0 + self.delta_angle
        else:
            self.thetaf = (
                self.theta0
                + self.turn_direction * abs(2 * pi - abs(self.delta_angle))
                )
        self.start_pose = [self.x0, self.y0, self.theta0]
        self.end_pose   = [self.xf, self.yf, self.thetaf]
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number
        )

    def resample(self, new_samples_number):
        """
        Resample the trajectory with a new number of samples.

        Updates the path coordinates, heading, velocity, and time grid.

        :param new_samples_number: Desired number of trajectory samples.
        :type new_samples_number: int
        """
        self.samples_number = new_samples_number
        angles = np.linspace(self.epsilon,
                            self.epsilon + self.turn_direction * self.iota,
                            new_samples_number)
        self.path_coordinates = np.column_stack((
            self.xc + self.radius * np.cos(angles),
            self.yc + self.radius * np.sin(angles),
        ))

        self.forward_velocity = self.v * np.ones((new_samples_number))
        self.angular_velocity = self.omega * np.ones((new_samples_number))
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            new_samples_number
        )
        self.time_grid  = np.linspace(
            self.t0,
            self.tf,
            new_samples_number
        ) #linspace because includes automatically the last point

    def reverse(self):
        """
        Reverse the trajectory direction.

        Flips the start and end orientations, inverts the motion direction,
        and updates angular quantities and trajectory profiles accordingly.
        """
        self.theta0 += pi
        self.thetaf += pi
        self.turn_direction = - self.turn_direction
        self.v  = -self.v
        self.omega  = -self.omega
        self.wrapped_theta0 = wrapPositiveAngle(self.theta0)
        self.wrapped_thetaf = wrapPositiveAngle(self.thetaf)
        self.delta_angle    = atan2(
            sin(self.wrapped_thetaf - self.wrapped_theta0),
            cos(self.wrapped_thetaf - self.wrapped_theta0)
            )    
        self.start_pose     = [self.x0, self.y0, self.theta0]
        self.end_pose       = [self.xf, self.yf, self.thetaf]

        self.forward_velocity = self.v * np.ones((self.samples_number))
        self.angular_velocity = self.omega * np.ones((self.samples_number))
        self.theta_trajectory = np.linspace(
            self.theta0,
            self.thetaf,
            self.samples_number
        )
  
    def plot_circle(self, figure=None, color="b", linestyle="dashed", linewidth=0.5):
        """
        Plot the circle associated with this trajectory/vehicle.
        """
        return plot_circle(
            self,
            figure=figure,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


class BicycleTrajectory(Trajectory):
    """ Trajectory object for representation of piece of trajectory of a unicycle vehicle.
    """
    def __init__(self, time_grid, x, y, theta, forward_velocity, steering_angle):
        """ Constructor.

        :param time_grid: time grid of the trajectory
        :type time_grid: list of floats or numpy.ndarray

        :param x: vector storing the x coordinates
            of the path in the provided time grid
        :type x: list of floats or numpy.ndarray
        
        :param y: vector storing the y coordinates
            of the path in the provided time grid
        :type y: list of floats or numpy.ndarray

        :param theta: vector storing the heading of the unicycle
            theta along the path in the provided time grid
        :type theta: list of floats or numpy.ndarray

        :param forward_velocity: vector storing the forward velocity
            of the unicycle in the provided time grid
        :type forward_velocity: list of floats or numpy.ndarray  

        :param steering_angle: vector storing the steering rate
            of the unicycle in the provided time grid
        :type steering_angle: list of floats or numpy.ndarray
        """
        self.time_grid = time_grid
        self.x0 = x[0]
        self.y0 = y[0]
        self.theta0 = theta[0]
        self.xf = x[-1]
        self.yf = y[-1]
        self.thetaf = theta[-1]
        self.path_coordinates = np.column_stack([x,y])
        self.theta = theta
        self.forward_velocity = forward_velocity
        self.steering_angle = steering_angle
        self.t0 = time_grid[0]
        self.tf = time_grid[-1]

    def __str__(self):
        return 'BicycleTrajectory object'
    
class BicycleTrajectoryOptimal(BicycleTrajectory):
    """ Optimal trajectory object for representation of piece of trajectory of a bicycle vehicle.
    """

    def __init__(self, time_grid, x, y, theta,
                 v, delta, control_time_grid, 
                control_x, control_y, control_theta,
                control_forward_velocity,
                control_steering_angle,
                x_initial_guess, y_initial_guess,
                total_time):
        super().__init__(time_grid, x, y, theta, v, delta)
        self.control_time_grid = control_time_grid
        self.control_x = control_x
        self.control_y = control_y
        self.control_theta = control_theta
        self.control_v = control_forward_velocity
        self.control_delta = control_steering_angle
        self.x_initial_guess = x_initial_guess
        self.y_initial_guess = y_initial_guess
        self.total_time = total_time
        self.theta_trajectory = theta

    def __str__(self):
        return 'BicycleTrajectoryOptimal object'

    # def plot_control_points(self, figure):
    #     import matplotlib.pyplot as plt

    #     if figure is None: figure = plt.figure()
    #     # plt.plot(self.control_x, self.control_y, 'r.',  label = 'control points')
    #     plt.step(self.control_x, self.control_y, 'r.',  label = 'control points')
    #     plt.legend()
    #     return figure
        
    # def plot_initial_guess(self, figure):
    #     import matplotlib.pyplot as plt

    #     if figure is None: figure = plt.figure()
    #     plt.plot(self.x_initial_guess, self.y_initial_guess, 'b--', label = 'initial guess')
    #     plt.legend()
    #     return figure
    
