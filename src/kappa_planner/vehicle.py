import numpy as np 
import math as m
from copy import copy
import yaml 
import os
import warnings


class Vehicle:
    """Vehicle object for representation of all types of vehicles.
    """
    
    def __init__(self):
        """Constructor. Vehicle class is the parent of all types of vehicles.
        """
        pass

    def __str__(self):
        return 'Vehicle object'

    def copy(self):
        """Generate a copy of the vehicle object.

        :return: copy of the vehicle
        :rtype: object of vehicle class
        """
        return copy(self)

    def simulate_one_step(self, y0, u0, dt):
        """Simulate the dynamics of the vehicle one step forward with Runge-Kutta4 simulation scheme.

        :param y0: initial state
        :type y0: numpy.ndarray

        :param u0: control inputs
        :type u0: numpy.ndarray

        :dt: time interval
        :type: float

        :return y: final state after one simulation step
        :rtype: numpy.ndarray
        """
        k1 = self.dynamics(y0, u0)
        k2 = self.dynamics(y0 + dt*k1/2., u0)
        k3 = self.dynamics(y0 + dt*k2/2., u0)
        k4 = self.dynamics(y0 + dt*k3, u0)
        y = y0 + (dt / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        return y

    def simulate_trajectory(self, initial_state, input_trajectory, time_interval):
        """
        Simulate the vehicle dynamics N steps forward using a 4th-order Runge–Kutta scheme.

        The number of simulation steps N is inferred from the input trajectory.

        :param initial_state: Initial vehicle state vector.
        :type initial_state: numpy.ndarray
        :param input_trajectory: Control input sequence (shape: [nu, N]).
        :type input_trajectory: numpy.ndarray
        :param time_interval: Time interval.
        :type time_interval: float
        :returns: The full state trajectory, including the initial state.
        :rtype: numpy.ndarray
        """
        nx = initial_state.shape[0]
        n_steps = input_trajectory.shape[1]

        # Preallocate and initialize trajectory array
        state_traj = np.empty((nx, n_steps + 1))
        state_traj[:, 0] = initial_state

        # Simulate each step forward
        for k in range(n_steps):
            state_traj[:, k + 1] = self.simulate_one_step(
                state_traj[:, k],
                input_trajectory[:, k],
                time_interval,
            )

        return state_traj


class Unicycle(Vehicle):
    """
    Unicycle vehicle model.

    Inherits from the :class:`Vehicle` base class and represents a simple
    nonholonomic robot with a single wheel or equivalent differential drive.
    The unicycle configuration is described by the state vector
    [x, y, theta]^T, where:

        - x and y are the planar coordinates of the vehicle position.
        - theta is the vehicle orientation in radians.

    The kinematic model follows:

        xxdot = v * cos(theta)  
        ydot = v * sin(theta)  
        thetadot = omega

    where v is the linear velocity and omega is the angular velocity.
    """

    def __init__(
        self,
        state=np.zeros(3),
        width=0.430,
        length=0.508,
        v_max=0.5,
        v_min=-0.5,
        omega_max=0.5,
        omega_min=-0.5,
        model=None,
    ):
        """
        Initialize a Unicycle object.

        All arguments are optional. If no model name is provided, the unicycle
        is initialized with default parameters corresponding to a Jackal robot.

        :param state: State vector of the unicycle [x, y, theta]^T.
                      Defaults to zeros.
        :type state: numpy.ndarray
        :param width: Width of the unicycle in meters. Defaults to 0.430 m.
        :type width: float
        :param length: Length of the unicycle in meters. Defaults to 0.508 m.
        :type length: float
        :param v_max: Maximum forward velocity (m/s). Defaults to 0.5.
        :type v_max: float
        :param v_min: Minimum forward velocity (m/s). Defaults to −0.5.
        :type v_min: float
        :param omega_max: Maximum angular velocity (rad/s). Defaults to 0.5.
        :type omega_max: float
        :param omega_min: Minimum angular velocity (rad/s). Defaults to −0.5.
        :type omega_min: float
        :param model: Name of the model defined in ``unicycle_library.yaml``.
                      If provided, the parameters are loaded from that file.
        :type model: str, optional
        """
        if model is not None:
            # Load parameters from YAML model library
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, "vehicle_library/unicycle_library.yaml")

            try:
                with open(yaml_path, "r") as f:
                    unicycle_library = yaml.safe_load(f)

                if model not in unicycle_library:
                    raise KeyError(f"Model '{model}' not found in YAML library.")

                params = unicycle_library[model]
                self.state = state
                self.width = params["width"]
                self.length = params["length"]
                self.v_max = params["v_max"]
                self.v_min = params["v_min"]
                self.omega_max = params["omega_max"]
                self.omega_min = params["omega_min"]

            except (OSError, yaml.YAMLError, KeyError) as e:
                warnings.warn(f"Failed to load unicycle model '{model}': {e}")
                # fallback to default parameters
                self.state = state
                self.width = width
                self.length = length
                self.v_max = v_max
                self.v_min = v_min
                self.omega_max = omega_max
                self.omega_min = omega_min
        else:
            # parameters provided by user or default parameters (Jackal)
            self.state = state
            self.width = width
            self.length = length
            self.v_max = v_max
            self.v_min = v_min
            self.omega_max = omega_max
            self.omega_min = -omega_max

        # Derived property
        self.max_radius = abs(self.v_max / self.omega_max)

    def __str__(self):
        return 'Unicycle object'

    def get_pose(self):
        """Returns the state of the unicycle.
        
        :return self.state: state of the unicycle
        :rtype: numpy.ndarray
        """
        return self.state[:3]
    
    def update(self, **kwargs):
        """
        Update attributes of the MotionPlanner instance.

        Keyword arguments correspond to existing attributes of the object.
        Attributes not found in the instance are ignored with a warning.
        If either v_max or omega_max is updated, max_radius is
        automatically recalculated as abs(v_max / omega_max).

        Example:
            planner.update(start_pose=new_start, end_pose=new_end, v_max=0.6)
        """
        # Track whether speed parameters are updated
        vmax_updated = "v_max" in kwargs
        omegamax_updated = "omega_max" in kwargs

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Ignoring unknown attribute '{key}' in update().")

        # Update derived parameter if needed
        if vmax_updated or omegamax_updated:
            v_max = getattr(self, "v_max", None)
            omega_max = getattr(self, "omega_max", None)

            if v_max is not None and omega_max not in (None, 0):
                self.max_radius = abs(v_max / omega_max)
                self.omega_min = -self.omega_max
            else:
                warnings.warn(
                    "Cannot compute max_radius: omega_max is zero or undefined."
                )

    def dynamics(self, x, u):
        """
        Compute the time derivatives of the unicycle state.

        Applies the unicycle kinematic model to compute the rate of change
        of the state vector given the current state and control inputs.

        :param x: Current state vector [x, y, theta].
        :type x: numpy.ndarray or list[float]
        :param u: Control input vector [v, omega].
        :type u: numpy.ndarray or list[float]
        :returns: State derivatives [x_dot, y_dot, theta_dot] as a NumPy array.
        :rtype: numpy.ndarray
        """
        v, omega = u  # unpack control inputs
        theta = x[2]

        x_dot = v * m.cos(theta)
        y_dot = v * m.sin(theta)
        theta_dot = omega

        return np.array([x_dot, y_dot, theta_dot])

    
class Bicycle(Vehicle):
    """
    Bicycle object.

    Inherits from the :class:`Vehicle` class. The bicycle model represents
    a vehicle with two wheels — a steerable front wheel and a fixed rear wheel.

    Its configuration is defined by the state vector [x, y, theta]^T, where:
        - x and y are the coordinates of the rear wheel contact point.
        - theta is the vehicle orientation (the angle of the line connecting
          the wheel centers).

    The kinematic model is:
        xdot = v * cos(theta),
        ydot = v * sin(theta),
        thetadot = v * tan(theta) / L

    where v is the forward velocity, delta is the steering angle,
    and L is the wheelbase.
    """

    def __init__(
        self,
        state=np.zeros(3),
        width=0.430,
        length=0.508,
        wheelbase=0.4,
        v_max=0.5,
        v_min=-0.5,
        delta_max=0.5,
        delta_min=-0.5,
        model=None,
    ):
        """
        Initialize a Bicycle object.

        All arguments are optional. If no model name is provided, the bicycle
        is initialized with default parameters corresponding to a Jackal robot.

        :param state: State vector of the bicycle [x, y, theta]^T. Defaults to zeros.
        :type state: numpy.ndarray
        :param width: Width of the bicycle in meters. Defaults to 0.430 m.
        :type width: float
        :param length: Length of the bicycle in meters. Defaults to 0.508 m.
        :type length: float
        :param wheelbase: Distance between front and rear axles (m). Defaults to 0.4.
        :type wheelbase: float
        :param v_max: Maximum forward velocity (m/s). Defaults to 0.5.
        :type v_max: float
        :param v_min: Minimum forward velocity (m/s). Defaults to −0.5.
        :type v_min: float
        :param delta_max: Maximum steering angle (rad). Defaults to 0.5.
        :type delta_max: float
        :param delta_min: Minimum steering angle (rad). Defaults to −0.5.
        :type delta_min: float
        :param model: Name of the model defined in ``bicycle_library.yaml``.
                      If provided, the parameters are loaded from that file.
        :type model: str, optional
        """
        if model is not None:
            # Load parameters from YAML model library
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, "vehicle_library/bicycle_library.yaml")

            try:
                with open(yaml_path, "r") as f:
                    bicycle_library = yaml.safe_load(f)

                if model not in bicycle_library:
                    raise KeyError(f"Model '{model}' not found in YAML library.")

                params = bicycle_library[model]
                self.state = state
                self.width = params["width"]
                self.length = params["length"]
                self.wheelbase = params["wheelbase"]
                self.v_max = params["v_max"]
                self.v_min = params["v_min"]
                self.delta_max = params["delta_max"]
                self.delta_min = params["delta_min"]

            except (OSError, yaml.YAMLError, KeyError) as e:
                warnings.warn(f"Failed to load bicycle model '{model}': {e}")
                # fallback to default parameters
                self.state = state
                self.width = width
                self.length = length
                self.wheelbase = wheelbase
                self.v_max = v_max
                self.v_min = v_min
                self.delta_max = delta_max
                self.delta_min = delta_min
        else:
            # parameters provided by user or default parameters (Jackal)
            self.state = state
            self.width = width
            self.length = length
            self.wheelbase = wheelbase
            self.v_max = v_max
            self.v_min = v_min
            self.delta_max = delta_max
            self.delta_min = delta_min

        # Derived parameters
        self.omega_max = self.v_max * m.tan(self.delta_max) / self.wheelbase
        self.omega_min = -self.omega_max

        if abs(m.tan(self.delta_max)) < 1e-8:
            self.max_radius = np.inf
        else:
            self.max_radius = abs(self.wheelbase / m.tan(self.delta_max))

    def __str__(self):
        return 'Bicycle object'

    def get_pose(self):
        """Returns the state of the bicycle.
        
        :return self.state: state of the bicycle
        :rtype: numpy.ndarray
        """
        return self.state[:3]
    
    def update(self, **kwargs):
        """
        Update attributes of the Bicycle instance.

        Keyword arguments correspond to existing attributes of the object.
        Attributes not found in the instance are ignored with a warning.

        If any of the following attributes are updated:
            - ``v_max``
            - ``delta_max``
            - ``wheelbase``

        then the dependent parameters ``omega_max``, ``omega_min``, and
        ``max_radius`` are automatically recalculated as:

            ``omega_max = v_max * tan(delta_max) / wheelbase``  
            ``omega_min = -omega_max``  
            ``max_radius = abs(wheelbase / tan(delta_max))``

        Example:
            bicycle.update(v_max=1.0, delta_max=0.6)
        """
        # Track updates to parameters that affect derived quantities
        v_updated = "v_max" in kwargs
        delta_updated = "delta_max" in kwargs
        wheelbase_updated = "wheelbase" in kwargs

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Ignoring unknown attribute '{key}' in update().")

        # Recompute derived parameters if needed
        if v_updated or delta_updated or wheelbase_updated:
            v_max = getattr(self, "v_max", None)
            delta_max = getattr(self, "delta_max", None)
            wheelbase = getattr(self, "wheelbase", None)

            if None in (v_max, delta_max, wheelbase):
                warnings.warn(
                    "Cannot update derived parameters: v_max, delta_max, or wheelbase is undefined."
                )
                return

            if abs(m.tan(delta_max)) < 1e-8:
                warnings.warn(
                    "Cannot compute derived parameters: tan(delta_max) is too small."
                )
                self.omega_max = np.inf
                self.omega_min = -np.inf
                self.max_radius = np.inf
            else:
                self.omega_max = v_max * m.tan(delta_max) / wheelbase
                self.omega_min = -self.omega_max
                self.max_radius = abs(wheelbase / m.tan(delta_max))

            # Consistency check for omega_min symmetry
            if not np.isclose(self.omega_min, -self.omega_max, rtol=1e-6, atol=1e-8):
                warnings.warn(
                    f"Inconsistent angular limits: omega_min = {self.omega_min:.3f}, "
                    f"-omega_max = {-self.omega_max:.3f}. Expected omega_min = -omega_max.",
                    UserWarning,
                )

    def dynamics(self, x, u):
        """
        Compute the time derivatives of the bicycle state.

        Applies the bicycle kinematic model to compute the rate of change
        of the state vector given the current state and control inputs.

        :param x: Current state vector [x, y, theta].
        :type x: numpy.ndarray or list[float]
        :param u: Control input vector u = [v, delta].
        :type u: numpy.ndarray or list[float]
        :returns: State derivatives [xdot, ydot, thetadot] as a NumPy array.
        :rtype: numpy.ndarray
        """
        v, delta = u
        theta = x[2]

        xdot = v * m.cos(theta)
        ydot = v * m.sin(theta)
        thetadot = v * m.tan(delta)/self.wheelbase   
        return np.array([xdot, ydot, thetadot])
    

class Bicycle_Acceleration(Vehicle):
    """
    Kinematic bicycle model with acceleration and steering rate limits.

    Inherits from the :class:`Vehicle` base class and represents a
    car-like vehicle with front-wheel steering and bounded acceleration
    and steering rate dynamics.

    The vehicle configuration is defined by the state vector
    ``[x, y, theta, v, delta]^T``, where:
        - ``x, y`` are the coordinates of the rear wheel contact point (m),
        - ``theta`` is the vehicle orientation (rad),
        - ``v`` is the longitudinal velocity (m/s),
        - ``delta`` is the front wheel steering angle (rad).

    The kinematic model follows:

        ``xdot = v * cos(theta)``  
        ``ydot = v * sin(theta)``  
        ``thetadot = v * tan(delta) / L``  
        ``vdot = a``  
        ``deltadot = deltadot``  

    where:
        - ``L`` is the wheelbase,
        - ``a`` is the longitudinal acceleration,
        - ``deltadot`` is the steering rate.

    The model includes bounds on velocity, acceleration, steering angle,
    and steering rate to represent real vehicle constraints.
    """

    def __init__(
        self,
        state=np.zeros(5),
        width=1.8,
        length_front=3.0,
        length_rear=1.5,
        wheelbase=2.7,
        v_max=30.0,
        v_min=0.0,
        a_max=3.0,
        a_min=-6.0,
        delta_max=0.6,
        delta_min=-0.6,
        delta_dot_max=0.6,
        delta_dot_min=-0.6,
        model=None,
    ):
        """
        Initialize a Bicycle_Acceleration object.

        All arguments are optional. If no model name is provided, the vehicle
        is initialized with default parameters representing a passenger car.

        :param state: Vehicle state vector ``[x, y, theta, v, delta]``.
                      Defaults to zeros.
        :type state: numpy.ndarray
        :param width: Vehicle width in meters. Defaults to 1.8 m.
        :type width: float
        :param length_front: Distance from vehicle CG to front axle (m).
        :type length_front: float
        :param length_rear: Distance from vehicle CG to rear axle (m).
        :type length_rear: float
        :param wheelbase: Distance between front and rear axles (m).
        :type wheelbase: float
        :param v_max: Maximum forward velocity (m/s). Defaults to 30.
        :type v_max: float
        :param v_min: Minimum forward velocity (m/s). Defaults to 0.
        :type v_min: float
        :param a_max: Maximum acceleration (m/s²). Defaults to 3.
        :type a_max: float
        :param a_min: Minimum acceleration (m/s²). Defaults to -6.
        :type a_min: float
        :param delta_max: Maximum steering angle (rad). Defaults to 0.6.
        :type delta_max: float
        :param delta_min: Minimum steering angle (rad). Defaults to -0.6.
        :type delta_min: float
        :param delta_dot_max: Maximum steering rate (rad/s). Defaults to 0.6.
        :type delta_dot_max: float
        :param delta_dot_min: Minimum steering rate (rad/s). Defaults to -0.6.
        :type delta_dot_min: float
        :param model: Optional model name from ``bicycle_acceleration_library.yaml``.
                      If provided, parameters are loaded from that file.
        :type model: str, optional
        """
        if model is not None:
            # Load parameters from YAML model library
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, "vehicle_library/bicycle_acceleration_library.yaml")

            try:
                with open(yaml_path, "r") as f:
                    bicycle_acceleration_library = yaml.safe_load(f)

                if model not in bicycle_acceleration_library:
                    raise KeyError(f"Model '{model}' not found in YAML library.")

                params = bicycle_acceleration_library[model]
                self.state = state
                self.width = params["width"]
                self.length_front = params["length_front"]
                self.length_rear = params["length_rear"]
                self.wheelbase = params["wheelbase"]
                self.v_max = params["v_max"]
                self.v_min = params["v_min"]
                self.a_max = params["a_max"]
                self.a_min = params["a_min"]
                self.delta_max = params["delta_max"]
                self.delta_min = params["delta_min"]
                self.delta_dot_max = params["delta_dot_max"]
                self.delta_dot_min = params["delta_dot_min"]

            except (OSError, yaml.YAMLError, KeyError) as e:
                warnings.warn(f"Failed to load bicycle model '{model}': {e}")
                # fallback to default parameters
                self.state = state
                self.width = width
                self.length_front = length_front
                self.length_rear = length_rear
                self.wheelbase = wheelbase
                self.v_max = v_max
                self.v_min = v_min
                self.a_max = a_max
                self.a_min = a_min
                self.delta_max = delta_max
                self.delta_min = delta_min
                self.delta_dot_max = delta_dot_max
                self.delta_dot_min = delta_dot_min
        else:
            # parameters provided by user or default parameters (passenger car)
            self.state = state
            self.width = width
            self.length_front = length_front
            self.length_rear = length_rear
            self.wheelbase = wheelbase
            self.v_max = v_max
            self.v_min = v_min
            self.a_max = a_max
            self.a_min = a_min
            self.delta_max = delta_max
            self.delta_min = delta_min
            self.delta_dot_max = delta_dot_max
            self.delta_dot_min = delta_dot_min

        # Derived parameters
        self.omega_max = self.v_max * m.tan(self.delta_max) / self.wheelbase
        self.omega_min = -self.omega_max
        self.max_radius = (
            np.inf if abs(m.tan(self.delta_max)) < 1e-8
            else abs(self.wheelbase / m.tan(self.delta_max))
        )

    def __str__(self):
        return 'BicycleAcceleration object'

    def get_pose(self):
        """Returns the state of the bicycle.
        
        :return self.state: state of the bicycle
        :rtype: numpy.ndarray
        """
        return self.state[:3]
    
    def update(self, **kwargs):
        """
        Update attributes of the Bicycle_Acceleration instance.

        Keyword arguments correspond to existing attributes of the object.
        Attributes not found in the instance are ignored with a warning.

        If any of the following attributes are updated:
            - ``v_max``
            - ``delta_max``
            - ``wheelbase``

        then the dependent parameters ``omega_max``, ``omega_min``, and
        ``max_radius`` are automatically recalculated as:

            ``omega_max = v_max * tan(delta_max) / wheelbase``  
            ``omega_min = -omega_max``  
            ``max_radius = abs(wheelbase / tan(delta_max))``

        Example:
            car.update(v_max=20.0, delta_max=0.5, a_max=4.0)

        :param kwargs: Keyword arguments matching instance attributes.
        :type kwargs: dict
        """
        # Track which parameters were updated
        v_updated = "v_max" in kwargs
        delta_updated = "delta_max" in kwargs
        wheelbase_updated = "wheelbase" in kwargs

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Ignoring unknown attribute '{key}' in update().")

        # Recompute derived parameters if needed
        if v_updated or delta_updated or wheelbase_updated:
            v_max = getattr(self, "v_max", None)
            delta_max = getattr(self, "delta_max", None)
            wheelbase = getattr(self, "wheelbase", None)

            if None in (v_max, delta_max, wheelbase):
                warnings.warn(
                    "Cannot update derived parameters: v_max, delta_max, or wheelbase is undefined."
                )
                return

            if abs(m.tan(delta_max)) < 1e-8:
                warnings.warn(
                    "Cannot compute derived parameters: tan(delta_max) is too small."
                )
                self.omega_max = np.inf
                self.omega_min = -np.inf
                self.max_radius = np.inf
            else:
                self.omega_max = v_max * m.tan(delta_max) / wheelbase
                self.omega_min = -self.omega_max
                self.max_radius = abs(wheelbase / m.tan(delta_max))

            # Consistency check
            if not np.isclose(self.omega_min, -self.omega_max, rtol=1e-6, atol=1e-8):
                warnings.warn(
                    f"Inconsistent angular limits: omega_min = {self.omega_min:.3f}, "
                    f"-omega_max = {-self.omega_max:.3f}. Expected omega_min = -omega_max.",
                    UserWarning,
                )

        # Optional consistency checks for acceleration and steering-rate limits
        if "a_max" in kwargs and "a_min" in kwargs:
            if not np.isclose(self.a_min, -abs(self.a_max), rtol=1e-3, atol=1e-6):
                warnings.warn(
                    f"a_min ({self.a_min:.3f}) is not equal to -a_max ({-self.a_max:.3f}). "
                    "Acceleration limits are asymmetric.",
                    UserWarning,
                )

        if "delta_dot_max" in kwargs and "delta_dot_min" in kwargs:
            if not np.isclose(self.delta_dot_min, -self.delta_dot_max, rtol=1e-3, atol=1e-6):
                warnings.warn(
                    f"delta_dot_min ({self.delta_dot_min:.3f}) is not equal to "
                    f"-delta_dot_max ({-self.delta_dot_max:.3f}). Steering rate limits are asymmetric.",
                    UserWarning,
                )

    def dynamics(self, x, u):
        """
        Compute the time derivatives of the bicycle-acceleration model state.

        Applies the extended kinematic bicycle model to compute the rate of change
        of the vehicle state, including longitudinal acceleration and steering
        rate as control inputs.

        The state vector is defined as ``x = [x, y, theta, v, delta]``, where:
            - ``x`` and ``y`` are the coordinates of the rear wheel contact point (m),
            - ``theta`` is the vehicle orientation (rad),
            - ``v`` is the longitudinal velocity (m/s),
            - ``delta`` is the steering angle (rad).

        The control input vector is ``u = [a, delta_dot]``, where:
            - ``a`` is the longitudinal acceleration (m/s²),
            - ``delta_dot`` is the steering rate (rad/s).

        The kinematic equations are:

            ``xdot = v * cos(theta)``  
            ``ydot = v * sin(theta)``  
            ``thetadot = v * tan(delta) / L``  
            ``vdot = a``  
            ``deltadot = delta_dot``  

        where ``L`` is the wheelbase.

        :param x: Current state vector ``[x, y, theta, v, delta]``.
        :type x: numpy.ndarray or list[float]
        :param u: Control input vector ``[a, delta_dot]``.
        :type u: numpy.ndarray or list[float]
        :returns: Time derivatives of the states ``[xdot, ydot, thetadot, vdot, deltadot]``.
        :rtype: numpy.ndarray
        """
        a, delta_dot = u
        _, _, theta, v, delta = x

        x_dot = v * m.cos(theta)
        y_dot = v * m.sin(theta)
        theta_dot = v * m.tan(delta) / self.wheelbase
        v_dot = a
        delta_dot = delta_dot

        return np.array([x_dot, y_dot, theta_dot, v_dot, delta_dot])




