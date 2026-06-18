import numpy as np
from .helpers.geometry_operations import wrapPositiveAngle, efficient_sign
from math import cos, sin, pi, sqrt, atan2
import warnings
from .geometry import Point


class CorridorWorld:
    """Corridor object for representation of free space between obstacles.
    """
    FWD = 0
    RGT = 1
    BCK = 2
    LFT = 3

    def __init__(self, width, height, center, tilt=0,
                 label=None, number=None):
        """Constructor.

        :param width: corridor width
        :type width: float

        :param height: corridor height
        :type height: float

        :param center: corridor center
        :type center: numpy.ndarray

        :param tilt: corridor tilt
        :type tilt: float

        :param label: label of the corridor
        :type label: string

        :param number: number of the corridor
        :type number: int
        """
        # Parameters
        self.height = height
        self.width = width
        self.tilt = tilt
        self.center = center
        self.label = label
        self.number = number

        # Derived attributes
        self.vector =  [self.center[0] + (self.height * 0.5)*cos(self.tilt),
                        self.center[1] + (self.height * 0.5)*sin(self.tilt)]
        self.unit_vector = [cos(self.tilt), sin(self.tilt)]
        self.tail = [self.center[0] - (self.height * 0.5)*cos(self.tilt), 
                     self.center[1] - (self.height * 0.5)*sin(self.tilt)]
        self.head = [self.center[0] + (self.height * 0.5)*cos(self.tilt),
                    self.center[1] + (self.height * 0.5)*sin(self.tilt)]
        self.W = self._init_W_from_width_height_tilt() # edge parameter vectors
        self.wf, self.wr, self.wb, self.wl = self.W.T # individual edge parameter vectors
        self.corners = self.get_corners()
        self.outward_normals = self._init_outward_normals()

        
    def _init_outward_normals(self):
        """Return outward unit normals for the corridor edges.

        Edge order:
            FWD = 0
            RGT = 1
            BCK = 2
            LFT = 3
        """

        ux = cos(self.tilt)
        uy = sin(self.tilt)

        # Right normal to the forward direction
        rx = sin(self.tilt)
        ry = -cos(self.tilt)

        return {
            self.FWD: np.array([ux, uy]),
            self.RGT: np.array([rx, ry]),
            self.BCK: np.array([-ux, -uy]),
            self.LFT: np.array([-rx, -ry]),
        }
     
    def __str__(self):
        """Return a string representation of the corridor."""
        parts = ["Corridor"]

        if self.number is not None:
            parts.append(str(self.number))
        if self.label is not None:
            parts.append(f"labelled {self.label}")

        description = " ".join(parts)
        return (
            f"{description} with center {self.center}, "
            f"width {self.width}, height {self.height}, tilt {self.tilt}"
    )

    def _init_W_from_width_height_tilt(self):
        """Initialize the edge parameters of the corridor from width, height and center.

        :param width: corridor width
        :type width: float

        :param height: corridor height
        :type height: float

        :param center: corridor center
        :type center: numpy.ndarray

        :return: W = edge parameter vectors
        :rtype: numpy.ndarray
        """
        # rotation matrix
        R = np.array([[cos(self.tilt), -sin(self.tilt)],
                      [sin(self.tilt), cos(self.tilt)]])
        
        Nrot = R @ np.array([[1, 0, -1, 0],
                             [0, -1, 0, 1]])
        
        T0_rot = R @ np.array([[0.5*self.height, 0, -0.5*self.height, 0],
                               [0, -0.5*self.width, 0, 0.5*self.width]])
        
        T = T0_rot + np.array([[self.center[0]], [self.center[1]]])

        Nrot_T = np.array([Nrot[:, 0] @ T[:, 0], Nrot[:, 1] @ T[:, 1],
                           Nrot[:, 2] @ T[:, 2], Nrot[:, 3] @ T[:, 3]])

        return np.vstack([Nrot, -Nrot_T])
    
    def get_corners(self):
        """
        Compute and store the coordinates of the corridor corners.

        The corners are returned and stored in counter-clockwise order:
            1. Top-right
            2. Bottom-right
            3. Bottom-left
            4. Top-left

        :return: Array of corner coordinates (4x2).
        :rtype: numpy.ndarray
        """
        # Local rectangle corners before rotation (width × height)
        local_corners = np.array([
            [ self.width * 0.5,  self.height * 0.5],   # top-right
            [ self.width * 0.5, -self.height * 0.5],   # bottom-right
            [-self.width * 0.5, -self.height * 0.5],   # bottom-left
            [-self.width * 0.5,  self.height * 0.5],   # top-left
        ])

        # Rotation matrix (tilt measured from x-axis)
        rot_angle = self.tilt - 0.5 * pi
        R = np.array([
            [cos(rot_angle), -sin(rot_angle)],
            [sin(rot_angle),  cos(rot_angle)]
        ])

        # Rotate and translate corners
        self.corners = local_corners @ R.T + np.array(self.center)

        return self.corners

    def shrink_with_direction(self, shrink_direction, offset):
        """
        Shrink the corridor height by ``offset`` meters from either its head or tail.

        :param shrink_direction: Which end to shrink, either ``"head"`` or ``"tail"``.
        :type shrink_direction: str
        :param offset: Amount to reduce the height (m).
        :type offset: float
        :return: New, shrunken :class:`CorridorWorld` instance.
        :rtype: CorridorWorld
        """
        # Determine sign for offset direction
        sign = -1 if shrink_direction == "head" else 1
        remove = sign * abs(offset) * 0.5

        # Compute new center position and dimensions
        new_center = [
            self.center[0] + remove * cos(self.tilt),
            self.center[1] + remove * sin(self.tilt)
        ]
        new_height = self.height - abs(offset)

        return CorridorWorld(
            width=self.width,
            height=new_height,
            center=new_center,
            tilt=self.tilt,
            label=self.label
        )

    def shrink(self, margin):
        """
        Create a new corridor with reduced width and height.

        The corridor center and tilt remain unchanged, while both dimensions
        are reduced by ``2 × margin``.

        :param margin: Reduction margin in meters.
        :type margin: float
        :return: New, shrunken :class:`CorridorWorld` instance.
        :rtype: CorridorWorld
        """
        return CorridorWorld(
            width=self.width - 2 * margin,
            height=self.height - 2 * margin,
            center=self.center,
            tilt=self.tilt,
            label=self.label
        )

    def update(self, **kwargs):
        """
        Update one or more corridor attributes and automatically refresh
        all derived geometric parameters.

        The method reinitializes the object using the existing attributes,
        updating only those provided in ``kwargs``. This ensures that all
        dependent quantities (e.g. head, tail, corners, edge vectors)
        remain consistent.

        Example:
            corridor.update(width=5.0, tilt=np.pi/6)

        :param kwargs: Attributes to update (e.g., ``width``, ``height``, ``center``, ``tilt``).
        :type kwargs: dict
        """
        # Valid constructor parameters
        valid_keys = {"width", "height", "center", "tilt", "label", "number"}

        # Warn if unknown keys are provided
        for key in kwargs:
            if key not in valid_keys:
                warnings.warn(f"Ignoring unknown attribute '{key}' in update().")

        # Gather current attributes, then override with provided ones
        params = {
            "width": self.width,
            "height": self.height,
            "center": self.center,
            "tilt": self.tilt,
            "label": self.label,
            "number": self.number,
        }
        params.update({k: v for k, v in kwargs.items() if k in valid_keys})

        # Reinitialize the object (recomputes all derived properties)
        self.__init__(**params)

    def update_vector(self, tail, head, width, margin):
        """
        Redefine the corridor using a new vector and dimensions.

        The new corridor is defined by its ``head`` and ``tail`` points,
        a specified ``width``, and an additional ``margin`` added to the
        computed height. The corridor center and tilt are derived
        automatically from the input vector.

        :param tail: Point defining the tail of the corridor vector.
        :type tail: list[float] or numpy.ndarray
        :param head: Point defining the head of the corridor vector.
        :type head: list[float] or numpy.ndarray
        :param width: Corridor width in meters.
        :type width: float
        :param margin: Additional margin (m) added to the computed height.
                    The corridor center remains unchanged.
        :type margin: float
        """
        # Compute geometric properties from vector
        delta_x = head[0] - tail[0]
        delta_y = head[1] - tail[1]
        center = [
            tail[0] + 0.5 * delta_x,
            tail[1] + 0.5 * delta_y
        ]
        tilt = wrapPositiveAngle(atan2(delta_y, delta_x))
        height = sqrt(delta_x**2 + delta_y**2) + margin

        # Reinitialize corridor with new parameters
        self.__init__(width, height, center, tilt)

    def compute_relative_turn_direction(self, corridor2):
        """
        Compute the relative turning direction between two corridors.

        The direction is obtained from the 2D cross product between their
        unit vectors:

            ``turn = x1·y2 - x2·y1``

        where:
            - ``turn > 0`` → left turn  
            - ``turn < 0`` → right turn  
            - ``turn ≈ 0`` → straight (aligned)

        :param corridor2: Second corridor to compare against.
        :type corridor2: CorridorWorld
        :return: Turning direction indicator (1 = left, -1 = right, 0 = straight).
        :rtype: float
        """
        tol = 1e-5  # numerical tolerance for "straight"
        turn = (
            self.unit_vector[0] * corridor2.unit_vector[1]
            - corridor2.unit_vector[0] * self.unit_vector[1]
        )

        if abs(turn) < tol:
            return 0.0

        return efficient_sign(turn)  # or np.sign(turn)
 
    def rotate_corridor(self, angle):
        """Return a new corridor that is a rotated
        version of the current corridor.

        :param angle: angle in radians
        :type angle: float
        """
        return CorridorWorld(width=self.width,
                             height=self.height,
                             center=self.center,
                             tilt=self.tilt + angle,
                             label=self.label,
                             number=self.number)

    def closest_point_on_corridor(self, point):
        """
        Return the closest point on or inside the rectangular corridor
        to an external point.

        :param point: Point-like object with x/y or index access
        :return: Point
        """
        px, py = point[0], point[1]
        cx, cy = self.center[0], self.center[1]

        # Corridor longitudinal direction
        ux = cos(self.tilt)
        uy = sin(self.tilt)

        # Perpendicular direction across corridor width
        vx = -sin(self.tilt)
        vy = cos(self.tilt)

        # Vector from corridor center to point
        dx = px - cx
        dy = py - cy

        # Coordinates in corridor-local frame
        local_forward = dx * ux + dy * uy
        local_side = dx * vx + dy * vy

        # Clamp to rectangle bounds
        local_forward = max(-self.height * 0.5, min(self.height * 0.5, local_forward))
        local_side = max(-self.width * 0.5, min(self.width * 0.5, local_side))

        # Transform back to world frame
        closest_x = cx + local_forward * ux + local_side * vx
        closest_y = cy + local_forward * uy + local_side * vy

        return Point(closest_x, closest_y)


