## Geometry.py
## Geometrical entities for the Arena package
from math import cos, sin, pi, sqrt, atan2
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return (self.x, self.y)[idx]

    def __setitem__(self, idx, value):
        if idx == 0:
            self.x = value
        elif idx == 1:
            self.y = value
        else:
            raise IndexError("Point index out of range")

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y

    @property
    def xy(self):
        return [self.x, self.y]

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Circle:
    '''Representation of a circle in a 2D space.
    '''

    def __init__(self, center, radius):
        '''Constructor.

        :param center: x and y coordinates of the circle center
        :type center: point

        :param radius: radius of the circle
        :type radius: float
        '''
        # Parameters
        self.center = center
        self.radius = radius
        self.xc = center.x
        self.yc = center.y


class Pose:
    """Representation of the pose of a robot.
    Pose [x, y, theta]: position + orientation
    """

    def __init__(self, position, theta):
        self.position = position
        self.theta = theta
        self.x = position.x
        self.y = position.y

    # --- sequence behavior ---
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.theta
        raise IndexError("Pose index out of range (0..2)")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
            self.position.x = value
        elif index == 1:
            self.y = value
            self.position.y = value
        elif index == 2:
            self.theta = value
        else:
            raise IndexError("Pose index out of range (0..2)")

    def __len__(self):
        return 3

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.theta

    # --- copy ---
    def copy(self):
        """Return a deep copy of the pose."""
        return Pose(
            position=Point(self.x, self.y),
            theta=self.theta,
        )

    # --- reverse pose ---
    def reversed(self):
        """
        Return a new pose with opposite heading.
        Position stays unchanged.
        """
        theta_reversed = (self.theta + pi) % (2 * pi)

        return Pose(
            position=Point(self.x, self.y),
            theta=theta_reversed,
        )

class IntermediateCircle(Circle):
    """A circle used in the algorithm, associated with a corner point and a turn direction."""

    def __init__(
        self,
        center,
        radius,
        corner_point,
        turn_direction,
        index=None,
        s_max=None,
        edge_pair=None,
        door_point=None,
        door_type=None,
        start_angle_arc=None,
        rho=None,
        swept_radius=None,        # R + r
        admissible_radius=None,   # R - r
        lower_bound_x=None,       # a
        lower_bound_y=None,       # b
        lower_bound_x_unclamped=None,  # a'
        lower_bound_y_unclamped=None,  # b'
        ex=None,
        ey=None,
        forbidden_points=None,
        merged = False,
        shifted = False,
        skip = False,
        skip_reason = None
    ):
        super().__init__(center=center, radius=radius)

        if turn_direction not in (-1, 0, 1):
            raise ValueError("turn_direction must be -1, 0, or 1")

        self.canonical_center = Point(center.x, center.y)
        # self.center = Point(center.x, center.y)  # mutable center that can be shifted along the bisector
        # self.xc = self.center.x
        # self.yc = self.center.y

        self.corner_point = corner_point          # Point
        self.turn_direction = turn_direction      # -1, 0, +1
        self.index = index
        self.bisector_direction = atan2(
            corner_point.y - self.canonical_center.y,
            corner_point.x - self.canonical_center.x
            )  # precompute for efficiency
        self.s = 0
        self.s_max = 10000 if s_max is None else s_max  # Optional limit on how far along the bisector we can go
        self.number_of_shifts = 0

        self.start_angle_arc = start_angle_arc
        self.rho = rho
        self.swept_radius = swept_radius
        self.admissible_radius = admissible_radius
        self.lower_bound_x = lower_bound_x
        self.lower_bound_y = lower_bound_y
        self.lower_bound_x_unclamped = lower_bound_x_unclamped
        self.lower_bound_y_unclamped = lower_bound_y_unclamped
        self.ex = ex
        self.ey = ey
        self.forbidden_points = forbidden_points if forbidden_points is not None else []
        self.shifted = shifted
        self.skip = skip
        self.skip_reason = skip_reason

        if self.start_angle_arc is not None and rho is not None:
            self.end_angle_arc = self.start_angle_arc - turn_direction * pi / 2

            n_arc_samples = 50

            angles = np.linspace(
                self.start_angle_arc,
                self.end_angle_arc,
                n_arc_samples
            )

            self.arc_coordinates = [
                Point(
                    self.corner_point.x + rho * cos(angle),
                    self.corner_point.y + rho * sin(angle)
                )
                for angle in angles
            ]
        else:
            self.end_angle_arc = None
            self.arc_coordinates = None
            
        self.edge_pair = edge_pair
        if self.turn_direction == 1: 
            self.door_point_left = corner_point
            self.door_point_right = door_point
        else: 
            self.door_point_right = corner_point
            self.door_point_left = door_point

        self.door_type = door_type
        self.merged = merged

    def update_s(self, s):
        """Update parameter s and move the circle center along the bisector."""

        if s < 0 or s > self.s_max:
            raise ValueError(f"s must be in [0, {self.s_max}]")

        self.s = s

        # Compute new center along bisector
        dx = cos(self.bisector_direction)
        dy = sin(self.bisector_direction)

        new_x = self.canonical_center.x + s * dx
        new_y = self.canonical_center.y + s * dy

        # Update center consistently
        self.center.x = new_x
        self.center.y = new_y

        # Keep parent class fields in sync
        self.xc = new_x
        self.yc = new_y
        
        self.number_of_shifts += 1


class IntermediateCirclesSequence:
    """Ordered sequence of IntermediateCircle objects with basic sequence operations."""


    def __init__(self, items=None):
        # items can be None or any iterable of IntermediateCircle
        self._items = list(items) if items is not None else []

        # # Optional validation (comment out if you want max speed)
        # for i, c in enumerate(self._items):
        #     if not isinstance(c, IntermediateCircle):
        #         raise TypeError(
        #             f"All items must be IntermediateCircle. Got {type(c).__name__} at index {i}."
        #         )

    # --- Python sequence protocol ---
    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        if not isinstance(value, IntermediateCircle):
            raise TypeError(f"Expected IntermediateCircle, got {type(value).__name__}")
        self._items[index] = value

    def __iter__(self):
        return iter(self._items)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._items!r})"

    # --- Convenience properties ---
    @property
    def first(self):
        return self._items[0]

    @property
    def last(self):
        return self._items[-1]

    # --- Mutating operations ---
    def append(self, circle: "IntermediateCircle"):
        if not isinstance(circle, IntermediateCircle):
            raise TypeError(f"Expected IntermediateCircle, got {type(circle).__name__}")
        self._items.append(circle)

    def insert(self, index: int, circle: "IntermediateCircle"):
        if not isinstance(circle, IntermediateCircle):
            raise TypeError(f"Expected IntermediateCircle, got {type(circle).__name__}")
        self._items.insert(index, circle)

    def remove_at(self, index: int):
        """Remove item at index (in-place)."""
        del self._items[index]

    def pop(self, index: int = -1):
        """Remove and return item at index (default last)."""
        return self._items.pop(index)

    def clear(self):
        self._items.clear()

    # --- Domain helpers ---
    def corner_points(self):
        """Return the corner points in the same order."""
        return [c.corner_point for c in self._items]

    def turn_directions(self):
        """Return the turn directions in the same order."""
        return [c.turn_direction for c in self._items]

    def centers(self):
        return [c.center for c in self._items]

    def radii(self):
        return [c.radius for c in self._items]
    

class IntermediateCircleChoice:
    """
    Possible IntermediateCircle candidates associated with one corridor transition.

    Usually contains one circle.
    In ambiguous cases, contains two circles: one left and one right.
    """

    def __init__(
        self,
        candidates=None,
        index=None,
        corridor_index_start=None,
        corridor_index_end=None,
    ):
        self._candidates = list(candidates) if candidates is not None else []

        # Position of this choice in the current choice sequence
        self.index = index

        # Indices of the original corridors connected by this choice
        self.corridor_index_start = corridor_index_start
        self.corridor_index_end = corridor_index_end

        self.preferred_turn_direction = None

        for k, circle in enumerate(self._candidates):
            if not isinstance(circle, IntermediateCircle):
                raise TypeError(
                    f"Expected IntermediateCircle at candidate {k}, "
                    f"got {type(circle).__name__}"
                )

    # --- Python sequence protocol ---
    def __len__(self):
        return len(self._candidates)

    def __getitem__(self, index):
        return self._candidates[index]

    def __iter__(self):
        return iter(self._candidates)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index={self.index!r}, "
            f"corridors=({self.corridor_index_start!r}, "
            f"{self.corridor_index_end!r}), "
            f"candidates={self._candidates!r})"
        )

    # --- Convenience properties ---
    @property
    def candidates(self):
        return self._candidates

    @property
    def first(self):
        return self._candidates[0]

    @property
    def is_ambiguous(self):
        return len(self._candidates) > 1

    @property
    def num_candidates(self):
        return len(self._candidates)

    # --- Mutating operations ---
    def append(self, circle):
        if not isinstance(circle, IntermediateCircle):
            raise TypeError(f"Expected IntermediateCircle, got {type(circle).__name__}")
        self._candidates.append(circle)

    # --- Domain helpers ---
    def turn_directions(self):
        return [c.turn_direction for c in self._candidates]

    def centers(self):
        return [c.center for c in self._candidates]

    def radii(self):
        return [c.radius for c in self._candidates]

    def corner_points(self):
        return [c.corner_point for c in self._candidates]


class IntermediateCircleChoicesSequence:
    """
    Ordered sequence of IntermediateCircleChoice objects.

    Each item represents one corridor transition.
    Each item may contain one or more candidate circles.
    """


    def __init__(self, choices=None):
        self._choices = list(choices) if choices is not None else []

        for i, choice in enumerate(self._choices):
            if not isinstance(choice, IntermediateCircleChoice):
                raise TypeError(
                    f"Expected IntermediateCircleChoice at index {i}, "
                    f"got {type(choice).__name__}"
                )

    # --- Python sequence protocol ---
    def __len__(self):
        return len(self._choices)

    def __getitem__(self, index):
        return self._choices[index]

    def __iter__(self):
        return iter(self._choices)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._choices!r})"

    # --- Convenience properties ---
    @property
    def first(self):
        return self._choices[0]

    @property
    def last(self):
        return self._choices[-1]

    @property
    def choices(self):
        return self._choices

    # --- Mutating operations ---
    def append(self, choice):
        if not isinstance(choice, IntermediateCircleChoice):
            raise TypeError(f"Expected IntermediateCircleChoice, got {type(choice).__name__}")
        self._choices.append(choice)

    def clear(self):
        self._choices.clear()

    # --- Domain helpers ---
    def ambiguous_indices(self):
        return [i for i, choice in enumerate(self._choices) if choice.is_ambiguous]

    def num_candidates_per_choice(self):
        return [len(choice) for choice in self._choices]

    def has_ambiguities(self):
        return any(choice.is_ambiguous for choice in self._choices)

    def selected_sequence(self, candidate_indices=None):
        """
        Build one concrete IntermediateCirclesSequence by selecting one candidate
        from each IntermediateCircleChoice.

        If candidate_indices is None, select candidate 0 everywhere.
        """

        if candidate_indices is None:
            candidate_indices = [0] * len(self._choices)

        if len(candidate_indices) != len(self._choices):
            raise ValueError(
                "candidate_indices must have the same length as the number of choices"
            )

        selected_circles = [None] * len(self._choices)

        for i, candidate_index in enumerate(candidate_indices):
            selected_circles[i] = self._choices[i][candidate_index]

        return IntermediateCirclesSequence(selected_circles)
    
    def replace_two_with_one(self, index, new_choice):
        """
        Replace choices[index] and choices[index + 1] with one new choice.
        """

        if not isinstance(new_choice, IntermediateCircleChoice):
            raise TypeError(
                f"Expected IntermediateCircleChoice, got {type(new_choice).__name__}"
            )

        self._choices[index] = new_choice
        del self._choices[index + 1]

    def remove_at(self, index: int):
        """Remove choice at index in-place."""
        del self._choices[index]