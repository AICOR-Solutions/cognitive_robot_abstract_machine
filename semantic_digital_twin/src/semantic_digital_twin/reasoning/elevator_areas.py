from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)
from semantic_digital_twin.spatial_types import Point3

# %% areas people occupy around an elevator


@dataclass
class ElevatorArea(ABC):
    """
    A region of space belonging to an elevator, in which the robot has to reckon with
    people when it uses that elevator.
    """

    elevator: Elevator
    """
    The elevator the area belongs to.
    """

    @abstractmethod
    def contains(self, point: Point3) -> bool:
        """
        :param point: A point with a reference frame in the elevator's world.
        :return: Whether the point lies inside this area.
        """

    @abstractmethod
    def footprint(self) -> list[Point3]:
        """
        :return: The corners of the area's outline on its floor, counter-clockwise, in
            the world root frame.
        """

    def _point_in_cabin_frame(self, point: Point3) -> Point3:
        """
        :return: ``point`` expressed in the frame of the elevator's cabin.
        """
        return self.elevator._world.transform(point, self.elevator.root)

    def _rectangle_corners(
        self, x_min: float, x_max: float, y_min: float, y_max: float, z: float
    ) -> list[Point3]:
        """
        :return: The corners of the rectangle at height ``z`` that is axis-aligned in
            the cabin frame, in the world root frame and counter-clockwise.
        """
        world = self.elevator._world
        return [
            world.transform(
                Point3(x, y, z, reference_frame=self.elevator.root), world.root
            )
            for x, y in ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
        ]


@dataclass
class ElevatorCabinArea(ElevatorArea):
    """
    The inside of the elevator cabin, wherever the cabin currently is.
    """

    def contains(self, point: Point3) -> bool:
        cabin_P_point = self._point_in_cabin_frame(point)
        scale = self.elevator.scale
        return (
            abs(float(cabin_P_point.x)) <= scale.x / 2
            and abs(float(cabin_P_point.y)) <= scale.y / 2
            and abs(float(cabin_P_point.z)) <= scale.z / 2
        )

    def footprint(self) -> list[Point3]:
        scale = self.elevator.scale
        return self._rectangle_corners(
            -scale.x / 2, scale.x / 2, -scale.y / 2, scale.y / 2, -scale.z / 2
        )


@dataclass
class ElevatorWaitingArea(ElevatorArea):
    """
    The space in front of the elevator's doors on one level, where people wait for the
    cabin and where the robot waits before boarding.
    """

    level: Level
    """
    The level on which the area lies.
    """

    depth: float = 1.5
    """
    How far the area reaches out from the elevator's doors.
    """

    side_margin: float = 0.3
    """
    How far the area reaches beyond the cabin's width on either side.
    """

    height: float = 2.5
    """
    How far the area reaches up from the level's floor.
    """

    def contains(self, point: Point3) -> bool:
        cabin_P_point = self._point_in_cabin_frame(point)
        door_x = -self.elevator.scale.x / 2
        half_width = self.elevator.scale.y / 2 + self.side_margin
        floor_height = float(self.level.floor_plane[0].z)
        point_height = float(
            self.elevator._world.transform(point, self.elevator._world.root).z
        )
        return (
            door_x - self.depth <= float(cabin_P_point.x) <= door_x
            and abs(float(cabin_P_point.y)) <= half_width
            and floor_height <= point_height <= floor_height + self.height
        )

    def footprint(self) -> list[Point3]:
        door_x = -self.elevator.scale.x / 2
        half_width = self.elevator.scale.y / 2 + self.side_margin
        floor_height = float(self.level.floor_plane[0].z)
        corners = self._rectangle_corners(
            door_x - self.depth, door_x, -half_width, half_width, 0.0
        )
        return [
            Point3(
                corner.x, corner.y, floor_height, reference_frame=corner.reference_frame
            )
            for corner in corners
        ]
