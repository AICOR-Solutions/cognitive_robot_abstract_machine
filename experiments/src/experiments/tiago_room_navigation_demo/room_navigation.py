"""
Driving a robot from the room it is in to another room of its building, riding the
elevator when the other room lies on another level.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import SequentialAction
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    PathPlanningNavigateAction,
)
from cramph.node import StatechartNode
from experiments.tiago_room_navigation_demo.exceptions import (
    NoElevatorError,
    RoomOnNoLevelError,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
    Room,
)

# %% navigating to a room


@dataclass(eq=False, repr=False, kw_only=True)
class NavigateToRoomAction(SequentialAction):
    """
    Drives the robot to the middle of a room along paths through the free space of each
    level it crosses.

    A room on the robot's own level is driven to directly. For a room on another level
    the robot drives in front of the elevator's doors, rides the elevator to the room's
    level and drives on from there.

    .. note:: The robot rides the first elevator of its world.
    """

    target_room: Room
    """
    The room the robot should end up in.
    """

    elevator_waiting_distance: float = 1.5
    """
    How far in front of the elevator's doors the robot drives before riding it.
    """

    elevator_arrival_threshold: float = 0.01
    """
    Position error within which the elevator's drive counts as having arrived.
    """

    def create_steps(self) -> list[StatechartNode]:
        """
        :raises RoomOnNoLevelError: If the target room lies on no level.
        :raises NoElevatorError: If the target room lies on another level than the
            robot and the world holds no elevator.
        """
        target_level = Level.under(self.target_room.floor.root)
        if target_level is None:
            raise RoomOnNoLevelError(self.target_room.name.name)
        drive_to_room = PathPlanningNavigateAction(target=self.target_room.floor_centre)
        if Level.under(self.robot.root) is target_level:
            return [drive_to_room]
        elevators = self.world.get_semantic_annotations_by_type(Elevator)
        if not elevators:
            raise NoElevatorError(self.target_room.name.name)
        elevator = elevators[0]
        return [
            PathPlanningNavigateAction(
                target=elevator.pose_in_front_of_doors(self.elevator_waiting_distance)
            ),
            *self._ride_elevator(elevator, target_level),
            drive_to_room,
        ]

    def _ride_elevator(
        self, elevator: Elevator, target_level: Level
    ) -> list[StatechartNode]:
        """
        :param elevator: The elevator the robot waits in front of.
        :param target_level: The level the robot rides to.
        :return: The steps that take the robot from in front of the elevator's doors
            out of the cabin on ``target_level``.
        """
        return [
            ElevatorNavigation(
                elevator=elevator,
                target_floor=target_level,
                arrival_threshold=self.elevator_arrival_threshold,
            )
        ]
