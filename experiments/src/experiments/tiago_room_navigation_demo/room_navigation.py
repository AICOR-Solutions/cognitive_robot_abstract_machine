"""
Driving a robot from the room it is in to another room of its building, riding the
elevator when the other room lies on another level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    PathPlanningNavigateAction,
)
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


@dataclass
class NavigateToRoomAction(ActionDescription):
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

    elevator_waiting_distance: float = field(default=1.5, kw_only=True)
    """
    How far in front of the elevator's doors the robot drives before riding it.
    """

    elevator_arrival_threshold: float = field(default=0.01, kw_only=True)
    """
    Position error within which the elevator's drive counts as having arrived.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        :raises RoomOnNoLevelError: If the target room lies on no level.
        :raises NoElevatorError: If the target room lies on another level than the
            robot and the world holds no elevator.
        """
        target_level = Level.under(self.target_room.floor.root)
        if target_level is None:
            raise RoomOnNoLevelError(self.target_room.name.name)
        drive_to_room = PathPlanningNavigateAction(self.target_room.floor_centre)
        if Level.under(self.robot.root) is target_level:
            return sequential([drive_to_room])
        elevators = self.world.get_semantic_annotations_by_type(Elevator)
        if not elevators:
            raise NoElevatorError(self.target_room.name.name)
        elevator = elevators[0]
        return sequential(
            [
                PathPlanningNavigateAction(
                    elevator.pose_in_front_of_doors(self.elevator_waiting_distance)
                ),
                ElevatorNavigation(
                    elevator,
                    target_level,
                    arrival_threshold=self.elevator_arrival_threshold,
                ),
                drive_to_room,
            ]
        )
