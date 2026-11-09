"""
Tiago drives from a start room to a given room, riding the elevator when the room lies
on another level.

In a kinematic simulation an
:class:`~experiments.tiago_elevator_demo.demo.ElevatorOperator` sends the elevator to the
target room's level once Tiago boards it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.tiago_arise_gazebo.entity_teleport import SimulatedEntityTeleport
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.exceptions import (
    RobotNotPlacedError,
    RoomOnNoLevelError,
    TargetRoomNotReachedError,
)
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Level,
)
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoRoomNavigationDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives from a start room into a target room.
    """

    ros_node_name: str = "tiago_room_navigation_demo_node"

    target_room_name: str = field(kw_only=True)
    """
    The name of the room annotation Tiago drives to.
    """

    start_room_name: str | None = field(default=None, kw_only=True)
    """
    The name of the room annotation Tiago starts in, see :meth:`start_pose`, or None to
    start where Tiago was spawned.
    """

    obstacle_clearance: float = field(default=0.0, kw_only=True)
    """
    How far Tiago's base keeps from obstacles on top of its base radius while it drives.
    """

    elevator_exit_clearance: float = field(
        default=NavigateToRoomAction.elevator_exit_clearance, kw_only=True
    )
    """
    Distance Tiago keeps from the elevator's opening after driving out of the cabin, on
    top of half the cabin's depth.
    """

    entity_teleport_service: str | None = field(default=None, kw_only=True)
    """
    The service through which a real run moves the simulated Tiago into the start room,
    or None where Tiago is a real robot that starts wherever it stands.
    """

    simulated_robot_name: str = field(default="tiago", kw_only=True)
    """
    The name of Tiago's model in the simulated world.
    """

    placement_timeout: float = field(default=30.0, kw_only=True)
    """
    Seconds a real run waits for the world to show Tiago in the start room.
    """

    placement_tolerance: float = field(default=0.05, kw_only=True)
    """
    How far Tiago may stand from :meth:`start_pose` and still count as placed there.
    """

    def acquire_world(self) -> World:
        """
        Differs from :meth:`TiagoElevatorRideDemonstration.acquire_world` in putting
        Tiago into the start room, see :meth:`place_robot_in_start_room`.
        """
        world = super().acquire_world()
        self.place_robot_in_start_room(world)
        return world

    def start_pose(self, world: World) -> Pose:
        """
        :return: The middle of the start room's floor at the height of the floor's
            frame, facing along the x-axis of ``world``'s root.
        """
        start_room = world.get_semantic_annotation_by_name(self.start_room_name)
        centre = start_room.floor_centre
        return Pose.from_xyz_rpy(
            x=float(centre.x),
            y=float(centre.y),
            z=float(start_room.floor.root.global_transform.z),
            reference_frame=world.root,
        )

    def place_robot_in_start_room(self, world: World) -> None:
        """
        Stand Tiago on the map at :meth:`start_pose`.

        A real run moves the simulated Tiago there and waits for the world to show it
        there; a real Tiago, and Tiago without a start room, are left where they stand.

        :raises RobotNotPlacedError: If a real run's world does not show Tiago in the
            start room within :attr:`placement_timeout`.
        """
        if self.start_room_name is None:
            return
        if self.runs_on_a_real_robot and self.entity_teleport_service is None:
            return
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        start = self.start_pose(world)
        self._stand_robot_on_map(robot, float(start.z))
        if not self.runs_on_a_real_robot:
            robot.root.parent_connection.origin = start.homogeneous_matrix
            return
        SimulatedEntityTeleport(
            node=self.ros_node, service_name=self.entity_teleport_service
        ).teleport(self.simulated_robot_name, start)
        self._wait_until_robot_stands_at(robot, start)

    def _stand_robot_on_map(self, robot: AbstractRobot, height: float) -> None:
        """
        Hang ``robot``'s drive off the map at ``height``, keeping where on the map it
        stands.
        """
        world = robot._world
        world.move_branch(robot.root, world.root)
        lift = height - float(robot.root.global_transform.z)
        if np.isclose(lift, 0.0):
            return
        drive = robot.root.parent_connection
        lifted_drive = drive.copy_with_new_parent(
            world.root,
            HomogeneousTransformationMatrix(
                (
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=lift)
                    @ drive.parent_T_connection_expression
                ).evaluate()
            ),
        )
        with world.modify_world():
            world.remove_connection(drive)
            world.add_connection(lifted_drive)

    def _wait_until_robot_stands_at(self, robot: AbstractRobot, pose: Pose) -> None:
        """
        :raises RobotNotPlacedError: If ``robot`` does not stand within
            :attr:`placement_tolerance` of ``pose`` within :attr:`placement_timeout`.
        """
        deadline = time.monotonic() + self.placement_timeout
        while (
            np.linalg.norm(
                robot.root.global_transform.position.to_np()[:2]
                - pose.position.to_np()[:2]
            )
            > self.placement_tolerance
        ):
            if time.monotonic() > deadline:
                raise RobotNotPlacedError(self.start_room_name)
            time.sleep(0.1)

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Drive to the target room.
        """
        statechart = Statechart(context=context)
        statechart.add_node(
            NavigateToRoomAction(
                target_room=context.world.get_semantic_annotation_by_name(
                    self.target_room_name
                ),
                elevator_arrival_threshold=self.elevator_arrival_threshold,
                obstacle_clearance=self.obstacle_clearance,
                elevator_exit_clearance=self.elevator_exit_clearance,
            )
        )
        return statechart

    def destination_level(self, world: World) -> Level:
        """
        :return: The level the target room lies on.
        :raises RoomOnNoLevelError: If the target room lies on no level.
        """
        target_room = world.get_semantic_annotation_by_name(self.target_room_name)
        level = Level.under(target_room.floor.root)
        if level is None:
            raise RoomOnNoLevelError(self.target_room_name)
        return level

    def conclude(self, world: World) -> None:
        """
        :raises TargetRoomNotReachedError: If Tiago does not stand on the target room's
            floor.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        target_room = world.get_semantic_annotation_by_name(self.target_room_name)
        if Floor.under(robot.root) is not target_room.floor:
            raise TargetRoomNotReachedError(self.target_room_name)


# %% the IAI's office


@dataclass
class TiagoIAIOfficeRoomNavigationDemonstration(TiagoRoomNavigationDemonstration):
    """
    Tiago drives from a start room into a target room of the IAI's office.
    """

    office: ElevatorBuilding = field(default_factory=IAIOffice)

    start_room_name: str | None = field(default="0.31", kw_only=True)

    target_room_name: str = field(default="1.54", kw_only=True)

    obstacle_clearance: float = field(default=0.15, kw_only=True)

    elevator_exit_clearance: float = field(default=0.7, kw_only=True)
