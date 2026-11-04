from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


from coraplex.exceptions import NoFloorBelowRobot, NotOnASingleLevelException
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import (
    Action,
    GoalAction,
    ParallelAction,
    SequentialAction,
)
from coraplex.robot_plans.actions.core.speech import SayAction
from coraplex.robot_plans.mixins import ReattachesBranches
from cramph.composites import PausedUntilTrue
from giskardpy.motion_statechart.goals.base_pose_goals import (
    MobileBasePoseGoal,
    DifferentialDriveBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.monitors.joint_monitors import (
    JointPositionReached,
)
from giskardpy.motion_statechart.monitors.sensor_monitors import PassageClear
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.robots.exceptions import MissingLidarError
from semantic_digital_twin.robots.robot_part_mixins import HasLidar
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.semantic_annotations.mixins import HasPlanarFreeSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Level,
    Elevator,
    Floor,
)
from krrood_spatial_math.spatial_types import (
    Pose,
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Region

# %% driving the base


@dataclass(eq=False, repr=False, kw_only=True)
class DrivesBase(Action, ABC):
    """
    Base class for the actions that move the robot's base to a pose.
    """

    base_pose_threshold: float = DifferentialDriveBaseGoal.threshold
    """
    How far the base may be from a pose it drives to, in meters and in radians, to count
    as arrived.
    """

    base_goal_type: type[MobileBasePoseGoal] = field(
        default=HolonomicBaseGoal, init=False
    )
    """
    The kind of node putting the base at the poses this action drives it to, as set by
    the plan transformations configuring the action.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NavigateAction(DrivesBase, GoalAction):
    """
    Navigates the Robot to a position.
    """

    target_location: Pose
    """
    Where the robot should stand, and which way it should face given as the pose's
    x-axis.
    """

    def create_goal(self) -> StatechartNode:
        return self.base_goal_type.moving_base_of(
            self.robot,
            self.robot.mobile_base.pose_facing(self.target_location),
            self.base_pose_threshold,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class LookAtAction(ParallelAction):
    """
    Lets the robot look at a position, following it while the look runs when it is given
    relative to a body that moves.
    """

    target: Pose
    """
    Position at which the robot should look, given as 6D pose.
    """

    camera: Camera | None = None
    """
    Camera that should be looking at the target.
    """

    def create_steps(self) -> list[StatechartNode]:
        camera = self.camera or self.robot.get_default_camera()
        pointing = Pointing(
            root_link=self.robot.get_torso().root,
            tip_link=camera.root,
            goal_point=self._world_P_target(),
            pointing_axis=camera.viewing_direction,
        )
        return [pointing, *self.keep_base_still()]

    def _world_P_target(self) -> Point3:
        """
        :return: Where :attr:`target` is in the world frame, as an expression of where
            the body it is given relative to stands while the look runs.
        """
        target_frame = self.target.reference_frame or self.world.root
        world_T_target_frame = self.world.compose_forward_kinematics_expression(
            self.world.root, target_frame
        )
        world_T_target_frame.reference_frame = self.world.root
        world_T_target_frame.child_frame = target_frame
        return world_T_target_frame @ self.target.position


@dataclass(eq=False, repr=False, kw_only=True)
class FaceAtAction(ParallelAction):
    """
    Turns the robot's base on the spot until its front faces a target.

    The base keeps the position it has when the action starts, so the turn is towards
    the target from wherever an earlier action left it.
    """

    target: Pose
    """
    What to face; only its horizontal position matters.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            Pointing(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_point=self._target_at_base_height(),
                pointing_axis=Vector3(
                    *self.robot.mobile_base.forward_axis.to_np()[:3],
                    reference_frame=self.robot.root,
                ),
            ),
            CartesianPosition(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_point=Point3(reference_frame=self.robot.root),
            ),
        ]

    def _target_at_base_height(self) -> Point3:
        """
        :return: :attr:`target` moved vertically to the height of the base, which can
            only turn about the vertical and so can only point level.
        """
        root_P_target = self.world.transform(self.target, self.world.root).position
        root_P_target.z = self.robot.root.global_pose.z
        return root_P_target


@dataclass(eq=False, repr=False, kw_only=True)
class PathPlanningNavigateAction(DrivesBase, SequentialAction):
    """
    Navigates the robot to a pose along a path through the environment's free space.

    The free space is decomposed into a graph of convex sets, so the robot drives around
    the furniture and walls between it and the target instead of straight at them. It is
    the free space of the level the robot is on, or, outside every level, of the floor
    it stands on.

    This works for obstacles which are known in the environment beforehand, not for
    those added during navigation.
    """

    target: Pose
    """
    Where the robot should stand at the end of the path, with its base.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            self.base_goal_type.moving_base_of(
                self.robot, waypoint, self.base_pose_threshold
            )
            for waypoint in self._path()
        ]

    @property
    def _ground(self) -> HasPlanarFreeSpace:
        """
        Where the path is laid out: the level whose region alone holds the robot, or
        else the floor it stands on.

        :raises NoFloorBelowRobot: If the robot is on no single level and stands over no
            annotated floor.
        """
        levels = Level.holding(self.robot.bodies_with_collision[0])
        if len(levels) == 1:
            return levels[0]
        return self._floor

    @property
    def _floor(self) -> Floor:
        """
        The floor the robot stands on, whose free space the path is laid out in.

        A world with several storeys puts more than one floor below the robot; the one
        it stands on is the topmost of those it stands within the footprint of.

        :raises NoFloorBelowRobot: If the robot stands over no annotated floor.
        :return: The floor the robot drives on.
        """
        floors_below = [
            floor
            for floor in self.world.get_semantic_annotations_by_type(Floor)
            if self._stands_on(floor)
        ]
        if not floors_below:
            raise NoFloorBelowRobot(self.robot)
        return max(floors_below, key=lambda floor: self._extent_of(floor).max_z)

    def _extent_of(self, floor: Floor) -> VolumetricBoundingBox:
        """
        :param floor: The floor to measure.
        :return: The floor's bounding box in the world's root frame.
        """
        return floor.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=self.world.root)
        ).bounding_box()

    def _stands_on(self, floor: Floor) -> bool:
        """
        :param floor: The floor to test.
        :return: Whether the robot's base rests within this floor's footprint and no
            lower than its top.
        """
        extent = self._extent_of(floor)
        base_pose = self.robot.root.global_pose
        return (
            extent.min_x <= float(base_pose.x) <= extent.max_x
            and extent.min_y <= float(base_pose.y) <= extent.max_y
            and extent.max_z <= float(base_pose.z)
        )

    def _path(self) -> list[Pose]:
        """
        The poses the robot drives to, one per leg of the path.

        Each pose faces the waypoint after it, so the leg leaving a waypoint no longer
        has to begin by turning. The waypoint the robot already stands on is left out,
        and the last pose is the requested target.

        .. note::
            The orientation aims the base's x-axis, which is the axis a drive travels
            along, rather than the base's
            :attr:`~semantic_digital_twin.robots.robot_parts.MobileBase.forward_axis`.
            The two differ on a base whose front is not its direction of travel, and it
            is travel that these orientations exist to line up.

        :return: The poses to drive to, in order.
        """
        waypoints = self._waypoints()
        base_height = self.world.transform(
            self.robot.root.global_transform, waypoints[0].reference_frame
        ).z
        poses = [
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(waypoint.x, waypoint.y, base_height, waypoint.reference_frame),
                RotationMatrix.from_vectors(
                    x=Vector3(
                        next_waypoint.x - waypoint.x,
                        next_waypoint.y - waypoint.y,
                        0,
                        reference_frame=waypoint.reference_frame,
                    ),
                    z=Vector3.Z(),
                    reference_frame=waypoint.reference_frame,
                ),
                reference_frame=waypoint.reference_frame,
            ).pose
            for waypoint, next_waypoint in zip(waypoints[1:], waypoints[2:])
        ]
        return poses + [self.target]

    def _waypoints(self) -> list[Point2]:
        """
        The points the robot travels through to get from where it stands to the target.

        :return: The path, beginning at the robot's own position and ending at the
            target's.
        """
        base_pose = self.robot.root.global_pose
        free_space = self._ground.planar_free_space(
            max_height=self.robot.as_bounding_box_collection_in_frame(self.robot.root)
            .bounding_box()
            .scale.z,
            bloat_obstacles=self.robot.mobile_base.base_radius,
        )
        return free_space.path_from_to(
            Point2.from_pose(base_pose), Point2.from_pose(self.target)
        )


# %% passing a door


@dataclass(eq=False, repr=False, kw_only=True)
class PassDoorAction(SequentialAction):
    """
    Drives the robot through a doorway towards a destination: it lines up in front of
    the doorway on the side away from the destination, waits until its lidar sees a
    passage as wide as its base, and drives through to the destination's side.
    """

    doors: list[Door]
    """
    The doors covering the doorway, such as both leaves of a double door.
    """

    destination: Point3
    """
    A point on the side of the doorway the robot should end up on.
    """

    approach_distance: float = 0.5
    """
    Distance from the doorway at which the robot waits for the passage to clear.
    """

    exit_distance: float = 0.5
    """
    Distance from the doorway at which the robot stops on the other side.
    """

    waiting_announcement: str | None = None
    """
    What the robot says once it stands in front of the doorway, before it waits for the
    passage to clear, or None to wait silently.
    """

    def create_steps(self) -> list[StatechartNode]:
        side = self._destination_side
        steps = [
            NavigateAction(
                target_location=self._pose_at(-side * self.approach_distance, side)
            )
        ]
        if self.waiting_announcement is not None:
            steps.append(SayAction(text=self.waiting_announcement))
        steps.append(
            PausedUntilTrue(
                monitor=self._passage_clear(),
                monitored_node=NavigateAction(
                    target_location=self._pose_at(side * self.exit_distance, side)
                ),
            )
        )
        return steps

    @property
    def _doorway_frame(self) -> Region:
        """
        The frame the doorway is measured in: the entry way of the first door, whose
        x-axis points through the doorway.
        """
        return self.doors[0].entry_way.root

    @property
    def _doorway(self) -> VolumetricBoundingBox:
        """
        The space all entry ways of the doors span together, in :attr:`_doorway_frame`.
        """
        return BoundingBoxCollection(
            shapes=[
                box
                for door in self.doors
                for box in door.entry_way.root.area.as_bounding_box_collection_in_frame(
                    self._doorway_frame
                )
            ],
            reference_frame=self._doorway_frame,
        ).bounding_box()

    @property
    def _destination_side(self) -> int:
        """
        ``1`` if :attr:`destination` lies on the positive x side of the doorway, ``-1``
        otherwise.
        """
        destination_x = self.world.transform(self.destination, self._doorway_frame).x
        return 1 if float(destination_x) >= float(self._doorway.center.x) else -1

    @property
    def _height_in_doorway_frame(self) -> float:
        """
        The robot's height in :attr:`_doorway_frame`, which its drive cannot change.
        """
        return float(
            self.world.transform(
                self.robot.root.global_transform, self._doorway_frame
            ).z
        )

    def _pose_at(self, distance: float, heading: int) -> Pose:
        """
        :param distance: Signed distance from the doorway's centre along the doorway
            frame's x-axis.
        :param heading: ``1`` to face along the doorway frame's x-axis, ``-1`` against
            it.
        :return: A pose centred on the doorway at that distance.
        """
        return Pose.from_xyz_rpy(
            x=float(self._doorway.center.x) + distance,
            y=float(self._doorway.center.y),
            z=self._height_in_doorway_frame,
            yaw=0.0 if heading > 0 else np.pi,
            reference_frame=self._doorway_frame,
        )

    def _passage_clear(self) -> PassageClear:
        """
        Observes True from the first time the robot's lidar sees a passage as wide as
        its base through the centre of the doorway.

        :raises MissingLidarError: If the robot's mobile base carries no lidar.
        """
        mobile_base = self.robot.mobile_base
        if not isinstance(mobile_base, HasLidar):
            raise MissingLidarError(robot_part=mobile_base)
        passage_clear = PassageClear(
            lidar=mobile_base.lidar,
            passage_start=Point3(
                self._doorway.center.x,
                self._doorway.center.y - mobile_base.base_radius,
                self._height_in_doorway_frame,
                reference_frame=self._doorway_frame,
            ),
            passage_end=Point3(
                self._doorway.center.x,
                self._doorway.center.y + mobile_base.base_radius,
                self._height_in_doorway_frame,
                reference_frame=self._doorway_frame,
            ),
            name="DoorwayClear",
            stays_clear_once_seen=True,
        )
        return passage_clear


# %% riding an elevator


class ElevatorRideAnnouncement(StrEnum):
    """
    What a robot riding an elevator says about where it believes it is and where it is
    headed, with the floors left open.
    """

    DEPARTING = "I am on the {current_floor} and heading to the {target_floor}."
    """
    Said before the robot sets off towards the elevator.
    """

    WAITING_AT_DOORS = (
        "I am in front of the elevator and waiting for it on the {current_floor}."
    )
    """
    Said once the robot stands in front of the elevator's doors.
    """

    RIDING = "I am inside the elevator and riding to the {target_floor}."
    """
    Said once the robot has boarded the cabin.
    """

    ARRIVED = "I have arrived on the {target_floor}."
    """
    Said once the robot has left the cabin on the floor it was headed to.
    """

    def verbalize(self, current_floor: Level, target_floor: Level) -> str:
        """
        :param current_floor: The floor the robot sets off from.
        :param target_floor: The floor the robot is headed to.
        :return: This announcement with the floors filled in by their spoken names.
        """
        return self.format(
            current_floor=current_floor.spoken_name,
            target_floor=target_floor.spoken_name,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ElevatorNavigation(ReattachesBranches, SequentialAction):
    """
    Navigates a robot to another level of a building using an elevator, the robot drives
    in the elevator, turns towards the doors and waits there until its lidar sees the
    doors open again and the elevator is at the right level.

    Along the way the robot says where it believes it is and where it is headed, see
    :class:`ElevatorRideAnnouncement`.
    """

    elevator: Elevator
    """
    Elevator the robot rides.
    """

    target_floor: Level
    """
    Level of the building the robot should end up on.
    """

    exit_clearance: float = 0.5
    """
    Distance the robot keeps from the elevator's opening after driving out, on top of
    half the cabin's depth.
    """

    arrival_threshold: float = 0.01
    """
    Position error within which the elevator's drive counts as having arrived.
    """

    def create_steps(self) -> list[StatechartNode]:
        half_cabin_depth = float(self.elevator.scale.x) / 2
        current_floor = self._current_floor
        return [
            SayAction(
                text=ElevatorRideAnnouncement.DEPARTING.verbalize(
                    current_floor, self.target_floor
                )
            ),
            PausedUntilTrue(
                monitor=self._cabin_at_floor(current_floor),
                monitored_node=PassDoorAction(
                    doors=self.elevator.doors,
                    destination=Point3(reference_frame=self.elevator.root),
                    approach_distance=self.exit_clearance,
                    exit_distance=half_cabin_depth,
                    waiting_announcement=ElevatorRideAnnouncement.WAITING_AT_DOORS.verbalize(
                        current_floor, self.target_floor
                    ),
                ),
            ),
            *self.reattach(self.robot.root, self.elevator.root),
            SayAction(
                text=ElevatorRideAnnouncement.RIDING.verbalize(
                    current_floor, self.target_floor
                )
            ),
            PausedUntilTrue(
                monitor=self._cabin_at_floor(self.target_floor),
                monitored_node=PassDoorAction(
                    doors=self.elevator.doors,
                    destination=self._point_outside_cabin,
                    approach_distance=half_cabin_depth,
                    exit_distance=self.exit_clearance,
                ),
            ),
            *self.reattach(self.robot.root, self.world.root),
            SayAction(
                text=ElevatorRideAnnouncement.ARRIVED.verbalize(
                    current_floor, self.target_floor
                )
            ),
        ]

    @property
    def _current_floor(self) -> Level:
        """
        Finds the floor the robot is currently on, based on its position in the world.

        :raises NotOnASingleLevelException: If the robot is not on any floor or on
            multiple floors at once.
        :return: The semantic annotation for the floor
        """
        current_floor = Level.holding(self.robot.bodies_with_collision[0])
        if len(current_floor) == 0:
            raise NotOnASingleLevelException("Robot is not on any recognized floor.")
        if len(current_floor) > 1:
            raise NotOnASingleLevelException("Robot is on multiple floors at once.")
        return current_floor[0]

    @property
    def _point_outside_cabin(self) -> Point3:
        """
        A point a cabin depth beyond the elevator's opening.
        """
        cabin_depth = float(self.elevator.scale.x)
        hole_direction = self.elevator.hole_direction
        return Point3(
            float(hole_direction[0]) * cabin_depth,
            float(hole_direction[1]) * cabin_depth,
            float(hole_direction[2]) * cabin_depth,
            reference_frame=self.elevator.root,
        )

    def _cabin_at_floor(self, floor: Level) -> JointPositionReached:
        """
        Observes True once the cabin serves ``floor``.

        :raises MissingMovableJointError: If the elevator has no drive.
        """
        if self.elevator.movable_joint is None:
            raise MissingMovableJointError(self.elevator)
        return JointPositionReached(
            connection=self.elevator.movable_joint,
            position=self.elevator.drive_position_for_floor(floor),
            threshold=self.arrival_threshold,
            name="ElevatorAtFloor",
        )
