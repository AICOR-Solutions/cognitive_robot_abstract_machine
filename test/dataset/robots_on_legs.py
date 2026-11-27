"""
A robot whose base walks on a leg, and a walking controller that records what it is
told, for tests of a physical simulation walking a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, PositionServo
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.walking import (
    PlanarVelocity,
    StandingBalance,
    WalkingController,
    WalkingPace,
    WalkingSenses,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
    ScrewAxisConnection,
)

from .robots_on_a_drive import RootBodyBase

# %% a controller recording what it is told


@dataclass
class RecordingWalkingController(WalkingController):
    """
    Drives one joint, named ``leg``, to a fixed position and records every call.
    """

    leg_target: float = 0.5
    """
    The position the leg is always driven to.
    """

    calls: list[tuple[WalkingSenses, PlanarVelocity]] = field(default_factory=list)
    """
    What the controller was told at every call since the last reset.
    """

    @classproperty
    def pace(cls) -> WalkingPace:
        return WalkingPace(
            slowing_rate=1.0,
            slowest_speed=0.1,
            fastest_speed=0.5,
            slowest_turn=0.1,
            fastest_turn=0.5,
            arrival_tolerance=0.05,
        )

    @classproperty
    def balance(cls) -> StandingBalance:
        return StandingBalance(
            foot_names=["shin"],
            support_margin=0.01,
            maximal_center_of_mass_speed=0.1,
            maximal_center_of_mass_acceleration=0.5,
        )

    @property
    def control_period(self) -> float:
        return 0.02

    @property
    def observed_joint_names(self) -> list[str]:
        return ["leg"]

    @property
    def driven_joint_names(self) -> list[str]:
        return ["leg"]

    @property
    def standing_posture(self) -> dict[str, float]:
        return {"leg": 0.3}

    def reset(self) -> None:
        self.calls = []

    def leg_targets(
        self, senses: WalkingSenses, command: PlanarVelocity
    ) -> dict[str, float]:
        self.calls.append((senses, command))
        return {"leg": self.leg_target}


# %% a robot standing on a leg


@dataclass(eq=False)
class BaseOnALeg(RootBodyBase[OmniDrive]):
    """
    A mobile base that is the root body of its robot and walks with a
    :class:`RecordingWalkingController`.
    """

    @classproperty
    def walking_controller_type(cls) -> type[RecordingWalkingController]:
        return RecordingWalkingController


@dataclass(eq=False)
class RobotOnALeg(MinimalRobot, HasMobileBase[BaseOnALeg]):
    """
    A minimal robot whose root body is its mobile base, which walks on a leg.
    """


# %% a world with the robot


def robot_on_a_leg_drive(odometry_height: float = 1.0) -> OmniDrive:
    """
    :return: The drive of a box-shaped base hanging a servoed leg, whose robot walks with
        a :class:`RecordingWalkingController`. The base's box is turned within the base,
        so its principal axes differ from the base's own.

    :param odometry_height: How high above the world's root the drive's odometry is
        localized.
    """
    robot_world = World()
    with robot_world.modify_world():
        pelvis = Body(name=PrefixedName("pelvis"))
        pelvis.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        yaw=0.5, roll=0.3, reference_frame=pelvis
                    ),
                    scale=Scale(0.4, 0.2, 0.1),
                )
            ],
            reference_frame=pelvis,
        )
        robot_world.add_body(pelvis)
        shin = Body(name=PrefixedName("shin"))
        shin.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-0.2, reference_frame=shin
                    ),
                    scale=Scale(0.05, 0.05, 0.3),
                )
            ],
            reference_frame=shin,
        )
        leg_position = DegreeOfFreedom(
            name=PrefixedName("leg"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0
            ),
        )
        robot_world.add_degree_of_freedom(leg_position)
        robot_world.add_connection(
            ScrewAxisConnection(
                name=PrefixedName("leg"),
                parent=pelvis,
                child=shin,
                screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
                    axis=Vector3.Y(reference_frame=shin)
                ),
                raw_dof=leg_position,
            )
        )
        RobotOnALeg.from_world(robot_world)
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
        odometry = Body(name=PrefixedName("odom"))
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=odometry,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=odometry_height, reference_frame=world.root
                ),
            )
        )
        drive = OmniDrive.create_with_dofs(
            world=world, parent=odometry, child=robot_world.root
        )
        world.merge_world(robot_world, drive)
        servo = PositionServo(
            name=PrefixedName("leg_servo"),
            gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=20.0),
        )
        servo.add_dof(world.get_connection_by_name("leg").raw_dof)
        world.add_actuator(servo)
    return drive
