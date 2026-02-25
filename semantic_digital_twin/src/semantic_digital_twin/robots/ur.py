from __future__ import annotations

from collections import defaultdict
from dataclasses import field, dataclass
from typing import Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm, Manipulator
from semantic_digital_twin.robots.robot_mixins import SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import Vector3, Quaternion
from semantic_digital_twin.world import World


@dataclass
class EndEffectorTool(Manipulator):

    def assign_to_robot(self, robot: AbstractRobot):
        """ """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"EndEffectorTool {self.name} is already part of another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def __hash__(self):
        """
        Returns the hash of the kinematic chain, which is based on the root and tip bodies.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.name, self.root, self.tool_frame))


@dataclass(eq=False)
class UR(AbstractRobot, SpecifiesLeftRightArm):
    """
    Represents a ur arm.
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def setup_collision_config(self): ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a UR robot semantic annotation from the given world.

        :param world: The world from which to create the robot semantic annotation.

        :return: A UR robot semantic annotation.
        """
        with world.modify_world():
            robot = cls(
                name=PrefixedName(name="ur", prefix=world.name),
                root=world.get_body_by_name("base_link"),
                _world=world,
            )
            manipulator = EndEffectorTool(
                tool_frame=world.get_body_by_name("tool1"),
                front_facing_axis=Vector3(z=1),
                front_facing_orientation=Quaternion(0.5, -0.5, 0.5, -0.5),
            )
            arm = Arm(
                name=PrefixedName("arm", prefix=robot.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("tool0"),
                manipulator=manipulator,
                _world=world,
            )

            robot.add_arm(arm)
            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(lambda: 0.2)
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

            return robot


@dataclass(eq=False)
class UR10Bolt(AbstractRobot, SpecifiesLeftRightArm):
    """
    Represents a ur arm.
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def setup_collision_config(self): ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a UR robot semantic annotation from the given world.

        :param world: The world from which to create the robot semantic annotation.

        :return: A UR robot semantic annotation.
        """
        with world.modify_world():
            robot = cls(
                name=PrefixedName(name="ur", prefix=world.name),
                root=world.get_body_by_name("base_link"),
                _world=world,
            )
            manipulator = EndEffectorTool(
                tool_frame=world.get_body_by_name("tool1"),
                front_facing_axis=Vector3(z=1),
                front_facing_orientation=Quaternion(0.5, -0.5, 0.5, -0.5),
                _world=world,
            )
            arm = Arm(
                name=PrefixedName("arm", prefix=robot.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("tool0"),
                manipulator=manipulator,
                _world=world,
            )

            robot.add_arm(arm)
            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(lambda: 0.1)
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

            return robot
