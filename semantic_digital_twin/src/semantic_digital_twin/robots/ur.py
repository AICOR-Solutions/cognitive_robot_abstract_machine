from __future__ import annotations

from collections import defaultdict
from dataclasses import field, dataclass
from typing import Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm, Manipulator
from semantic_digital_twin.robots.robot_mixins import HasArms, SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import Vector3, Quaternion
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)


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
class UR10Bolt(AbstractRobot, HasArms):
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
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="ur", prefix=world.name),
            root=world.get_body_by_name("base_link"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        manipulator = EndEffectorTool(
            name=PrefixedName("end_effector", prefix=self.name.name),
            root=self._world.get_body_by_name("tool0"),
            tool_frame=self._world.get_body_by_name("tool1"),
            front_facing_axis=Vector3(z=1),
            front_facing_orientation=Quaternion(0.5, -0.5, 0.5, -0.5),
            _world=self._world,
        )
        arm = Arm(
            name=PrefixedName("arm", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("tool0"),
            manipulator=manipulator,
            _world=self._world,
        )
        self.add_arm(arm)

    def _setup_collision_rules(self):
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 0.5)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        # Mark all active joints of the UR10 arm as having hardware interfaces
        if not self.arms:
            return
        for connection in self.arms[0].connections:
            if isinstance(connection, ActiveConnection):
                connection.has_hardware_interface = True

    def _setup_joint_states(self):
        # Create a simple PARK joint state with zeros for the 6 arm joints
        if not self.arms:
            return
        arm = self.arms[0]
        active_conns = [
            c for c in arm.connections if not isinstance(c, FixedConnection)
        ]
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(zip(active_conns, [0.0] * len(active_conns))),
            state_type=StaticJointState.PARK,
        )
        arm.add_joint_state(arm_park)
