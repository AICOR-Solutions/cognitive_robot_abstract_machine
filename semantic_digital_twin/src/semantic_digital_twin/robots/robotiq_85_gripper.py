from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.ormatic.utils import classproperty
from typing_extensions import Generic

from semantic_digital_twin.robots.robot_part_mixins import (
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
)
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.exceptions import CommandedJointNotInJointState
from semantic_digital_twin.robots.robot_parts import (
    EndEffector,
    HasGripperActionServers,
    ServoedRobotPart,
)
from semantic_digital_twin.world_description.connection_properties import (
    JointDrive,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


# %% driving a gripper through a parallel gripper command server


@dataclass(eq=False)
class HasParallelGripperCommandServer(HasGripperActionServers, ABC):
    """
    A gripper whose driver serves a ``control_msgs/ParallelGripperCommand`` action,
    which brings the real gripper into every state.
    """

    @classproperty
    @abstractmethod
    def parallel_gripper_command_topic(cls) -> str:
        """
        The name the driver serves the parallel gripper command action under.
        """

    @property
    @abstractmethod
    def commanded_joint(self) -> ActiveConnection1DOF:
        """
        The joint whose position the server is commanded, which every other joint of the
        gripper follows.
        """

    def commanded_position(self, joint_state: JointState) -> float:
        """
        :param joint_state: A joint state of this gripper.
        :return: The position the server is commanded to bring the gripper into
            `joint_state`: the position `joint_state` sets for :attr:`commanded_joint`,
            in that joint's units.
        :raises CommandedJointNotInJointState: If `joint_state` sets no position for
            :attr:`commanded_joint`.
        """
        for connection, position in joint_state.items():
            if connection is self.commanded_joint:
                return position
        raise CommandedJointNotInJointState(gripper=self, joint_state=joint_state)

    def action_topic_serving(self, state_type: JointStateType | None) -> str:
        return self.parallel_gripper_command_topic


# %% the Robotiq 2F-85


@dataclass(eq=False)
class Robotiq85Gripper(
    EndEffector,
    HasTwoFingers[TGenericLeftFinger, TGenericRightFinger],
    ServoedRobotPart,
    Generic[TGenericLeftFinger, TGenericRightFinger],
    ABC,
):
    """
    A Robotiq 2F-85 parallel gripper: two fingers coupled by a mimic linkage, driven by
    the knuckle joint of the thumb.

    Knows what a grasp and a physical simulation need from it: the joint that drives
    the fingers and the servo driving it; the pads that meet an object are the tips of
    its thumb and finger.

    The servo is raised empirically, since no pre-tuned reference exists for the
    Robotiq 2F-85; its armature gives the coupled mechanism the numerical damping that
    keeps it from chattering under load.
    """

    @property
    def knuckle_joint(self) -> ActiveConnection1DOF:
        """
        The joint that actually drives the gripper; every other finger joint in the
        mimic linkage follows it.
        """
        return self.thumb.root.parent_connection

    @classproperty
    def holds_its_own_weight(cls) -> bool:
        return True

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        Every finger joint of the linkage, with the same drive.
        """
        finger_drive = JointDrive(
            gains=ServoGains(stiffness=100.0, damping=10.0, torque_limit=10.0),
            armature=0.05,
            joint_damping=0.0,
        )
        return {
            connection.name.name: finger_drive
            for connection in self.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        }
