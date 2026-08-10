"""
Tests for the synchronizers that write ROS topics into the world state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import pytest
from nav_msgs.msg import Odometry
from numpy.testing import assert_allclose
from sensor_msgs.msg import JointState

from giskardpy.middleware.ros2.exceptions import UnboundMessageTypeError
from giskardpy.middleware.ros2.input_synchronization import (
    LatestJointStateSynchronizer,
    OdometrySynchronizer,
    PendingJointStateSynchronizer,
    TopicInputSynchronizer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body

# %% mimics and helpers


@dataclass
class UnparameterizedTopicSynchronizer(TopicInputSynchronizer):
    """
    A topic synchronizer that never bound the type of its messages.
    """

    def apply_message(self, message: Any) -> None:
        pass


def latest_message_field_type(synchronizer_type: type) -> Any:
    """
    The declared type of the buffered message of a synchronizer class.
    """
    [message_field] = [
        field for field in fields(synchronizer_type) if field.name == "latest_message"
    ]
    return message_field.type


def joint_state_message(joint_name: str, position: float) -> JointState:
    """
    A joint state message that reports one position for one joint.
    """
    message = JointState()
    message.name = [joint_name]
    message.position = [position]
    return message


def odometry_message(pose: HomogeneousTransformationMatrix) -> Odometry:
    """
    An odometry message that reports the given pose.
    """
    quaternion = pose.to_rotation_matrix().to_quaternion().to_np()
    position = pose.to_position().to_np()
    message = Odometry()
    message.pose.pose.position.x = float(position[0])
    message.pose.pose.position.y = float(position[1])
    message.pose.pose.position.z = float(position[2])
    message.pose.pose.orientation.x = float(quaternion[0])
    message.pose.pose.orientation.y = float(quaternion[1])
    message.pose.pose.orientation.z = float(quaternion[2])
    message.pose.pose.orientation.w = float(quaternion[3])
    return message


@pytest.fixture()
def omni_drive_world() -> World:
    """
    A world whose root is connected to a base body by an omni drive.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        base = Body(name=PrefixedName("base"))
        world.add_connection(
            OmniDrive.create_with_dofs(world=world, parent=root, child=base)
        )
    return world


# %% message type resolution


def test_joint_state_synchronizers_read_joint_state_messages():
    assert PendingJointStateSynchronizer.message_type() is JointState
    assert LatestJointStateSynchronizer.message_type() is JointState


def test_odometry_synchronizer_reads_odometry_messages():
    assert OdometrySynchronizer.message_type() is Odometry


def test_synchronizer_without_bound_message_type_is_rejected():
    with pytest.raises(UnboundMessageTypeError):
        UnparameterizedTopicSynchronizer.message_type()
    with pytest.raises(UnboundMessageTypeError):
        TopicInputSynchronizer.message_type()


def test_joint_state_synchronizers_buffer_joint_state_messages():
    assert (
        latest_message_field_type(PendingJointStateSynchronizer) == Optional[JointState]
    )
    assert (
        latest_message_field_type(LatestJointStateSynchronizer) == Optional[JointState]
    )


def test_odometry_synchronizer_buffers_odometry_messages():
    assert latest_message_field_type(OdometrySynchronizer) == Optional[Odometry]


# %% writing joint states


def test_pending_joint_state_synchronizer_writes_a_message_once(
    init_rospy, mini_world: World
):
    [connection] = mini_world.connections
    synchronizer = PendingJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )
    synchronizer.latest_message = joint_state_message(connection.name.name, 0.42)

    assert synchronizer.apply() is True
    assert mini_world.state[connection.raw_dof.id].position == 0.42
    assert synchronizer.apply() is False


def test_latest_joint_state_synchronizer_rewrites_its_message_every_cycle(
    init_rospy, mini_world: World
):
    [connection] = mini_world.connections
    synchronizer = LatestJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )
    synchronizer.latest_message = joint_state_message(connection.name.name, 0.42)

    assert synchronizer.apply() is True
    mini_world.state[connection.raw_dof.id].position = 1.0
    assert synchronizer.apply() is True
    assert mini_world.state[connection.raw_dof.id].position == 0.42


def test_synchronizer_writes_nothing_without_a_message(init_rospy, mini_world: World):
    [connection] = mini_world.connections
    position_before_apply = mini_world.state[connection.raw_dof.id].position
    synchronizer = PendingJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )

    assert synchronizer.apply() is False
    assert mini_world.state[connection.raw_dof.id].position == position_before_apply


# %% writing the base pose


def test_odometry_synchronizer_writes_the_pose_into_the_drive(
    init_rospy, omni_drive_world: World
):
    connection = omni_drive_world.get_connection_by_name("root_T_base")
    expected_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.5, y=-2.5, yaw=0.75
    )
    synchronizer = OdometrySynchronizer(
        world=omni_drive_world, topic_name="odom", connection=connection
    )
    synchronizer.latest_message = odometry_message(expected_pose)

    assert synchronizer.apply() is True
    assert_allclose(
        connection.origin.to_np().astype(float),
        expected_pose.to_np().astype(float),
        atol=1e-9,
    )
