import time

import numpy as np
import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.localization import PublishInitialPose
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% a base standing on a map


@pytest.fixture
def world_with_a_base() -> World:
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        base = Body(name=PrefixedName("base"))
        world.add_connection(
            FixedConnection(
                parent=map_body,
                child=base,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=2, yaw=0.5, reference_frame=map_body
                ),
            )
        )
    return world


# %% publishing the initial pose


def test_the_initial_pose_is_where_the_world_holds_the_base(
    rclpy_node, world_with_a_base
):
    topic_name = "initial_pose_test"
    base = world_with_a_base.get_kinematic_structure_entity_by_name("base")
    messages = []
    subscription = rclpy_node.create_subscription(
        PoseWithCovarianceStamped, topic_name, messages.append, 10
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=world_with_a_base),
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(publish := PublishInitialPose(topic_name=topic_name, base=base))
    statechart.add_node(EndMotion.when_true(publish))
    executor.compile(statechart=statechart)
    # gives the subscription time to discover the publisher created while compiling
    time.sleep(0.5)

    executor.tick_until_end(timeout=100)
    deadline = time.monotonic() + 2.0
    while not messages and time.monotonic() < deadline:
        time.sleep(0.05)
    rclpy_node.destroy_subscription(subscription)

    [message] = messages
    assert message.header.frame_id == str(world_with_a_base.root.name)
    assert message.pose.pose == PoseToRos2Converter.convert(base.global_pose)
    expected_covariance = np.zeros((6, 6))
    expected_covariance[0, 0] = publish.position_standard_deviation**2
    expected_covariance[1, 1] = publish.position_standard_deviation**2
    expected_covariance[5, 5] = publish.yaw_standard_deviation**2
    assert np.array(message.pose.covariance) == pytest.approx(
        expected_covariance.flatten()
    )
