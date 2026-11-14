from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.srv import LoadMap as LoadMapService
from rclpy.node import Node
from rclpy.service import Service
from typing_extensions import Iterator

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.localization import (
    LoadMap,
    PublishInitialPose,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% a map server


@dataclass
class MapServerMimic:
    """
    Answers map loading requests with :attr:`result` and remembers what was requested.
    """

    node: Node
    """
    The node the service is offered on.
    """

    service_name: str
    """
    The name the service is offered under.
    """

    result: int = LoadMapService.Response.RESULT_SUCCESS
    """
    The result every request is answered with.
    """

    requested_map_urls: list[str] = field(default_factory=list, init=False)
    """
    The map of every request, in the order they came in.
    """

    service: Service = field(init=False)
    """
    The service answering the requests.
    """

    def __post_init__(self):
        self.service = self.node.create_service(
            LoadMapService, self.service_name, self.answer
        )

    def answer(
        self, request: LoadMapService.Request, response: LoadMapService.Response
    ) -> LoadMapService.Response:
        """
        Remember the requested map and answer with :attr:`result`.
        """
        self.requested_map_urls.append(request.map_url)
        response.result = self.result
        return response

    def destroy(self) -> None:
        """
        Stop offering the service.
        """
        self.node.destroy_service(self.service)


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


# %% loading a map


@pytest.fixture
def map_server(rclpy_node) -> Iterator[MapServerMimic]:
    server = MapServerMimic(node=rclpy_node, service_name="load_map_test")
    yield server
    server.destroy()


def run_map_loading(rclpy_node: Node, world: World, node: LoadMap) -> Statechart:
    """
    :return: The statechart that ran `node` until it ended.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node)
    statechart.add_node(EndMotion.when_true(node))
    statechart.add_node(EndMotion.when_failed(node))
    executor.compile(statechart=statechart)
    deadline = time.monotonic() + 2.0
    while not statechart.is_ended() and time.monotonic() < deadline:
        executor.tick()
        # gives the map server time to answer between two ticks
        time.sleep(0.01)
    return statechart


def test_the_map_server_is_asked_to_load_the_map(
    rclpy_node, world_with_a_base, map_server
):
    loading = LoadMap(service_name=map_server.service_name, map_url="/maps/first.yaml")

    run_map_loading(rclpy_node, world_with_a_base, loading)

    assert map_server.requested_map_urls == [loading.map_url]
    assert loading.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_map_the_server_cannot_load_fails_the_loading(
    rclpy_node, world_with_a_base, map_server
):
    map_server.result = LoadMapService.Response.RESULT_MAP_DOES_NOT_EXIST
    loading = LoadMap(service_name=map_server.service_name, map_url="/maps/none.yaml")

    run_map_loading(rclpy_node, world_with_a_base, loading)

    assert loading.life_cycle_state == LifeCycleValues.FAILED
