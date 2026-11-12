from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs import srv as nav2_services
from rclpy.client import Client
from rclpy.task import Future

import krrood.symbolic_math.symbolic_math as sm
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from cramph.node import (
    FailsOnObservingFalse,
    NodeArtifacts,
    SucceedsOnObservingTrue,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicPublisherNode
from giskardpy.motion_statechart.ros_context import RosContextExtension
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

logger = logging.getLogger(__name__)

# %% switching the map


@dataclass(eq=False, repr=False, kw_only=True)
class LoadMap(SucceedsOnObservingTrue, FailsOnObservingFalse, MotionStatechartNode):
    """
    Has a map server load a map once it starts: observes True once the map server loaded
    it, False once the map server could not, and Unknown until it answered.
    """

    map_url: str
    """
    Where the map server finds the map's description, such as the path of its ``.yaml``
    file.
    """

    service_name: str = "map_server/load_map"
    """
    Name of the map server's service loading a map.
    """

    _client: Client = field(init=False, repr=False)
    """
    The client calling the map server, created in :meth:`set_up`.
    """

    _result: int | None = field(init=False, default=None, repr=False)
    """
    The map server's answer to the current request, None until it answered.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    def set_up(self) -> None:
        """
        Create the client and wait for the map server to offer its service.
        """
        super().set_up()
        ros_node = self.context.require_extension(RosContextExtension).ros_node
        self._client = ros_node.create_client(nav2_services.LoadMap, self.service_name)
        logger.info(f"Waiting for service {self.service_name}")
        self._client.wait_for_service()

    def on_start(self) -> None:
        self._result = None
        future = self._client.call_async(
            nav2_services.LoadMap.Request(map_url=self.map_url)
        )
        future.add_done_callback(self._remember_result)

    def _remember_result(self, future: Future) -> None:
        """
        Keep the map server's answer for :meth:`on_tick`.
        """
        self._result = future.result().result

    def on_tick(self) -> ObservationStateValues:
        if self._result is None:
            return ObservationStateValues.UNKNOWN
        if self._result == nav2_services.LoadMap.Response.RESULT_SUCCESS:
            return ObservationStateValues.TRUE
        logger.error(
            f"{self.service_name} could not load {self.map_url}: {self._result}"
        )
        return ObservationStateValues.FALSE


# %% telling a localization where the robot is


@dataclass(eq=False, repr=False, kw_only=True)
class PublishInitialPose(TopicPublisherNode[PoseWithCovarianceStamped]):
    """
    Publishes the pose the world holds for :attr:`base`, in the frame of the world's
    root, as the initial pose of a localization once it starts, and observes True from
    then on.

    .. note:: A localization such as AMCL listens for this message on its
        ``initialpose`` topic and starts its estimate over around the pose.
    """

    base: KinematicStructureEntity
    """
    The body whose pose is published, such as the root of a mobile robot.
    """

    position_standard_deviation: float = 0.1
    """
    How far, in meters, the published position may be off along either axis of the
    floor.
    """

    yaw_standard_deviation: float = 0.1
    """
    How far, in radians, the published heading may be off.
    """

    msg_type: type[PoseWithCovarianceStamped] = field(
        default=PoseWithCovarianceStamped, init=False
    )

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self) -> None:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = str(self.context.world.root.name)
        message.header.stamp = self.ros2_node.get_clock().now().to_msg()
        message.pose.pose = PoseToRos2Converter.convert(self.base.global_pose)
        covariance = np.zeros((6, 6))
        covariance[0, 0] = covariance[1, 1] = self.position_standard_deviation**2
        covariance[5, 5] = self.yaw_standard_deviation**2
        message.pose.covariance = covariance.flatten().tolist()
        self._publisher.publish(message)
