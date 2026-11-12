from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import NodeArtifacts
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicPublisherNode
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

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
