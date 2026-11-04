from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.visualization.lidar_viz_marker import (
    LidarVisualizationMarkerPublisher,
)
from ..test_sensors.test_lidar import (
    FAR_WALL_DISTANCE,
    simulated_lidar,
    three_beam_pattern,
    world_with_walls,
)

# %% recording what is published


@dataclass
class LidarMarkerRecorder:
    """
    Keeps the last marker array published in the lidar marker's namespace.
    """

    namespace: str
    """
    The namespace the lidar marker is published in.
    """

    last_message: MarkerArray | None = field(init=False, default=None)
    """
    The most recently received marker array of that namespace.
    """

    def __call__(self, message: MarkerArray) -> None:
        if message.markers[0].ns == self.namespace:
            self.last_message = message


# %% publishing the hit points


def test_publishes_the_hit_points_of_the_lidar_in_the_world_root(rclpy_node):
    world, mount = world_with_walls(FAR_WALL_DISTANCE)
    lidar = simulated_lidar(mount, three_beam_pattern())
    publisher = LidarVisualizationMarkerPublisher(node=rclpy_node, lidar=lidar)
    recorder = LidarMarkerRecorder(namespace=publisher.namespace)
    rclpy_node.create_subscription(
        MarkerArray, publisher.topic_name, recorder, publisher.qos_profile
    )

    for _ in range(30):
        if recorder.last_message is not None:
            break
        sleep(0.1)
    publisher.stop()

    [marker] = recorder.last_message.markers
    assert marker.type == Marker.POINTS
    assert marker.header.frame_id == str(world.root.name)
    published_points = [[point.x, point.y, point.z] for point in marker.points]
    assert np.allclose(published_points, lidar.get_lidar_reading().root_P_hit_points)
