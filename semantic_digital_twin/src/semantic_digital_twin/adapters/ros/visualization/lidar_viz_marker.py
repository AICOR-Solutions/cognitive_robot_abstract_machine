from __future__ import annotations

from dataclasses import dataclass, field

from geometry_msgs.msg import Point as RosPoint
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.timer import Timer
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.sensors.lidar import Lidar

# %% publisher


@dataclass
class LidarVisualizationMarkerPublisher(HasROS2Node):
    """
    Periodically publishes where the beams of a lidar hit a surface as an RViz marker.

    The points are placed where the world believes the lidar to be, so a scan that does
    not line up with the drawn world shows a mismatch between the world and reality.

    .. warning:: To see something in Rviz add a MarkerArray plugin, set the topic
        name, and make sure the fixed frame is the tf root.
    """

    lidar: Lidar
    """
    The lidar whose readings are drawn.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the marker is published.
    """

    namespace: str = "__lidar_hit_points__"
    """
    The namespace of the marker.
    """

    publish_period: float = field(kw_only=True, default=0.1)
    """
    Time between two publishes, in seconds.
    """

    point_size: float = field(kw_only=True, default=0.03)
    """
    Width and height of every drawn point, in meters.
    """

    color: ColorRGBA = field(
        kw_only=True, default_factory=lambda: ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    )
    """
    The color the points are drawn with.
    """

    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """
    QoS profile for the publisher.

    Uses TRANSIENT_LOCAL because it shares a topic with the VizMarkerPublisher.
    """

    _publisher: Publisher = field(init=False)
    """
    The ROS publisher for the marker.
    """

    _timer: Timer = field(init=False)
    """
    The timer triggering every publish.
    """

    def __post_init__(self):
        self._publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        self._timer = self.node.create_timer(self.publish_period, self.publish)

    def stop(self) -> None:
        """
        Stop publishing.
        """
        self.node.destroy_timer(self._timer)
        self.node.destroy_publisher(self._publisher)

    def publish(self) -> None:
        """
        Publish the hit points of the most recent sweep, if the lidar has taken one.
        """
        if not self.lidar.source.has_reading:
            return
        marker = Marker()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.ns = self.namespace
        marker.id = 0
        marker.header.frame_id = str(self.lidar.root._world.root.name)
        marker.scale.x = self.point_size
        marker.scale.y = self.point_size
        marker.color = self.color
        marker.pose.orientation.w = 1.0
        marker.points = [
            RosPoint(x=float(x), y=float(y), z=float(z))
            for x, y, z in self.lidar.get_lidar_reading().root_P_hit_points
        ]
        self._publisher.publish(MarkerArray(markers=[marker]))
