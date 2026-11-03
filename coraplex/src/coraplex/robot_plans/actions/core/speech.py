from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.publisher import Publisher
from std_msgs.msg import String

from coraplex.robot_plans.actions.base import SequentialAction
from cramph.node import EndedByOwner, NodeArtifacts, StatechartNode
from giskardpy.motion_statechart.ros_context import RosContextExtension
from krrood.symbolic_math.symbolic_math import Scalar

# %% speaking aloud


@dataclass(eq=False, repr=False, kw_only=True)
class SpeakOnStart(EndedByOwner, StatechartNode):
    """
    Publishes a sentence once when it starts, and stays silent if it runs where there is
    no ROS node or if there is no topic to speak on.
    """

    text: str
    """
    The sentence to publish.
    """

    topic_name: str | None = None
    """
    The topic the sentence is published on, None to stay silent.
    """

    message_queue_depth: int = 10
    """
    How many sentences the publisher keeps for a receiver that has not taken them yet.
    """

    _publisher: Publisher | None = field(init=False, default=None)
    """
    Publishes the sentence, None when this node stays silent.
    """

    def set_up(self) -> None:
        super().set_up()
        if self.topic_name is None:
            return
        ros_context_extension = self.context.get_extension(RosContextExtension)
        if ros_context_extension is None or ros_context_extension.ros_node is None:
            return
        self._publisher = ros_context_extension.ros_node.create_publisher(
            String, self.topic_name, self.message_queue_depth
        )

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_start(self) -> None:
        if self._publisher is not None:
            self._publisher.publish(String(data=self.text))


@dataclass(eq=False, repr=False, kw_only=True)
class SayAction(SequentialAction):
    """
    Says a sentence aloud on the robot's text to speech topic.

    The robot stays silent if it cannot speak, because it declares no
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.text_to_speech_topic`,
    or if the plan runs where there is no ROS node to publish the sentence with.
    """

    text: str
    """
    The sentence the robot says.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            SpeakOnStart(text=self.text, topic_name=self.robot.text_to_speech_topic)
        ]
