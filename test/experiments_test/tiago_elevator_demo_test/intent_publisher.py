import json
from dataclasses import dataclass, field

from hri_actions_msgs.msg import Intent
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.timer import Timer

from experiments.tiago_room_request_demo.room_request import (
    IntentRole,
    Ros4HriInteractionTopic,
)

# %% a person asking the robot to go somewhere


@dataclass
class RepeatedIntent:
    """
    Publishes the same intent over and over, so that it reaches a subscriber whenever
    that subscriber is connected.
    """

    node: Node
    """
    The node the intent is published on.
    """

    intent_name: str
    """
    The intent published, one of the constants of :class:`Intent`.
    """

    goal: str
    """
    What the intent's goal role names.
    """

    period: float = 0.1
    """
    Seconds between two publications.
    """

    _publisher: Publisher = field(init=False)
    """
    Publishes the intent.
    """

    _timer: Timer = field(init=False)
    """
    Triggers the publications.
    """

    def __post_init__(self) -> None:
        self._publisher = self.node.create_publisher(
            Intent, Ros4HriInteractionTopic.INTENTS, 10
        )
        self._timer = self.node.create_timer(self.period, self._publish)

    def _publish(self) -> None:
        self._publisher.publish(
            Intent(
                intent=self.intent_name,
                data=json.dumps({IntentRole.GOAL: self.goal}),
            )
        )

    def close(self) -> None:
        """
        Stop publishing.
        """
        self.node.destroy_timer(self._timer)
        self.node.destroy_publisher(self._publisher)
