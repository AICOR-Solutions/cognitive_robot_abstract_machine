import threading
from dataclasses import dataclass, field

from rclpy.node import Node
from rclpy.subscription import Subscription
from std_msgs.msg import String

# %% listening to what a robot says


@dataclass
class SpeechListener:
    """
    Records the sentences published on a text to speech topic, in the order they were
    said.
    """

    node: Node
    """
    The node the topic is listened to on.
    """

    topic_name: str
    """
    The text to speech topic listened to.
    """

    sentences: list[str] = field(default_factory=list, init=False)
    """
    The sentences heard so far.
    """

    _heard: threading.Condition = field(default_factory=threading.Condition, init=False)
    """
    Notified whenever a sentence is heard.
    """

    _subscription: Subscription = field(init=False)
    """
    The subscription to :attr:`topic_name`.
    """

    def __post_init__(self) -> None:
        self._subscription = self.node.create_subscription(
            String, self.topic_name, self._hear, 10
        )

    def _hear(self, message: String) -> None:
        with self._heard:
            self.sentences.append(message.data)
            self._heard.notify_all()

    def wait_for_sentences(self, count: int, timeout: float = 5.0) -> list[str]:
        """
        :param count: How many sentences to wait for.
        :param timeout: Seconds to wait at most.
        :return: The sentences heard once ``count`` arrived or the timeout passed.
        """
        with self._heard:
            self._heard.wait_for(lambda: len(self.sentences) >= count, timeout)
            return list(self.sentences)

    def close(self) -> None:
        """
        Stop listening.
        """
        self.node.destroy_subscription(self._subscription)
