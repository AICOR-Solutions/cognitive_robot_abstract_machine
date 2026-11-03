import threading
from dataclasses import dataclass, field

from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_action

from coraplex.robot_plans.actions.core.speech import SaySkillInterface

# %% listening to what a robot says


@dataclass
class SpeechListener:
    """
    Serves a text to speech action, recording the sentences it is sent in the order they
    were said.
    """

    node: Node
    """
    The node the action is served on.
    """

    action_name: str
    """
    The name of the text to speech action served.
    """

    sentences: list[str] = field(default_factory=list, init=False)
    """
    The sentences heard so far.
    """

    _heard: threading.Condition = field(default_factory=threading.Condition, init=False)
    """
    Notified whenever a sentence is heard.
    """

    _server: ActionServer = field(init=False)
    """
    The served action.
    """

    def __post_init__(self) -> None:
        self._server = ActionServer(
            self.node,
            get_action(SaySkillInterface.ACTION_TYPE),
            self.action_name,
            execute_callback=self._hear,
        )

    def _hear(self, goal_handle: ServerGoalHandle) -> object:
        """
        :return: The result of the goal saying a sentence.
        """
        with self._heard:
            self.sentences.append(goal_handle.request.input)
            self._heard.notify_all()
        goal_handle.succeed()
        return get_action(SaySkillInterface.ACTION_TYPE).Result()

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
        Stop serving the action.
        """
        self._server.destroy()
