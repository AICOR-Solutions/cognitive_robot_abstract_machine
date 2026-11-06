"""
Asking people a question they answer with yes or no, by saying it through the ROS4HRI
``say`` skill and listening for the answer in the recognized speech.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum

from action_msgs.msg import GoalStatus
from communication_skills.action import Say
from hri_msgs.msg import LiveSpeech
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.task import Future
from std_skills.msg import Result as SkillResult

from experiments.tiago_room_request_demo.exceptions import SaySkillNotServedError

# %% what people answer


class Ros4HriVoiceTopic(StrEnum):
    """
    Topics of REP-155 on which the robot hears people.
    """

    ANONYMOUS_SPEAKER_SPEECH = "/humans/voices/anonymous_speaker/speech"


class AgreeingWord(StrEnum):
    """
    Words with which a person agrees.
    """

    YES = "yes"
    YEAH = "yeah"
    YEP = "yep"
    SURE = "sure"
    OKAY = "okay"
    OK = "ok"


class RefusingWord(StrEnum):
    """
    Words with which a person refuses, which outweigh any :class:`AgreeingWord` in the
    same sentence.
    """

    NO = "no"
    NOPE = "nope"
    NOT = "not"
    DO_NOT = "don't"
    SORRY = "sorry"


# %% asking


@dataclass
class YesNoAnswerInProgress:
    """
    A question to a person, which the robot is saying through the ``say`` skill or the
    person is answering.
    """

    answer_timeout: float
    """
    Seconds the person may take to answer.
    """

    asked_at: float = field(default_factory=time.monotonic)
    """
    When the question was asked, on the monotonic clock.
    """

    agreed: bool | None = field(default=None, init=False)
    """
    Whether the person agreed, None until that is known.
    """

    _goal_handle: ClientGoalHandle | None = field(default=None, init=False)
    """
    The accepted goal, None until the skill accepted it.
    """

    _question_ended: bool = field(default=False, init=False)
    """
    Whether the robot finished saying the question, which is when answers start to
    count.
    """

    def on_goal_response(self, future: Future) -> None:
        """
        Wait for the result of an accepted goal; a rejected goal counts as not agreeing.
        """
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.agreed = False
            return
        self._goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, future: Future) -> None:
        """
        Start listening once the question was said; a failed skill counts as not
        agreeing.
        """
        response = future.result()
        if (
            response is None
            or response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.result.error_code != SkillResult.ROS_ENOERR
        ):
            self.agreed = False
            return
        self._question_ended = True

    def on_speech(self, sentence: str) -> None:
        """
        Take a sentence holding a yes or a no as the answer, and keep listening
        otherwise.

        :param sentence: What the robot heard.
        """
        if not self._question_ended or self.agreed is not None:
            return
        words = set(re.findall(r"[\w']+", sentence.casefold()))
        if words & set(RefusingWord):
            self.agreed = False
        elif words & set(AgreeingWord):
            self.agreed = True

    def poll(self) -> bool | None:
        """
        :return: Whether the person agreed, None while that is not known yet. No clear
            answer within :attr:`answer_timeout` counts as not agreeing.
        """
        if self.agreed is None and (
            time.monotonic() - self.asked_at > self.answer_timeout
        ):
            if self._goal_handle is not None and not self._question_ended:
                self._goal_handle.cancel_goal_async()
            self.agreed = False
        return self.agreed


@dataclass
class YesNoQuestionAsker:
    """
    Asks people questions they answer with yes or no.

    .. warning:: The recognized speech names no speaker, so anybody can answer.
    """

    node: Node
    """
    The node the skill is called from, which an executor has to spin elsewhere.
    """

    text_to_speech_action: str
    """
    The name of the action that makes the robot speak.
    """

    speech_topic: str = Ros4HriVoiceTopic.ANONYMOUS_SPEAKER_SPEECH
    """
    The topic the recognized speech of people is read from.
    """

    server_timeout: float = 10.0
    """
    Seconds to wait for the skill to be served.
    """

    answer_timeout: float = 60.0
    """
    Seconds a person may take to answer.
    """

    answer_check_period: float = 0.1
    """
    Seconds between two looks at whether the answer is known, while waiting for it.
    """

    _client: ActionClient = field(init=False)
    """
    The client calling the skill.
    """

    _speech_subscription: Subscription = field(init=False)
    """
    The subscription to the recognized speech.
    """

    _answer: YesNoAnswerInProgress | None = field(default=None, init=False)
    """
    The question asked last, None until one was asked.
    """

    def __post_init__(self) -> None:
        self._client = ActionClient(self.node, Say, self.text_to_speech_action)
        self._speech_subscription = self.node.create_subscription(
            LiveSpeech, self.speech_topic, self._hear, 10
        )

    def _hear(self, speech: LiveSpeech) -> None:
        """
        Pass what the robot heard on to the question asked last.
        """
        if self._answer is not None:
            self._answer.on_speech(speech.final)

    @property
    def is_served(self) -> bool:
        """
        :return: Whether the skill can be called right now.
        """
        return self._client.server_is_ready()

    def ask(self, question: str, person_id: str = "") -> YesNoAnswerInProgress:
        """
        Ask without waiting for the answer.

        :param question: What to ask.
        :param person_id: The ROS4HRI ID of the person to ask, anybody if empty.
        :return: The question, which tells the answer once it is known.
        """
        self._answer = YesNoAnswerInProgress(answer_timeout=self.answer_timeout)
        self._client.send_goal_async(
            Say.Goal(person_id=person_id, input=question)
        ).add_done_callback(self._answer.on_goal_response)
        return self._answer

    def ask_until_answered(self, question: str, person_id: str = "") -> bool:
        """
        Ask and wait for the answer, which needs an executor spinning :attr:`node`
        elsewhere.

        :param question: What to ask.
        :param person_id: The ROS4HRI ID of the person to ask, anybody if empty.
        :return: Whether the person agreed.
        :raises SaySkillNotServedError: If the skill is not served within
            :attr:`server_timeout`.
        """
        if not self._client.wait_for_server(timeout_sec=self.server_timeout):
            raise SaySkillNotServedError(
                self.text_to_speech_action, self.server_timeout
            )
        answer = self.ask(question, person_id)
        while (agreed := answer.poll()) is None:
            time.sleep(self.answer_check_period)
        return agreed

    def close(self) -> None:
        """
        Stop calling the skill and listening.
        """
        self._client.destroy()
        self.node.destroy_subscription(self._speech_subscription)
