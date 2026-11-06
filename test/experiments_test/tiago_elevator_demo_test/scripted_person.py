import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import pytest
from hri_msgs.msg import LiveSpeech
from rclpy.action import GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.publisher import Publisher
from rclpy.timer import Timer
from semantic_digital_twin.robots.tiago import Tiago

from experiments.tiago_room_request_demo.boarding_permission import BoardingQuestion
from experiments.tiago_room_request_demo.yes_no_question import Ros4HriVoiceTopic

from .speech_listener import SpeechListener

# %% a person with fixed answers


class PersonAnswer(StrEnum):
    """
    How the scripted person answers the question of the robot.
    """

    YES = "yes"
    NO = "no"
    UNCLEAR = "the weather is lovely today"
    REJECT = "reject"
    KEEP_SILENT = "keep_silent"


@dataclass
class ScriptedPerson(SpeechListener):
    """
    Serves the robot's speech and answers each boarding question with the next of
    :attr:`answers` and every later one with the last of them, repeating the answer
    until the next question.
    """

    answers: list[PersonAnswer] = field(default_factory=list)
    """
    How the questions are answered, in order.
    """

    room_confirmations: list[PersonAnswer] = field(
        default_factory=lambda: [PersonAnswer.YES]
    )
    """
    How the questions confirming a room are answered, in order.
    """

    speech_repetition_period: float = 0.1
    """
    Seconds between two repetitions of the answer.
    """

    after_refusal: Callable[[], None] = field(
        default_factory=lambda: lambda: None, init=False
    )
    """
    Called for each question answered with anything but yes.
    """

    goals: list = field(default_factory=list, init=False)
    """
    The goals asking the question that were accepted, in order.
    """

    confirmation_goals: list = field(default_factory=list, init=False)
    """
    The goals asking to confirm a room that were accepted, in order.
    """

    answer_times: list[float] = field(default_factory=list, init=False)
    """
    When each accepted question was answered, on the monotonic clock.
    """

    refusals: int = field(default=0, init=False)
    """
    How many questions were answered with anything but yes.
    """

    _decided_questions: int = field(default=0, init=False)
    """
    How many questions were decided on so far.
    """

    _repeated_speech: str | None = field(default=None, init=False)
    """
    What the person keeps saying, None while they say nothing.
    """

    _speech_publisher: Publisher = field(init=False)
    """
    Publishes what the person says.
    """

    _speech_timer: Timer = field(init=False)
    """
    Repeats what the person says.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._speech_publisher = self.node.create_publisher(
            LiveSpeech, Ros4HriVoiceTopic.ANONYMOUS_SPEAKER_SPEECH, 10
        )
        self._speech_timer = self.node.create_timer(
            self.speech_repetition_period, self._repeat_speech
        )

    def _repeat_speech(self) -> None:
        """
        Say the answer again, if the person is answering.
        """
        if self._repeated_speech is not None:
            self._speech_publisher.publish(LiveSpeech(final=self._repeated_speech))

    def _next_answer(self) -> PersonAnswer:
        """
        :return: The answer for the next question, or the last one once they ran out.
        """
        return self.answers[min(self._decided_questions, len(self.answers) - 1)]

    def _refuse(self) -> None:
        """
        Count a refusal and tell whoever waits for it.
        """
        self.refusals += 1
        self.after_refusal()

    def decide(self, goal: object) -> GoalResponse:
        if (
            goal.input != BoardingQuestion.ASKING_TO_BOARD
            or self._next_answer() is not PersonAnswer.REJECT
        ):
            return GoalResponse.ACCEPT
        self._decided_questions += 1
        self._refuse()
        return GoalResponse.REJECT

    def _hear(self, goal_handle: ServerGoalHandle) -> object:
        """
        Start answering once a question ended.
        """
        sentence = goal_handle.request.input
        if sentence != BoardingQuestion.ASKING_TO_BOARD:
            if not sentence.endswith("?"):
                return super()._hear(goal_handle)
            confirmation = self.room_confirmations[
                min(len(self.confirmation_goals), len(self.room_confirmations) - 1)
            ]
            self.confirmation_goals.append(goal_handle.request)
            self._repeated_speech = None
            result = super()._hear(goal_handle)
            if confirmation is not PersonAnswer.KEEP_SILENT:
                self._repeated_speech = confirmation.value
            return result
        answer = self._next_answer()
        self._decided_questions += 1
        self._repeated_speech = None
        self.goals.append(goal_handle.request)
        self.answer_times.append(time.monotonic())
        if answer is not PersonAnswer.YES:
            self._refuse()
        result = super()._hear(goal_handle)
        if answer is not PersonAnswer.KEEP_SILENT:
            self._repeated_speech = answer.value
        return result

    def close(self) -> None:
        """
        Stop serving the speech and answering.
        """
        self.node.destroy_timer(self._speech_timer)
        self.node.destroy_publisher(self._speech_publisher)
        super().close()


@pytest.fixture
def scripted_person(rclpy_node, request):
    module = ScriptedPerson(
        node=rclpy_node,
        action_name=Tiago.text_to_speech_action,
        answers=request.param,
    )
    yield module
    module.close()
