import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import pytest
from communication_skills.action import Ask
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from experiments.tiago_room_request_demo.boarding_permission import (
    BoardingAnswerKey,
    CommunicationSkill,
)

# %% an ask skill with fixed answers


class AskAnswer(StrEnum):
    """
    How the scripted ask skill answers a question.
    """

    YES = "yes"
    NO = "no"
    UNANSWERED = "unanswered"
    REJECT = "reject"
    KEEP_SILENT = "keep_silent"


@dataclass
class ScriptedAsk:
    """
    Serves the ``ask`` skill, answering each goal with the next of :attr:`answers` and
    every later one with the last of them.
    """

    node: Node
    """
    The node the skill is served on.
    """

    answers: list[AskAnswer]
    """
    How the goals are answered, in order.
    """

    after_refusal: Callable[[], None] = field(default=lambda: None, init=False)
    """
    Called for each goal answered with anything but yes.
    """

    goals: list[Ask.Goal] = field(default_factory=list, init=False)
    """
    The goals accepted, in order.
    """

    answer_times: list[float] = field(default_factory=list, init=False)
    """
    When each accepted goal was answered, on the monotonic clock.
    """

    refusals: int = field(default=0, init=False)
    """
    How many goals were answered with anything but yes.
    """

    _decided_goals: int = field(default=0, init=False)
    """
    How many goals were decided on so far.
    """

    released: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set to end the silence of goals answered with :attr:`AskAnswer.KEEP_SILENT`.
    """

    server: ActionServer = field(init=False)
    """
    The served skill.
    """

    def __post_init__(self) -> None:
        self.server = ActionServer(
            self.node,
            Ask,
            CommunicationSkill.ASK,
            execute_callback=self.execute,
            goal_callback=self.decide,
        )

    def _take_answer(self) -> AskAnswer:
        """
        :return: The answer for the next goal, or the last one once they ran out.
        """
        answer = self.answers[min(self._decided_goals, len(self.answers) - 1)]
        self._decided_goals += 1
        return answer

    def _refuse(self) -> None:
        """
        Count a refusal and tell whoever waits for it.
        """
        self.refusals += 1
        self.after_refusal()

    def decide(self, goal: Ask.Goal) -> GoalResponse:
        """
        :return: Whether the next answer lets the goal be accepted at all.
        """
        if (
            self.answers[min(self._decided_goals, len(self.answers) - 1)]
            is not AskAnswer.REJECT
        ):
            return GoalResponse.ACCEPT
        self._take_answer()
        self._refuse()
        return GoalResponse.REJECT

    def execute(self, goal_handle: ServerGoalHandle) -> Ask.Result:
        """
        :return: The result carrying the next answer.
        """
        answer = self._take_answer()
        self.goals.append(goal_handle.request)
        self.answer_times.append(time.monotonic())
        if answer is AskAnswer.KEEP_SILENT:
            self.released.wait()
        if answer is not AskAnswer.YES:
            self._refuse()
        goal_handle.succeed()
        answers = (
            {BoardingAnswerKey.MAY_BOARD: True}
            if answer is AskAnswer.YES
            else {BoardingAnswerKey.MAY_BOARD: False} if answer is AskAnswer.NO else {}
        )
        return Ask.Result(answers=json.dumps(answers))

    def close(self) -> None:
        """
        Stop serving the skill.
        """
        self.released.set()
        self.server.destroy()


@pytest.fixture
def scripted_ask(rclpy_node, request):
    module = ScriptedAsk(node=rclpy_node, answers=request.param)
    yield module
    module.close()
