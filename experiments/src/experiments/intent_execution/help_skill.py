"""
Asking people for help through Speech2Plan's ``ask_for_help`` skill, which says the
question and hears the yes or no itself, so the answer is not taken for an instruction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from speech2plan_msgs.action import AskForHelp
from std_skills.msg import Result as SkillResult

from experiments.intent_execution.exceptions import HelpSkillNotServedError
from experiments.tiago_room_request_demo.yes_no_question import (
    YesNoAnswerInProgress,
    YesNoAsker,
)


@dataclass
class HelpAnswerInProgress(YesNoAnswerInProgress):
    """
    A question asked through the help skill, which ends with the answer.
    """

    def on_result(self, future: Future) -> None:
        """
        Take the answer the skill heard; a failed skill counts as not agreeing.
        """
        response = future.result()
        self._question_ended = True
        self.agreed = bool(
            response is not None
            and response.status == GoalStatus.STATUS_SUCCEEDED
            and response.result.result.error_code == SkillResult.ROS_ENOERR
            and response.result.yes
        )


@dataclass
class HelpSkillAsker(YesNoAsker):
    """
    Asks people for help with :attr:`problem` through the help skill.
    """

    node: Node = field(kw_only=True)
    """
    The node the skill is called from, which an executor has to spin elsewhere.
    """

    skill_path: str = field(kw_only=True)
    """
    Where the ``speech2plan_msgs/action/AskForHelp`` skill is served.
    """

    problem: str = field(kw_only=True)
    """
    Why the robot asks, one of the problems of ``AskForHelp``.
    """

    server_timeout: float = 10.0
    """
    Seconds to wait for the skill to be served.
    """

    answer_timeout: float = 180.0
    """
    Seconds people may take to answer.
    """

    answer_check_period: float = 0.1
    """
    Seconds between two looks at whether the answer is known, while waiting for it.
    """

    _client: ActionClient = field(init=False)
    """
    The client calling the skill.
    """

    def __post_init__(self) -> None:
        self._client = ActionClient(
            self.node,
            AskForHelp,
            self.skill_path,
            callback_group=ReentrantCallbackGroup(),
        )

    @property
    def is_served(self) -> bool:
        return self._client.server_is_ready()

    def not_served(self) -> HelpSkillNotServedError:
        return HelpSkillNotServedError(
            skill_path=self.skill_path, timeout=self.server_timeout
        )

    def ask(self, question: str, person_id: str = "") -> HelpAnswerInProgress:
        answer = HelpAnswerInProgress(answer_timeout=self.answer_timeout)
        self._client.send_goal_async(
            AskForHelp.Goal(problem=self.problem, text=question, person_id=person_id)
        ).add_done_callback(answer.on_goal_response)
        return answer

    def wait_until_served(self) -> bool:
        """
        :return: Whether the skill is served within :attr:`server_timeout`, which needs
            an executor spinning :attr:`node` elsewhere.
        """
        return self._client.wait_for_server(timeout_sec=self.server_timeout)

    def answer_to(self, question: str) -> bool:
        """
        Ask anybody and wait for the answer, which needs an executor spinning
        :attr:`node` elsewhere.

        :return: Whether someone answered yes; no answer within :attr:`answer_timeout`
            counts as no.
        """
        answer = self.ask(question)
        while (agreed := answer.poll()) is None:
            time.sleep(self.answer_check_period)
        return agreed

    def close(self) -> None:
        self._client.destroy()
