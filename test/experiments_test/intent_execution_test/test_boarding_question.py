"""
Tiago asks to board the elevator through Speech2Plan's help skill.
"""

from __future__ import annotations

import json
import time

import pytest
from speech2plan_msgs.action import AskForHelp

from experiments.intent_execution.exceptions import HelpSkillNotServedError
from experiments.intent_execution.help_skill import HelpSkillAsker
from experiments.tiago_room_request_demo.boarding_permission import BoardingQuestion
from experiments.tiago_room_request_demo.yes_no_question import (
    YesNoAnswerInProgress,
    YesNoAsker,
)
from krrood.adapters.json_serializer import from_json, to_json

from .conftest import Office


def wait_for_the_skill(asker: YesNoAsker, timeout: float = 5.0) -> None:
    """
    :raises AssertionError: If the skill is not served within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while not asker.is_served:
        assert time.monotonic() < deadline, f"not served within {timeout} s"
        time.sleep(0.01)


def wait_for_the_answer(answer: YesNoAnswerInProgress, timeout: float = 5.0) -> bool:
    """
    :return: What the people answered.
    :raises AssertionError: If that is not known within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while (agreed := answer.poll()) is None:
        assert time.monotonic() < deadline, f"no answer within {timeout} s"
        time.sleep(0.01)
    return agreed


def boarding_asker(office: Office) -> YesNoAsker:
    """
    :return: What asks the people at the elevator whether Tiago may board.
    """
    asker = office.demonstration.boarding_asker(office.context)
    wait_for_the_skill(asker)
    return asker


@pytest.mark.parametrize("answer", [True, False])
def test_the_boarding_question_is_answered_through_the_help_skill(office, answer):
    office.people.answers = [answer]
    asker = boarding_asker(office)

    agreed = wait_for_the_answer(
        asker.ask(BoardingQuestion.ASKING_TO_BOARD, "person_at_elevator")
    )
    asker.close()

    assert agreed is answer
    [question] = office.people.asked
    assert question.problem == AskForHelp.Goal.BOARD_ELEVATOR
    assert question.text == BoardingQuestion.ASKING_TO_BOARD
    assert question.person_id == "person_at_elevator"


def test_a_help_skill_nobody_serves_ends_the_plan_with_where_it_was_expected(office):
    asker = HelpSkillAsker(
        node=office.demonstration.ros_node,
        skill_path="/test/unserved/ask_for_help",
        problem=AskForHelp.Goal.BOARD_ELEVATOR,
        server_timeout=0.1,
    )

    served = asker.wait_until_served()
    failure = asker.not_served()
    asker.close()

    assert not served
    assert failure == HelpSkillNotServedError(
        skill_path="/test/unserved/ask_for_help", timeout=0.1
    )


def test_the_unserved_help_skill_survives_the_way_to_the_controller():
    """
    The controller runs the child ending the plan, and reports its exception back.
    """
    failure = HelpSkillNotServedError(skill_path="/skill/ask_for_help", timeout=0.5)

    assert from_json(json.loads(json.dumps(to_json(failure)))) == failure
