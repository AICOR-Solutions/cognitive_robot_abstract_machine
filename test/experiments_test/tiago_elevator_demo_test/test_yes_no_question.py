from types import SimpleNamespace

import pytest

pytest.importorskip("communication_skills")

from action_msgs.msg import GoalStatus
from communication_skills.action import Say
from rclpy.task import Future

from experiments.tiago_room_request_demo.exceptions import SaySkillNotServedError
from experiments.tiago_room_request_demo.room_request import RoomRequestAnnouncement
from experiments.tiago_room_request_demo.yes_no_question import (
    YesNoAnswerInProgress,
    YesNoQuestionAsker,
)
from semantic_digital_twin.robots.tiago import Tiago

from .scripted_person import PersonAnswer, scripted_person

# %% understanding what a person says


def answer_after_the_question() -> YesNoAnswerInProgress:
    """
    :return: A question the robot has finished saying.
    """
    answer = YesNoAnswerInProgress(answer_timeout=60.0)
    question_ended = Future()
    question_ended.set_result(
        SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED, result=Say.Result())
    )
    answer.on_result(question_ended)
    return answer


@pytest.mark.parametrize(
    "sentence, expected_agreement",
    [
        ("yes", True),
        ("Yes, please.", True),
        ("sure", True),
        ("no", False),
        ("no thank you", False),
        ("not now", False),
        ("yes no", False),
    ],
)
def test_what_a_person_says_decides_whether_they_agree(sentence, expected_agreement):
    answer = answer_after_the_question()

    answer.on_speech(sentence)

    assert answer.poll() is expected_agreement


@pytest.mark.parametrize("sentence", ["", "maybe later", "yesterday", "nobody knows"])
def test_a_sentence_without_a_clear_answer_leaves_the_answer_open(sentence):
    answer = answer_after_the_question()

    answer.on_speech(sentence)

    assert answer.poll() is None


def test_speech_during_the_question_does_not_count_as_an_answer():
    answer = YesNoAnswerInProgress(answer_timeout=60.0)

    answer.on_speech("yes")

    assert answer.poll() is None


# %% waiting for the answer


@pytest.fixture
def question_asker(rclpy_node):
    asker = YesNoQuestionAsker(
        rclpy_node, Tiago.text_to_speech_action, server_timeout=0.5, answer_timeout=5.0
    )
    yield asker
    asker.close()


@pytest.mark.parametrize(
    "scripted_person, expected_agreement",
    [([PersonAnswer.YES], True), ([PersonAnswer.NO], False)],
    indirect=["scripted_person"],
)
def test_asking_until_answered_returns_the_answer(
    scripted_person, question_asker, expected_agreement
):
    scripted_person.room_confirmations = scripted_person.answers

    assert (
        question_asker.ask_until_answered(
            RoomRequestAnnouncement.CONFIRMING.verbalize("kitchen")
        )
        is expected_agreement
    )


def test_asking_when_the_skill_is_not_served_raises(question_asker):
    with pytest.raises(SaySkillNotServedError):
        question_asker.ask_until_answered(
            RoomRequestAnnouncement.CONFIRMING.verbalize("kitchen")
        )
