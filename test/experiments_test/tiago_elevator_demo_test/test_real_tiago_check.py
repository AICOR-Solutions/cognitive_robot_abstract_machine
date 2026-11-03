from dataclasses import dataclass
from enum import StrEnum

import pytest

pytest.importorskip("communication_skills")

from json_msgs.action import JsonAction
from rclpy.action import ActionServer
from std_msgs.msg import String

from experiments.tiago_room_request_demo import real_tiago_check
from experiments.tiago_room_request_demo.real_tiago_check import (
    DriveConfirmation,
    RealTiagoCheck,
    RealTiagoPreflight,
    RequiredAction,
    RequiredTopic,
    Stage,
)

# %% a robot serving only part of what is needed


class CheckedName(StrEnum):
    """
    Names the preflight is told to look for.
    """

    SERVED_ACTION = "/served_action"
    MISSING_ACTION = "/missing_action"
    PUBLISHED_TOPIC = "/published_topic"
    SILENT_TOPIC = "/silent_topic"


@dataclass
class SessionOfNode:
    """
    Stands in for a ROS session by holding only its node.
    """

    node: object
    """
    The node of the session.
    """


@pytest.fixture
def preflight_of_partly_served_robot(rclpy_node):
    server = ActionServer(
        rclpy_node, JsonAction, CheckedName.SERVED_ACTION, lambda _: None
    )
    publisher = rclpy_node.create_publisher(String, CheckedName.PUBLISHED_TOPIC, 10)
    yield RealTiagoPreflight(
        rclpy_node,
        actions=[
            RequiredAction(CheckedName.SERVED_ACTION, JsonAction, [Stage.SAY]),
            RequiredAction(CheckedName.MISSING_ACTION, JsonAction, [Stage.SAY]),
        ],
        topics=[
            RequiredTopic(CheckedName.PUBLISHED_TOPIC, [Stage.PEOPLE]),
            RequiredTopic(CheckedName.SILENT_TOPIC, [Stage.PEOPLE]),
        ],
        discovery_time=0.5,
    )
    rclpy_node.destroy_publisher(publisher)
    server.destroy()


# %% looking at the robot


def test_preflight_tells_what_is_there_from_what_is_missing(
    preflight_of_partly_served_robot,
):
    findings = preflight_of_partly_served_robot.find()

    assert [finding.found for finding in findings] == [True, False, True, False]


# %% speaking through the skill


def test_the_skill_speaks_the_sentence_it_is_given(rclpy_node, tiago_speech):
    sentence = "Hello"

    seconds = RealTiagoCheck(SessionOfNode(rclpy_node)).say_through_skill(sentence)

    assert seconds >= 0.0
    assert tiago_speech.sentences == [sentence]


def test_saying_through_a_skill_nobody_serves_fails(rclpy_node):
    check = RealTiagoCheck(SessionOfNode(rclpy_node), server_timeout=0.5)

    assert check.say_through_skill("Hello") is None


# %% moving the robot only when told to


@pytest.fixture
def demonstration_runs(monkeypatch):
    """
    :return: A list that grows by one with every demonstration that was run.
    """
    runs = []

    @dataclass
    class RecordedDemonstration:
        used_robot: object
        executor_type: object

        def run(self) -> None:
            runs.append(self)

    monkeypatch.setattr(
        real_tiago_check, "TiagoRoomRequestDemonstration", RecordedDemonstration
    )
    return runs


def test_drive_stays_put_when_something_is_missing(
    rclpy_node, demonstration_runs, monkeypatch
):
    monkeypatch.setattr(RealTiagoCheck, "preflight", lambda self: False)
    monkeypatch.setattr("builtins.input", lambda _: DriveConfirmation.PHRASE.value)

    assert RealTiagoCheck(SessionOfNode(rclpy_node)).drive() is False
    assert demonstration_runs == []


def test_drive_stays_put_without_the_confirmation(
    rclpy_node, demonstration_runs, monkeypatch
):
    monkeypatch.setattr(RealTiagoCheck, "preflight", lambda self: True)
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    assert RealTiagoCheck(SessionOfNode(rclpy_node)).drive() is False
    assert demonstration_runs == []


def test_drive_runs_the_demonstration_once_confirmed(
    rclpy_node, demonstration_runs, monkeypatch
):
    monkeypatch.setattr(RealTiagoCheck, "preflight", lambda self: True)
    monkeypatch.setattr("builtins.input", lambda _: DriveConfirmation.PHRASE.value)

    assert RealTiagoCheck(SessionOfNode(rclpy_node)).drive() is True
    assert len(demonstration_runs) == 1
