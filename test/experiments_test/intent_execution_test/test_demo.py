"""
The demonstration taking tasks through its skill, serving its world and carrying the
tasks out in a kinematic simulation of the IAI office.
"""

from __future__ import annotations

import json
import time

import pytest
from typing_extensions import Any, Iterator

intent_messages = pytest.importorskip("hri_actions_msgs.msg")
pytest.importorskip("speech2plan_msgs.action")

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from speech2plan_msgs.action import CarryOutIntent
from std_skills.msg import Result as SkillResult

from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.speech import SaySkillInterface
from experiments.intent_execution.demo import IntentExecutionDemonstration
from experiments.intent_execution.plan_builder import IntentRole, PlannableIntent
from experiments.tiago_room_request_demo.room_request import RoomRequestAnnouncement
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Room,
)
from semantic_digital_twin.world import World

from ..tiago_elevator_demo_test.speech_listener import SpeechListener
from .conftest import OfficeRoom

# %% fixtures


@pytest.fixture
def demonstration(rclpy_node: Node) -> Iterator[IntentExecutionDemonstration]:
    demonstration = IntentExecutionDemonstration(
        used_robot=Tiago, executor_type=KinematicallySimulatedPlanExecutor
    )
    yield demonstration
    demonstration.tear_down()


@pytest.fixture
def served_world(demonstration: IntentExecutionDemonstration) -> World:
    return demonstration.acquire_world()


@pytest.fixture
def tiago_speech(rclpy_node: Node) -> Iterator[SpeechListener]:
    pytest.importorskip(SaySkillInterface.PACKAGE)
    listener = SpeechListener(node=rclpy_node, action_name=Tiago.text_to_speech_action)
    yield listener
    listener.close()


@pytest.fixture
def task_client(
    rclpy_node: Node, demonstration: IntentExecutionDemonstration
) -> Iterator[ActionClient]:
    client = ActionClient(
        rclpy_node, CarryOutIntent, str(demonstration.task_skill_path)
    )
    yield client
    client.destroy()


def spoken_name_of(
    demonstration: IntentExecutionDemonstration, world: World, room: Room
) -> str:
    """
    :return: How people speak of ``room``.
    """
    return next(
        spoken_name
        for spoken_name, known_room in demonstration.office.rooms_by_spoken_name(
            world
        ).items()
        if known_room is room
    )


def room_upstairs(world: World) -> Room:
    """
    :return: A room two storeys above the elevator room, where the robot starts.
    """
    return world.get_semantic_annotation_by_name(OfficeRoom.UPSTAIRS)


def task(intent: str, roles: dict[str, Any]) -> CarryOutIntent.Goal:
    """
    :return: The goal of carrying out ``intent`` with ``roles``.
    """
    return CarryOutIntent.Goal(
        intent=intent_messages.Intent(intent=intent, data=json.dumps(roles))
    )


def completed(future: Future, seconds: float = 30.0) -> Any:
    """
    :return: The result of ``future``, once the node's executor completed it.
    """
    deadline = time.monotonic() + seconds
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert future.done(), "the future did not complete in time"
    return future.result()


def sent(
    client: ActionClient,
    goal: CarryOutIntent.Goal,
    feedback: list[Any] | None = None,
) -> ClientGoalHandle:
    """
    :return: The handle of ``goal``, once the skill answered whether it takes it on.
    """
    assert client.wait_for_server(timeout_sec=30.0)
    callback = None if feedback is None else lambda message: feedback.append(message)
    return completed(client.send_goal_async(goal, feedback_callback=callback))


# %% the intent vocabulary


def test_the_plannable_intents_are_spelled_as_the_message_defines_them() -> None:
    assert [str(intent) for intent in PlannableIntent] == [
        intent_messages.Intent.MOVE_TO,
        intent_messages.Intent.BRING_OBJECT,
    ]


# %% serving the world and taking tasks


def test_the_served_world_can_be_fetched_by_another_node(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    rclpy_node: Node,
) -> None:
    demonstration.serve(served_world)

    fetched = fetch_world_from_service(node=rclpy_node, timeout_seconds=60.0)

    assert (
        fetched.get_semantic_annotation_by_name(OfficeRoom.UPSTAIRS).name
        == served_world.get_semantic_annotation_by_name(OfficeRoom.UPSTAIRS).name
    )


def test_a_task_sent_to_the_skill_is_carried_out_step_by_step(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    task_client: ActionClient,
) -> None:
    tasks = demonstration.serve(served_world)
    room = room_upstairs(served_world)
    feedback = []
    handle = sent(
        task_client,
        task(
            intent_messages.Intent.MOVE_TO,
            {IntentRole.GOAL: spoken_name_of(demonstration, served_world, room)},
        ),
        feedback,
    )

    demonstration.perform(served_world, tasks.get(timeout=10.0))

    response = completed(handle.get_result_async())
    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert response.result.result.error_code == SkillResult.ROS_ENOERR
    assert [
        (message.feedback.step, message.feedback.state) for message in feedback
    ] == [(f"drive to {room.name}", CarryOutIntent.Feedback.RUNNING)]
    robot = served_world.get_semantic_annotations_by_type(Tiago)[0]
    assert Floor.under(robot.root) is room.floor


def test_a_task_sent_while_another_is_carried_out_is_turned_down(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    task_client: ActionClient,
) -> None:
    tasks = demonstration.serve(served_world)
    greeting = task(intent_messages.Intent.GREET, {})
    first = sent(task_client, greeting)

    second = sent(task_client, greeting)

    assert first.accepted
    assert not second.accepted
    demonstration.perform(served_world, tasks.get(timeout=10.0))
    response = completed(first.get_result_async())
    assert response.status == GoalStatus.STATUS_ABORTED
    assert response.result.result.error_code == SkillResult.ROS_ENOTSUP


def test_a_task_canceled_before_it_starts_is_not_carried_out(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    task_client: ActionClient,
) -> None:
    tasks = demonstration.serve(served_world)
    robot = served_world.get_semantic_annotations_by_type(Tiago)[0]
    start = robot.root.global_pose.to_np()
    room = room_upstairs(served_world)
    handle = sent(
        task_client,
        task(intent_messages.Intent.MOVE_TO, {IntentRole.GOAL: room.name.name}),
    )
    completed(handle.cancel_goal_async())

    demonstration.perform(served_world, tasks.get(timeout=10.0))

    response = completed(handle.get_result_async())
    assert response.status == GoalStatus.STATUS_CANCELED
    assert response.result.result.error_code == SkillResult.ROS_ECANCELED
    assert robot.root.global_pose.to_np() == pytest.approx(start)


def test_a_task_naming_nothing_to_act_on_fails_without_stopping_the_skill(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    task_client: ActionClient,
) -> None:
    tasks = demonstration.serve(served_world)
    handle = sent(task_client, task(intent_messages.Intent.BRING_OBJECT, {}))

    demonstration.perform(served_world, tasks.get(timeout=10.0))

    response = completed(handle.get_result_async())
    assert response.status == GoalStatus.STATUS_ABORTED
    assert response.result.result.error_code == SkillResult.ROS_EBADMSG
    assert sent(task_client, task(intent_messages.Intent.GREET, {})).accepted
    demonstration.perform(served_world, tasks.get(timeout=10.0))


# %% carrying intents out


def test_a_robot_sent_to_a_room_says_that_it_heads_there(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
    tiago_speech: SpeechListener,
) -> None:
    robot = served_world.get_semantic_annotations_by_type(Tiago)[0]
    room = next(
        room
        for room in served_world.get_semantic_annotations_by_type(Room)
        if Floor.under(robot.root) is room.floor
    )
    spoken_name = spoken_name_of(demonstration, served_world, room)

    demonstration.carry_out(
        served_world,
        intent_messages.Intent(
            intent=intent_messages.Intent.MOVE_TO,
            data=json.dumps({IntentRole.GOAL: spoken_name}),
        ),
    )

    [heading_to, *_] = tiago_speech.wait_for_sentences(1)
    assert heading_to == RoomRequestAnnouncement.HEADING_TO.verbalize(spoken_name)


def test_an_intent_no_plan_is_built_for_is_skipped(
    demonstration: IntentExecutionDemonstration,
    served_world: World,
) -> None:
    robot = served_world.get_semantic_annotations_by_type(Tiago)[0]
    start = robot.root.global_pose.to_np()

    outcome = demonstration.carry_out(
        served_world,
        intent_messages.Intent(intent=intent_messages.Intent.GREET, data="{}"),
    )

    assert outcome.result.error_code == SkillResult.ROS_ENOTSUP
    assert robot.root.global_pose.to_np() == pytest.approx(start)
