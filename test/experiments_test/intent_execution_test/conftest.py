"""
The IAI office with a Tiago in its elevator room, and the plans built in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

import pytest
from hri_actions_msgs.msg import Intent
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from speech2plan_msgs.action import AskForHelp, CarryOutIntent
from std_skills.msg import Result as SkillResult
from typing_extensions import Any, Iterator

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from experiments.intent_execution.demo import IntentExecutionDemonstration, SkillPath
from experiments.intent_execution.plan_builder import (
    IntentPlanBuilder,
    PlannableIntent,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World

# %% the people the robot asks


@dataclass
class StandInPeople:
    """
    Serves the help skill in place of Speech2Plan, answering from a script.
    """

    node: Node = field(kw_only=True)
    """
    The node serving the skill.
    """

    answers: list[bool] = field(kw_only=True, default_factory=lambda: [True])
    """
    The answers in turn, the last one again once the others are given.
    """

    asked: list[AskForHelp.Goal] = field(kw_only=True, default_factory=list)
    """
    The questions put to the people, in order.
    """

    server: ActionServer = field(init=False)
    """
    Serves the help skill on :attr:`node`.
    """

    def __post_init__(self) -> None:
        self.server = ActionServer(
            self.node,
            AskForHelp,
            str(SkillPath.ASK_FOR_HELP),
            execute_callback=self.answer,
            callback_group=ReentrantCallbackGroup(),
        )

    def answer(self, goal: ServerGoalHandle) -> AskForHelp.Result:
        """
        Answer the question of ``goal`` with the next scripted answer.
        """
        self.asked.append(goal.request)
        yes = self.answers[min(len(self.asked), len(self.answers)) - 1]
        goal.succeed()
        return AskForHelp.Result(yes=yes)

    @property
    def texts(self) -> list[str]:
        """
        What the robot asked, in order.
        """
        return [question.text for question in self.asked]


# %% the office with a robot


class OfficeRoom(StrEnum):
    """
    Rooms of the office the tests send the robot to.
    """

    UPSTAIRS = "2.58"
    """
    A room two storeys above the elevator room the robot starts in.
    """


@dataclass
class Office:
    """
    The office with a Tiago in its elevator room, and the context its plans run in.
    """

    demonstration: IntentExecutionDemonstration = field(kw_only=True)
    """
    The demonstration that built the office and carries out the intents.
    """

    world: World = field(kw_only=True)
    """
    The world holding the office and the robot.
    """

    people: StandInPeople = field(kw_only=True)
    """
    The people the robot asks for help.
    """

    @property
    def context(self) -> StatechartContext:
        """
        A context the plans of the robot run in.
        """
        return StatechartContext.with_extensions(
            self.world, self.demonstration.build_context_extensions(self.world)
        )

    @property
    def robot(self) -> AbstractRobot:
        """
        The robot carrying out the intents.
        """
        return self.context.require_extension(RobotAccess).robot

    @property
    def plan_builder(self) -> IntentPlanBuilder:
        """
        A builder of the plans of the robot.
        """
        return IntentPlanBuilder(
            context=self.context,
            demonstration=self.demonstration,
            pickup_room=self.demonstration.pickup_room,
        )

    def room(self, name: str) -> Room:
        """
        :return: The room of that name.
        """
        return self.world.get_semantic_annotation_by_name(name)

    def spoken_name_of(self, room: Room) -> str:
        """
        :return: How people speak of ``room``.
        """
        return self.plan_builder.spoken_name_of(room)

    def carry_out(
        self, intent: PlannableIntent, roles: dict[str, Any]
    ) -> CarryOutIntent.Result:
        """
        Carry out ``intent`` with ``roles`` in a kinematic simulation.

        :return: How carrying it out ended.
        """
        return self.demonstration.carry_out(
            self.world, Intent(intent=intent, data=json.dumps(roles))
        )

    def carry_out_without_error(
        self, intent: PlannableIntent, roles: dict[str, Any]
    ) -> CarryOutIntent.Result:
        """
        Carry out ``intent`` with ``roles``, which must end without an error.
        """
        outcome = self.carry_out(intent, roles)
        assert outcome.result.error_code == SkillResult.ROS_ENOERR, (
            outcome.result.error_msg
        )
        return outcome


@pytest.fixture
def office() -> Iterator[Office]:
    demonstration = IntentExecutionDemonstration(
        used_robot=Tiago,
        executor_type=KinematicallySimulatedPlanExecutor,
        help_server_timeout=timedelta(seconds=5),
    )
    world = demonstration.acquire_world()
    yield Office(
        demonstration=demonstration,
        world=world,
        people=StandInPeople(node=demonstration.ros_node),
    )
    demonstration.tear_down()
