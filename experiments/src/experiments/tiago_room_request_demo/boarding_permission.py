"""
Asking the people at an elevator whether the robot may take it with them, through the
ROS4HRI ``ask`` skill, and waiting for the next cabin while they say no.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

from action_msgs.msg import GoalStatus
from communication_skills.action import Ask
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from std_skills.msg import Result as SkillResult

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import NotOnASingleLevelException
from coraplex.plans.executables import Executable, GiskardExecutable
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import ExecutionBoundaryNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from experiments.tiago_room_request_demo.exceptions import (
    AskSkillNotServedError,
    BoardingPermissionNotGrantedError,
    ElevatorCabinTimedOutError,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% the question


class CommunicationSkill(StrEnum):
    """
    Names of the ROS4HRI communication skills the robot calls.
    """

    ASK = "/skill/ask"


class BoardingQuestion(StrEnum):
    """
    What the robot asks the people at an elevator.
    """

    ASKING_TO_BOARD = "May I take the elevator with you?"
    """
    Asked before the robot boards an elevator people stand in or wait at.
    """


class BoardingAnswerKey(StrEnum):
    """
    The pieces of information the robot asks the people at an elevator for.
    """

    MAY_BOARD = "may_board"
    """
    Whether the person agrees that the robot boards.
    """


class BoardingPermissionFile(Enum):
    """
    Files describing the question to the people at an elevator.
    """

    ANSWERS_SCHEMA = Path(__file__).with_name("boarding_answers_schema.json")
    """
    The answers the ``ask`` skill is to retrieve, keyed by :class:`BoardingAnswerKey`.
    """


@dataclass
class BoardingAnswers:
    """
    What a person answered the robot asking to board.
    """

    may_board: bool | None
    """
    Whether the person agrees that the robot boards, None if the answer could not be
    retrieved.
    """

    @classmethod
    def from_json(cls, answers: str) -> BoardingAnswers:
        """
        :param answers: The answers of an ``ask`` skill result.
        :return: The answers about boarding they hold.
        """
        return cls(may_board=json.loads(answers).get(BoardingAnswerKey.MAY_BOARD))


# %% asking a person


@dataclass
class BoardingQuestionAsker:
    """
    Asks a person through the ``ask`` skill whether the robot may board.
    """

    node: Node
    """
    The node the skill is called from, which an executor has to spin elsewhere.
    """

    server_timeout: float = 10.0
    """
    Seconds to wait for the skill to be served.
    """

    answer_timeout: float = 60.0
    """
    Seconds a person may take to answer.
    """

    _client: ActionClient = field(init=False)
    """
    The client calling the skill.
    """

    def __post_init__(self) -> None:
        self._client = ActionClient(self.node, Ask, CommunicationSkill.ASK)

    def may_board(self, person_id: str) -> bool:
        """
        :param person_id: The ROS4HRI ID of the person to ask.
        :return: Whether the person agreed. A rejected goal, a failed skill, no answer
            within :attr:`answer_timeout` and an answer that could not be retrieved
            count as not agreeing.
        :raises AskSkillNotServedError: If the skill is not served within
            :attr:`server_timeout`.
        """
        if not self._client.wait_for_server(timeout_sec=self.server_timeout):
            raise AskSkillNotServedError(CommunicationSkill.ASK, self.server_timeout)
        goal_handle = self._wait_for(
            self._client.send_goal_async(
                Ask.Goal(
                    person_id=person_id,
                    question=BoardingQuestion.ASKING_TO_BOARD,
                    answers_schema=BoardingPermissionFile.ANSWERS_SCHEMA.value.read_text(),
                )
            )
        )
        if goal_handle is None or not goal_handle.accepted:
            return False
        response = self._wait_for(goal_handle.get_result_async())
        if response is None:
            goal_handle.cancel_goal_async()
            return False
        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.result.error_code != SkillResult.ROS_ENOERR
        ):
            return False
        return BoardingAnswers.from_json(response.result.answers).may_board is True

    def _wait_for(self, future: Future) -> object | None:
        """
        :return: What ``future`` resolved to, None if it did not within
            :attr:`answer_timeout`.
        """
        resolved = threading.Event()
        future.add_done_callback(lambda _: resolved.set())
        if not resolved.wait(self.answer_timeout):
            return None
        return future.result()

    def close(self) -> None:
        """
        Stop calling the skill.
        """
        self._client.destroy()


# %% asking until allowed


@dataclass
class AskToBoardExecutable(Executable):
    """
    Asks to board whenever the cabin serves the robot's level, and after every refusal
    waits for the cabin to leave before waiting for the next one.

    Ends once nobody is at the elevator or a person agreed.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator the robot wants to board.
    """

    level: Level = field(kw_only=True)
    """
    The level the robot boards from.
    """

    robot_root: KinematicStructureEntity = field(kw_only=True)
    """
    The robot's root, from which the distance to the people is measured.
    """

    persons_at_elevator: PersonsAtElevator = field(kw_only=True)
    """
    Finds the people to ask.
    """

    maximum_attempts: int = field(kw_only=True)
    """
    How many times the robot asks before it gives up.
    """

    arrival_threshold: float = field(kw_only=True)
    """
    Position error of the elevator's drive within which the cabin serves :attr:`level`.
    """

    cabin_timeout: float = field(kw_only=True)
    """
    Seconds to wait for the cabin to arrive or to leave.
    """

    cabin_check_period: float = field(default=0.05, kw_only=True)
    """
    Seconds between two looks at where the cabin is.
    """

    execution_scope: Callable[[], AbstractContextManager[None]] = field(
        default=nullcontext, kw_only=True, repr=False, compare=False
    )
    """
    Tracks the execution of the plan node this executable belongs to.
    """

    def execute(self) -> None:
        """
        :raises BoardingPermissionNotGrantedError: If the cabin left after the last of
            :attr:`maximum_attempts` refusals.
        :raises ElevatorCabinTimedOutError: If the cabin did not arrive or leave within
            :attr:`cabin_timeout`.
        :raises AskSkillNotServedError: If somebody is to be asked and the ``ask`` skill
            is not served.
        """
        if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
            return
        with self.execution_scope():
            asker = BoardingQuestionAsker(self.context.ros_node)
            try:
                self._ask_until_allowed(asker)
            finally:
                asker.close()

    def _ask_until_allowed(self, asker: BoardingQuestionAsker) -> None:
        """
        Return once nobody is at the elevator or a person agreed.

        :param asker: Asks the people.
        """
        for _ in range(self.maximum_attempts):
            self._wait_until_cabin_serves_level(True)
            person_id = self.persons_at_elevator.nearest_to(self.robot_root, self.level)
            if person_id is None or asker.may_board(person_id):
                return
            self._wait_until_cabin_serves_level(False)
        raise BoardingPermissionNotGrantedError(self.maximum_attempts)

    def _wait_until_cabin_serves_level(self, serving: bool) -> None:
        """
        :param serving: Whether to wait for the cabin to serve :attr:`level` or for it
            not to.
        :raises ElevatorCabinTimedOutError: If that did not happen within
            :attr:`cabin_timeout`.
        """
        target_position = self.elevator.drive_position_for_floor(self.level)
        deadline = time.monotonic() + self.cabin_timeout
        while (
            bool(
                abs(self.elevator.movable_joint.position - target_position)
                < self.arrival_threshold
            )
            != serving
        ):
            if time.monotonic() > deadline:
                raise ElevatorCabinTimedOutError(self.cabin_timeout)
            time.sleep(self.cabin_check_period)


# Between two motions, because a motion waiting this long on the controller would be ended
# by any other process changing the world model.
@dataclass(eq=False)
class AskToBoardNode(ExecutionBoundaryNode):
    """
    Node between two motions in which the robot asks to board an elevator.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator the robot wants to board.
    """

    level: Level = field(kw_only=True)
    """
    The level the robot boards from.
    """

    robot_root: KinematicStructureEntity = field(kw_only=True)
    """
    The robot's root, from which the distance to the people is measured.
    """

    persons_at_elevator: PersonsAtElevator = field(kw_only=True)
    """
    Finds the people to ask.
    """

    maximum_attempts: int = field(kw_only=True)
    """
    How many times the robot asks before it gives up.
    """

    arrival_threshold: float = field(kw_only=True)
    """
    Position error of the elevator's drive within which the cabin serves :attr:`level`.
    """

    cabin_timeout: float = field(kw_only=True)
    """
    Seconds to wait for the cabin to arrive or to leave.
    """

    def parse(self) -> AskToBoardExecutable:
        return AskToBoardExecutable(
            context=self.context,
            elevator=self.elevator,
            level=self.level,
            robot_root=self.robot_root,
            persons_at_elevator=self.persons_at_elevator,
            maximum_attempts=self.maximum_attempts,
            arrival_threshold=self.arrival_threshold,
            cabin_timeout=self.cabin_timeout,
            execution_scope=self.execution_scope,
        )


# %% actions


@dataclass
class AskToBoardAction(ActionDescription):
    """
    Waits in front of an elevator until the people at it let the robot board, asking
    them whenever the cabin serves the robot's level.
    """

    elevator: Elevator
    """
    The elevator the robot wants to board.
    """

    persons_at_elevator: PersonsAtElevator = field(kw_only=True)
    """
    Finds the people to ask.
    """

    maximum_attempts: int = field(default=3, kw_only=True)
    """
    How many times the robot asks before it gives up.
    """

    arrival_threshold: float = field(default=0.01, kw_only=True)
    """
    Position error of the elevator's drive within which the cabin serves the robot's
    level.
    """

    cabin_timeout: float = field(default=600.0, kw_only=True)
    """
    Seconds to wait for the cabin to arrive or to leave.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        :raises NotOnASingleLevelException: If the robot stands on no level.
        :raises MissingMovableJointError: If the elevator has no drive.
        """
        level = Level.under(self.robot.root)
        if level is None:
            raise NotOnASingleLevelException("Robot is not on any recognized floor.")
        if self.elevator.movable_joint is None:
            raise MissingMovableJointError(self.elevator)
        # The areas join the world on first access, which must not happen while a
        # motion runs.
        self.elevator.cabin_area
        self.elevator.waiting_areas
        return execute_single(
            AskToBoardNode(
                elevator=self.elevator,
                level=level,
                robot_root=self.robot.root,
                persons_at_elevator=self.persons_at_elevator,
                maximum_attempts=self.maximum_attempts,
                arrival_threshold=self.arrival_threshold,
                cabin_timeout=self.cabin_timeout,
            )
        )


@dataclass
class NavigateToRoomAskingToBoardAction(NavigateToRoomAction):
    """
    Unlike :class:`NavigateToRoomAction`, asks the people at the elevator whether the
    robot may board before riding it, see :class:`AskToBoardAction`.
    """

    persons_at_elevator: PersonsAtElevator = field(kw_only=True)
    """
    Finds the people to ask.
    """

    maximum_boarding_attempts: int = field(default=3, kw_only=True)
    """
    How many times the robot asks to board before it gives up.
    """

    def _ride_elevator(
        self, elevator: Elevator, target_level: Level
    ) -> list[ActionDescription]:
        return [
            AskToBoardAction(
                elevator,
                persons_at_elevator=self.persons_at_elevator,
                maximum_attempts=self.maximum_boarding_attempts,
                arrival_threshold=self.elevator_arrival_threshold,
            ),
            *super()._ride_elevator(elevator, target_level),
        ]
