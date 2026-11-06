"""
Asking the people at an elevator whether the robot may take it with them, and waiting for
the next cabin while they say no.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto


from coraplex.exceptions import NotOnASingleLevelException
from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.base import GoalAction
from cramph.composites import ChildChooser, CompositeNodeChoosingItsChild
from cramph.monitors import CountTicks
from cramph.node import EndedByOwner, NodeArtifacts, StatechartNode
from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import Scalar
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from experiments.tiago_room_request_demo.exceptions import (
    BoardingPermissionNotGrantedError,
    ElevatorCabinTimedOutError,
    SaySkillNotServedError,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from experiments.tiago_room_request_demo.yes_no_question import (
    YesNoAnswerInProgress,
    YesNoQuestionAsker,
)
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)

# %% the question


class BoardingQuestion(StrEnum):
    """
    What the robot asks the people at an elevator.
    """

    ASKING_TO_BOARD = "May I take the elevator with you?"
    """
    Asked before the robot boards an elevator people stand in or wait at.
    """


# %% asking until allowed


@dataclass(eq=False, repr=False, kw_only=True)
class AskToBoardNode(CompositeNodeChoosingItsChild):
    """
    Holds the plan until the people at an elevator let the robot board: it succeeds once
    :class:`AskToBoardChooser` chose a child for it, which only happens then.

    .. note:: The question is asked where the plan was built, so no process running this
        node has to know who stands at the elevator.
    """

    elevator: Elevator
    """
    The elevator the robot wants to board.
    """

    level: Level
    """
    The level the robot boards from.
    """

    maximum_attempts: int
    """
    How many times the robot asks before it gives up.
    """

    arrival_threshold: float
    """
    Position error of the elevator's drive within which the cabin serves :attr:`level`.
    """

    cabin_timeout: float
    """
    Seconds to wait for the cabin to arrive or to leave.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class AbortOnStart(EndedByOwner, StatechartNode):
    """
    Ends the plan with an exception as soon as it starts, wherever the plan runs, which
    reports the exception to whoever started the plan.
    """

    failure: DataclassException
    """
    What the plan ends with.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_trinary_unknown())

    def on_start(self) -> None:
        raise self.failure


class BoardingStage(Enum):
    """
    What the robot is waiting for while it asks to board.
    """

    CABIN_ARRIVING = auto()
    """
    The cabin has to serve the robot's level before anybody is asked.
    """

    SKILL_SERVED = auto()
    """
    There is somebody to ask, and the ``say`` skill has to be served.
    """

    ANSWER = auto()
    """
    A person was asked and has not answered yet.
    """

    CABIN_LEAVING = auto()
    """
    The people refused, and their cabin has to leave before the next one can come.
    """


@dataclass
class BoardingConversation:
    """
    Asks to board whenever the cabin serves the robot's level, and after every refusal
    waits for the cabin to leave before waiting for the next one.
    """

    node: AskToBoardNode
    """
    The node this conversation chooses the child of.
    """

    persons_at_elevator: PersonsAtElevator
    """
    Finds the people to ask.
    """

    asker: YesNoQuestionAsker
    """
    Asks the people.
    """

    stage: BoardingStage = BoardingStage.CABIN_ARRIVING
    """
    What the robot is waiting for.
    """

    stage_started_at: float = field(default_factory=time.monotonic)
    """
    When the robot started waiting for that, on the monotonic clock.
    """

    refusals: int = 0
    """
    How many times the people refused so far.
    """

    person_id: str | None = None
    """
    The person to ask, None until somebody was found.
    """

    answer: YesNoAnswerInProgress | None = None
    """
    The question being answered, None while nobody is being asked.
    """

    def advance(self) -> StatechartNode | None:
        """
        Go as far as what happened so far allows, without waiting for anything.

        :return: The child that ends the node's waiting, once there is one: one that
            succeeds right away once the robot may board, or one ending the plan with
            what went wrong. None while the conversation goes on.
        """
        match self.stage:
            case BoardingStage.CABIN_ARRIVING:
                return self._advance_while_cabin_arrives()
            case BoardingStage.SKILL_SERVED:
                return self._advance_while_skill_is_not_served()
            case BoardingStage.ANSWER:
                return self._advance_while_person_answers()
            case BoardingStage.CABIN_LEAVING:
                return self._advance_while_cabin_leaves()

    def _advance_while_cabin_arrives(self) -> StatechartNode | None:
        """
        Once the cabin serves the robot's level, find the person to ask, and let the robot
        board if nobody is there.
        """
        if not self._cabin_serves_level():
            return self._abort_if_cabin_timed_out()
        self.person_id = self.persons_at_elevator.nearest_to(
            self.node.context.require_extension(RobotAccess).robot.root,
            self.node.level,
        )
        if self.person_id is None:
            return self._allowed()
        self._enter(BoardingStage.SKILL_SERVED)
        return self._advance_while_skill_is_not_served()

    def _advance_while_skill_is_not_served(self) -> StatechartNode | None:
        """
        Ask the person once the skill is served, and give up if it does not get served.
        """
        if not self.asker.is_served:
            if self._seconds_in_stage() > self.asker.server_timeout:
                return self._aborting(
                    SaySkillNotServedError(
                        self.asker.text_to_speech_action, self.asker.server_timeout
                    )
                )
            return None
        self.answer = self.asker.ask(BoardingQuestion.ASKING_TO_BOARD, self.person_id)
        self._enter(BoardingStage.ANSWER)
        return None

    def _advance_while_person_answers(self) -> StatechartNode | None:
        """
        Let the robot board if the person agreed, and otherwise wait for their cabin to
        leave.
        """
        may_board = self.answer.poll()
        if may_board is None:
            return None
        if may_board:
            return self._allowed()
        self.refusals += 1
        self._enter(BoardingStage.CABIN_LEAVING)
        return None

    def _advance_while_cabin_leaves(self) -> StatechartNode | None:
        """
        Once the refused cabin left, wait for the next one, or give up if the robot asked
        as often as it may.
        """
        if self._cabin_serves_level():
            return self._abort_if_cabin_timed_out()
        if self.refusals >= self.node.maximum_attempts:
            return self._aborting(
                BoardingPermissionNotGrantedError(self.node.maximum_attempts)
            )
        self._enter(BoardingStage.CABIN_ARRIVING)
        return None

    def _enter(self, stage: BoardingStage) -> None:
        """
        :param stage: What the robot is waiting for from now on.
        """
        self.stage = stage
        self.stage_started_at = time.monotonic()

    def _seconds_in_stage(self) -> float:
        """
        :return: For how long the robot has been waiting for what it waits for now.
        """
        return time.monotonic() - self.stage_started_at

    def _cabin_serves_level(self) -> bool:
        """
        :return: Whether the cabin stands at the robot's level.
        """
        elevator = self.node.elevator
        return bool(
            abs(
                elevator.movable_joint.position
                - elevator.drive_position_for_floor(self.node.level)
            )
            < self.node.arrival_threshold
        )

    def _abort_if_cabin_timed_out(self) -> StatechartNode | None:
        """
        :return: A child ending the plan if the cabin took too long, None otherwise.
        """
        if self._seconds_in_stage() <= self.node.cabin_timeout:
            return None
        return self._aborting(ElevatorCabinTimedOutError(self.node.cabin_timeout))

    def _allowed(self) -> StatechartNode:
        """
        :return: A child that succeeds right away, which ends the node's waiting.
        """
        return CountTicks(name=f"{self.node.name}/allowed", ticks=0)

    def _aborting(self, failure: DataclassException) -> StatechartNode:
        """
        :param failure: What went wrong.
        :return: A child ending the plan with `failure`.
        """
        return AbortOnStart(name=f"{self.node.name}/gave_up", failure=failure)


@dataclass
class AskToBoardChooser(ChildChooser[AskToBoardNode]):
    """
    Answers every :class:`AskToBoardNode` once its :class:`BoardingConversation` came to
    an end.
    """

    persons_at_elevator: PersonsAtElevator
    """
    Finds the people to ask.
    """

    asker: YesNoQuestionAsker
    """
    Asks the people.
    """

    _conversations: dict[int, BoardingConversation] = field(
        default_factory=dict, init=False
    )
    """
    The conversation of each node, by the node's index.
    """

    _outcomes: dict[int, StatechartNode] = field(default_factory=dict, init=False)
    """
    The child each node whose conversation ended runs, by the node's index.
    """

    def has_choice_for(self, node: AskToBoardNode) -> bool:
        if node.index not in self._outcomes:
            conversation = self._conversations.setdefault(
                node.index,
                BoardingConversation(
                    node=node,
                    persons_at_elevator=self.persons_at_elevator,
                    asker=self.asker,
                ),
            )
            outcome = conversation.advance()
            if outcome is None:
                return False
            self._outcomes[node.index] = outcome
        return True

    def choose_child(self, node: AskToBoardNode) -> StatechartNode | None:
        return self._outcomes.pop(node.index)

    def cleanup(self) -> None:
        self.asker.close()


# %% actions


@dataclass(eq=False, repr=False, kw_only=True)
class AskToBoardAction(GoalAction):
    """
    Waits in front of an elevator until the people at it let the robot board, asking
    them whenever the cabin serves the robot's level.

    .. warning:: The plan needs an :class:`AskToBoardChooser` among the choosers of its
        context.
    """

    elevator: Elevator
    """
    The elevator the robot wants to board.
    """

    maximum_attempts: int = 3
    """
    How many times the robot asks before it gives up.
    """

    arrival_threshold: float = 0.01
    """
    Position error of the elevator's drive within which the cabin serves the robot's
    level.
    """

    cabin_timeout: float = 600.0
    """
    Seconds to wait for the cabin to arrive or to leave.
    """

    def create_goal(self) -> AskToBoardNode:
        """
        :raises NotOnASingleLevelException: If the robot stands on no level.
        :raises MissingMovableJointError: If the elevator has no drive.
        """
        level = Level.under(self.robot.root)
        if level is None:
            raise NotOnASingleLevelException("Robot is not on any recognized floor.")
        if self.elevator.movable_joint is None:
            raise MissingMovableJointError(self.elevator)
        # The areas join the world on first access, which is cheapest before the plan
        # runs.
        self.elevator.cabin_area
        self.elevator.waiting_areas
        return AskToBoardNode(
            elevator=self.elevator,
            level=level,
            maximum_attempts=self.maximum_attempts,
            arrival_threshold=self.arrival_threshold,
            cabin_timeout=self.cabin_timeout,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class NavigateToRoomAskingToBoardAction(NavigateToRoomAction):
    """
    Unlike :class:`NavigateToRoomAction`, asks the people at the elevator whether the
    robot may board before riding it, see :class:`AskToBoardAction`.
    """

    maximum_boarding_attempts: int = 3
    """
    How many times the robot asks to board before it gives up.
    """

    def _ride_elevator(
        self, elevator: Elevator, target_level: Level
    ) -> list[StatechartNode]:
        return [
            AskToBoardAction(
                elevator=elevator,
                maximum_attempts=self.maximum_boarding_attempts,
                arrival_threshold=self.elevator_arrival_threshold,
            ),
            *super()._ride_elevator(elevator, target_level),
        ]
