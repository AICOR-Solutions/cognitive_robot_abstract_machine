"""
Checking a plan before it runs: its steps are tried one after another on a copy of the
world, kinematically simulated, so nothing real moves.

A step left underspecified is
grounded the way a run grounds it, by trying its candidates, and a detection is answered
by :class:`~coraplex.perception.ValidationPerception`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.enums import DetectionGrounding
from coraplex.exceptions import NothingPerceived, UnidentifiedDetections
from coraplex.perception import (
    PerceptionInterface,
    PerceptionQuery,
    ValidationPerception,
)
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.failures import (
    EmptyUnderspecified,
    EndEffectorDidNotReachTarget,
    MotionMadeNoProgress,
    MotionViolatedCollisionAvoidance,
    NavigationGoalNotReachedError,
    RobotInCollision,
    TargetLeftFieldOfView,
)
from coraplex.robot_plans.actions.base import Action
from cramph.composites import StatementNode
from cramph.context import ContextExtension, StatechartContext
from cramph.exceptions import ExecutionFailure
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% verdicts


class ValidationVerdict(StrEnum):
    """
    How a step of a plan fared when it was checked.
    """

    VALID = "valid"
    """
    The step succeeded in the copy of the world.
    """

    ASSUMED = "assumed"
    """
    The step succeeded on what the world model assumes, since the robot cannot perceive
    it now; it is perceived again when the plan runs.
    """

    FAILED = "failed"
    """
    The step did not succeed.
    """


class FailureCause(StrEnum):
    """
    Why a step failed its check, as a person reads it.
    """

    UNREACHABLE = "unreachable"
    """
    The robot could not bring its arm or its base where the step asks.
    """

    COLLISION = "collision"
    """
    The step's motion would bring the robot into collision.
    """

    NOT_PERCEIVED = "not_perceived"
    """
    No object matching the detection was seen in its region.
    """

    AMBIGUOUS = "ambiguous"
    """
    Several objects matched the detection and none of them could be told apart.
    """

    OUT_OF_VIEW = "out_of_view"
    """
    An object the step was to keep in view lay outside the camera's field of view.
    """

    UNSUCCESSFUL = "unsuccessful"
    """
    The step failed for another reason, which the failure itself names.
    """

    @classmethod
    def from_failure(cls, failure: Exception) -> FailureCause:
        """
        :param failure: What ended a step.
        :return: The cause it stands for.
        """
        match failure:
            case (
                MotionMadeNoProgress()
                | EndEffectorDidNotReachTarget()
                | NavigationGoalNotReachedError()
            ):
                return cls.UNREACHABLE
            case MotionViolatedCollisionAvoidance() | RobotInCollision():
                return cls.COLLISION
            case NothingPerceived():
                return cls.NOT_PERCEIVED
            case UnidentifiedDetections():
                return cls.AMBIGUOUS
            case TargetLeftFieldOfView():
                return cls.OUT_OF_VIEW
        return cls.UNSUCCESSFUL


# %% checks


@dataclass
class StepCheck:
    """
    How one step of a plan fared when it was checked.
    """

    verdict: ValidationVerdict
    """
    Whether the step succeeded, on perceived or on assumed objects.
    """

    failure: Exception | None = None
    """
    What ended the step, if it failed.
    """

    candidate_failures: list[ExecutionFailure] = field(default_factory=list)
    """
    Why each candidate of an underspecified step failed, if the step ran out of them.
    """

    grounded_action: Action | None = None
    """
    The action the step ran as: the step itself, or the candidate an underspecified step
    was grounded to; None for a detection or a failed underspecified step.
    """

    grounding: DetectionGrounding | None = None
    """
    Where the objects a detection reports come from; None for any other step.
    """

    found: list[Body] = field(default_factory=list)
    """
    The bodies a detection found, where it moved them in the copy or added them to it.
    """

    @property
    def cause(self) -> FailureCause | None:
        """
        Why the step failed: of an underspecified step, why its first candidate failed;
        None if the step did not fail.
        """
        if self.failure is None:
            return None
        if self.candidate_failures:
            return FailureCause.from_failure(self.candidate_failures[0])
        return FailureCause.from_failure(self.failure)


# %% validation


@dataclass
class PlanValidation:
    """
    Checks the steps of a plan one after another on a copy of a world, without moving
    the robot, each starting where the step before it ended.

    The steps are built from the entities of :attr:`copied_world`; once a step failed,
    the copy is left where the failure left it.
    """

    world: World
    """
    The world the plan runs in, which is never modified.
    """

    extensions: list[ContextExtension]
    """
    The context extensions the plan runs with, referring to :attr:`world`.
    """

    perception: PerceptionInterface
    """
    The robot's perception, asked where its camera sees a detection's region now.
    """

    copied_world: World = field(init=False)
    """
    The copy the steps are checked in.
    """

    def __post_init__(self) -> None:
        self.copied_world = deepcopy(self.world)

    def check(self, step: StatechartNode) -> StepCheck:
        """
        Run a step in the copy, kinematically simulated in a statechart of its own; an
        underspecified step is grounded by trying its candidates.

        :param step: The step, built from the entities of :attr:`copied_world`.
        :return: How it fared.
        """
        executor = KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(
                self.copied_world,
                [
                    self.copied_world.rebind_world_entities(extension)
                    for extension in self.extensions
                ],
            )
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(step)
        # the boundary where a step that did not succeed becomes its verdict
        try:
            executor.compile(statechart)
            executor.execute()
        except ExecutionFailure as failure:
            return StepCheck(
                verdict=ValidationVerdict.FAILED,
                failure=failure,
                candidate_failures=(
                    executor.child_chooser.rejections.get(failure.node, [])
                    if isinstance(failure, EmptyUnderspecified)
                    else []
                ),
            )
        finally:
            executor.close()
        if isinstance(step, StatementNode):
            return StepCheck(
                verdict=ValidationVerdict.VALID,
                grounded_action=step.grounded_nodes[-1],
            )
        return StepCheck(
            verdict=ValidationVerdict.VALID,
            grounded_action=step if isinstance(step, Action) else None,
        )

    def check_detection(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> StepCheck:
        """
        Answer a detection as :class:`~coraplex.perception.ValidationPerception` does
        and write the object it reports into the copy, as the detection does when it
        runs.

        :param query: What the detection looks for, in :attr:`copied_world`.
        :param accept_first_if_multiple: Whether several objects reported may be
            resolved by taking the first one.
        :return: How it fared: valid where the object was perceived, assumed where it
            was taken from the world model.
        """
        grounded = ValidationPerception(self.perception).look_for(query)
        detections = grounded.detections
        if not detections:
            return StepCheck(
                verdict=ValidationVerdict.FAILED,
                failure=NothingPerceived(query.semantic_annotation),
                grounding=grounded.grounding,
            )
        if len(detections) > 1 and not accept_first_if_multiple:
            return StepCheck(
                verdict=ValidationVerdict.FAILED,
                failure=UnidentifiedDetections(
                    query.semantic_annotation, len(detections)
                ),
                grounding=grounded.grounding,
            )
        [annotation, *_] = detections[0].apply_to(
            self.copied_world,
            trust_orientation=query.trust_detected_orientation,
            region=query.region,
        )
        return StepCheck(
            verdict=(
                ValidationVerdict.ASSUMED
                if grounded.grounding is DetectionGrounding.ASSUMED
                else ValidationVerdict.VALID
            ),
            grounding=grounded.grounding,
            found=[annotation.root],
        )
