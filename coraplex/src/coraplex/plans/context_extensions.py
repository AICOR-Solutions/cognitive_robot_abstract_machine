from __future__ import annotations

from dataclasses import dataclass, field


from cramph.context import ContextExtension
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    QueryBackend,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what a plan's nodes read from the context


@dataclass(kw_only=True)
class RobotAccess(ContextExtension):
    """
    Gives the nodes of a statechart the robot that performs the plan.
    """

    robot: AbstractRobot
    """
    The robot performing the plan.
    """

    @property
    def controlled_root(self) -> KinematicStructureEntity:
        """
        :return: The topmost entity the robot may move, which a Cartesian goal is
            expressed relative to. Driving the base while manipulating moves the robot
            relative to the world, so the world root is the only frame that holds still
            then; otherwise the robot's own root does.
        """
        if (
            isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
        ):
            return self.robot._world.root
        return self.robot.root


@dataclass(kw_only=True)
class StatementGrounding(ContextExtension):
    """
    How the underspecified statements of a plan are grounded into actions.
    """

    query_backend: QueryBackend = field(
        default_factory=EntityQueryLanguageGenerativeBackend
    )
    """
    The backend answering the underspecified statements.

    Defaults to the deterministic generative backend, since underspecified actions are
    constructed (generated).
    """

    candidates_to_try: int = 50
    """
    How many candidates an underspecified step tries before giving up, unless the step
    has a limit of its own.
    """

    sampling_seed: int | None = 0
    """
    Seed for the locations the plan's actions sample, so a run repeats; ``None`` samples
    afresh each run.
    """


@dataclass(kw_only=True)
class WalkingBasesStand(ContextExtension):
    """
    Lets the base of a robot that walks on legs stand while an action keeps the base
    still, instead of holding it at the pose it started at, and keeps its center of mass
    over its feet, see
    :class:`~giskardpy.motion_statechart.tasks.balance_tasks.CenterOfMassOverTheFeet`;
    only for :attr:`~semantic_digital_twin.robots.walking.PhysicalBaseMode.WALKING`.
    """


@dataclass(kw_only=True)
class WalkingBasesHeld(ContextExtension):
    """
    Holds the base of a robot that walks on legs where it stands once it walked there,
    in a physical simulation, see
    :attr:`~semantic_digital_twin.robots.walking.PhysicalBaseMode.WALK_THEN_FIX`. The
    hold is a simulation shortcut: the real robot keeps balancing on its legs.
    """

    simulation: MujocoSim
    """
    The physical simulation holding the bases.
    """
