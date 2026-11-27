"""
Walking on legs: a controller a legged robot brings along, which balances its base on
its legs and walks it at a commanded velocity.

The controller is independent of any simulator. A physical simulation
hands it what the robot senses once every
:attr:`WalkingController.control_period` and drives the joints it names
towards the positions it answers with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy
from krrood.patterns.classproperty import classproperty
from numpy.typing import NDArray

# %% how a physical simulation moves a walking base


class PhysicalBaseMode(StrEnum):
    """
    How a physical simulation moves the base of a robot that walks on legs.

    Holding the base is a simulation shortcut: the real robot balances on its legs all
    the time.
    """

    FIXED = "fixed"
    """
    The base is held where it starts, and nothing walks it.
    """

    WALK_THEN_FIX = "walk_then_fix"
    """
    The walking controller walks the base to every pose it is commanded to; once the
    base stands still there, it is held where it stands until it is commanded to walk
    again. A base that never walks is held where it starts.
    """

    WALKING = "walking"
    """
    The walking controller balances the base on its legs all the time, also while it
    stands.
    """


# %% what a walking controller is told


@dataclass(kw_only=True)
class PlanarVelocity:
    """
    A velocity of a base across the floor, along and about the axes of the base itself.
    """

    forward: float = 0.0
    """
    How fast the base moves along its own x-axis, in metres per second.
    """

    sideways: float = 0.0
    """
    How fast the base moves along its own y-axis, in metres per second.
    """

    turn: float = 0.0
    """
    How fast the base turns about the vertical, in radians per second.
    """

    def to_array(self) -> NDArray[numpy.float64]:
        """
        :return: The forward, sideways and turning velocity, in this order.
        """
        return numpy.array([self.forward, self.sideways, self.turn])


@dataclass(kw_only=True)
class WalkingSenses:
    """
    What a legged robot senses of itself at one instant: its base's inertial unit and
    the encoders of its joints.
    """

    angular_velocity: NDArray[numpy.float64]
    """
    How fast the base turns, in radians per second about the base's own axes.
    """

    gravity_direction: NDArray[numpy.float64]
    """The unit vector gravity pulls along, in the base's own axes; ``(0, 0,
    -1)`` while the base stands upright."""

    joint_positions: dict[str, float]
    """
    The position of each joint the controller observes, by the joint's name.
    """

    joint_velocities: dict[str, float]
    """
    The velocity of each joint the controller observes, by the joint's name.
    """


# %% how fast a walking robot approaches a pose


@dataclass(kw_only=True)
class WalkingPace:
    """
    How fast a robot walks up to a pose: in proportion to how far it still is, between
    the slowest pace its controller still steps at and the fastest it walks safely, and
    how close it has to come to count as arrived.
    """

    slowing_rate: float
    """
    The speed, per metre or radian still to go, the robot approaches at, in 1/s.
    """

    slowest_speed: float
    """
    The slowest speed, in metres per second, the controller still walks at.
    """

    fastest_speed: float
    """
    The fastest speed, in metres per second, the robot approaches a pose at.
    """

    slowest_turn: float
    """
    The slowest turn, in radians per second, the controller still turns at.
    """

    fastest_turn: float
    """
    The fastest turn, in radians per second, the robot turns towards a heading at.
    """

    arrival_tolerance: float
    """
    How far the base may stand from a pose, in metres and in radians, to count as
    arrived; the controller cannot step closer reliably.
    """


# %% how a robot standing on its legs keeps its balance


@dataclass(kw_only=True)
class StandingBalance:
    """
    Where a robot standing on its legs keeps its center of mass while its upper body
    moves, so its walking controller can stand still: above its feet, and moved there
    gently.
    """

    foot_names: list[str]
    """
    The names of the bodies the robot stands on, which the soles belong to.
    """

    support_margin: float
    """
    How far, in metres, the center of mass stays inside the box around both feet, seen
    from above.
    """

    maximal_center_of_mass_speed: float
    """
    How fast, in metres per second, the center of mass may move across the floor.
    """

    maximal_center_of_mass_acceleration: float
    """
    How fast, in metres per second squared, its speed across the floor may change.
    """


# %% the controller


@dataclass(kw_only=True)
class WalkingController(ABC):
    """
    Balances a robot's base on its legs and walks it at a commanded velocity, standing
    on the spot while the velocity is zero.

    A controller keeps what it remembers between two calls of
    :meth:`leg_targets`, such as its gait's phase, until :meth:`reset`
    forgets it.
    """

    @classproperty
    @abstractmethod
    def pace(cls) -> WalkingPace:
        """
        How fast the controller walks the robot up to a pose.
        """

    @classproperty
    @abstractmethod
    def balance(cls) -> StandingBalance:
        """
        Where the robot keeps its center of mass while it stands and its upper body
        moves, in a physical simulation that walks it all the time, see
        :attr:`PhysicalBaseMode.WALKING`.
        """

    @property
    @abstractmethod
    def control_period(self) -> float:
        """
        How long, in seconds, the joints are driven towards one answer of
        :meth:`leg_targets` before the controller is asked again.
        """

    @property
    @abstractmethod
    def observed_joint_names(self) -> list[str]:
        """
        The joints whose positions and velocities the controller is told.
        """

    @property
    @abstractmethod
    def driven_joint_names(self) -> list[str]:
        """
        The joints the controller drives; nothing else commands them while it walks.
        """

    @property
    @abstractmethod
    def standing_posture(self) -> dict[str, float]:
        """
        The position of each driven joint the robot stands in before it is first asked,
        by the joint's name.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Forget everything remembered, as for a robot just put down in its
        :attr:`standing_posture`.
        """

    @abstractmethod
    def leg_targets(
        self, *, senses: WalkingSenses, command: PlanarVelocity
    ) -> dict[str, float]:
        """
        :param senses: What the robot senses now.
        :param command: The velocity the base is commanded to walk at.
        :return: The position each driven joint is driven towards until the next call,
            by the joint's name.
        """
