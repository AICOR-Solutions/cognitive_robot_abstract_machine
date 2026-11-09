"""
Checking, before a step acts, that the bodies it acts on still lie where the plan was
grounded with them, so a plan whose world changed since can be grounded anew rather than
acting on what no longer holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.failures import BodiesMovedSinceGrounding, BodyDisplacement
from cramph.data_types import ObservationStateValues
from cramph.node import EndedByOwner, StatechartNode
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% expected positions


@dataclass
class ExpectedPosition:
    """
    Where a body is expected to lie when a step starts.
    """

    body: Body
    """
    The body.
    """

    position: Point3
    """
    Where the origin of :attr:`body` is expected.
    """

    def displacement(self, world: World) -> BodyDisplacement:
        """
        :param world: The world holding the body.
        :return: How far the body lies from :attr:`position` now.
        """
        expected = world.transform(self.position, world.root)
        current = self.body.global_pose.position
        return BodyDisplacement(
            body=self.body, distance=float(expected.euclidean_distance(current))
        )


# %% checking them


@dataclass(eq=False, repr=False, kw_only=True)
class BodiesWhereExpected(EndedByOwner, StatechartNode):
    """
    Observes ``TRUE`` once every body of :attr:`expected` lies within :attr:`tolerance`
    of its expected position; otherwise it ends the statechart with
    :class:`~coraplex.plans.failures.BodiesMovedSinceGrounding`.
    """

    expected: list[ExpectedPosition]
    """
    The bodies and where they are expected.
    """

    tolerance: float = 0.02
    """
    How far, in metres, a body may lie from its expected position.
    """

    def displacements(self) -> list[BodyDisplacement]:
        """
        :return: The bodies lying farther than :attr:`tolerance` from where they are
            expected, and how far.
        """
        return [
            displacement
            for displacement in (
                expected.displacement(self.context.world) for expected in self.expected
            )
            if displacement.distance > self.tolerance
        ]

    def on_tick(self) -> ObservationStateValues | None:
        """
        :raises BodiesMovedSinceGrounding: When a body lies farther than
            :attr:`tolerance` from where it is expected.
        """
        displacements = self.displacements()
        if displacements:
            raise BodiesMovedSinceGrounding(displacements=displacements)
        return ObservationStateValues.TRUE
