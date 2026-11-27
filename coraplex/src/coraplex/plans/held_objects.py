"""
Checking, while a hand lifts an object, that the object rises with it, where nothing but
contact carries the object and a grasp that missed leaves it behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.failures import ObjectLeftBehindByTheHand
from cramph.data_types import ObservationStateValue
from cramph.node import EndedByOwner, StatechartNode
from semantic_digital_twin.world_description.world_entity import Body

# %% rising with the hand


@dataclass(eq=False, repr=False, kw_only=True)
class ObjectRisesWithTheHand(EndedByOwner, StatechartNode):
    """
    Observes ``TRUE`` while :attr:`held_body` has risen as far as :attr:`tool_frame`
    since this node started, within :attr:`tolerance`; otherwise it ends the statechart
    with :class:`~coraplex.plans.failures.ObjectLeftBehindByTheHand`.
    """

    held_body: Body
    """
    The body the hand is to carry.
    """

    tool_frame: Body
    """
    The tool frame of the hand lifting :attr:`held_body`.
    """

    tolerance: float = 0.02
    """
    How much less, in metres, the body may rise than the hand: what it may slip in the
    hand, well under how far a pick lifts an object.
    """

    _start_heights: tuple[float, float] = field(init=False, repr=False)
    """
    The heights of :attr:`held_body` and :attr:`tool_frame` when this node started.
    """

    def on_start(self) -> None:
        """
        Also records the heights of :attr:`held_body` and :attr:`tool_frame`, see
        :meth:`~cramph.node.StatechartNode.on_start`.
        """
        super().on_start()
        self._start_heights = self._heights()

    def on_tick(self) -> ObservationStateValue | None:
        """
        :raises ObjectLeftBehindByTheHand: When the body rose less than the hand by more
            than :attr:`tolerance`.
        """
        (body_start, hand_start), (body_now, hand_now) = (
            self._start_heights,
            self._heights(),
        )
        shortfall = (hand_now - hand_start) - (body_now - body_start)
        if shortfall > self.tolerance:
            raise ObjectLeftBehindByTheHand(
                held_body=self.held_body, shortfall=shortfall
            )
        return ObservationStateValue.TRUE

    def _heights(self) -> tuple[float, float]:
        """
        :return: The heights of :attr:`held_body` and :attr:`tool_frame` in the world
            now, in that order.
        """
        return (
            float(self.held_body.global_pose.position.z),
            float(self.tool_frame.global_pose.position.z),
        )
