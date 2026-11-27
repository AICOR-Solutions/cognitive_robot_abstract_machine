"""
Holding the base of a robot that walks on legs where it stands once it walked there, a
shortcut a physical simulation takes, see
:class:`~coraplex.plans.context_extensions.WalkingBasesHeld`.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.context_extensions import WalkingBasesHeld
from cramph.data_types import ObservationStateValue
from cramph.node import StatechartNode, SucceedsOnObservingTrue
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% holding a base once it stands still


@dataclass(eq=False, repr=False, kw_only=True)
class BaseHeldWhereItStands(SucceedsOnObservingTrue, StatechartNode):
    """
    Observes ``TRUE`` once the simulation holds the base of the robot rooted at
    :attr:`robot_root` where it stands, which it does once the base stood still, see
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoWalking.hold_when_still`.
    """

    robot_root: KinematicStructureEntity
    """
    The root of the robot whose base is held.
    """

    def on_tick(self) -> ObservationStateValue | None:
        """
        Observes ``TRUE`` once the base stands still and is held, ``UNKNOWN`` until
        then, see :meth:`~cramph.node.StatechartNode.on_tick`.
        """
        walking = self.context.require_extension(
            extension_type=WalkingBasesHeld
        ).simulation.walking_robot_rooted_at(self.robot_root)
        if walking.hold_when_still():
            return ObservationStateValue.TRUE
        return ObservationStateValue.UNKNOWN
