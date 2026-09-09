from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from giskardpy.executor import Executor
from giskardpy.middleware.ros2 import rospy
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    StateHistoryItem,
)


@dataclass
class StatechartStateLogger:
    """
    Logs every state change of the running motion statechart to the console.

    The statechart only records a snapshot when its state actually changed, so every
    recorded snapshot is a transition. Logging all of them shows what the robot is doing
    without printing anything once per control cycle.
    """

    executor: Executor
    """
    The executor holding the motion statechart that is reported on.
    """

    last_history_length: int = field(init=False, default=0)
    """
    Length of the statechart history at the most recent log, used to find the snapshots
    recorded since then.
    """

    _statechart: Optional[MotionStatechart] = field(init=False, default=None)
    """
    The statechart the history length above belongs to, so that the next goal starts
    logging from its own first snapshot.
    """

    def log_if_changed(self) -> None:
        """
        Log every snapshot the statechart recorded since the last call.
        """
        motion_statechart = self.executor.motion_statechart
        if motion_statechart is None:
            return
        if motion_statechart is not self._statechart:
            self._statechart = motion_statechart
            self.last_history_length = 0
        history = motion_statechart.history.history
        for index in range(self.last_history_length, len(history)):
            previous = history[index - 1] if index > 0 else None
            self.log_item(history[index], previous)
        self.last_history_length = len(history)

    def log_item(
        self, item: StateHistoryItem, previous: Optional[StateHistoryItem]
    ) -> None:
        """
        Log the nodes whose state differs between `previous` and `item`.

        :param item: The snapshot that was recorded.
        :param previous: The snapshot before it, or None for the first one of a goal.
        """
        changes = self.describe_changes(item, previous)
        if not changes:
            return
        rospy.get_node().get_logger().info(
            f"statechart cycle {item.control_cycle}: {', '.join(changes)}"
        )

    @staticmethod
    def describe_changes(
        item: StateHistoryItem, previous: Optional[StateHistoryItem]
    ) -> List[str]:
        """
        Describe every node that changed as ``name observation|life_cycle``, with the
        state it came from when there is one.

        :param item: The snapshot that was recorded.
        :param previous: The snapshot before it, or None for the first one of a goal.
        :return: One description per changed node, empty if nothing changed.
        """
        changes: List[str] = []
        for node, life_cycle in item.life_cycle_state.items():
            now = f"{item.observation_state[node].name}|{life_cycle.name}"
            if previous is None:
                changes.append(f"{node.name} {now}")
                continue
            before = (
                f"{previous.observation_state[node].name}"
                f"|{previous.life_cycle_state[node].name}"
            )
            if before != now:
                changes.append(f"{node.name} {before} -> {now}")
        return changes
