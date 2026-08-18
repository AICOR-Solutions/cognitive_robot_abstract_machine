from __future__ import division

from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional

from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import (
    sum,
    trinary_logic_and,
    trinary_logic_not,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TerminalNode,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountStarts


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Takes a list of nodes and wires their start/end conditions such that they are
    executed in order.

    Its observation is the observation of the last node in the sequence.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: MotionStatechartContext) -> None:
        last_node: Optional[MotionStatechartNode] = None
        for i, node in enumerate(self.nodes):
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.observation_variable
            # A node that ends the motion has nothing left to transition to.
            if not isinstance(node, TerminalNode):
                node.end_condition = node.observation_variable
            last_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.nodes[-1].observation_variable)


@dataclass(repr=False, eq=False)
class Parallel(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    This nodes' observation state turns True when up to `minimum_success` nodes are
    True.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    Defines the minimum number of nodes that must be True for the goal to be achieved.

    Defaults to None, which means that all nodes must be True.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        true_observation_variables = [
            x.observation_variable == True for x in self.nodes
        ]
        minimum_success = (
            self.minimum_success
            if self.minimum_success is not None
            else len(self.nodes)
        )
        return NodeArtifacts(
            observation=minimum_success <= sum(*true_observation_variables)
        )


@dataclass(repr=False, eq=False)
class Retry(Goal):
    """
    Runs a node again from the start whenever it observes False, and cancels the motion
    once the attempts are used up.

    Its observation is the observation of the retried node, so it is True exactly when an
    attempt succeeded.

    ..note:: A failed attempt has to be reported as a False observation. A node that
        raises instead aborts the whole chart before the retry can see it.
    """

    retried_node: MotionStatechartNode = field(kw_only=True)
    """
    The node that is run again after a failed attempt.
    """

    attempts: int = field(default=3, kw_only=True)
    """
    How often the node may be run in total, the first run included.
    """

    exception: DataclassException = field(kw_only=True)
    """
    Raised once the attempts are used up.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        attempt_counter = CountStarts(
            name=f"{self.name}/attempts", starts=self.attempts
        )
        cancel = CancelMotion(exception=self.exception)
        self.add_nodes([self.retried_node, attempt_counter, cancel])

        attempt_failed = trinary_logic_not(self.retried_node.observation_variable)
        attempt_left = trinary_logic_and(
            attempt_failed, trinary_logic_not(attempt_counter.observation_variable)
        )
        # A failed attempt starts the counter, which counts that attempt, and the next
        # tick sends both back to NOT_STARTED so the node runs again and the counter is
        # ready to count the attempt after it.
        attempt_counter.start_condition = attempt_failed
        attempt_counter.reset_condition = attempt_left
        self.retried_node.reset_condition = attempt_left
        cancel.start_condition = attempt_counter.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.retried_node.observation_variable)
