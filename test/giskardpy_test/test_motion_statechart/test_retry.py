from __future__ import annotations

import json

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.templates import Retry
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountStarts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    SucceedsAfterFailures,
    TestNodeAssertionError,
)
from semantic_digital_twin.world import World

# %% helpers


def retry_chart(retried_node: SucceedsAfterFailures, attempts: int) -> MotionStatechart:
    """
    Build a chart holding nothing but a retried node.

    :param retried_node: The node the retry runs.
    :param attempts: How often the retry may run it.
    :return: The chart, ready to be compiled.
    """
    motion_statechart = MotionStatechart()
    retry = Retry(
        retried_node=retried_node,
        attempts=attempts,
        exception=TestNodeAssertionError(reason="out of attempts"),
    )
    motion_statechart.add_node(retry)
    motion_statechart.add_node(EndMotion.when_true(retry))
    return motion_statechart


def run_until_end(motion_statechart: MotionStatechart) -> None:
    """
    Tick a chart until the motion ends.

    :param motion_statechart: The chart to run.
    """
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end(100)


def run_retry(retried_node: SucceedsAfterFailures, attempts: int) -> None:
    """
    Run a chart holding nothing but a retried node until the motion ends.

    :param retried_node: The node the retry runs.
    :param attempts: How often the retry may run it.
    """
    run_until_end(retry_chart(retried_node, attempts))


# %% counting starts


def test_count_starts_turns_true_on_its_last_start():
    """
    Counting restarts is the point, so the count has to survive the reset that causes
    the next start rather than being cleared by it.
    """
    counter = CountStarts(starts=2)
    context = MotionStatechartContext(world=World())

    counter.on_start(context)
    first_observation = counter.on_tick(context)
    counter.on_reset(context)
    counter.on_start(context)

    assert first_observation == ObservationStateValues.FALSE
    assert counter.on_tick(context) == ObservationStateValues.TRUE


# %% retrying until success


def test_retry_runs_the_node_again_until_it_succeeds():
    """
    A node that fails once is not necessarily a node that cannot succeed, so the motion
    goes on after the retries that were needed.
    """
    node = SucceedsAfterFailures(failures=2)

    run_retry(node, attempts=3)

    assert node.runs == 3


def test_retry_runs_a_node_that_succeeds_only_once():
    """
    Retrying costs an attempt and a query, so nothing may be repeated while the node is
    reporting success.
    """
    node = SucceedsAfterFailures(failures=0)

    run_retry(node, attempts=3)

    assert node.runs == 1


def test_retry_cancels_the_motion_once_the_attempts_are_used_up():
    """
    A node that keeps failing has to reach the caller as the failure the retry was given
    for it, rather than as a motion that ran out of ticks.
    """
    node = SucceedsAfterFailures(failures=10)

    with pytest.raises(TestNodeAssertionError):
        run_retry(node, attempts=3)

    assert node.runs == 3


# %% crossing to a controller


def test_a_restored_retry_still_runs_the_node_again():
    """
    On the real robot the chart is run by the controller it was serialized to, so the
    retry has to survive the trip as the retry it was and not just as its node.
    """
    motion_statechart = retry_chart(SucceedsAfterFailures(failures=2), attempts=3)
    restored = MotionStatechart.from_json(
        json.loads(json.dumps(motion_statechart.to_json()))
    )

    run_until_end(restored)

    restored_nodes = restored.get_nodes_by_type(SucceedsAfterFailures)
    assert len(restored_nodes) == 1
    assert restored_nodes[0].runs == 3
