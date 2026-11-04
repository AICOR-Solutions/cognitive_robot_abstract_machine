import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from json_msgs.action import JsonAction
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from typing_extensions import Any

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.ros2_nodes.await_action_result import (
    AwaitActionResult,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from semantic_digital_twin.world import World

# %% running the node


class AwaitedAction(StrEnum):
    """
    Actions the tests send goals to.
    """

    GREETING = "/greeting"


@dataclass(eq=False, repr=False, kw_only=True)
class GreetingGoal(AwaitActionResult):
    """
    Sends a goal of an action type every checkout has.
    """

    action_name: str | None = AwaitedAction.GREETING

    def find_action_type(self) -> type:
        return JsonAction

    def build_action_goal(self) -> Any:
        return JsonAction.Goal(goal="Hello")


def perform_greeting_goal(rclpy_node, result_timeout: float = 30.0) -> None:
    """
    Tick a statechart holding only a :class:`GreetingGoal` until the node observed true.

    :param rclpy_node: The node the goal is sent from.
    :param result_timeout: Seconds the node waits for the goal to end.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=World()),
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node := GreetingGoal(result_timeout=result_timeout))
    statechart.add_node(EndMotion.when_true(node))
    executor.compile(statechart=statechart)
    executor.tick_until_end(timeout=1_000_000)


def serve_greeting(rclpy_node, execute: Callable, **options) -> ActionServer:
    """
    :return: A server of :attr:`AwaitedAction.GREETING` running ``execute``.
    """
    return ActionServer(
        rclpy_node, JsonAction, AwaitedAction.GREETING, execute, **options
    )


# %% waiting for the goal to end


def test_the_node_ends_only_after_the_goal_ended(rclpy_node):
    goal_ended = threading.Event()

    def execute(goal_handle: ServerGoalHandle) -> JsonAction.Result:
        time.sleep(0.5)
        goal_ended.set()
        goal_handle.succeed()
        return JsonAction.Result()

    server = serve_greeting(rclpy_node, execute)

    perform_greeting_goal(rclpy_node)

    assert goal_ended.is_set()
    server.destroy()


# %% never hanging on the action


def test_the_node_ends_when_the_goal_is_rejected(rclpy_node):
    goal_decided = threading.Event()

    def reject(_) -> GoalResponse:
        goal_decided.set()
        return GoalResponse.REJECT

    server = serve_greeting(rclpy_node, lambda _: None, goal_callback=reject)

    perform_greeting_goal(rclpy_node)

    assert goal_decided.is_set()
    server.destroy()


def test_the_node_ends_when_the_goal_fails(rclpy_node):
    goal_failed = threading.Event()

    def execute(goal_handle: ServerGoalHandle) -> JsonAction.Result:
        goal_handle.abort()
        goal_failed.set()
        return JsonAction.Result()

    server = serve_greeting(rclpy_node, execute)

    perform_greeting_goal(rclpy_node)

    assert goal_failed.is_set()
    server.destroy()


def test_the_node_ends_when_nobody_serves_the_action(rclpy_node):
    started_at = time.monotonic()

    perform_greeting_goal(rclpy_node)

    assert time.monotonic() - started_at >= GreetingGoal().server_timeout


def test_the_node_stops_waiting_after_its_result_timeout(rclpy_node):
    release = threading.Event()
    goal_running = threading.Event()

    def execute(goal_handle: ServerGoalHandle) -> JsonAction.Result:
        goal_running.set()
        release.wait()
        goal_handle.succeed()
        return JsonAction.Result()

    server = serve_greeting(rclpy_node, execute)
    result_timeout = 0.5

    started_at = time.monotonic()
    perform_greeting_goal(rclpy_node, result_timeout)

    assert time.monotonic() - started_at >= result_timeout
    assert goal_running.is_set()
    release.set()
    server.destroy()


# %% reaching the controller


def test_the_node_survives_the_serialization_a_controller_receives_it_in():
    node = GreetingGoal()
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(node)

    received = Statechart.from_json(
        json.loads(json.dumps(statechart.to_json())),
        context=StatechartContext(world=World()),
    ).nodes[0]

    assert received.action_name == node.action_name
