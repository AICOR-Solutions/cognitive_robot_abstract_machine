from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.task import Future
from typing_extensions import Any

from cramph.data_types import ObservationStateValues
from cramph.node import EndedByOwner, StatechartNode
from giskardpy.motion_statechart.ros_context import RosContextExtension

logger = logging.getLogger(__name__)


# %% sending an action goal and waiting for it to end


@dataclass(eq=False, repr=False, kw_only=True)
class AwaitActionResult(EndedByOwner, StatechartNode, ABC):
    """
    Sends a goal to a ROS action server when it starts and observes true once the goal
    ended, whatever its outcome.

    An action nobody serves, a rejected goal, a goal that did not succeed and a goal
    still running after :attr:`result_timeout` only log a warning, so they never fail
    the plan. Observes true right away where there is no ROS node, no
    :attr:`action_name` or no action type.
    """

    action_name: str | None = None
    """
    Name of the ROS action the goal is sent to, None to send nothing.
    """

    server_timeout: float = 2.0
    """
    Seconds to wait during :meth:`set_up` for the action to be served.
    """

    result_timeout: float = 30.0
    """
    Seconds after sending the goal at which this node stops waiting for it to end.
    """

    tick_pause_while_waiting: float = 0.005
    """
    Seconds every tick sleeps while the goal runs, which lets the callbacks reporting
    its end run while a simulated plan ticks unpaced.
    """

    _action_client: ActionClient | None = field(init=False, default=None)
    """
    The client the goal is sent with, None where nothing is sent.
    """

    _goal_ended: bool = field(init=False, default=False)
    """
    Whether the goal sent last ended.
    """

    _deadline: float = field(init=False, default=0.0)
    """
    When to stop waiting for the goal sent last, on the monotonic clock.
    """

    @abstractmethod
    def find_action_type(self) -> type | None:
        """
        :return: The type of the action, None if it is not available here.
        """

    @abstractmethod
    def build_action_goal(self) -> Any:
        """
        :return: The goal to send.
        """

    def set_up(self) -> None:
        super().set_up()
        if self.action_name is None:
            return
        ros_context_extension = self.context.get_extension(RosContextExtension)
        if ros_context_extension is None or ros_context_extension.ros_node is None:
            return
        action_type = self.find_action_type()
        if action_type is None:
            return
        self._action_client = ros_context_extension.get_or_create_action_client(
            action_type, self.action_name
        )
        self._action_client.wait_for_server(timeout_sec=self.server_timeout)

    def on_start(self) -> None:
        if self._action_client is None:
            self._goal_ended = True
            return
        if not self._action_client.server_is_ready():
            logger.warning(f"Action {self.action_name} is not served, skipping it.")
            self._goal_ended = True
            return
        self._goal_ended = False
        self._deadline = time.monotonic() + self.result_timeout
        self._action_client.send_goal_async(self.build_action_goal()).add_done_callback(
            self._receive_goal_response
        )

    def _receive_goal_response(self, future: Future) -> None:
        """
        Wait for the result of an accepted goal, or end a rejected one.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.warning(f"Action {self.action_name} rejected its goal.")
            self._goal_ended = True
            return
        goal_handle.get_result_async().add_done_callback(self._receive_result)

    def _receive_result(self, future: Future) -> None:
        """
        End the goal, whose outcome ``future`` holds.
        """
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            logger.warning(f"Action {self.action_name} ended with status {status}.")
        self._goal_ended = True

    def on_tick(self) -> ObservationStateValues:
        if self._goal_ended:
            return ObservationStateValues.TRUE
        if time.monotonic() < self._deadline:
            time.sleep(self.tick_pause_while_waiting)
            return ObservationStateValues.UNKNOWN
        logger.warning(
            f"Action {self.action_name} did not end within {self.result_timeout} s."
        )
        return ObservationStateValues.TRUE
