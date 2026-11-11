"""
The loop that runs a motion statechart on the robot, one control cycle at a time.
"""

from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

from typing_extensions import Iterator

from cramph.executor import StatechartExecutor
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.client_presence import ClientWatchdog
from giskardpy.middleware.ros2.command_publishing import CommandPublisher
from giskardpy.middleware.ros2.exceptions import (
    ClientDisconnectedError,
    ExecutionCanceledException,
    WorldModelModifiedDuringMotionError,
)
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from semantic_digital_twin.input_synchronization import WorldStateInputs
from giskardpy.middleware.ros2.world_updates import IncomingWorldUpdates
from giskardpy.motion_control import MotionControl
from semantic_digital_twin.world import World


# %% halting a stalled cycle


@dataclass
class StalledCycleHalt:
    """
    Commands the robot zero velocities while a control cycle stalls, such as while a
    node blocks its tick waiting for a reply, so its controllers do not keep moving it
    with the last command they received.
    """

    stall_timeout: float = 0.2
    """
    How long, in seconds, after the last command a cycle counts as stalled.
    """

    check_period: float = 0.01
    """
    How often, in seconds, the time since the last command is checked.
    """

    halted: bool = field(init=False, default=False)
    """
    Whether the robot was halted since the last command of a cycle.
    """

    last_command_time: float = field(init=False, default_factory=time.monotonic)
    """
    The :func:`time.monotonic` time the last cycle commanded the robot.
    """

    lock: Lock = field(init=False, default_factory=Lock, repr=False)
    """
    Keeps a halt from interleaving with the command of a cycle.
    """

    @contextmanager
    def watching(self, command_publishers: list[CommandPublisher]) -> Iterator[None]:
        """
        Halt the robot through `command_publishers` whenever a cycle stalls while the
        block runs.

        :param command_publishers: The publishers commanding the robot.
        """
        with ExitStack() as watching:
            self.last_command_time = time.monotonic()
            stopped = Event()
            watcher = Thread(
                target=self.halt_stalled_cycles,
                args=(command_publishers, stopped),
                name="stalled cycle halt",
                daemon=True,
            )
            watcher.start()
            watching.callback(watcher.join)
            watching.callback(stopped.set)
            yield

    def halt_stalled_cycles(
        self, command_publishers: list[CommandPublisher], stopped: Event
    ) -> None:
        """
        Halt the robot once per stall until `stopped` is set.

        :param command_publishers: The publishers commanding the robot.
        :param stopped: Set once the watching ends.
        """
        while not stopped.wait(self.check_period):
            with self.lock:
                if self.halted:
                    continue
                if time.monotonic() - self.last_command_time < self.stall_timeout:
                    continue
                for command_publisher in command_publishers:
                    command_publisher.stop()
                self.halted = True

    @contextmanager
    def commanding(self) -> Iterator[bool]:
        """
        Command the robot for a cycle while the block runs, which no halt interleaves
        with.

        :return: Whether the robot was halted while the cycle stalled.
        """
        with self.lock:
            yield self.halted
            self.halted = False
            self.last_command_time = time.monotonic()


@dataclass
class ControlLoop:
    """
    Runs a compiled motion statechart until it ends.

    Every cycle reads the inputs of the robot, ticks the controller and sends the
    resulting velocities back to the robot.
    """

    executor: StatechartExecutor
    """
    Computes the next command from the motion statechart.
    """

    action_server: ActionServerHandler
    """
    The action server the running goal belongs to; polled for cancel requests.
    """

    feedback_publisher: ActionFeedbackPublisher
    """
    Reports the state of the motion statechart to the action client.
    """

    inputs: WorldStateInputs
    """
    Writes the state of the robot into the world at the start of every cycle.
    """

    cycle_counter: CycleCounter
    """
    Ticked at the end of every cycle, shared with the idle loop of the motion server.
    """

    client_watchdog: ClientWatchdog
    """
    Watches the client of the running goal; polled for its disconnect.
    """

    world_updates: IncomingWorldUpdates
    """
    Delivers the world updates of other processes at the start of every cycle.

    State updates are applied right away; a model change would invalidate the compiled
    motion statechart, so it terminates the motion and is applied by the idle loop
    instead.
    """

    command_publishers: list[CommandPublisher] = field(default_factory=list)
    """
    Sends the computed velocities to the robot at the end of every cycle.
    """

    stalled_cycle_halt: StalledCycleHalt = field(default_factory=StalledCycleHalt)
    """
    Halts the robot while a cycle stalls.
    """

    @property
    def world(self) -> World:
        return self.executor.context.world

    def run(self) -> None:
        """
        Run cycles until the motion statechart reaches an end motion, halting the robot
        while a cycle stalls.

        :raises ExecutionCanceledException: If the goal was canceled.
        :raises ClientDisconnectedError: If the client of the goal disconnected.
        """
        with self.stalled_cycle_halt.watching(self.command_publishers):
            while True:
                self.run_cycle()
                if self.executor.statechart.is_ended():
                    return
                self.executor.pacer.sleep()

    def run_cycle(self) -> None:
        """
        Synchronize the inputs, compute the next command and publish it.

        :raises ExecutionCanceledException: If the goal was canceled.
        :raises ClientDisconnectedError: If the client of the goal disconnected.
        :raises WorldModelModifiedDuringMotionError: If another process modified the
            world model.
        """
        self.apply_world_updates()
        self.inputs.synchronize()
        self.raise_if_canceled()
        self.raise_if_client_disconnected()
        self.executor.tick()
        self.publish_commands()
        self.feedback_publisher.publish_if_changed()
        self.cycle_counter.tick()

    def apply_world_updates(self) -> None:
        """
        Take over the state of other processes and stop on a model change.

        :raises WorldModelModifiedDuringMotionError: If another process modified the
            world model, or is in the middle of doing so.
        """
        if self.world.world_is_being_modified:
            raise WorldModelModifiedDuringMotionError()
        self.world_updates.apply_state_updates()
        if self.world_updates.has_pending_model_change:
            raise WorldModelModifiedDuringMotionError()

    def raise_if_canceled(self) -> None:
        """
        :raises ExecutionCanceledException: If the client canceled the goal or a new
            goal superseded it.
        """
        if not self.action_server.is_cancel_requested():
            return
        self.action_server.loginfo("canceled")
        raise ExecutionCanceledException(
            action_server_name=self.action_server.action_name,
            goal_id=self.action_server.goal_id,
        )

    def raise_if_client_disconnected(self) -> None:
        """
        :raises ClientDisconnectedError: If the client that sent the goal is gone.
        """
        if not self.client_watchdog.is_client_gone():
            return
        self.action_server.loginfo("client disconnected")
        raise ClientDisconnectedError(client=self.client_watchdog.client)

    def publish_commands(self) -> None:
        """
        Send the velocities of the current cycle to the robot; after a stall halted it,
        stop it instead, so the next cycle starts from rest.
        """
        with self.stalled_cycle_halt.commanding() as halted:
            if halted:
                self.stop()
                return
            for command_publisher in self.command_publishers:
                command_publisher.publish()

    def stop(self) -> None:
        """
        Bring the robot to a halt and clear the commanded velocities.
        """
        for command_publisher in self.command_publishers:
            command_publisher.stop()
        MotionControl.set_velocity_acceleration_jerk_to_zero(self.world)
        self.world.notify_state_change()
