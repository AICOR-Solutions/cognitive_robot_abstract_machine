"""
Tiago carries out what a person asks for: it serves the ``carry_out_intent`` skill,
through which Speech2Plan hands it the ``hri_actions_msgs/Intent`` of each task a person
confirmed, and builds and performs a plan for each in the IAI office, one at a time. On
its drives it says where it heads, asks the people at the elevator through Speech2Plan's
``ask_for_help`` skill whether it may board and says where it rides.

Bringing an item, it asks people through Speech2Plan's ``ask_for_help`` skill to put the
item on its tray and to take it off again, and goes on once they say they did. When a
step fails it asks them whether to try again, and on a yes plans anew from where it
stands, without collecting an item it already has.

A simulated run serves the office on
:class:`~semantic_digital_twin.adapters.ros.world_fetcher.FetchWorldServer` and keeps it
synchronized, so Speech2Plan grounds what it hears in the world the plans are performed
in. Start Speech2Plan next to it::

    ./scripts/run_node.sh
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from functools import partial
from queue import Queue

import rclpy
from hri_actions_msgs.msg import Intent
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.impl.rcutils_logger import RcutilsLogger
from speech2plan_msgs.action import AskForHelp, CarryOutIntent
from std_skills.msg import Result as SkillResult
from typing_extensions import Any

from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
)
from cramph.composites import ChildChooserAccess
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from krrood.exceptions import DataclassException
from experiments.intent_execution.help_skill import HelpSkillAsker
from experiments.intent_execution.plan_builder import (
    DrivingStep,
    IntentPlan,
    IntentPlanBuilder,
    PlanStep,
    PlannableIntent,
    TrayStep,
)
from experiments.tiago_room_request_demo.communicating_navigation import (
    TiagoCommunicatingNavigationDemonstration,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world import World

# %% the skills between Speech2Plan and the robot


class SkillPath(StrEnum):
    """
    Where the skills between Speech2Plan and the robot are served.
    """

    CARRY_OUT_INTENT = "/skill/carry_out_intent"
    """
    The ``speech2plan_msgs/action/CarryOutIntent`` skill this demonstration serves,
    which Speech2Plan calls.
    """

    ASK_FOR_HELP = "/skill/ask_for_help"
    """
    The ``speech2plan_msgs/action/AskForHelp`` skill Speech2Plan serves, which this
    demonstration calls.
    """


class HelpRequest(StrEnum):
    """
    What the robot asks when a step failed, with the step left open.
    """

    TRY_AGAIN = "I could not {doing}. Should I try again?"

    def verbalize(self, step: PlanStep) -> str:
        """
        :return: The request about ``step``.
        """
        doing = (
            f"get to the {step.destination.spoken_name}"
            if isinstance(step, DrivingStep)
            else step.summary
        )
        return self.format(doing=doing)


@dataclass
class TaskInProgress:
    """
    A task the skill took on, waiting for the main loop to carry it out.
    """

    goal: ServerGoalHandle
    """
    The goal of the task, with the intent to carry out.
    """

    result: CarryOutIntent.Result = field(default_factory=CarryOutIntent.Result)
    """
    How carrying the task out ended, once :attr:`finished` is set.
    """

    finished: threading.Event = field(default_factory=threading.Event)
    """
    Set once the main loop is done with the task.
    """


# %% a task being carried out


@dataclass
class StepFailure:
    """
    A step of a plan that failed, and why.
    """

    step: PlanStep
    """
    The step that failed.
    """

    number: int
    """
    The number of the step in its plan, counted from 1.
    """

    count: int
    """
    How many steps its plan has.
    """

    error: DataclassException
    """
    Why it failed.
    """


@dataclass
class TaskRun:
    """
    One intent being carried out: what it asks for, how far Tiago got with it, and how
    it ended.
    """

    intent: str
    """
    The name of the intent.
    """

    roles: dict[str, Any]
    """
    The ``data`` object of the intent.
    """

    goal: ServerGoalHandle | None = None
    """
    The goal the intent arrived with, on which Tiago reports what it does; None for an
    intent that arrived otherwise.
    """

    collected: bool = False
    """
    Whether the item is on the tray, so a plan made anew does not collect it again.
    """

    retries: int = 0
    """
    How often Tiago planned anew after people told it to try again.
    """

    result: CarryOutIntent.Result = field(default_factory=CarryOutIntent.Result)
    """
    How carrying the intent out ended, with what happened at the tray.
    """

    @property
    def is_canceled(self) -> bool:
        """
        Whether the caller of :attr:`goal` asked to cancel the intent.
        """
        return self.goal is not None and self.goal.is_cancel_requested

    def end(self, error_code: int, error_msg: str = "") -> None:
        """
        Let carrying the intent out end with ``error_code``, ``error_msg`` saying why.
        """
        self.result.result = SkillResult(error_code=error_code, error_msg=error_msg)

    def report(
        self, step: PlanStep, number: int, count: int, state: int, problem: str = ""
    ) -> None:
        """
        Tell the caller of :attr:`goal` that Tiago is in ``state`` at step ``number`` of
        ``count``, if the intent arrived as a goal.
        """
        if self.goal is None:
            return
        self.goal.publish_feedback(
            CarryOutIntent.Feedback(
                state=state,
                step=step.summary,
                step_number=number,
                step_count=count,
                problem=problem,
            )
        )


# %% the demonstration


@dataclass
class IntentExecutionDemonstration(TiagoCommunicatingNavigationDemonstration):
    """
    Carries out every task that arrives, one at a time, in the IAI office, turning
    down a task while it carries another out.
    """

    ros_node_name: str = "intent_execution"

    task_skill_path: str = SkillPath.CARRY_OUT_INTENT
    """
    Where the tasks arrive.
    """

    help_skill_path: str = SkillPath.ASK_FOR_HELP
    """
    Where people are asked for help.
    """

    pickup_room: str = "1.53"
    """
    The name of the room items are collected from, standing in for the pharmacy: on the
    first floor, so a delivery from it rides the elevator.
    """

    maximum_tray_requests: int = 2
    """
    How often Tiago asks the people at its tray before it gives the task up.
    """

    maximum_retries: int = 2
    """
    How often Tiago plans anew after people told it to try a failed step again.
    """

    help_server_timeout: timedelta = timedelta(seconds=10)
    """
    How long Tiago waits for Speech2Plan to serve the help skill.
    """

    help_answer_timeout: timedelta = timedelta(minutes=3)
    """
    How long Tiago waits for people to answer it.
    """

    step_in_progress: PlanStep | None = field(init=False, default=None)
    """
    The step of a plan being performed, None between intents.
    """

    busy: bool = field(init=False, default=False)
    """
    Whether a task was taken on and is not finished yet.
    """

    taking_tasks: threading.Lock = field(init=False, default_factory=threading.Lock)
    """
    Held while deciding whether to take on a task, so only one is taken on at a time.
    """

    @property
    def logger(self) -> RcutilsLogger:
        """
        The logger reporting the intents carried out, named after the demonstration's
        node.
        """
        return rclpy.logging.get_logger(self.ros_node_name)

    def plan_for(
        self,
        intent: str,
        roles: dict[str, Any],
        context: StatechartContext,
        collected: bool = False,
    ) -> IntentPlan:
        """
        :return: The plan carrying out ``intent`` with ``roles`` in ``context`` from
            where Tiago stands, without collecting an item already ``collected``.
        """
        plan = IntentPlanBuilder(
            context=context, demonstration=self, pickup_room=self.pickup_room
        ).plan_for(intent, roles, collected=collected)
        steps = "\n".join(
            f"  {number}. {step.summary}"
            for number, step in enumerate(plan.steps, start=1)
        )
        self.logger.info(f"plan for {plan.intent}:\n{steps}")
        return plan

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Perform :attr:`step_in_progress`, a drive.
        """
        return self.step_in_progress.build_statechart(context)

    # %% taking tasks

    def serve(self, world: World) -> Queue[TaskInProgress]:
        """
        Serve the task skill on :attr:`task_skill_path`, and ``world`` in a simulated
        run.

        :param world: The world the tasks are carried out in.
        :return: The tasks taken on, in the order they were, for the main loop to carry
            out with :meth:`perform`.
        """
        if not self.runs_on_a_real_robot:
            FetchWorldServer(node=self.ros_node, world=world)
            WorldSynchronizer(_world=world, node=self.ros_node)
        tasks: Queue[TaskInProgress] = Queue()
        ActionServer(
            self.ros_node,
            CarryOutIntent,
            str(self.task_skill_path),
            execute_callback=partial(self.hand_to_main_loop, tasks),
            goal_callback=self.take,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.logger.info(f"carrying out the tasks sent to {self.task_skill_path}")
        return tasks

    def take(self, goal: CarryOutIntent.Goal) -> GoalResponse:
        """
        :return: Whether to take on the task of ``goal``: unless another one is still
            being carried out.
        """
        with self.taking_tasks:
            if self.busy:
                self.logger.info(f"busy, turning down {goal.intent.intent}")
                return GoalResponse.REJECT
            self.busy = True
            return GoalResponse.ACCEPT

    def hand_to_main_loop(
        self, tasks: Queue[TaskInProgress], goal: ServerGoalHandle
    ) -> CarryOutIntent.Result:
        """
        Put the task of ``goal`` into ``tasks`` and wait until the main loop is done
        with it, since plans are performed on the main thread.

        :return: How carrying the task out ended.
        """
        task = TaskInProgress(goal=goal)
        tasks.put(task)
        task.finished.wait()
        outcome = task.result.result.error_code
        if outcome == SkillResult.ROS_ENOERR:
            goal.succeed()
        elif outcome == SkillResult.ROS_ECANCELED:
            goal.canceled()
        else:
            goal.abort()
        with self.taking_tasks:
            self.busy = False
        return task.result

    def perform(self, world: World, task: TaskInProgress) -> None:
        """
        Carry the task out in ``world``, reporting each step on its goal, and let the
        skill know once it is done with it.
        """
        try:
            task.result = self.carry_out(world, task.goal.request.intent, task.goal)
        finally:
            task.finished.set()

    # %% carrying an intent out

    def carry_out(
        self, world: World, intent: Intent, goal: ServerGoalHandle | None = None
    ) -> CarryOutIntent.Result:
        """
        Plan ``intent`` in ``world`` and perform the plan step by step. When a step
        fails, ask people whether to try again, and on a yes plan anew from where Tiago
        stands, at most :attr:`maximum_retries` times. An intent no plan is built for is
        skipped.

        :param goal: The goal ``intent`` arrived with, on which each step is reported as
            it starts, and whose cancellation stops the plan before the next step; None
            for an intent that arrived otherwise.
        :return: How carrying the intent out ended, with what happened at the tray.
        """
        run = TaskRun(intent=intent.intent, roles=json.loads(intent.data), goal=goal)
        if intent.intent not in set(PlannableIntent):
            self.logger.info(f"no plan is built for {intent.intent}, skipping it")
            run.end(SkillResult.ROS_ENOTSUP, f"no plan is built for {intent.intent}")
            return run.result
        failure = self.attempt(world, run)
        while failure is not None and self.may_try_again(run, failure):
            failure = self.attempt(world, run)
        return run.result

    def attempt(self, world: World, run: TaskRun) -> StepFailure | None:
        """
        Plan ``run`` from where Tiago stands and perform the plan, ending ``run`` unless
        a step fails.

        :return: The step that failed, or None when ``run`` ended.
        """
        executor = self.create_executor(world)
        try:
            plan = self.plan_of(run, executor.context)
            if plan is None:
                return None
            return self.perform_plan(world, executor, plan, run)
        finally:
            executor.close()
            self.step_in_progress = None

    def plan_of(self, run: TaskRun, context: StatechartContext) -> IntentPlan | None:
        """
        :return: The plan of ``run`` in ``context``, or None when its roles name nothing
            a plan is built for, which ends ``run``.
        """
        try:
            return self.plan_for(run.intent, run.roles, context, run.collected)
        except DataclassException as error:
            self.logger.error(f"cannot plan {run.intent}: {error}")
            run.end(SkillResult.ROS_EBADMSG, str(error))
            return None

    def perform_plan(
        self, world: World, executor: PlanExecutor, plan: IntentPlan, run: TaskRun
    ) -> StepFailure | None:
        """
        Perform the steps of ``plan`` one after the other, ending ``run`` once they are
        done, canceled or the people at the tray did not confirm.

        :return: The step that failed, or None when ``run`` ended.
        """
        count = len(plan.steps)
        for number, step in enumerate(plan.steps, start=1):
            if run.is_canceled:
                self.logger.info(f"canceled before '{step.summary}'")
                run.end(SkillResult.ROS_ECANCELED, f"canceled before '{step.summary}'")
                return None
            if isinstance(step, TrayStep):
                run.report(
                    step, number, count, CarryOutIntent.Feedback.WAITING_FOR_PEOPLE
                )
                if not self.have_tray_handled(step, run):
                    return None
                continue
            run.report(step, number, count, CarryOutIntent.Feedback.RUNNING)
            error = self.drive(world, executor, step)
            if error is not None:
                return StepFailure(step=step, number=number, count=count, error=error)
        run.end(SkillResult.ROS_ENOERR)
        return None

    def drive(
        self, world: World, executor: PlanExecutor, step: DrivingStep
    ) -> DataclassException | None:
        """
        Perform ``step`` with ``executor``, heading to its room as it starts and asking
        to board the elevator anew.

        :return: Why the drive failed, or None when Tiago arrived.
        """
        choosers = executor.context.require_extension(ChildChooserAccess).choosers
        self.step_in_progress = step
        self.head_to(step.destination, world)
        asking_to_board = self.ask_to_board_in(executor.context)
        try:
            executor.compile(self.build_statechart(executor.context))
            executor.execute()
        except DataclassException as error:
            return error
        finally:
            choosers.remove(asking_to_board)
        return None

    def may_try_again(self, run: TaskRun, failure: StepFailure) -> bool:
        """
        Ask people whether to try the failed step of ``run`` again, at most
        :attr:`maximum_retries` times, ending ``run`` when they do not say yes.

        :return: Whether to plan anew.
        """
        self.logger.error(f"'{failure.step.summary}' failed: {failure.error}")
        run.report(
            failure.step,
            failure.number,
            failure.count,
            CarryOutIntent.Feedback.ASKING_FOR_HELP,
            problem=str(failure.error),
        )
        if run.retries >= self.maximum_retries or not self.ask_for_help(
            AskForHelp.Goal.STEP_FAILED, HelpRequest.TRY_AGAIN.verbalize(failure.step)
        ):
            run.end(SkillResult.ROS_EOTHER, str(failure.error))
            return False
        run.retries += 1
        self.logger.info("trying again, planning from where Tiago stands")
        return True

    # %% asking people

    def have_tray_handled(self, step: TrayStep, run: TaskRun) -> bool:
        """
        Ask the people at the tray to load or unload it until they say they did, at
        most :attr:`maximum_tray_requests` times, recording on ``run`` that they did,
        and ending ``run`` when they did not.

        :return: Whether they said they did.
        """
        problem = (
            AskForHelp.Goal.UNLOAD_TRAY if step.unloading else AskForHelp.Goal.LOAD_TRAY
        )
        handled = any(
            self.ask_for_help(problem, step.request)
            for _ in range(self.maximum_tray_requests)
        )
        if not handled:
            run.end(SkillResult.ROS_EOTHER, f"nobody confirmed: {step.summary}")
            return False
        run.result.records.append(step.record)
        run.collected = run.collected or not step.unloading
        self.logger.info(step.record)
        return True

    def ask_for_help(self, problem: str, text: str) -> bool:
        """
        Ask the people around ``text`` through the help skill of Speech2Plan.

        :param problem: Why Tiago asks, one of the problems of ``AskForHelp``.
        :return: Whether someone answered yes; no when nobody serves the skill or
            answers in time.
        """
        asker = self.help_asker(problem)
        served = asker.wait_until_served()
        if not served:
            self.logger.error(f"nobody serves {self.help_skill_path}, cannot ask")
        agreed = served and asker.answer_to(text)
        asker.close()
        return agreed

    def boarding_asker(self, context: StatechartContext) -> HelpSkillAsker:
        """
        :return: The help skill of Speech2Plan asking the people at the elevator, which
            hears their answer without taking it for an instruction.
        """
        return self.help_asker(AskForHelp.Goal.BOARD_ELEVATOR)

    def help_asker(self, problem: str) -> HelpSkillAsker:
        """
        :param problem: Why Tiago asks, one of the problems of ``AskForHelp``.
        :return: An asker of the people around through the help skill.
        """
        return HelpSkillAsker(
            node=self.ros_node,
            skill_path=str(self.help_skill_path),
            problem=problem,
            server_timeout=self.help_server_timeout.total_seconds(),
            answer_timeout=self.help_answer_timeout.total_seconds(),
        )

    def run(self) -> World:
        """
        Acquire the world and carry out every task that arrives until ROS shuts down.

        :return: The world the tasks were carried out in.
        """
        try:
            world = self.acquire_world()
            self.start_simulation(world)
            tasks = self.serve(world)
            while rclpy.ok():
                self.perform(world, tasks.get())
        finally:
            self.tear_down()
        return world


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> None:
    """
    Carry out the intents that arrive.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real (or Gazebo) robot or simulates it.
    """
    IntentExecutionDemonstration(used_robot=Tiago, executor_type=executor_type).run()


if __name__ == "__main__":
    main()
