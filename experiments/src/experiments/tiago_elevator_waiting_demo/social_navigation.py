"""
Boarding an elevator by handing the robot's base to IRII's social navigation module,
which drives the robot into the cabin around the people near the elevator.

The module serves the ``ra_fetch_interfaces/action/BoardElevator`` action, a stand-in
until the interface with IRII is agreed; ``mock_social_navigation`` in
``tiago_arise_humans`` plays the module in the Gazebo simulator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import krrood.symbolic_math.symbolic_math as sm
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Vector3 as RosVector3
from ra_fetch_interfaces.action import BoardElevator
from ra_fetch_interfaces.msg import TrackedPerson

from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion
from experiments.tiago_elevator_waiting_demo.exceptions import (
    BoardingFailedError,
    BoardingTimedOutError,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import CancelMotion, Goal, NodeArtifacts
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import ActionServerTask
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    Point3ToRos2Converter,
    PoseToRos2StampedConverter,
)
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    ElevatorArea,
    Human,
    Level,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% the action the social navigation module serves


class SocialNavigationAction(StrEnum):
    """
    Names of the actions the social navigation module serves.
    """

    BOARD_ELEVATOR = "/ra_fetch/social_navigation/board_elevator"


@dataclass(eq=False, repr=False)
class BoardElevatorActionServerTask(
    ActionServerTask[
        BoardElevator,
        BoardElevator.Goal,
        BoardElevator.Impl.GetResultService.Response,
        BoardElevator.Feedback,
    ]
):
    """
    Hands the robot's base to the social navigation module, which drives the robot into
    an elevator's cabin around the persons near it.

    Observes nothing until the module gives the base back, then whether the robot's base
    stands inside the cabin, or false if the module rejected the goal or did not
    succeed.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator to board.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose base is handed over.
    """

    floor_tolerance: float = field(default=0.2, kw_only=True)
    """
    How far a level's floor may lie above the robot's base for the persons in front of
    the doors on that level to count as waiting.
    """

    _goal_accepted: bool | None = field(init=False, default=None)
    """
    Whether the module accepted the goal, None until it answered.
    """

    def build_msg(self, context: MotionStatechartContext) -> None:
        self._msg = self.boarding_goal()

    def boarding_goal(self) -> BoardElevator.Goal:
        """
        :return: The goal asking the module to drive the robot into the cabin, with the
            persons inside the cabin and those waiting in front of its doors on the
            robot's level.
        """
        world = self.elevator._world
        robot_level = Level.under(self.robot.root, self.floor_tolerance)
        return BoardElevator.Goal(
            robot_pose=PoseToRos2StampedConverter.convert(
                world.transform(Pose(reference_frame=self.robot.root), world.root)
            ),
            doorway_pose=PoseToRos2StampedConverter.convert(
                world.transform(self.elevator.doorway_pose, world.root)
            ),
            cabin_size=RosVector3(
                x=self.elevator.scale.x,
                y=self.elevator.scale.y,
                z=self.elevator.scale.z,
            ),
            persons_in_cabin=self._persons_in(self.elevator.cabin_area),
            persons_waiting=(
                []
                if robot_level is None
                else self._persons_in(self.elevator.waiting_area_on(robot_level))
            ),
        )

    def _persons_in(self, area: ElevatorArea) -> list[TrackedPerson]:
        """
        :return: The persons whose humans stand inside ``area``.
        """
        world = self.elevator._world
        return [
            TrackedPerson(
                person_id=human.name.name,
                position=Point3ToRos2Converter.convert(
                    world.transform(Point3(reference_frame=human.root), world.root)
                ),
            )
            for human in world.get_semantic_annotations_by_type(Human)
            if InsideRegion(human.root, area.root)()
        ]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Observe whether the robot's base stands inside the cabin's footprint.
        """
        cabin_T_base = context.world.compose_forward_kinematics_expression(
            self.elevator.cabin_area.root, self.robot.root
        )
        return NodeArtifacts(
            observation=sm.trinary_logic_and(
                sm.abs(cabin_T_base.x) <= self.elevator.scale.x / 2,
                sm.abs(cabin_T_base.y) <= self.elevator.scale.y / 2,
            )
        )

    def goal_response_callback(self, future) -> None:
        self._goal_accepted = future.result().accepted
        super().goal_response_callback(future)

    def result_callback(self, future) -> None:
        super().result_callback(future)
        logging.getLogger(__name__).info(
            f"The social navigation module gave the base back: "
            f"{self._result.result.reason}"
        )

    def on_tick(
        self, context: MotionStatechartContext
    ) -> ObservationStateValues | None:
        """
        Unlike :meth:`ActionServerTask.on_tick`, defers to the observation of
        :meth:`build_artifacts` once the module succeeded.
        """
        if self._goal_accepted is False:
            return ObservationStateValues.FALSE
        if self._result is None:
            return ObservationStateValues.UNKNOWN
        if self._result.status != GoalStatus.STATUS_SUCCEEDED:
            return ObservationStateValues.FALSE
        return None


# %% boarding inside a motion


@dataclass(eq=False, repr=False)
class SocialNavigationBoarding(Goal):
    """
    Boards an elevator through the social navigation module and ends the motion if the
    robot does not end up inside the cabin, or if the module keeps the base too long.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator to board.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose base is handed over.
    """

    timeout: float = field(kw_only=True)
    """
    Seconds of controller time the module may keep the base before the motion gives up.
    """

    action_name: str = field(
        default=SocialNavigationAction.BOARD_ELEVATOR, kw_only=True
    )
    """
    The name the module serves the boarding action under.
    """

    _boarding: BoardElevatorActionServerTask | None = field(init=False, default=None)
    """
    The handover to the module.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self._boarding = BoardElevatorActionServerTask(
            name=f"{self.name}/boarding",
            action_topic=self.action_name,
            message_type=BoardElevator,
            elevator=self.elevator,
            robot=self.robot,
        )
        failed = CancelMotion(
            name=f"{self.name}/failed",
            exception=BoardingFailedError(elevator_name=str(self.elevator.name)),
        )
        timer = CountSimulationTimeSeconds(
            name=f"{self.name}/timer", seconds=self.timeout
        )
        timed_out = CancelMotion(
            name=f"{self.name}/timed_out",
            exception=BoardingTimedOutError(timeout=self.timeout),
        )
        self._add_children_to_motion_statechart(
            [self._boarding, failed, timer, timed_out]
        )
        failed.start_condition = sm.trinary_logic_not(
            self._boarding.observation_variable
        )
        timed_out.start_condition = timer.observation_variable

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._boarding.observation_variable)


@dataclass
class SocialNavigationBoardingMotion(BaseMotion):
    """
    Lets the social navigation module drive the robot into an elevator's cabin.
    """

    elevator: Elevator
    """
    The elevator to board.
    """

    timeout: float
    """
    Seconds the social navigation module may keep the base.
    """

    def perform(self) -> None:
        return

    @property
    def _motion_chart(self) -> SocialNavigationBoarding:
        return SocialNavigationBoarding(
            elevator=self.elevator, robot=self.robot, timeout=self.timeout
        )


@dataclass
class BoardElevatorAction(ActionDescription):
    """
    Boards an elevator the robot waits in front of, leaving the way around the people
    near it to the social navigation module.
    """

    elevator: Elevator
    """
    The elevator to board.
    """

    timeout: float = 120.0
    """
    Seconds the social navigation module may keep the base before boarding fails.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            SocialNavigationBoardingMotion(self.elevator, self.timeout)
        )
