from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from typing_extensions import Any

from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import (
    Arm,
    EndEffector,
)
from krrood_spatial_math.spatial_types import Point3, Pose

from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.robot_plans.actions.base import (
    GoalAction,
    ParallelAction,
    SequentialAction,
)
from coraplex.robot_plans.actions.core.exceptions import WidthBeyondGripperOpening
from coraplex.robot_plans.actions.core.pick_up import HasGraspChoice
from coraplex.robot_plans.mixins import (
    HasMaxJointVelocity,
    MovesToolCenterPoint,
    OperatesGripper,
)
from cramph.composites import Parallel, Sequence
from cramph.node import StatechartNode
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)


@dataclass(eq=False, repr=False, kw_only=True)
class MoveTorsoAction(SequentialAction):
    """
    Move the torso of the robot up and down.
    """

    torso_state: TorsoState
    """
    The state of the torso that should be set.
    """

    def create_steps(self) -> list[StatechartNode]:
        joint_state = self.robot.get_torso().get_joint_state_by_type(self.torso_state)
        return [JointPositionList(goal_state=joint_state)]


@dataclass(eq=False, repr=False, kw_only=True)
class SetGripperAction(OperatesGripper, GoalAction):
    """
    Moves a gripper to the given configuration.
    """

    configuration: GripperConfiguration
    """
    The configuration the gripper is to reach.
    """

    def create_goal(self) -> StatechartNode:
        return self.gripper_goal_type.moving_gripper_to(self.configuration)


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingersToWidthAction(GoalAction):
    """
    Closes a two-finger gripper until its fingertips stand a given distance apart,
    rather than driving it to a gripper state of its own.

    Closing a gripper all the way onto an object that is not centred between the
    fingers wedges it sideways instead of holding it, so an object held by contact alone
    is better met at its own width. Giskard is given the distance between the fingertips
    as the goal and finds the joint positions that realise it.

    .. note:: This only means anything where something pushes back, see
        :class:`~coraplex.robot_plans.plan_transformations.CloseFingersToObjectWidth`.
    """

    gripper: HasTwoFingers
    """
    The gripper whose fingers close.
    """

    width: float
    """
    How far apart the fingertips' facing surfaces are to stand, in metres.
    """

    squeeze: float = 0.0
    """
    How far past that width each fingertip is sent, in metres, so the fingers press into
    an object rather than merely touching it. The object stops them short of that, so
    fingers squeezing it count as done once they stopped approaching it.

    Keep it small, since it is commanded penetration into a body that does not yield.
    """

    threshold: float = 0.0005
    """
    How close, in metres, the fingertips have to come to the goal for the action to
    count as done: well under a squeeze, so the squeeze is not lost to the tolerance.
    """

    def create_goal(self) -> StatechartNode:
        """
        :raises WidthBeyondGripperOpening: If :attr:`width` is wider than the gripper
            opens.
        """
        if self.width > self.gripper.separation_when_open:
            raise WidthBeyondGripperOpening(gripper=self.gripper, width=self.width)
        return MoveGripper(
            gripper=self.gripper,
            width=max(0.0, self.width - 2 * self.squeeze),
            threshold=self.threshold,
            tolerate_stall=self.squeeze > 0.0,
            name="MoveFingersToWidth",
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ParkArmsAction(ParallelAction, HasMaxJointVelocity):
    """
    Park the arms of the robot.
    """

    arms: list[Arm]
    """
    The arms that should be parked.
    """

    def create_steps(self) -> list[StatechartNode]:
        park_state = self.park_joint_state()
        joint_goal = self._through_raised_arms(JointPositionList(goal_state=park_state))
        if self.max_joint_velocity is None:
            return [joint_goal, *self.keep_base_still()]
        return [
            joint_goal,
            *self.keep_base_still(),
            JointVelocityLimit(
                connections=list(park_state.connections),
                max_velocity=self.max_joint_velocity,
            ),
        ]

    def _through_raised_arms(self, parking: JointPositionList) -> StatechartNode:
        """
        :param parking: The goal folding the arms into their park position.
        :return: `parking`, after a goal raising the arms that declare a
            :attr:`~semantic_digital_twin.datastructures.definitions.StaticJointState.RAISED`
            state and either took a grasp earlier in the plan or hold their hands
            farther out than that state does (see :meth:`_stands_out`), which draw their
            hands straight to where that state holds them and from there straight to
            where the parked state does: back over a table they reached over, then down
            beside it.
        """
        reaching_arms = {
            reach.arm
            for reach in self.statechart.get_preceding_nodes_by_type(
                self, HasGraspChoice
            )
        }
        raised_arms = [
            arm
            for arm in self.arms
            if arm.has_joint_state_of_type(StaticJointState.RAISED)
            and (arm in reaching_arms or self._stands_out(arm))
        ]
        if not raised_arms:
            return parking
        raising, lowering = (
            Parallel(
                nodes=[
                    CartesianPosition(
                        root_link=arm.root,
                        tip_link=arm.end_effector.tool_frame,
                        goal_point=self._tool_position_in_state(arm, state_type),
                    )
                    for arm in raised_arms
                ]
            )
            for state_type in (StaticJointState.RAISED, StaticJointState.PARK)
        )
        return Sequence(nodes=[raising, lowering, parking])

    def _stands_out(self, arm: Arm) -> bool:
        """
        A park performed as a plan of its own, as one step of a plan run step by step,
        knows of no grasp before it; where the arm's hand stands tells instead.

        :param arm: An arm declaring a raised and a parked state.
        :return: Whether the arm's hand now stands farther from where its parked state
            holds it than where its raised state holds it.
        """
        parked = self._tool_position_in_state(arm, StaticJointState.PARK).to_np()[:3]
        raised = self._tool_position_in_state(arm, StaticJointState.RAISED).to_np()[:3]
        now = self.world.transform(
            target_frame=arm.root,
            spatial_object=Point3(reference_frame=arm.end_effector.tool_frame),
        ).to_np()[:3]
        return bool(np.linalg.norm(now - parked) > np.linalg.norm(raised - parked))

    def _tool_position_in_state(self, arm: Arm, state_type: StaticJointState) -> Point3:
        """
        :param arm: An arm declaring a joint state of `state_type`.
        :param state_type: The joint state to stand the arm in.
        :return: Where the arm's tool frame stands with the arm in that state, in the
            frame of the arm's root, so a torso bent or turned while the hand is drawn
            back neither helps nor hinders it.
        """
        with self.world.reset_state_context():
            arm.get_joint_state_by_type(state_type).apply_to(self.world)
            return self.world.transform(
                target_frame=arm.root,
                spatial_object=Point3(reference_frame=arm.end_effector.tool_frame),
            )

    def park_joint_state(self) -> JointState:
        """
        :return: The joint state that puts every arm this action parks into its park
            position, and the robot's torso and neck too where they declare one, so a
            humanoid that bent its waist or turned its head to reach parks upright.
        """
        parked_body_parts = [
            part
            for part in [self.robot.get_torso_if_specified(), *self.robot.all_necks]
            if part is not None and part.has_joint_state_of_type(StaticJointState.PARK)
        ]
        connections = []
        target_values = []
        for part in [*self.arms, *parked_body_parts]:
            joint_state = part.get_joint_state_by_type(StaticJointState.PARK)
            connections.extend(joint_state.connections)
            target_values.extend(joint_state.target_values)
        return JointState(connections=connections, target_values=target_values)


@dataclass(eq=False, repr=False, kw_only=True)
class FollowToolCenterPointPathAction(MovesToolCenterPoint, ParallelAction):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a path of
    poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    arm: Arm
    """
    The arm to use.
    """

    position_threshold: float = CartesianPose.translation_threshold
    """
    Distance threshold in meters within which the tool center point passes a waypoint.
    """

    orientation_threshold: float = CartesianPose.orientation_threshold
    """
    Rotation threshold in rad within which the tool center point passes a waypoint.
    """

    def create_steps(self) -> list[StatechartNode]:
        waypoints = Sequence(
            nodes=[self._waypoint_goal(pose) for pose in self.target_locations.poses]
        )
        return [waypoints, *self.keep_base_still()]

    def _waypoint_goal(self, target: Pose) -> CartesianPose:
        """
        :param target: The waypoint the tool center point passes through.
        :return: The task reaching that waypoint.
        """
        return CartesianPose(
            root_link=self.controlled_root,
            tip_link=self.arm.end_effector.tool_frame,
            goal_pose=target,
            translation_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )

    def validate(
        self,
        result: Any | None = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass


@dataclass(eq=False, repr=False, kw_only=True)
class MoveManipulatorAction(MovesToolCenterPoint, GoalAction):
    """
    Move the end_effector to a specific pose.
    """

    target_pose: Pose
    """
    The pose where the end_effector should be moved to.
    """

    end_effector: EndEffector
    """
    The end_effector that should be moved.
    """

    allow_gripper_collision: bool
    """
    If the gripper can collide with something.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for :attr:`end_effector` to touch its surroundings if
            :attr:`allow_gripper_collision`, none otherwise.
        """
        if not self.allow_gripper_collision:
            return []
        return [AllowCollisionForEndEffector(end_effector=self.end_effector)]

    def create_goal(self) -> StatechartNode:
        goal = CartesianPose(
            root_link=self.controlled_root,
            tip_link=self.end_effector.tool_frame,
            goal_pose=self.target_pose,
            translation_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        for node in self.keep_base_still():
            goal.add_node(node)
        return goal
