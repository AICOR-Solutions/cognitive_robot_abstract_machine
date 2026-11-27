"""
Tasks moving the base a drive carries along the floor and about the vertical, relative
to the parent of that drive, and the nodes putting the base of a robot at a pose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Self

import krrood.symbolic_math.symbolic_math as sm
from cramph.composites import Parallel, Sequence
from cramph.node import (
    CompositeNode,
    FailsOnObservingFalse,
    NodeArtifacts,
    StatechartNode,
    SucceedsOnObservingTrue,
    expanded_child_field,
)
from giskardpy.motion_statechart.forward_kinematics_binding import (
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.exceptions import (
    RobotDoesNotWalkError,
    RobotWithoutMobileBaseError,
)
from giskardpy.motion_statechart.graph_node import (
    BindsGoalReferenceFrame,
    ConvergingTask,
    MotionNodeArtifacts,
)
from giskardpy.motion_statechart.mixins import (
    HasCruiseAngularVelocityParameter,
    HasCruiseVelocityParameter,
    HasGoalPoseParameter,
    HasThresholdParameter,
    HasWeightParameter,
)
from giskardpy.motion_statechart.weights import VelocityScale
from krrood.symbolic_math.symbolic_math import Scalar, trinary_if_cases
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, MobileBase
from semantic_digital_twin.robots.walking import WalkingPace
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% moving a driven base


@dataclass(eq=False, repr=False, kw_only=True)
class DrivenBaseTask(ConvergingTask, HasGoalPoseParameter, ABC):
    """
    Moves the base a drive carries towards :attr:`goal_pose`, relative to the parent of
    the drive as it is whenever the task is built.

    A base moved under another parent while the statechart runs is therefore driven
    relative to that parent, and the joints above the drive are never moved for it.
    """

    driven_body: KinematicStructureEntity
    """
    The body the drive carries.
    """

    parent_T_goal_reference_frame: HomogeneousTransformationMatrix
    """
    The pose of the reference frame of :attr:`goal_pose` relative to the parent of the
    drive, as the goal running this task fixed it.
    """

    @property
    def drive_parent(self) -> KinematicStructureEntity:
        """
        :return: The body the drive carrying :attr:`driven_body` is attached to now.
        """
        return self.driven_body.parent_kinematic_structure_entity

    @property
    def parent_T_goal(self) -> Pose:
        """
        :return: :attr:`goal_pose` relative to :attr:`drive_parent`.
        """
        parent_T_goal = self.parent_T_goal_reference_frame @ self.goal_pose
        parent_T_goal.reference_frame = self.drive_parent
        return parent_T_goal

    def parent_T_base(self) -> HomogeneousTransformationMatrix:
        """
        :return: The pose of :attr:`driven_body` relative to :attr:`drive_parent`, over
            the degrees of freedom of the drive.
        """
        return self.context.world.compose_forward_kinematics_expression(
            self.drive_parent, self.driven_body
        )


@dataclass(eq=False, repr=False, kw_only=True)
class BasePosition(DrivenBaseTask, HasCruiseVelocityParameter):
    """
    Drives the base along the floor to below the position of :attr:`goal_pose`, with an
    error in meters.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The artifacts of this task, whose error is the distance along the floor
            between the base and the goal position.
        """
        artifacts = MotionNodeArtifacts()
        parent_P_goal = self.parent_T_goal.position
        parent_P_base = self.parent_T_base().position
        parent_V_base_to_goal = parent_P_goal - parent_P_base
        parent_V_base_to_goal.z = 0
        for base_coordinate, goal_coordinate, cruise_velocity in zip(
            (parent_P_base.x, parent_P_base.y),
            (parent_P_goal.x, parent_P_goal.y),
            self.cruise_velocities(parent_V_base_to_goal),
        ):
            artifacts.geometry.add_position_constraint(
                expression_current=base_coordinate,
                expression_goal=goal_coordinate,
                cruise_velocity=cruise_velocity,
                velocity_scale=VelocityScale.TRANSLATION,
                quadratic_weight=self.weight,
            )
        artifacts.error = parent_V_base_to_goal.norm()
        return artifacts

    def cruise_velocities(
        self, parent_V_base_to_goal: Vector3
    ) -> tuple[sm.ScalarData, sm.ScalarData]:
        """
        :param parent_V_base_to_goal: The way from the base to the goal along the floor,
            relative to the drive's parent.
        :return: The speed the base approaches the goal at along the parent's x and y.
        """
        return self.cruise_velocity, self.cruise_velocity


@dataclass(eq=False, repr=False, kw_only=True)
class DrivenBaseRotation(DrivenBaseTask, HasCruiseAngularVelocityParameter, ABC):
    """
    Turns the base about the vertical until it faces the way the orientation
    :meth:`parent_R_goal` describes does, with an error in radians.
    """

    @abstractmethod
    def parent_R_goal(
        self, *, parent_T_base: HomogeneousTransformationMatrix
    ) -> RotationMatrix:
        """
        :param parent_T_base: The pose of the base relative to the drive's parent.
        :return: The orientation the base should have, relative to the drive's parent.
        """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The artifacts of this task, whose error is the angle about the vertical
            between the orientation of the base and the goal orientation.
        """
        artifacts = MotionNodeArtifacts()
        parent_T_base = self.parent_T_base()
        parent_R_base = parent_T_base.rotation_matrix
        parent_R_goal = self.parent_R_goal(parent_T_base=parent_T_base)
        yaw_error = sm.shortest_angular_distance(
            parent_R_base.rpy[2], parent_R_goal.rpy[2]
        )
        artifacts.geometry.add_yaw_goal_constraint(
            frame_R_current=parent_R_base,
            frame_R_goal=parent_R_goal,
            cruise_velocity=self.turning_cruise_velocity(sm.abs(yaw_error)),
            quadratic_weight=self.weight,
        )
        artifacts.error = sm.abs(yaw_error)
        return artifacts

    def turning_cruise_velocity(self, angle_to_go: sm.Scalar) -> sm.ScalarData:
        """
        :param angle_to_go: How far the base still has to turn, in radians.
        :return: The speed the base turns towards the goal orientation at.
        """
        return self.cruise_angular_velocity


@dataclass(eq=False, repr=False, kw_only=True)
class BaseOrientation(DrivenBaseRotation):
    """
    Turns the base to the orientation of :attr:`goal_pose`.
    """

    def parent_R_goal(
        self, *, parent_T_base: HomogeneousTransformationMatrix
    ) -> RotationMatrix:
        return self.parent_T_goal.rotation_matrix


@dataclass(eq=False, repr=False, kw_only=True)
class BaseHeading(DrivenBaseRotation):
    """
    Turns the front of the base towards the position of :attr:`goal_pose`, from wherever
    the base is.
    """

    def parent_R_goal(
        self, *, parent_T_base: HomogeneousTransformationMatrix
    ) -> RotationMatrix:
        parent_V_base_to_goal = self.parent_T_goal.position - self.parent_P_viewpoint(
            parent_T_base=parent_T_base
        )
        parent_V_base_to_goal.scale(1)
        return RotationMatrix.from_vectors(
            x=parent_V_base_to_goal,
            z=Vector3.Z(reference_frame=self.drive_parent),
            reference_frame=self.drive_parent,
        )

    def parent_P_viewpoint(
        self, *, parent_T_base: HomogeneousTransformationMatrix
    ) -> Point3:
        """
        :param parent_T_base: The pose of the base relative to the drive's parent.
        :return: The point relative to the drive's parent the base faces the position of
            :attr:`goal_pose` from.
        """
        return parent_T_base.position


@dataclass(eq=False, repr=False, kw_only=True)
class BaseTurnTowardsGoalPosition(BaseHeading):
    """
    Turns the front of the base towards the position of :attr:`goal_pose`, as seen from
    where the base stood when the task started, so that driving does not bring the base
    closer to facing it.
    """

    _parent_T_start_binding: ForwardKinematicsBinding = field(init=False, repr=False)
    """
    Fixes the pose of the base relative to the drive's parent when the task starts.
    """

    def set_up(self) -> None:
        """
        Bind the pose of the base, which the base faces the goal position from.
        """
        super().set_up()
        self._parent_T_start_binding = ForwardKinematicsBinding(
            name=PrefixedName("parent_T_start", str(self.name)),
            kinematic_chain=KinematicChain(
                root=self.drive_parent, tip=self.driven_body
            ),
            float_variable_data=self.context.float_variable_data,
        )
        self._parent_T_start_binding.bind(world=self.context.world)

    def on_start(self) -> None:
        self._parent_T_start_binding.bind(world=self.context.world)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Differs from :meth:`DrivenBaseRotation.build_artifacts` in first expressing the
        bound start pose relative to the drive's new parent, if the drive was moved
        under another parent.
        """
        if self._parent_T_start_binding.kinematic_chain.root is not self.drive_parent:
            self._parent_T_start_binding.change_root(
                root=self.drive_parent, world=self.context.world
            )
        return super().build_artifacts()

    def parent_P_viewpoint(
        self, *, parent_T_base: HomogeneousTransformationMatrix
    ) -> Point3:
        return self._parent_T_start_binding.root_T_tip.position


# %% walking up to a pose


@dataclass(eq=False, repr=False, kw_only=True)
class PacedBasePosition(BasePosition):
    """
    Walks the base to below the position of :attr:`goal_pose` at :attr:`pace`: in
    proportion to the distance still to go, so a base that follows its commands late
    still slows down before it arrives.
    """

    pace: WalkingPace
    """
    How fast the base walks up to the goal.
    """

    def cruise_velocities(
        self, parent_V_base_to_goal: Vector3
    ) -> tuple[sm.ScalarData, sm.ScalarData]:
        distance = parent_V_base_to_goal.norm()
        speed = sm.limit(
            self.pace.slowing_rate * distance,
            self.pace.slowest_speed,
            self.pace.fastest_speed,
        )
        share = speed / sm.max(distance, self.pace.arrival_tolerance)
        return (
            share * sm.abs(parent_V_base_to_goal.x),
            share * sm.abs(parent_V_base_to_goal.y),
        )


@dataclass(eq=False, repr=False, kw_only=True)
class PacedTurn(DrivenBaseRotation, ABC):
    """
    Turns the base at :attr:`pace`: in proportion to the angle still to go.
    """

    pace: WalkingPace
    """
    How fast the base turns.
    """

    def turning_cruise_velocity(self, angle_to_go: sm.Scalar) -> sm.ScalarData:
        return sm.limit(
            self.pace.slowing_rate * angle_to_go,
            self.pace.slowest_turn,
            self.pace.fastest_turn,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class PacedBaseOrientation(PacedTurn, BaseOrientation):
    """
    Turns the base to the orientation of :attr:`goal_pose` at its pace.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class PacedBaseTurnTowardsGoalPosition(PacedTurn, BaseTurnTowardsGoalPosition):
    """
    Turns the front of the base towards the position of :attr:`goal_pose`, as seen from
    where it stood when the task started, at its pace.
    """


# %% putting the base of a robot at a pose


class MobileBasePoseGoal(ABC):
    """
    A node putting the base of a robot at a pose, built for any robot the same way, so
    which kind of node does it can be chosen without knowing what each one needs.
    """

    @classmethod
    @abstractmethod
    def moving_base_of(
        cls, *, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        :param robot: The robot whose base is put at `goal_pose`.
        :param goal_pose: Where the root of `robot` should end up.
        :param threshold: How far the base may be from `goal_pose`, in meters and in
            radians, to count as arrived.
        :return: A new node putting the base of `robot` at `goal_pose`.
        """


# %% driving the base along the floor


@dataclass(eq=False, repr=False, kw_only=True)
class DrivenBaseGoal(
    SucceedsOnObservingTrue,
    FailsOnObservingFalse,
    CompositeNode,
    MobileBasePoseGoal,
    BindsGoalReferenceFrame,
    HasGoalPoseParameter,
    HasWeightParameter,
    HasThresholdParameter,
    ABC,
):
    """
    Drives a base along the floor and turns it about the vertical until it stands below
    :attr:`goal_pose`, facing its way. Its threshold holds in meters and in radians.

    The base is moved relative to the body it is attached to, as it is whenever the
    tasks are built, so the goal follows a base moved under another parent while the
    statechart runs. Where the goal lies is read from its reference frame as
    :attr:`goal_binding_policy` says.
    """

    mobile_base: MobileBase
    """
    The base moved, by the drive attaching its robot to its parent.
    """

    root_T_goal_reference_frame: HomogeneousTransformationMatrix | None = (
        expanded_child_field(default=None)
    )
    """
    Differs from :attr:`BindsGoalReferenceFrame.root_T_goal_reference_frame` in being
    created by :meth:`expand` and sent along with this goal once it expanded, since the
    tasks of :attr:`motion` are expressed over it.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_pose.reference_frame

    @property
    def driven_body(self) -> KinematicStructureEntity:
        """
        :return: The body the drive of :attr:`mobile_base` carries.
        """
        return self.mobile_base.drive.child

    @property
    @abstractmethod
    def motion(self) -> StatechartNode:
        """
        :return: The node running the tasks, once :meth:`_create_motion` created it.
        """

    @abstractmethod
    def _create_motion(self, *, task_parameters: dict[str, Any]) -> None:
        """
        Creates the node running the tasks that move :attr:`driven_body`.

        :param task_parameters: The parameters every task is created with.
        """

    @classmethod
    def moving_base_of(
        cls, *, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        Differs from :meth:`MobileBasePoseGoal.moving_base_of` in requiring `robot` to
        have a mobile base.

        :raises RobotWithoutMobileBaseError: If `robot` has no mobile base.
        """
        if not isinstance(robot, HasMobileBase) or robot.mobile_base is None:
            raise RobotWithoutMobileBaseError(robot=robot)
        return cls(
            mobile_base=robot.mobile_base, goal_pose=goal_pose, threshold=threshold
        )

    @property
    def _drive_parent(self) -> KinematicStructureEntity:
        """
        :return: The body :attr:`driven_body` is attached to now.
        """
        return self.driven_body.parent_kinematic_structure_entity

    @property
    def goal_binding_root(self) -> KinematicStructureEntity:
        return self._drive_parent

    def _root_T_goal_reference_frame_to_bind(self) -> HomogeneousTransformationMatrix:
        """
        Differs from
        :meth:`BindsGoalReferenceFrame._root_T_goal_reference_frame_to_bind` in binding
        the matrix :meth:`expand` created, which the tasks are expressed over.
        """
        return self.root_T_goal_reference_frame

    def expand(self) -> None:
        """
        Add the node running the tasks.
        """
        self.root_T_goal_reference_frame = (
            HomogeneousTransformationMatrix.create_with_variables(
                str(self._goal_reference_frame_name)
            )
        )
        self._create_motion(
            task_parameters=dict(
                driven_body=self.driven_body,
                goal_pose=self.goal_pose,
                parent_T_goal_reference_frame=self.root_T_goal_reference_frame,
                weight=self.weight,
                threshold=self.threshold,
            )
        )
        self._add_child_to_statechart(node=self.motion)
        self._succeed_at_its_goal(child=self.motion)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Report the outcome of :attr:`motion`, and express
        :attr:`root_T_goal_reference_frame` relative to the new parent of the base if it
        was moved under another parent, since the tasks are rebuilt against it.

        Where the goal's reference frame was bound is kept.
        """
        if (
            self._goal_reference_frame_binding.kinematic_chain.root
            is not self._drive_parent
        ):
            self._goal_reference_frame_binding.change_root(
                root=self._drive_parent, world=self.context.world
            )
        return NodeArtifacts(
            observation=trinary_if_cases(
                cases=[
                    (self.motion.is_succeeded, Scalar.const_true()),
                    (self.motion.is_failed_or_interrupted, Scalar.const_false()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(eq=False, repr=False, kw_only=True)
class HolonomicBaseGoal(DrivenBaseGoal):
    """
    Moves the base to the pose in whatever direction it lies, sideways included, while
    turning it.
    """

    parallel: Parallel = expanded_child_field()
    """
    The parallel running the position and the orientation task.
    """

    @property
    def motion(self) -> StatechartNode:
        return self.parallel

    def _create_motion(self, *, task_parameters: dict[str, Any]) -> None:
        self.parallel = Parallel(
            name=f"{self.name}/parallel",
            nodes=[
                BasePosition(name=f"{self.name}/position", **task_parameters),
                BaseOrientation(name=f"{self.name}/orientation", **task_parameters),
            ],
        )


@dataclass(eq=False, repr=False, kw_only=True)
class WalkingBaseGoal(DrivenBaseGoal):
    """
    Walks a base on legs to the pose at the pace its walking controller walks up to a
    pose, see :attr:`~semantic_digital_twin.robots.walking.WalkingController.pace`,
    running these steps in one :class:`~cramph.composites.Sequence`, since legs walk
    forward best:

    1. Turn towards the goal position, without walking
    2. Walk to the goal position, facing it as seen from the start, so a base that
       overshoots does not turn round
    3. Turn to the goal orientation, keeping the position
    """

    pace: WalkingPace
    """
    How fast the base walks up to the pose.
    """

    sequence: Sequence = expanded_child_field()
    """
    The sequence running the three steps.
    """

    @property
    def motion(self) -> StatechartNode:
        return self.sequence

    @classmethod
    def moving_base_of(
        cls, *, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        Differs from :meth:`DrivenBaseGoal.moving_base_of` in walking at the pace of the
        robot's walking controller.

        :raises RobotDoesNotWalkError: If the base of `robot` does not walk.
        """
        if not isinstance(robot, HasMobileBase) or robot.mobile_base is None:
            raise RobotWithoutMobileBaseError(robot=robot)
        controller_type = robot.mobile_base.walking_controller_type
        if controller_type is None:
            raise RobotDoesNotWalkError(robot=robot)
        return cls(
            mobile_base=robot.mobile_base,
            goal_pose=goal_pose,
            threshold=threshold,
            pace=controller_type.pace,
        )

    def _create_motion(self, *, task_parameters: dict[str, Any]) -> None:
        paced = dict(task_parameters, pace=self.pace)
        steps = [
            PacedBaseTurnTowardsGoalPosition(name=f"{self.name}/step1", **paced),
            Parallel(
                name=f"{self.name}/step2",
                nodes=[
                    PacedBasePosition(name=f"{self.name}/step2/position", **paced),
                    PacedBaseTurnTowardsGoalPosition(
                        name=f"{self.name}/step2/heading", **paced
                    ),
                ],
            ),
            Parallel(
                name=f"{self.name}/step3",
                nodes=[
                    PacedBasePosition(name=f"{self.name}/step3/position", **paced),
                    PacedBaseOrientation(
                        name=f"{self.name}/step3/orientation", **paced
                    ),
                ],
            ),
        ]
        self.sequence = Sequence(name=f"{self.name}/sequence", nodes=steps)


@dataclass(eq=False, repr=False, kw_only=True)
class DifferentialDriveBaseGoal(DrivenBaseGoal):
    """
    Moves the robot to a goal pose using a differential drive, running these steps in
    one :class:`~cramph.composites.Sequence`:

    1. Turn towards the goal position, without driving
    2. Drive to goal position, steering towards it
    3. Orient to goal orientation
    """

    mobile_base: MobileBase[DifferentialDrive]
    """
    The base moved, by the differential drive attaching its robot to its parent.
    """

    sequence: Sequence = expanded_child_field()
    """
    The sequence running the three steps.
    """

    @property
    def motion(self) -> StatechartNode:
        return self.sequence

    def _create_motion(self, *, task_parameters: dict[str, Any]) -> None:
        steps = [
            BaseTurnTowardsGoalPosition(name=f"{self.name}/step1", **task_parameters),
            Parallel(
                name=f"{self.name}/step2",
                nodes=[
                    BasePosition(name=f"{self.name}/step2/position", **task_parameters),
                    BaseHeading(name=f"{self.name}/step2/heading", **task_parameters),
                ],
            ),
            Parallel(
                name=f"{self.name}/step3",
                nodes=[
                    BasePosition(name=f"{self.name}/step3/position", **task_parameters),
                    BaseOrientation(
                        name=f"{self.name}/step3/orientation", **task_parameters
                    ),
                ],
            ),
        ]
        self.sequence = Sequence(name=f"{self.name}/sequence", nodes=steps)
