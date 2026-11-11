"""
Monitors of the safety constraints a robot's motions are held to: a workspace box its
tool stays inside, and how fast its joints and its tool may move.

A motion breaking one
is cancelled with a :class:`~giskardpy.motion_statechart.exceptions.SafetyConstraintViolated`
naming it, which nothing recovers from by trying again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from cramph.data_types import ObservationStateValues
from cramph.node import CancelStatechart, EndedByOwner, NodeArtifacts, StatechartNode
from giskardpy.motion_statechart.exceptions import SafetyConstraintViolated
from giskardpy.motion_statechart.graph_node import (
    MotionNodeArtifacts,
    MotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityScale
from krrood.ormatic.utils import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

# %% monitors


@dataclass(eq=False, repr=False, kw_only=True)
class SafetyMonitor(EndedByOwner, MotionStatechartNode, ABC):
    """
    Observes ``True`` while the safety constraint it watches is violated.
    """

    constraint: str
    """
    The watched constraint, as a person reads it.
    """

    @abstractmethod
    def describe_violation(self) -> str:
        """
        :return: What violates the constraint in the world as it is now, as a person
            reads it.
        """


@dataclass(eq=False, repr=False, kw_only=True)
class ToolOutsideWorkspace(SafetyMonitor):
    """
    Observes ``True`` while the origin of :attr:`tool` lies outside the axis-aligned box
    spanned by :attr:`minimum` and :attr:`maximum` in the frame both are expressed in.
    """

    tool: Body
    """
    The body that is to stay inside the box.
    """

    minimum: Point3
    """
    The box's corner of the smallest coordinates.
    """

    maximum: Point3
    """
    The box's corner of the largest coordinates, in the frame of :attr:`minimum`.
    """

    def build_artifacts(self) -> NodeArtifacts:
        box_P_tool = self.context.world.compose_forward_kinematics_expression(
            self.minimum.reference_frame, self.tool
        ).position
        return NodeArtifacts(
            observation=sm.logic_or(
                *[
                    sm.logic_or(
                        box_P_tool[axis] < self.minimum[axis],
                        box_P_tool[axis] > self.maximum[axis],
                    )
                    for axis in range(3)
                ]
            )
        )

    def describe_violation(self) -> str:
        box_P_tool = self.tool_position()
        minimum = self.minimum.to_np()[:3]
        maximum = self.maximum.to_np()[:3]
        distance = np.linalg.norm(
            np.maximum(0.0, np.maximum(minimum - box_P_tool, box_P_tool - maximum))
        )
        return (
            f"{self.tool.name} at {np.round(box_P_tool, 3).tolist()} m is "
            f"{distance:.3f} m outside the box from {minimum.tolist()} to "
            f"{maximum.tolist()} m"
        )

    def tool_position(self) -> np.ndarray:
        """
        :return: Where the origin of :attr:`tool` lies now, in the frame of the box.
        """
        return self.context.world.compute_forward_kinematics_np(
            self.minimum.reference_frame, self.tool
        )[:3, 3]


@dataclass(eq=False, repr=False, kw_only=True)
class JointVelocityExceeded(SafetyMonitor):
    """
    Observes ``True`` while any of :attr:`connections` moves faster than its own
    velocity limit, as the world describes it when the monitor is built, by more than
    :attr:`velocity_limit_tolerance`. A joint without a velocity limit is not watched.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints whose velocities are watched.
    """

    @classproperty
    def velocity_limit_tolerance(cls) -> float:
        """
        :return: How far, in radians or metres per second, a joint may move faster than
            its velocity limit before a violation is observed. The controller holds the
            velocity it plans to the limit, but the velocity it commands is integrated
            from the jerk the QP solver returns, which meets that integration only up to
            the solver's tolerance of 1e-6; a joint moving at its limit can pass it by
            that much.
        """
        return 1e-5

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(
            observation=sm.logic_or(
                Scalar.const_false(),
                *[
                    self.exceeds_its_limit(connection)
                    for connection in self.limited_connections
                ],
            )
        )

    @property
    def limited_connections(self) -> list[ActiveConnection1DOF]:
        """
        :return: Those of :attr:`connections` the world sets a velocity limit for.
        """
        return [
            connection
            for connection in self.connections
            if connection.raw_dof.limits.upper.velocity is not None
        ]

    @classmethod
    def exceeds_its_limit(cls, connection: ActiveConnection1DOF) -> sm.Scalar:
        """
        :param connection: A joint the world sets a velocity limit for.
        :return: Whether it moves faster than that limit, in either direction.
        """
        dof = connection.raw_dof
        velocity = dof.variables.velocity
        return sm.logic_or(
            velocity > dof.limits.upper.velocity + cls.velocity_limit_tolerance,
            velocity < dof.limits.lower.velocity - cls.velocity_limit_tolerance,
        )

    def describe_violation(self) -> str:
        fastest = max(self.limited_connections, key=self.share_of_limit)
        return (
            f"{fastest.name} moved at {self.joint_speed(fastest):.3f}, faster than its "
            f"limit of {fastest.raw_dof.limits.upper.velocity:.3f}"
        )

    def share_of_limit(self, connection: ActiveConnection1DOF) -> float:
        """
        :param connection: One of :attr:`limited_connections`.
        :return: How fast it moves now, as a share of its upper velocity limit.
        """
        return self.joint_speed(connection) / connection.raw_dof.limits.upper.velocity

    def joint_speed(self, connection: ActiveConnection1DOF) -> float:
        """
        :param connection: One of :attr:`connections`.
        :return: How fast it moves now.
        """
        return abs(float(self.context.world.state[connection.raw_dof.id].velocity))


@dataclass(eq=False, repr=False, kw_only=True)
class ToolVelocityExceeded(SafetyMonitor):
    """
    Observes ``True`` while the origin of :attr:`tool` moves faster than
    :attr:`maximum_velocity` relative to :attr:`root_link`, as the joints' velocities
    move it.
    """

    root_link: Body
    """
    The body the tool's velocity is measured against.
    """

    tool: Body
    """
    The body whose velocity is watched.
    """

    maximum_velocity: float
    """
    The speed the tool may not exceed, in metres per second.
    """

    def build_artifacts(self) -> NodeArtifacts:
        root_P_tool = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tool
        ).position
        degrees_of_freedom = self.context.world.active_degrees_of_freedom
        tool_velocity = sm.Vector(
            [root_P_tool.x, root_P_tool.y, root_P_tool.z]
        ).total_derivative(
            [dof.variables.position for dof in degrees_of_freedom],
            [dof.variables.velocity for dof in degrees_of_freedom],
        )
        return NodeArtifacts(observation=tool_velocity.norm() > self.maximum_velocity)

    def describe_violation(self) -> str:
        return f"{self.tool.name} moved faster than {self.maximum_velocity} m/s"


# %% cancelling the motion


@dataclass(eq=False, repr=False, kw_only=True)
class CancelOnSafetyViolation(CancelStatechart):
    """
    Cancels the motion once one of :attr:`monitors` observes its constraint violated,
    raising a :class:`~giskardpy.motion_statechart.exceptions.SafetyConstraintViolated`
    naming it.
    """

    monitors: list[SafetyMonitor]
    """
    The monitors of the constraints the motion is held to.
    """

    exception: Exception = field(init=False, default=Exception)
    """
    Set to ``init=False``, because this class creates its own exception once it knows
    which constraint is violated.
    """

    def build(self) -> NodeArtifacts:
        self.start_condition = sm.logic_or(
            Scalar.const_false(), *[monitor.observes_true for monitor in self.monitors]
        )
        return NodeArtifacts()

    def create_exception(self) -> SafetyConstraintViolated:
        """
        :return: The violation of the first of :attr:`monitors` observing its
            constraint violated when this node starts.
        """
        violated = next(
            monitor
            for monitor in self.monitors
            if monitor.observation_state == ObservationStateValues.TRUE
        )
        return SafetyConstraintViolated(
            constraint=violated.constraint, violation=violated.describe_violation()
        )


# %% holding motions inside the constraints


@dataclass(eq=False, repr=False, kw_only=True)
class ToolSpeedLimit(Task):
    """
    Keeps a tool slower than :attr:`maximum_velocity` relative to :attr:`root_link`,
    weighted above collision avoidance: each axis of its velocity is held to
    :attr:`maximum_velocity` divided by the square root of three. Not a hard
    constraint, which a tool already moving faster could not meet in one cycle.
    """

    root_link: Body
    """
    The body the tool's velocity is measured against.
    """

    tool: Body
    """
    The body held to the limit.
    """

    maximum_velocity: float
    """
    The speed the tool stays below, in metres per second.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        root_P_tool = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tool
        ).position
        axis_velocity = self.maximum_velocity / np.sqrt(3.0)
        for axis, coordinate in zip(
            "xyz", (root_P_tool.x, root_P_tool.y, root_P_tool.z)
        ):
            artifacts.constraints.add_velocity_constraint(
                lower_velocity_limit=-axis_velocity,
                upper_velocity_limit=axis_velocity,
                quadratic_weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                task_expression=coordinate,
                velocity_scale=VelocityScale.TRANSLATION,
                name=f"{self.name}/{axis}",
            )
        artifacts.observation = sm.Scalar.const_true()
        return artifacts


# %% the constraints a robot is held to


@dataclass(frozen=True)
class SafetyConstraint(ABC):
    """
    A constraint every motion of a robot is held to.
    """

    constraint: str
    """
    The constraint, as a person reads it.
    """

    @abstractmethod
    def create_monitor(self) -> SafetyMonitor:
        """
        :return: A new monitor of this constraint, for one statechart.
        """

    def create_motion_limits(self) -> list[MotionStatechartNode]:
        """
        :return: New tasks keeping every motion of one statechart inside this
            constraint, so that its monitor only observes a motion leaving it
            unexpectedly; none for a constraint only watched.
        """
        return []


@dataclass(frozen=True)
class WorkspaceBox(SafetyConstraint):
    """
    An axis-aligned box a tool stays inside.
    """

    tool: Body
    """
    The body that is to stay inside the box.
    """

    minimum: Point3
    """
    The box's corner of the smallest coordinates.
    """

    maximum: Point3
    """
    The box's corner of the largest coordinates, in the frame of :attr:`minimum`.
    """

    def create_monitor(self) -> ToolOutsideWorkspace:
        return ToolOutsideWorkspace(
            name=self.constraint,
            constraint=self.constraint,
            tool=self.tool,
            minimum=self.minimum,
            maximum=self.maximum,
        )


@dataclass(frozen=True)
class MaximumJointVelocity(SafetyConstraint):
    """
    Each joint of a group stays within its own velocity limit, as the world describes
    it: the limit the controller plans every motion within, so only a motion leaving it
    unexpectedly breaks this constraint.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints held to their limits.
    """

    def create_monitor(self) -> JointVelocityExceeded:
        return JointVelocityExceeded(
            name=self.constraint,
            constraint=self.constraint,
            connections=self.connections,
        )


@dataclass(frozen=True)
class MaximumToolVelocity(SafetyConstraint):
    """
    A speed a tool may not exceed.
    """

    root_link: Body
    """
    The body the tool's velocity is measured against.
    """

    tool: Body
    """
    The body held to the limit.
    """

    maximum_velocity: float
    """
    The speed the tool may not exceed, in metres per second.
    """

    held_fraction: float = 0.8
    """
    The share of :attr:`maximum_velocity` every motion holds the tool's speed to.
    """

    def create_motion_limits(self) -> list[MotionStatechartNode]:
        """
        :return: A task holding the tool's speed to :attr:`held_fraction` of
            :attr:`maximum_velocity`.
        """
        return [
            ToolSpeedLimit(
                name=f"{self.constraint} held",
                root_link=self.root_link,
                tool=self.tool,
                maximum_velocity=self.maximum_velocity * self.held_fraction,
            )
        ]

    def create_monitor(self) -> ToolVelocityExceeded:
        return ToolVelocityExceeded(
            name=self.constraint,
            constraint=self.constraint,
            root_link=self.root_link,
            tool=self.tool,
            maximum_velocity=self.maximum_velocity,
        )


@dataclass
class SafetyEnvelope:
    """
    The safety constraints every motion of a robot is held to.
    """

    constraints: list[SafetyConstraint] = field(default_factory=list)
    """
    The constraints.
    """

    def create_nodes(self) -> list[StatechartNode]:
        """
        :return: New nodes for one statechart: the tasks keeping motions inside the
            constraints, a monitor of every constraint, and the node cancelling the
            motion once one of them observes a violation; none without constraints.
        """
        if not self.constraints:
            return []
        limits = [
            limit
            for constraint in self.constraints
            for limit in constraint.create_motion_limits()
        ]
        monitors = [constraint.create_monitor() for constraint in self.constraints]
        return [*limits, *monitors, CancelOnSafetyViolation(monitors=monitors)]
