"""
Monitors of the safety constraints a robot's motions are held to.

A workspace box its tool stays inside, and how fast its joints and its tool may move. A
motion breaking one is cancelled with a
:class:`~giskardpy.motion_statechart.exceptions.SafetyConstraintViolated`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from cramph.data_types import ObservationStateValue
from cramph.node import CancelStatechart, EndedByOwner, NodeArtifacts, StatechartNode
from giskardpy.motion_statechart.exceptions import SafetyConstraintViolated
from giskardpy.motion_statechart.graph_node import (
    MotionNodeArtifacts,
    MotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.mixins import HasRootLinkParameter
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityScale
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from krrood.patterns.called_positionally import called_positionally
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    ScrewAxisConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

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
        """
        Observes whether the tool lies outside the box on any axis, see
        :meth:`MotionStatechartNode.build_artifacts`.
        """
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
        below, above = minimum - box_P_tool, box_P_tool - maximum
        axis = int(np.argmax(np.maximum(below, above)))
        if below[axis] > above[axis]:
            side, bound, distance = "lower", minimum[axis], below[axis]
        else:
            side, bound, distance = "upper", maximum[axis], above[axis]
        name = "xyz"[axis]
        return (
            f"{self.tool.name} is {distance * 1000:.1f} mm past the box's {side} "
            f"{name} bound: {name}={box_P_tool[axis]:.3f} m against {bound:.3f} m in "
            f"{self.minimum.reference_frame.name}"
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
    :attr:`velocity_limit_tolerance`.

    A joint without a velocity limit is not watched.
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
        """
        Observes whether any of :attr:`limited_connections` exceeds its limit, see
        :meth:`MotionStatechartNode.build_artifacts`.
        """
        return NodeArtifacts(
            observation=sm.logic_or(
                Scalar.const_false(),
                *[
                    self.exceeds_its_limit(connection=connection)
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
    def exceeds_its_limit(cls, *, connection: ActiveConnection1DOF) -> sm.Scalar:
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
        unit = (
            "m/s"
            if isinstance(fastest, ScrewAxisConnection)
            and fastest.behaves_like_a_prismatic_connection
            else "rad/s"
        )
        return (
            f"{fastest.name} moved at {self.joint_speed(connection=fastest):.3f} {unit}, faster "
            f"than its limit {fastest.raw_dof.limits.upper.velocity:.3f} {unit}"
        )

    @called_positionally
    def share_of_limit(self, connection: ActiveConnection1DOF) -> float:
        """
        :param connection: One of :attr:`limited_connections`.
        :return: How fast it moves now, as a share of its upper velocity limit.
        """
        return (
            self.joint_speed(connection=connection)
            / connection.raw_dof.limits.upper.velocity
        )

    def joint_speed(self, *, connection: ActiveConnection1DOF) -> float:
        """
        :param connection: One of :attr:`connections`.
        :return: How fast it moves now.
        """
        return abs(float(self.context.world.state[connection.raw_dof.id].velocity))


@dataclass(eq=False, repr=False, kw_only=True)
class ToolVelocityExceeded(SafetyMonitor, HasRootLinkParameter):
    """
    Observes ``True`` while the origin of :attr:`tool` moves faster than
    :attr:`speed_limit` relative to :attr:`root_link`, as the joints' velocities move
    it.
    """

    tool: Body
    """
    The body whose velocity is watched.
    """

    speed_limit: float
    """
    The speed the tool may not exceed, in metres per second.
    """

    tool_speed: sm.Scalar = field(init=False, repr=False)
    """
    How fast the tool moves, from the joints' positions and velocities; set once the
    monitor is built.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        Observes whether the tool's speed, derived from the joints' velocities, exceeds
        :attr:`speed_limit`, see :meth:`MotionStatechartNode.build_artifacts`.
        """
        root_P_tool = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tool
        ).position
        degrees_of_freedom = self.context.world.active_degrees_of_freedom
        tool_velocity = sm.Vector(
            [root_P_tool.x, root_P_tool.y, root_P_tool.z]
        ).total_derivative(
            variables=[dof.variables.position for dof in degrees_of_freedom],
            variables_dot=[dof.variables.velocity for dof in degrees_of_freedom],
        )
        self.tool_speed = tool_velocity.norm()
        return NodeArtifacts(observation=self.tool_speed > self.speed_limit)

    def describe_violation(self) -> str:
        speed = float(np.asarray(self.tool_speed.evaluate()).reshape(-1)[0])
        return (
            f"{self.tool.name} moved at {speed:.3f} m/s, faster than its limit "
            f"{self.speed_limit:.3f} m/s"
        )


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

    violation: SafetyConstraintViolated | None = field(
        init=False, default=None, repr=False
    )
    """
    The violation this node cancelled the motion with, kept as it was when it started,
    since the world moves on after the motion stopped; None while it did not start.
    """

    def build(self) -> NodeArtifacts:
        """
        Starts this node once any of :attr:`monitors` observes a violation, see
        :meth:`CancelStatechart.build`.
        """
        self.start_condition = sm.logic_or(
            Scalar.const_false(), *[monitor.observes_true for monitor in self.monitors]
        )
        return NodeArtifacts()

    def on_start(self) -> None:
        """
        Also keeps the violation as :attr:`violation`, see
        :meth:`CancelStatechart.on_start`.
        """
        super().on_start()
        self.violation = self.pending_exception

    def create_exception(self) -> SafetyConstraintViolated:
        """
        :return: The violation of the first of :attr:`monitors` observing its
            constraint violated when this node starts.
        """
        violated = next(
            monitor
            for monitor in self.monitors
            if monitor.observation_state == ObservationStateValue.TRUE
        )
        return SafetyConstraintViolated(
            constraint=violated.constraint, violation=violated.describe_violation()
        )


# %% holding motions inside the constraints


@dataclass(eq=False, repr=False, kw_only=True)
class ToolSpeedLimit(Task, HasRootLinkParameter):
    """
    Keeps a tool slower than :attr:`speed_limit` relative to :attr:`root_link`,
    weighted above collision avoidance: each axis of its velocity is held to
    :attr:`speed_limit` divided by the square root of three. Not a hard constraint,
    which a tool already moving faster could not meet in one cycle.
    """

    tool: Body
    """
    The body held to the limit.
    """

    speed_limit: float
    """
    The speed the tool stays below, in metres per second.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Holds each axis of the tool's velocity to its share of :attr:`speed_limit`, see
        :meth:`Task.build_artifacts`.
        """
        artifacts = MotionNodeArtifacts()
        root_P_tool = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tool
        ).position
        axis_velocity = self.speed_limit / np.sqrt(3.0)
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


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class MaximumToolVelocity(SafetyConstraint):
    """
    A speed a tool may not exceed.
    """

    reference_frame: KinematicStructureEntity
    """
    The entity the tool's velocity is measured against.
    """

    tool: Body
    """
    The body held to the limit.
    """

    speed_limit: float
    """
    The speed the tool may not exceed, in metres per second.
    """

    held_fraction: float = 0.8
    """
    The share of :attr:`speed_limit` every motion holds the tool's speed to.
    """

    def create_motion_limits(self) -> list[MotionStatechartNode]:
        """
        :return: A task holding the tool's speed to :attr:`held_fraction` of
            :attr:`speed_limit`.
        """
        return [
            ToolSpeedLimit(
                name=f"{self.constraint} held",
                root_link=self.reference_frame,
                tool=self.tool,
                speed_limit=self.speed_limit * self.held_fraction,
            )
        ]

    def create_monitor(self) -> ToolVelocityExceeded:
        return ToolVelocityExceeded(
            name=self.constraint,
            constraint=self.constraint,
            root_link=self.reference_frame,
            tool=self.tool,
            speed_limit=self.speed_limit,
        )


@dataclass(kw_only=True)
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
