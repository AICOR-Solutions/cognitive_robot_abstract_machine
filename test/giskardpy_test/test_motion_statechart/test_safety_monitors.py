"""
The safety monitors a robot's motions are held to, and the cancelling of a motion that
breaks one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import SafetyConstraintViolated
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.safety_monitors import (
    CancelOnSafetyViolation,
    JointVelocityExceeded,
    MaximumJointVelocity,
    SafetyEnvelope,
    SafetyMonitor,
    MaximumToolVelocity,
    WorkspaceBox,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

# %% helpers

BOUNDARY_TOLERANCE = 0.02
"""
How far, in metres, a tool cancelled at a box's boundary may have moved beyond it: the
control cycles in which the violation is observed and the motion cancelled.
"""


def _tool(world: World) -> Body:
    """
    :return: The tool frame of the right arm of the PR2.
    """
    return world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")


def _torso(world: World) -> ActiveConnection1DOF:
    """
    :return: The torso joint of the PR2, which lifts the tool straight up.
    """
    return world.get_connection_by_name("torso_lift_joint")


def _arm_joints(world: World) -> list[ActiveConnection1DOF]:
    """
    :return: The joints of the right arm of the PR2 from the shoulder to the forearm.
    """
    return [
        world.get_connection_by_name(name)
        for name in (
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
        )
    ]


def _box_around_tool(world: World, half_size: float) -> WorkspaceBox:
    """
    :return: A box centred on where the tool is now, in the world's root frame.
    """
    position = world.compute_forward_kinematics_np(world.root, _tool(world))[:3, 3]
    return WorkspaceBox(
        constraint="right arm workspace",
        tool=_tool(world),
        minimum=Point3(*(position - half_size), reference_frame=world.root),
        maximum=Point3(*(position + half_size), reference_frame=world.root),
    )


def _observe(world: World, monitor: SafetyMonitor) -> ObservationStateValues:
    """
    :return: What `monitor` observes of the world as it is, after one tick.
    """
    executor = StatechartExecutor(context=StatechartContext(world=world))
    statechart = Statechart(context=executor.context)
    statechart.add_node(monitor)
    executor.compile(statechart=statechart)
    executor.tick()
    return monitor.observation_state


# %% observing


def test_a_tool_inside_its_box_is_not_a_violation(pr2_world_state_reset):
    box = _box_around_tool(pr2_world_state_reset, 0.1)
    observed = _observe(pr2_world_state_reset, box.create_monitor())
    assert observed == ObservationStateValues.FALSE


def test_a_tool_outside_its_box_is_a_violation(pr2_world_state_reset):
    world = pr2_world_state_reset
    inside = _box_around_tool(world, 0.1)
    shifted = WorkspaceBox(
        constraint=inside.constraint,
        tool=inside.tool,
        minimum=Point3(*(inside.minimum.to_np()[:3] + 0.5), reference_frame=world.root),
        maximum=Point3(*(inside.maximum.to_np()[:3] + 0.5), reference_frame=world.root),
    )
    observed = _observe(world, shifted.create_monitor())
    assert observed == ObservationStateValues.TRUE


@pytest.mark.parametrize(
    "share_of_limit, expected",
    [
        (0.5, ObservationStateValues.FALSE),
        (1.0, ObservationStateValues.FALSE),
        (-1.0, ObservationStateValues.FALSE),
        (1.01, ObservationStateValues.TRUE),
        (-1.01, ObservationStateValues.TRUE),
    ],
)
def test_a_joint_faster_than_its_own_velocity_limit_is_a_violation(
    pr2_world_state_reset, share_of_limit, expected
):
    world = pr2_world_state_reset
    torso = _torso(world)
    world.state[torso.dof.id].velocity = (
        share_of_limit * torso.dof.limits.upper.velocity
    )
    limit = MaximumJointVelocity(constraint="torso velocity", connections=[torso])
    assert _observe(world, limit.create_monitor()) == expected


def test_a_joint_passing_its_limit_within_the_tolerance_is_no_violation(
    pr2_world_state_reset,
):
    world = pr2_world_state_reset
    torso = _torso(world)
    world.state[torso.dof.id].velocity = (
        torso.dof.limits.upper.velocity
        + JointVelocityExceeded.velocity_limit_tolerance / 2
    )
    limit = MaximumJointVelocity(constraint="torso velocity", connections=[torso])
    assert _observe(world, limit.create_monitor()) == ObservationStateValues.FALSE


def test_each_joint_is_held_to_its_own_velocity_limit(pr2_world_state_reset):
    """
    The shoulder pan joint's limit is far below the elbow's: a velocity the elbow may
    move at breaks the shoulder's limit.
    """
    world = pr2_world_state_reset
    arm = _arm_joints(world)
    shoulder_joint, elbow_joint = arm[0], arm[3]
    velocity = elbow_joint.dof.limits.upper.velocity * 0.9
    assert velocity > shoulder_joint.dof.limits.upper.velocity
    world.state[shoulder_joint.dof.id].velocity = velocity
    world.state[elbow_joint.dof.id].velocity = velocity
    limit = MaximumJointVelocity(
        constraint="arm velocity", connections=[shoulder_joint, elbow_joint]
    )
    monitor = limit.create_monitor()

    assert _observe(world, monitor) == ObservationStateValues.TRUE
    assert monitor.describe_violation().startswith(str(shoulder_joint.name))


def test_a_joint_is_held_to_its_limit_as_the_robot_tightened_it(
    pr2_world_state_reset,
):
    world = pr2_world_state_reset
    torso = _torso(world)
    description_limit = torso.dof.limits.upper.velocity
    torso.raw_dof.limits.upper.velocity = description_limit / 2
    torso.raw_dof.limits.lower.velocity = -description_limit / 2
    world.state[torso.dof.id].velocity = description_limit * 0.75
    limit = MaximumJointVelocity(constraint="torso velocity", connections=[torso])
    assert _observe(world, limit.create_monitor()) == ObservationStateValues.TRUE


@pytest.mark.parametrize(
    "velocity, expected",
    [(0.05, ObservationStateValues.FALSE), (0.2, ObservationStateValues.TRUE)],
)
def test_a_tool_faster_than_its_limit_is_a_violation(
    pr2_world_state_reset, velocity, expected
):
    world = pr2_world_state_reset
    world.state[_torso(world).dof.id].velocity = velocity
    limit = MaximumToolVelocity(
        constraint="right tool velocity",
        root_link=world.root,
        tool=_tool(world),
        maximum_velocity=0.1,
    )
    assert _observe(world, limit.create_monitor()) == expected


# %% cancelling


def test_an_envelope_without_constraints_adds_nothing():
    assert SafetyEnvelope().create_nodes() == []


def test_an_envelope_watches_each_constraint_and_cancels_on_any(pr2_world_state_reset):
    box = _box_around_tool(pr2_world_state_reset, 0.1)
    nodes = SafetyEnvelope(constraints=[box]).create_nodes()
    *monitors, cancel = nodes
    assert [monitor.constraint for monitor in monitors] == [box.constraint]
    assert isinstance(cancel, CancelOnSafetyViolation)
    assert cancel.monitors == monitors


def test_a_motion_leaving_the_box_is_cancelled_at_its_boundary(pr2_world_state_reset):
    world = pr2_world_state_reset
    box = _box_around_tool(world, 0.05)
    start = world.compute_forward_kinematics_np(world.root, box.tool)[:3, 3]
    goal = Point3(*(start + [0.3, 0.0, 0.0]), reference_frame=world.root)
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    reach = CartesianPosition(root_link=world.root, tip_link=box.tool, goal_point=goal)
    statechart.add_node(reach)
    statechart.add_node(EndMotion.when_true(reach))
    statechart.add_nodes(SafetyEnvelope(constraints=[box]).create_nodes())
    executor.compile(statechart=statechart)

    with pytest.raises(SafetyConstraintViolated) as violated:
        executor.tick_until_end()

    stopped = world.compute_forward_kinematics_np(world.root, box.tool)[:3, 3]
    overshoot = stopped[0] - box.maximum.to_np()[0]
    assert violated.value.constraint == box.constraint
    assert 0.0 < overshoot < BOUNDARY_TOLERANCE
    assert np.allclose(stopped[1:], start[1:], atol=0.01)


def test_a_motion_is_held_below_its_tools_velocity_limit(pr2_world_state_reset):
    world = pr2_world_state_reset
    tool = _tool(world)
    start = world.compute_forward_kinematics_np(world.root, tool)[:3, 3]
    goal = Point3(*(start + [0.3, 0.0, 0.0]), reference_frame=world.root)
    limit = MaximumToolVelocity(
        constraint="right tool velocity",
        root_link=world.root,
        tool=tool,
        maximum_velocity=0.05,
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    reach = CartesianPosition(root_link=world.root, tip_link=tool, goal_point=goal)
    statechart.add_node(reach)
    statechart.add_node(EndMotion.when_true(reach))
    statechart.add_nodes(SafetyEnvelope(constraints=[limit]).create_nodes())
    executor.compile(statechart=statechart)

    executor.tick_until_end()

    reached = world.compute_forward_kinematics_np(world.root, tool)[:3, 3]
    assert np.allclose(reached, goal.to_np()[:3], atol=0.01)


def test_a_motion_at_its_joints_velocity_limits_is_not_cancelled(
    pr2_world_state_reset,
):
    """
    Every joint of a far joint goal moves at its velocity limit for most of the motion,
    as fast as the controller lets it.
    """
    world = pr2_world_state_reset
    joints = [*_arm_joints(world), _torso(world)]
    goal = {}
    for joint in joints:
        position = float(world.state[joint.dof.id].position)
        lower, upper = joint.dof.limits.lower.position, joint.dof.limits.upper.position
        if upper is None:
            goal[joint] = position + 2.0
        elif upper - position > position - lower:
            goal[joint] = position + 0.9 * (upper - position)
        else:
            goal[joint] = position + 0.9 * (lower - position)
    limit = MaximumJointVelocity(constraint="arm velocity", connections=joints)
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    reach = JointPositionList(
        goal_state=JointState.from_mapping(goal), cruise_velocity=100.0
    )
    statechart.add_node(reach)
    statechart.add_node(EndMotion.when_true(reach))
    statechart.add_nodes(SafetyEnvelope(constraints=[limit]).create_nodes())
    executor.compile(statechart=statechart)

    executor.tick_until_end()

    for joint in joints:
        assert world.state[joint.dof.id].position == pytest.approx(
            goal[joint], abs=0.02
        )
