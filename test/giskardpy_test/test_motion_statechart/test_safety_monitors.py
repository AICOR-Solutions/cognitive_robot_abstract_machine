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
    MaximumJointVelocity,
    SafetyEnvelope,
    SafetyMonitor,
    MaximumToolVelocity,
    WorkspaceBox,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from krrood_spatial_math.spatial_types import Point3
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
    "velocity, expected",
    [(0.05, ObservationStateValues.FALSE), (0.2, ObservationStateValues.TRUE)],
)
def test_a_joint_faster_than_its_limit_is_a_violation(
    pr2_world_state_reset, velocity, expected
):
    world = pr2_world_state_reset
    torso = _torso(world)
    world.state[torso.dof.id].velocity = velocity
    limit = MaximumJointVelocity(
        constraint="torso velocity", connections=[torso], maximum_velocity=0.1
    )
    assert _observe(world, limit.create_monitor()) == expected


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
