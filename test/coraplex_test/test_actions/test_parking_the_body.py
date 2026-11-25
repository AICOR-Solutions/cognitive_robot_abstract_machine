"""
Parking the arms of a robot whose torso and neck declare parked states, as a humanoid's
do, parks those too.
"""

from __future__ import annotations

import pytest

from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart, Neck
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from ...plan_running import run_plan, simulated_executor, statechart_of

# %% parked states of the torso and the neck


def connection_named(world: World, name: str) -> ActiveConnection1DOF:
    """
    :return: The connection of `world` the joint `name` is.
    """
    [connection] = [
        connection for connection in world.connections if connection.name.name == name
    ]
    return connection


def declare_parked_state(
    part: AbstractRobotPart,
    positions: dict[ActiveConnection1DOF, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Let `part` declare a parked state at `positions` for the rest of the test.
    """
    parked = JointState.from_mapping(
        name=PrefixedName("parked", prefix=part.name.name),
        mapping=positions,
        state_type=StaticJointState.PARK,
    )
    monkeypatch.setattr(part, "joint_states", [*part.joint_states, parked])


def test_parking_the_arms_parks_a_torso_and_a_neck_that_declare_parked_states(
    pr2_apartment_context, monkeypatch
) -> None:
    world, pr2, extensions = pr2_apartment_context
    torso_lift = connection_named(world, PR2Joint.TORSO_LIFT)
    head_pan = connection_named(world, PR2Joint.HEAD_PAN)
    declare_parked_state(pr2.get_torso(), {torso_lift: 0.1}, monkeypatch)
    [neck] = [part for part in pr2._robot_parts if isinstance(part, Neck)]
    declare_parked_state(neck, {head_pan: 0.4}, monkeypatch)

    run_plan(ParkArmsAction(arms=pr2.all_arms), extensions)

    assert torso_lift.position == pytest.approx(0.1, abs=0.01)
    assert head_pan.position == pytest.approx(0.4, abs=0.01)


# %% where the base stands while the arms park


def test_parking_the_arms_holds_a_mobile_base_where_it_stands(
    pr2_apartment_context,
) -> None:
    """
    Collision avoidance buys clearance by driving a free base, which carries the parked
    arms into whatever the robot stands at.
    """
    world, pr2, extensions = pr2_apartment_context
    park = ParkArmsAction(arms=pr2.all_arms)
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, park))

    [hold] = park.statechart.get_nodes_by_type(HoldPose)
    assert (hold.root_link, hold.tip_link) == (world.root, pr2.root)
