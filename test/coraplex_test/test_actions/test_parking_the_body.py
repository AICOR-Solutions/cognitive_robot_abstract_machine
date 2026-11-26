"""
Parking the arms of a robot whose torso and neck declare parked states, as a humanoid's
do, parks those too.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    HoldPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart, Arm, Neck
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
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


# %% raising the arms clear of a table before they fold


@pytest.fixture
def left_arm_declaring_a_raised_state(pr2_apartment_context, monkeypatch):
    """
    The PR2's left arm, declaring a raised state that lifts its shoulder.
    """
    world, pr2, extensions = pr2_apartment_context
    arm = pr2.all_arms[0]
    raised = JointState.from_mapping(
        name=PrefixedName("raised", prefix=arm.name.name),
        mapping={arm.active_connections[1]: -0.3},
        state_type=StaticJointState.RAISED,
    )
    monkeypatch.setattr(arm, "joint_states", [*arm.joint_states, raised])
    return arm


def compiled_park(plan: list, extensions) -> ParkArmsAction:
    """
    :return: The last step of `plan`, a park, once `plan` is compiled.
    """
    executor = simulated_executor(extensions)
    executor.compile(statechart_of(executor, Sequence(nodes=plan)))
    return plan[-1]


def goals_of_park(park: ParkArmsAction, goal_type: type) -> list:
    """
    :return: The goals of `goal_type` below the compiled `park`.
    """
    return [
        node
        for node in park.statechart.get_nodes_by_type(goal_type)
        if park in node.path
    ]


def pick_up_the_milk(world: World, arm: Arm) -> PickUpAction:
    """
    :return: A pick-up of the milk carton in the apartment with `arm`.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    return PickUpAction(grasp=GraspCandidate.from_body_origin(milk), arm=arm)


def root_P_tool_in_state(
    world: World, root, arm: Arm, state_type: StaticJointState
) -> list[float]:
    """
    :return: Where the tool frame of `arm` stands in `root`'s frame with the arm in its
        joint state of `state_type`.
    """
    with world.reset_state_context():
        arm.get_joint_state_by_type(state_type).apply_to(world)
        return world.compute_forward_kinematics_np(
            root, arm.end_effector.tool_frame
        )[:3, 3]


def test_an_arm_that_reached_is_drawn_back_through_its_raised_state_before_it_parks(
    pr2_apartment_context, left_arm_declaring_a_raised_state
) -> None:
    """
    Folded straight into its parked state, an arm reaching over a table sweeps its hand
    down through the tabletop; drawn straight to where its raised state holds the hand
    and from there straight to where its parked state does, the hand stays above it.
    """
    world, pr2, extensions = pr2_apartment_context
    arm = left_arm_declaring_a_raised_state
    park = ParkArmsAction(arms=pr2.all_arms)

    compiled_park([pick_up_the_milk(world, arm), park], extensions)

    raising, lowering = goals_of_park(park, CartesianPosition)
    [parking] = goals_of_park(park, JointPositionList)
    for goal, state_type in (
        (raising, StaticJointState.RAISED),
        (lowering, StaticJointState.PARK),
    ):
        assert (goal.root_link, goal.tip_link) == (
            arm.root,
            arm.end_effector.tool_frame,
        )
        assert goal.goal_point.to_np()[:3] == pytest.approx(
            root_P_tool_in_state(world, arm.root, arm, state_type)
        )
    assert park.statechart.get_preceding_nodes_by_type(parking, CartesianPosition)[
        :2
    ] == [lowering, raising]


def test_an_arm_that_did_not_reach_and_stands_parked_parks_straight(
    pr2_apartment_context, left_arm_declaring_a_raised_state
) -> None:
    world, pr2, extensions = pr2_apartment_context
    left_arm_declaring_a_raised_state.get_joint_state_by_type(
        StaticJointState.PARK
    ).apply_to(world)
    other_arm = pr2.all_arms[1]
    park = ParkArmsAction(arms=pr2.all_arms)

    compiled_park([pick_up_the_milk(world, other_arm), park], extensions)

    assert goals_of_park(park, CartesianPosition) == []


def test_an_arm_standing_out_over_a_table_is_drawn_back_though_this_plan_did_not_reach(
    pr2_apartment_context, left_arm_declaring_a_raised_state
) -> None:
    """
    A park performed on its own, as a step of a plan run one step at a time, knows of
    no earlier grasp; an arm whose hand stands farther from where its parked state holds
    it than its raised state does is drawn back through its raised state all the same.
    """
    world, pr2, extensions = pr2_apartment_context
    arm = left_arm_declaring_a_raised_state
    for connection in arm.active_connections:
        connection.position = 0.0
    park_point = root_P_tool_in_state(world, arm.root, arm, StaticJointState.PARK)
    raised_point = root_P_tool_in_state(world, arm.root, arm, StaticJointState.RAISED)
    tool_point = world.compute_forward_kinematics_np(
        arm.root, arm.end_effector.tool_frame
    )[:3, 3]
    assert np.linalg.norm(tool_point - park_point) > np.linalg.norm(
        raised_point - park_point
    )
    park = ParkArmsAction(arms=pr2.all_arms)

    compiled_park([park], extensions)

    raising, lowering = goals_of_park(park, CartesianPosition)
    assert raising.tip_link is arm.end_effector.tool_frame
