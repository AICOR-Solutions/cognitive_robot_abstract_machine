"""
Tests for plans moving a robot whose base walks on legs: it walks to a pose at its
controller's pace, and in physics it stands rather than holding a pose while it works.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from coraplex.plans.context_extensions import RobotAccess, WalkingBasesStand
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.plan_transformations import WalkTheBaseOnItsLegs
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from giskardpy.motion_statechart.goals.base_pose_goals import WalkingBaseGoal
from giskardpy.motion_statechart.tasks.balance_tasks import CenterOfMassOverTheFeet
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from ...dataset.robots_on_a_drive import RobotOnAnOmniDrive
from ...dataset.robots_on_legs import (
    RecordingWalkingController,
    RobotOnALeg,
    robot_on_a_leg_drive,
)

# %% fixtures


@pytest.fixture
def robot_on_a_leg() -> RobotOnALeg:
    world = robot_on_a_leg_drive()._world
    [robot] = world.get_semantic_annotations_by_type(RobotOnALeg)
    return robot


def navigation_of(robot: AbstractRobot, *extensions) -> NavigateAction:
    """
    :return: A navigation of `robot` in a statechart whose context has `extensions`.
    """
    world = robot._world
    action = NavigateAction(target_location=Pose(reference_frame=world.root))
    context = StatechartContext.with_extensions(
        world, [RobotAccess(robot), *extensions]
    )
    Statechart(context=context).add_node(action)
    return action


# %% walking to a pose


def test_a_walking_robot_walks_to_where_it_navigates(robot_on_a_leg):
    action = navigation_of(robot_on_a_leg)
    walking = WalkTheBaseOnItsLegs()

    assert walking.is_applicable(action)
    walking.apply(action)
    assert action.base_goal_type is WalkingBaseGoal
    assert (
        action.base_pose_threshold == RecordingWalkingController.pace.arrival_tolerance
    )


def test_a_driven_robot_is_left_to_its_drive(cylinder_bot_world):
    [robot] = cylinder_bot_world.get_semantic_annotations_by_type(RobotOnAnOmniDrive)

    assert not WalkTheBaseOnItsLegs().is_applicable(navigation_of(robot))


def test_a_physics_run_walks_walking_robots():
    [adaptations] = [
        field
        for field in fields(PhysicallySimulatedPlanExecutor)
        if field.name == "plan_adaptations"
    ]

    assert any(
        isinstance(adaptation, WalkTheBaseOnItsLegs)
        for adaptation in adaptations.default_factory()
    )


# %% keeping the base still


def test_a_walking_base_stands_where_the_context_lets_it(robot_on_a_leg):
    """
    The base is not walked back to where it started, however far it swayed.
    """
    action = navigation_of(robot_on_a_leg, WalkingBasesStand())

    [stand, _] = action.keep_base_still()

    assert isinstance(stand, HoldPose)
    assert stand.cruise_linear_velocity == NavigateAction.creeping_velocity
    assert stand.cruise_angular_velocity == NavigateAction.creeping_velocity
    assert (stand.translation_threshold, stand.orientation_threshold) == (
        math.inf,
        math.inf,
    )


def test_a_walking_base_is_held_at_its_pose_elsewhere(robot_on_a_leg):
    action = navigation_of(robot_on_a_leg)

    assert [type(task) for task in action.keep_base_still()] == [HoldPose]


def test_a_standing_walking_base_keeps_its_center_of_mass_over_its_feet(
    robot_on_a_leg,
):
    action = navigation_of(robot_on_a_leg, WalkingBasesStand())

    [_, balance] = action.keep_base_still()

    assert isinstance(balance, CenterOfMassOverTheFeet)
    assert balance.balance == RecordingWalkingController.balance
    assert balance.robot_root is robot_on_a_leg.root
    assert [foot.name.name for foot in balance.feet] == (
        RecordingWalkingController.balance.foot_names
    )


def test_a_driven_base_is_only_held(cylinder_bot_world):
    """
    A robot without a walking controller gets exactly the task it got before balancing.
    """
    [robot] = cylinder_bot_world.get_semantic_annotations_by_type(RobotOnAnOmniDrive)
    action = navigation_of(robot, WalkingBasesStand())

    [hold] = action.keep_base_still()

    assert isinstance(hold, HoldPose)
    assert (hold.root_link, hold.tip_link) == (cylinder_bot_world.root, robot.root)
    assert (hold.cruise_linear_velocity, hold.translation_threshold) == (
        HoldPose.cruise_linear_velocity,
        HoldPose.translation_threshold,
    )
