"""
Tests for the goal walking a base on legs to a pose at the pace its walking controller
sets.
"""

from __future__ import annotations

import pytest

from cramph.context import StatechartContext
from cramph.statechart import Statechart

from giskardpy.motion_statechart.exceptions import RobotDoesNotWalkError
from giskardpy.motion_statechart.goals.base_pose_goals import WalkingBaseGoal
from giskardpy.motion_statechart.tasks.drive_tasks import (
    PacedBaseOrientation,
    PacedBasePosition,
    PacedBaseTurnTowardsGoalPosition,
)
from krrood_spatial_math.spatial_types import Pose

from ...dataset.robots_on_a_drive import RobotOnAnOmniDrive
from ...dataset.robots_on_legs import (
    RecordingWalkingController,
    RobotOnALeg,
    robot_on_a_leg_drive,
)

# %% building the goal


@pytest.fixture
def robot_on_a_leg() -> RobotOnALeg:
    world = robot_on_a_leg_drive()._world
    [robot] = world.get_semantic_annotations_by_type(RobotOnALeg)
    return robot


def test_a_walking_goal_walks_at_the_pace_of_the_robots_controller(robot_on_a_leg):
    target = Pose.from_xyz_rpy(
        1, 2, 0, yaw=0.5, reference_frame=robot_on_a_leg._world.root
    )

    goal = WalkingBaseGoal.moving_base_of(robot_on_a_leg, target, threshold=0.05)

    assert goal.mobile_base is robot_on_a_leg.mobile_base
    assert goal.pace == RecordingWalkingController.pace


def test_a_walking_goal_cannot_walk_a_base_that_does_not_walk(
    cylinder_bot_world,
):
    [robot] = cylinder_bot_world.get_semantic_annotations_by_type(RobotOnAnOmniDrive)
    target = Pose.from_xyz_rpy(1, 2, 0, reference_frame=cylinder_bot_world.root)

    with pytest.raises(RobotDoesNotWalkError) as raised:
        WalkingBaseGoal.moving_base_of(robot, target, threshold=0.05)

    assert raised.value.robot is robot


def test_a_walking_goal_turns_walks_and_turns_at_its_pace(robot_on_a_leg):
    target = Pose.from_xyz_rpy(
        1, 2, 0, yaw=0.5, reference_frame=robot_on_a_leg._world.root
    )
    goal = WalkingBaseGoal.moving_base_of(robot_on_a_leg, target, threshold=0.05)
    Statechart(context=StatechartContext(world=robot_on_a_leg._world)).add_node(goal)
    goal.expand()

    [turn, walk, settle] = goal.sequence.nodes

    assert isinstance(turn, PacedBaseTurnTowardsGoalPosition)
    assert [type(node) for node in walk.nodes] == [
        PacedBasePosition,
        PacedBaseTurnTowardsGoalPosition,
    ]
    assert [type(node) for node in settle.nodes] == [
        PacedBasePosition,
        PacedBaseOrientation,
    ]
    assert all(node.pace is goal.pace for node in [turn, *walk.nodes, *settle.nodes])
