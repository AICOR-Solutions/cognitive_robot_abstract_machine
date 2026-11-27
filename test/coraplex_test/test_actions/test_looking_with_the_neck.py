"""
A robot whose torso costs more to move than its other joints looks with its neck alone.
"""

from __future__ import annotations

from coraplex.robot_plans.actions.core.navigation import LookAtAction
from giskardpy.motion_statechart.tasks.pointing import Pointing
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.robots.robot_parts import Neck
from ...plan_running import simulated_executor, statechart_of

# %% where a look turns the camera from


def pointing_of_a_look(world, extensions) -> Pointing:
    """
    :return: The task turning the camera of a look at a point low in front of the robot.
    """
    look = LookAtAction(
        look_target=Pose.from_xyz_rpy(x=1.0, z=0.5, reference_frame=world.root)
    )
    executor = simulated_executor(extensions)
    executor.compile(statechart=statechart_of(executor, look))
    [pointing] = look.statechart.get_nodes_by_type(node_type=Pointing)
    return pointing


def test_a_robot_whose_torso_costs_more_to_move_looks_with_its_neck(
    pr2_apartment_context, monkeypatch
) -> None:
    world, pr2, extensions = pr2_apartment_context
    monkeypatch.setattr(pr2.get_torso(), "motion_cost", 50.0)
    [neck] = [part for part in pr2._robot_parts if isinstance(part, Neck)]

    assert pointing_of_a_look(world, extensions).kinematic_chain.root is neck.root


def test_a_robot_whose_torso_moves_as_cheaply_as_any_joint_looks_from_the_torso(
    pr2_apartment_context,
) -> None:
    world, pr2, extensions = pr2_apartment_context

    assert (
        pointing_of_a_look(world, extensions).kinematic_chain.root
        is pr2.get_torso().root
    )
