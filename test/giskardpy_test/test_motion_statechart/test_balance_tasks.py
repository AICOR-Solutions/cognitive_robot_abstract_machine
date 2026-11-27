"""
Tests for keeping the center of mass of a robot standing on its feet over them while its
upper body moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.tasks.balance_tasks import CenterOfMassOverTheFeet
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.walking import StandingBalance
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.derivatives import DerivativeMap
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% a robot standing on two feet, reaching forward with a heavy arm


@pytest.fixture
def standing_world() -> World:
    """
    A pelvis of 10 kg standing a metre above two feet of 1 kg, each 0.2 m long, and an
    arm of 10 kg sliding forward from the pelvis.
    """
    world = World()
    with world.modify_world():
        floor = Body(name=PrefixedName("floor"))
        pelvis = Body(name=PrefixedName("pelvis"), inertial=Inertial(mass=10.0))
        world.add_connection(
            FixedConnection(
                parent=floor,
                child=pelvis,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0
                ),
            )
        )
        for side, y in (("left", 0.1), ("right", -0.1)):
            foot = Body(name=PrefixedName(f"{side}_foot"), inertial=Inertial(mass=1.0))
            foot.collision = ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix(reference_frame=foot),
                        scale=Scale(0.2, 0.1, 0.02),
                    )
                ],
                reference_frame=foot,
            )
            world.add_connection(
                FixedConnection(
                    parent=pelvis,
                    child=foot,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=y, z=-1.0
                    ),
                )
            )
        arm = Body(name=PrefixedName("arm"), inertial=Inertial(mass=10.0))
        reach = DegreeOfFreedom(
            name=PrefixedName("reach"),
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(-2.0, -1.0, None, None),
                upper=DerivativeMap(2.0, 1.0, None, None),
            ),
            has_hardware_interface=True,
        )
        world.add_degree_of_freedom(reach)
        world.add_connection(
            PrismaticConnection(
                parent=pelvis, child=arm, raw_dof=reach, axis=Vector3.X()
            )
        )
    MinimalRobot.from_world(world)
    return world


@pytest.fixture
def balance() -> StandingBalance:
    return StandingBalance(
        foot_names=["left_foot", "right_foot"],
        support_margin=0.02,
        maximal_center_of_mass_speed=0.1,
        maximal_center_of_mass_acceleration=0.5,
    )


def center_of_mass_ahead_of_the_feet(world: World) -> float:
    """
    :return: How far, in metres, the center of mass of the robot lies ahead of its feet.
    """
    pelvis = world.get_body_by_name("pelvis")
    return float(
        world.compose_center_of_mass_expression(pelvis, world.root).evaluate()[0]
    )


def reach_forward(
    world: World, balance: StandingBalance | None, ticks: int
) -> list[float]:
    """
    Slide the arm a metre forward, balanced as `balance` says, if at all.

    :return: Where the center of mass lies ahead of the feet after every tick.
    """
    config = QPControllerConfig.create_with_simulation_defaults()
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[MotionControl(qp_controller_config=config)],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        JointPositionList(
            goal_state=JointState.from_mapping(
                {world.get_connection_by_name("pelvis_T_arm"): 1.0}
            )
        )
    )
    if balance is not None:
        pelvis = world.get_body_by_name("pelvis")
        statechart.add_node(
            CenterOfMassOverTheFeet(
                robot_root=pelvis,
                feet=[world.get_body_by_name(name) for name in balance.foot_names],
                balance=balance,
            )
        )
    executor.compile(statechart=statechart)
    trajectory = []
    for _ in range(ticks):
        executor.tick()
        trajectory.append(center_of_mass_ahead_of_the_feet(world))
    return trajectory


# %% the center of mass of a branch


def test_the_center_of_mass_weighs_every_body_of_the_branch(standing_world):
    standing_world.get_connection_by_name("pelvis_T_arm").position = 0.44

    # pelvis 10 kg and feet 2 kg at x=0, arm 10 kg at x=0.44
    assert center_of_mass_ahead_of_the_feet(standing_world) == pytest.approx(0.2)


# %% keeping it over the feet


def test_an_unbalanced_reach_carries_the_center_of_mass_past_the_feet(
    standing_world,
):
    trajectory = reach_forward(standing_world, None, ticks=100)

    assert trajectory[-1] == pytest.approx(10.0 / 22.0, abs=1e-3)


def test_the_center_of_mass_stays_over_the_feet(standing_world, balance):
    trajectory = reach_forward(standing_world, balance, ticks=200)

    front_of_the_feet = 0.1
    assert max(trajectory) <= front_of_the_feet - balance.support_margin + 1e-3
    assert trajectory[-1] == pytest.approx(
        front_of_the_feet - balance.support_margin, abs=5e-3
    )


def test_the_center_of_mass_moves_no_faster_than_its_limits(standing_world, balance):
    trajectory = reach_forward(standing_world, balance, ticks=200)

    time_step = QPControllerConfig.create_with_simulation_defaults().control_time_step
    velocities = np.diff(trajectory) / time_step.total_seconds()
    accelerations = np.diff(velocities) / time_step.total_seconds()
    assert np.max(np.abs(velocities)) <= balance.maximal_center_of_mass_speed + 1e-3
    assert (
        np.max(np.abs(accelerations))
        <= balance.maximal_center_of_mass_acceleration + 1e-2
    )
