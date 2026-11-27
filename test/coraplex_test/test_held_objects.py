"""
A check that an object a hand lifts rises with it.
"""

from __future__ import annotations

import pytest

from coraplex.plans.failures import ObjectLeftBehindByTheHand
from coraplex.plans.held_objects import ObjectRisesWithTheHand
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% a hand and a cube, both free


@pytest.fixture
def hand_and_cube() -> tuple[Body, Body]:
    """
    :return: A free tool frame and a free cube at the origin of a world of their own.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    hand = Body(name=PrefixedName("tool_frame"))
    cube = Body(name=PrefixedName("cube"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        for body in (hand, cube):
            world.add_connection(
                Connection6DoF.create_with_dofs(world=world, parent=root, child=body)
            )
    return hand, cube


def raise_by(body: Body, height: float) -> None:
    """
    Put a body at a height above the world's origin.
    """
    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=height, reference_frame=body._world.root
    )


def lift(hand: Body, cube: Body, hand_rise: float, cube_rise: float):
    """
    Start a check, then raise the hand and the cube and tick it again.

    :return: The check, once it ticked after the lift.
    """
    world = hand._world
    executor = StatechartExecutor(context=StatechartContext(world=world))
    statechart = Statechart(context=executor.context)
    node = ObjectRisesWithTheHand(held_body=cube, tool_frame=hand)
    statechart.add_node(node=node)
    executor.compile(statechart=statechart)
    executor.tick()
    raise_by(hand, hand_rise)
    raise_by(cube, cube_rise)
    executor.tick()
    return node


# %% checking


def test_an_object_rising_with_the_hand_passes_the_check(hand_and_cube):
    hand, cube = hand_and_cube
    node = lift(hand, cube, hand_rise=0.1, cube_rise=0.095)
    assert node.observation_state == ObservationStateValue.TRUE


def test_an_object_left_on_the_table_fails_the_check_naming_the_shortfall(
    hand_and_cube,
):
    hand, cube = hand_and_cube
    with pytest.raises(ObjectLeftBehindByTheHand) as raised:
        lift(hand, cube, hand_rise=0.1, cube_rise=0.0)
    assert raised.value.held_body is cube
    assert raised.value.shortfall == pytest.approx(0.1)
