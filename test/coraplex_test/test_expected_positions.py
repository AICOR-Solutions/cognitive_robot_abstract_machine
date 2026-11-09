"""
A step checking that the bodies it acts on still lie where the plan was grounded with
them.
"""

from __future__ import annotations

import pytest

from coraplex.plans.expected_positions import BodiesWhereExpected, ExpectedPosition
from coraplex.plans.failures import BodiesMovedSinceGrounding
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% a free body


@pytest.fixture
def cube() -> Body:
    """
    :return: A free body at the origin of a world of its own.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    cube = Body(name=PrefixedName("cube"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=cube)
        )
    return cube


def check(cube: Body, tolerance: float) -> BodiesWhereExpected:
    """
    Run a check expecting the cube at the world's origin.

    :return: The check, once it ticked.
    """
    world = cube._world
    executor = StatechartExecutor(context=StatechartContext(world=world))
    statechart = Statechart(context=executor.context)
    node = BodiesWhereExpected(
        expected=[
            ExpectedPosition(body=cube, position=Point3(reference_frame=world.root))
        ],
        tolerance=tolerance,
    )
    statechart.add_node(node)
    executor.compile(statechart)
    executor.tick()
    return node


def move(cube: Body, x: float) -> None:
    """
    Move the cube along the world's x-axis.
    """
    cube.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x, reference_frame=cube._world.root
    )


# %% checking


def test_a_body_within_the_tolerance_is_where_expected(cube):
    move(cube, 0.01)
    assert check(cube, tolerance=0.02).observation_state == ObservationStateValues.TRUE


def test_a_body_beyond_the_tolerance_fails_the_check_naming_how_far(cube):
    move(cube, 0.05)
    with pytest.raises(BodiesMovedSinceGrounding) as raised:
        check(cube, tolerance=0.02)
    [displacement] = raised.value.displacements
    assert displacement.body is cube
    assert displacement.distance == pytest.approx(0.05)
