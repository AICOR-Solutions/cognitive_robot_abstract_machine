"""
Tests for the center of mass of a branch of the world.
"""

from __future__ import annotations

import pytest

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WeightlessBranchError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ScrewAxisConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% a branch of a weighed body and a body to weigh by its shape


def cube_of_water(name: str) -> Body:
    """
    :return: A body without inertial properties, a cube of 0.1 m, holding a litre.
    """
    cube = Body(name=PrefixedName(name))
    cube.collision = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=cube),
                scale=Scale(0.1, 0.1, 0.1),
            )
        ],
        reference_frame=cube,
    )
    return cube


@pytest.fixture
def sliding_world() -> World:
    """
    A root of 1 kg carrying a cube of water, 1 kg, that slides along its x-axis.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"), inertial=Inertial(mass=1.0))
        slide = DegreeOfFreedom(name=PrefixedName("slide"))
        world.add_degree_of_freedom(slide)
        world.add_connection(
            ScrewAxisConnection(
                parent=root,
                child=cube_of_water("cube"),
                raw_dof=slide,
                screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                    axis=Vector3.X()
                ),
            )
        )
    return world


# %% weighing the branch


def test_a_body_without_inertial_is_weighed_by_its_shapes(sliding_world):
    root = sliding_world.get_body_by_name("root")
    sliding_world.get_connection_by_name("root_T_cube").position = 1.0

    center = sliding_world.compose_center_of_mass_expression(
        branch_root=root, reference_frame=root
    ).evaluate()

    assert center[0] == pytest.approx(0.5)


def test_the_center_of_mass_follows_the_joints(sliding_world):
    root = sliding_world.get_body_by_name("root")
    center = sliding_world.compose_center_of_mass_expression(
        branch_root=root, reference_frame=root
    )

    sliding_world.get_connection_by_name("root_T_cube").position = 0.4

    assert center.evaluate()[0] == pytest.approx(0.2)


def test_a_body_attached_to_the_branch_is_weighed(sliding_world):
    root = sliding_world.get_body_by_name("root")
    with sliding_world.modify_world():
        sliding_world.add_connection(
            FixedConnection(
                parent=sliding_world.get_body_by_name("cube"),
                child=cube_of_water("held"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5
                ),
            )
        )

    center = sliding_world.compose_center_of_mass_expression(
        branch_root=root, reference_frame=root
    ).evaluate()

    assert center[0] == pytest.approx(0.5)


def test_a_branch_that_weighs_nothing_has_no_center_of_mass():
    world = World()
    with world.modify_world():
        frame = Body(name=PrefixedName("frame"))
        world.add_kinematic_structure_entity(frame)

    with pytest.raises(WeightlessBranchError):
        world.compose_center_of_mass_expression(
            branch_root=frame, reference_frame=frame
        )
