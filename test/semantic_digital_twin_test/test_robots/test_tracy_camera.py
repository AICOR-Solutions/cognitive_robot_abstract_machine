"""
Tracy's fixed camera: where it looks and what it sees; skipped where Tracy's description
is not installed.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not Tracy.is_description_installed(),
    reason="iai_tracy_description is not installed",
)


@pytest.fixture
def tracy() -> Tracy:
    """
    :return: Tracy in a world of its own.
    """
    world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).create_world()
    return Tracy.from_world(world)


# %% where the camera looks


def test_tracys_camera_looks_along_its_camera_links_x_axis(tracy: Tracy) -> None:
    """
    The camera link is a ROS body frame looking along its x-axis, so the camera's
    optical z-axis is that x-axis.
    """
    camera = tracy.get_default_camera()
    world = camera.root._world
    camera_link = world.get_body_by_name("camera_link")

    camera_link_T_camera = world.compute_forward_kinematics_np(camera_link, camera.root)

    np.testing.assert_allclose(camera_link_T_camera[:3, 2], [1, 0, 0], atol=1e-12)
    assert camera.root.parent_connection.parent is camera_link


def test_tracys_camera_looks_down_at_its_table(tracy: Tracy) -> None:
    camera = tracy.get_default_camera()

    viewing_direction = camera.root.global_transform.to_np()[:3, 2]

    assert viewing_direction[2] < 0


# %% what the camera sees


def test_tracys_camera_sees_a_block_on_its_table_past_its_own_housing(
    tracy: Tracy,
) -> None:
    """
    The housing around the camera link encloses the optical frame; the camera sees
    through it onto the table.
    """
    world = tracy._world
    block = Body(
        name=PrefixedName("block"),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=block,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.55, z=tracy.table.top_z + 0.025, reference_frame=world.root
                ),
            )
        )
    camera = tracy.get_default_camera()

    frame = world.ray_tracer.render(camera)

    seen_bodies = {
        world.kinematic_structure[index] for index in np.unique(frame.segmentation)
    }
    assert block in seen_bodies
