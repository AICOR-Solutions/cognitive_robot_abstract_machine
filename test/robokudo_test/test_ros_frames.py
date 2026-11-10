"""
ROS frame identifier normalization and validation.
"""

from __future__ import annotations

# %% Imports
import pytest

from robokudo.exceptions import InvalidROSFrameId
from robokudo.utils.ros_frames import find_bodies_for_ros_frame, normalize_ros_frame_id
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% Invalid identifiers


@pytest.mark.parametrize("frame_id", ["", "/", "///"])
def test_normalization_rejects_empty_ros_identifier(frame_id: str) -> None:
    """
    Invalid ROS identifiers raise a frame-specific exception with the supplied value.
    """
    with pytest.raises(InvalidROSFrameId) as raised:
        normalize_ros_frame_id(frame_id)
    assert raised.value.frame_id == frame_id


def test_body_lookup_rejects_empty_ros_identifier() -> None:
    """
    Body lookup preserves the frame-specific normalization error.
    """
    with pytest.raises(InvalidROSFrameId):
        find_bodies_for_ros_frame(World(), "")


# %% frames of a robot whose bodies carry its name as prefix


@pytest.fixture
def robot_world() -> World:
    """
    :return: A world holding a robot's camera link, named with the robot's prefix.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName(name="camera_link", prefix="tracy")))
    return world


def test_a_frame_named_with_the_prefix_finds_the_body(robot_world: World) -> None:
    [body] = robot_world.bodies

    assert find_bodies_for_ros_frame(robot_world, "/tracy/camera_link") == [body]


def test_a_frame_named_without_the_prefix_finds_the_body(robot_world: World) -> None:
    """
    A driver outside the world, such as a camera's, names its frames without prefix.
    """
    [body] = robot_world.bodies

    assert find_bodies_for_ros_frame(robot_world, "camera_link") == [body]


def test_a_full_name_is_preferred_to_a_name_without_prefix(robot_world: World) -> None:
    [camera_link] = robot_world.bodies
    unprefixed = Body(name=PrefixedName(name="camera_link"))
    with robot_world.modify_world():
        robot_world.add_body(unprefixed)
        robot_world.add_connection(
            Connection6DoF.create_with_dofs(
                parent=camera_link, child=unprefixed, world=robot_world
            )
        )

    assert find_bodies_for_ros_frame(robot_world, "camera_link") == [unprefixed]
