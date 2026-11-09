"""
Naming a color as people do, and giving a body a named color.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.color_names import ColorName
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% naming a color


@pytest.mark.parametrize("color_name", list(ColorName))
def test_every_name_covers_its_own_representative_color(color_name: ColorName):
    assert ColorName.from_color(color_name.representative_color) is color_name


@pytest.mark.parametrize(
    "red, green, blue, expected",
    [
        (0.8, 0.1, 0.1, ColorName.RED),
        (0.95, 0.8, 0.1, ColorName.YELLOW),
        (0.1, 0.25, 0.8, ColorName.BLUE),
        (0.2, 0.7, 0.3, ColorName.GREEN),
        (0.9, 0.1, 0.5, ColorName.MAGENTA),
    ],
)
def test_a_shade_is_named_by_the_hue_it_is_nearest(
    red: float, green: float, blue: float, expected: ColorName
):
    assert ColorName.from_color(Color.from_rgba(red, green, blue)) is expected


def test_a_hue_at_the_far_end_of_the_wheel_is_still_red():
    assert ColorName.from_color(Color(hue=0.97, saturation=0.9, value=0.9)) is (
        ColorName.RED
    )


def test_a_color_too_pale_for_a_hue_is_white():
    pale = Color(
        hue=0.0,
        saturation=ColorName.minimum_chromatic_saturation / 2,
        value=1.0,
    )

    assert ColorName.from_color(pale) is ColorName.WHITE


def test_a_color_too_dark_for_a_hue_is_black():
    dark = Color(hue=0.0, saturation=1.0, value=ColorName.maximum_black_value)

    assert ColorName.from_color(dark) is ColorName.BLACK


def test_a_dim_color_without_a_hue_is_grey():
    dim = Color(
        hue=0.0,
        saturation=0.0,
        value=(ColorName.maximum_black_value + ColorName.minimum_white_value) / 2,
    )

    assert ColorName.from_color(dim) is ColorName.GREY


def test_a_shape_collection_shows_the_colors_of_its_shapes():
    shapes = ShapeCollection(
        [
            Box(color=ColorName.RED.representative_color),
            Box(color=ColorName.BLUE.representative_color),
        ]
    )

    assert shapes.color_names() == {ColorName.RED, ColorName.BLUE}


# %% giving a body a color


@pytest.fixture
def grey_cube() -> Body:
    """
    :return: A grey cube free in a world of its own.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    cube = Body(
        name=PrefixedName("cube"),
        visual=ShapeCollection(
            [Box(scale=Scale(0.05, 0.05, 0.05), color=Color.GREY())]
        ),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=cube)
        )
    return cube


def test_a_body_given_a_color_shows_it_on_every_visual_shape(grey_cube: Body):
    red = ColorName.RED.representative_color

    with grey_cube._world.modify_world():
        grey_cube.update_visual_color(red)

    assert [shape.color for shape in grey_cube.visual] == [red]


def test_giving_a_body_a_color_keeps_its_geometry(grey_cube: Body):
    visual_scales = [shape.scale for shape in grey_cube.visual]
    collision = grey_cube.collision

    with grey_cube._world.modify_world():
        grey_cube.update_visual_color(ColorName.RED.representative_color)

    assert [shape.scale for shape in grey_cube.visual] == visual_scales
    assert grey_cube.collision is collision


def test_giving_a_body_a_color_is_a_change_of_the_world_model(grey_cube: Body):
    world = grey_cube._world
    version = world.get_world_model_manager().version

    with world.modify_world():
        grey_cube.update_visual_color(ColorName.RED.representative_color)

    assert world.get_world_model_manager().version > version
