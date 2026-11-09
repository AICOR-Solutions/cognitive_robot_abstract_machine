"""
Detecting an object by its color: a red, a yellow and a blue cube of one type stand on
Tracy's table, and asking for the red one finds only it, from the world model and live
through RoboKudo's cube localization.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pytest

from coraplex.datastructures.enums import DetectionTechnique
from coraplex.perception import (
    Detection,
    PerceptionQuery,
    WorldPerception,
)
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.plan_transformations import DetectWithRoboKudo
from cramph.data_types import LifeCycleValues
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.color_names import ColorName
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from ..plan_running import run_plan

# %% cubes on Tracy's table


@dataclass
class CubesOnTheTable:
    """
    Cubes of one type and different colors standing on Tracy's table.
    """

    world: World
    """
    The world holding Tracy and the cubes.
    """

    tracy: Tracy
    """
    The robot looking at the cubes.
    """

    cubes: dict[ColorName, Body]
    """
    The body of each cube by the name of its color.
    """

    table_region: VolumetricBoundingBox
    """
    The region above Tracy's table, holding every cube.
    """

    edge_length: float = 0.05
    """
    Edge length of every cube, in metres.
    """

    def add_cube(self, name: str, color: Color, x: float, y: float) -> Body:
        """
        Put a cube annotated as milk on the table.

        :param name: Name of its body.
        :param color: Color of its visual geometry.
        :param x: Position of its centre in the world frame, in metres.
        :param y: Position of its centre in the world frame, in metres.
        :return: Its body.
        """
        scale = Scale(*[self.edge_length] * 3)
        body = Body(
            name=PrefixedName(name),
            visual=ShapeCollection([Box(scale=scale, color=color)]),
            collision=ShapeCollection([Box(scale=scale)]),
        )
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world, parent=self.world.root, child=body
                )
            )
            self.world.add_semantic_annotation(Milk(root=body))
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x,
            y=y,
            z=self.tracy.table.top_z + self.edge_length / 2,
            reference_frame=self.world.root,
        )
        return body

    def query(self, color: ColorName | None) -> PerceptionQuery:
        """
        :param color: The color asked for, None for any.
        :return: A query for a cube of that color on the table.
        """
        return PerceptionQuery(
            Milk, self.table_region, self.tracy, self.world, color=color
        )


@pytest.fixture
def table_without_cubes(tracy_context) -> CubesOnTheTable:
    """
    :return: Tracy with its arms parked out of its camera's view and its table, no cube
        on it yet.
    """
    world, tracy, extensions = tracy_context
    for arm in tracy.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    top = tracy.table.top_z
    return CubesOnTheTable(
        world=world,
        tracy=tracy,
        cubes={},
        table_region=VolumetricBoundingBox(
            origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            min_x=0.3,
            min_y=-0.4,
            min_z=top - 0.05,
            max_x=0.9,
            max_y=0.4,
            max_z=top + 0.2,
        ),
    )


@pytest.fixture
def three_cubes(table_without_cubes: CubesOnTheTable) -> CubesOnTheTable:
    """
    :return: A red, a yellow and a blue cube, all annotated as milk, on Tracy's table.
    """
    scene = table_without_cubes
    shades = {
        ColorName.RED: (Color.from_rgba(red=0.8, green=0.1, blue=0.1), -0.15),
        ColorName.YELLOW: (Color.from_rgba(red=0.95, green=0.8, blue=0.1), 0.0),
        ColorName.BLUE: (Color.from_rgba(red=0.1, green=0.25, blue=0.8), 0.15),
    }
    for color_name, (color, y) in shades.items():
        scene.cubes[color_name] = scene.add_cube(
            f"{color_name.value}_cube", color, x=0.6, y=y
        )
    return scene


def position_of(body: Body) -> np.ndarray:
    """
    :return: Where the world holds the body, in world coordinates.
    """
    return body.global_pose.position.to_np().flatten()[:3]


# %% answering from the world


def test_the_world_answers_a_color_query_with_the_cube_of_that_color(
    three_cubes: CubesOnTheTable,
):
    assert three_cubes.query(ColorName.RED).from_world() == [
        three_cubes.cubes[ColorName.RED]
    ]


def test_the_world_answers_a_query_without_a_color_with_every_cube(
    three_cubes: CubesOnTheTable,
):
    assert set(three_cubes.query(None).from_world()) == set(three_cubes.cubes.values())


def test_the_world_reports_the_cube_in_the_color_asked_for(
    three_cubes: CubesOnTheTable,
):
    detection = WorldPerception().detect(three_cubes.query(ColorName.RED))

    assert detection.color is ColorName.RED
    np.testing.assert_allclose(
        detection.pose.position.to_np().flatten()[:3],
        position_of(three_cubes.cubes[ColorName.RED]),
        atol=1e-9,
    )


def test_a_detect_action_for_red_succeeds_among_cubes_of_one_type(
    three_cubes: CubesOnTheTable, tracy_context
):
    """
    Three cubes of the type asked for would be ambiguous; the color tells them apart.
    """
    world, tracy, extensions = tracy_context
    detection = DetectAction(
        technique=DetectionTechnique.TYPES,
        object_sem_annotation=Milk,
        detection_box=three_cubes.table_region,
        color=ColorName.RED,
    )

    run_plan(detection, extensions)

    assert detection.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% writing a detection of a color into the world


def test_a_detection_of_a_color_moves_the_cube_of_that_color(
    three_cubes: CubesOnTheTable,
):
    red_cube = three_cubes.cubes[ColorName.RED]
    others = {
        cube: cube.global_pose.to_np()
        for color, cube in three_cubes.cubes.items()
        if color is not ColorName.RED
    }
    perceived_position = position_of(red_cube) + np.array([0.02, 0.01, 0.0])

    [annotation] = Detection(
        semantic_annotation=Milk,
        pose=Pose.from_xyz_rpy(
            *perceived_position, reference_frame=three_cubes.world.root
        ),
        color=ColorName.RED,
    ).apply_to(three_cubes.world, region=three_cubes.table_region)

    assert annotation.root is red_cube
    np.testing.assert_allclose(position_of(red_cube), perceived_position, atol=1e-9)
    for cube, pose in others.items():
        np.testing.assert_allclose(cube.global_pose.to_np(), pose, atol=1e-9)


def test_a_detected_color_is_written_into_a_cube_not_showing_it(
    table_without_cubes: CubesOnTheTable,
):
    """
    Where the world did not know the color, the detection teaches it, so the cube is
    shown in the color it was seen in.
    """
    cube = table_without_cubes.add_cube("cube", Color.GREY(), x=0.6, y=0.0)

    Detection(
        semantic_annotation=Milk, pose=cube.global_pose, color=ColorName.RED
    ).apply_to(table_without_cubes.world, region=table_without_cubes.table_region)

    assert [shape.color for shape in cube.visual] == [
        ColorName.RED.representative_color
    ]


def test_a_cube_showing_the_detected_color_keeps_its_shade(
    three_cubes: CubesOnTheTable,
):
    red_cube = three_cubes.cubes[ColorName.RED]
    shades = [shape.color for shape in red_cube.visual]

    Detection(
        semantic_annotation=Milk, pose=red_cube.global_pose, color=ColorName.RED
    ).apply_to(three_cubes.world, region=three_cubes.table_region)

    assert [shape.color for shape in red_cube.visual] == shades


# %% answering live through RoboKudo


@pytest.fixture
def cube_localization(three_cubes: CubesOnTheTable, rclpy_node):
    """
    RoboKudo's cube localization answering queries on the cubes' world, as a simulated
    run hosts it.
    """
    if find_spec("robokudo_msgs") is None:
        pytest.skip("RoboKudo's messages are not installed")
    from robokudo.descriptors.analysis_engines.tracy_cubes import AnalysisEngine
    from robokudo.query_service import QueryService

    service = QueryService(analysis_engine=AnalysisEngine(), world=three_cubes.world)
    service.start()
    yield service
    service.stop()


def test_a_live_detect_action_for_red_succeeds_among_cubes_of_one_type(
    three_cubes: CubesOnTheTable, tracy_context, rclpy_node, cube_localization
):
    """
    RoboKudo sees three cubes of the type asked for, which would be ambiguous; the color
    it is asked for tells them apart.
    """
    world, tracy, extensions = tracy_context
    detection = DetectAction(
        technique=DetectionTechnique.TYPES,
        object_sem_annotation=Milk,
        detection_box=three_cubes.table_region,
        color=ColorName.RED,
    )

    run_plan(
        detection,
        extensions,
        ros_node=rclpy_node,
        plan_adaptations=[DetectWithRoboKudo()],
    )

    assert detection.life_cycle_state == LifeCycleValues.SUCCEEDED
