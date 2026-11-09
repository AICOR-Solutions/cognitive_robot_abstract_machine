"""
Localizing the cubes on Tracy's table through Tracy's simulated camera; skipped where
Tracy's description is not installed.
"""

from __future__ import annotations

# %% Imports
import math
from dataclasses import dataclass

import numpy as np
import py_trees
import pytest
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

import robokudo.pipeline
import robokudo.utils.tree_execution
import robokudo.world
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from robokudo.annotators.cluster_color import Color as SemanticColorName
from robokudo.behaviours.clear_annotator_outputs import ClearAnnotatorOutputs
from robokudo.cas import CAS
from robokudo.descriptors.analysis_engines.tracy_cubes import AnalysisEngine
from robokudo.query_service import QueryService
from robokudo.types.annotation import PoseAnnotation, SemanticColor
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not Tracy.is_description_installed(),
    reason="iai_tracy_description is not installed",
)

# %% a scene of cubes


@dataclass
class PlacedCube:
    """
    A cube standing on Tracy's table where the test put it.
    """

    color_name: SemanticColorName
    """
    The color the cube should be perceived in.
    """

    color: Color
    """
    The color of its visual geometry.
    """

    x: float
    """
    Position of its centre in the world frame, in metres.
    """

    y: float
    """
    Position of its centre in the world frame, in metres.
    """

    yaw: float
    """
    Rotation about the vertical axis, in radians.
    """


@dataclass
class CubeScene:
    """
    Tracy with its arms parked and cubes standing on its table.
    """

    world: World
    """
    The world holding Tracy and the cubes.
    """

    cubes: list[PlacedCube]
    """
    The cubes as placed.
    """

    table_height: float
    """
    Height of the table top in the world frame, in metres.
    """

    edge_length: float = 0.05
    """
    Edge length of every cube, in metres.
    """


def place_cubes_at_random(seed: int, minimum_distance: float = 0.1) -> list[PlacedCube]:
    """
    :param seed: Seed of the random poses.
    :param minimum_distance: The least distance between two cube centres, in metres.
    :return: A red, a yellow and a blue cube at random poses in the camera's view.
    """
    generator = np.random.default_rng(seed)
    colors = {
        SemanticColorName.RED: Color.from_rgba(red=0.8, green=0.1, blue=0.1),
        SemanticColorName.YELLOW: Color.from_rgba(red=0.95, green=0.8, blue=0.1),
        SemanticColorName.BLUE: Color.from_rgba(red=0.1, green=0.25, blue=0.8),
    }
    cubes = []
    while len(cubes) < len(colors):
        x, y = generator.uniform([0.45, -0.3], [0.8, 0.3])
        if any(
            math.dist((x, y), (cube.x, cube.y)) < minimum_distance for cube in cubes
        ):
            continue
        color_name = list(colors)[len(cubes)]
        cubes.append(
            PlacedCube(
                color_name=color_name,
                color=colors[color_name],
                x=float(x),
                y=float(y),
                yaw=float(generator.uniform(-math.pi, math.pi)),
            )
        )
    return cubes


@pytest.fixture(params=[0, 1, 2])
def cube_scene(request: pytest.FixtureRequest) -> CubeScene:
    """
    :return: Tracy with three cubes at random poses, the world RoboKudo runs in.
    """
    world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).create_world()
    tracy = Tracy.from_world(world)
    for arm in tracy.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    scene = CubeScene(
        world=world,
        cubes=place_cubes_at_random(request.param),
        table_height=tracy.table.top_z,
    )
    for cube in scene.cubes:
        body = Body(
            name=PrefixedName(f"{cube.color_name.name.lower()}_cube"),
            visual=ShapeCollection(
                [Box(scale=Scale(*[scene.edge_length] * 3), color=cube.color)]
            ),
            collision=ShapeCollection([Box(scale=Scale(*[scene.edge_length] * 3))]),
        )
        with world.modify_world():
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world, parent=world.root, child=body
                )
            )
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=cube.x,
            y=cube.y,
            z=scene.table_height + scene.edge_length / 2,
            yaw=cube.yaw,
            reference_frame=world.root,
        )
    robokudo.world.set_world(world)
    return scene


# %% what the pipeline reports


@dataclass
class PerceivedCube:
    """
    One cube as the pipeline reports it.
    """

    color_names: list[str]
    """
    The semantic colors annotated on its cluster.
    """

    position: np.ndarray
    """
    Its centre in the world frame, in metres.
    """

    yaw: float
    """
    Its rotation about the vertical axis, in radians.
    """

    @classmethod
    def from_hypothesis(cls, hypothesis: ObjectHypothesis, cas: CAS) -> PerceivedCube:
        """
        :param hypothesis: The cluster of one cube.
        :param cas: The CAS of the frame it was seen in.
        :return: The cube with its pose in the world frame.
        """
        [pose] = [
            annotation
            for annotation in hypothesis.annotations
            if type(annotation) is PoseAnnotation
        ]
        pose_in_world = transform_pose_from_camera_to_world(cas, pose)
        return cls(
            color_names=[
                annotation.color
                for annotation in hypothesis.annotations
                if isinstance(annotation, SemanticColor)
            ],
            position=np.asarray(pose_in_world.translation),
            yaw=float(Rotation.from_quat(pose_in_world.rotation).as_euler("xyz")[2]),
        )


def perceive_cubes(node) -> list[PerceivedCube]:
    """
    :param node: The ROS node the pipeline runs on.
    :return: The cubes the pipeline perceives in one frame of Tracy's camera.
    """
    pipeline = robokudo.pipeline.Pipeline("TracyCubesTestPipeline")
    pipeline.add_children(
        [ClearAnnotatorOutputs(), *AnalysisEngine().perception_annotators()]
    )
    result = robokudo.utils.tree_execution.run_tree_once(
        pipeline, node, max_iterations=80, tick_rate=20
    )
    assert result is py_trees.common.Status.SUCCESS
    return [
        PerceivedCube.from_hypothesis(hypothesis, pipeline.cas)
        for hypothesis in pipeline.cas.filter_annotations_by_type(ObjectHypothesis)
    ]


def yaw_difference_modulo_quarter_turn(first: float, second: float) -> float:
    """
    :return: How far apart two yaws of a cube are, which looks the same every quarter
        turn, in radians.
    """
    difference = (first - second) % (math.pi / 2)
    return min(difference, math.pi / 2 - difference)


# %% accuracy


def test_every_cube_is_perceived_in_its_color_within_a_centimetre(
    node, cube_scene: CubeScene
) -> None:
    perceived = perceive_cubes(node)

    for cube in cube_scene.cubes:
        [match] = [
            perceived_cube
            for perceived_cube in perceived
            if cube.color_name.name.lower() in perceived_cube.color_names
        ]
        true_position = np.array(
            [cube.x, cube.y, cube_scene.table_height + cube_scene.edge_length / 2]
        )
        assert np.linalg.norm(match.position - true_position) < 0.01


def test_every_cube_is_perceived_at_its_yaw_up_to_a_quarter_turn(
    node, cube_scene: CubeScene
) -> None:
    perceived = perceive_cubes(node)

    for cube in cube_scene.cubes:
        [match] = [
            perceived_cube
            for perceived_cube in perceived
            if cube.color_name.name.lower() in perceived_cube.color_names
        ]
        assert yaw_difference_modulo_quarter_turn(match.yaw, cube.yaw) < math.radians(5)


# %% answering queries in the process holding the world


@pytest.fixture
def query_service(cube_scene: CubeScene) -> QueryService:
    """
    :return: The cube localization answering queries on the scene's world.
    """
    service = QueryService(analysis_engine=AnalysisEngine(), world=cube_scene.world)
    service.start()
    yield service
    service.stop()


def ask_for_objects(node) -> list:
    """
    Send one query for any object to RoboKudo's query action, as a perception client
    does.

    :param node: The node the client is created on.
    :return: The object designators of the answer.
    """
    query_messages = pytest.importorskip("robokudo_msgs.action")
    client = ActionClient(node, query_messages.Query, "robokudo/query")
    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)
    assert client.wait_for_server(timeout_sec=10.0)
    goal_handle_future = client.send_goal_async(query_messages.Query.Goal())
    executor.spin_until_future_complete(goal_handle_future, timeout_sec=30.0)
    result_future = goal_handle_future.result().get_result_async()
    executor.spin_until_future_complete(result_future, timeout_sec=30.0)
    executor.remove_node(node)
    client.destroy()
    return list(result_future.result().result.res)


def test_a_query_is_answered_with_the_cubes_in_the_world_frame(
    node, cube_scene: CubeScene, query_service: QueryService
) -> None:
    designators = ask_for_objects(node)

    for cube in cube_scene.cubes:
        [match] = [
            designator
            for designator in designators
            if cube.color_name.name.lower() in designator.color
        ]
        [pose] = match.pose
        frame = cube_scene.world.get_kinematic_structure_entity_by_name(
            pose.header.frame_id
        )
        assert frame is cube_scene.world.root
        position = pose.pose.position
        assert math.dist((position.x, position.y), (cube.x, cube.y)) < 0.01
