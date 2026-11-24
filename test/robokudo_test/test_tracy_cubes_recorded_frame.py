"""
Localizing the cubes on Tracy's table in a frame recorded by Tracy's real camera, an
Orbbec Femto Bolt, over the polished metal table whose depth it measures poorly.
"""

from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from typing_extensions import Any, Self

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.analysis_engines.tracy_cubes import (
    AnalysisEngine,
    TracyCubesSettings,
)
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import PoseAnnotation, SemanticColor
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

# %% the recorded frame


@dataclass
class RecordedCamera:
    """
    The camera a frame was recorded with: its pinhole model, at the frame's resolution,
    and its pose in Tracy's world frame.
    """

    width: int
    """
    Width of the frame, in pixels.
    """

    height: int
    """
    Height of the frame, in pixels.
    """

    focal_length_x: float
    """
    Horizontal focal length, in pixels.
    """

    focal_length_y: float
    """
    Vertical focal length, in pixels.
    """

    principal_point_x: float
    """
    Column of the principal point.
    """

    principal_point_y: float
    """
    Row of the principal point.
    """

    camera_position_in_world: list[float]
    """
    Where the camera's optical frame is in the world frame, in metres.
    """

    camera_orientation_in_world_xyzw: list[float]
    """
    How the camera's optical frame is turned in the world frame, as a quaternion.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """
        :param data: The camera's fields by name.
        :return: The camera written in the data.
        """
        return cls(**data)

    def to_camera_observation(self) -> CameraObservation:
        """
        :return: The camera seeing the frame from its pose in the world frame.
        """
        model = PinholeCameraModel(
            image_resolution=CameraResolution(width=self.width, height=self.height),
            focal_length_x=self.focal_length_x,
            focal_length_y=self.focal_length_y,
            principal_point=Point2(x=self.principal_point_x, y=self.principal_point_y),
        )
        camera_body = Body(name=PrefixedName(name="camera_color_optical_frame"))
        x, y, z = self.camera_position_in_world
        quat_x, quat_y, quat_z, quat_w = self.camera_orientation_in_world_xyzw
        return CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"), root=camera_body, camera_model=model
            ),
            effective_camera_model=model,
            world_T_camera=HomogeneousTransformationMatrix(
                data=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=x,
                    pos_y=y,
                    pos_z=z,
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w,
                ),
                reference_frame=Body(name=PrefixedName(name="map")),
                child_frame=camera_body,
            ),
            timestamp_nanoseconds=1,
        )


@dataclass
class RecordedFrame:
    """
    A frame of Tracy's real camera, saved at half its resolution: the color image, the
    depth image registered to it, the camera and the settings it is perceived with.
    """

    folder: Path
    """
    The folder the frame is saved in.
    """

    def settings(self) -> TracyCubesSettings:
        """
        :return: The settings the real Tracy perceives its cubes with.
        """
        return TracyCubesSettings.from_file(self.folder / "settings.yaml")

    def cube_positions(self) -> dict[str, np.ndarray]:
        """
        :return: Per cube color, where the cube stands on the table, in the world frame.
        """
        cubes = yaml.safe_load((self.folder / "cubes.yaml").read_text())
        return {
            color: np.array([position["x"], position["y"]])
            for color, position in cubes.items()
        }

    def fill(self, cas: CAS) -> None:
        """
        Put the frame into a CAS, as the collection reader does with a camera's frame.

        :param cas: The CAS the frame goes into.
        """
        camera = RecordedCamera.from_json(
            json.loads((self.folder / "camera.json").read_text())
        )
        cas.camera_observation = camera.to_camera_observation()
        cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
        cas.set(CASViews.COLOR_IMAGE, cv2.imread(str(self.folder / "color.png")))
        cas.set(
            CASViews.DEPTH_IMAGE,
            cv2.imread(str(self.folder / "depth.png"), cv2.IMREAD_UNCHANGED),
        )


@pytest.fixture
def three_cubes() -> RecordedFrame:
    """
    :return: A red, a yellow and a blue cube in a row across Tracy's metal table.
    """
    return RecordedFrame(
        Path(__file__).parent
        / "dataset"
        / "tracy_cubes"
        / "three_cubes_on_the_metal_table"
    )


# %% what the pipeline reports


@dataclass
class LocalizedCube:
    """
    One cube as the pipeline reports it.
    """

    color_names: list[str]
    """
    The semantic colors annotated on it.
    """

    position: np.ndarray
    """
    Its centre in the world frame, in metres.
    """


def localize_cubes(frame: RecordedFrame) -> list[LocalizedCube]:
    """
    :param frame: A recorded frame.
    :return: The cubes the pipeline localizes in it.
    """
    annotators = [
        annotator
        for annotator in AnalysisEngine(
            settings=frame.settings()
        ).perception_annotators()
        if not isinstance(annotator, CollectionReaderAnnotator)
    ]
    pipeline = Pipeline("TracyCubesRecordedFrame")
    pipeline.add_children(annotators)
    outputs = AnnotatorOutputPerPipelineMap()
    outputs.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", outputs)
    cas = pipeline.cas
    frame.fill(cas)
    for annotator in annotators:
        assert annotator.update() is Status.SUCCESS
    cubes = []
    for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
        [pose] = [
            annotation
            for annotation in hypothesis.annotations
            if type(annotation) is PoseAnnotation
        ]
        cubes.append(
            LocalizedCube(
                color_names=[
                    annotation.color
                    for annotation in hypothesis.annotations
                    if isinstance(annotation, SemanticColor)
                ],
                position=np.asarray(
                    transform_pose_from_camera_to_world(cas, pose).translation
                ),
            )
        )
    return cubes


# %% the cubes on the metal table


def test_every_cube_on_the_metal_table_is_found_once_in_its_color(
    three_cubes: RecordedFrame,
) -> None:
    cubes = localize_cubes(three_cubes)

    assert sorted(cube.color_names for cube in cubes) == sorted(
        [color] for color in three_cubes.cube_positions()
    )


def test_every_cube_on_the_metal_table_is_found_within_one_and_a_half_centimetres_of_where_it_stands(
    three_cubes: RecordedFrame,
) -> None:
    cubes = localize_cubes(three_cubes)

    for color, position in three_cubes.cube_positions().items():
        [cube] = [cube for cube in cubes if cube.color_names == [color]]
        assert np.linalg.norm(cube.position[:2] - position) < 0.015
