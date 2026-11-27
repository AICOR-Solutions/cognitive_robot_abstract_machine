"""
Localizing the cubes in a frame recorded by Tracy's real camera, an Orbbec Femto Bolt,
over the polished metal table whose depth it measures poorly.
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

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.analysis_engines.tracy_cubes import (
    AnalysisEngine,
    TracyCubesSettings,
)
from robokudo.types.camera import CameraObservation
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

from .test_color_cluster_extractor import perceive, pipeline_seeing
from .test_tracy_cubes import PerceivedCube

# %% the recorded frame


@dataclass
class RecordedCamera:
    """
    The camera a frame was recorded with, as ``camera.json`` holds it.
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
    Position of the camera's optical frame in the world frame, in metres.
    """

    camera_orientation_in_world_xyzw: list[float]
    """
    Orientation of the camera's optical frame in the world frame, as a quaternion.
    """

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


@pytest.fixture
def three_cubes() -> Path:
    """
    :return: The folder of a frame of a red, a yellow and a blue cube in a row, saved at
        half resolution with the settings the real Tracy uses.
    """
    return (
        Path(__file__).parent
        / "dataset"
        / "tracy_cubes"
        / "three_cubes_on_the_metal_table"
    )


# %% the cubes on the metal table


def test_every_cube_on_the_metal_table_is_found_once_where_it_stands(
    three_cubes: Path,
) -> None:
    settings = TracyCubesSettings.from_file(three_cubes / "settings.yaml")
    annotators = [
        annotator
        for annotator in AnalysisEngine(settings=settings).perception_annotators()
        if not isinstance(annotator, CollectionReaderAnnotator)
    ]
    camera = RecordedCamera(**json.loads((three_cubes / "camera.json").read_text()))
    expected = yaml.safe_load((three_cubes / "cubes.yaml").read_text())

    cas = perceive(
        pipeline_seeing(
            annotators,
            camera.to_camera_observation(),
            cv2.imread(str(three_cubes / "color.png")),
            cv2.imread(str(three_cubes / "depth.png"), cv2.IMREAD_UNCHANGED),
        )
    )

    cubes = [
        PerceivedCube.from_hypothesis(hypothesis=hypothesis, cas=cas)
        for hypothesis in cas.filter_annotations_by_type(
            type_to_include=ObjectHypothesis
        )
    ]
    assert sorted(cube.color_names for cube in cubes) == sorted(
        [color] for color in expected
    )
    for cube in cubes:
        [color] = cube.color_names
        position = [expected[color]["x"], expected[color]["y"]]
        assert np.linalg.norm(cube.position[:2] - position) < 0.015
