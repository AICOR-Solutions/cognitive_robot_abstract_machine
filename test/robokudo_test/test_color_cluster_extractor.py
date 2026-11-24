"""
Finding objects by their color in the color image, each placed by the depth under it.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import IntEnum

import cv2
import numpy as np
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.annotators.color_cluster_extractor import ColorClusterExtractor
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import DepthNotMeasured
from robokudo.utils.error_handling import (
    clear_blackboard_exception,
    get_blackboard_exception,
)
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import SemanticColor
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

# %% a frame of colored squares


class Hue(IntEnum):
    """
    Hues of colored squares, on OpenCV's full scale (0 to 255).
    """

    RED = 5
    YELLOW = 40
    BLUE = 170
    RED_ABOVE_MAGENTA = 250
    """
    A red past the hue of magenta, where the hue circle closes.
    """

    BLUE_OF_TRACYS_CUBE = 147
    """
    The hue of the real blue cube on Tracy's table, which the evenly divided hue circle
    names cyan.
    """


@dataclass
class ColoredSquare:
    """
    A square of one color in the frame, at one depth.
    """

    hue: Hue
    """
    Its hue.
    """

    left: int
    """
    Its leftmost column.
    """

    top: int
    """
    Its topmost row.
    """

    size: int = 40
    """
    Its width and height, in pixels.
    """

    depth_millimetres: int = 900
    """
    How far it is from the camera; 0 where the camera measured nothing.
    """


@dataclass
class Frame:
    """
    A grey surface with colored squares on it, seen by a camera at the world origin
    looking along the world's z axis, so a point's height in the world is its depth.
    """

    squares: list[ColoredSquare] = field(default_factory=list)
    """
    The squares on the surface.
    """

    size: int = 200
    """
    Width and height of the frame, in pixels.
    """

    surface_depth_millimetres: int = 1000
    """
    How far the grey surface is from the camera.
    """

    def color_image(self) -> np.ndarray:
        """
        :return: The frame's color image, blue first.
        """
        hsv = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        hsv[...] = (0, 10, 150)
        for square in self.squares:
            hsv[
                square.top : square.top + square.size,
                square.left : square.left + square.size,
            ] = (square.hue, 220, 200)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

    def depth_image(self) -> np.ndarray:
        """
        :return: The frame's depth image, in millimetres.
        """
        depth = np.full(
            (self.size, self.size), self.surface_depth_millimetres, dtype=np.uint16
        )
        for square in self.squares:
            depth[
                square.top : square.top + square.size,
                square.left : square.left + square.size,
            ] = square.depth_millimetres
        return depth

    def camera_observation(self) -> CameraObservation:
        """
        :return: The camera at the world origin seeing the frame.
        """
        model = PinholeCameraModel(
            image_resolution=CameraResolution(width=self.size, height=self.size),
            focal_length_x=200.0,
            focal_length_y=200.0,
            principal_point=Point2(x=(self.size - 1) / 2, y=(self.size - 1) / 2),
        )
        camera_body = Body(name=PrefixedName(name="camera"))
        return CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"), root=camera_body, camera_model=model
            ),
            effective_camera_model=model,
            world_T_camera=HomogeneousTransformationMatrix(
                data=HomogeneousTransformationMatrix.from_xyz_quaternion(quat_w=1.0),
                reference_frame=Body(name=PrefixedName(name="world")),
                child_frame=camera_body,
            ),
            timestamp_nanoseconds=1,
        )


def pipeline_seeing(frame: Frame, annotators: list[BaseAnnotator]) -> Pipeline:
    """
    :param frame: The frame seen.
    :param annotators: The annotators that run on the frame after its point cloud is
        built.
    :return: A pipeline of the annotators, behind the one building the point cloud, its
        CAS holding the frame.
    """
    preprocessor = ImagePreprocessorAnnotator()
    pipeline = Pipeline("ColorClusterExtractorTest")
    pipeline.add_children([preprocessor, *annotators])
    outputs = AnnotatorOutputPerPipelineMap()
    outputs.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", outputs)
    cas = preprocessor.get_cas()
    cas.camera_observation = frame.camera_observation()
    cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
    cas.set(CASViews.COLOR_IMAGE, frame.color_image())
    cas.set(CASViews.DEPTH_IMAGE, frame.depth_image())
    return pipeline


def perceive(frame: Frame, annotators: list[BaseAnnotator]) -> CAS:
    """
    :param frame: The frame seen.
    :param annotators: The annotators run on the frame after its point cloud is built.
    :return: The CAS of the frame once every annotator ran.
    """
    pipeline = pipeline_seeing(frame, annotators)
    for annotator in pipeline.get_annotators():
        assert annotator.update() is Status.SUCCESS
    return pipeline.cas


def extractor_below_the_surface() -> ColorClusterExtractor:
    """
    :return: An extractor keeping points up to 0.95 m high, so the surface itself is
        never an object.
    """
    extractor = ColorClusterExtractor()
    extractor.descriptor.parameters.highest_height = 0.95
    return extractor


def extract(
    frame: Frame, extractor: ColorClusterExtractor | None = None
) -> list[ObjectHypothesis]:
    """
    :param frame: The frame seen.
    :param extractor: The extractor finding the objects; by default
        :func:`extractor_below_the_surface`.
    :return: The objects the extractor found in the frame.
    """
    extractor = extractor or extractor_below_the_surface()
    return perceive(frame, [extractor]).filter_annotations_by_type(ObjectHypothesis)


def colors_of(hypothesis: ObjectHypothesis) -> list[str]:
    """
    :return: The semantic colors annotated on the hypothesis.
    """
    return [
        annotation.color
        for annotation in hypothesis.annotations
        if isinstance(annotation, SemanticColor)
    ]


# %% finding the objects


def test_every_area_of_one_color_is_an_object_named_by_that_color() -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(Hue.BLUE, left=120, top=120),
        ]
    )

    hypotheses = extract(frame)

    assert sorted(colors_of(hypothesis) for hypothesis in hypotheses) == [
        ["blue"],
        ["red"],
    ]


def test_two_areas_of_one_color_are_two_objects() -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(Hue.RED, left=120, top=120),
        ]
    )

    hypotheses = extract(frame)

    assert [colors_of(hypothesis) for hypothesis in hypotheses] == [["red"], ["red"]]


def test_an_object_holds_the_points_under_its_area_where_the_camera_measured_them() -> (
    None
):
    [hypothesis] = extract(Frame([ColoredSquare(Hue.RED, left=80, top=80)]))

    points = np.asarray(hypothesis.points.points)
    assert len(points) > 100
    assert np.allclose(points[:, 2], 0.9)
    assert np.all(np.abs(points[:, :2]) < 0.1)


def test_an_object_carries_the_region_of_its_area() -> None:
    [hypothesis] = extract(Frame([ColoredSquare(Hue.RED, left=30, top=60)]))

    region = hypothesis.roi.roi
    assert (region.pos.x, region.pos.y, region.width, region.height) == (30, 60, 40, 40)


def test_a_hue_above_magentas_bound_is_red_again() -> None:
    [hypothesis] = extract(
        Frame([ColoredSquare(Hue.RED_ABOVE_MAGENTA, left=80, top=80)])
    )

    assert colors_of(hypothesis) == ["red"]


def test_the_hue_bounds_decide_the_color() -> None:
    extractor = extractor_below_the_surface()
    extractor.descriptor.parameters.hue_upper_bounds = {
        **extractor.descriptor.parameters.hue_upper_bounds,
        Color.CYAN: 130,
    }

    [hypothesis] = extract(
        Frame([ColoredSquare(Hue.BLUE_OF_TRACYS_CUBE, left=80, top=80)]), extractor
    )

    assert colors_of(hypothesis) == ["blue"]


def test_the_default_hue_bounds_and_thresholds_are_those_the_colors_are_named_by() -> (
    None
):
    parameters = ColorClusterExtractor.Descriptor.Parameters()
    color_parameters = ClusterColorAnnotator.Descriptor.Parameters()

    assert parameters.hue_upper_bounds == color_parameters.chromatic_hue_upper_bounds()
    assert parameters.minimum_saturation == color_parameters.min_saturation_color
    assert parameters.minimum_value == color_parameters.min_value_color


# %% what is no object


def test_an_area_outside_the_heights_is_no_object() -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(Hue.YELLOW, left=120, top=120, depth_millimetres=990),
        ]
    )

    hypotheses = extract(frame)

    assert [colors_of(hypothesis) for hypothesis in hypotheses] == [["red"]]


def test_an_area_whose_depth_the_camera_did_not_measure_is_no_object() -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(Hue.BLUE, left=120, top=120, depth_millimetres=0),
        ]
    )

    hypotheses = extract(frame)

    assert [colors_of(hypothesis) for hypothesis in hypotheses] == [["red"]]


def test_a_speck_of_color_is_no_object() -> None:
    frame = Frame([ColoredSquare(Hue.RED, left=80, top=80, size=3)])

    assert extract(frame) == []


def test_an_area_with_too_few_points_is_no_object() -> None:
    extractor = extractor_below_the_surface()
    extractor.descriptor.parameters.minimum_points = 10_000

    assert extract(Frame([ColoredSquare(Hue.RED, left=80, top=80)]), extractor) == []


def test_a_grey_frame_has_no_objects() -> None:
    assert extract(Frame()) == []


def test_a_frame_whose_depth_image_measured_nothing_is_refused() -> None:
    """
    A camera that measured no depth sees no object, which is not the same as there
    being none.
    """
    frame = Frame(
        [ColoredSquare(Hue.RED, left=80, top=80, depth_millimetres=0)],
        surface_depth_millimetres=0,
    )

    preprocessor, extractor = pipeline_seeing(
        frame, [extractor_below_the_surface()]
    ).get_annotators()
    preprocessor.update()

    status = extractor.update()
    error = get_blackboard_exception()
    clear_blackboard_exception()

    assert status is Status.FAILURE
    assert type(error) is DepthNotMeasured


# %% what it drew


def test_it_draws_the_outline_of_every_object_in_its_color() -> None:
    extractor = extractor_below_the_surface()

    extract(Frame([ColoredSquare(Hue.BLUE, left=80, top=80)]), extractor)

    drawn = extractor.get_annotator_output_struct().image
    blue = ClusterColorAnnotator.color_name_to_bgr_values[Color.BLUE]
    assert np.any(np.all(drawn == blue, axis=2))
