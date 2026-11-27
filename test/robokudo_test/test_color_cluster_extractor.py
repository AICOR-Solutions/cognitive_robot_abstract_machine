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
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import DepthNotMeasured
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import SemanticColor
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import (
    clear_blackboard_exception,
    get_blackboard_exception,
)
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

# %% running annotators on a frame


def pipeline_seeing(
    annotators: list[BaseAnnotator],
    camera_observation: CameraObservation,
    color_image: np.ndarray,
    depth_image: np.ndarray,
) -> Pipeline:
    """
    :return: A pipeline of the annotators whose CAS holds the frame, as the collection
        reader leaves it.
    """
    pipeline = Pipeline("PerceptionTest")
    pipeline.add_children(annotators)
    outputs = AnnotatorOutputPerPipelineMap()
    outputs.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", outputs)
    cas = pipeline.cas
    cas.camera_observation = camera_observation
    cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
    cas.set(view_name=CASViews.COLOR_IMAGE, value=color_image)
    cas.set(view_name=CASViews.DEPTH_IMAGE, value=depth_image)
    return pipeline


def perceive(pipeline: Pipeline) -> CAS:
    """
    :return: The pipeline's CAS once every annotator ran successfully.
    """
    for annotator in pipeline.get_annotators():
        assert annotator.update() is Status.SUCCESS
    return pipeline.cas


def colors_of(hypothesis: ObjectHypothesis) -> list[str]:
    """
    :return: The semantic colors annotated on the hypothesis.
    """
    return [
        annotation.color
        for annotation in hypothesis.annotations
        if isinstance(annotation, SemanticColor)
    ]


# %% a frame of colored squares


class Hue(IntEnum):
    """
    Hues of colored squares, on OpenCV's full scale (0 to 255).
    """

    RED = 5
    YELLOW = 40
    BLUE = 170
    RED_ABOVE_MAGENTA = 250
    BLUE_OF_TRACYS_CUBE = 147
    """
    The real blue cube's hue, which the evenly divided hue circle names cyan.
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
    looking along the world's z axis, so a point's height is its depth.
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

    def create_pipeline(self, annotators: list[BaseAnnotator]) -> Pipeline:
        """
        :return: A pipeline seeing the frame, of an image preprocessor and the annotators.
        """
        hsv = np.full((self.size, self.size, 3), (0, 10, 150), dtype=np.uint8)
        depth = np.full(
            (self.size, self.size), self.surface_depth_millimetres, dtype=np.uint16
        )
        for square in self.squares:
            rows = slice(square.top, square.top + square.size)
            columns = slice(square.left, square.left + square.size)
            hsv[rows, columns] = (square.hue, 220, 200)
            depth[rows, columns] = square.depth_millimetres
        model = PinholeCameraModel(
            image_resolution=CameraResolution(width=self.size, height=self.size),
            focal_length_x=200.0,
            focal_length_y=200.0,
            principal_point=Point2(x=(self.size - 1) / 2, y=(self.size - 1) / 2),
        )
        camera_body = Body(name=PrefixedName(name="camera"))
        camera_observation = CameraObservation(
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
        return pipeline_seeing(
            [ImagePreprocessorAnnotator(), *annotators],
            camera_observation,
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL),
            depth,
        )


def extractor_below_the_surface() -> ColorClusterExtractor:
    """
    :return: An extractor keeping points up to 0.95 m high, so never the surface.
    """
    extractor = ColorClusterExtractor()
    extractor.descriptor.parameters.highest_height = 0.95
    return extractor


def extract(
    frame: Frame, extractor: ColorClusterExtractor | None = None
) -> list[ObjectHypothesis]:
    """
    :return: The objects the extractor, by default :func:`extractor_below_the_surface`,
        finds in the frame.
    """
    extractor = extractor or extractor_below_the_surface()
    cas = perceive(frame.create_pipeline([extractor]))
    return cas.filter_annotations_by_type(type_to_include=ObjectHypothesis)


# %% finding the objects


def test_every_area_of_one_color_is_an_object_named_by_that_color() -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(Hue.RED, left=120, top=20),
            ColoredSquare(Hue.BLUE, left=120, top=120),
        ]
    )

    hypotheses = extract(frame)

    assert sorted(colors_of(hypothesis) for hypothesis in hypotheses) == [
        ["blue"],
        ["red"],
        ["red"],
    ]


def test_an_object_holds_the_points_and_the_region_of_its_area() -> None:
    [hypothesis] = extract(Frame([ColoredSquare(Hue.RED, left=80, top=80)]))

    points = np.asarray(hypothesis.points.points)
    assert len(points) > 100
    assert np.allclose(points[:, 2], 0.9)
    assert np.all(np.abs(points[:, :2]) < 0.1)
    region = hypothesis.roi.roi
    assert (region.pos.x, region.pos.y, region.width, region.height) == (80, 80, 40, 40)


@pytest.mark.parametrize(
    "hue, hue_upper_bounds, color",
    [
        (Hue.RED_ABOVE_MAGENTA, {}, "red"),
        (Hue.BLUE_OF_TRACYS_CUBE, {Color.CYAN: 130}, "blue"),
    ],
)
def test_the_hue_bounds_decide_the_color(
    hue: Hue, hue_upper_bounds: dict[Color, float], color: str
) -> None:
    extractor = extractor_below_the_surface()
    extractor.descriptor.parameters.hue_upper_bounds = {
        **extractor.descriptor.parameters.hue_upper_bounds,
        **hue_upper_bounds,
    }

    [hypothesis] = extract(Frame([ColoredSquare(hue, left=80, top=80)]), extractor)

    assert colors_of(hypothesis) == [color]


# %% what is no object


@pytest.mark.parametrize("depth_millimetres", [990, 0], ids=["too_high", "unmeasured"])
def test_an_area_outside_the_heights_or_unmeasured_is_no_object(
    depth_millimetres: int,
) -> None:
    frame = Frame(
        [
            ColoredSquare(Hue.RED, left=20, top=20),
            ColoredSquare(
                Hue.YELLOW, left=120, top=120, depth_millimetres=depth_millimetres
            ),
        ]
    )

    hypotheses = extract(frame)

    assert [colors_of(hypothesis) for hypothesis in hypotheses] == [["red"]]


def test_a_speck_of_color_is_no_object() -> None:
    assert extract(Frame([ColoredSquare(Hue.RED, left=80, top=80, size=3)])) == []


def test_an_area_with_too_few_points_is_no_object() -> None:
    extractor = extractor_below_the_surface()
    extractor.descriptor.parameters.minimum_points = 10_000

    assert extract(Frame([ColoredSquare(Hue.RED, left=80, top=80)]), extractor) == []


def test_a_frame_whose_depth_image_measured_nothing_is_refused() -> None:
    frame = Frame(
        [ColoredSquare(Hue.RED, left=80, top=80, depth_millimetres=0)],
        surface_depth_millimetres=0,
    )
    preprocessor, extractor = frame.create_pipeline(
        [extractor_below_the_surface()]
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
