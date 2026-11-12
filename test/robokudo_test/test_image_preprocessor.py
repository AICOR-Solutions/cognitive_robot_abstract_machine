"""
Building the point cloud of a camera's color and depth images.
"""

from __future__ import annotations

# %% Imports
import numpy as np
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.cas import CASViews
from robokudo.exceptions import DepthImageNotRegistered
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ColorToDepthRatio
from robokudo.utils.error_handling import (
    clear_blackboard_exception,
    get_blackboard_exception,
)
from robokudo.utils.query import QueryHandler

from .test_utils_annotator_helper import _camera_observation

# %% Unregistered depth


def queued_feedback() -> list[str]:
    """
    :return: The feedback texts waiting to be sent to the client of a query, taken out
        of the queue.
    """
    queue = QueryHandler.get_feedback_queue()
    texts = []
    while not queue.empty():
        texts.append(queue.get_nowait().feedback)
    return texts


def preprocessor_seeing(
    color_shape: tuple[int, int], depth_shape: tuple[int, int]
) -> ImagePreprocessorAnnotator:
    """
    :param color_shape: Height and width of the color image in the CAS.
    :param depth_shape: Height and width of the depth image in the CAS.
    :return: An image preprocessor in a pipeline of its own, its CAS holding one frame
        with images of these sizes.
    """
    annotator = ImagePreprocessorAnnotator()
    pipeline = Pipeline("Sequence")
    pipeline.add_child(annotator)
    outputs = AnnotatorOutputPerPipelineMap()
    outputs.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", outputs)
    see_frame(annotator, color_shape, depth_shape)
    return annotator


def see_frame(
    annotator: ImagePreprocessorAnnotator,
    color_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> None:
    """
    Put a frame of the camera into the annotator's CAS.

    :param annotator: The image preprocessor whose CAS gets the frame.
    :param color_shape: Height and width of the color image.
    :param depth_shape: Height and width of the depth image.
    """
    cas = annotator.get_cas()
    cas.camera_observation = _camera_observation(
        HomogeneousTransformationMatrix.from_xyz_quaternion(quat_w=1.0)
    )
    cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
    cas.set(CASViews.COLOR_IMAGE, np.zeros((*color_shape, 3), dtype=np.uint8))
    cas.set(CASViews.DEPTH_IMAGE, np.ones(depth_shape, dtype=np.uint16))


def test_a_depth_image_not_registered_to_the_color_image_is_reported_to_the_query() -> (
    None
):
    """
    The client of a query that cannot be answered learns that the depth image has to be
    registered to the color image.
    """
    annotator = preprocessor_seeing(color_shape=(20, 30), depth_shape=(10, 15))
    queued_feedback()

    annotator.update()

    [feedback] = queued_feedback()
    assert "depth_registration:=true" in feedback
    clear_blackboard_exception()


def test_a_depth_image_not_registered_fails_the_frame_and_aborts_its_query() -> None:
    """
    An unregistered depth image fails the frame instead of raising out of the tick,
    which would stop the pipeline from answering any later query, and leaves the error
    on the blackboard, where the query server aborts the query with it.
    """
    annotator = preprocessor_seeing(color_shape=(20, 30), depth_shape=(10, 15))

    status = annotator.update()

    assert status is Status.FAILURE
    error = get_blackboard_exception()
    assert isinstance(error, DepthImageNotRegistered)
    assert "depth_registration:=true" in str(error)
    clear_blackboard_exception()


def test_a_registered_frame_after_an_unregistered_one_is_turned_into_a_point_cloud() -> (
    None
):
    annotator = preprocessor_seeing(color_shape=(20, 30), depth_shape=(10, 15))
    annotator.update()
    clear_blackboard_exception()
    see_frame(annotator, color_shape=(20, 30), depth_shape=(20, 30))

    status = annotator.update()

    assert status is Status.SUCCESS
    assert annotator.get_cas().get(CASViews.CLOUD) is not None
