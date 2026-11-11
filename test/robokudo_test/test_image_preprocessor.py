"""
Building the point cloud of a camera's color and depth images.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.blackboard import Blackboard

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ColorToDepthRatio
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


def test_a_depth_image_not_registered_to_the_color_image_is_reported_to_the_query() -> (
    None
):
    """
    The client of a query that cannot be answered learns that the depth image has to be
    registered to the color image.
    """
    annotator = ImagePreprocessorAnnotator()
    pipeline = Pipeline("Sequence")
    pipeline.add_child(annotator)
    outputs = AnnotatorOutputPerPipelineMap()
    outputs.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", outputs)
    cas = annotator.get_cas()
    cas.camera_observation = _camera_observation(
        HomogeneousTransformationMatrix.from_xyz_quaternion(quat_w=1.0)
    )
    cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
    cas.set(CASViews.COLOR_IMAGE, np.zeros((20, 30, 3), dtype=np.uint8))
    cas.set(CASViews.DEPTH_IMAGE, np.zeros((10, 15), dtype=np.uint16))
    queued_feedback()

    with pytest.raises(RuntimeError):
        annotator.update()

    [feedback] = queued_feedback()
    assert "depth_registration:=true" in feedback
