"""
Placing an object's box around its points, in the world frame.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.types.annotation import BoundingBox3DAnnotation
from robokudo.types.scene import ObjectHypothesis

from .test_color_cluster_extractor import (
    ColoredSquare,
    Frame,
    Hue,
    extractor_below_the_surface,
    perceive,
)

# %% objects standing on a surface


@pytest.mark.parametrize("support_height, bottom_height", [(0.85, 0.85), (None, 0.9)])
def test_a_box_reaches_down_to_the_surface_its_object_stands_on(
    support_height: float | None, bottom_height: float
) -> None:
    """
    The square's points are 0.9 m high; without a surface the box spans them only.
    """
    pose_annotator = ClusterPoseBBAnnotator()
    pose_annotator.descriptor.parameters.support_height = support_height
    frame = Frame([ColoredSquare(Hue.RED, left=80, top=80, depth_millimetres=900)])

    cas = perceive(
        frame.create_pipeline([extractor_below_the_surface(), pose_annotator])
    )

    [hypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
    [box] = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, BoundingBox3DAnnotation)
    ]
    assert np.isclose(box.z_length, 0.9 - bottom_height)
    assert np.isclose(box.pose.translation[2], (0.9 + bottom_height) / 2)
