"""
Placing an object's box around its points, in the world frame.
"""

from __future__ import annotations

# %% Imports
import numpy as np

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


def box_of_a_square_seen_at(
    depth_millimetres: int, pose_annotator: ClusterPoseBBAnnotator
) -> BoundingBox3DAnnotation:
    """
    :param depth_millimetres: How far the square is from the camera, so how high it is
        in the world.
    :param pose_annotator: The annotator placing the box.
    :return: The box placed around the square's points.
    """
    cas = perceive(
        Frame(
            [
                ColoredSquare(
                    Hue.RED, left=80, top=80, depth_millimetres=depth_millimetres
                )
            ]
        ),
        [extractor_below_the_surface(), pose_annotator],
    )
    [hypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
    [box] = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, BoundingBox3DAnnotation)
    ]
    return box


def test_a_box_reaches_down_to_the_surface_its_object_stands_on() -> None:
    pose_annotator = ClusterPoseBBAnnotator()
    pose_annotator.descriptor.parameters.support_height = 0.85

    box = box_of_a_square_seen_at(900, pose_annotator)

    assert np.isclose(box.z_length, 0.9 - 0.85)
    assert np.isclose(box.pose.translation[2], (0.9 + 0.85) / 2)


def test_without_a_surface_a_box_spans_its_points_only() -> None:
    box = box_of_a_square_seen_at(900, ClusterPoseBBAnnotator())

    assert np.isclose(box.z_length, 0.0)
    assert np.isclose(box.pose.translation[2], 0.9)
