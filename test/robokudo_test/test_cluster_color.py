"""
Naming the color of a cluster by the hue range each chromatic color covers.
"""

from __future__ import annotations

# %% Imports
import numpy as np

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color

# %% hue ranges


def count_colors_of_one_pixel(
    annotator: ClusterColorAnnotator, hue: int
) -> dict[Color, int]:
    """
    :param annotator: The annotator counting the colors.
    :param hue: The hue of a fully saturated, bright pixel on OpenCV's full scale.
    :return: How many pixels the annotator counts in each color.
    """
    hsv_image = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    mask = np.array([[255]], dtype=np.uint8)
    _, color_count = annotator.count_colors_numba(hsv_image, mask)
    return color_count


def test_the_hue_circle_is_divided_evenly_by_default() -> None:
    annotator = ClusterColorAnnotator()
    color_range = annotator.descriptor.parameters.color_range

    assert annotator.color_hue_positions[Color.RED] == color_range / 2 + 0.5
    assert annotator.color_hue_positions[Color.BLUE] == 4.5 * color_range + 0.5


def test_a_hue_is_named_by_the_configured_upper_bounds() -> None:
    descriptor = ClusterColorAnnotator.Descriptor()
    descriptor.parameters.hue_upper_bounds = {
        Color.RED: 30,
        Color.YELLOW: 64,
        Color.GREEN: 107,
        Color.CYAN: 149,
        Color.BLUE: 192,
        Color.MAGENTA: 235,
    }
    annotator = ClusterColorAnnotator(descriptor=descriptor)

    assert count_colors_of_one_pixel(annotator, hue=25)[Color.RED] == 1
    assert count_colors_of_one_pixel(annotator, hue=240)[Color.RED] == 1
    assert count_colors_of_one_pixel(annotator, hue=180)[Color.BLUE] == 1
