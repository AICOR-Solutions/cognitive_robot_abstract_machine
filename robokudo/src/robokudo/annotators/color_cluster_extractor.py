"""
Finding objects by their color in the color image, each placed by the depth under it.
"""

from __future__ import annotations

# %% Imports
import math
from dataclasses import dataclass, field
from timeit import default_timer

import cv2
import numpy as np
import open3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import DepthNotMeasured
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.types.annotation import SemanticColor
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import get_camera_to_world_transform_matrix
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy.typing as npt


# %% areas of one color


@dataclass
class ColorArea:
    """
    A connected area of pixels of one chromatic color and the points under it.
    """

    color: Color
    """
    The color of every pixel of the area.
    """

    outline: npt.NDArray
    """
    The area's outer contour, in the depth image's pixels.
    """

    points: open3d.geometry.PointCloud = field(
        default_factory=open3d.geometry.PointCloud
    )
    """
    The points under the area between the heights, in the camera frame.
    """


# %% annotator


class ColorClusterExtractor(BaseAnnotator):
    """
    Turns every connected area of one chromatic color into an object of that color,
    holding the points under it that lie between two heights in the world frame.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of how pixels are named by color and which areas become objects.
        """

        class Parameters:
            """
            Parameters of the color areas and of the points kept under them.
            """

            def __init__(self) -> None:
                color_parameters = ClusterColorAnnotator.Descriptor.Parameters()
                self.hue_upper_bounds: dict[Color, float] = (
                    color_parameters.chromatic_hue_upper_bounds()
                )
                """
                Per chromatic color, the hue (0 to 255) a pixel lies below to take it.
                """
                self.minimum_saturation: int = color_parameters.min_saturation_color
                """
                The saturation a pixel must exceed to take a color.
                """
                self.minimum_value: int = color_parameters.min_value_color
                """
                The brightness a pixel must exceed to take a color.
                """
                self.speck_size: int = 5
                """
                Width and height, in pixels, of the specks of a color left out.
                """
                self.edge_width: int = 4
                """
                Pixels left out at an area's edge, whose depth may be the background's.
                """
                self.lowest_height: float = -math.inf
                """
                World height, in metres, below which no point of an object lies.
                """
                self.highest_height: float = math.inf
                """
                World height, in metres, above which no point of an object lies.
                """
                self.minimum_points: int = 100
                """
                The fewest points between the heights an area needs to become an object.
                """
                self.outlier_neighbors: int = 20
                """
                How many neighbors a point's distance is averaged over to find outliers.
                """
                self.outlier_standard_deviations: float = 2.0
                """
                How far above the mean neighbor distance a point may lie, in deviations.
                """
                self.depth_truncation: float = 9.0
                """
                Distance from the camera, in metres, beyond which depth is not read.
                """

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ColorClusterExtractor",
        descriptor: ColorClusterExtractor.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        start = default_timer()
        cas = self.get_cas()
        color_image = get_scaled_color_image_for_depth_image(
            cas, cas.get(view_name=CASViews.COLOR_IMAGE)
        )
        depth_image = cas.get(view_name=CASViews.DEPTH_IMAGE)
        if not depth_image.any():
            raise DepthNotMeasured(context=self.name)
        areas = self.color_areas(color_image)
        drawing = color_image.copy()
        objects = 0
        for area in areas:
            area.points = self.points_under(area, color_image, depth_image)
            if len(area.points.points) < self.descriptor.parameters.minimum_points:
                self.draw_area(drawing, area, Color.GREY, 1)
                continue
            hypothesis = ObjectHypothesis()
            hypothesis.id = str(objects)
            hypothesis.source = self.name
            hypothesis.points = area.points
            left, top, width, height = cv2.boundingRect(area.outline)
            hypothesis.roi.roi.pos.x, hypothesis.roi.roi.pos.y = left, top
            hypothesis.roi.roi.width, hypothesis.roi.roi.height = width, height
            hypothesis.roi.mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(
                hypothesis.roi.mask,
                [area.outline],
                -1,
                255,
                cv2.FILLED,
                offset=(-left, -top),
            )
            color = SemanticColor()
            color.color = area.color.name.lower()
            color.source = self.name
            color.ratio = 1.0
            hypothesis.annotations.append(color)
            cas.annotations.append(hypothesis)
            objects += 1
            self.draw_area(drawing, area, area.color, 3)
        self.get_annotator_output_struct().set_image(img=drawing)
        self.feedback_message = (
            f"{objects} objects of {len(areas)} color areas in "
            f"{default_timer() - start:.3f}s"
        )
        return Status.SUCCESS

    def color_areas(self, color_image: npt.NDArray) -> list[ColorArea]:
        """
        :param color_image: A color image, blue first.
        :return: The connected areas of one chromatic color in it, specks left out.
        """
        parameters = self.descriptor.parameters
        colors = sorted(parameters.hue_upper_bounds, key=lambda color: color.value)
        hue, saturation, value = cv2.split(
            cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV_FULL)
        )
        bounds = [parameters.hue_upper_bounds[color] for color in colors]
        named = np.searchsorted(bounds, hue, side="right") % len(bounds)
        named[
            (saturation <= parameters.minimum_saturation)
            | (value <= parameters.minimum_value)
        ] = -1
        speck = np.ones((parameters.speck_size, parameters.speck_size), np.uint8)
        areas = []
        for index, color in enumerate(colors):
            mask = cv2.morphologyEx(
                (named == index).astype(np.uint8) * 255, cv2.MORPH_OPEN, speck
            )
            outlines, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            areas.extend(ColorArea(color, outline) for outline in outlines)
        return areas

    def points_under(
        self, area: ColorArea, color_image: npt.NDArray, depth_image: npt.NDArray
    ) -> open3d.geometry.PointCloud:
        """
        :param color_image: The color image, blue first, registered to the depth image.
        :return: The points under the area, in the camera frame, without its edge, the
            outliers and the points outside the heights.
        """
        parameters = self.descriptor.parameters
        mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [area.outline], -1, 255, cv2.FILLED)
        mask = cv2.erode(mask, np.ones((2 * parameters.edge_width + 1,) * 2, np.uint8))
        if np.count_nonzero(mask) < parameters.minimum_points:
            return open3d.geometry.PointCloud()
        masked_depth = np.where(mask == 255, depth_image, 0)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)),
            open3d.geometry.Image(
                np.ascontiguousarray(
                    masked_depth, dtype=depth_image.dtype.newbyteorder("=")
                )
            ),
            convert_rgb_to_intensity=False,
            depth_trunc=parameters.depth_truncation,
        )
        points = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            Open3DCameraModelAdapter.to_intrinsic(
                camera_model=self.get_cas().pointcloud_camera_model
            ),
        )
        if len(points.points) < parameters.minimum_points:
            return points
        points, _ = points.remove_statistical_outlier(
            nb_neighbors=parameters.outlier_neighbors,
            std_ratio=parameters.outlier_standard_deviations,
        )
        camera_to_world = get_camera_to_world_transform_matrix(self.get_cas())
        heights = (
            np.asarray(points.points) @ camera_to_world[2, :3] + camera_to_world[2, 3]
        )
        return points.select_by_index(
            np.flatnonzero(
                (heights >= parameters.lowest_height)
                & (heights <= parameters.highest_height)
            )
        )

    def draw_area(
        self, drawing: npt.NDArray, area: ColorArea, color: Color, thickness: int
    ) -> None:
        """
        Outline an area on the drawing in a color and write its color and point count.
        """
        bgr = ClusterColorAnnotator.color_name_to_bgr_values[color]
        cv2.drawContours(drawing, [area.outline], -1, bgr, thickness)
        left, top, _, _ = cv2.boundingRect(area.outline)
        cv2.putText(
            drawing,
            f"{area.color.name.lower()} ({len(area.points.points)})",
            (left, max(top - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bgr,
            1,
        )
