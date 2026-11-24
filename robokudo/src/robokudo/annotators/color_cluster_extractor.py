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
    A connected area of pixels of one chromatic color, and the points the depth image
    holds under it.
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
    The points under the area, in the camera frame, that lie between the heights the
    objects are looked for at.
    """

    def to_object_hypothesis(self, identifier: str, source: str) -> ObjectHypothesis:
        """
        :param identifier: The identifier the hypothesis takes.
        :param source: The name of the annotator that found the area.
        :return: An object of the area's points, outlined by its region and annotated
            with its color.
        """
        hypothesis = ObjectHypothesis()
        hypothesis.id = identifier
        hypothesis.source = source
        hypothesis.points = self.points
        left, top, width, height = cv2.boundingRect(self.outline)
        hypothesis.roi.roi.pos.x = left
        hypothesis.roi.roi.pos.y = top
        hypothesis.roi.roi.width = width
        hypothesis.roi.roi.height = height
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(
            mask, [self.outline], -1, 255, cv2.FILLED, offset=(-left, -top)
        )
        hypothesis.roi.mask = mask
        color = SemanticColor()
        color.color = self.color.name.lower()
        color.source = source
        color.ratio = 1.0
        hypothesis.annotations.append(color)
        return hypothesis


# %% annotator


class ColorClusterExtractor(BaseAnnotator):
    """
    Turns every connected area of one chromatic color in the color image into an object
    of that color, holding the points of the depth image under the area that lie between
    two heights in the world frame. Needs the point cloud's camera model in the CAS.
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
                Per chromatic color, the hue on OpenCV's full scale a pixel lies below to
                take it, tried from red to magenta; hues above magenta's are red again.
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
                How many pixels at an area's edge are left out, since their depth may be
                that of what lies behind.
                """
                self.lowest_height: float = -math.inf
                """
                Height in the world frame, in metres, below which no point of an object
                lies.
                """
                self.highest_height: float = math.inf
                """
                Height in the world frame, in metres, above which no point of an object
                lies.
                """
                self.minimum_points: int = 100
                """
                The fewest points between the heights an area needs to become an object.
                """
                self.outlier_neighbors: int = 20
                """
                How many neighbors a point's distance to them is averaged over, to leave
                out the points of an area far from the others.
                """
                self.outlier_standard_deviations: float = 2.0
                """
                How many standard deviations above the mean a point's average distance to
                its neighbors may lie before it is left out.
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
            cas, cas.get(CASViews.COLOR_IMAGE)
        )
        depth_image = cas.get(CASViews.DEPTH_IMAGE)
        if not depth_image.any():
            raise DepthNotMeasured(context=self.name)
        areas = self.color_areas(color_image)
        drawing = color_image.copy()
        objects = 0
        for area in areas:
            area.points = self.points_under(area, color_image, depth_image)
            if len(area.points.points) < self.descriptor.parameters.minimum_points:
                self.draw_area(
                    drawing,
                    area,
                    ClusterColorAnnotator.color_name_to_bgr_values[Color.GREY],
                    1,
                )
                continue
            cas.annotations.append(area.to_object_hypothesis(str(objects), self.name))
            objects += 1
            self.draw_area(
                drawing,
                area,
                ClusterColorAnnotator.color_name_to_bgr_values[area.color],
                3,
            )
        self.get_annotator_output_struct().set_image(drawing)
        self.feedback_message = (
            f"{objects} objects of {len(areas)} color areas in "
            f"{default_timer() - start:.3f}s"
        )
        return Status.SUCCESS

    # %% naming pixels by color

    def color_areas(self, color_image: npt.NDArray) -> list[ColorArea]:
        """
        :param color_image: A color image, blue first.
        :return: The connected areas of one chromatic color in it, specks left out.
        """
        parameters = self.descriptor.parameters
        colors = sorted(parameters.hue_upper_bounds, key=lambda color: color.value)
        named = self.chromatic_colors(color_image, colors)
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

    def chromatic_colors(
        self, color_image: npt.NDArray, colors: list[Color]
    ) -> npt.NDArray:
        """
        :param color_image: A color image, blue first.
        :param colors: The chromatic colors with a hue bound, from red to magenta.
        :return: Per pixel, the index in the colors of the one it takes, -1 where it is
            too pale or too dark to take one.
        """
        parameters = self.descriptor.parameters
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV_FULL)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        bounds = np.array([parameters.hue_upper_bounds[color] for color in colors])
        named = np.searchsorted(bounds, hue, side="right")
        named[named == len(bounds)] = 0
        named[
            (saturation <= parameters.minimum_saturation)
            | (value <= parameters.minimum_value)
        ] = -1
        return named

    # %% the points under an area

    def points_under(
        self, area: ColorArea, color_image: npt.NDArray, depth_image: npt.NDArray
    ) -> open3d.geometry.PointCloud:
        """
        :param area: An area of the color image.
        :param color_image: The color image, blue first, registered to the depth image.
        :param depth_image: The depth image.
        :return: The points under the area, its edge left out, in the camera frame,
            those far from the others and those outside the heights left out.
        """
        parameters = self.descriptor.parameters
        mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [area.outline], -1, 255, cv2.FILLED)
        edge = np.ones((2 * parameters.edge_width + 1,) * 2, np.uint8)
        mask = cv2.erode(mask, edge)
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
                self.get_cas().pointcloud_camera_model
            ),
        )
        if len(points.points) < parameters.minimum_points:
            return points
        points, _ = points.remove_statistical_outlier(
            nb_neighbors=parameters.outlier_neighbors,
            std_ratio=parameters.outlier_standard_deviations,
        )
        return points.select_by_index(np.flatnonzero(self.between_the_heights(points)))

    def between_the_heights(self, points: open3d.geometry.PointCloud) -> npt.NDArray:
        """
        :param points: Points in the camera frame.
        :return: Per point, whether its height in the world frame lies between
            :attr:`Parameters.lowest_height` and :attr:`Parameters.highest_height`.
        """
        camera_to_world = get_camera_to_world_transform_matrix(self.get_cas())
        heights = (
            np.asarray(points.points) @ camera_to_world[2, :3] + camera_to_world[2, 3]
        )
        return (heights >= self.descriptor.parameters.lowest_height) & (
            heights <= self.descriptor.parameters.highest_height
        )

    # %% what it drew

    def draw_area(
        self,
        drawing: npt.NDArray,
        area: ColorArea,
        outline_color: tuple[int, ...],
        thickness: int,
    ) -> None:
        """
        Outline an area on the drawing and write its color and number of points next to
        it.

        :param drawing: The image drawn on, blue first.
        :param area: The area.
        :param outline_color: The color of the outline and the text, blue first.
        :param thickness: The outline's thickness in pixels.
        """
        cv2.drawContours(drawing, [area.outline], -1, outline_color, thickness)
        left, top, _, _ = cv2.boundingRect(area.outline)
        cv2.putText(
            drawing,
            f"{area.color.name.lower()} ({len(area.points.points)})",
            (left, max(top - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            outline_color,
            1,
        )
