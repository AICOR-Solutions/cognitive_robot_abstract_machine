"""
Analysis engine localizing the cubes on Tracy's table through Tracy's fixed camera.

It answers queries with one object designator per cube: its pose in the world frame,
upright with its yaw known up to a quarter turn, and its semantic color. A query naming
a color is answered with the cubes of that color only, whatever type it names. Only
existing annotators are used: plane, cluster extraction, cluster color and bounding-box
pose. Its thresholds come from a settings file, so they can be tuned without a rebuild.
"""

from __future__ import annotations

# %% Imports
import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import yaml
from py_trees.behaviour import Behaviour
from typing_extensions import Any, Self

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors.analysis_engines.exceptions import UnknownSettingsSection
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    RuntimeRobotWorldSource,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from semantic_digital_twin.datastructures.camera_rendering import (
    VisualColorRenderingMode,
)

# %% Settings


class SettingsEnvironmentVariable(StrEnum):
    """
    Environment variables naming the settings file of an analysis engine.
    """

    TRACY_CUBES_SETTINGS = "ROBOKUDO_TRACY_CUBES_SETTINGS"
    """
    Path of the settings file of Tracy's cube localization; unset keeps the defaults.
    """


@dataclass
class TableTopRegion:
    """
    Where the points of the table top and of the cubes on it lie.
    """

    lowest_height: float = 0.8
    """
    Height above the world origin, in metres, below which no point belongs to the table
    top or to a cube on it.
    """

    highest_height: float = 1.1
    """
    Height above the world origin, in metres, above which no point belongs to the table
    top or to a cube on it.
    """

    plane_distance_threshold: float = 0.005
    """
    How far, in metres, a point may lie from the table plane and still belong to it.
    """

    @classmethod
    def from_json(cls, data: dict[str, float]) -> Self:
        """
        :param data: The region's fields by name.
        :return: The region written in the data.
        """
        return cls(**data)


@dataclass
class CubeClusters:
    """
    Which clusters of points above the table become objects.
    """

    minimum_cluster_size: int = 100
    """
    The fewest points a cluster above the table needs to become an object; a cube seen
    from Tracy's simulated camera has about a thousand.
    """

    @classmethod
    def from_json(cls, data: dict[str, int]) -> Self:
        """
        :param data: The thresholds by field name.
        :return: The thresholds written in the data.
        """
        return cls(**data)


@dataclass
class ColorThresholds:
    """
    How the pixels of a cluster are named by color, on OpenCV's full HSV scale (0 to
    255), and how many of them a color needs to be annotated.
    """

    hue_upper_bounds: dict[Color, float] | None = None
    """
    Per chromatic color, the hue a pixel lies below to take it, tried from red to
    magenta; hues above magenta's bound are red again. None divides the hue evenly.
    """

    minimum_saturation: int = field(
        default_factory=lambda: (
            ClusterColorAnnotator.Descriptor.Parameters().min_saturation_color
        )
    )
    """
    The saturation a pixel must exceed to be named by its hue.
    """

    minimum_value: int = field(
        default_factory=lambda: (
            ClusterColorAnnotator.Descriptor.Parameters().min_value_color
        )
    )
    """
    The brightness a pixel must exceed to be named by its hue.
    """

    maximum_value_black: int = field(
        default_factory=lambda: (
            ClusterColorAnnotator.Descriptor.Parameters().max_value_black
        )
    )
    """
    The brightness up to which a pixel without a hue is black.
    """

    minimum_value_white: int = field(
        default_factory=lambda: (
            ClusterColorAnnotator.Descriptor.Parameters().min_value_white
        )
    )
    """
    The brightness above which a pixel without a hue is white; between black and white
    it is grey.
    """

    minimum_color_share: float = field(
        default_factory=lambda: (
            ClusterColorAnnotator.Descriptor.Parameters().ratio_annotation_threshold
        )
    )
    """
    The share of a cluster's pixels a color must exceed to be annotated on it.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """
        :param data: The thresholds by field name, the hue bounds by color name.
        :return: The thresholds written in the data.
        """
        thresholds = dict(data)
        hue_upper_bounds = thresholds.pop("hue_upper_bounds", None)
        if hue_upper_bounds is None:
            return cls(**thresholds)
        return cls(
            **thresholds,
            hue_upper_bounds={
                Color[color_name.upper()]: bound
                for color_name, bound in hue_upper_bounds.items()
            },
        )

    def create_annotator(self) -> ClusterColorAnnotator:
        """
        :return: An annotator naming the colors of clusters by these thresholds.
        """
        descriptor = ClusterColorAnnotator.Descriptor()
        descriptor.parameters.hue_upper_bounds = self.hue_upper_bounds
        descriptor.parameters.min_saturation_color = self.minimum_saturation
        descriptor.parameters.min_value_color = self.minimum_value
        descriptor.parameters.max_value_black = self.maximum_value_black
        descriptor.parameters.min_value_white = self.minimum_value_white
        descriptor.parameters.ratio_annotation_threshold = self.minimum_color_share
        return ClusterColorAnnotator(descriptor=descriptor)


@dataclass
class TracyCubesSettings:
    """
    The tunable settings of Tracy's cube localization, as written in its settings file.
    """

    table_top: TableTopRegion = field(default_factory=TableTopRegion)
    """
    Where the table top and the cubes on it are looked for.
    """

    clusters: CubeClusters = field(default_factory=CubeClusters)
    """
    Which clusters above the table become cubes.
    """

    colors: ColorThresholds = field(default_factory=ColorThresholds)
    """
    How the cubes are named by color.
    """

    real_camera: OrbbecCameraConfig = field(default_factory=OrbbecCameraConfig)
    """
    Topics and frames of the real camera, Tracy's Orbbec Femto Bolt.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """
        :param data: The settings by section, each section's entries by field name.
        :return: The settings written in the data.
        :raises UnknownSettingsSection: If the data holds a section these settings lack.
        """
        known_sections = [settings_field.name for settings_field in fields(cls)]
        unknown_sections = [
            section for section in data if section not in known_sections
        ]
        if unknown_sections:
            raise UnknownSettingsSection(
                unknown_sections=unknown_sections, known_sections=known_sections
            )
        return cls(
            table_top=TableTopRegion.from_json(data.get("table_top", {})),
            clusters=CubeClusters.from_json(data.get("clusters", {})),
            colors=ColorThresholds.from_json(data.get("colors", {})),
            real_camera=OrbbecCameraConfig(**data.get("real_camera", {})),
        )

    @classmethod
    def from_file(cls, settings_file: Path) -> Self:
        """
        :param settings_file: A YAML file with the sections of these settings; a section
            or entry left out keeps its default.
        :return: The settings written in the file.
        """
        return cls.from_json(yaml.safe_load(settings_file.read_text()) or {})

    @classmethod
    def from_environment(cls) -> Self:
        """
        :return: The settings in the file
            :attr:`SettingsEnvironmentVariable.TRACY_CUBES_SETTINGS` names, else the
            defaults.
        """
        settings_file = os.environ.get(SettingsEnvironmentVariable.TRACY_CUBES_SETTINGS)
        if not settings_file:
            return cls()
        return cls.from_file(Path(settings_file))


# %% Analysis engine


@dataclass
class AnalysisEngine(AnalysisEngineInterface):
    """
    Localizes the cubes standing on Tracy's table and names their colors.
    """

    camera: BaseCameraConfig = field(
        default_factory=lambda: RayTracingCameraConfiguration(
            source=RuntimeRobotWorldSource(), rgb_mode=VisualColorRenderingMode()
        )
    )
    """
    The camera the cubes are seen through; by default Tracy's camera rendered from the
    world RoboKudo runs in.
    """

    settings: TracyCubesSettings = field(
        default_factory=TracyCubesSettings.from_environment
    )
    """
    The thresholds the cubes are found and named by.
    """

    def name(self) -> str:
        return "tracy_cubes"

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("TracyCubesPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                *self.perception_annotators(),
                GenerateQueryResult(descriptor=self.query_result()),
                ActionServerCheck(),
            ]
        )
        return pipeline

    @staticmethod
    def query_result() -> GenerateQueryResult.Descriptor:
        """
        :return: The configuration of the answer: the cubes matching the query, each
            matching every type since the cubes are not classified.
        """
        descriptor = GenerateQueryResult.Descriptor()
        descriptor.parameters.filter_by_query = True
        descriptor.parameters.unclassified_objects_match_every_type = True
        return descriptor

    def perception_annotators(self) -> list[Behaviour]:
        """
        :return: The annotators that read one camera frame and leave one object
            hypothesis per cube in it, each with a pose and a semantic color.
        """
        crop = PointcloudCropAnnotator.Descriptor()
        crop.parameters.relative_to_world = True
        crop.parameters.min_z = self.settings.table_top.lowest_height
        crop.parameters.max_z = self.settings.table_top.highest_height
        plane = PlaneAnnotator.Descriptor()
        plane.parameters.distance_threshold = (
            self.settings.table_top.plane_distance_threshold
        )
        plane.parameters.random_seed = 0
        clusters = PointCloudClusterExtractor.Descriptor()
        clusters.parameters.min_cluster_count = (
            self.settings.clusters.minimum_cluster_size
        )
        return [
            CollectionReaderAnnotator(
                descriptor=CollectionReaderDescriptorFactory.create_descriptor_from_camera_configuration(
                    self.camera
                )
            ),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(descriptor=crop),
            PlaneAnnotator(descriptor=plane),
            PointCloudClusterExtractor(descriptor=clusters),
            self.settings.colors.create_annotator(),
            ClusterPoseBBAnnotator(),
        ]
