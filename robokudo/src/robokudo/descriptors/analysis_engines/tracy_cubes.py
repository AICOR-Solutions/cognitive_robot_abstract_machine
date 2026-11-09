"""
Analysis engine localizing the cubes on Tracy's table through Tracy's fixed camera.

It answers queries with one object designator per cube: its pose in the world frame,
upright with its yaw known up to a quarter turn, and its semantic color. A query naming
a color is answered with the cubes of that color only, whatever type it names. Only
existing annotators are used: plane, cluster extraction, cluster color and bounding-box
pose.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from py_trees.behaviour import Behaviour

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.behaviours.action_server_checks import ActionServerCheck
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

# %% Analysis engine


@dataclass
class AnalysisEngine(AnalysisEngineInterface):
    """
    Localizes the cubes standing on Tracy's table and names their colors.
    """

    camera: CollectionReaderAnnotator.Descriptor = field(
        default_factory=lambda: CollectionReaderDescriptorFactory.create_descriptor(
            RayTracingCameraConfiguration.registry_name,
            source=RuntimeRobotWorldSource(),
            rgb_mode=VisualColorRenderingMode(),
        )
    )
    """
    The camera the cubes are seen through; by default Tracy's camera rendered from the
    world RoboKudo runs in.
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

    minimum_cluster_size: int = 100
    """
    The fewest points a cluster above the table needs to become an object; a cube seen
    from Tracy's camera has about a thousand.
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
        crop.parameters.min_z = self.lowest_height
        crop.parameters.max_z = self.highest_height
        plane = PlaneAnnotator.Descriptor()
        plane.parameters.distance_threshold = self.plane_distance_threshold
        plane.parameters.random_seed = 0
        clusters = PointCloudClusterExtractor.Descriptor()
        clusters.parameters.min_cluster_count = self.minimum_cluster_size
        return [
            CollectionReaderAnnotator(descriptor=self.camera),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(descriptor=crop),
            PlaneAnnotator(descriptor=plane),
            PointCloudClusterExtractor(descriptor=clusters),
            ClusterColorAnnotator(),
            ClusterPoseBBAnnotator(),
        ]
