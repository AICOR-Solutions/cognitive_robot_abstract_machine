"""
Reading the thresholds of Tracy's cube localization from a settings file.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import pytest
import yaml

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.analysis_engines.exceptions import UnknownSettingsSection
from robokudo.descriptors.analysis_engines.tracy_cubes import (
    AnalysisEngine,
    SettingsEnvironmentVariable,
    TracyCubesSettings,
)
from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig

# %% fixtures


@pytest.fixture
def settings_file() -> Path:
    """
    :return: A settings file setting every threshold away from its default.
    """
    return Path(__file__).parent / "dataset" / "tracy_cubes" / "tuned_settings.yaml"


@pytest.fixture
def written_settings(settings_file: Path) -> dict:
    """
    :return: The settings as written in the file.
    """
    return yaml.safe_load(settings_file.read_text())


# %% reading the file


def test_the_table_top_and_clusters_are_read_from_the_file(
    settings_file: Path, written_settings: dict
) -> None:
    settings = TracyCubesSettings.from_file(settings_file)

    table_top = written_settings["table_top"]
    assert settings.table_top.lowest_height == table_top["lowest_height"]
    assert settings.table_top.highest_height == table_top["highest_height"]
    assert (
        settings.table_top.plane_distance_threshold
        == table_top["plane_distance_threshold"]
    )
    assert (
        settings.clusters.minimum_cluster_size
        == written_settings["clusters"]["minimum_cluster_size"]
    )


def test_the_color_thresholds_are_read_from_the_file(
    settings_file: Path, written_settings: dict
) -> None:
    settings = TracyCubesSettings.from_file(settings_file)

    colors = written_settings["colors"]
    assert settings.colors.hue_upper_bounds == {
        Color[color_name.upper()]: bound
        for color_name, bound in colors["hue_upper_bounds"].items()
    }
    assert settings.colors.minimum_saturation == colors["minimum_saturation"]
    assert settings.colors.minimum_value == colors["minimum_value"]
    assert settings.colors.maximum_value_black == colors["maximum_value_black"]
    assert settings.colors.minimum_value_white == colors["minimum_value_white"]
    assert settings.colors.minimum_color_share == colors["minimum_color_share"]


def test_the_real_camera_is_read_from_the_file(
    settings_file: Path, written_settings: dict
) -> None:
    settings = TracyCubesSettings.from_file(settings_file)

    assert settings.real_camera == OrbbecCameraConfig(**written_settings["real_camera"])


def test_a_file_leaving_a_section_out_keeps_its_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"clusters": {"minimum_cluster_size": 7}}))

    settings = TracyCubesSettings.from_file(settings_file)

    assert settings.table_top == TracyCubesSettings().table_top
    assert settings.colors == TracyCubesSettings().colors
    assert settings.real_camera == OrbbecCameraConfig()


def test_a_misspelled_section_is_refused(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"colours": {"minimum_value": 7}}))

    with pytest.raises(UnknownSettingsSection):
        TracyCubesSettings.from_file(settings_file)


def test_the_default_colors_are_the_annotators_defaults() -> None:
    colors = TracyCubesSettings().colors
    parameters = ClusterColorAnnotator.Descriptor.Parameters()

    assert colors.hue_upper_bounds is None
    assert colors.minimum_saturation == parameters.min_saturation_color
    assert colors.minimum_value == parameters.min_value_color
    assert colors.maximum_value_black == parameters.max_value_black
    assert colors.minimum_value_white == parameters.min_value_white
    assert colors.minimum_color_share == parameters.ratio_annotation_threshold


# %% the analysis engine


def test_the_analysis_engine_reads_the_file_named_in_the_environment(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        SettingsEnvironmentVariable.TRACY_CUBES_SETTINGS, str(settings_file)
    )

    assert AnalysisEngine().settings == TracyCubesSettings.from_file(settings_file)


def test_the_analysis_engine_keeps_the_defaults_without_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SettingsEnvironmentVariable.TRACY_CUBES_SETTINGS, raising=False)

    assert AnalysisEngine().settings == TracyCubesSettings()


def test_the_annotators_take_the_thresholds_of_the_settings(
    settings_file: Path,
) -> None:
    settings = TracyCubesSettings.from_file(settings_file)

    annotators = AnalysisEngine(settings=settings).perception_annotators()

    [crop] = [a for a in annotators if isinstance(a, PointcloudCropAnnotator)]
    [plane] = [a for a in annotators if isinstance(a, PlaneAnnotator)]
    [clusters] = [a for a in annotators if isinstance(a, PointCloudClusterExtractor)]
    [colors] = [a for a in annotators if isinstance(a, ClusterColorAnnotator)]
    assert crop.descriptor.parameters.min_z == settings.table_top.lowest_height
    assert crop.descriptor.parameters.max_z == settings.table_top.highest_height
    assert (
        plane.descriptor.parameters.distance_threshold
        == settings.table_top.plane_distance_threshold
    )
    assert (
        clusters.descriptor.parameters.min_cluster_count
        == settings.clusters.minimum_cluster_size
    )
    assert colors.color_hue_positions == settings.colors.hue_upper_bounds
    color_parameters = colors.descriptor.parameters
    assert color_parameters.min_saturation_color == settings.colors.minimum_saturation
    assert color_parameters.min_value_color == settings.colors.minimum_value
    assert color_parameters.max_value_black == settings.colors.maximum_value_black
    assert color_parameters.min_value_white == settings.colors.minimum_value_white
    assert (
        color_parameters.ratio_annotation_threshold
        == settings.colors.minimum_color_share
    )
