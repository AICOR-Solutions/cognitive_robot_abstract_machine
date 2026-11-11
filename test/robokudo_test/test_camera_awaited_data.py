"""
What a live RGB-D camera interface reports waiting for while it has no new frame, so a
log tells which topic, synchronization or transform holds a query up.
"""

from __future__ import annotations

# %% Imports
from collections.abc import Iterator

import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig
from robokudo.io.camera_interface import KinectCameraInterface

# %% camera interface


@pytest.fixture
def camera_config() -> OrbbecCameraConfig:
    """
    :return: An Orbbec camera read uncompressed on its driver's default topics.
    """
    return OrbbecCameraConfig(
        topic_color="/camera/color/image_raw",
        color_hints="raw",
        tf_to="camera_link",
    )


@pytest.fixture
def camera(node, camera_config: OrbbecCameraConfig) -> Iterator[KinectCameraInterface]:
    """
    :return: The live interface of the camera, subscribed on the registered node.
    """
    yield KinectCameraInterface(camera_config)


def stamped(seconds: int) -> Header:
    """
    :param seconds: The stamp's seconds.
    :return: A header stamped then.
    """
    return Header(stamp=Time(sec=seconds))


# %% waiting


def test_a_camera_without_messages_names_every_topic_it_waits_on(
    camera: KinectCameraInterface, camera_config: OrbbecCameraConfig
) -> None:
    assert camera.describe_awaited_data() == (
        f"no message yet on {camera_config.topic_color}, "
        f"{camera_config.topic_depth}, {camera_config.topic_camera_info}"
    )


def test_a_camera_names_only_the_topics_without_messages(
    camera: KinectCameraInterface, camera_config: OrbbecCameraConfig
) -> None:
    camera.color_subscriber.signalMessage(Image(header=stamped(0)))

    assert camera.describe_awaited_data() == (
        f"no message yet on {camera_config.topic_depth}, "
        f"{camera_config.topic_camera_info}"
    )


def test_a_camera_whose_messages_never_match_in_time_says_so(
    camera: KinectCameraInterface, camera_config: OrbbecCameraConfig
) -> None:
    camera.color_subscriber.signalMessage(Image(header=stamped(0)))
    camera.depth_subscriber.signalMessage(Image(header=stamped(10)))
    camera.camera_info_subscriber.signalMessage(CameraInfo(header=stamped(20)))

    assert camera.describe_awaited_data() == (
        f"messages arrive (1 on {camera_config.topic_color}, "
        f"1 on {camera_config.topic_depth}, 1 on {camera_config.topic_camera_info}) "
        f"but never within {camera.synchronization_slop} s of each other"
    )


def test_a_camera_without_its_pose_on_tf_names_the_frames(
    camera: KinectCameraInterface, camera_config: OrbbecCameraConfig
) -> None:
    for topic in camera.received_messages:
        camera.received_messages[topic] = 1
    camera.synchronized_frames = 1
    camera.transform_missing = True

    assert camera.describe_awaited_data() == (
        f"no transform from {camera_config.tf_from} to {camera_config.tf_to} on TF"
    )
