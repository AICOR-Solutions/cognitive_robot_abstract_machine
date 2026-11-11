"""
Localizing the cubes on Tracy's table through Tracy's real camera, with RoboKudo in a
process of its own: the images arrive on the topics of the Orbbec Femto Bolt's driver,
rendered here from a simulated scene.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

# %% Imports
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from rclpy.publisher import Publisher
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from typing_extensions import Any

from robokudo.annotators.cluster_color import Color as SemanticColorName
from robokudo.descriptors.analysis_engines.tracy_cubes import TracyCubesSettings
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.camera_rendering import (
    VisualColorRenderingMode,
)
from semantic_digital_twin.robots.tracy import Tracy, TracyCamera

from .test_tracy_cubes import CubeScene, ask_client_for_objects, build_cube_scene

pytestmark = pytest.mark.skipif(
    not Tracy.is_description_installed(),
    reason="iai_tracy_description is not installed",
)

query_messages = pytest.importorskip("robokudo_msgs.action")
designator_messages = pytest.importorskip("robokudo_msgs.msg")

# %% the camera driver's stream


@dataclass
class CameraStream:
    """
    One frame of Tracy's camera as the camera's driver publishes it, sent repeatedly.
    """

    node: Node
    """
    The node publishing the frame.
    """

    color: Image
    """
    The color image, in the driver's encoding.
    """

    depth: Image
    """
    The depth image registered to the color image, in millimetres.
    """

    camera_info: CameraInfo
    """
    The calibration of the color image.
    """

    settings: TracyCubesSettings
    """
    The settings naming the driver's topics.
    """

    color_publisher: Publisher = field(init=False)
    """
    Publishes the color image.
    """

    depth_publisher: Publisher = field(init=False)
    """
    Publishes the depth image.
    """

    camera_info_publisher: Publisher = field(init=False)
    """
    Publishes the calibration.
    """

    def publish(self) -> None:
        """
        Publish the frame once, stamped now.
        """
        stamp = self.node.get_clock().now().to_msg()
        for message in (self.color, self.depth, self.camera_info):
            message.header.stamp = stamp
        self.color_publisher.publish(self.color)
        self.depth_publisher.publish(self.depth)
        self.camera_info_publisher.publish(self.camera_info)

    def __post_init__(self) -> None:
        camera = self.settings.real_camera
        self.color_publisher = self.node.create_publisher(Image, camera.topic_color, 1)
        self.depth_publisher = self.node.create_publisher(Image, camera.topic_depth, 1)
        self.camera_info_publisher = self.node.create_publisher(
            CameraInfo, camera.topic_camera_info, 1
        )


@pytest.fixture(scope="module")
def settings_file() -> Path:
    """
    :return: Settings reading the Femto Bolt on its driver's default topics and frames.
    """
    return (
        Path(__file__).parent / "dataset" / "tracy_cubes" / "femto_bolt_settings.yaml"
    )


@pytest.fixture(scope="module")
def cube_scene() -> CubeScene:
    """
    :return: Tracy with three cubes at random poses, the scene the camera sees.
    """
    return build_cube_scene(0)


@pytest.fixture(scope="module")
def camera_driver(cube_scene: CubeScene, settings_file: Path) -> Iterator[CameraStream]:
    """
    Publish what Tracy's camera sees of the scene as the camera's driver does, with the
    optical frame below Tracy's camera link on TF, and Tracy's description.

    :return: The stream of the camera's frame.
    """
    settings = TracyCubesSettings.from_file(settings_file)
    world = cube_scene.world
    [camera] = world.get_semantic_annotations_by_type(TracyCamera)
    frame = world.ray_tracer.render(camera, rgb_mode=VisualColorRenderingMode())
    bridge = CVBridgeWorkaround()
    optical_frame = settings.real_camera.tf_from
    color = bridge.cv2_to_imgmsg(
        np.ascontiguousarray(frame.color_bgr.swapaxes(0, 1)[..., ::-1]), "rgb8"
    )
    color.header.frame_id = optical_frame
    depth = bridge.cv2_to_imgmsg(
        np.ascontiguousarray(frame.depth_in_millimeters.swapaxes(0, 1)).astype(
            np.uint16
        ),
        "16UC1",
    )
    depth.header.frame_id = optical_frame
    node = rclpy.create_node("tracy_camera_driver")
    stream = CameraStream(
        node=node,
        color=color,
        depth=depth,
        camera_info=RosCameraModelAdapter.to_camera_info(
            frame.camera_model, optical_frame
        ),
        settings=settings,
    )
    camera_link = world.get_kinematic_structure_entity_by_name(
        settings.real_camera.tf_to
    )
    link_T_optical = world.compute_forward_kinematics(camera_link, camera.root).to_np()
    transform = TransformStamped()
    transform.header.frame_id = settings.real_camera.tf_to
    transform.child_frame_id = optical_frame
    (
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ) = link_T_optical[:3, 3].tolist()
    (
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
    ) = Rotation.from_matrix(link_T_optical[:3, :3]).as_quat().tolist()
    StaticTransformBroadcaster(node).sendTransform(transform)
    node.create_publisher(
        String,
        "/robot_description",
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    ).publish(String(data=URDFParser.from_file(Tracy.get_ros_file_path()).urdf))
    node.create_timer(0.2, stream.publish)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield stream
    executor.shutdown()
    spinner.join()
    node.destroy_node()


@dataclass
class PerceptionClient:
    """
    A client of RoboKudo's query action, with the node it is spun on.
    """

    node: Node
    """
    The node the client is created on.
    """

    client: ActionClient
    """
    The client of RoboKudo's query action.
    """

    process: subprocess.Popen
    """
    The perception process answering the client.
    """


stopping_time = 5.0
"""
How long the perception process may take to stop once it receives SIGTERM, in seconds.
"""


@pytest.fixture
def perception_process(
    request: pytest.FixtureRequest, settings_file: Path, camera_driver: CameraStream
) -> Iterator[PerceptionClient]:
    """
    Run Tracy's cube localization through the real camera in a process of its own, one
    per test so that each query reaches an idle analysis engine.

    :return: A client of its query action, once its server is up.
    """
    node = rclpy.create_node("tracy_cubes_perception_client")
    client = ActionClient(node, query_messages.Query, "robokudo/query")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "robokudo.scripts.perceive_tracy_cubes",
            "--settings",
            str(settings_file),
        ],
        env=os.environ.copy(),
    )
    request.addfinalizer(process.kill)
    started = time.monotonic()
    while not client.wait_for_server(timeout_sec=1.0):
        assert process.poll() is None, "the perception process ended"
        assert time.monotonic() - started < 180.0
    yield PerceptionClient(node=node, client=client, process=process)
    process.terminate()
    assert process.wait(timeout=stopping_time) == 0
    client.destroy()
    node.destroy_node()


# %% answering queries


def test_every_cube_is_localized_in_tracys_world_frame(
    cube_scene: CubeScene, perception_process: PerceptionClient
) -> None:
    designators = ask_client_for_objects(
        perception_process.client, perception_process.node
    )

    for cube in cube_scene.cubes:
        [match] = [
            designator
            for designator in designators
            if cube.color_name.name.lower() in designator.color
        ]
        [pose] = match.pose
        frame = cube_scene.world.get_kinematic_structure_entity_by_name(
            pose.header.frame_id
        )
        assert frame is cube_scene.world.root
        position = pose.pose.position
        assert (
            math.dist(
                (position.x, position.y, position.z),
                (
                    cube.x,
                    cube.y,
                    cube_scene.table_height + cube_scene.edge_length / 2,
                ),
            )
            < 0.01
        )


def test_every_cube_is_localized_in_the_frame_tf_names_tracys_root_link(
    cube_scene: CubeScene, perception_process: PerceptionClient
) -> None:
    designators = ask_client_for_objects(
        perception_process.client, perception_process.node
    )

    assert {
        pose.header.frame_id for designator in designators for pose in designator.pose
    } == {cube_scene.world.root.name.name}


def test_a_query_for_a_color_is_answered_with_the_cube_of_that_color_only(
    cube_scene: CubeScene, perception_process: PerceptionClient
) -> None:
    red = SemanticColorName.RED.name.lower()

    designators = ask_client_for_objects(
        perception_process.client,
        perception_process.node,
        designator_messages.ObjectDesignator(type="block", color=[red]),
    )

    [red_cube] = [
        cube for cube in cube_scene.cubes if cube.color_name.name.lower() == red
    ]
    [designator] = designators
    assert red in designator.color
    position = designator.pose[0].pose.position
    assert math.dist((position.x, position.y), (red_cube.x, red_cube.y)) < 0.01


def send_query(perception: PerceptionClient) -> Future:
    """
    Send a query for any object and wait until the perception process accepts it.

    :param perception: The client of the perception process.
    :return: The future of the query's result.
    """
    executor = SingleThreadedExecutor(context=perception.node.context)
    executor.add_node(perception.node)
    goal_handle_future = perception.client.send_goal_async(query_messages.Query.Goal())
    executor.spin_until_future_complete(goal_handle_future, timeout_sec=10.0)
    executor.remove_node(perception.node)
    goal_handle = goal_handle_future.result()
    assert goal_handle is not None and goal_handle.accepted
    return goal_handle.get_result_async()


def wait_for_result(perception: PerceptionClient, result_future: Future) -> Any:
    """
    :param perception: The client of the perception process.
    :param result_future: The future of a query's result.
    :return: The query's result with its status, once it arrives within ten seconds.
    """
    executor = SingleThreadedExecutor(context=perception.node.context)
    executor.add_node(perception.node)
    executor.spin_until_future_complete(result_future, timeout_sec=10.0)
    executor.remove_node(perception.node)
    assert result_future.done(), "the query was not answered"
    return result_future.result()


def assert_every_cube_is_answered(cube_scene: CubeScene, result: Any) -> None:
    """
    :param cube_scene: The scene the camera sees.
    :param result: The answer to a query for any object.
    """
    colors = {color for designator in result.res for color in designator.color}
    for cube in cube_scene.cubes:
        assert cube.color_name.name.lower() in colors


def test_a_query_sent_right_after_an_answer_is_answered(
    cube_scene: CubeScene, perception_process: PerceptionClient
) -> None:
    first = wait_for_result(perception_process, send_query(perception_process))
    second = wait_for_result(perception_process, send_query(perception_process))

    for result in (first, second):
        assert result.status == GoalStatus.STATUS_SUCCEEDED
        assert_every_cube_is_answered(cube_scene, result.result)


def test_sigterm_stops_the_process_while_a_query_is_answered(
    perception_process: PerceptionClient,
) -> None:
    send_query(perception_process)
    perception_process.process.terminate()

    assert perception_process.process.wait(timeout=stopping_time) == 0


def show_depth(
    camera_driver: CameraStream, perception: PerceptionClient, depth: Image
) -> None:
    """
    Let the camera's driver publish a depth image from now on, and wait until it has
    been published twice, so that the perception process holds no earlier frame.

    :param camera_driver: The stream of the camera's frame.
    :param perception: The client of the perception process, whose node listens.
    :param depth: The depth image to publish.
    """
    received = []
    subscription = perception.node.create_subscription(
        Image,
        camera_driver.settings.real_camera.topic_depth,
        lambda message: received.append(bytes(message.data) == bytes(depth.data)),
        10,
    )
    camera_driver.depth = depth
    executor = SingleThreadedExecutor(context=perception.node.context)
    executor.add_node(perception.node)
    started = time.monotonic()
    while received.count(True) < 2:
        assert time.monotonic() - started < 10.0, "the depth image was not published"
        executor.spin_once(timeout_sec=0.5)
    executor.remove_node(perception.node)
    perception.node.destroy_subscription(subscription)


def test_a_query_failing_on_a_frame_fails_and_the_next_query_is_answered(
    cube_scene: CubeScene,
    camera_driver: CameraStream,
    perception_process: PerceptionClient,
) -> None:
    depth = camera_driver.depth
    show_depth(
        camera_driver,
        perception_process,
        Image(
            header=depth.header,
            height=depth.height,
            width=depth.width,
            encoding=depth.encoding,
            is_bigendian=depth.is_bigendian,
            step=depth.step,
            data=bytes(len(depth.data)),
        ),
    )
    failed = wait_for_result(perception_process, send_query(perception_process))
    show_depth(camera_driver, perception_process, depth)
    answered = wait_for_result(perception_process, send_query(perception_process))

    assert failed.status == GoalStatus.STATUS_ABORTED
    assert answered.status == GoalStatus.STATUS_SUCCEEDED
    assert_every_cube_is_answered(cube_scene, answered.result)


def test_a_depth_image_not_registered_to_the_color_image_is_logged(
    camera_driver: CameraStream, settings_file: Path, tmp_path: Path
) -> None:
    """
    The Femto Bolt's driver sends a depth image of its own size unless asked to register
    it to the color image; the log has to say so, as no point cloud can be built.
    """
    depth = camera_driver.depth
    unregistered = CVBridgeWorkaround().cv2_to_imgmsg(
        np.zeros((depth.height // 2, depth.width // 2), dtype=np.uint16), "16UC1"
    )
    unregistered.header = depth.header
    log_file = tmp_path / "perception.log"
    node = rclpy.create_node("tracy_cubes_unregistered_depth_client")
    client = ActionClient(node, query_messages.Query, "robokudo/query")
    with log_file.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robokudo.scripts.perceive_tracy_cubes",
                "--settings",
                str(settings_file),
            ],
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    perception = PerceptionClient(node=node, client=client, process=process)
    started = time.monotonic()
    while not client.wait_for_server(timeout_sec=1.0):
        assert process.poll() is None, "the perception process ended"
        assert time.monotonic() - started < 180.0
    show_depth(camera_driver, perception, unregistered)
    send_query(perception)
    started = time.monotonic()
    while "register the depth image" not in log_file.read_text():
        assert time.monotonic() - started < 20.0, log_file.read_text()[-2000:]
        time.sleep(0.5)
    process.kill()
    process.wait()
    show_depth(camera_driver, perception, depth)
    client.destroy()
    node.destroy_node()
