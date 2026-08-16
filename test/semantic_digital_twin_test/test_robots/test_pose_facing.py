import numpy as np
import pytest

from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# The three distinct forward axes the repository declares: PR2 keeps the default x,
# Stretch reaches out along negative y and Armar7 along y.
ROBOTS_WITH_DISTINCT_FORWARD_AXES = [PR2, Stretch, Armar7]

HEADINGS = [0.0, np.pi / 2, -np.pi / 2, np.pi, 0.3]


def _spawn(robot_type: type) -> tuple[World, AbstractRobot]:
    """
    A world holding nothing but ``robot_type``.
    """
    world = World.create_with_root_body("root")
    try:
        robot = RobotSpecification(semantic_annotation_type=robot_type).spawn(world)
    except ParsingError as error:
        pytest.skip(f"{robot_type.__name__} URDF not available: {error}")
    return world, robot


# %% the forward axis ends up pointing along the heading


@pytest.mark.parametrize("robot_type", ROBOTS_WITH_DISTINCT_FORWARD_AXES)
@pytest.mark.parametrize("heading_yaw", HEADINGS)
def test_forward_axis_points_along_the_heading(robot_type: type, heading_yaw: float):
    world, robot = _spawn(robot_type)
    mobile_base = robot.mobile_base
    heading = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=heading_yaw, reference_frame=world.root
    )

    base_pose = mobile_base.pose_facing(heading)

    world_V_forward = base_pose.to_rotation_matrix() @ mobile_base.forward_axis
    np.testing.assert_allclose(
        world_V_forward.to_np()[:3].flatten(),
        [np.cos(heading_yaw), np.sin(heading_yaw), 0.0],
        atol=1e-9,
    )


@pytest.mark.parametrize("robot_type", ROBOTS_WITH_DISTINCT_FORWARD_AXES)
def test_the_position_is_the_headings_own(robot_type: type):
    world, robot = _spawn(robot_type)
    heading = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.3, reference_frame=world.root)

    base_pose = robot.mobile_base.pose_facing(heading)

    np.testing.assert_allclose(
        base_pose.to_position().to_np(), heading.to_position().to_np(), atol=1e-9
    )
    assert base_pose.reference_frame is heading.reference_frame


# %% a base whose front is already the x-axis needs no correction


def test_an_x_forward_base_takes_the_heading_unchanged():
    world, robot = _spawn(PR2)
    heading = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.3, reference_frame=world.root)

    np.testing.assert_allclose(
        robot.mobile_base.pose_facing(heading).to_np(), heading.to_np(), atol=1e-9
    )
