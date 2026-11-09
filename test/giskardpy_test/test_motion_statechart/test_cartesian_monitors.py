import numpy as np
import pytest

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    PoseReached,
    PositionReached,
    OrientationReached,
    PointingAt,
    VectorsAligned,
    DistanceToLine,
    InFieldOfView,
    OutOfFieldOfView,
)
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianOrientation,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def _create_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with motion control acting in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )


def _run(executor: StatechartExecutor, statechart: Statechart) -> None:
    """
    Compiles `statechart` with `executor` and ticks it until it ends.
    """
    executor.compile(statechart=statechart)
    executor.tick_until_end()


def test_position_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.6, -0.3, 1.0, reference_frame=root)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(root_link=root, tip_link=tip, goal_point=goal_point)
    monitor = PositionReached(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


@pytest.mark.parametrize(
    "binding_policy",
    [GoalBindingPolicy.Bind_at_build, GoalBindingPolicy.Bind_on_start],
)
def test_position_reached_binding_policies(
    pr2_world_state_reset: World, binding_policy: GoalBindingPolicy
):
    """
    The robot is driven toward a goal expressed relative to the tip.

    The monitor captures the same goal with the same policy and serves as the success
    condition, so the motion only ends once the monitor detects the goal was reached.
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.2, 0, 0, reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=binding_policy,
    )
    monitor = PositionReached(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=binding_policy,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_orientation_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_orientation = RotationMatrix.from_axis_angle(
        AxisAngle(Vector3.Z(reference_frame=root), 0.3)
    )
    goal_orientation.reference_frame = root

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianOrientation(
        root_link=root, tip_link=tip, goal_orientation=goal_orientation
    )
    monitor = OrientationReached(
        root_link=root,
        tip_link=tip,
        goal_orientation=goal_orientation,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pose_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.6, y=-0.3, z=1.0, reference_frame=root
    )

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPose(root_link=root, tip_link=tip, goal_pose=goal_pose)
    monitor = PoseReached(
        root_link=root,
        tip_link=tip,
        goal_pose=goal_pose,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pointing_at(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = Pointing(
        root_link=root, tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis
    )
    monitor = PointingAt(
        root_link=root, tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_vectors_aligned(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.X(reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    monitor = VectorsAligned(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_distance_to_line(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    center_point = Point3(0.6, -0.3, 1.0, reference_frame=root)
    line_axis = Vector3.Y(reference_frame=root)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(root_link=root, tip_link=tip, goal_point=center_point)
    monitor = DistanceToLine(
        root_link=root,
        tip_link=tip,
        center_point=center_point,
        line_axis=line_axis,
        line_length=0.4,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


# %% field of view


@pytest.fixture
def camera_and_target(pr2_world_copy: World) -> tuple[Camera, Body]:
    """
    :return: The PR2's camera and a free body to put somewhere in its view.
    """
    world = pr2_world_copy
    target = Body(name=PrefixedName("target"))
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=target
            )
        )
    [camera] = world.get_semantic_annotations_by_type(Camera)
    return camera, target


def _observe_at(monitor: InFieldOfView, camera_P_target: Point3) -> float:
    """
    Put the target of `monitor` at a point of its camera and let it observe it.

    :return: What `monitor` observes then.
    """
    world = monitor.camera._world
    root_P_target = world.transform(camera_P_target, world.root)
    monitor.target.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=root_P_target.x,
            y=root_P_target.y,
            z=root_P_target.z,
            reference_frame=world.root,
        )
    )
    executor = _create_executor(world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(monitor)
    executor.compile(statechart=statechart)
    executor.tick()
    return monitor.observation_state


@pytest.mark.parametrize(
    "x, y, depth",
    [(0.0, 0.0, 1.0), (0.3, -0.2, 1.0), (2.0, 0.0, 1.0), (0.0, 0.0, -1.0)],
)
def test_in_field_of_view_agrees_with_the_camera(camera_and_target, x, y, depth):
    camera, target = camera_and_target
    point = Point3(x, y, depth, reference_frame=camera.root)

    observed = _observe_at(InFieldOfView(camera=camera, target=target), point)

    expected = camera.is_in_field_of_view(point)
    assert observed == (
        ObservationStateValues.TRUE if expected else ObservationStateValues.FALSE
    )


def test_a_target_within_the_margin_of_an_edge_is_out_of_view(camera_and_target):
    camera, target = camera_and_target
    half_angle = camera.field_of_view.horizontal_angle / 2
    margin = 0.1
    near_the_edge = Point3(
        np.tan(half_angle - margin / 2), 0.0, 1.0, reference_frame=camera.root
    )

    observed = _observe_at(
        InFieldOfView(camera=camera, target=target, margin=margin), near_the_edge
    )

    assert camera.is_in_field_of_view(near_the_edge)
    assert observed == ObservationStateValues.FALSE


def test_out_of_field_of_view_observes_a_target_outside(camera_and_target):
    camera, target = camera_and_target
    beside = Point3(2.0, 0.0, 1.0, reference_frame=camera.root)

    observed = _observe_at(OutOfFieldOfView(camera=camera, target=target), beside)

    assert observed == ObservationStateValues.TRUE
