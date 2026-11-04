from copy import deepcopy

import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.goals.tracebot import InsertCylinder
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_insert_cylinder_with_tracy(tracy_world):
    world = deepcopy(tracy_world)
    robot = world.get_semantic_annotations_by_type(Tracy)[0]
    tool_frame = world.get_body_by_name("r_gripper_tool_frame")
    hole_point = Point3(0.8, -0.3, 0.88, reference_frame=world.root)
    cylinder_height = 0.1

    with world.modify_world():
        cylinder = Body(
            name=PrefixedName("cylinder"),
            visual=ShapeCollection([Cylinder(width=0.04, height=cylinder_height)]),
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=tool_frame,
                child=cylinder,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.0, 0.0, 0.02, reference_frame=tool_frame
                ),
            )
        )
        hole = Body(
            name=PrefixedName("hole"),
            visual=ShapeCollection([Cylinder(width=0.06, height=0.01)]),
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=hole,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=hole_point, reference_frame=world.root
                ),
            )
        )

    msc = MotionStatechart()
    goal = InsertCylinder(
        tip_link=cylinder,
        tip_P_tool=Point3(0.0, 0.0, cylinder_height / 2, reference_frame=cylinder),
        tip_V_axis=Vector3(0.0, 0.0, -1.0, reference_frame=cylinder),
        up_axis=Vector3(0.0, 0.0, 1.0, reference_frame=world.root),
        hole_point=hole_point,
        pre_grasp_height=0.1,
    )
    close_gripper = JointPositionList(
        goal_state=robot.right_arm.end_effector.get_joint_state_by_type(
            GripperState.CLOSE
        )
    )
    msc.add_node(sequence := Sequence([close_gripper, goal]))
    msc.add_node(EndMotion.when_true(sequence))

    executor = Executor(
        MotionStatechartContext(world=world),
    )
    executor.compile(motion_statechart=msc)
    executor.tick_until_end(3000)

    root_T_cylinder = world.compute_forward_kinematics(world.root, cylinder)
    root_V_axis = root_T_cylinder.rotation_matrix.z_vector()
    root_P_bottom = root_T_cylinder.position + root_V_axis * (cylinder_height / 2)
    assert np.allclose(root_P_bottom.to_np(), hole_point.to_np(), atol=0.02)
    assert np.allclose(root_V_axis.to_np(), Vector3(0.0, 0.0, -1.0).to_np(), atol=0.02)


def test_insert_cylinder_untilted_into_leaning_hole(tracy_world):
    """
    An untilted cylinder goes down a hole that leans off the vertical, staying lined up
    with it, without the joint velocities jumping once it is nearly lined up.
    """
    world = deepcopy(tracy_world)
    tool_frame = world.get_body_by_name("r_gripper_tool_frame")
    hole_point = Point3(0.8, -0.3, 0.88, reference_frame=world.root)
    cylinder_height = 0.1
    lean = 0.35
    up_axis = Vector3(np.sin(lean), 0.0, np.cos(lean), reference_frame=world.root)

    with world.modify_world():
        cylinder = Body(
            name=PrefixedName("cylinder"),
            visual=ShapeCollection([Cylinder(width=0.017, height=cylinder_height)]),
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=tool_frame,
                child=cylinder,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.0, 0.0, 0.02, reference_frame=tool_frame
                ),
            )
        )

    msc = MotionStatechart()
    goal = InsertCylinder(
        tip_link=cylinder,
        tip_P_tool=Point3(0.0, 0.0, cylinder_height / 2, reference_frame=cylinder),
        tip_V_axis=Vector3(0.0, 0.0, -1.0, reference_frame=cylinder),
        up_axis=up_axis,
        hole_point=hole_point,
        pre_grasp_height=0.1,
        tilt=0.0,
    )
    msc.add_node(goal)
    msc.add_node(EndMotion.when_true(goal))

    executor = Executor(
        MotionStatechartContext(world=world),
    )
    executor.compile(motion_statechart=msc)
    executor.tick_until_end(3000)

    root_T_cylinder = world.compute_forward_kinematics(world.root, cylinder)
    root_V_axis = root_T_cylinder.rotation_matrix.z_vector()
    root_P_bottom = root_T_cylinder.position + root_V_axis * (cylinder_height / 2)
    assert np.allclose(root_P_bottom.to_np(), hole_point.to_np(), atol=0.02)
    assert np.allclose(root_V_axis.to_np(), (-up_axis).to_np(), atol=0.02)
