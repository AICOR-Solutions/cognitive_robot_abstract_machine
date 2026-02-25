import os
import sys
import traceback

import rclpy
from rclpy.executors import SingleThreadedExecutor
import threading

from ament_index_python.packages import get_package_share_directory

# giskard / motion statechart
from giskardpy_ros.python_interface.python_interface import GiskardWrapper
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionStraight,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList

from krrood.symbolic_math.symbolic_math import trinary_logic_and

# semantic digital twin
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.robots.ur import UR10Bolt
from semantic_digital_twin.datastructures.joint_state import JointState

# utils for loading xacro (same as used in world_utils.py)
from giskardpy_ros.utils.utils import load_xacro


# Minimal inline versions of WorldObjectSpec + add_objects to avoid importing world_utils.py
class WorldBuildError(Exception):
    pass


class WorldObjectSpec:
    def __init__(
        self,
        urdf_path: str,
        prefix: str | None = None,
        parent: KinematicStructureEntity | None = None,
        transform: HomogeneousTransformationMatrix | None = None,
    ) -> None:
        self.urdf_path = urdf_path
        self.prefix = prefix
        self.parent = parent
        self.transform = transform or HomogeneousTransformationMatrix.from_xyz_rpy()


def add_objects(world: World, objects: list[WorldObjectSpec]) -> None:
    if not objects:
        return
    with world.modify_world():
        for spec in objects:
            if not spec.urdf_path:
                raise WorldBuildError("Missing URDF path for world object.")
            urdf_text = load_xacro(spec.urdf_path)
            object_world = URDFParser(urdf_text, prefix=spec.prefix).parse()
            # skip if same root already exists
            root_name = object_world.root.name
            if world.get_kinematic_structure_entities_by_name(root_name):
                continue
            parent_entity = spec.parent or world.root
            fixed = FixedConnection(
                parent=parent_entity,
                child=object_world.root,
                parent_T_connection_expression=spec.transform,
            )
            world.merge_world(object_world, fixed)


def build_simple_approach_statechart(
    world: World,
    base_link: KinematicStructureEntity,
    tool_link: KinematicStructureEntity,
    approach_points: list[Point3],
    align_vector: Vector3,
) -> MotionStatechart:
    """Inline copy of the essential 'move_approach' logic from skills.py.

    Creates a two-step Cartesian approach with plane alignment.
    """
    assert len(approach_points) == 2, "approach_points must contain exactly 2 points"

    pos1 = CartesianPositionStraight(
        name="position1",
        root_link=base_link,
        tip_link=tool_link,
        goal_point=world.transform(approach_points[0], world.root),
    )
    pos2 = CartesianPositionStraight(
        name="position2",
        root_link=base_link,
        tip_link=tool_link,
        goal_point=world.transform(approach_points[1], world.root),
    )

    align_planes = AlignPlanes(
        root_link=base_link,
        tip_link=tool_link,
        goal_normal=world.transform(align_vector, world.root),
        tip_normal=Vector3(0, 0, 1, reference_frame=tool_link),
        reference_velocity=0.5,
    )

    msc = MotionStatechart()
    msc.add_nodes([pos1, pos2, align_planes])
    # gate transition of pos2 on both first pos and alignment
    pos1.end_condition = trinary_logic_and(
        pos1.observation_variable, align_planes.observation_variable
    )
    pos2.start_condition = trinary_logic_and(
        pos1.observation_variable, align_planes.observation_variable
    )
    msc.add_node(EndMotion.when_all_true([pos2, align_planes]))
    return msc


def main():
    # Init ROS 2 and node
    rclpy.init()
    node = rclpy.create_node("repro_ur_approach")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    # Spin the node in the background so that callbacks/services work
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Connect to running giskard and fetch its world/robot (same as ur_demo external path)
    giskard = GiskardWrapper(node)
    world = giskard.world

    # Add the same workobject as in ur_demo.py
    workobject_spec = WorldObjectSpec(
        urdf_path=os.path.join("workobject.urdf"),
        transform=HomogeneousTransformationMatrix.from_xyz_rpy(y=1.0, x=0.1, yaw=3.14),
    )
    add_objects(world, [workobject_spec])

    # Set initial panel orientation like in ur_demo
    try:
        world.get_connection_by_name("column_to_panel").position = 3.14
    except Exception:
        pass  # ignore if not present yet

    # Get robot semantic annotation and links
    robot = world.get_semantic_annotations_by_type(UR10Bolt)[0]
    base_link = robot.root
    tool_link = list(robot.manipulators)[0].tool_frame

    # Move to the initial viewing pose (same as UR demo's DemoConfig.view_pose)
    try:
        view_pose = {
            "shoulder_pan_joint": -1.54,
            "shoulder_lift_joint": -1.52,
            "elbow_joint": -1.31,
            "wrist_1_joint": -1.0,
            "wrist_2_joint": 1.6,
            "wrist_3_joint": -0.008,
        }

        joint_goal_state = JointState.from_mapping(
            {world.get_connection_by_name(k): v for k, v in view_pose.items()}
        )

        joint_msc = MotionStatechart()
        joint_msc.add_node(joint_goal := JointPositionList(goal_state=joint_goal_state))
        joint_msc.add_node(EndMotion.when_true(joint_goal))

        print("Moving to initial viewing pose...")
        giskard.execute(joint_msc)
        print("Reached initial viewing pose.")
    except Exception:
        print("Failed to move to initial viewing pose; proceeding anyway.")

    # Choose a segment and create a simple approach towards it
    segment = world.get_body_by_name("seg_top")

    # Two-point approach in segment frame: descend along local +Z towards surface
    approach_height = 0.15
    approach_points = [
        Point3(z=approach_height, reference_frame=segment),
        Point3(z=0.0, reference_frame=segment),
    ]
    align_vec = Vector3(0, 0, -1, reference_frame=segment)

    msc = build_simple_approach_statechart(
        world=world,
        base_link=base_link,
        tool_link=tool_link,
        approach_points=approach_points,
        align_vector=align_vec,
    )

    # Execute and let any internal errors surface
    try:
        print("Executing approach...")
        giskard.execute(msc)
        print("Approach finished.")
    except Exception as e:
        print("Exception during execution:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean shutdown of ROS resources
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""
Before running this have giskard standalone running via:
python ur10_velocity.py /home/path/to/this/test/folder/ur10_femto_bolt.urdf.xacro --standalone

the ur10_velocity file can be found in this repo:
git@github.com:AICOR-Solutions/giskardpy_ros.git branch cleanup

at the location: giskardpy_ros/scripts/other_robots/ur10


The qp not solvable error happens kinda regularly for me. Bu i assume it only happens because the velocity limit at
cognitive_robot_abstract_machine/semantic_digital_twin/src/semantic_digital_twin/robots/ur.py

is set to 0.1
"""
