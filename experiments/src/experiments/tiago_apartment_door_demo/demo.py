"""
Tiago drives out of the apartment through a doorway and back in again.

Running with :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` drives the actual
robot, fetches the world from the running controller and reads the base lidar from the
robot's ``/scan``. The default runs the whole plan in simulation against a world built from
the Tiago URDF, so nothing on the network is needed.

The apartment's wall mesh only models the frame of the doorway, so the demonstration adds a
door annotation into it. Its leaf stays open in the model: whether the real door is open is
decided by the lidar, which the robot waits on in front of the doorway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import PassDoorAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.predefined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale

# %% demonstration


@dataclass
class TiagoDoorPassingDemonstration(RobotDemonstration):
    """
    Tiago passes the left doorway of the apartment's long wall to the outside and comes
    back in again.

    The plan ends where it started, so it can be repeated against the same scene.
    """

    ros_node_name: str = "tiago_door_demo_node"

    door_name: str = "apartment_door"
    """
    Name of the door this demonstration adds, which also marks whether the scene was
    already spawned.
    """

    world_T_door: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-6.25, y=-0.385, z=2.25 / 2, yaw=np.pi / 2
        )
    )
    """
    Centre of the doorway, with the x-axis pointing through the wall into the apartment.

    Measured from the frame of the left of the two doorways in the apartment's wall
    mesh, as :class:`ApartmentEnvironment` places it.
    """

    door_scale: Scale = field(default_factory=lambda: Scale(0.04, 1.46, 2.25))
    """
    Thickness, width and height of the door, sized to the doorway in the wall mesh.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-6.25, y=0.615, yaw=-np.pi / 2
        )
    )
    """
    Where a simulated Tiago starts: :attr:`distance_from_door` in front of the doorway
    in the apartment, facing it, which is also where the plan brings it back to.
    """

    distance_from_door: float = 1.5
    """
    How far from the doorway the robot waits in front of it and stops behind it.
    """

    def build_simulated_world(self) -> World:
        """
        Put the Tiago on its drive in a world holding nothing but a map body.
        """
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=self.odom_T_robot_start,
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(self.door_name)

    def populate_scene(self, world: World) -> None:
        """
        Spawn the apartment and a door into the doorway, with its leaf open.
        """
        ApartmentEnvironment().populate(world)
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name=self.door_name,
                world=world,
                world_root_T_self=self.world_T_door,
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.Y(),
                    dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                        lower_position=0.0, upper_position=self.door_scale.y
                    ),
                ),
                scale=self.door_scale,
            )
        door.movable_joint.position = self.door_scale.y

    def acquire_world(self) -> World:
        """
        On a real run, the base lidar is read from the robot instead of the world.
        """
        world = super().acquire_world()
        if self.execution_type is ExecutionType.REAL:
            lidar = world.get_semantic_annotations_by_type(self.used_robot)[
                0
            ].mobile_base.lidar
            lidar.use_source(lidar.real_source(self.ros_node))
        return world

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the Tiago in ``world``.

        ..note:: The ROS node has to be in the context for a real robot.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=self.debug,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Pass the doorway to the outside and back into the apartment.
        """
        door = context.world.get_semantic_annotation_by_name(self.door_name)
        outside = Point3(-2 * self.distance_from_door, 0, 0, reference_frame=door.root)
        inside = Point3(2 * self.distance_from_door, 0, 0, reference_frame=door.root)

        VizMarkerPublisher(_world=context.world, node=context.ros_node)

        return sequential(
            [
                ParkArmsAction(context.robot.all_arms),
                PassDoorAction(
                    [door],
                    destination=outside,
                    approach_distance=self.distance_from_door,
                    exit_distance=self.distance_from_door,
                ),
                PassDoorAction(
                    [door],
                    destination=inside,
                    approach_distance=self.distance_from_door,
                    exit_distance=self.distance_from_door,
                ),
            ],
            context=context,
        )


def main(
    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED,
    repetitions: int = 1,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param repetitions: How often to pass the doorway out and back in.
    """
    TiagoDoorPassingDemonstration(
        used_robot=Tiago, execution_type=execution_type, repetitions=repetitions
    ).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.PHYSICALLY_SIMULATED)
