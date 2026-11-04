"""
Tiago drives out of the apartment through a doorway and back in again.

Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives the actual
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

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
    SimulatedPlanExecutor,
)
from coraplex.robot_plans.actions.core.navigation import PassDoorAction
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from semantic_digital_twin.predefined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from krrood_spatial_math.spatial_types import (
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
            x=-0.824, y=3.731, z=2.25 / 2, yaw=-0.138
        )
    )
    """
    Centre of the doorway, with the x-axis pointing through the wall into the apartment.

    Measured in the lab map at the second of the apartment's two doors.
    """

    world_T_apartment: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-1.270, y=-2.264, yaw=-1.6938
        )
    )
    """
    Pose of the apartment's origin in the lab map, fitted to the walls the robot mapped.
    """

    door_scale: Scale = field(default_factory=lambda: Scale(0.04, 0.91, 2.25))
    """
    Thickness, width and height of the door, sized to the real doorway.
    """

    room_distance_from_door: float = 2
    """
    How far from the doorway the robot waits and stops on the side of the apartment.
    """

    hallway_distance_from_door: float = 1.5
    """
    How far from the doorway the robot stops and waits on the side of the hallway.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing but a map body, which :meth:`populate_scene`
            spawns the apartment into.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The Tiago on its drive :attr:`room_distance_from_door` in front of the
            doorway in the apartment, facing it, which is also where the plan brings it
            back to.
        """
        world_T_start = (
            self.world_T_door
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.room_distance_from_door, yaw=np.pi
            )
        ).to_np()
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(world_T_start[0, 3]),
                y=float(world_T_start[1, 3]),
                yaw=float(np.arctan2(world_T_start[1, 0], world_T_start[0, 0])),
            ),
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(self.door_name)

    def populate_scene(self, world: World) -> None:
        """
        Spawn the apartment and a door into the doorway, with its leaf open.
        """
        world.merge_world_at_pose(
            ApartmentEnvironment().create_world(), self.world_T_apartment
        )
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
        if not issubclass(self.executor_type, SimulatedPlanExecutor):
            lidar = world.get_semantic_annotations_by_type(self.used_robot)[
                0
            ].mobile_base.lidar
            lidar.use_source(lidar.real_source(self.ros_node))
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Give the plan the Tiago in ``world``.
        """
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Pass the doorway to the outside and back into the apartment.
        """
        door = context.world.get_semantic_annotation_by_name(self.door_name)
        outside = Point3(
            -2 * self.hallway_distance_from_door, 0, 0, reference_frame=door.root
        )
        inside = Point3(
            2 * self.room_distance_from_door, 0, 0, reference_frame=door.root
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    PassDoorAction(
                        doors=[door],
                        destination=outside,
                        approach_distance=self.room_distance_from_door,
                        exit_distance=self.hallway_distance_from_door,
                    ),
                    PassDoorAction(
                        doors=[door],
                        destination=inside,
                        approach_distance=self.hallway_distance_from_door,
                        exit_distance=self.room_distance_from_door,
                    ),
                ]
            )
        )
        return statechart


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
    repetitions: int = 1,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot or simulates it.
    :param repetitions: How often to pass the doorway out and back in.
    """
    TiagoDoorPassingDemonstration(
        used_robot=Tiago, executor_type=executor_type, repetitions=repetitions
    ).run()


if __name__ == "__main__":
    main(executor_type=RealRobotPlanExecutor)
