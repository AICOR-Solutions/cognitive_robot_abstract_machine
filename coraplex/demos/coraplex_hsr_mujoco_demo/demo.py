"""
The HSR carries a milk carton from a counter to a table in MuJoCo. Run it from this
folder::

    python demo.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum

import mujoco
import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.querying.predicates import IsAmongTheClosestGraspsTo
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from, under the coraplex resources.
    """

    MILK = os.path.join("objects", "milk.stl")

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", self.value
        )


@dataclass
class Stand:
    """
    A box standing on the floor, at a distance from the robot straight ahead of or
    behind it, with the robot facing it from its long side.
    """

    name: str
    """
    The name of the stand's body.
    """

    distance: float
    """
    Where the stand's centre is along the robot's forward axis, in meters; negative
    behind the robot.
    """

    scale: Scale
    """
    The stand's depth along the robot's forward axis, its width and its height.
    """

    @property
    def facing_yaw(self) -> float:
        """
        Which way the robot faces when it faces the stand.
        """
        return 0.0 if self.distance > 0 else np.pi

    def point_behind_the_near_edge(self, depth: float) -> float:
        """
        :param depth: How far behind the edge nearest the robot's start, in meters.
        :return: Where that point is along the robot's forward axis.
        """
        direction = np.sign(self.distance)
        return self.distance - direction * (self.scale.x / 2 - depth)

    def spawn(self, world: World) -> None:
        """
        Stand the box on the floor.

        :param world: The world to spawn into.
        """
        BodySpecification.box(self.name, self.scale).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.distance, z=self.scale.z / 2
            ),
        )


# %% the demonstration


@dataclass
class MilkTransportDemonstration:
    """
    The HSR carries the milk from the counter to the table, in MuJoCo.
    """

    counter: Stand = field(
        default_factory=lambda: Stand("counter", 1.0, Scale(0.6, 1.0, 0.75))
    )
    """
    The counter the milk starts on, in front of the robot.
    """

    table: Stand = field(
        default_factory=lambda: Stand("table", -1.0, Scale(0.6, 1.0, 0.7))
    )
    """
    The table the milk is carried to, behind the robot.
    """

    milk_depth: float = 0.15
    """
    How far behind the near edge of the counter the milk starts, and of the table it is
    set down, in meters.
    """

    standing_distance: float = 0.5
    """
    How far from the milk the robot's base stands to take it and to set it down, in
    meters.
    """

    grasp_height: float = 0.05
    """
    How far above the milk's origin the hand takes hold of it, in meters.
    """

    headless: bool = False
    """
    Whether the physics runs without a viewer.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a held carton creep between the
    fingers.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held carton stays where the
    fingers hold it.
    """

    settling_duration: timedelta = timedelta(seconds=1)
    """
    How long the milk is left to come to rest on the counter before the robot moves.
    """

    def build_world(self) -> World:
        """
        :return: The floor, the counter, the table, the milk standing on the counter and
            the HSR between them, all below a root body.
        """
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
        BodySpecification.box("floor", Scale(4.0, 4.0, 0.02)).spawn(
            world, parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01)
        )
        self.counter.spawn(world)
        self.table.spawn(world)
        milk = SemanticAnnotationWithRootSpecification(
            name="milk",
            semantic_annotation_type=Milk,
            root_specification=BodySpecification.mesh(
                "milk",
                SceneFile.MILK.path,
                connection_specification=Connection6DoFSpecification(),
            ),
        ).spawn(world)
        ContactParameters.create_for_grasped_object().apply_to([milk.root])
        milk.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.counter.point_behind_the_near_edge(self.milk_depth),
                z=self.counter.scale.z + self._height_of_the_milk_origin(milk),
                reference_frame=world.root,
            )
        )
        RobotSpecification(HSRB).spawn(world)
        return world

    @staticmethod
    def _height_of_the_milk_origin(milk: Milk) -> float:
        """
        :param milk: The milk.
        :return: How high the milk's origin lies above the bottom of the carton.
        """
        return -(
            milk.root.collision.as_bounding_box_collection_in_frame(milk.root)
            .bounding_box()
            .min_z
        )

    def build_plan(self, context: Context) -> Plan:
        """
        Raise the torso, park the arm, take the milk from the counter, carry it to the
        table and set it down there.
        """
        world = context.world
        hsrb = context.robot
        [milk] = world.get_semantic_annotations_by_type(Milk)
        [arm] = hsrb.all_arms
        taking_pose = self._standing_pose_facing(self.counter, world)
        target = Pose.from_xyz_rpy(
            x=self.table.point_behind_the_near_edge(self.milk_depth),
            z=self.table.scale.z + self._height_of_the_milk_origin(milk),
            yaw=self.table.facing_yaw,
            reference_frame=world.root,
        )
        return sequential(
            [
                MoveTorsoAction(TorsoState.HIGH),
                ParkArmsAction(hsrb.all_arms),
                NavigateAction(taking_pose),
                PickUpAction(grasp=self._grasp_from(taking_pose, milk), arm=arm),
                NavigateAction(self._standing_pose_facing(self.table, world)),
                PlaceAction(object_designator=milk, target_location=target),
                ParkArmsAction(hsrb.all_arms),
            ],
            context=context,
        ).plan

    def _standing_pose_facing(self, stand: Stand, world: World) -> Pose:
        """
        :param stand: The stand the robot takes the milk from or sets it down on.
        :param world: The world the stand is in.
        :return: Where the robot's base stands, facing the stand,
            :attr:`standing_distance` before the milk's spot on it.
        """
        milk_spot = stand.point_behind_the_near_edge(self.milk_depth)
        return Pose.from_xyz_rpy(
            x=milk_spot - np.sign(stand.distance) * self.standing_distance,
            yaw=stand.facing_yaw,
            reference_frame=world.root,
        )

    def _grasp_from(self, standing_pose: Pose, milk: Milk) -> GraspCandidate:
        """
        :param standing_pose: Where the robot stands to take the milk.
        :param milk: The milk.
        :return: The milk's grasp closest to the robot standing there, raised
            :attr:`grasp_height` above the milk's origin.
        """
        grasps = milk.grasp_candidates()
        [closest] = [
            grasp
            for grasp in grasps
            if IsAmongTheClosestGraspsTo(
                grasp, standing_pose, grasps, number_of_grasps=1
            )()
        ]
        milk_T_raised_grasp = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.grasp_height, reference_frame=milk.root
            )
            @ milk.root._world.transform(
                closest.grasp_pose, milk.root
            ).homogeneous_matrix
        )
        return replace(closest, grasp_pose=milk_T_raised_grasp.pose)

    def run(self) -> World:
        """
        Build the scene, and carry the milk to the table in MuJoCo.

        :return: The world the milk was carried in.
        """
        world = self.build_world()
        [hsrb] = world.get_semantic_annotations_by_type(HSRB)
        plan = self.build_plan(Context(world=world, robot=hsrb))
        simulation = MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(self.settling_duration)
            with PhysicallySimulatedRobot(simulation=simulation):
                plan.perform()
        finally:
            simulation.stop_simulation()
        return world


def main(headless: bool = False) -> World:
    """
    Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the milk was carried in.
    """
    return MilkTransportDemonstration(headless=headless).run()


if __name__ == "__main__":
    main()
