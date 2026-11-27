"""
Tasks keeping a robot that stands on its legs balanced while its upper body moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts, Task
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityScale
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.robots.walking import StandingBalance
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% the floor plan the balance is kept in


@dataclass
class FloorPlan:
    """
    Coordinates across the floor, seen from above: along and across the way a robot
    faces, measured from the point midway between its feet.
    """

    origin: Point3
    """
    The point midway between the feet, in the world's root frame.
    """

    forward: tuple[sm.Scalar, sm.Scalar]
    """
    The horizontal unit vector the robot faces along, in the world's root frame.
    """

    def coordinates(self, point: Point3) -> tuple[sm.Scalar, sm.Scalar]:
        """
        :param point: A point in the world's root frame.
        :return: How far the point lies ahead of and to the left of :attr:`origin`.
        """
        x, y = point.x - self.origin.x, point.y - self.origin.y
        cos, sin = self.forward
        return cos * x + sin * y, cos * y - sin * x


# %% the task


@dataclass(eq=False, repr=False, kw_only=True)
class CenterOfMassOverTheFeet(Task):
    """
    Keeps the center of mass of a robot standing on its feet above the box around both
    of them, seen from above, and bounds how fast it moves and speeds up across the
    floor, as :attr:`balance` says.

    An invariant rather than a goal: it observes nothing and is always true.
    """

    robot_root: KinematicStructureEntity
    """
    The root of the robot, whose branch is weighed, a held object included.
    """

    feet: list[Body]
    """
    The bodies the robot stands on.
    """

    balance: StandingBalance
    """
    How far inside the box the center of mass stays, and how gently it moves.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Task priority relative to other tasks.

    Above collision avoidance by default, because a robot that falls over avoids
    nothing.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Constrain the center of mass, along and across the way the robot faces, to the
        box around the feet shrunk by the margin, and its velocity and acceleration to
        the limits.
        """
        artifacts = MotionNodeArtifacts()
        floor_plan = self._floor_plan()
        world = self.context.world
        center_of_mass = floor_plan.coordinates(
            world.compose_center_of_mass_expression(self.robot_root, world.root)
        )
        corners = [
            floor_plan.coordinates(corner) for corner in self._corners_of_the_feet()
        ]
        for axis, direction in enumerate(("forward", "sideways")):
            lowest = reduce(sm.min, [corner[axis] for corner in corners])
            highest = reduce(sm.max, [corner[axis] for corner in corners])
            self._keep_within(
                artifacts,
                center_of_mass[axis],
                lowest + self.balance.support_margin,
                highest - self.balance.support_margin,
                direction,
            )
        artifacts.observation = sm.Scalar.const_true()
        return artifacts

    def _floor_plan(self) -> FloorPlan:
        """
        :return: The floor plan measured from midway between the feet, facing the way
            :attr:`robot_root` faces.
        """
        world = self.context.world
        feet = [
            world.compose_forward_kinematics_expression(world.root, foot).position
            for foot in self.feet
        ]
        origin = Point3(
            *(sum(foot[axis] for foot in feet) / len(feet) for axis in range(3)),
            reference_frame=world.root,
        )
        facing = world.compose_forward_kinematics_expression(
            world.root, self.robot_root
        ).rotation_matrix.x_vector()
        length = sm.sqrt(facing.x**2 + facing.y**2)
        return FloorPlan(origin=origin, forward=(facing.x / length, facing.y / length))

    def _corners_of_the_feet(self) -> list[Point3]:
        """
        :return: The corners of the box around each foot's collision shapes, in the
            world's root frame.
        """
        world = self.context.world
        return [
            world.compose_forward_kinematics_expression(world.root, foot).dot(corner)
            for foot in self.feet
            for corner in foot.collision.as_bounding_box_collection_in_frame(foot)
            .bounding_box()
            .get_points()
        ]

    def _keep_within(
        self,
        artifacts: MotionNodeArtifacts,
        coordinate: sm.Scalar,
        lowest: sm.Scalar,
        highest: sm.Scalar,
        name: str,
    ) -> None:
        """
        Keep `coordinate` between `lowest` and `highest`, and its velocity and
        acceleration within :attr:`balance`'s limits.
        """
        speed = self.balance.maximal_center_of_mass_speed
        acceleration = self.balance.maximal_center_of_mass_acceleration
        artifacts.constraints.add_inequality_constraint(
            task_expression=coordinate,
            lower_error=lowest - coordinate,
            upper_error=highest - coordinate,
            cruise_velocity=speed,
            velocity_scale=VelocityScale.TRANSLATION,
            quadratic_weight=self.weight,
            name=f"{name}/position",
        )
        artifacts.constraints.add_velocity_constraint(
            lower_velocity_limit=-speed,
            upper_velocity_limit=speed,
            task_expression=coordinate,
            velocity_scale=VelocityScale.TRANSLATION,
            quadratic_weight=self.weight,
            name=f"{name}/velocity",
        )
        artifacts.constraints.add_acceleration_constraint(
            lower_acceleration_limit=-acceleration,
            upper_acceleration_limit=acceleration,
            task_expression=coordinate,
            velocity_scale=VelocityScale.TRANSLATION,
            quadratic_weight=self.weight,
            name=f"{name}/acceleration",
        )
