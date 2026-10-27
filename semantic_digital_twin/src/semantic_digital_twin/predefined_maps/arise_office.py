from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    ElevatorCabinArea,
    ElevatorWaitingArea,
    FirstFloor,
    Floor,
    GroundFloor,
    Level,
    Wall,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Region

# %% the office of the Gazebo simulator


class WallSide(StrEnum):
    """
    The edge of a storey an outer wall stands on, by compass direction with north
    towards +y.
    """

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class AriseOfficeBuilding:
    """
    World-independent description of the two-storey office of the TIAGo ARISE Gazebo
    simulator (``office_area.sdf`` and ``elevator.urdf`` of the
    ``tiago_arise_navigation`` package).

    The world root is the Gazebo world frame. The elevator's doors face -y.

    .. note:: The Gazebo elevator joint is 0 when the cabin serves the ground floor, so
        it equals the semantic drive position minus :attr:`ground_floor_height`.
    """

    storey_scale: Scale = field(default_factory=lambda: Scale(16.0, 8.0, 4.0))
    """
    Extents of each storey around the world origin and the height of its walls.
    """

    ground_floor_height: float = 0.1
    """
    Height of the top of the ground floor slab.
    """

    first_floor_height: float = 4.15
    """
    Height of the top of the first floor slab.
    """

    wall_thickness: float = 0.1
    """
    Thickness of the outer walls.
    """

    cabin_centre: Point3 = field(default_factory=lambda: Point3(0.0, 2.925, 0.0))
    """
    Centre of the elevator cabin's footprint in the world frame; its height is ignored.
    """

    cabin_scale: Scale = field(default_factory=lambda: Scale(1.6, 1.6, 2.0))
    """
    Depth from the doors to the back wall, width and height of the elevator cabin.
    """

    cabin_wall_thickness: float = 0.05
    """
    Thickness of the cabin's walls.
    """

    cabin_door_thickness: float = 0.04
    """
    Thickness of each of the two sliding cabin doors.
    """

    cabin_maximum_speed: float = 0.5
    """
    Maximum speed of the elevator drive, as limited in ``elevator.urdf``.
    """

    shaft_opening_width: float = 1.85
    """
    Width of the gap the first floor slab leaves for the shaft.
    """

    shaft_front: float = 2.125
    """
    Y at which the first floor slab in front of the shaft ends.
    """

    waiting_area_depth: float = 1.5
    """
    How far the waiting area on each level reaches out from the elevator's doors.
    """

    waiting_area_side_margin: float = 0.3
    """
    How far the waiting area on each level reaches beyond the cabin's width on either
    side.
    """

    waiting_area_height: float = 2.5
    """
    How far the waiting area on each level reaches up from the level's floor.
    """

    def spawn(self, world: World, name: str = "arise_office") -> Elevator:
        """
        Materialize the office in ``world``: floor slabs and outer walls of both
        storeys, a level annotation per storey, and the elevator serving the ground
        floor with the area inside its cabin and a waiting area in front of it on each
        level.

        :param world: The world the office is added to.
        :param name: Prefix for the names of every entity this office creates.
        :return: The elevator.
        """
        self._spawn_slab(
            world,
            f"{name}_ground_floor",
            Point3(0, 0, self.ground_floor_height),
            self.storey_scale.x,
            self.storey_scale.y,
        )
        self._spawn_first_floor_slabs(world, f"{name}_first_floor")
        for level_type, height in (
            (GroundFloor, self.ground_floor_height),
            (FirstFloor, self.first_floor_height),
        ):
            self._spawn_outer_walls(world, f"{name}_{level_type.__name__}", height)
            self._spawn_level(
                world, f"{name}_{level_type.__name__}_level", level_type, height
            )
        elevator = self._spawn_elevator(world, f"{name}_elevator")
        ElevatorCabinArea.spawn_for(elevator)
        for level in world.get_semantic_annotations_by_type(Level):
            ElevatorWaitingArea.spawn_for(
                elevator,
                level,
                depth=self.waiting_area_depth,
                side_margin=self.waiting_area_side_margin,
                height=self.waiting_area_height,
            )
        return elevator

    # %% storeys

    def _spawn_first_floor_slabs(self, world: World, name: str) -> None:
        """
        Three slabs around the shaft: one on either side and one in front of it.
        """
        side_width = (self.storey_scale.x - self.shaft_opening_width) / 2
        side_centre = (self.shaft_opening_width + side_width) / 2
        front_depth = self.shaft_front + self.storey_scale.y / 2
        height = self.first_floor_height
        self._spawn_slab(
            world,
            f"{name}_left",
            Point3(-side_centre, 0, height),
            side_width,
            self.storey_scale.y,
        )
        self._spawn_slab(
            world,
            f"{name}_right",
            Point3(side_centre, 0, height),
            side_width,
            self.storey_scale.y,
        )
        self._spawn_slab(
            world,
            f"{name}_front",
            Point3(0, self.shaft_front - front_depth / 2, height),
            self.shaft_opening_width,
            front_depth,
        )

    @staticmethod
    def _spawn_slab(
        world: World, name: str, world_P_centre: Point3, width: float, depth: float
    ) -> None:
        """
        A flat floor whose top is at the height of ``world_P_centre``.
        """
        Floor.get_annotation_specification(
            name,
            Floor.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(width, depth, 0)
            ),
        ).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                world_P_centre.x, world_P_centre.y, world_P_centre.z
            ),
        )

    def _spawn_outer_walls(self, world: World, name: str, height: float) -> None:
        """
        A wall along each edge of the storey whose floor top is at ``height``.
        """
        half_x, half_y = self.storey_scale.x / 2, self.storey_scale.y / 2
        for side, world_T_wall, length in (
            (
                WallSide.NORTH,
                HomogeneousTransformationMatrix.from_xyz_rpy(0, half_y, yaw=np.pi / 2),
                self.storey_scale.x,
            ),
            (
                WallSide.SOUTH,
                HomogeneousTransformationMatrix.from_xyz_rpy(0, -half_y, yaw=np.pi / 2),
                self.storey_scale.x,
            ),
            (
                WallSide.EAST,
                HomogeneousTransformationMatrix.from_xyz_rpy(half_x, 0),
                self.storey_scale.y,
            ),
            (
                WallSide.WEST,
                HomogeneousTransformationMatrix.from_xyz_rpy(-half_x, 0),
                self.storey_scale.y,
            ),
        ):
            Wall.get_annotation_specification(
                f"{name}_wall_{side}",
                Wall.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(self.wall_thickness, length, self.storey_scale.z)
                ),
            ).spawn(
                world,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=height + self.storey_scale.z / 2
                )
                @ world_T_wall,
            )

    def _spawn_level(
        self, world: World, name: str, level_type: type[Level], height: float
    ) -> None:
        """
        A region spanning one storey from the top of its floor up, annotated as a level.
        """
        with world.modify_world():
            region = Region(
                name=PrefixedName(name),
                area=ShapeCollection(shapes=[Box(scale=self.storey_scale)]),
            )
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=region,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=height + self.storey_scale.z / 2
                    ),
                )
            )
            world.add_semantic_annotation(level_type(root=region))

    # %% elevator

    def _spawn_elevator(self, world: World, name: str) -> Elevator:
        """
        The cabin on a vertical drive with two sliding doors in its opening, built like
        the elevator of the multi-storey test building.

        The cabin's x-axis points from its doors into the cabin.
        """
        elevator_world = World.create_with_root_body(f"{name}_shaft")
        with elevator_world.modify_world():
            elevator = Elevator.get_annotation_specification(
                name,
                Elevator.get_default_root_kinematic_structure_entity_specification(
                    scale=self.cabin_scale, wall_thickness=self.cabin_wall_thickness
                ),
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                        maximum_speed=self.cabin_maximum_speed
                    ),
                ),
            ).spawn(elevator_world)
            door_scale = Scale(
                self.cabin_door_thickness, self.cabin_scale.y / 2, self.cabin_scale.z
            )
            for i, door_y in enumerate(
                (-self.cabin_scale.y / 4, self.cabin_scale.y / 4)
            ):
                elevator.add(
                    Door.create_with_new_body_in_world(
                        name=f"{name}_door{i}",
                        world=elevator_world,
                        world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            Point3(-self.cabin_scale.x / 2, door_y, 0),
                            reference_frame=elevator_world.root,
                        ),
                        parent_connection_specification=PrismaticConnectionSpecification(
                            axis=(Vector3.Y() * ((-1) ** (i + 1))),
                            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                                lower_position=0.0, upper_position=door_scale.y
                            ),
                        ),
                        scale=door_scale,
                    )
                )
            elevator_world.add_semantic_annotation(elevator)

        world.merge_world(
            elevator_world,
            FixedConnection(
                parent=world.root,
                child=elevator_world.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.cabin_centre.x,
                    self.cabin_centre.y,
                    self.cabin_scale.z / 2,
                    yaw=np.pi / 2,
                ),
            ),
        )
        elevator = world.get_semantic_annotation_by_name(name)
        elevator.drive_to_floor(world.get_semantic_annotations_by_type(GroundFloor)[0])
        return elevator

    def create_world(self, root_name: str = "map") -> World:
        """
        :param root_name: Name of the world's root body, the frame the office is in.
        :return: A world holding nothing but this office.
        """
        world = World.create_with_root_body(root_name)
        self.spawn(world)
        return world
