from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import DoorLeafNotFoundError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% a sliding door in an otherwise empty world


@pytest.fixture
def sliding_door() -> Door:
    """
    A door whose leaf slides one meter along its own y axis, with a handle.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
            parent_connection_specification=PrismaticConnectionSpecification(
                axis=Vector3.Y(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=0.0, upper_position=1.0
                ),
            ),
            scale=Scale(0.05, 1, 2),
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        door.add(handle)
    return door


# %% what moves and what stays


def test_the_movable_joint_of_a_door_connects_its_frame_and_its_leaf(sliding_door):
    joint = sliding_door.movable_joint

    assert isinstance(joint, PrismaticConnection)
    assert joint.parent is sliding_door.root
    assert joint.child is sliding_door.leaf


def test_the_frame_of_a_door_is_fixed_to_its_parent(sliding_door):
    assert isinstance(sliding_door.root.parent_connection, FixedConnection)


def test_the_leaf_of_a_door_carries_its_geometry(sliding_door):
    assert sliding_door.leaf.collision.shapes
    assert not sliding_door.root.collision.shapes


def test_the_entry_way_of_a_door_stays_put_when_the_door_opens(sliding_door):
    world_T_entry_way = sliding_door.entry_way.root.global_transform.to_np()

    sliding_door.movable_joint.position = 1.0

    np.testing.assert_allclose(
        sliding_door.entry_way.root.global_transform.to_np(), world_T_entry_way
    )


def test_the_handle_of_a_door_moves_with_its_leaf(sliding_door):
    assert sliding_door.handle.root.parent_kinematic_structure_entity is (
        sliding_door.leaf
    )


# %% moving the whole door


def test_adding_a_door_to_an_elevator_carries_everything_the_door_holds(
    sliding_door,
):
    world = sliding_door._world
    with world.modify_world():
        elevator = Elevator.create_with_new_body_in_world(
            name="elevator", world=world, scale=Scale(2, 2, 2)
        )
        elevator.add(sliding_door)

    door_branch = world.get_kinematic_structure_entities_of_branch(elevator.root)
    assert sliding_door.leaf in door_branch
    assert sliding_door.handle.root in door_branch
    assert sliding_door.entry_way.root in door_branch


# %% doors around an existing leaf


def test_a_door_built_around_a_leaf_keeps_the_leafs_joint_and_pose():
    world = World.create_with_root_body("root")
    with world.modify_world():
        leaf_body = Body(name=PrefixedName("leaf"))
        world.add_body(leaf_body)
        PrismaticConnectionSpecification(axis=Vector3.Y()).connect(
            world,
            child=leaf_body,
            parent=world.root,
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0),
        )
    joint = leaf_body.parent_connection
    world_T_leaf = leaf_body.global_transform.to_np()

    door = Door.from_leaf(leaf_body)

    assert door.leaf is leaf_body
    assert type(door.movable_joint) is type(joint)
    assert door.movable_joint.dof.id == joint.dof.id
    assert door.movable_joint.parent is door.root
    np.testing.assert_allclose(leaf_body.global_transform.to_np(), world_T_leaf)


def test_a_door_whose_frame_holds_no_leaf_says_so():
    world = World.create_with_root_body("root")
    with world.modify_world():
        frame = Body(name=PrefixedName("frame"))
        world.add_body(frame)
        world.add_connection(FixedConnection(parent=world.root, child=frame))
    door = Door(root=frame)

    with pytest.raises(DoorLeafNotFoundError) as raised:
        door.leaf

    assert raised.value.door is door
