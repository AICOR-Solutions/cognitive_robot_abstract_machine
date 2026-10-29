import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Human,
    Level,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

# %% elevator


def test_cabin_floor_is_level_with_the_ground_floor(arise_office_world, arise_office):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    cabin_bottom = float(elevator.root.global_transform.z) - elevator.scale.z / 2
    assert cabin_bottom == pytest.approx(arise_office.ground_floor_height)


def test_elevator_drives_to_the_first_floor_slab(arise_office_world, arise_office):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert elevator.drive_position_for_floor(first_floor) == pytest.approx(
        arise_office.first_floor_height
    )


def test_elevator_opens_towards_negative_y(arise_office_world):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    world_V_hole = arise_office_world.transform(
        elevator.hole_direction, arise_office_world.root
    )
    assert world_V_hole.to_np()[:3] == pytest.approx(Vector3(0, -1, 0).to_np()[:3])


def test_elevator_has_two_doors(arise_office_world):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    assert len(elevator.doors) == 2


# %% levels


def test_levels_start_at_the_slab_tops(arise_office_world, arise_office):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert float(ground_floor.floor_plane[0].z) == pytest.approx(
        arise_office.ground_floor_height
    )
    assert float(first_floor.floor_plane[0].z) == pytest.approx(
        arise_office.first_floor_height
    )


def test_level_under_a_person_is_the_storey_they_stand_on(
    arise_office_world, arise_office
):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    downstairs, upstairs = (
        Human.create_with_new_body_in_world(
            name=f"person_at_{height}",
            world=arise_office_world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=height),
        )
        for height in (
            arise_office.ground_floor_height,
            arise_office.first_floor_height,
        )
    )
    assert Level.under(downstairs.root) is ground_floor
    assert Level.under(upstairs.root) is first_floor


def test_level_under_an_entity_below_every_storey_is_unknown(arise_office_world):
    basement = Human.create_with_new_body_in_world(
        name="person_in_the_basement",
        world=arise_office_world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-3.0),
    )
    assert Level.under(basement.root) is None


def test_levels_are_spoken_of_by_the_storey_they_stand_for(arise_office_world):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert ground_floor.spoken_name == "ground floor"
    assert first_floor.spoken_name == "first floor"


def test_doorway_lies_in_the_middle_of_the_doors_on_the_cabin_floor(
    arise_office_world, arise_office
):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    world_T_doorway = arise_office_world.transform(
        elevator.doorway_pose, arise_office_world.root
    ).to_np()
    assert world_T_doorway[:3, 3] == pytest.approx(
        [
            float(arise_office.cabin_centre.x),
            float(arise_office.cabin_centre.y) - arise_office.cabin_scale.x / 2,
            arise_office.ground_floor_height,
        ]
    )
    # The office's doors face -y, so the doorway's x-axis points along +y into the cabin
    assert world_T_doorway[:3, 0] == pytest.approx([0.0, 1.0, 0.0])


def test_level_under_a_person_riding_the_elevator_is_the_storey_it_arrived_at(
    arise_office_world, arise_office
):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    rider = Human.create_with_new_body_in_world(
        name="rider",
        world=arise_office_world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            arise_office.cabin_centre.x,
            arise_office.cabin_centre.y,
            arise_office.ground_floor_height,
        ),
    )
    with arise_office_world.modify_world():
        arise_office_world.move_branch(rider.root, elevator.root)
    elevator.drive_to_floor(first_floor)
    assert Level.under(rider.root) is first_floor
