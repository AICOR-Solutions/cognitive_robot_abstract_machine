import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.spatial_types import Vector3

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
