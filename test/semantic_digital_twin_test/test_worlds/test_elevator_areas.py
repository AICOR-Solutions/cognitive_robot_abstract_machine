import pytest

from semantic_digital_twin.reasoning.elevator_areas import (
    ElevatorCabinArea,
    ElevatorWaitingArea,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.spatial_types import Point3

# %% fixtures


@pytest.fixture
def elevator(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(Elevator)[0]


@pytest.fixture
def ground_floor(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]


@pytest.fixture
def first_floor(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]


def head_of_person_at(world, x: float, y: float, floor_height: float) -> Point3:
    return Point3(x, y, floor_height + 1.65, reference_frame=world.root)


# %% cabin


def test_person_inside_the_cabin_is_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = head_of_person_at(
        arise_office_world,
        arise_office.cabin_centre.x + 0.25,
        arise_office.cabin_centre.y + 0.2,
        arise_office.ground_floor_height,
    )
    assert ElevatorCabinArea(elevator).contains(person)


def test_person_in_front_of_the_doors_is_not_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = head_of_person_at(
        arise_office_world, 0.0, 1.5, arise_office.ground_floor_height
    )
    assert not ElevatorCabinArea(elevator).contains(person)


def test_cabin_area_moves_with_the_cabin(
    arise_office_world, elevator, first_floor, arise_office
):
    upstairs = head_of_person_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.first_floor_height,
    )
    downstairs = head_of_person_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    elevator.drive_to_floor(first_floor)
    assert ElevatorCabinArea(elevator).contains(upstairs)
    assert not ElevatorCabinArea(elevator).contains(downstairs)


def test_cabin_area_footprint_is_the_cabin_floor(elevator, arise_office):
    corners = {
        (
            round(float(corner.x), 6),
            round(float(corner.y), 6),
            round(float(corner.z), 6),
        )
        for corner in ElevatorCabinArea(elevator).footprint()
    }
    assert corners == {
        (round(x, 6), round(y, 6), round(arise_office.ground_floor_height, 6))
        for x in (
            float(arise_office.cabin_centre.x) - elevator.scale.y / 2,
            float(arise_office.cabin_centre.x) + elevator.scale.y / 2,
        )
        for y in (
            float(arise_office.cabin_centre.y) - elevator.scale.x / 2,
            float(arise_office.cabin_centre.y) + elevator.scale.x / 2,
        )
    }


# %% waiting area


def test_person_in_front_of_the_doors_waits_on_its_level(
    arise_office_world, elevator, ground_floor, first_floor, arise_office
):
    person = head_of_person_at(
        arise_office_world, 0.45, 1.2, arise_office.ground_floor_height
    )
    assert ElevatorWaitingArea(elevator, ground_floor).contains(person)
    assert not ElevatorWaitingArea(elevator, first_floor).contains(person)


def test_person_inside_the_cabin_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    person = head_of_person_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    assert not ElevatorWaitingArea(elevator, ground_floor).contains(person)


def test_person_beyond_the_waiting_depth_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    area = ElevatorWaitingArea(elevator, ground_floor)
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    person = head_of_person_at(
        arise_office_world,
        0.0,
        door_y - area.depth - 0.1,
        arise_office.ground_floor_height,
    )
    assert not area.contains(person)


def test_waiting_area_footprint_lies_in_front_of_the_doors_on_the_level(
    elevator, ground_floor, arise_office
):
    area = ElevatorWaitingArea(elevator, ground_floor)
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    half_width = elevator.scale.y / 2 + area.side_margin
    corners = {
        (
            round(float(corner.x), 6),
            round(float(corner.y), 6),
            round(float(corner.z), 6),
        )
        for corner in area.footprint()
    }
    assert corners == {
        (round(x, 6), round(y, 6), round(arise_office.ground_floor_height, 6))
        for x in (-half_width, half_width)
        for y in (door_y - area.depth, door_y)
    }
