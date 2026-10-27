import pytest

from semantic_digital_twin.exceptions import MissingSemanticAnnotationError
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    ElevatorCabinArea,
    FirstFloor,
    GroundFloor,
    Human,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale

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


def person_standing_at(world, x: float, y: float, floor_height: float) -> Human:
    """
    A person the size of an adult standing on ``floor_height`` at ``x``, ``y``.
    """
    scale = Scale(0.4, 0.4, 1.7)
    return Human.create_with_new_body_in_world(
        name=f"person_{x}_{y}_{floor_height}",
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, floor_height + scale.z / 2
        ),
        scale=scale,
    )


def rounded(corners) -> set[tuple[float, float, float]]:
    return {
        (
            round(float(corner.x), 6),
            round(float(corner.y), 6),
            round(float(corner.z), 6),
        )
        for corner in corners
    }


# %% cabin


def test_person_inside_the_cabin_is_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x + 0.25,
        arise_office.cabin_centre.y + 0.2,
        arise_office.ground_floor_height,
    )
    assert InsideRegion(person.root, elevator.cabin_area.root)()


def test_person_in_front_of_the_doors_is_not_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = person_standing_at(
        arise_office_world, 0.0, 1.5, arise_office.ground_floor_height
    )
    assert not InsideRegion(person.root, elevator.cabin_area.root)()


def test_cabin_area_moves_with_the_cabin(
    arise_office_world, elevator, first_floor, arise_office
):
    upstairs = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.first_floor_height,
    )
    downstairs = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    elevator.drive_to_floor(first_floor)
    assert InsideRegion(upstairs.root, elevator.cabin_area.root)()
    assert not InsideRegion(downstairs.root, elevator.cabin_area.root)()


def test_cabin_area_footprint_is_the_cabin_floor(elevator, arise_office):
    assert rounded(elevator.cabin_area.footprint()) == {
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


def test_elevator_without_cabin_area_reports_it_missing(arise_office_world, elevator):
    with arise_office_world.modify_world():
        arise_office_world.remove_semantic_annotation(elevator.cabin_area)
    with pytest.raises(MissingSemanticAnnotationError):
        elevator.cabin_area


# %% waiting area


def test_person_in_front_of_the_doors_waits_on_its_level(
    arise_office_world, elevator, ground_floor, first_floor, arise_office
):
    person = person_standing_at(
        arise_office_world, 0.45, 1.2, arise_office.ground_floor_height
    )
    assert InsideRegion(person.root, elevator.waiting_area_on(ground_floor).root)()
    assert not InsideRegion(person.root, elevator.waiting_area_on(first_floor).root)()


def test_person_inside_the_cabin_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    person = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    assert not InsideRegion(person.root, elevator.waiting_area_on(ground_floor).root)()


def test_person_beyond_the_waiting_depth_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    person = person_standing_at(
        arise_office_world,
        0.0,
        door_y - arise_office.waiting_area_depth - 0.3,
        arise_office.ground_floor_height,
    )
    assert not InsideRegion(person.root, elevator.waiting_area_on(ground_floor).root)()


def test_waiting_area_footprint_lies_in_front_of_the_doors_on_the_level(
    elevator, ground_floor, arise_office
):
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    half_width = elevator.scale.y / 2 + arise_office.waiting_area_side_margin
    assert rounded(elevator.waiting_area_on(ground_floor).footprint()) == {
        (round(x, 6), round(y, 6), round(arise_office.ground_floor_height, 6))
        for x in (-half_width, half_width)
        for y in (door_y - arise_office.waiting_area_depth, door_y)
    }


def test_waiting_area_stays_on_its_level_while_the_cabin_rides(
    elevator, ground_floor, first_floor, arise_office
):
    before = rounded(elevator.waiting_area_on(ground_floor).footprint())
    elevator.drive_to_floor(first_floor)
    assert rounded(elevator.waiting_area_on(ground_floor).footprint()) == before


def test_office_spawns_one_cabin_area_and_a_waiting_area_per_level(
    arise_office_world, elevator, ground_floor, first_floor
):
    assert [
        area.elevator
        for area in arise_office_world.get_semantic_annotations_by_type(
            ElevatorCabinArea
        )
    ] == [elevator]
    assert elevator.waiting_area_on(ground_floor).level is ground_floor
    assert elevator.waiting_area_on(first_floor).level is first_floor
