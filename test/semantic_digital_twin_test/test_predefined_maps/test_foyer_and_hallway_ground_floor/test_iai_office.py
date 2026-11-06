from types import SimpleNamespace

import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    StoreyHighRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    OfficeRoom,
    OutlineFrame,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    BathroomCabin,
    Elevator,
    ElevatorLobby,
    Floor,
    Hallway,
)
from krrood_spatial_math.spatial_types import Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

# %% fixtures


@pytest.fixture(scope="module")
def iai_office_world() -> tuple[World, list]:
    world = World.create_with_root_body("map")
    levels = IAIOffice().spawn(world, "office")
    return world, levels


# %% floor plans


def test_office_has_a_floor_plan_for_every_storey():
    floor_plans = IAIOffice().floor_plans
    assert [floor_plan.storey for floor_plan in floor_plans] == list(Storey)


def test_storey_high_room_walls_reach_the_next_storey():
    room = StoreyHighRoom(
        outline=[Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0), Point2(0.0, 1.0)]
    )
    assert room.wall_height == Storey.height


# %% elevator


def test_elevator_travels_no_higher_than_the_top_storey(iai_office_world):
    world, levels = iai_office_world
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    assert elevator.movable_joint.dof.limits.upper.position == pytest.approx(
        elevator.drive_position_for_floor(levels[-1])
    )


def test_the_simulator_reports_the_drive_under_the_name_of_its_connection(
    iai_office_world,
):
    world, _ = iai_office_world
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    assert (
        IAIOffice().simulated_drive_joint_name(elevator)
        == elevator.movable_joint.name.name
    )


# %% kinds of rooms


def test_every_room_takes_the_type_its_floor_plan_shows(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()

    mistyped_rooms = [
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if type(world.get_semantic_annotation_by_name(office_room.number))
        is not office_room.room_type
    ]

    assert mistyped_rooms == []


def test_every_bathroom_lists_the_cabins_its_floor_plan_shows(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()

    for floor_plan in office.floor_plans:
        for office_room in floor_plan.rooms:
            if not office_room.cabin_numbers:
                continue
            bathroom = world.get_semantic_annotation_by_name(office_room.number)
            assert [
                cabin.name.name for cabin in bathroom.cabins
            ] == office_room.cabin_numbers


# %% room names


def test_every_room_is_named_by_its_number_on_its_storey(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()

    misnamed_rooms = [
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if world.get_semantic_annotation_by_name(office_room.number).name
        != PrefixedName(office_room.number, f"office_{floor_plan.storey.name.lower()}")
    ]

    assert misnamed_rooms == []


def test_every_toilet_cabin_belongs_to_one_bathroom(iai_office_world):
    world, _ = iai_office_world
    cabins_of_bathrooms = [
        cabin
        for bathroom in world.get_semantic_annotations_by_type(Bathroom)
        for cabin in bathroom.cabins
    ]

    assert sorted(cabin.name.name for cabin in cabins_of_bathrooms) == sorted(
        cabin.name.name
        for cabin in world.get_semantic_annotations_by_type(BathroomCabin)
    )


def test_only_bathrooms_list_cabins():
    office = IAIOffice()

    rooms_listing_cabins = [
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if office_room.cabin_numbers and office_room.room_type is not Bathroom
    ]

    assert rooms_listing_cabins == []


def test_every_storey_has_one_hallway(iai_office_world):
    world, levels = iai_office_world

    assert len(world.get_semantic_annotations_by_type(Hallway)) == len(levels)


def test_the_elevator_room_is_the_only_elevator_lobby(iai_office_world):
    world, _ = iai_office_world

    assert [
        lobby.name for lobby in world.get_semantic_annotations_by_type(ElevatorLobby)
    ] == [PrefixedName("elevator_room", "office_ground_floor")]


# %% floors


def test_a_rooms_floor_ends_at_its_surface_not_at_its_walls(iai_office_world):
    world, _ = iai_office_world

    extent = world.get_semantic_annotation_by_name("1.85").floor.extent

    assert extent.max_z == pytest.approx(Storey.FIRST_FLOOR.elevation)


def test_the_middle_of_a_room_lies_on_its_floor(iai_office_world):
    world, _ = iai_office_world

    middle = world.get_semantic_annotation_by_name("1.85").floor_centre

    assert float(middle.z) == pytest.approx(Storey.FIRST_FLOOR.elevation)


@pytest.mark.parametrize("room_number", ["0.30", "1.85", "2.50"])
def test_a_body_in_a_room_stands_on_that_rooms_floor_not_on_the_hallways(
    iai_office_world, room_number
):
    world, _ = iai_office_world
    room = world.get_semantic_annotation_by_name(room_number)

    standing_in_the_room = SimpleNamespace(
        global_transform=room.floor_centre, _world=world
    )

    assert Floor.under(standing_in_the_room) is room.floor


# %% rooms spoken of


@pytest.mark.parametrize(
    "number, spoken_name",
    [
        ("1.53", "room one point five three"),
        ("T.0.4.1", "room t point zero point four point one"),
        ("2.16f.1", "room two point one six f point one"),
    ],
)
def test_a_rooms_number_is_said_digit_by_digit(number, spoken_name):
    room = OfficeRoom(number, OutlineFrame.ELEVATOR_ROOM, layout=None)

    assert room.spoken_name == spoken_name


def test_every_room_a_robot_may_enter_can_be_named(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()
    rooms = office.rooms_by_spoken_name(world)

    named = {room.name.name for room in rooms.values()}
    assert named == {"elevator_room"} | {
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if office_room.navigation_target and office_room.room_type is not BathroomCabin
    }


def test_a_room_a_robot_does_not_fit_into_cannot_be_named(iai_office_world):
    world, _ = iai_office_world

    assert (
        IAIOffice().find_room_by_spoken_name(world, "room zero point three nine")
        is None
    )


@pytest.mark.parametrize(
    "sentence, expected_number",
    [
        ("go to room zero point three four please", "0.34"),
        ("go to room zero point three four point one please", "0.34.1"),
        ("take me to the elevator room", "elevator_room"),
    ],
)
def test_a_sentence_names_the_most_specific_room_it_holds(
    iai_office_world, sentence, expected_number
):
    world, _ = iai_office_world

    assert [
        room.name.name for room in IAIOffice().rooms_named_in(world, sentence).values()
    ] == [expected_number]
