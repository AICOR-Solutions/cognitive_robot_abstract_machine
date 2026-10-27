import pytest

from geometry_msgs.msg import TransformStamped
from hri_msgs.msg import IdsList

from experiments.ra_fetch_elevator_people.elevator_people import (
    ElevatorPeople,
    TrackedPersons,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Human,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture
def elevator(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(Elevator)[0]


@pytest.fixture
def tracked_persons(rclpy_node, arise_office_world):
    tracked_persons = TrackedPersons(
        world=arise_office_world,
        node=rclpy_node,
        topic_name="/test/humans/persons/tracked",
    )
    yield tracked_persons
    tracked_persons.close()


def place_person(
    tracked_persons: TrackedPersons,
    person_id: str,
    x: float,
    y: float,
    floor_height: float,
) -> HomogeneousTransformationMatrix:
    """
    Let TF know the person's frame at head height above ``x``, ``y`` on
    ``floor_height``.

    :return: The person's frame in the world root frame.
    """
    world = tracked_persons.world
    head_height = floor_height + 1.65
    transform = TransformStamped()
    transform.header.frame_id = world.root.name.name
    transform.child_frame_id = TrackedPersons.frame_of(person_id)
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = head_height
    transform.transform.rotation.w = 1.0
    tracked_persons.person_frames.tf_wrapper.tf_buffer.set_transform_static(
        transform, "test"
    )
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, head_height, reference_frame=world.root
    )


def track(tracked_persons: TrackedPersons, *person_ids: str) -> None:
    """
    Receive a message naming ``person_ids`` as tracked and apply it.
    """
    tracked_persons.buffer_message(IdsList(ids=list(person_ids)))
    tracked_persons.apply()


# %% tracking persons


def test_tracked_person_becomes_a_human_at_its_frame(
    tracked_persons, arise_office_world, arise_office
):
    world_T_person = place_person(
        tracked_persons, "anna", 1.0, -2.0, arise_office.ground_floor_height
    )
    track(tracked_persons, "anna")
    human = tracked_persons.humans["anna"]
    assert human in arise_office_world.get_semantic_annotations_by_type(Human)
    assert human.root.global_transform.to_np() == pytest.approx(world_T_person.to_np())


def test_tracked_person_follows_its_frame(tracked_persons, arise_office):
    place_person(tracked_persons, "anna", 1.0, -2.0, arise_office.ground_floor_height)
    track(tracked_persons, "anna")
    first = tracked_persons.humans["anna"]
    world_T_person = place_person(
        tracked_persons, "anna", 2.0, -1.0, arise_office.ground_floor_height
    )
    track(tracked_persons, "anna")
    assert tracked_persons.humans["anna"] is first
    assert first.root.global_transform.to_np() == pytest.approx(world_T_person.to_np())


def test_tracked_person_without_a_frame_gets_no_human(tracked_persons):
    track(tracked_persons, "ghost")
    assert tracked_persons.humans == {}


def test_persons_no_longer_tracked_leave_the_world(
    tracked_persons, arise_office_world, arise_office
):
    place_person(tracked_persons, "anna", 1.0, -2.0, arise_office.ground_floor_height)
    place_person(tracked_persons, "ben", 2.0, -2.0, arise_office.ground_floor_height)
    track(tracked_persons, "anna", "ben")
    gone = tracked_persons.humans["anna"]
    track(tracked_persons, "ben")
    assert list(tracked_persons.humans) == ["ben"]
    assert gone not in arise_office_world.get_semantic_annotations_by_type(Human)
    assert not arise_office_world.is_kinematic_structure_entity_in_world_by_name(
        gone.root.name.name
    )


def test_closing_removes_every_human(tracked_persons, arise_office_world, arise_office):
    place_person(tracked_persons, "anna", 1.0, -2.0, arise_office.ground_floor_height)
    track(tracked_persons, "anna")
    tracked_persons.close()
    assert tracked_persons.humans == {}
    assert arise_office_world.get_semantic_annotations_by_type(Human) == []


def test_persons_are_sorted_into_the_elevator_areas(
    tracked_persons, arise_office_world, elevator, arise_office
):
    place_person(
        tracked_persons,
        "rider",
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    place_person(
        tracked_persons, "waiting", 0.45, 1.2, arise_office.ground_floor_height
    )
    place_person(
        tracked_persons, "walker", -5.0, -2.5, arise_office.ground_floor_height
    )
    track(tracked_persons, "rider", "waiting", "walker")
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert tracked_persons.persons_in(elevator.cabin_area) == ["rider"]
    assert tracked_persons.persons_in(elevator.waiting_area_on(ground_floor)) == [
        "waiting"
    ]
    assert tracked_persons.persons_in(elevator.waiting_area_on(first_floor)) == []


# %% the robot's level


@pytest.fixture
def office_with_tiago(arise_office) -> World:
    """
    The ARISE office with a Tiago standing on the ground floor.
    """
    world = WorldSpecification(
        world_parser=None,
        robots=[RobotSpecification(semantic_annotation_type=Tiago)],
    ).to_domain_object()
    arise_office.spawn(world)
    return world


def test_robot_stands_on_the_ground_floor_at_the_start(rclpy_node, office_with_tiago):
    robot = office_with_tiago.get_semantic_annotations_by_type(Tiago)[0]
    elevator_people = ElevatorPeople.for_world(rclpy_node, office_with_tiago, robot)
    ground_floor = office_with_tiago.get_semantic_annotations_by_type(GroundFloor)[0]
    assert elevator_people.robot_level() is ground_floor


def test_robot_riding_the_elevator_stands_on_the_level_it_arrived_at(
    rclpy_node, office_with_tiago
):
    robot = office_with_tiago.get_semantic_annotations_by_type(Tiago)[0]
    elevator = office_with_tiago.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = office_with_tiago.get_semantic_annotations_by_type(FirstFloor)[0]
    elevator_people = ElevatorPeople.for_world(rclpy_node, office_with_tiago, robot)
    office_with_tiago.move_branch(robot.root, elevator.root)
    elevator.drive_to_floor(first_floor)
    assert elevator_people.robot_level() is first_floor
