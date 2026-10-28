from dataclasses import dataclass, field
from time import sleep
from typing import Any

import pytest

from geometry_msgs.msg import PolygonStamped, TransformStamped
from hri_msgs.msg import IdsList
from rclpy.node import Node

from coraplex.datastructures.enums import ExecutionType
from experiments.ra_fetch_elevator_people.elevator_people import (
    ElevatorPeoplePublisher,
    ElevatorPeopleTopic,
    TrackedPersons,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Human,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% fixtures


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
    assert human.name.name == "anna"
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


# %% publishing who is where


@dataclass
class TopicRecorder:
    """
    Keeps the last message received on each topic it subscribes to.
    """

    last_messages: dict[str, Any] = field(default_factory=dict)
    """
    The last message received on each topic.
    """

    def subscribe(self, node: Node, message_type: type, topic: str) -> None:
        """
        Record the messages ``node`` receives on ``topic``.
        """
        node.create_subscription(
            message_type,
            topic,
            lambda message: self.last_messages.__setitem__(topic, message),
            1,
        )

    def wait_for(self, *topics: str) -> None:
        """
        Wait up to three seconds until a message arrived on each of ``topics``.
        """
        for _ in range(30):
            if all(topic in self.last_messages for topic in topics):
                return
            sleep(0.1)


@pytest.fixture
def office_with_tiago() -> World:
    return TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    ).build_simulated_world()


@pytest.fixture
def elevator_people_publisher(rclpy_node, office_with_tiago) -> ElevatorPeoplePublisher:
    return ElevatorPeoplePublisher(
        node=rclpy_node,
        elevator=office_with_tiago.get_semantic_annotations_by_type(Elevator)[0],
        robot=office_with_tiago.get_semantic_annotations_by_type(Tiago)[0],
    )


@pytest.fixture
def recorder(rclpy_node) -> TopicRecorder:
    recorder = TopicRecorder()
    recorder.subscribe(rclpy_node, IdsList, ElevatorPeopleTopic.CABIN_PERSONS)
    recorder.subscribe(rclpy_node, PolygonStamped, ElevatorPeopleTopic.CABIN_FOOTPRINT)
    recorder.subscribe(rclpy_node, IdsList, ElevatorPeopleTopic.WAITING_AREA_PERSONS)
    recorder.subscribe(
        rclpy_node, PolygonStamped, ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT
    )
    return recorder


def person_standing_at(
    world: World, name: str, x: float, y: float, floor_height: float
) -> Human:
    scale = Scale(0.4, 0.4, 1.7)
    return Human.create_with_new_body_in_world(
        name=name,
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, floor_height + scale.z / 2
        ),
        scale=scale,
    )


def corners(footprint: PolygonStamped) -> set[tuple[float, float, float]]:
    return {
        (round(point.x, 5), round(point.y, 5), round(point.z, 5))
        for point in footprint.polygon.points
    }


def test_publishes_the_persons_inside_the_cabin_and_its_outline(
    elevator_people_publisher, recorder, office_with_tiago, arise_office
):
    person_standing_at(
        office_with_tiago,
        "rider",
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    person_standing_at(
        office_with_tiago, "walker", -5.0, -2.5, arise_office.ground_floor_height
    )

    elevator_people_publisher.publish()
    recorder.wait_for(
        ElevatorPeopleTopic.CABIN_PERSONS, ElevatorPeopleTopic.CABIN_FOOTPRINT
    )

    elevator = elevator_people_publisher.elevator
    assert recorder.last_messages[ElevatorPeopleTopic.CABIN_PERSONS].ids == ["rider"]
    assert corners(recorder.last_messages[ElevatorPeopleTopic.CABIN_FOOTPRINT]) == {
        (round(x, 5), round(y, 5), round(arise_office.ground_floor_height, 5))
        for x in (
            float(arise_office.cabin_centre.x) - elevator.scale.y / 2,
            float(arise_office.cabin_centre.x) + elevator.scale.y / 2,
        )
        for y in (
            float(arise_office.cabin_centre.y) - elevator.scale.x / 2,
            float(arise_office.cabin_centre.y) + elevator.scale.x / 2,
        )
    }


def test_publishes_the_persons_waiting_on_the_robots_level_and_their_area(
    elevator_people_publisher, recorder, office_with_tiago, arise_office
):
    person_standing_at(
        office_with_tiago, "waiting", 0.45, 1.2, arise_office.ground_floor_height
    )
    person_standing_at(
        office_with_tiago, "upstairs", 0.45, 1.2, arise_office.first_floor_height
    )

    elevator_people_publisher.publish()
    recorder.wait_for(
        ElevatorPeopleTopic.WAITING_AREA_PERSONS,
        ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT,
    )

    elevator = elevator_people_publisher.elevator
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    half_width = elevator.scale.y / 2 + elevator.waiting_area_side_margin
    assert recorder.last_messages[ElevatorPeopleTopic.WAITING_AREA_PERSONS].ids == [
        "waiting"
    ]
    assert corners(
        recorder.last_messages[ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT]
    ) == {
        (round(x, 5), round(y, 5), round(arise_office.ground_floor_height, 5))
        for x in (-half_width, half_width)
        for y in (door_y - elevator.waiting_area_depth, door_y)
    }
