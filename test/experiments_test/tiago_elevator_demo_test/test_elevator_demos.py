import json

import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType
from experiments.ra_fetch_elevator_people.elevator_people import ElevatorPeople
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.demo import (
    BoardingRequest,
    BoardingResult,
    BoardingStatus,
    ElevatorAreaName,
    ElevatorDescription,
    PersonNearElevator,
    TiagoElevatorWaitingDemonstration,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
)
from semantic_digital_twin.world import World

from ..ra_fetch_elevator_people_test.test_elevator_people import (
    place_person,
    track,
)

# %% riding the elevator


def test_tiago_rides_the_elevator_to_the_first_floor(rclpy_node):
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(first_floor.floor_plane[0].z) - demonstration.office.ground_floor_height
    )
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )


# %% handing the base over to the social navigation module


@pytest.fixture
def waiting_demonstration() -> TiagoElevatorWaitingDemonstration:
    return TiagoElevatorWaitingDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )


@pytest.fixture
def office_world(waiting_demonstration) -> World:
    world = waiting_demonstration.build_simulated_world()
    waiting_demonstration.populate_scene(world)
    return world


def test_tiago_waits_in_front_of_the_doors_facing_the_cabin(
    waiting_demonstration, office_world
):
    elevator = office_world.get_semantic_annotations_by_type(Elevator)[0]

    waiting_pose = office_world.transform(
        waiting_demonstration.waiting_pose(elevator), office_world.root
    ).to_np()

    description = ElevatorDescription.of(elevator)
    inward = np.array(
        [np.cos(description.inward_yaw), np.sin(description.inward_yaw), 0.0]
    )
    assert waiting_pose[:3, 3] == pytest.approx(
        np.array(description.door_center)
        - waiting_demonstration.distance_from_doors * inward
    )
    assert np.arctan2(waiting_pose[1, 0], waiting_pose[0, 0]) == pytest.approx(
        description.inward_yaw
    )


def test_elevator_description_gives_doors_and_cabin_on_the_cabin_floor(
    waiting_demonstration, office_world
):
    office = waiting_demonstration.office
    elevator = office_world.get_semantic_annotations_by_type(Elevator)[0]

    description = ElevatorDescription.of(elevator)

    cabin_x, cabin_y = float(office.cabin_centre.x), float(office.cabin_centre.y)
    assert description.cabin_center == pytest.approx(
        [cabin_x, cabin_y, office.ground_floor_height]
    )
    assert description.door_center == pytest.approx(
        [cabin_x, cabin_y - office.cabin_scale.x / 2, office.ground_floor_height]
    )
    # The office's doors face -y
    assert description.inward_yaw == pytest.approx(np.pi / 2)
    assert (description.cabin_width, description.cabin_depth) == pytest.approx(
        (office.cabin_scale.y, office.cabin_scale.x)
    )


def test_boarding_request_names_the_persons_in_the_cabin_and_the_waiting_area(
    rclpy_node, waiting_demonstration, office_world
):
    office = waiting_demonstration.office
    elevator = office_world.get_semantic_annotations_by_type(Elevator)[0]
    robot = office_world.get_semantic_annotations_by_type(Tiago)[0]
    elevator_people = ElevatorPeople.for_world(rclpy_node, office_world, robot)
    tracked_persons = elevator_people.tracked_persons
    world_T_rider = place_person(
        tracked_persons,
        "rider",
        office.cabin_centre.x,
        office.cabin_centre.y,
        office.ground_floor_height,
    )
    place_person(tracked_persons, "waiting", 0.45, 1.2, office.ground_floor_height)
    place_person(tracked_persons, "walker", -5.0, -2.5, office.ground_floor_height)
    track(tracked_persons, "rider", "waiting", "walker")

    request = waiting_demonstration._boarding_request(elevator_people, elevator, robot)

    assert request.request_id == waiting_demonstration.request_id
    assert [(person.person_id, person.area) for person in request.persons] == [
        ("rider", ElevatorAreaName.CABIN),
        ("waiting", ElevatorAreaName.WAITING_AREA),
    ]
    assert request.persons[0].position == pytest.approx(
        world_T_rider.to_np()[:3, 3].tolist()
    )


def test_boarding_request_is_plain_json_for_the_social_navigation_module():
    request = BoardingRequest(
        request_id=3,
        frame_id="map",
        robot_position=[0.0, 0.625, 0.0],
        robot_yaw=1.57,
        elevator=ElevatorDescription(
            door_center=[0.0, 2.125, 0.0],
            cabin_center=[0.0, 2.925, 0.0],
            inward_yaw=1.57,
            cabin_width=1.6,
            cabin_depth=1.6,
        ),
        persons=[
            PersonNearElevator(
                person_id="in_cabin",
                area=ElevatorAreaName.CABIN,
                position=[0.35, 3.2, 1.75],
            )
        ],
    )

    assert json.loads(request.to_message().data) == {
        "request_id": 3,
        "frame_id": "map",
        "robot_position": [0.0, 0.625, 0.0],
        "robot_yaw": 1.57,
        "elevator": {
            "door_center": [0.0, 2.125, 0.0],
            "cabin_center": [0.0, 2.925, 0.0],
            "inward_yaw": 1.57,
            "cabin_width": 1.6,
            "cabin_depth": 1.6,
        },
        "persons": [
            {
                "person_id": "in_cabin",
                "area": ElevatorAreaName.CABIN.value,
                "position": [0.35, 3.2, 1.75],
            }
        ],
    }


def test_boarding_result_is_read_from_the_social_navigation_modules_json():
    result = BoardingResult.from_json(
        json.loads('{"request_id": 3, "status": "SUCCEEDED", "reason": "aboard"}')
    )

    assert result == BoardingResult(
        request_id=3, status=BoardingStatus.SUCCEEDED, reason="aboard"
    )
