import pytest

from coraplex.robot_plans.actions.core.navigation import ElevatorRideAnnouncement
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)

# %% riding the elevator


def test_tiago_rides_the_elevator_to_the_first_floor(rclpy_node):
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(first_floor.floor_plane[0].z)
    )
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )


def test_tiago_says_where_it_is_and_where_it_is_headed_while_riding(tiago_speech):
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)

    world = demonstration.run()

    ground_floor = world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    expected_sentences = [
        announcement.verbalize(current_floor=ground_floor, target_floor=first_floor)
        for announcement in (
            ElevatorRideAnnouncement.DEPARTING,
            ElevatorRideAnnouncement.WAITING_AT_DOORS,
            ElevatorRideAnnouncement.RIDING,
            ElevatorRideAnnouncement.ARRIVED,
        )
    ]
    assert (
        tiago_speech.wait_for_sentences(len(expected_sentences)) == expected_sentences
    )


def test_simulated_tiago_stands_on_the_ground_floor():
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)

    world = demonstration.build_simulated_world()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    lobby = demonstration.office.get_room(world, GroundFloor, AriseOfficeRoom.LOBBY)
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(lobby.floor.root.global_transform.z)
    )
