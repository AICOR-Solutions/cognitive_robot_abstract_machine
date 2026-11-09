import pytest
from typing_extensions import Iterator

from coraplex.robot_plans.actions.core.navigation import ElevatorRideAnnouncement
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Region

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


# %% calling the cabin


@pytest.fixture
def ride_demonstration() -> Iterator[TiagoElevatorRideDemonstration]:
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)
    yield demonstration
    demonstration.tear_down()


@pytest.fixture
def operated_world(ride_demonstration: TiagoElevatorRideDemonstration) -> World:
    """
    The office with Tiago in the lobby and the cabin sent up to the first floor, run by
    the demonstration's elevator operator.
    """
    world = ride_demonstration.build_simulated_world()
    ride_demonstration.build_context_extensions(world)
    world.get_semantic_annotations_by_type(Elevator)[0].serve_floor(
        world.get_semantic_annotations_by_type(FirstFloor)[0]
    )
    return world


def move_robot_into(robot: AbstractRobot, region: Region) -> None:
    """
    Put ``robot``'s base on the middle of ``region``, on the level the robot stands on.
    """
    world = robot._world
    odom = robot.root.parent_kinematic_structure_entity
    odom_T_region = world.transform(
        HomogeneousTransformationMatrix(reference_frame=region), odom
    )
    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        float(odom_T_region.x), float(odom_T_region.y), 0, reference_frame=odom
    )
    world.notify_state_change()


def test_the_cabin_comes_to_a_robot_waiting_on_another_level(operated_world):
    elevator = operated_world.get_semantic_annotations_by_type(Elevator)[0]
    ground_floor = operated_world.get_semantic_annotations_by_type(GroundFloor)[0]
    robot = operated_world.get_semantic_annotations_by_type(Tiago)[0]

    move_robot_into(robot, elevator.waiting_areas[ground_floor])

    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(ground_floor)
    )
    assert [door.movable_joint.position for door in elevator.doors] == [
        door.movable_joint.dof.limits.upper.position for door in elevator.doors
    ]


def test_the_cabin_stays_while_no_robot_waits_for_it(operated_world):
    elevator = operated_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = operated_world.get_semantic_annotations_by_type(FirstFloor)[0]

    operated_world.notify_state_change()

    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )
