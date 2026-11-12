from dataclasses import dataclass, field

import pytest

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    ElevatorRideAnnouncement,
    NavigateAction,
)
from experiments.tiago_elevator_demo.demo import (
    RelocalizeFacingTheElevatorDoors,
    TiagoElevatorRideDemonstration,
)
from giskardpy.motion_statechart.ros2_nodes.localization import (
    LoadMap,
    PublishInitialPose,
)
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

from ...plan_running import simulated_executor, statechart_of

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


@dataclass
class RideRememberingWhereTheRobotHung(TiagoElevatorRideDemonstration):
    """
    Rides the elevator and remembers the entity the robot hung from before the ride.
    """

    parent_before_the_ride: KinematicStructureEntity | None = field(
        default=None, init=False
    )
    """
    The entity the robot hung from once the world was acquired.
    """

    def acquire_world(self) -> World:
        world = super().acquire_world()
        self.parent_before_the_ride = world.get_semantic_annotations_by_type(
            self.used_robot
        )[0].root.parent_kinematic_structure_entity
        return world


def test_tiago_hangs_from_where_it_hung_before_the_ride_once_out_of_the_cabin(
    rclpy_node,
):
    demonstration = RideRememberingWhereTheRobotHung(used_robot=Tiago)

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    assert (
        robot.root.parent_kinematic_structure_entity
        is demonstration.parent_before_the_ride
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


# %% relocalizing on a real run


def test_the_robot_is_relocalized_right_after_it_turned_to_the_doors_in_the_cabin():
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    ride = ElevatorNavigation(
        elevator=world.get_semantic_annotations_by_type(Elevator)[0],
        target_floor=world.get_semantic_annotations_by_type(FirstFloor)[0],
    )
    transformation = RelocalizeFacingTheElevatorDoors()

    executor = simulated_executor(
        [RobotAccess(robot), PlanRewriting(transformations=[transformation])]
    )

    executor.prepare(statechart_of(executor, ride))

    [turn_to_the_doors] = [
        step for step in ride.nodes if isinstance(step, NavigateAction)
    ]
    relocalization = ride.nodes[ride.nodes.index(turn_to_the_doors) + 1]
    assert isinstance(relocalization, PublishInitialPose)
    assert relocalization.base is robot.root
    assert relocalization.topic_name == transformation.topic_name
    demonstration.tear_down()


def test_the_target_floors_map_is_loaded_before_the_robot_is_relocalized():
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    ride = ElevatorNavigation(
        elevator=world.get_semantic_annotations_by_type(Elevator)[0],
        target_floor=world.get_semantic_annotations_by_type(FirstFloor)[0],
    )
    transformation = RelocalizeFacingTheElevatorDoors(
        floor_map_urls={FirstFloor: "/maps/first_floor.yaml"}
    )
    executor = simulated_executor(
        [RobotAccess(robot), PlanRewriting(transformations=[transformation])]
    )

    executor.prepare(statechart_of(executor, ride))

    [turn_to_the_doors] = [
        step for step in ride.nodes if isinstance(step, NavigateAction)
    ]
    turn_index = ride.nodes.index(turn_to_the_doors)
    map_loading, relocalization = ride.nodes[turn_index + 1 : turn_index + 3]
    assert isinstance(map_loading, LoadMap)
    assert map_loading.map_url == transformation.floor_map_urls[FirstFloor]
    assert isinstance(relocalization, PublishInitialPose)
    demonstration.tear_down()


def test_simulated_tiago_stands_on_the_ground_floor():
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)

    world = demonstration.build_simulated_world()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    lobby = demonstration.office.get_room(world, GroundFloor, AriseOfficeRoom.LOBBY)
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(lobby.floor.root.global_transform.z)
    )
