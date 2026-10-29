import pytest

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.actions.core.navigation import ElevatorRideAnnouncement
from coraplex.robot_plans.actions.core.speech import SayAction
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.demo import (
    ElevatorBoardingAnnouncement,
    TiagoElevatorWaitingDemonstration,
)
from experiments.tiago_elevator_waiting_demo.exceptions import BoardingFailedError
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.world import World

from .scripted_social_navigation import BoardingAnswer

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
        float(first_floor.floor_plane[0].z)
    )
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )


def test_tiago_says_where_it_is_and_where_it_is_headed_while_riding(tiago_speech):
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

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
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.build_simulated_world()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    ground_floor_slab = demonstration.office.ground_floor_slab(world)
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(ground_floor_slab.root.global_transform.z)
    )


# %% handing the base over to the social navigation module


@pytest.fixture
def waiting_demonstration() -> TiagoElevatorWaitingDemonstration:
    return TiagoElevatorWaitingDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )


@pytest.fixture
def office_world(waiting_demonstration) -> World:
    return waiting_demonstration.build_simulated_world()


def test_tiago_waits_in_front_of_the_doors_facing_the_cabin(
    waiting_demonstration, office_world
):
    elevator = office_world.get_semantic_annotations_by_type(Elevator)[0]

    world_T_waiting = office_world.transform(
        waiting_demonstration.waiting_pose(elevator), office_world.root
    ).to_np()

    world_T_doorway = office_world.transform(
        elevator.doorway_pose, office_world.root
    ).to_np()
    inward = world_T_doorway[:3, 0]
    assert world_T_waiting[:3, 3] == pytest.approx(
        world_T_doorway[:3, 3] - waiting_demonstration.distance_from_doors * inward
    )
    assert world_T_waiting[:3, 0] == pytest.approx(inward)


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.ABORT], indirect=True)
def test_tiago_hands_its_base_over_from_where_it_waits(
    social_navigation, waiting_demonstration
):
    office = waiting_demonstration.office
    with pytest.raises(BoardingFailedError):
        waiting_demonstration.run()

    [goal] = social_navigation.goals
    doors_y = float(office.cabin_centre.y) - office.cabin_scale.x / 2
    position = goal.robot_pose.pose.position
    assert [position.x, position.y] == pytest.approx(
        [
            float(office.cabin_centre.x),
            doors_y - waiting_demonstration.distance_from_doors,
        ],
        abs=0.1,
    )


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.ABORT], indirect=True)
def test_tiago_says_it_waits_to_board_in_front_of_the_elevator(
    social_navigation, tiago_speech, waiting_demonstration, office_world
):
    with pytest.raises(BoardingFailedError):
        waiting_demonstration.run()

    ground_floor = office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    expected_sentence = ElevatorBoardingAnnouncement.WAITING_TO_BOARD.verbalize(
        current_floor=ground_floor
    )
    assert tiago_speech.wait_for_sentences(1) == [expected_sentence]


def test_tiago_says_it_is_inside_the_elevator_once_it_boarded(
    waiting_demonstration, office_world
):
    context = waiting_demonstration.build_context(office_world)

    plan = waiting_demonstration.build_plan(context)

    last_step = plan.children[-1].designator
    assert isinstance(last_step, SayAction)
    assert last_step.text == ElevatorBoardingAnnouncement.BOARDED
