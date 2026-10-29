from collections import Counter

import pytest
from typing_extensions import Iterator

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    PathPlanningNavigateAction,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.demo import (
    TiagoRoomNavigationDemonstration,
)
from experiments.tiago_room_navigation_demo.exceptions import (
    NoElevatorError,
    RoomOnNoLevelError,
    TargetRoomNotReachedError,
)
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from experiments.tiago_room_navigation_demo.trials import (
    RoomNavigationTrialResult,
    RoomNavigationTrials,
    RoomNavigationTrialSummary,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Room,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale

# %% Tiago standing in the lobby


@pytest.fixture
def lobby_demonstration() -> Iterator[TiagoElevatorRideDemonstration]:
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )
    yield demonstration
    demonstration.tear_down()


@pytest.fixture
def lobby_context(lobby_demonstration) -> Context:
    return lobby_demonstration.build_context(
        lobby_demonstration.build_simulated_world()
    )


def planned_steps(action: NavigateToRoomAction, context: Context) -> list:
    """
    :return: The actions ``action`` expands into for the robot of ``context``.
    """
    execute_single(action, context=context)
    return [child.designator for child in action._action_plan.children]


# %% planning the way to a room


def test_a_room_on_the_robots_level_is_driven_to_directly(
    lobby_demonstration, lobby_context
):
    east_office = lobby_demonstration.office.get_room(
        lobby_context.world, GroundFloor, AriseOfficeRoom.EAST_OFFICE
    )

    [drive] = planned_steps(NavigateToRoomAction(east_office), lobby_context)

    assert isinstance(drive, PathPlanningNavigateAction)
    assert drive.target.to_np() == pytest.approx(east_office.floor_centre.to_np())


def test_a_room_on_another_level_is_reached_by_riding_the_elevator(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    west_office = lobby_demonstration.office.get_room(
        world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
    )
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    action = NavigateToRoomAction(west_office)

    steps = planned_steps(action, lobby_context)

    assert [type(step) for step in steps] == [
        PathPlanningNavigateAction,
        ElevatorNavigation,
        PathPlanningNavigateAction,
    ]
    drive_to_elevator, ride, drive_to_room = steps
    assert drive_to_elevator.target.to_np() == pytest.approx(
        elevator.pose_in_front_of_doors(action.elevator_waiting_distance).to_np()
    )
    assert ride.target_floor is world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert drive_to_room.target.to_np() == pytest.approx(
        west_office.floor_centre.to_np()
    )


def test_a_room_on_no_level_cannot_be_planned_for(lobby_context):
    world = lobby_context.world
    basement_floor = Floor.get_annotation_specification(
        "basement_floor",
        Floor.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(2, 2, 0)
        ),
    ).spawn(world, parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-3))
    basement = Room(name=PrefixedName("basement"), floor=basement_floor)
    with world.modify_world():
        world.add_semantic_annotation(basement)

    with pytest.raises(RoomOnNoLevelError):
        planned_steps(NavigateToRoomAction(basement), lobby_context)


def test_a_room_on_another_level_cannot_be_planned_for_without_an_elevator(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    with world.modify_world():
        world.remove_semantic_annotation(
            world.get_semantic_annotations_by_type(Elevator)[0]
        )
    landing = lobby_demonstration.office.get_room(
        world, FirstFloor, AriseOfficeRoom.LANDING
    )

    with pytest.raises(NoElevatorError):
        planned_steps(NavigateToRoomAction(landing), lobby_context)


def test_path_planning_lays_its_path_out_over_the_level_the_robot_is_on(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    east_office = lobby_demonstration.office.get_room(
        world, GroundFloor, AriseOfficeRoom.EAST_OFFICE
    )
    action = PathPlanningNavigateAction(east_office.floor_centre)

    execute_single(action, context=lobby_context)

    assert action._ground is world.get_semantic_annotations_by_type(GroundFloor)[0]


# %% driving to a room


@pytest.mark.parametrize(
    "level_type, room",
    [
        (GroundFloor, AriseOfficeRoom.WEST_OFFICE),
        (FirstFloor, AriseOfficeRoom.EAST_OFFICE),
    ],
)
def test_tiago_reaches_the_target_room(
    rclpy_node, arise_office, arise_office_world, level_type, room
):
    target_room_name = arise_office.get_room(
        arise_office_world, level_type, room
    ).name.name

    result = RoomNavigationTrials().perform_trial(target_room_name)

    assert result.failure is None


def test_tiago_outside_the_target_room_fails_the_demonstration(
    arise_office, arise_office_world
):
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        target_room_name=arise_office.get_room(
            arise_office_world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
        ).name.name,
    )

    world = demonstration.build_simulated_world()

    with pytest.raises(TargetRoomNotReachedError):
        demonstration.conclude(world)
    demonstration.tear_down()


# %% trials


def test_targets_are_sampled_from_every_room_but_the_one_tiago_starts_in(
    arise_office, arise_office_world
):
    lobby = arise_office.get_room(
        arise_office_world, GroundFloor, AriseOfficeRoom.LOBBY
    )
    other_rooms = {
        room.name.name
        for room in arise_office_world.get_semantic_annotations_by_type(Room)
        if room is not lobby
    }
    trials = RoomNavigationTrials(number_of_trials=30, seed=0)

    target_room_names = trials.target_room_names()

    assert len(target_room_names) == trials.number_of_trials
    assert set(target_room_names) == other_rooms


def test_the_same_seed_samples_the_same_targets():
    assert (
        RoomNavigationTrials(number_of_trials=10, seed=3).target_room_names()
        == RoomNavigationTrials(number_of_trials=10, seed=3).target_room_names()
    )


def test_a_failing_trial_is_recorded_instead_of_raised(rclpy_node):
    result = RoomNavigationTrials().perform_trial("no_such_room")

    assert result.target_room_name == "no_such_room"
    assert isinstance(result.failure, WorldEntityNotFoundError)


# %% overview of the trials


@pytest.fixture
def summary_with_one_failure() -> RoomNavigationTrialSummary:
    return RoomNavigationTrialSummary(
        [
            RoomNavigationTrialResult("office"),
            RoomNavigationTrialResult("landing", TargetRoomNotReachedError("landing")),
            RoomNavigationTrialResult("landing"),
        ]
    )


def test_summary_counts_successes_and_failures(summary_with_one_failure):
    assert summary_with_one_failure.number_of_successes == 2
    assert summary_with_one_failure.number_of_failures == 1


def test_summary_groups_failures_by_type(summary_with_one_failure):
    assert summary_with_one_failure.failures_by_type == Counter(
        {TargetRoomNotReachedError.__name__: 1}
    )


def test_summary_groups_trials_by_target_room(summary_with_one_failure):
    office, failed_landing, landing = summary_with_one_failure.results
    assert summary_with_one_failure.trials_by_target_room == {
        "office": [office],
        "landing": [failed_landing, landing],
    }


def test_report_names_each_failed_trial_with_why_it_failed(summary_with_one_failure):
    assert (
        f"#2 to landing: {TargetRoomNotReachedError.__name__}"
        in summary_with_one_failure.report()
    )
