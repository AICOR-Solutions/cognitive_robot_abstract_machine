from dataclasses import dataclass

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.exceptions import (
    BoardingFailedError,
    BoardingTimedOutError,
)
from experiments.tiago_elevator_waiting_demo.social_navigation import (
    BoardElevatorAction,
    BoardElevatorActionServerTask,
    SocialNavigationAction,
)
from ra_fetch_interfaces.action import BoardElevator

from .scripted_social_navigation import BoardingAnswer
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Human,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% boarding from where the robot starts


@dataclass
class BoardingWithoutDriving(TiagoElevatorRideDemonstration):
    """
    Boards the elevator through the social navigation module from wherever the robot
    starts.
    """

    boarding_timeout: float = 120.0
    """
    Seconds the social navigation module may keep the base.
    """

    def build_plan(self, context: Context) -> PlanNode:
        elevator = context.world.get_semantic_annotations_by_type(Elevator)[0]
        return sequential(
            [BoardElevatorAction(elevator, self.boarding_timeout)], context=context
        )


def starting_in_the_cabin(arise_office) -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        arise_office.cabin_centre.x, arise_office.cabin_centre.y, yaw=np.pi / 2
    )


def starting_in_front_of_the_doors(arise_office) -> HomogeneousTransformationMatrix:
    doors_y = float(arise_office.cabin_centre.y) - arise_office.cabin_scale.x / 2
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        arise_office.cabin_centre.x, doors_y - 1.5, yaw=np.pi / 2
    )


def boarding_from(
    odom_T_robot_start: HomogeneousTransformationMatrix, boarding_timeout: float = 120.0
) -> BoardingWithoutDriving:
    return BoardingWithoutDriving(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        odom_T_robot_start=odom_T_robot_start,
        boarding_timeout=boarding_timeout,
    )


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.SUCCEED], indirect=True)
def test_boarding_ends_once_the_module_succeeded_with_the_robot_in_the_cabin(
    social_navigation, arise_office
):
    boarding_from(starting_in_the_cabin(arise_office)).run()
    assert len(social_navigation.goals) == 1


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.SUCCEED], indirect=True)
def test_boarding_fails_when_the_module_succeeded_without_the_robot_in_the_cabin(
    social_navigation, arise_office
):
    with pytest.raises(BoardingFailedError):
        boarding_from(starting_in_front_of_the_doors(arise_office)).run()


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.ABORT], indirect=True)
def test_boarding_fails_when_the_module_aborts(social_navigation, arise_office):
    with pytest.raises(BoardingFailedError):
        boarding_from(starting_in_the_cabin(arise_office)).run()


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.REJECT], indirect=True)
def test_boarding_fails_when_the_module_rejects_the_goal(
    social_navigation, arise_office
):
    with pytest.raises(BoardingFailedError):
        boarding_from(starting_in_the_cabin(arise_office)).run()


@pytest.mark.parametrize(
    "social_navigation", [BoardingAnswer.KEEP_THE_BASE], indirect=True
)
def test_boarding_times_out_when_the_module_keeps_the_base(
    social_navigation, arise_office
):
    with pytest.raises(BoardingTimedOutError):
        boarding_from(starting_in_the_cabin(arise_office), boarding_timeout=0.5).run()


# %% the boarding goal


@pytest.fixture
def office_with_tiago(arise_office) -> World:
    demonstration = boarding_from(starting_in_front_of_the_doors(arise_office))
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return world


def person_standing_at(
    world: World, name: str, x: float, y: float, floor_height: float
):
    scale = Scale(0.4, 0.4, 1.7)
    return Human.create_with_new_body_in_world(
        name=name,
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, floor_height + scale.z / 2
        ),
        scale=scale,
    )


def boarding_task(world: World) -> BoardElevatorActionServerTask:
    return BoardElevatorActionServerTask(
        action_topic=SocialNavigationAction.BOARD_ELEVATOR,
        message_type=BoardElevator,
        elevator=world.get_semantic_annotations_by_type(Elevator)[0],
        robot=world.get_semantic_annotations_by_type(Tiago)[0],
    )


def test_boarding_goal_names_the_persons_in_the_cabin_and_waiting_on_the_robots_level(
    office_with_tiago, arise_office
):
    floor = arise_office.ground_floor_height
    person_standing_at(
        office_with_tiago,
        "rider",
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        floor,
    )
    person_standing_at(office_with_tiago, "waiting", 0.45, 1.2, floor)
    person_standing_at(office_with_tiago, "walker", -5.0, -2.5, floor)
    person_standing_at(
        office_with_tiago, "upstairs", 0.45, 1.2, arise_office.first_floor_height
    )

    goal = boarding_task(office_with_tiago).boarding_goal()

    assert [person.person_id for person in goal.persons_in_cabin] == ["rider"]
    assert [person.person_id for person in goal.persons_waiting] == ["waiting"]


def test_boarding_goal_describes_the_doorway_and_the_cabin(
    office_with_tiago, arise_office
):
    elevator = office_with_tiago.get_semantic_annotations_by_type(Elevator)[0]
    world_T_doorway = office_with_tiago.transform(
        elevator.doorway_pose, office_with_tiago.root
    ).to_np()

    goal = boarding_task(office_with_tiago).boarding_goal()

    position = goal.doorway_pose.pose.position
    assert [position.x, position.y, position.z] == pytest.approx(
        world_T_doorway[:3, 3].tolist()
    )
    assert goal.doorway_pose.header.frame_id == str(office_with_tiago.root.name)
    assert [goal.cabin_size.x, goal.cabin_size.y, goal.cabin_size.z] == pytest.approx(
        [elevator.scale.x, elevator.scale.y, elevator.scale.z]
    )


def test_boarding_goal_gives_the_robots_pose(office_with_tiago):
    robot = office_with_tiago.get_semantic_annotations_by_type(Tiago)[0]

    goal = boarding_task(office_with_tiago).boarding_goal()

    position = goal.robot_pose.pose.position
    assert [position.x, position.y, position.z] == pytest.approx(
        robot.root.global_transform.to_np()[:3, 3].tolist()
    )
