import json
from dataclasses import dataclass, field

import pytest

pytest.importorskip("communication_skills")

from geometry_msgs.msg import TransformStamped
from hri_msgs.msg import IdsList

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from experiments.ra_fetch_elevator_people.elevator_people import (
    Ros4HriTopic,
    TrackedPersons,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_request_demo.boarding_permission import (
    AskToBoardExecutable,
    BoardingAnswerKey,
    BoardingQuestion,
    BoardingQuestionAsker,
    CommunicationSkill,
    NavigateToRoomAskingToBoardAction,
)
from experiments.tiago_room_request_demo.exceptions import (
    AskSkillNotServedError,
    BoardingPermissionNotGrantedError,
    ElevatorCabinTimedOutError,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
    AriseOfficeRoom,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Level,
)
from semantic_digital_twin.world import World

from .next_elevator import NextElevator
from .scripted_ask import AskAnswer, ScriptedAsk, scripted_ask

# %% people the person manager tracks


@dataclass
class PersonAtTheElevator:
    """
    Where a person stands in a test.
    """

    person_id: str
    """
    The person's ROS4HRI ID.
    """

    x: float
    """
    The person's position along the world's x-axis.
    """

    y: float
    """
    The person's position along the world's y-axis.
    """

    level_type: type[Level] = GroundFloor
    """
    The storey the person stands on.
    """


def person_waiting() -> PersonAtTheElevator:
    """
    :return: A person in the waiting area, beside the robot's way to the doors.
    """
    return PersonAtTheElevator("waiting", 0.85, 1.0)


def person_riding(office: AriseOfficeBuilding) -> PersonAtTheElevator:
    """
    :return: A person in the middle of the cabin.
    """
    return PersonAtTheElevator(
        "rider", float(office.cabin_centre.x), float(office.cabin_centre.y)
    )


def track_people(
    persons_at_elevator: PersonsAtElevator,
    office: AriseOfficeBuilding,
    people: list[PersonAtTheElevator],
    head_height_above_floor: float = 1.65,
) -> None:
    """
    Let TF know where ``people`` are and receive the message naming them as tracked.
    """
    floor_heights = {
        GroundFloor: office.ground_floor_height,
        FirstFloor: office.first_floor_height,
    }
    for person in people:
        transform = TransformStamped()
        transform.header.frame_id = persons_at_elevator.elevator._world.root.name.name
        transform.child_frame_id = TrackedPersons.frame_of(person.person_id)
        transform.transform.translation.x = person.x
        transform.transform.translation.y = person.y
        transform.transform.translation.z = (
            floor_heights[person.level_type] + head_height_above_floor
        )
        transform.transform.rotation.w = 1.0
        persons_at_elevator.tf_wrapper.tf_buffer.set_transform_static(transform, "test")
    persons_at_elevator.buffer_message(
        IdsList(ids=[person.person_id for person in people])
    )


# %% finding the person to ask


@pytest.fixture
def persons_at_elevator(rclpy_node, arise_office_world):
    persons = PersonsAtElevator(
        node=rclpy_node,
        topic_name="/test/humans/persons/tracked",
        elevator=arise_office_world.get_semantic_annotations_by_type(Elevator)[0],
    )
    yield persons
    persons.close()


@pytest.fixture
def ground_floor(arise_office_world) -> GroundFloor:
    return arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]


def test_nobody_tracked_means_nobody_to_ask(
    persons_at_elevator, arise_office_world, ground_floor
):
    assert persons_at_elevator.nearest_to(arise_office_world.root, ground_floor) is None


def test_a_tracked_person_without_a_frame_is_not_asked(
    persons_at_elevator, arise_office_world, ground_floor
):
    persons_at_elevator.buffer_message(IdsList(ids=["ghost"]))

    assert persons_at_elevator.nearest_to(arise_office_world.root, ground_floor) is None


def test_the_person_nearest_to_the_entity_is_chosen(
    persons_at_elevator, arise_office, arise_office_world, ground_floor
):
    nearest = person_waiting()
    track_people(
        persons_at_elevator, arise_office, [person_riding(arise_office), nearest]
    )

    assert (
        persons_at_elevator.nearest_to(arise_office_world.root, ground_floor)
        == nearest.person_id
    )


def test_a_person_in_the_cabin_counts_as_being_at_the_elevator(
    persons_at_elevator, arise_office, arise_office_world, ground_floor
):
    rider = person_riding(arise_office)
    track_people(persons_at_elevator, arise_office, [rider])

    assert (
        persons_at_elevator.nearest_to(arise_office_world.root, ground_floor)
        == rider.person_id
    )


def test_a_person_waiting_on_another_level_is_not_at_the_elevator(
    persons_at_elevator, arise_office, arise_office_world, ground_floor
):
    waiting = person_waiting()
    track_people(
        persons_at_elevator,
        arise_office,
        [PersonAtTheElevator("upstairs", waiting.x, waiting.y, FirstFloor)],
    )

    assert persons_at_elevator.nearest_to(arise_office_world.root, ground_floor) is None


def test_a_person_elsewhere_on_the_level_is_not_at_the_elevator(
    persons_at_elevator, arise_office, arise_office_world, ground_floor
):
    track_people(
        persons_at_elevator,
        arise_office,
        [PersonAtTheElevator("walker", -5.0, -2.5)],
    )

    assert persons_at_elevator.nearest_to(arise_office_world.root, ground_floor) is None


def test_tracking_people_leaves_the_world_model_alone(
    persons_at_elevator, arise_office, arise_office_world, ground_floor
):
    # The areas join the world on first access, which the action does before any motion.
    persons_at_elevator.elevator.cabin_area
    persons_at_elevator.elevator.waiting_areas
    degrees_of_freedom_before = len(arise_office_world.state.keys())
    version_before = arise_office_world.get_world_model_manager().version
    track_people(persons_at_elevator, arise_office, [person_waiting()])

    persons_at_elevator.nearest_to(arise_office_world.root, ground_floor)

    assert len(arise_office_world.state.keys()) == degrees_of_freedom_before
    assert arise_office_world.get_world_model_manager().version == version_before


# %% driving upstairs past people at the elevator


@dataclass
class RidingPastPeopleAtTheElevator(TiagoElevatorRideDemonstration):
    """
    Tiago drives from the lobby to the east office on the first floor, with people
    standing at the elevator and a next cabin coming after each refusal.
    """

    people: list[PersonAtTheElevator] = field(default_factory=list)
    """
    The people the person manager tracks.
    """

    ask: ScriptedAsk | None = None
    """
    The skill whose refusals send the cabin off.
    """

    maximum_boarding_attempts: int = 3
    """
    How many times Tiago asks to board before it gives up.
    """

    next_elevator: NextElevator | None = field(init=False, default=None)
    """
    Moves the cabin after a refusal.
    """

    persons_at_elevator: PersonsAtElevator | None = field(init=False, default=None)
    """
    Finds the people to ask.
    """

    plan_world: World | None = field(init=False, default=None)
    """
    The world the plan was built for.
    """

    def build_plan(self, context: Context) -> PlanNode:
        world = context.world
        self.plan_world = world
        elevator = world.get_semantic_annotations_by_type(Elevator)[0]
        self.next_elevator = NextElevator(
            elevator,
            home_level=world.get_semantic_annotations_by_type(GroundFloor)[0],
            away_level=world.get_semantic_annotations_by_type(FirstFloor)[0],
        )
        self.ask.after_refusal = self.next_elevator.start_round
        self.persons_at_elevator = PersonsAtElevator(
            node=context.ros_node,
            topic_name=Ros4HriTopic.TRACKED_PERSONS,
            elevator=elevator,
        )
        track_people(self.persons_at_elevator, self.office, self.people)
        return sequential(
            [
                NavigateToRoomAskingToBoardAction(
                    self.office.get_room(
                        world, FirstFloor, AriseOfficeRoom.EAST_OFFICE
                    ),
                    persons_at_elevator=self.persons_at_elevator,
                    elevator_arrival_threshold=self.elevator_arrival_threshold,
                    maximum_boarding_attempts=self.maximum_boarding_attempts,
                )
            ],
            context=context,
        )

    def tear_down(self) -> None:
        if self.next_elevator is not None:
            self.next_elevator.wait_for_rounds()
        if self.persons_at_elevator is not None:
            self.persons_at_elevator.close()
            self.persons_at_elevator = None
        super().tear_down()


def riding_past(
    ask: ScriptedAsk,
    people: list[PersonAtTheElevator],
    maximum_boarding_attempts: int = 3,
) -> RidingPastPeopleAtTheElevator:
    """
    :return: A kinematically simulated demonstration with ``people`` at the elevator.
    """
    return RidingPastPeopleAtTheElevator(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        people=people,
        ask=ask,
        maximum_boarding_attempts=maximum_boarding_attempts,
    )


def stands_in_the_east_office_upstairs(
    world: World, demonstration: RidingPastPeopleAtTheElevator
) -> bool:
    """
    :return: Whether the robot of ``world`` reached the demonstration's target room.
    """
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    target_room = demonstration.office.get_room(
        world, FirstFloor, AriseOfficeRoom.EAST_OFFICE
    )
    return Floor.under(robot.root) is target_room.floor


# %% boarding


@pytest.mark.parametrize("scripted_ask", [[AskAnswer.NO]], indirect=True)
def test_tiago_boards_without_asking_when_nobody_is_at_the_elevator(scripted_ask):
    demonstration = riding_past(scripted_ask, people=[])

    world = demonstration.run()

    assert scripted_ask.goals == []
    assert stands_in_the_east_office_upstairs(world, demonstration)


@pytest.mark.parametrize("scripted_ask", [[AskAnswer.YES]], indirect=True)
def test_tiago_asks_the_person_waiting_and_boards_once_allowed(scripted_ask):
    person = person_waiting()
    demonstration = riding_past(scripted_ask, people=[person])

    world = demonstration.run()

    [goal] = scripted_ask.goals
    assert goal.person_id == person.person_id
    assert goal.question == BoardingQuestion.ASKING_TO_BOARD
    assert set(json.loads(goal.answers_schema)) == {BoardingAnswerKey.MAY_BOARD}
    assert stands_in_the_east_office_upstairs(world, demonstration)


@pytest.mark.parametrize("scripted_ask", [[AskAnswer.NO, AskAnswer.YES]], indirect=True)
def test_tiago_asks_again_once_the_next_cabin_arrived_after_a_refusal(scripted_ask):
    demonstration = riding_past(scripted_ask, people=[person_waiting()])

    world = demonstration.run()

    assert len(scripted_ask.goals) == 2
    [cabin_came_back] = demonstration.next_elevator.return_times
    assert scripted_ask.answer_times[0] < cabin_came_back
    assert cabin_came_back < scripted_ask.answer_times[1]
    assert stands_in_the_east_office_upstairs(world, demonstration)


# %% not boarding


@pytest.mark.parametrize("scripted_ask", [[AskAnswer.NO]], indirect=True)
def test_tiago_gives_up_after_being_refused_every_time(scripted_ask):
    maximum_boarding_attempts = 2
    demonstration = riding_past(
        scripted_ask,
        people=[person_waiting()],
        maximum_boarding_attempts=maximum_boarding_attempts,
    )

    with pytest.raises(BoardingPermissionNotGrantedError):
        demonstration.run()

    assert len(scripted_ask.goals) == maximum_boarding_attempts
    world = demonstration.plan_world
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    assert (
        Level.under(robot.root)
        is world.get_semantic_annotations_by_type(GroundFloor)[0]
    )


@pytest.mark.parametrize(
    "scripted_ask", [[AskAnswer.UNANSWERED], [AskAnswer.REJECT]], indirect=True
)
def test_anything_but_a_yes_counts_as_a_refusal(scripted_ask):
    demonstration = riding_past(
        scripted_ask, people=[person_waiting()], maximum_boarding_attempts=1
    )

    with pytest.raises(BoardingPermissionNotGrantedError):
        demonstration.run()

    assert scripted_ask.refusals == 1


# %% asking a person


@pytest.fixture
def boarding_question_asker(rclpy_node):
    asker = BoardingQuestionAsker(rclpy_node, server_timeout=0.2, answer_timeout=0.5)
    yield asker
    asker.close()


def test_asking_fails_when_nothing_serves_the_skill(boarding_question_asker):
    with pytest.raises(AskSkillNotServedError) as error:
        boarding_question_asker.may_board("waiting")

    assert error.value.action_name == CommunicationSkill.ASK
    assert error.value.timeout == boarding_question_asker.server_timeout


@pytest.mark.parametrize("scripted_ask", [[AskAnswer.KEEP_SILENT]], indirect=True)
def test_a_person_who_does_not_answer_in_time_does_not_agree(
    scripted_ask, boarding_question_asker
):
    assert boarding_question_asker.may_board("waiting") is False


@pytest.mark.parametrize(
    "scripted_ask, expected_agreement",
    [([AskAnswer.YES], True), ([AskAnswer.NO], False)],
    indirect=["scripted_ask"],
)
def test_the_answer_of_a_person_decides_whether_they_agree(
    scripted_ask, boarding_question_asker, expected_agreement
):
    assert boarding_question_asker.may_board("waiting") is expected_agreement


# %% waiting for the cabin


@pytest.fixture
def asking_at_a_cabin_that_is_elsewhere(rclpy_node):
    """
    :return: An executable asking on the ground floor, with the cabin on the first floor.
    """
    world = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    ).build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    elevator.drive_to_floor(world.get_semantic_annotations_by_type(FirstFloor)[0])
    persons = PersonsAtElevator(
        node=rclpy_node, topic_name="/test/humans/persons/tracked", elevator=elevator
    )
    yield AskToBoardExecutable(
        context=Context(world=world, robot=robot, ros_node=rclpy_node),
        elevator=elevator,
        level=world.get_semantic_annotations_by_type(GroundFloor)[0],
        robot_root=robot.root,
        persons_at_elevator=persons,
        maximum_attempts=1,
        arrival_threshold=0.01,
        cabin_timeout=0.2,
    )
    persons.close()


def test_waiting_gives_up_when_the_cabin_does_not_arrive(
    asking_at_a_cabin_that_is_elsewhere,
):
    with pytest.raises(ElevatorCabinTimedOutError) as error:
        asking_at_a_cabin_that_is_elsewhere.execute()

    assert error.value.timeout == asking_at_a_cabin_that_is_elsewhere.cabin_timeout
