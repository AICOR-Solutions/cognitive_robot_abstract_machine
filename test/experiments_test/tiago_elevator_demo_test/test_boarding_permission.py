import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

pytest.importorskip("communication_skills")

from action_msgs.msg import GoalStatus
from communication_skills.action import Say
from geometry_msgs.msg import TransformStamped
from hri_msgs.msg import IdsList
from rclpy.task import Future

from coraplex.plans.context_extensions import RobotAccess
from cramph.composites import ChildChooserAccess
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.ra_fetch_elevator_people.elevator_people import (
    Ros4HriTopic,
    TrackedPersons,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_request_demo.boarding_permission import (
    AbortOnStart,
    AskToBoardChooser,
    AskToBoardNode,
    BoardingAnswerInProgress,
    BoardingConversation,
    BoardingQuestion,
    BoardingQuestionAsker,
    NavigateToRoomAskingToBoardAction,
)
from experiments.tiago_room_request_demo.exceptions import (
    BoardingPermissionNotGrantedError,
    ElevatorCabinTimedOutError,
    SaySkillNotServedError,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from krrood.adapters.json_serializer import from_json, to_json
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
from .scripted_person import PersonAnswer, ScriptedPerson, scripted_person

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

    person: ScriptedPerson | None = None
    """
    The person whose refusals send the cabin off.
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

    def build_statechart(self, context: StatechartContext) -> Statechart:
        world = context.world
        self.plan_world = world
        elevator = world.get_semantic_annotations_by_type(Elevator)[0]
        self.next_elevator = NextElevator(
            elevator,
            home_level=world.get_semantic_annotations_by_type(GroundFloor)[0],
            away_level=world.get_semantic_annotations_by_type(FirstFloor)[0],
        )
        self.person.after_refusal = self.next_elevator.start_round
        self.persons_at_elevator = PersonsAtElevator(
            node=self.ros_node,
            topic_name=Ros4HriTopic.TRACKED_PERSONS,
            elevator=elevator,
        )
        track_people(self.persons_at_elevator, self.office, self.people)
        context.require_extension(ChildChooserAccess).choosers.append(
            AskToBoardChooser(
                persons_at_elevator=self.persons_at_elevator,
                asker=BoardingQuestionAsker(self.ros_node, Tiago.text_to_speech_action),
            )
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            NavigateToRoomAskingToBoardAction(
                target_room=self.office.get_room(
                    world, FirstFloor, AriseOfficeRoom.EAST_OFFICE
                ),
                elevator_arrival_threshold=self.elevator_arrival_threshold,
                maximum_boarding_attempts=self.maximum_boarding_attempts,
            )
        )
        return statechart

    def tear_down(self) -> None:
        if self.next_elevator is not None:
            self.next_elevator.wait_for_rounds()
        if self.persons_at_elevator is not None:
            self.persons_at_elevator.close()
            self.persons_at_elevator = None
        super().tear_down()


def riding_past(
    person: ScriptedPerson,
    people: list[PersonAtTheElevator],
    maximum_boarding_attempts: int = 3,
) -> RidingPastPeopleAtTheElevator:
    """
    :return: A kinematically simulated demonstration with ``people`` at the elevator.
    """
    return RidingPastPeopleAtTheElevator(
        used_robot=Tiago,
        people=people,
        person=person,
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


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.NO]], indirect=True)
def test_tiago_boards_without_asking_when_nobody_is_at_the_elevator(scripted_person):
    demonstration = riding_past(scripted_person, people=[])

    world = demonstration.run()

    assert scripted_person.goals == []
    assert stands_in_the_east_office_upstairs(world, demonstration)


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
def test_tiago_asks_the_person_waiting_and_boards_once_allowed(scripted_person):
    person = person_waiting()
    demonstration = riding_past(scripted_person, people=[person])

    world = demonstration.run()

    [goal] = scripted_person.goals
    assert goal.person_id == person.person_id
    assert goal.input == BoardingQuestion.ASKING_TO_BOARD
    assert stands_in_the_east_office_upstairs(world, demonstration)


@pytest.mark.parametrize(
    "scripted_person", [[PersonAnswer.NO, PersonAnswer.YES]], indirect=True
)
def test_tiago_asks_again_once_the_next_cabin_arrived_after_a_refusal(scripted_person):
    demonstration = riding_past(scripted_person, people=[person_waiting()])

    world = demonstration.run()

    assert len(scripted_person.goals) == 2
    [cabin_came_back] = demonstration.next_elevator.return_times
    assert scripted_person.answer_times[0] < cabin_came_back
    assert cabin_came_back < scripted_person.answer_times[1]
    assert stands_in_the_east_office_upstairs(world, demonstration)


# %% not boarding


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.NO]], indirect=True)
def test_tiago_gives_up_after_being_refused_every_time(scripted_person):
    maximum_boarding_attempts = 2
    demonstration = riding_past(
        scripted_person,
        people=[person_waiting()],
        maximum_boarding_attempts=maximum_boarding_attempts,
    )

    with pytest.raises(BoardingPermissionNotGrantedError):
        demonstration.run()

    assert len(scripted_person.goals) == maximum_boarding_attempts
    world = demonstration.plan_world
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    assert (
        Level.under(robot.root)
        is world.get_semantic_annotations_by_type(GroundFloor)[0]
    )


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.REJECT]], indirect=True)
def test_a_rejected_question_counts_as_a_refusal(scripted_person):
    demonstration = riding_past(
        scripted_person, people=[person_waiting()], maximum_boarding_attempts=1
    )

    with pytest.raises(BoardingPermissionNotGrantedError):
        demonstration.run()

    assert scripted_person.refusals == 1


# %% asking a person


@pytest.fixture
def boarding_question_asker(rclpy_node):
    asker = BoardingQuestionAsker(
        rclpy_node,
        Tiago.text_to_speech_action,
        server_timeout=0.2,
        answer_timeout=0.5,
    )
    yield asker
    asker.close()


def wait_for_the_answer(answer: BoardingAnswerInProgress, timeout: float = 5.0) -> bool:
    """
    :return: What the person answered.
    :raises AssertionError: If that is not known within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        may_board = answer.poll()
        if may_board is not None:
            return may_board
        time.sleep(0.01)
    raise AssertionError(f"no answer within {timeout} s")


def wait_for_the_skill(asker: BoardingQuestionAsker, timeout: float = 5.0) -> None:
    """
    :raises AssertionError: If the skill is not served within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if asker.is_served:
            return
        time.sleep(0.01)
    raise AssertionError(f"the skill was not served within {timeout} s")


@pytest.mark.parametrize(
    "scripted_person",
    [[PersonAnswer.KEEP_SILENT], [PersonAnswer.UNCLEAR]],
    indirect=True,
)
def test_a_person_who_gives_no_clear_answer_in_time_does_not_agree(
    scripted_person, boarding_question_asker
):
    wait_for_the_skill(boarding_question_asker)

    assert wait_for_the_answer(boarding_question_asker.ask("waiting")) is False


@pytest.mark.parametrize(
    "scripted_person, expected_agreement",
    [([PersonAnswer.YES], True), ([PersonAnswer.NO], False)],
    indirect=["scripted_person"],
)
def test_the_answer_of_a_person_decides_whether_they_agree(
    scripted_person, boarding_question_asker, expected_agreement
):
    wait_for_the_skill(boarding_question_asker)

    assert (
        wait_for_the_answer(boarding_question_asker.ask("waiting"))
        is expected_agreement
    )


# %% understanding what a person says


def answer_after_the_question() -> BoardingAnswerInProgress:
    """
    :return: A question the robot has finished saying.
    """
    answer = BoardingAnswerInProgress(answer_timeout=60.0)
    question_ended = Future()
    question_ended.set_result(
        SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED, result=Say.Result())
    )
    answer.on_result(question_ended)
    return answer


@pytest.mark.parametrize(
    "sentence, expected_agreement",
    [
        ("yes", True),
        ("Yes, please.", True),
        ("sure", True),
        ("no", False),
        ("no thank you", False),
        ("not now", False),
        ("yes no", False),
    ],
)
def test_what_a_person_says_decides_whether_they_agree(sentence, expected_agreement):
    answer = answer_after_the_question()

    answer.on_speech(sentence)

    assert answer.poll() is expected_agreement


@pytest.mark.parametrize("sentence", ["", "maybe later", "yesterday", "nobody knows"])
def test_a_sentence_without_a_clear_answer_leaves_the_answer_open(sentence):
    answer = answer_after_the_question()

    answer.on_speech(sentence)

    assert answer.poll() is None


def test_speech_during_the_question_does_not_count_as_an_answer():
    answer = BoardingAnswerInProgress(answer_timeout=60.0)

    answer.on_speech("yes")

    assert answer.poll() is None


# %% the conversation going wrong


@dataclass
class ConversationInAnOffice:
    """
    A conversation that has not started yet, on the ground floor of the office.
    """

    conversation: BoardingConversation
    """
    The conversation.
    """

    elevator: Elevator
    """
    The elevator it is about.
    """

    first_floor: FirstFloor
    """
    The storey above the robot.
    """


@pytest.fixture
def conversation_in_an_office(rclpy_node, boarding_question_asker):
    demonstration = TiagoElevatorRideDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    persons = PersonsAtElevator(
        node=rclpy_node, topic_name="/test/humans/persons/tracked", elevator=elevator
    )
    track_people(persons, demonstration.office, [person_waiting()])
    node = AskToBoardNode(
        elevator=elevator,
        level=world.get_semantic_annotations_by_type(GroundFloor)[0],
        maximum_attempts=1,
        arrival_threshold=0.01,
        cabin_timeout=0.0,
    )
    statechart = Statechart(
        context=StatechartContext.with_extensions(world, [RobotAccess(robot)])
    )
    statechart.add_node(node)
    yield ConversationInAnOffice(
        conversation=BoardingConversation(
            node=node, persons_at_elevator=persons, asker=boarding_question_asker
        ),
        elevator=elevator,
        first_floor=world.get_semantic_annotations_by_type(FirstFloor)[0],
    )
    persons.close()


def test_the_plan_ends_when_the_cabin_does_not_arrive(
    conversation_in_an_office,
):
    conversation_in_an_office.elevator.drive_to_floor(
        conversation_in_an_office.first_floor
    )
    time.sleep(0.01)

    child = conversation_in_an_office.conversation.advance()

    assert isinstance(child, AbortOnStart)
    assert isinstance(child.failure, ElevatorCabinTimedOutError)
    assert (
        child.failure.timeout
        == conversation_in_an_office.conversation.node.cabin_timeout
    )


def test_the_plan_ends_when_nothing_serves_the_skill(
    conversation_in_an_office,
):
    conversation = conversation_in_an_office.conversation
    conversation.asker.server_timeout = 0.0
    time.sleep(0.01)

    child = conversation.advance()

    assert isinstance(child, AbortOnStart)
    assert isinstance(child.failure, SaySkillNotServedError)
    assert child.failure.action_name == conversation.asker.text_to_speech_action
    assert child.failure.timeout == conversation.asker.server_timeout


@pytest.mark.parametrize(
    "failure",
    [
        BoardingPermissionNotGrantedError(3),
        ElevatorCabinTimedOutError(0.5),
        SaySkillNotServedError(Tiago.text_to_speech_action, 0.5),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_what_went_wrong_survives_the_way_to_the_controller(failure):
    """
    The controller runs the child ending the plan, and reports its exception back.
    """
    assert from_json(json.loads(json.dumps(to_json(failure)))) == failure


def test_the_node_ending_the_plan_survives_the_way_to_the_controller():
    context = StatechartContext.with_extensions(World(), [])
    statechart = Statechart(context=context)
    statechart.add_node(AbortOnStart(failure=ElevatorCabinTimedOutError(0.5)))

    received = Statechart.from_json(
        json.loads(json.dumps(statechart.to_json())), context=context
    )

    [node] = received.get_nodes_by_type(AbortOnStart)
    assert node.failure == ElevatorCabinTimedOutError(0.5)
