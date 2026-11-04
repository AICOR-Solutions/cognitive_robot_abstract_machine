import json

import pytest

pytest.importorskip("communication_skills")
pytest.importorskip("hri_actions_msgs")

from geometry_msgs.msg import TransformStamped
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from experiments.ra_fetch_elevator_people.elevator_people import (
    Ros4HriTopic,
    TrackedPersons,
)
from experiments.tiago_room_request_demo.demo import TiagoRoomRequestDemonstration
from experiments.tiago_room_request_demo.exceptions import NoRoomRequestedError
from experiments.tiago_room_request_demo.room_request import (
    MoveToIntentData,
    RawUserInputData,
    RawUserInputKey,
    RoomRequestAnnouncement,
    RoomRequests,
)
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
    AriseOfficeRoom,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    FirstFloor,
    Floor,
)
from semantic_digital_twin.world import World

from .intent_publisher import RepeatedIntent
from .scripted_person import PersonAnswer, scripted_person

# %% fixtures


@pytest.fixture
def room_requests(rclpy_node, arise_office, arise_office_world):
    listener = RoomRequests(
        node=rclpy_node,
        world=arise_office_world,
        office=arise_office,
        text_to_speech_action=Tiago.text_to_speech_action,
    )
    yield listener
    listener.close()


@pytest.fixture
def repeated_intent(rclpy_node, request):
    intent_name, goal = request.param
    publisher = RepeatedIntent(
        node=rclpy_node, intent_name=intent_name, spoken_text=goal
    )
    yield publisher
    publisher.close()


@pytest.fixture
def repeated_raw_input(rclpy_node, request):
    publisher = RepeatedIntent(
        node=rclpy_node,
        intent_name=Intent.RAW_USER_INPUT,
        spoken_text=request.param,
        data_key=RawUserInputKey.SENTENCE,
    )
    yield publisher
    publisher.close()


@pytest.fixture
def person_manager_tracking_someone_waiting(rclpy_node, arise_office):
    """
    Publishes a person in the waiting area the way a ROS4HRI person manager does.

    :return: The person's ID.
    """
    person_id = "anna"
    transform = TransformStamped()
    transform.header.stamp = rclpy_node.get_clock().now().to_msg()
    transform.header.frame_id = World.create_with_root_body().root.name.name
    transform.child_frame_id = TrackedPersons.frame_of(person_id)
    transform.transform.translation.x = 0.85
    transform.transform.translation.y = 1.0
    transform.transform.translation.z = arise_office.ground_floor_height + 1.65
    transform.transform.rotation.w = 1.0
    frames = StaticTransformBroadcaster(rclpy_node)
    frames.sendTransform(transform)
    tracked = rclpy_node.create_publisher(IdsList, Ros4HriTopic.TRACKED_PERSONS, 10)
    timer = rclpy_node.create_timer(
        0.1, lambda: tracked.publish(IdsList(ids=[person_id]))
    )
    yield person_id
    rclpy_node.destroy_timer(timer)
    rclpy_node.destroy_publisher(tracked)


def east_office_upstairs() -> str:
    """
    :return: How people speak of the room requested in these tests.
    """
    return AriseOfficeBuilding.spoken_name_of(FirstFloor, AriseOfficeRoom.EAST_OFFICE)


# %% the arguments of an intent


def test_the_goal_of_a_move_to_intent_is_read_from_its_data():
    assert MoveToIntentData.from_json('{"goal": "lobby"}').goal == "lobby"


def test_a_move_to_intent_without_data_names_no_goal():
    assert MoveToIntentData.from_json("").goal is None


def test_the_sentence_of_a_raw_user_input_is_read_from_its_data():
    data = json.dumps({RawUserInputKey.SENTENCE: "go to the lobby"})

    assert RawUserInputData.from_json(data).sentence == "go to the lobby"


def test_a_raw_user_input_without_data_holds_no_sentence():
    assert RawUserInputData.from_json("").sentence == ""


# %% listening for requests


@pytest.mark.parametrize(
    "repeated_intent",
    [(Intent.MOVE_TO, east_office_upstairs())],
    indirect=True,
)
def test_a_move_to_intent_naming_a_room_requests_it(
    room_requests, repeated_intent, arise_office, arise_office_world
):
    requested_room = room_requests.wait_for_room(timeout=5.0)

    assert requested_room.spoken_name == repeated_intent.spoken_text
    assert requested_room.room is arise_office.get_room(
        arise_office_world, FirstFloor, AriseOfficeRoom.EAST_OFFICE
    )


@pytest.mark.parametrize(
    "repeated_intent",
    [(Intent.GREET, east_office_upstairs())],
    indirect=True,
)
def test_intents_other_than_moving_request_no_room(room_requests, repeated_intent):
    with pytest.raises(NoRoomRequestedError):
        room_requests.wait_for_room(timeout=1.0)


@pytest.mark.parametrize(
    "repeated_intent", [(Intent.MOVE_TO, "kitchen")], indirect=True
)
def test_a_room_the_office_does_not_have_is_said_to_be_unknown(
    room_requests, repeated_intent, tiago_speech
):
    [first_sentence, *_] = tiago_speech.wait_for_sentences(1)

    assert first_sentence == RoomRequestAnnouncement.UNKNOWN_ROOM.verbalize(
        repeated_intent.spoken_text
    )
    with pytest.raises(NoRoomRequestedError):
        room_requests.wait_for_room(timeout=0.5)


@pytest.mark.parametrize(
    "repeated_raw_input",
    [f"please go to the {east_office_upstairs()} now"],
    indirect=True,
)
def test_a_raw_user_input_naming_a_room_requests_it(
    room_requests, repeated_raw_input, arise_office, arise_office_world
):
    requested_room = room_requests.wait_for_room(timeout=5.0)

    assert requested_room.spoken_name == east_office_upstairs()
    assert requested_room.room is arise_office.get_room(
        arise_office_world, FirstFloor, AriseOfficeRoom.EAST_OFFICE
    )


@pytest.mark.parametrize(
    "repeated_raw_input",
    [
        "go into the kitchen",
        "east office",
        f"{east_office_upstairs()} or west office on the first floor",
    ],
    indirect=True,
)
def test_a_raw_user_input_naming_no_single_room_is_ignored_silently(
    room_requests, repeated_raw_input, tiago_speech
):
    with pytest.raises(NoRoomRequestedError):
        room_requests.wait_for_room(timeout=1.0)

    assert tiago_speech.sentences == []


# %% confirming a room


def request_the_room_once(room_requests: RoomRequests) -> None:
    """
    Let a person ask for the room once, the way the chatbot passes on what it heard.
    """
    room_requests._hear(
        Intent(
            intent=Intent.RAW_USER_INPUT,
            data=json.dumps({RawUserInputKey.SENTENCE: east_office_upstairs()}),
        )
    )


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
def test_a_room_counts_once_the_person_confirms_it(room_requests, scripted_person):
    request_the_room_once(room_requests)

    requested_room = room_requests.wait_for_confirmed_room(timeout=5.0)

    assert requested_room.spoken_name == east_office_upstairs()
    [goal] = scripted_person.confirmation_goals
    assert goal.input == RoomRequestAnnouncement.CONFIRMING.verbalize(
        east_office_upstairs()
    )


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
def test_a_refused_room_is_forgotten(room_requests, scripted_person):
    scripted_person.room_confirmations = [PersonAnswer.NO]
    request_the_room_once(room_requests)

    with pytest.raises(NoRoomRequestedError):
        room_requests.wait_for_confirmed_room(timeout=1.0)

    assert len(scripted_person.confirmation_goals) == 1


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
def test_the_robot_says_that_it_stays_when_a_room_is_refused(
    room_requests, scripted_person
):
    scripted_person.room_confirmations = [PersonAnswer.NO]
    request_the_room_once(room_requests)

    with pytest.raises(NoRoomRequestedError):
        room_requests.wait_for_confirmed_room(timeout=0.5)

    [_, acknowledgement] = scripted_person.wait_for_sentences(2)
    assert acknowledgement == RoomRequestAnnouncement.STAYING


def test_the_robot_does_not_name_the_room_when_it_stays():
    """
    The robot hearing itself name a room would take that for a new request.
    """
    assert (
        RoomRequestAnnouncement.STAYING.verbalize(east_office_upstairs())
        == RoomRequestAnnouncement.STAYING
    )


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
@pytest.mark.parametrize("repeated_raw_input", [east_office_upstairs()], indirect=True)
def test_the_robot_asks_again_when_a_refused_room_is_requested_again(
    room_requests, scripted_person, repeated_raw_input
):
    scripted_person.room_confirmations = [PersonAnswer.NO, PersonAnswer.YES]

    room_requests.wait_for_confirmed_room(timeout=5.0)

    assert len(scripted_person.confirmation_goals) == 2


# %% the demonstration


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
@pytest.mark.parametrize(
    "repeated_intent",
    [(Intent.MOVE_TO, east_office_upstairs())],
    indirect=True,
)
def test_tiago_drives_to_the_room_it_was_told(
    scripted_person, repeated_intent, arise_office
):
    demonstration = TiagoRoomRequestDemonstration(used_robot=Tiago)

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    target_room = arise_office.get_room(world, FirstFloor, AriseOfficeRoom.EAST_OFFICE)
    assert Floor.under(robot.root) is target_room.floor
    [confirmation, heading_to, *_] = scripted_person.wait_for_sentences(2)
    assert confirmation == RoomRequestAnnouncement.CONFIRMING.verbalize(
        repeated_intent.spoken_text
    )
    assert heading_to == RoomRequestAnnouncement.HEADING_TO.verbalize(
        repeated_intent.spoken_text
    )


@pytest.mark.parametrize("scripted_person", [[PersonAnswer.YES]], indirect=True)
@pytest.mark.parametrize(
    "repeated_intent",
    [(Intent.MOVE_TO, east_office_upstairs())],
    indirect=True,
)
def test_tiago_asks_the_person_the_person_manager_tracks_at_the_elevator(
    scripted_person, repeated_intent, person_manager_tracking_someone_waiting
):
    demonstration = TiagoRoomRequestDemonstration(used_robot=Tiago)

    demonstration.run()

    [goal] = scripted_person.goals
    assert goal.person_id == person_manager_tracking_someone_waiting


def test_tiago_gives_up_when_nobody_tells_it_a_room(rclpy_node):
    demonstration = TiagoRoomRequestDemonstration(
        used_robot=Tiago,
        room_request_timeout=0.2,
    )

    with pytest.raises(NoRoomRequestedError):
        demonstration.run()
