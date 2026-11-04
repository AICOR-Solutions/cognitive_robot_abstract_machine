"""
Being told which room of the ARISE office to drive to, through ROS4HRI intents.

A chatbot or a dialogue manager publishes an ``hri_actions_msgs/Intent`` once a person
says something. A ``MOVE_TO`` intent names the room in its ``goal`` role, and a
``RAW_USER_INPUT`` holds the sentence a chatbot could not interpret, which may hold the
room. Rooms are named as people speak of them, such as *east office on the first floor*.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import StrEnum

from communication_skills.action import Say
from hri_actions_msgs.msg import Intent
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.subscription import Subscription

from experiments.tiago_room_request_demo.exceptions import NoRoomRequestedError
from experiments.tiago_room_request_demo.yes_no_question import YesNoQuestionAsker
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeBuilding
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World

# %% the intents


class Ros4HriInteractionTopic(StrEnum):
    """
    Topics of the ROS4HRI interaction architecture this module reads.
    """

    INTENTS = "/intents"


class IntentRole(StrEnum):
    """
    Thematic roles of an intent's data this module reads.
    """

    GOAL = "goal"
    """
    Where a move-to intent asks the robot to go.
    """


@dataclass
class MoveToIntentData:
    """
    The arguments of a move-to intent.
    """

    goal: str | None
    """
    Where the robot is asked to go, None if the intent does not say.
    """

    @classmethod
    def from_json(cls, data: str) -> MoveToIntentData:
        """
        :param data: The data of a move-to intent, empty if it carries none.
        :return: The arguments it holds.
        """
        return cls(goal=json.loads(data or "{}").get(IntentRole.GOAL))


class RawUserInputKey(StrEnum):
    """
    Keys of a raw user input's data this module reads.
    """

    SENTENCE = "input"
    """
    What the person said.
    """


@dataclass
class RawUserInputData:
    """
    The arguments of a raw user input.
    """

    sentence: str
    """
    What the person said, empty if the input does not say.
    """

    @classmethod
    def from_json(cls, data: str) -> RawUserInputData:
        """
        :param data: The data of a raw user input, empty if it carries none.
        :return: The arguments it holds.
        """
        return cls(sentence=json.loads(data or "{}").get(RawUserInputKey.SENTENCE, ""))


# %% what the robot says


class RoomRequestAnnouncement(StrEnum):
    """
    What the robot says about a room it was asked to go to, with any name left open.
    """

    CONFIRMING = "Should I go to the {room}?"
    """
    Asked before the robot sets off to a room it was asked to go to.
    """

    HEADING_TO = "I am heading to the {room}."
    """
    Said once the robot sets off to a room it knows.
    """

    STAYING = "Okay, I will stay here."
    """
    Said when a person did not confirm a room, without naming it, as the robot hearing
    itself name a room would take that for a new request.
    """

    UNKNOWN_ROOM = "I do not know the {room}."
    """
    Said when the robot is asked to go to a room it does not know.
    """

    def verbalize(self, room: str) -> str:
        """
        :param room: The room as it was spoken of.
        :return: This announcement with the room filled in.
        """
        return self.format(room=room)


# %% listening for requests


@dataclass
class RequestedRoom:
    """
    A room the robot was asked to go to.
    """

    spoken_name: str
    """
    How the room was spoken of.
    """

    room: Room
    """
    The room annotation it names.
    """


@dataclass
class RoomRequests:
    """
    Listens for people asking for a room of an office, keeping the first room the office
    has, until it is confirmed or refused.

    Says aloud when a move-to intent names a room the office does not have. A sentence
    that does not name exactly one room is ignored silently.
    """

    node: Node
    """
    The node the intents are listened to on.
    """

    world: World
    """
    The world the office was spawned into.
    """

    office: AriseOfficeBuilding
    """
    The office whose rooms can be requested.
    """

    text_to_speech_action: str
    """
    The name of the action that makes the robot speak.
    """

    _requested_room: RequestedRoom | None = field(init=False, default=None)
    """
    The first room requested that the office has.
    """

    _heard: threading.Condition = field(default_factory=threading.Condition, init=False)
    """
    Notified once a room of the office was requested.
    """

    _subscription: Subscription = field(init=False)
    """
    The subscription to the intents.
    """

    _speech: ActionClient = field(init=False)
    """
    Sends what the robot says.
    """

    _confirmation_asker: YesNoQuestionAsker = field(init=False)
    """
    Asks whether the robot understood the room correctly.
    """

    def __post_init__(self) -> None:
        self._subscription = self.node.create_subscription(
            Intent, Ros4HriInteractionTopic.INTENTS, self._hear, 10
        )
        self._speech = ActionClient(self.node, Say, self.text_to_speech_action)
        self._confirmation_asker = YesNoQuestionAsker(
            self.node, self.text_to_speech_action
        )

    def _hear(self, intent: Intent) -> None:
        """
        Keep the room an intent asks for, or say that the office does not have the room a
        move-to intent names.
        """
        if intent.intent == Intent.RAW_USER_INPUT:
            rooms = self.office.rooms_named_in(
                self.world, RawUserInputData.from_json(intent.data).sentence
            )
            if len(rooms) == 1:
                self._keep(RequestedRoom(*next(iter(rooms.items()))))
            return
        if intent.intent != Intent.MOVE_TO:
            return
        spoken_name = MoveToIntentData.from_json(intent.data).goal
        if spoken_name is None:
            return
        room = self.office.find_room_by_spoken_name(self.world, spoken_name)
        if room is None:
            self._speech.send_goal_async(
                Say.Goal(
                    input=RoomRequestAnnouncement.UNKNOWN_ROOM.verbalize(spoken_name)
                )
            )
            return
        self._keep(RequestedRoom(spoken_name.strip(), room))

    def _keep(self, requested_room: RequestedRoom) -> None:
        """
        Keep ``requested_room``, unless a room is waiting for its confirmation already.
        """
        with self._heard:
            if self._requested_room is None:
                self._requested_room = requested_room
            self._heard.notify_all()

    def wait_for_room(self, timeout: float) -> RequestedRoom:
        """
        :param timeout: Seconds to wait at most.
        :return: The first room requested that the office has.
        :raises NoRoomRequestedError: If no such room was requested within
            ``timeout``.
        """
        with self._heard:
            self._heard.wait_for(lambda: self._requested_room is not None, timeout)
            if self._requested_room is None:
                raise NoRoomRequestedError(timeout)
            return self._requested_room

    def wait_for_confirmed_room(self, timeout: float) -> RequestedRoom:
        """
        Ask about each room requested, saying so and forgetting those not confirmed.

        :param timeout: Seconds to wait at most for each request.
        :return: The first room a person confirmed.
        :raises NoRoomRequestedError: If no room was requested within ``timeout``.
        :raises SaySkillNotServedError: If the robot cannot ask.
        """
        while True:
            requested_room = self.wait_for_room(timeout)
            if self._confirmation_asker.ask_until_answered(
                RoomRequestAnnouncement.CONFIRMING.verbalize(requested_room.spoken_name)
            ):
                return requested_room
            self._speech.send_goal_async(
                Say.Goal(input=RoomRequestAnnouncement.STAYING)
            )
            with self._heard:
                self._requested_room = None

    def close(self) -> None:
        """
        Stop listening and speaking.
        """
        self.node.destroy_subscription(self._subscription)
        self._speech.destroy()
        self._confirmation_asker.close()
