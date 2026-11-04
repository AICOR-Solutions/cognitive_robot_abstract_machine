"""
Being told which room of the ARISE office to drive to, through ROS4HRI move-to intents.

A chatbot or a dialogue manager publishes an ``hri_actions_msgs/Intent`` with the intent
``MOVE_TO`` once a person asks the robot to go somewhere. The intent's ``goal`` role
names the room as people speak of it, such as *east office on the first floor*.
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


# %% what the robot says


class RoomRequestAnnouncement(StrEnum):
    """
    What the robot says about a room it was asked to go to, with the room left open.
    """

    HEADING_TO = "I am heading to the {room}."
    """
    Said once the robot sets off to a room it knows.
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
    Listens for move-to intents naming a room of an office, keeping the first one that
    names a room the office has.

    Says aloud when an intent names a room the office does not have, unless
    :attr:`text_to_speech_action` is None.
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

    text_to_speech_action: str | None
    """
    The name of the action that makes the robot speak, or None if it cannot speak.
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

    _speech: ActionClient | None = field(init=False, default=None)
    """
    Sends what the robot says, None if it cannot speak.
    """

    def __post_init__(self) -> None:
        self._subscription = self.node.create_subscription(
            Intent, Ros4HriInteractionTopic.INTENTS, self._hear, 10
        )
        if self.text_to_speech_action is not None:
            self._speech = ActionClient(self.node, Say, self.text_to_speech_action)

    def _hear(self, intent: Intent) -> None:
        """
        Keep the room a move-to intent names, or say that the office does not have it.
        """
        if intent.intent != Intent.MOVE_TO:
            return
        spoken_name = MoveToIntentData.from_json(intent.data).goal
        if spoken_name is None:
            return
        room = self.office.find_room_by_spoken_name(self.world, spoken_name)
        if room is None:
            self._say(RoomRequestAnnouncement.UNKNOWN_ROOM.verbalize(spoken_name))
            return
        with self._heard:
            if self._requested_room is None:
                self._requested_room = RequestedRoom(spoken_name.strip(), room)
            self._heard.notify_all()

    def _say(self, sentence: str) -> None:
        """
        Say ``sentence`` aloud, if the robot can speak.
        """
        if self._speech is None:
            return
        self._speech.send_goal_async(Say.Goal(input=sentence))

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

    def close(self) -> None:
        """
        Stop listening and speaking.
        """
        self.node.destroy_subscription(self._subscription)
        if self._speech is not None:
            self._speech.destroy()
