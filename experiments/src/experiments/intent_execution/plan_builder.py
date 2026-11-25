"""
Builds the plans of the medicine-transport mission from ``hri_actions_msgs/Intent``
(REP-155): going to a place, and bringing an item to one.

Bringing, the robot collects the item in the room items are dispensed from, where people
put it on its tray, and drives to the place it goes, riding the elevator between
storeys, where people take it off again. The item is never looked for, so the intent
names it only in words; the places are entities of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from experiments.intent_execution.exceptions import (
    EntityInNoRoom,
    IntentNotPlannable,
    RoleNotGrounded,
)
from experiments.tiago_room_request_demo.communicating_navigation import (
    TiagoCommunicatingNavigationDemonstration,
)
from experiments.tiago_room_request_demo.room_request import RequestedRoom
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Room,
)

# %% the intent vocabulary


class PlannableIntent(StrEnum):
    """
    The intents a plan is built for, spelled as ``hri_actions_msgs/Intent`` defines
    them.
    """

    MOVE_TO = "__intent_move_to__"
    BRING_OBJECT = "__intent_bring_object__"


class IntentRole(StrEnum):
    """
    The thematic roles of an intent's ``data`` object a plan is built from.
    """

    OBJECT = "object"
    """
    What the robot brings, in the speaker's words.
    """

    GOAL = "goal"
    """
    The place the robot goes, or brings the item to: a room, or what stands in one.
    """


class TrayRequest(StrEnum):
    """
    What the robot asks the people who load and unload its tray, with the item left
    open; they answer with yes or no.
    """

    LOAD = "Please put the {item} on my tray. Is it on the tray now?"
    """
    Asked in the pickup room.
    """

    UNLOAD = "I have brought the {item}. Please take it off my tray. Have you taken it?"
    """
    Asked where the item goes.
    """

    def verbalize(self, item: str) -> str:
        """
        :return: The request, naming ``item``.
        """
        return self.format(item=item)


class TrayHandling(StrEnum):
    """
    What people do at the robot's tray, as a plan step names it, with the item left
    open.
    """

    LOADING = "have the {item} put on the tray"
    """
    People put the item on the tray.
    """

    UNLOADING = "have the {item} taken off the tray"
    """
    People take the item off the tray, delivering it.
    """

    def verbalize(self, item: str) -> str:
        """
        :return: The step's name, naming ``item``.
        """
        return self.format(item=item)


class TrayRecord(StrEnum):
    """
    What happened at the robot's tray once people handled it, as it is logged, with the
    item and the room left open.
    """

    COLLECTED = "collected the {item} in {room}"
    """
    People put the item on the tray in the pickup room.
    """

    DELIVERED = "delivered the {item} to {room}"
    """
    People took the item off the tray where it goes.
    """

    def verbalize(self, item: str, room: Room) -> str:
        """
        :return: The record, naming ``item`` and ``room``.
        """
        return self.format(item=item, room=room.name)


# %% plans


@dataclass
class PlanStep:
    """
    One step of a plan, as a person would name it.
    """

    summary: str = field(kw_only=True)
    """
    What the step does, in words.
    """


@dataclass
class DrivingStep(PlanStep):
    """
    The drive to the middle of a room, riding the elevator when the room lies on another
    storey than the robot stands on when the step starts.
    """

    destination: RequestedRoom = field(kw_only=True)
    """
    The room the robot drives to, with how it speaks of the room.
    """

    nodes: list[StatechartNode] = field(kw_only=True)
    """
    The nodes driving there, in order.
    """

    summary: str = field(init=False)

    def __post_init__(self) -> None:
        self.summary = f"drive to {self.destination.room.name}"

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        :return: The statechart performing :attr:`nodes` one after another, in
            ``context``.
        """
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=self.nodes))
        return statechart


@dataclass
class TrayStep(PlanStep):
    """
    People put an item on the robot's tray, or take it off, once the robot asked them
    to and they said they did.
    """

    item: str = field(kw_only=True)
    """
    The item, in the speaker's words.
    """

    room: Room = field(kw_only=True)
    """
    The room the robot stands in while people handle its tray.
    """

    unloading: bool = field(kw_only=True)
    """
    Whether people take the item off the tray, delivering it, rather than put it on.
    """

    summary: str = field(init=False)

    def __post_init__(self) -> None:
        handling = TrayHandling.UNLOADING if self.unloading else TrayHandling.LOADING
        self.summary = handling.verbalize(self.item)

    @property
    def request(self) -> str:
        """
        :return: What the robot asks the people at its tray.
        """
        request = TrayRequest.UNLOAD if self.unloading else TrayRequest.LOAD
        return request.verbalize(self.item)

    @property
    def record(self) -> str:
        """
        :return: What happened once people handled the tray, as it is logged.
        """
        record = TrayRecord.DELIVERED if self.unloading else TrayRecord.COLLECTED
        return record.verbalize(self.item, self.room)


@dataclass
class IntentPlan:
    """
    The steps carrying out one intent, each performed once the step before it is done.
    """

    intent: str = field(kw_only=True)
    """
    The name of the intent carried out.
    """

    steps: list[PlanStep] = field(kw_only=True)
    """
    The steps carrying out the intent, in order.
    """


# %% intent to plan


@dataclass
class IntentPlanBuilder:
    """
    Builds the plan of a move or a delivery from wherever the robot stands, so a plan
    built again after people helped carries on from there.
    """

    context: StatechartContext = field(kw_only=True)
    """
    The context the plan runs in.
    """

    demonstration: TiagoCommunicatingNavigationDemonstration = field(kw_only=True)
    """
    The demonstration the plan runs in, whose office's rooms a goal may name as people
    speak of them, and which drives the robot between them.
    """

    pickup_room: str = field(kw_only=True)
    """
    The name of the room items are collected from, standing in for the pharmacy.
    """

    def plan_for(
        self, intent: str, roles: dict[str, Any], collected: bool = False
    ) -> IntentPlan:
        """
        :param intent: The name of the intent.
        :param roles: The ``data`` object of the intent.
        :param collected: Whether the item is on the tray already, so a delivery does
            not collect it again.
        :return: The plan carrying out the intent.
        :raises IntentNotPlannable: If no plan is built for the intent.
        :raises RoleNotGrounded: If the intent names no goal, or a delivery no item.
        """
        if intent == PlannableIntent.MOVE_TO:
            steps = [self.driving_to(self.goal_room(intent, roles))]
        elif intent == PlannableIntent.BRING_OBJECT:
            steps = self.delivery(intent, roles, collected)
        else:
            raise IntentNotPlannable(
                intent=intent, plannable=[str(name) for name in PlannableIntent]
            )
        return IntentPlan(intent=intent, steps=steps)

    def delivery(
        self, intent: str, roles: dict[str, Any], collected: bool
    ) -> list[PlanStep]:
        """
        :return: Unless the item was ``collected``, the drive to the pickup room and
            people putting the item on the tray there; then the drive to the room of
            the goal, and people taking the item off the tray there.
        """
        item = roles.get(IntentRole.OBJECT)
        if not item:
            raise RoleNotGrounded(role=IntentRole.OBJECT, intent=intent)
        destination = self.goal_room(intent, roles)
        steps: list[PlanStep] = []
        if not collected:
            pickup = self.context.world.get_semantic_annotation_by_name(
                self.pickup_room
            )
            steps += [
                self.driving_to(pickup),
                TrayStep(item=item, room=pickup, unloading=False),
            ]
        return steps + [
            self.driving_to(destination),
            TrayStep(item=item, room=destination, unloading=True),
        ]

    def driving_to(self, room: Room) -> DrivingStep:
        """
        :return: The drive to the middle of ``room``, saying where the robot heads.
        """
        destination = RequestedRoom(spoken_name=self.spoken_name_of(room), room=room)
        return DrivingStep(
            destination=destination, nodes=self.demonstration.drive_to(destination)
        )

    # %% the places of the roles

    def goal_room(self, intent: str, roles: dict[str, Any]) -> Room:
        """
        :return: The room the goal names as people speak of it, or that the entity
            of that name is or stands in.
        :raises RoleNotGrounded: If the intent names no goal.
        """
        name = roles.get(IntentRole.GOAL)
        if name is None:
            raise RoleNotGrounded(role=IntentRole.GOAL, intent=intent)
        room = self.demonstration.office.find_room_by_spoken_name(
            self.context.world, name
        )
        if room is not None:
            return room
        prefix, _, unprefixed_name = name.rpartition("/")
        goal = self.context.world.get_semantic_annotation_by_name(
            PrefixedName(name=unprefixed_name, prefix=prefix) if prefix else name
        )
        return goal if isinstance(goal, Room) else self.room_of(goal)

    def spoken_name_of(self, room: Room) -> str:
        """
        :return: How people speak of ``room``, or its name where they have no word for
            it.
        """
        return next(
            (
                spoken_name
                for spoken_name, known_room in self.demonstration.office.rooms_by_spoken_name(
                    self.context.world
                ).items()
                if known_room is room
            ),
            room.name.name,
        )

    def room_of(self, entity: HasRootBody) -> Room:
        """
        :return: The room whose floor ``entity`` stands over.
        :raises EntityInNoRoom: If ``entity`` stands over the floor of no room.
        """
        floor = Floor.under(entity.root)
        room = next(
            (
                room
                for room in self.context.world.get_semantic_annotations_by_type(Room)
                if floor is not None and room.floor is floor
            ),
            None,
        )
        if room is None:
            raise EntityInNoRoom(entity=str(entity.name))
        return room
