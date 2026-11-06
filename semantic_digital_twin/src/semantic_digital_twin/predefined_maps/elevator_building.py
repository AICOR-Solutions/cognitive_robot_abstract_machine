from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
    from semantic_digital_twin.world import World

# %% buildings with an elevator


@dataclass
class ElevatorBuilding(ABC):
    """
    World-independent description of a building whose storeys one elevator connects,
    with rooms a person can name when speaking.
    """

    ground_floor_height: float = 0.0
    """
    Height of the lowest storey's floor above the world's origin.
    """

    @abstractmethod
    def spawn(self, world: World, name: str) -> object:
        """
        Add the building's storeys, with a level annotation each, and its elevator to
        ``world``.

        :param world: The world the building is added to.
        :param name: Prefix for the names of every entity the building creates.
        :return: Whatever the building hands back about what it added.
        """

    @abstractmethod
    def rooms_by_spoken_name(self, world: World) -> dict[str, Room]:
        """
        :param world: The world the building was spawned into.
        :return: The rooms a robot may be sent into, by how a person names them when
            speaking.
        """

    def find_room_by_spoken_name(self, world: World, spoken_name: str) -> Room | None:
        """
        :param world: The world the building was spawned into.
        :param spoken_name: How a person named the room, in any letter case.
        :return: The room named so by :meth:`rooms_by_spoken_name`, or None if there is
            none.
        """
        requested = spoken_name.strip().casefold()
        for known_name, room in self.rooms_by_spoken_name(world).items():
            if known_name.casefold() == requested:
                return room
        return None

    def rooms_named_in(self, world: World, sentence: str) -> dict[str, Room]:
        """
        :param world: The world the building was spawned into.
        :param sentence: What a person said, in any letter case.
        :return: The rooms whose whole name by :meth:`rooms_by_spoken_name` the sentence
            holds, by that name. A name that is only part of another named room's name
            does not count.
        """
        heard = sentence.casefold()
        named = {
            known_name: room
            for known_name, room in self.rooms_by_spoken_name(world).items()
            if re.search(rf"\b{re.escape(known_name.casefold())}\b", heard)
        }
        return {
            known_name: room
            for known_name, room in named.items()
            if not any(
                known_name != other and known_name.casefold() in other.casefold()
                for other in named
            )
        }
