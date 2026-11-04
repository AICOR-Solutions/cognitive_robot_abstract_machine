"""
Who stands at an elevator, from the people a ROS4HRI (REP-155) person manager tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from hri_msgs.msg import IdsList
from rclpy.duration import Duration
from rclpy.time import Time

from experiments.ra_fetch_elevator_people.elevator_people import TrackedPersons
from semantic_digital_twin.adapters.ros.latest_message_subscriber import (
    LatestMessageSubscriber,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% persons


@dataclass
class PersonPosition:
    """
    Where a tracked person is.
    """

    person_id: str
    """
    The person's ROS4HRI person ID.
    """

    head_position: Point3
    """
    Where the person's frame is, at head height, in the world root frame.
    """


# The persons stay out of the world: a model change stops the robot until the controller
# compiled again, and the state of a body only this process knows makes its state updates
# unknown there.
@dataclass
class PersonsAtElevator(LatestMessageSubscriber[IdsList]):
    """
    Finds the tracked persons inside an elevator's cabin and in front of its doors,
    without adding them to the world.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator the persons are looked for at.
    """

    tf_wrapper: TFWrapper = field(init=False)
    """
    Provides the frames of the persons.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tf_wrapper = TFWrapper(node=self.node)

    def positions(self) -> list[PersonPosition]:
        """
        :return: Where the tracked persons are, for those whose frames TF knows.
        """
        if self.latest_message is None:
            return []
        world_root = self.elevator._world.root
        root_frame = world_root.name.name
        positions = []
        for person_id in self.latest_message.ids:
            person_frame = TrackedPersons.frame_of(person_id)
            if not self.tf_wrapper.wait_for_transform(
                root_frame, person_frame, Time(), Duration()
            ):
                continue
            point = self.tf_wrapper.lookup_point(root_frame, person_frame).point
            positions.append(
                PersonPosition(
                    person_id,
                    Point3(point.x, point.y, point.z, reference_frame=world_root),
                )
            )
        return positions

    def nearest_to(self, entity: KinematicStructureEntity, level: Level) -> str | None:
        """
        :param entity: The entity the distance to the persons is measured from.
        :param level: The level whose waiting area is looked at.
        :return: The ID of the person nearest to ``entity`` among those inside the cabin
            and those waiting in front of its doors on ``level``, None if there is none.
        """
        world_root = self.elevator._world.root
        bounding_boxes = [
            area.area.as_bounding_box_collection_in_frame(world_root)
            for area in (self.elevator.cabin_area, self.elevator.waiting_areas[level])
        ]
        persons_at_elevator = [
            person
            for person in self.positions()
            if any(boxes.contains(person.head_position) for boxes in bounding_boxes)
        ]
        if not persons_at_elevator:
            return None
        entity_position = entity.global_transform.to_np()[:3, 3]
        return min(
            persons_at_elevator,
            key=lambda person: np.linalg.norm(
                person.head_position.to_np()[:3] - entity_position
            ),
        ).person_id
