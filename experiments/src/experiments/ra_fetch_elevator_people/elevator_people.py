"""
Who is inside the elevator and who waits in front of it, from the people a ROS4HRI
(REP-155) pipeline tracks.

Tracked persons are mirrored into the semantic digital twin as :class:`Human` bodies at
their ``person_<personID>`` TF frame. Each update publishes, as ``hri_msgs/IdsList`` of
person IDs, who is inside the cabin and who is in the waiting area in front of its doors
on the level the robot is on, along with both areas' outlines.

Running :func:`main` against the TIAGo ARISE Gazebo simulator with its
``tiago_arise_humans`` launch gives the real data flow without a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import rclpy
from geometry_msgs.msg import Point32, PolygonStamped
from hri_msgs.msg import IdsList
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Time
from typing_extensions import Self

from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive, SimulatorTopic
from semantic_digital_twin.adapters.ros.input_synchronization import (
    TfFrameSynchronizer,
    TopicInputSynchronizer,
)
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeBuilding
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    ElevatorArea,
    Human,
    Level,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% topics


class Ros4HriTopic(StrEnum):
    """
    Topics of REP-155 this module reads.
    """

    TRACKED_PERSONS = "/humans/persons/tracked"


class RobotFrame(StrEnum):
    """
    TF frames of the robot this module looks up.
    """

    BASE = "base_footprint"


class ElevatorPeopleTopic(StrEnum):
    """
    Topics this module publishes.
    """

    CABIN_PERSONS = "/ra_fetch/elevator/cabin/persons"
    CABIN_FOOTPRINT = "/ra_fetch/elevator/cabin/footprint"
    WAITING_AREA_PERSONS = "/ra_fetch/elevator/waiting_area/persons"
    WAITING_AREA_FOOTPRINT = "/ra_fetch/elevator/waiting_area/footprint"


# %% tracked persons


@dataclass
class TrackedPersons(TopicInputSynchronizer[IdsList]):
    """
    Keeps one :class:`Human` in the world per person a ROS4HRI person manager tracks, at
    the person's TF frame, for as long as TF knows that frame.
    """

    body_radius: float = field(default=0.3, kw_only=True)
    """
    Radius of the cylinder standing in for a person's body.
    """

    body_height: float = field(default=1.8, kw_only=True)
    """
    Height of that cylinder, which hangs down from the person's frame at head height.
    """

    humans: dict[str, Human] = field(default_factory=dict, init=False)
    """
    The humans of the tracked persons by their ROS4HRI person ID.
    """

    person_frames: TfFrameSynchronizer = field(init=False)
    """
    Moves each human to its person's frame.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.person_frames = TfFrameSynchronizer(world=self.world, node=self.node)

    def apply_message(self, message: IdsList) -> None:
        """
        Make the world hold exactly the humans of the tracked persons whose frames TF
        knows, at those frames.
        """
        located_ids = [
            person_id
            for person_id in message.ids
            if self.person_frames.tf_wrapper.wait_for_transform(
                self.world.root.name.name,
                self.frame_of(person_id),
                Time(),
                Duration(),
            )
        ]
        for person_id in [
            person_id for person_id in self.humans if person_id not in located_ids
        ]:
            self._remove_human(person_id)
        for person_id in located_ids:
            if person_id not in self.humans:
                self.humans[person_id] = self._add_human(person_id)
        self.person_frames.apply()

    def close(self) -> None:
        """
        Unlike :meth:`InputSynchronizer.close`, also removes the humans of all persons
        from the world.
        """
        for person_id in list(self.humans):
            self._remove_human(person_id)
        super().close()

    def persons_in(self, area: ElevatorArea) -> list[str]:
        """
        :return: The IDs of the tracked persons inside ``area``.
        """
        return [
            person_id
            for person_id, human in self.humans.items()
            if InsideRegion(human.root, area.root)()
        ]

    @staticmethod
    def frame_of(person_id: str) -> str:
        """
        :return: The TF frame REP-155 publishes the person with ``person_id`` at.
        """
        return f"person_{person_id}"

    def _add_human(self, person_id: str) -> Human:
        """
        Add a human for ``person_id`` at the world root, as a cylinder of a person's
        size on a connection that follows the person's frame.

        :return: The new human.
        """
        body = Body(name=PrefixedName(self.frame_of(person_id), prefix="ros4hri"))
        cylinder = Cylinder(width=2 * self.body_radius, height=self.body_height)
        cylinder.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-self.body_height / 2, reference_frame=body
        )
        body.collision = ShapeCollection([cylinder], reference_frame=body)
        body.visual = body.collision
        human = Human(root=body)
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(body)
            connection = Connection6DoF.create_with_dofs(
                parent=self.world.root, child=body, world=self.world
            )
            self.world.add_connection(connection)
            self.world.add_semantic_annotation(human)
        self.person_frames.track(
            connection, self.world.root.name.name, self.frame_of(person_id)
        )
        return human

    def _remove_human(self, person_id: str) -> None:
        """
        Remove the human of ``person_id`` from the world.
        """
        human = self.humans.pop(person_id)
        self.person_frames.untrack(human.root.parent_connection)
        with self.world.modify_world():
            self.world.remove_semantic_annotation(human)
            self.world.remove_kinematic_structure_entity(human.root)


# %% publishing


@dataclass
class ElevatorPeople(HasROS2Node):
    """
    Publishes who of the tracked persons is inside an elevator and who waits in front of
    it on the robot's level.
    """

    elevator: Elevator
    """
    The elevator whose areas are watched.
    """

    tracked_persons: TrackedPersons = field(kw_only=True)
    """
    The persons to watch.
    """

    tf: TFWrapper = field(kw_only=True)
    """
    Where the persons' and the robot's frames are looked up.
    """

    robot_frame: str = field(default=RobotFrame.BASE, kw_only=True)
    """
    TF frame of the robot's base, whose height decides the robot's level.
    """

    floor_tolerance: float = field(default=0.2, kw_only=True)
    """
    How far a level's floor may lie above the robot's base frame for the robot to count
    as standing on it.
    """

    publishers: dict[ElevatorPeopleTopic, Publisher] = field(init=False)
    """
    The publisher of each topic this class publishes.
    """

    def __post_init__(self) -> None:
        message_types = {
            ElevatorPeopleTopic.CABIN_PERSONS: IdsList,
            ElevatorPeopleTopic.CABIN_FOOTPRINT: PolygonStamped,
            ElevatorPeopleTopic.WAITING_AREA_PERSONS: IdsList,
            ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT: PolygonStamped,
        }
        self.publishers = {
            topic: self.node.create_publisher(message_type, topic, 1)
            for topic, message_type in message_types.items()
        }

    @classmethod
    def for_world(
        cls,
        node: Node,
        world: World,
        robot_frame: str,
    ) -> Self:
        """
        Watch the only elevator of ``world``.
        """
        return cls(
            node=node,
            elevator=world.get_semantic_annotations_by_type(Elevator)[0],
            tracked_persons=TrackedPersons(
                world=world, node=node, topic_name=Ros4HriTopic.TRACKED_PERSONS
            ),
            tf=TFWrapper(node=node),
            robot_frame=robot_frame,
        )

    def update(self) -> None:
        """
        Update the tracked persons and publish both areas with the persons inside them.
        """
        self.tracked_persons.apply()
        self._publish(
            self.elevator.cabin_area,
            ElevatorPeopleTopic.CABIN_PERSONS,
            ElevatorPeopleTopic.CABIN_FOOTPRINT,
        )
        robot_level = self.robot_level()
        if robot_level is None:
            return
        self._publish(
            self.elevator.waiting_area_on(robot_level),
            ElevatorPeopleTopic.WAITING_AREA_PERSONS,
            ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT,
        )

    def robot_level(self) -> Level | None:
        """
        :return: The highest level whose floor is at most :attr:`floor_tolerance` above
            the robot's base, or None if the robot's frame is unknown or below every level.
        """
        world_T_robot = self._lookup(self.robot_frame)
        if world_T_robot is None:
            return None
        robot_height = float(world_T_robot.z)
        levels_below = [
            level
            for level in self.elevator._world.get_semantic_annotations_by_type(Level)
            if float(level.floor_plane[0].z) <= robot_height + self.floor_tolerance
        ]
        if not levels_below:
            return None
        return max(levels_below, key=lambda level: float(level.floor_plane[0].z))

    def _lookup(self, frame: str) -> HomogeneousTransformationMatrix | None:
        """
        :return: The pose of ``frame`` in the world root frame, or None if TF does not
            know it yet.
        """
        world = self.elevator._world
        if not self.tf.tf_buffer.can_transform(world.root.name.name, frame, Time()):
            return None
        transform = self.tf.lookup_transform(world.root.name.name, frame).transform
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=transform.translation.x,
            pos_y=transform.translation.y,
            pos_z=transform.translation.z,
            quat_x=transform.rotation.x,
            quat_y=transform.rotation.y,
            quat_z=transform.rotation.z,
            quat_w=transform.rotation.w,
            reference_frame=world.root,
        )

    def _publish(
        self,
        area: ElevatorArea,
        persons_topic: ElevatorPeopleTopic,
        footprint_topic: ElevatorPeopleTopic,
    ) -> None:
        """
        Publish the persons inside ``area`` and the area's footprint.
        """
        stamp = self.node.get_clock().now().to_msg()
        frame_id = self.elevator._world.root.name.name
        persons = IdsList(ids=self.tracked_persons.persons_in(area))
        persons.header.stamp, persons.header.frame_id = stamp, frame_id
        self.publishers[persons_topic].publish(persons)
        footprint = PolygonStamped()
        footprint.header.stamp, footprint.header.frame_id = stamp, frame_id
        footprint.polygon.points = [
            Point32(x=float(corner.x), y=float(corner.y), z=float(corner.z))
            for corner in area.footprint()
        ]
        self.publishers[footprint_topic].publish(footprint)


# %% running against the Gazebo simulator


def main(rate: float = 5.0, robot_frame: str = RobotFrame.BASE) -> None:
    """
    Watch the elevator of the ARISE Gazebo office.

    :param rate: Updates per second.
    :param robot_frame: TF frame of the simulated robot's base.
    """
    rclpy.init()
    node = rclpy.create_node("ra_fetch_elevator_people")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    office = AriseOfficeBuilding()
    world = office.create_world()
    elevator_people = ElevatorPeople.for_world(node, world, robot_frame)
    elevator_inputs = WorldStateInputs(world=world)
    elevator_inputs.add_input(
        ElevatorDrive(
            world=world,
            node=node,
            topic_name=SimulatorTopic.ELEVATOR_JOINT_STATES,
            elevator=elevator_people.elevator,
            position_offset=office.ground_floor_height,
        )
    )
    node.create_timer(1.0 / rate, elevator_inputs.synchronize)
    node.create_timer(1.0 / rate, elevator_people.update)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
