"""
Tiago is told which room of the ARISE office to drive to and drives there.

Riding the elevator, it asks the people at it whether it may board, and waits for the
next cabin while they say no.

The room arrives as a ROS4HRI move-to intent on ``/intents``, the answer through the
ROS4HRI ``ask`` skill on ``/skill/ask``; on the real TIAGo both come from PAL's
communication hub. The people at the elevator come from a ROS4HRI person manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.speech import SayAction
from experiments.ra_fetch_elevator_people.elevator_people import Ros4HriTopic
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.exceptions import (
    TargetRoomNotReachedError,
)
from experiments.tiago_room_request_demo.boarding_permission import (
    NavigateToRoomAskingToBoardAction,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from experiments.tiago_room_request_demo.room_request import (
    RequestedRoom,
    RoomRequestAnnouncement,
    RoomRequests,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Floor,
)
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoRoomRequestDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago waits to be told a room of the office, then drives there, asking the people at
    the elevator whether it may board on the way.
    """

    ros_node_name: str = "tiago_room_request_demo_node"

    room_request_timeout: float = 300.0
    """
    Seconds Tiago waits to be told a room it knows.
    """

    maximum_boarding_attempts: int = 3
    """
    How many times Tiago asks to board before it gives up.
    """

    requested_room: RequestedRoom | None = field(init=False, default=None)
    """
    The room Tiago was told to drive to, None until it was told.
    """

    room_requests: RoomRequests | None = field(init=False, default=None)
    """
    Listens for the room Tiago is told to drive to.
    """

    persons_at_elevator: PersonsAtElevator | None = field(init=False, default=None)
    """
    Finds the people Tiago asks to board.
    """

    def build_plan(self, context: Context) -> PlanNode:
        """
        Wait to be told a room, then drive there, asking to board the elevator.

        :raises NoRoomRequestedError: If nobody told a room the office has within
            :attr:`room_request_timeout`.
        """
        self.persons_at_elevator = PersonsAtElevator(
            node=context.ros_node,
            topic_name=Ros4HriTopic.TRACKED_PERSONS,
            elevator=context.world.get_semantic_annotations_by_type(Elevator)[0],
        )
        self.room_requests = RoomRequests(
            node=context.ros_node,
            world=context.world,
            office=self.office,
            text_to_speech_topic=context.robot.text_to_speech_topic,
        )
        self.requested_room = self.room_requests.wait_for_room(
            self.room_request_timeout
        )
        return sequential(
            [
                SayAction(
                    RoomRequestAnnouncement.HEADING_TO.verbalize(
                        self.requested_room.spoken_name
                    )
                ),
                NavigateToRoomAskingToBoardAction(
                    self.requested_room.room,
                    persons_at_elevator=self.persons_at_elevator,
                    elevator_arrival_threshold=self.elevator_arrival_threshold,
                    maximum_boarding_attempts=self.maximum_boarding_attempts,
                ),
            ],
            context=context,
        )

    def conclude(self, world: World) -> None:
        """
        :raises TargetRoomNotReachedError: If Tiago does not stand on the requested
            room's floor.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        if Floor.under(robot.root) is not self.requested_room.room.floor:
            raise TargetRoomNotReachedError(self.requested_room.room.name.name)

    def tear_down(self) -> None:
        if self.room_requests is not None:
            self.room_requests.close()
            self.room_requests = None
        if self.persons_at_elevator is not None:
            self.persons_at_elevator.close()
            self.persons_at_elevator = None
        super().tear_down()


def main(execution_type: ExecutionType = ExecutionType.REAL) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    TiagoRoomRequestDemonstration(used_robot=Tiago, execution_type=execution_type).run()


if __name__ == "__main__":
    main()
