"""
Tiago is told which room of the IAI office to drive to and drives there.

Riding the elevator, it asks the people at it whether it may board, and waits for the
next cabin while they say no.

The room arrives as a ROS4HRI intent on ``/intents``, and Tiago drives once a person
confirmed it. Questions are said through the ROS4HRI ``say`` skill and answers are read
from the recognized speech; on the real TIAGo these come from PAL's communication stack.
The people at the elevator come from a ROS4HRI person manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PlanExecutor, RealRobotPlanExecutor
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.tiago_room_navigation_demo.exceptions import (
    TargetRoomNotReachedError,
)
from experiments.tiago_room_request_demo.communicating_navigation import (
    TiagoCommunicatingNavigationDemonstration,
)
from experiments.tiago_room_request_demo.room_request import RoomRequests
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoRoomRequestDemonstration(TiagoCommunicatingNavigationDemonstration):
    """
    Tiago waits to be told a room of the office and confirms it, then drives there,
    asking the people at the elevator whether it may board on the way.
    """

    ros_node_name: str = "tiago_room_request_demo_node"

    room_request_timeout: float = 300.0
    """
    Seconds Tiago waits to be told a room it knows.
    """

    room_requests: RoomRequests | None = field(init=False, default=None)
    """
    Listens for the room Tiago is told to drive to.
    """

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Wait to be told a room and confirm it, then drive there, asking to board the
        elevator. The elevator takes Tiago to the room's level.

        :raises NoRoomRequestedError: If nobody told a room the office has within
            :attr:`room_request_timeout`.
        """
        self.ask_to_board_in(context)
        self.room_requests = RoomRequests(
            node=self.ros_node,
            world=context.world,
            office=self.office,
            text_to_speech_action=context.require_extension(
                RobotAccess
            ).robot.text_to_speech_action,
        )
        requested_room = self.room_requests.wait_for_confirmed_room(
            self.room_request_timeout
        )
        # Every intent would take time from the updates of the world while the plan runs.
        self.room_requests.close()
        self.room_requests = None
        self.head_to(requested_room, context.world)
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=self.drive_to(requested_room)))
        return statechart

    def conclude(self, world: World) -> None:
        """
        :raises TargetRoomNotReachedError: If Tiago does not stand on the requested
            room's floor.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        if Floor.under(robot.root) is not self.destination.room.floor:
            raise TargetRoomNotReachedError(self.destination.room.name.name)

    def tear_down(self) -> None:
        if self.room_requests is not None:
            self.room_requests.close()
            self.room_requests = None
        super().tear_down()


def main(executor_type: type[PlanExecutor] = RealRobotPlanExecutor) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real (or Gazebo) robot or simulates it.
    """
    TiagoRoomRequestDemonstration(used_robot=Tiago, executor_type=executor_type).run()


if __name__ == "__main__":
    main()
