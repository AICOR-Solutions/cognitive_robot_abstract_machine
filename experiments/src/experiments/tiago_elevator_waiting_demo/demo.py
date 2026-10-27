"""
Tiago waits in the waiting zone in front of the open elevator of the ARISE office and
hands its base over to the social navigation module (IRII), which drives it around the
people near the elevator into the cabin.

The people come from a ROS4HRI person manager: on the real TIAGo PAL's, in the Gazebo
simulator ``tiago_arise_humans`` (scenario ``elevator_waiting``). They are kept in the
world as humans, and those inside the cabin and in the waiting area in front of its
doors on the robot's level go into the boarding goal.

While the social navigation module has the base, the plan sends no motions; the module
drives the robot through ``/cmd_vel`` and gives the base back with its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from experiments.ra_fetch_elevator_people.elevator_people import (
    Ros4HriTopic,
    TrackedPersons,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.social_navigation import (
    BoardElevatorAction,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoElevatorWaitingDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives to the waiting zone in front of the elevator and hands its base over to
    the social navigation module to board.
    """

    ros_node_name: str = "tiago_elevator_waiting_demo_node"

    distance_from_doors: float = 1.5
    """
    How far in front of the elevator's doors the robot waits.
    """

    tracked_persons_update_period: float = 0.2
    """
    Seconds between two updates of the tracked persons in a real run.
    """

    tracked_persons: TrackedPersons | None = field(init=False, default=None)
    """
    The persons a ROS4HRI person manager tracks, kept in the world in a real run.
    """

    def waiting_pose(self, elevator: Elevator) -> Pose:
        """
        :return: Where the robot waits: :attr:`distance_from_doors` in front of the
            middle of the doors, facing the cabin, on the cabin's floor.
        """
        doorway = elevator.doorway_pose
        return Pose.from_xyz_rpy(
            x=float(doorway.x) - self.distance_from_doors,
            z=float(doorway.z),
            reference_frame=elevator.root,
        )

    def build_context(self, world: World) -> Context:
        """
        Unlike :meth:`TiagoElevatorRideDemonstration.build_context`, also keeps the
        persons a ROS4HRI person manager tracks in the world in a real run.
        """
        if self.execution_type is ExecutionType.REAL and self.tracked_persons is None:
            self._follow_tracked_persons(world)
        return super().build_context(world)

    def _follow_tracked_persons(self, world: World) -> None:
        """
        Keep a human in ``world`` for each person a ROS4HRI person manager tracks.
        """
        self.tracked_persons = TrackedPersons(
            world=world, node=self.ros_node, topic_name=Ros4HriTopic.TRACKED_PERSONS
        )
        tracked_persons_inputs = WorldStateInputs(world=world)
        tracked_persons_inputs.add_input(self.tracked_persons)
        self.ros_node.create_timer(
            self.tracked_persons_update_period, tracked_persons_inputs.synchronize
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Drive to the waiting zone and board the elevator.
        """
        elevator = context.world.get_semantic_annotations_by_type(Elevator)[0]
        return sequential(
            [
                NavigateAction(self.waiting_pose(elevator)),
                BoardElevatorAction(elevator),
            ],
            context=context,
        )

    def tear_down(self) -> None:
        if self.tracked_persons is not None:
            self.tracked_persons.close()
            self.tracked_persons = None
        super().tear_down()


def main(execution_type: ExecutionType = ExecutionType.REAL) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    TiagoElevatorWaitingDemonstration(
        used_robot=Tiago, execution_type=execution_type
    ).run()


if __name__ == "__main__":
    main()
