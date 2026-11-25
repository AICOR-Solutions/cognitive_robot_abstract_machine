"""
Tiago driving into rooms of the IAI office while it communicates: it says which room it
heads to, asks the people at the elevator whether it may board, and says where it rides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.robot_plans.plan_transformations import ParkArmsBeforeFirstAction
from coraplex.robot_plans.actions.core.speech import SayAction
from cramph.composites import ChildChooserAccess
from cramph.context import ContextExtension, StatechartContext
from cramph.node import StatechartNode
from experiments.ra_fetch_elevator_people.elevator_people import Ros4HriTopic
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.demo import (
    TiagoIAIOfficeRoomNavigationDemonstration,
)
from experiments.tiago_room_request_demo.boarding_permission import (
    AskToBoardChooser,
    NavigateToRoomAskingToBoardAction,
)
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from experiments.tiago_room_request_demo.room_request import (
    RequestedRoom,
    RoomRequestAnnouncement,
)
from experiments.tiago_room_request_demo.yes_no_question import (
    YesNoAsker,
    YesNoQuestionAsker,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoCommunicatingNavigationDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives into rooms of the IAI office, saying which room it heads to and asking
    the people at the elevator whether it may board on the way.
    """

    office: ElevatorBuilding = field(default_factory=IAIOffice)
    """
    The building the robot drives in.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-19.0, y=39.0
        )
    )
    """
    Where a simulated Tiago starts, in the elevator room of the office.
    """

    obstacle_clearance: float = (
        TiagoIAIOfficeRoomNavigationDemonstration.obstacle_clearance
    )
    """
    How far Tiago's base keeps from obstacles on top of its base radius while it drives.
    """

    elevator_exit_clearance: float = (
        TiagoIAIOfficeRoomNavigationDemonstration.elevator_exit_clearance
    )
    """
    Distance Tiago keeps from the elevator's opening after driving out of the cabin, on
    top of half the cabin's depth.
    """

    maximum_boarding_attempts: int = 3
    """
    How many times Tiago asks to board before it gives up.
    """

    destination: RequestedRoom | None = field(init=False, default=None)
    """
    The room Tiago drives to, None until it heads anywhere.
    """

    persons_at_elevator: PersonsAtElevator | None = field(init=False, default=None)
    """
    Finds the people Tiago asks to board.
    """

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Differs from :meth:`TiagoElevatorRideDemonstration.build_context_extensions` in
        parking Tiago's arms before every drive, so they hang clear of the floor, the
        doorways and the cabin.
        """
        extensions = super().build_context_extensions(world)
        rewriting = next(
            (
                extension
                for extension in extensions
                if isinstance(extension, PlanRewriting)
            ),
            None,
        )
        if rewriting is None:
            rewriting = PlanRewriting()
            extensions.append(rewriting)
        rewriting.transformations.append(ParkArmsBeforeFirstAction())
        return extensions

    def ask_to_board_in(self, context: StatechartContext) -> AskToBoardChooser:
        """
        Let the plans run in ``context`` ask the people at the elevator whether Tiago
        may board.

        :return: The chooser asking, among the choosers of ``context``; it stops asking
            once a run in ``context`` ended.
        """
        if self.persons_at_elevator is None:
            self.persons_at_elevator = PersonsAtElevator(
                node=self.ros_node,
                topic_name=Ros4HriTopic.TRACKED_PERSONS,
                elevator=context.world.get_semantic_annotations_by_type(Elevator)[0],
            )
        chooser = AskToBoardChooser(
            persons_at_elevator=self.persons_at_elevator,
            asker=self.boarding_asker(context),
        )
        context.require_extension(ChildChooserAccess).choosers.append(chooser)
        return chooser

    def boarding_asker(self, context: StatechartContext) -> YesNoAsker:
        """
        :return: What asks the people at the elevator whether Tiago may board: the robot
            of ``context`` saying the question and hearing the answer itself.
        """
        return YesNoQuestionAsker(
            self.ros_node,
            context.require_extension(RobotAccess).robot.text_to_speech_action,
        )

    def drive_to(self, room: RequestedRoom) -> list[StatechartNode]:
        """
        :return: Tiago saying that it heads to ``room``, and its drive there, asking to
            board the elevator.
        """
        return [
            SayAction(
                text=RoomRequestAnnouncement.HEADING_TO.verbalize(room.spoken_name)
            ),
            NavigateToRoomAskingToBoardAction(
                target_room=room.room,
                elevator_arrival_threshold=self.elevator_arrival_threshold,
                obstacle_clearance=self.obstacle_clearance,
                elevator_exit_clearance=self.elevator_exit_clearance,
                maximum_boarding_attempts=self.maximum_boarding_attempts,
            ),
        ]

    def head_to(self, room: RequestedRoom, world: World) -> None:
        """
        Make ``room`` the :attr:`destination`, before Tiago drives there, and tell
        whatever operates the elevator its level.
        """
        self.destination = room
        self.tell_elevator_operator_the_destination(world)

    def destination_level(self, world: World) -> Level:
        """
        :return: The level of :attr:`destination`, or the level Tiago stands on before
            it heads anywhere.
        """
        if self.destination is None:
            return Level.under(
                world.get_semantic_annotations_by_type(self.used_robot)[0].root
            )
        return Level.under(self.destination.room.floor.root)

    def tear_down(self) -> None:
        if self.persons_at_elevator is not None:
            self.persons_at_elevator.close()
            self.persons_at_elevator = None
        super().tear_down()
