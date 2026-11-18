"""
Tiago rides the elevator of the ARISE office from the ground floor to the first floor.

Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives the robot of
the TIAGo ARISE Gazebo simulator, or the real TIAGo: the world comes from the running
controller, the base lidar from ``/scan``, and the cabin's height from the elevator's joint
states. The elevator itself is operated outside the plan, by a person on the real robot and
by the simulator's elevator operator in Gazebo.

In a kinematic simulation an :class:`ElevatorOperator` stands in for whoever operates
the elevator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
    SimulatedPlanExecutor,
)
from coraplex.datastructures.enums import InsertionPosition
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    PlanRewriting,
    SequentialInsertion,
)
from coraplex.robot_plans.actions.core.navigation import (
    DrivesBase,
    ElevatorNavigation,
    NavigateAction,
    PassDoorAction,
)
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive, SimulatorTopic
from experiments.tiago_arise_gazebo.elevator_operator import SimulatedElevatorOperator
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.ros2_nodes.localization import (
    LoadMap,
    PublishInitialPose,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
)
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago, TiagoTopic
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Level,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Region

# %% operating the elevator in a kinematic simulation


@dataclass
class ElevatorOperator(TickingExecutorExtension):
    """
    Brings an elevator's cabin to a robot waiting in front of its doors after
    :attr:`call_duration` ticks, as if the robot pressed the call button.

    Once the robot boards, closes the doors, drives the cabin to a floor and opens the
    doors again after :attr:`ride_duration` ticks, standing in for whatever operates the
    elevator in the real world.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator this operator drives.
    """

    floor: Level = field(kw_only=True)
    """
    The floor the elevator is sent to once the robot is aboard.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose boarding sets the elevator off.
    """

    ride_duration: int = 500
    """
    How many ticks the doors stay closed after the robot boarded, long enough for it to
    turn towards them.
    """

    call_duration: int = 50
    """
    How many ticks a called cabin takes to arrive, during which the robot sees it away.
    """

    waiting_areas: dict[Level, Region] = field(init=False)
    """
    Where the robot waits for the cabin, per level.
    """

    robot_aboard: bool = field(default=False, init=False)
    """
    Whether the robot was aboard the elevator at the last tick.
    """

    ticks_until_doors_open: int | None = field(default=None, init=False)
    """
    How many more ticks the doors stay closed, None while the elevator is not riding.
    """

    ticks_until_cabin_arrives: int | None = field(default=None, init=False)
    """
    How many more ticks a called cabin takes to arrive, None while nobody calls it.
    """

    def __post_init__(self) -> None:
        # The areas join the world on first access, which must not happen mid-tick.
        self.waiting_areas = self.elevator.waiting_areas

    def before_tick(self, executor: StatechartExecutor) -> None:
        aboard = self.robot.root.parent_kinematic_structure_entity is self.elevator.root
        boarded = aboard and not self.robot_aboard
        self.robot_aboard = aboard
        if boarded:
            self.ticks_until_doors_open = self.ride_duration
            self.elevator.close()
            self.elevator.drive_to_floor(self.floor)
            return
        if self.ticks_until_doors_open is not None:
            self.ticks_until_doors_open -= 1
            if self.ticks_until_doors_open > 0:
                return
            self.ticks_until_doors_open = None
            self.elevator.open()
            return
        if not aboard:
            self.answer_call()

    def answer_call(self) -> None:
        """
        Call the cabin while the robot waits in front of the doors on a level the cabin
        is away from, and bring it there once :attr:`call_duration` ticks passed.
        """
        level = Level.under(self.robot.root)
        if (
            level is None
            or self.elevator.is_at_floor(level)
            or not InsideRegion(
                self.robot.bodies_with_collision[0], self.waiting_areas[level]
            )()
        ):
            self.ticks_until_cabin_arrives = None
            return
        if self.ticks_until_cabin_arrives is None:
            self.ticks_until_cabin_arrives = self.call_duration
        self.ticks_until_cabin_arrives -= 1
        if self.ticks_until_cabin_arrives > 0:
            return
        self.ticks_until_cabin_arrives = None
        self.elevator.serve_floor(level)


# %% driving the real base


@dataclass
class ArriveWithinTheAccuracyOfTheRealBase(ConfigurationTransformation[DrivesBase]):
    """
    Lets a real base count as arrived at a distance it can actually hold, which is wider
    than the one a simulated base reaches.
    """

    base_pose_threshold: float = 0.1
    """
    How far the base may be from a pose, in meters and in radians, to count as arrived.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return True

    def apply(self, plan_node: DrivesBase) -> None:
        plan_node.base_pose_threshold = self.base_pose_threshold


@dataclass
class RelocalizeFacingTheElevatorDoors(SequentialInsertion[ElevatorNavigation]):
    """
    Publishes where the world holds the robot as the new pose estimate of its
    localization, right after the robot turned towards the elevator's doors in the
    cabin.
    """

    topic_name: str = TiagoTopic.INITIAL_POSE
    """
    Topic the localization takes a new estimate of the robot's pose on.
    """

    floor_map_urls: dict[type[Level], str] = field(default_factory=dict)
    """
    The map the localization loads before the new estimate, for each floor the robot
    rides to; riding to a floor without one keeps the map the localization has.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.AFTER

    def is_applicable(self, plan_node: ElevatorNavigation) -> bool:
        return True

    def anchor(self, plan_node: ElevatorNavigation) -> StatechartNode:
        """
        :return: The step turning the robot towards the doors inside the cabin.
        """
        [turn_to_the_doors] = [
            step for step in plan_node.nodes if isinstance(step, NavigateAction)
        ]
        return turn_to_the_doors

    def nodes_to_insert(self, plan_node: ElevatorNavigation) -> list[StatechartNode]:
        relocalization = PublishInitialPose(
            topic_name=self.topic_name, base=plan_node.robot.root
        )
        map_url = self.floor_map_urls.get(type(plan_node.target_floor))
        if map_url is None:
            return [relocalization]
        return [LoadMap(map_url=map_url), relocalization]


@dataclass
class ArriveWithTheCabinOnTheTargetFloor(SequentialInsertion[ElevatorNavigation]):
    """
    Puts the elevator's cabin, with the robot aboard, at the height of the ride's target
    floor once the doors opened again, right before the robot leaves the cabin, for an
    elevator whose height nothing reports.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: ElevatorNavigation) -> bool:
        return True

    def anchor(self, plan_node: ElevatorNavigation) -> StatechartNode:
        """
        :return: The step moving the robot out from under the cabin, once the doors
            opened again.
        """
        exit_through_the_doors = [
            step for step in plan_node.nodes if isinstance(step, PassDoorAction)
        ][-1]
        [leaving_the_cabin] = [
            node
            for node in exit_through_the_doors.descendants
            if isinstance(node, MoveBranch)
        ]
        return leaving_the_cabin

    def nodes_to_insert(self, plan_node: ElevatorNavigation) -> list[StatechartNode]:
        elevator = plan_node.elevator
        return [
            SetSeedConfiguration(
                seed_configuration=JointState.from_mapping(
                    {
                        elevator.movable_joint: elevator.drive_position_for_floor(
                            plan_node.target_floor
                        )
                    }
                )
            )
        ]


# %% demonstration


@dataclass
class TiagoElevatorRideDemonstration(RobotDemonstration):
    """
    Tiago drives from the ground floor of the ARISE office into the elevator and leaves
    it on the first floor.
    """

    ros_node_name: str = "tiago_elevator_demo_node"

    office: ElevatorBuilding = field(default_factory=AriseOfficeBuilding)
    """
    The building the robot rides the elevator in.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where a simulated Tiago starts, as it is spawned in the Gazebo simulator.
    """

    elevator_joint_state_topic: str = SimulatorTopic.ELEVATOR_JOINT_STATES
    """
    Topic the elevator of a real run publishes its joint states on.
    """

    elevator_arrival_threshold: float = 0.03
    """
    How far the cabin may stand from a floor's height and still count as serving it; a
    loaded cabin sags by a few millimetres.
    """

    elevator_drive_update_period: float = 0.1
    """
    Seconds between two updates of the cabin's height from the elevator's joint states
    in a real run.
    """

    elevator_operator_parameter_service: str | None = None
    """
    The service through which a real run tells the simulator's elevator operator the
    destination floor, or None where a person operates the elevator.
    """

    elevator_operator: ElevatorOperator | None = field(init=False, default=None)
    """
    The operator of the elevator in a kinematic simulation.
    """

    elevator_inputs: WorldStateInputs | None = field(init=False, default=None)
    """
    Writes the cabin's height from the elevator's joint states in a real run.
    """

    @property
    def runs_on_a_real_robot(self) -> bool:
        """
        :return: Whether the plan drives the real (or Gazebo) robot.
        """
        return not issubclass(self.executor_type, SimulatedPlanExecutor)

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing but a map body, which :meth:`populate_scene`
            spawns the office into.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The Tiago on its drive on the office's ground floor.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.office.ground_floor_height
            ),
            odom_T_robot_start=self.odom_T_robot_start,
        )

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(Elevator))

    def populate_scene(self, world: World) -> None:
        """
        Spawn the office.

        In a kinematic simulation the elevator waits with open doors.
        """
        self.office.spawn(world)
        if not self.runs_on_a_real_robot:
            world.get_semantic_annotations_by_type(Elevator)[0].open()

    def acquire_world(self) -> World:
        """
        On a real run, the base lidar is read from the robot instead of the world.
        """
        world = super().acquire_world()
        if self.runs_on_a_real_robot:
            lidar = world.get_semantic_annotations_by_type(self.used_robot)[
                0
            ].mobile_base.lidar
            lidar.use_source(lidar.real_source(self.ros_node))
        return world

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Give the plan the Tiago in ``world`` and connect the elevator to whatever moves
        its cabin.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        elevator = world.get_semantic_annotations_by_type(Elevator)[0]
        if self.runs_on_a_real_robot:
            self.tell_elevator_operator_the_destination(world)
            if self.elevator_inputs is None:
                self._follow_elevator_drive(elevator)
            return [
                RobotAccess(robot),
                PlanRewriting(
                    transformations=[
                        ArriveWithinTheAccuracyOfTheRealBase(),
                        RelocalizeFacingTheElevatorDoors(
                            floor_map_urls={
                                GroundFloor: "/home/pal/maps/lab_map/map.yaml",
                                FirstFloor: "/home/pal/maps/first_floor/first_floor_map_aligned.yaml",
                            }
                        ),
                        ArriveWithTheCabinOnTheTargetFloor(),
                    ]
                ),
            ]
        if self.elevator_operator is None:
            self.elevator_operator = ElevatorOperator(
                elevator=elevator,
                floor=self.destination_level(world),
                robot=robot,
            )
        return [RobotAccess(robot)]

    def create_executor(self, world: World) -> PlanExecutor:
        """
        In a kinematic simulation the executor also ticks the :attr:`elevator_operator`.
        """
        executor = super().create_executor(world)
        if self.elevator_operator is not None:
            executor.extensions.append(self.elevator_operator)
        return executor

    def tell_elevator_operator_the_destination(self, world: World) -> None:
        """
        Have whatever operates the elevator of ``world`` take Tiago to
        :meth:`destination_level` once it boards: the :attr:`elevator_operator` of a
        kinematic simulation, or the simulator's elevator operator in a real run with an
        :attr:`elevator_operator_parameter_service`.

        A person operating the elevator is not told.
        """
        if not self.runs_on_a_real_robot:
            self.elevator_operator.floor = self.destination_level(world)
            return
        if self.elevator_operator_parameter_service is None:
            return
        SimulatedElevatorOperator(
            node=self.ros_node,
            parameter_service_name=self.elevator_operator_parameter_service,
        ).set_destination_floor(self.destination_floor_index(world))

    def _follow_elevator_drive(self, elevator: Elevator) -> None:
        """
        Keep the cabin's height in the world equal to the one the elevator reports.
        """
        self.elevator_inputs = WorldStateInputs(world=elevator._world)
        self.elevator_inputs.add_input(
            ElevatorDrive(
                world=elevator._world,
                node=self.ros_node,
                topic_name=self.elevator_joint_state_topic,
                elevator=elevator,
                joint_name=self.office.simulated_drive_joint_name(elevator),
                position_offset=self.office.ground_floor_height,
            )
        )
        self.ros_node.create_timer(
            self.elevator_drive_update_period, self.elevator_inputs.synchronize
        )

    def destination_level(self, world: World) -> Level:
        """
        :return: The level of ``world`` Tiago rides the elevator to, the first floor.
        """
        return world.get_semantic_annotations_by_type(FirstFloor)[0]

    def destination_floor_index(self, world: World) -> int:
        """
        :return: The position of :meth:`destination_level` among the floors of the
            elevator of ``world``, counted from the lowest.
        """
        return world.get_semantic_annotations_by_type(Elevator)[0].floors.index(
            self.destination_level(world)
        )

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Ride the elevator to :meth:`destination_level`.
        """
        statechart = Statechart(context=context)
        statechart.add_node(
            ElevatorNavigation(
                elevator=context.world.get_semantic_annotations_by_type(Elevator)[0],
                target_floor=self.destination_level(context.world),
            )
        )
        return statechart

    def tear_down(self) -> None:
        self.elevator_operator = None
        super().tear_down()


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real (or Gazebo) robot or simulates it.
    """
    TiagoElevatorRideDemonstration(used_robot=Tiago, executor_type=executor_type).run()


if __name__ == "__main__":
    main(executor_type=RealRobotPlanExecutor)
