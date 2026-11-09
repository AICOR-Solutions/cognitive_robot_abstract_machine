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
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    PlanRewriting,
)
from coraplex.robot_plans.actions.core.navigation import (
    DrivesBase,
    ElevatorNavigation,
)
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive, SimulatorTopic
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
)
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Level,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Region

# %% operating the elevator in a kinematic simulation


@dataclass(eq=False)
class ElevatorCallButton(StateChangeCallback):
    """
    Calls an elevator's cabin to the level a robot waits on in front of its doors,
    standing in for the robot pressing the elevator's call button.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator this button calls.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose waiting calls the cabin.
    """

    waiting_areas: dict[Level, Region] = field(init=False)
    """
    Where the robot waits for the cabin, per level.
    """

    def __post_init__(self) -> None:
        self.waiting_areas = self.elevator.waiting_areas
        super().__post_init__()

    def on_state_change(self, **kwargs) -> None:
        level = Level.under(self.robot.root)
        if level is None or self.elevator.is_at_floor(level):
            return
        if not InsideRegion(
            self.robot.bodies_with_collision[0], self.waiting_areas[level]
        )():
            return
        # Moving the doors and the cabin changes the state this button listens to
        self.pause()
        self.elevator.serve_floor(level)
        self.resume()


@dataclass(eq=False)
class ElevatorOperator(ModelChangeCallback):
    """
    Brings an elevator's cabin to a robot waiting for it and drives the elevator to a
    floor as soon as the robot boards it, standing in for whatever operates the elevator
    in the real world.
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

    call_button: ElevatorCallButton = field(init=False)
    """
    The call button the robot presses by waiting in front of the doors.
    """

    def __post_init__(self) -> None:
        self.call_button = ElevatorCallButton(
            _world=self._world, elevator=self.elevator, robot=self.robot
        )
        super().__post_init__()

    def on_model_change(self, **kwargs) -> None:
        if self.robot.root.parent_kinematic_structure_entity is not self.elevator.root:
            return
        self.elevator.serve_floor(self.floor)

    def stop(self) -> None:
        """
        Differs from :meth:`~semantic_digital_twin.callbacks.callback.Callback.stop` in
        stopping :attr:`call_button` too.
        """
        self.call_button.stop()
        super().stop()


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
            if self.elevator_inputs is None:
                self._follow_elevator_drive(elevator)
            return [
                RobotAccess(robot),
                PlanRewriting(transformations=[ArriveWithinTheAccuracyOfTheRealBase()]),
            ]
        if self.elevator_operator is None:
            self.elevator_operator = ElevatorOperator(
                _world=world,
                elevator=elevator,
                floor=self.destination_level(world),
                robot=robot,
            )
        return [RobotAccess(robot)]

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

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Ride the elevator to :meth:`destination_level`.
        """
        statechart = Statechart(context=context)
        statechart.add_node(
            ElevatorNavigation(
                elevator=context.world.get_semantic_annotations_by_type(Elevator)[0],
                target_floor=self.destination_level(context.world),
                arrival_threshold=self.elevator_arrival_threshold,
            )
        )
        return statechart

    def tear_down(self) -> None:
        if self.elevator_operator is not None:
            self.elevator_operator.stop()
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
