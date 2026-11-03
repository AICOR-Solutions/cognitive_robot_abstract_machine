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
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
)
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

# %% operating the elevator in a kinematic simulation


@dataclass(eq=False)
class ElevatorOperator(ModelChangeCallback):
    """
    Drives an elevator to a floor as soon as a robot boards it, standing in for whatever
    operates the elevator in the real world.
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

    def on_model_change(self, **kwargs) -> None:
        if self.robot.root.parent_kinematic_structure_entity is not self.elevator.root:
            return
        self.elevator.close()
        self.elevator.drive_to_floor(self.floor)
        self.elevator.open()


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

    office: AriseOfficeBuilding = field(default_factory=AriseOfficeBuilding)
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
        elevator = self.office.spawn(world)
        if not self.runs_on_a_real_robot:
            elevator.open()

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
                floor=world.get_semantic_annotations_by_type(FirstFloor)[0],
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
                position_offset=self.office.ground_floor_height,
            )
        )
        self.ros_node.create_timer(
            self.elevator_drive_update_period, self.elevator_inputs.synchronize
        )

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Ride the elevator to the first floor.
        """
        statechart = Statechart(context=context)
        statechart.add_node(
            ElevatorNavigation(
                elevator=context.world.get_semantic_annotations_by_type(Elevator)[0],
                target_floor=context.world.get_semantic_annotations_by_type(FirstFloor)[
                    0
                ],
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
