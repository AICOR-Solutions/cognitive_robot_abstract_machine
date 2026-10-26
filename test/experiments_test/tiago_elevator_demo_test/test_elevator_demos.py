import pytest

from coraplex.datastructures.enums import ExecutionType
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
)

# %% riding the elevator


def test_tiago_rides_the_elevator_to_the_first_floor(rclpy_node):
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(first_floor.floor_plane[0].z) - demonstration.office.ground_floor_height
    )
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )
