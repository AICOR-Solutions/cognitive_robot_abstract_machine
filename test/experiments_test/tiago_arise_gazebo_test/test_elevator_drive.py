import pytest
from sensor_msgs.msg import JointState

from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator

# %% fixtures


@pytest.fixture
def elevator(arise_office_world) -> Elevator:
    return arise_office_world.get_semantic_annotations_by_type(Elevator)[0]


@pytest.fixture
def drive(rclpy_node, elevator, arise_office) -> ElevatorDrive:
    drive = ElevatorDrive(
        world=elevator._world,
        node=rclpy_node,
        topic_name="/test/elevator/joint_states",
        elevator=elevator,
        position_offset=arise_office.ground_floor_height,
    )
    yield drive
    drive.close()


# %% writing the cabin's height


def test_drive_moves_the_cabin_to_the_received_joint_position(
    drive, elevator, arise_office
):
    drive.buffer_message(JointState(name=[drive.joint_name], position=[2.0]))
    assert drive.apply()
    assert elevator.movable_joint.position == pytest.approx(
        2.0 + arise_office.ground_floor_height
    )


def test_drive_writes_nothing_without_a_joint_state(drive, elevator):
    position = elevator.movable_joint.position
    assert not drive.apply()
    assert elevator.movable_joint.position == position


def test_drive_writes_nothing_for_joint_states_of_other_joints(drive, elevator):
    position = elevator.movable_joint.position
    drive.buffer_message(JointState(name=["another_joint"], position=[2.0]))
    assert not drive.apply()
    assert elevator.movable_joint.position == position


def test_drive_writes_nothing_while_the_cabin_stands_still(
    drive, elevator, arise_office
):
    drive.buffer_message(
        JointState(
            name=[drive.joint_name],
            position=[
                elevator.movable_joint.position - arise_office.ground_floor_height
            ],
        )
    )
    state_version = elevator._world.state.version
    assert not drive.apply()
    assert elevator._world.state.version == state_version
