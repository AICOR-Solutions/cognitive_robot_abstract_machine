import pytest
from sensor_msgs.msg import JointState

from geometry_msgs.msg import TransformStamped

from experiments.ra_fetch_elevator_people.elevator_people import (
    ElevatorDrive,
    ElevatorPeople,
    PersonMirror,
    RobotFrame,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Human,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

# %% fixtures


@pytest.fixture
def elevator(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(Elevator)[0]


def person_frame_at(
    world, x: float, y: float, floor_height: float
) -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x,
        y,
        floor_height + 1.65,
        reference_frame=world.root,
    )


# %% mirroring persons


def test_placed_person_becomes_a_human_at_its_frame(arise_office_world, arise_office):
    mirror = PersonMirror(arise_office_world)
    world_T_person = person_frame_at(
        arise_office_world, 1.0, -2.0, arise_office.ground_floor_height
    )
    human = mirror.place("anna", world_T_person)
    assert human in arise_office_world.get_semantic_annotations_by_type(Human)
    assert human.root.global_transform.to_np() == pytest.approx(world_T_person.to_np())


def test_placing_a_person_again_moves_its_human(arise_office_world, arise_office):
    mirror = PersonMirror(arise_office_world)
    first = mirror.place(
        "anna",
        person_frame_at(
            arise_office_world, 1.0, -2.0, arise_office.ground_floor_height
        ),
    )
    world_T_person = person_frame_at(
        arise_office_world, 2.0, -1.0, arise_office.ground_floor_height
    )
    second = mirror.place("anna", world_T_person)
    assert second is first
    assert second.root.global_transform.to_np() == pytest.approx(world_T_person.to_np())


def test_persons_no_longer_tracked_leave_the_world(arise_office_world, arise_office):
    mirror = PersonMirror(arise_office_world)
    gone = mirror.place(
        "anna",
        person_frame_at(
            arise_office_world, 1.0, -2.0, arise_office.ground_floor_height
        ),
    )
    mirror.place(
        "ben",
        person_frame_at(
            arise_office_world, 2.0, -2.0, arise_office.ground_floor_height
        ),
    )
    mirror.keep_only(["ben"])
    assert list(mirror.humans) == ["ben"]
    assert gone not in arise_office_world.get_semantic_annotations_by_type(Human)
    assert not arise_office_world.is_kinematic_structure_entity_in_world_by_name(
        gone.root.name.name
    )


def test_persons_are_sorted_into_the_elevator_areas(
    arise_office_world, elevator, arise_office
):
    mirror = PersonMirror(arise_office_world)
    mirror.place(
        "rider",
        person_frame_at(
            arise_office_world,
            arise_office.cabin_centre.x,
            arise_office.cabin_centre.y,
            arise_office.ground_floor_height,
        ),
    )
    mirror.place(
        "waiting",
        person_frame_at(
            arise_office_world, 0.45, 1.2, arise_office.ground_floor_height
        ),
    )
    mirror.place(
        "walker",
        person_frame_at(
            arise_office_world, -5.0, -2.5, arise_office.ground_floor_height
        ),
    )
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert mirror.persons_in(elevator.cabin_area) == ["rider"]
    assert mirror.persons_in(elevator.waiting_area_on(ground_floor)) == ["waiting"]
    assert mirror.persons_in(elevator.waiting_area_on(first_floor)) == []


# %% elevator drive


def test_drive_moves_the_cabin_to_the_received_joint_position(
    rclpy_node, elevator, arise_office
):
    drive = ElevatorDrive(
        node=rclpy_node,
        topic_name="/test/elevator/joint_states",
        position_offset=arise_office.ground_floor_height,
    )
    drive.buffer_message(JointState(name=[drive.joint_name], position=[2.0]))
    drive.apply_to(elevator)
    assert elevator.movable_joint.position == pytest.approx(
        2.0 + arise_office.ground_floor_height
    )


def test_drive_leaves_the_cabin_without_a_joint_state(rclpy_node, elevator):
    drive = ElevatorDrive(node=rclpy_node, topic_name="/test/elevator/joint_states")
    position = elevator.movable_joint.position
    drive.apply_to(elevator)
    assert elevator.movable_joint.position == position


def test_drive_leaves_the_world_alone_while_the_cabin_stands_still(
    rclpy_node, elevator
):
    drive = ElevatorDrive(node=rclpy_node, topic_name="/test/elevator/joint_states")
    drive.buffer_message(
        JointState(name=[drive.joint_name], position=[elevator.movable_joint.position])
    )
    state_version = elevator._world.state.version
    drive.apply_to(elevator)
    assert elevator._world.state.version == state_version


# %% the robot's level


def robot_base_at_height(world, height: float) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = world.root.name.name
    transform.child_frame_id = RobotFrame.BASE
    transform.transform.translation.z = height
    transform.transform.rotation.w = 1.0
    return transform


def test_robot_stands_on_the_level_under_its_base(
    rclpy_node, arise_office_world, arise_office
):
    elevator_people = ElevatorPeople.for_world(
        rclpy_node, arise_office_world, RobotFrame.BASE, None
    )
    elevator_people.tf.tf_buffer.set_transform_static(
        robot_base_at_height(arise_office_world, arise_office.first_floor_height),
        "test",
    )
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert elevator_people.robot_level() is first_floor


def test_robot_level_is_unknown_without_the_robots_frame(
    rclpy_node, arise_office_world
):
    elevator_people = ElevatorPeople.for_world(
        rclpy_node, arise_office_world, RobotFrame.BASE, None
    )
    assert elevator_people.robot_level() is None
