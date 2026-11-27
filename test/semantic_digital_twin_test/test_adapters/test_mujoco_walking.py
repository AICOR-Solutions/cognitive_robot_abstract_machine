"""
Tests for a robot walking on legs in a stepped MuJoCo simulation: its base floats, its
walking controller drives its legs, and the base's measured pose is the drive's
odometry.
"""

from __future__ import annotations

from datetime import timedelta

import mujoco
import numpy as np
import pytest

from cramph.executor import SteppedSimulationPacer
from semantic_digital_twin.robots.walking import (
    PhysicalBaseMode,
    PlanarVelocity,
    WalkingSenses,
)

from ...dataset.robots_on_legs import RecordingWalkingController, robot_on_a_leg_drive
from ...pytest_environment import runs_in_continuous_integration

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.adapters.multi_sim import (
    FloatingBaseJoint,
    MujocoBuilder,
    MujocoFloatingBase,
    MujocoSim,
    MujocoWalking,
)

needs_mujoco_in_ci = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% fixtures


@pytest.fixture
def drive_on_a_leg() -> OmniDrive:
    return robot_on_a_leg_drive()


@pytest.fixture
def walking_simulation(drive_on_a_leg):
    simulation = MujocoSim(world=drive_on_a_leg._world, headless=True)
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


@pytest.fixture
def walk_then_fix_simulation(drive_on_a_leg):
    simulation = MujocoSim(
        world=drive_on_a_leg._world,
        headless=True,
        walking_mode=PhysicalBaseMode.WALK_THEN_FIX,
    )
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


@pytest.fixture
def fixed_simulation(drive_on_a_leg):
    simulation = MujocoSim(
        world=drive_on_a_leg._world,
        headless=True,
        walking_mode=PhysicalBaseMode.FIXED,
    )
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


def _controller(simulation: MujocoSim) -> RecordingWalkingController:
    [walking] = simulation.walking
    return walking.controller


# %% the compiled model


def test_a_walking_base_floats_on_three_slides_and_a_ball(drive_on_a_leg, tmp_path):
    builder = MujocoBuilder()
    builder.build_world(world=drive_on_a_leg._world, file_path=str(tmp_path / "s.xml"))
    floating_base = MujocoFloatingBase(drive_on_a_leg, builder.element_names)
    [pelvis] = [body for body in builder.spec.bodies if body.name == "pelvis"]

    assert [(joint.name, joint.type) for joint in pelvis.joints] == [
        (floating_base.joint_name(FloatingBaseJoint.X), mujoco.mjtJoint.mjJNT_SLIDE),
        (floating_base.joint_name(FloatingBaseJoint.Y), mujoco.mjtJoint.mjJNT_SLIDE),
        (
            floating_base.joint_name(FloatingBaseJoint.HEIGHT),
            mujoco.mjtJoint.mjJNT_SLIDE,
        ),
        (
            floating_base.joint_name(FloatingBaseJoint.ORIENTATION),
            mujoco.mjtJoint.mjJNT_BALL,
        ),
    ]


# %% the walking controller in the physics


@needs_mujoco_in_ci
def test_the_driven_joint_starts_in_the_standing_posture(walking_simulation):
    world = walking_simulation.world
    controller = _controller(walking_simulation)

    assert world.state[world.get_connection_by_name("leg").raw_dof.id].position == (
        pytest.approx(controller.standing_posture["leg"])
    )


@needs_mujoco_in_ci
def test_the_controller_is_asked_once_every_control_period(walking_simulation):
    controller = _controller(walking_simulation)

    walking_simulation.step_simulation(timedelta(seconds=0.2))

    assert len(controller.calls) == round(0.2 / controller.control_period)


@needs_mujoco_in_ci
def test_the_leg_is_driven_to_the_controllers_answer(walking_simulation):
    world = walking_simulation.world
    controller = _controller(walking_simulation)

    walking_simulation.step_simulation(timedelta(seconds=1.0))

    assert world.state[world.get_connection_by_name("leg").raw_dof.id].position == (
        pytest.approx(controller.leg_target, abs=0.02)
    )


@needs_mujoco_in_ci
def test_the_controller_is_told_the_commanded_velocity(walking_simulation):
    world = walking_simulation.world
    [drive] = [walking.drive.connection for walking in walking_simulation.walking]
    world.state[drive.x_velocity.id].velocity = 0.4
    world.state[drive.y_velocity.id].velocity = -0.1
    world.state[drive.yaw.id].velocity = 0.2
    world.notify_state_change()

    walking_simulation.step_simulation(timedelta(milliseconds=1))

    _, command = _controller(walking_simulation).calls[-1]
    assert command.to_array() == pytest.approx([0.4, -0.1, 0.2])


@needs_mujoco_in_ci
def test_the_controller_senses_the_bases_turn_in_the_bases_own_axes(
    walking_simulation,
):
    """
    The base's box is turned within the base, so a turn measured in the box's principal
    axes would differ; the ball joint's velocity is the turn in the base's own axes.
    """
    simulator = walking_simulation.simulator
    model, data = simulator._mj_model, simulator._mj_data
    [walking] = walking_simulation.walking
    ball = model.joint(
        MujocoFloatingBase(
            walking.drive.connection, walking.drive.element_names
        ).joint_name(FloatingBaseJoint.ORIENTATION)
    )
    turn = np.array([0.3, -0.5, 0.7])
    data.qvel[ball.dofadr[0] : ball.dofadr[0] + 3] = turn
    data.qpos[ball.qposadr[0] : ball.qposadr[0] + 4] = [
        0.9,
        0.3,
        0.2,
        0.1,
    ] / np.linalg.norm([0.9, 0.3, 0.2, 0.1])

    walking_simulation.step_simulation(timedelta(milliseconds=1))

    senses, _ = _controller(walking_simulation).calls[-1]
    assert senses.angular_velocity == pytest.approx(turn)


@needs_mujoco_in_ci
def test_the_controller_senses_gravity_straight_down_while_the_base_is_upright(
    walking_simulation,
):
    walking_simulation.step_simulation(timedelta(milliseconds=1))

    senses, _ = _controller(walking_simulation).calls[-1]
    assert senses.gravity_direction == pytest.approx([0.0, 0.0, -1.0])


@needs_mujoco_in_ci
def test_the_bases_measured_pose_is_the_drives_odometry(walking_simulation):
    world = walking_simulation.world
    simulator = walking_simulation.simulator
    model, data = simulator._mj_model, simulator._mj_data
    [walking] = walking_simulation.walking
    drive = walking.drive.connection
    floating_base = MujocoFloatingBase(drive, walking.drive.element_names)
    yaw = 0.4
    data.qpos[model.joint(floating_base.joint_name(FloatingBaseJoint.X)).qposadr[0]] = (
        0.3
    )
    data.qpos[
        model.joint(floating_base.joint_name(FloatingBaseJoint.Y)).qposadr[0]
    ] = -0.2
    orientation = model.joint(
        floating_base.joint_name(FloatingBaseJoint.ORIENTATION)
    ).qposadr[0]
    data.qpos[orientation : orientation + 4] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    walking_simulation.step_simulation(timedelta(milliseconds=1))

    assert [
        world.state[degree_of_freedom.id].position
        for degree_of_freedom in (drive.x, drive.y, drive.yaw)
    ] == pytest.approx([0.3, -0.2, yaw], abs=1e-3)


@needs_mujoco_in_ci
def test_a_robot_counts_as_still_while_its_walking_legs_keep_stepping(
    walking_simulation,
):
    """
    A balanced robot never stops moving its legs, so waiting for the robot to stand
    still waits for every other joint only.
    """
    controller = _controller(walking_simulation)

    def stepping(senses: WalkingSenses, command: PlanarVelocity) -> dict[str, float]:
        if len(controller.calls) % 10 == 0:
            controller.leg_target = -controller.leg_target
        return RecordingWalkingController.leg_targets(controller, senses, command)

    controller.leg_targets = stepping
    pacer = SteppedSimulationPacer(
        walking_simulation, standstill_time_limit=timedelta(seconds=5)
    )
    pacer.target_frequency = 100
    started = walking_simulation.simulator.current_simulation_time

    pacer.wait_for_standstill()

    waited = walking_simulation.simulator.current_simulation_time - started
    assert waited < 1.0


# %% holding the base where it stands


def _walking(simulation: MujocoSim) -> MujocoWalking:
    [walking] = simulation.walking
    return walking


def _pelvis_position(simulation: MujocoSim) -> np.ndarray:
    """
    :return: Where the pelvis of the walking robot is in the physics now.
    """
    walking = _walking(simulation)
    data = simulation.simulator._mj_data
    return data.body(
        walking.drive.element_names.of(walking.drive.connection.child)
    ).xpos.copy()


def _command_walking(simulation: MujocoSim, forward: float) -> None:
    world = simulation.world
    drive = _walking(simulation).drive.connection
    world.state[drive.x_velocity.id].velocity = forward
    world.notify_state_change()


@needs_mujoco_in_ci
def test_a_walking_base_is_built_with_a_hold_that_starts_released(drive_on_a_leg):
    simulation = MujocoSim(world=drive_on_a_leg._world, headless=True)
    simulation.start_stepped_simulation()
    walking = _walking(simulation)
    model = simulation.simulator._mj_model
    data = simulation.simulator._mj_data
    hold_name = MujocoFloatingBase(
        drive_on_a_leg, walking.drive.element_names
    ).hold_name

    assert model.equality(hold_name).type == mujoco.mjtEq.mjEQ_WELD
    assert not data.eq_active[model.equality(hold_name).id]
    assert not walking.held
    simulation.stop_simulation()


@needs_mujoco_in_ci
def test_a_base_that_walks_then_stands_is_held_from_the_start(
    walk_then_fix_simulation,
):
    """
    Nothing stands under the robot, so a base that was not held would fall.
    """
    started_at = _pelvis_position(walk_then_fix_simulation)

    walk_then_fix_simulation.step_simulation(timedelta(seconds=0.5))

    assert _walking(walk_then_fix_simulation).held
    assert _pelvis_position(walk_then_fix_simulation) == pytest.approx(
        started_at, abs=1e-3
    )


@needs_mujoco_in_ci
def test_a_held_base_is_balanced_by_its_controller_until_held_and_then_not_asked(
    walk_then_fix_simulation,
):
    controller = _controller(walk_then_fix_simulation)
    asked_until_held = len(controller.calls)

    walk_then_fix_simulation.step_simulation(timedelta(seconds=0.5))

    assert asked_until_held > 0
    assert len(controller.calls) == asked_until_held


@needs_mujoco_in_ci
def test_a_released_base_goes_on_from_what_its_controller_remembered(
    walk_then_fix_simulation,
):
    controller = _controller(walk_then_fix_simulation)
    remembered = list(controller.calls)

    walking = _walking(walk_then_fix_simulation)
    walking.release()
    walk_then_fix_simulation.step_simulation(
        timedelta(seconds=walking.landing_duration + 0.01)
    )

    assert controller.calls[: len(remembered)] == remembered
    assert len(controller.calls) > len(remembered)


@needs_mujoco_in_ci
def test_a_held_base_is_released_once_commanded_to_walk(walk_then_fix_simulation):
    controller = _controller(walk_then_fix_simulation)
    command = _walking(walk_then_fix_simulation).walking_share * (
        controller.pace.slowest_speed
    )
    _command_walking(walk_then_fix_simulation, command)

    walk_then_fix_simulation.step_simulation(
        timedelta(seconds=_walking(walk_then_fix_simulation).landing_duration + 0.1)
    )

    assert not _walking(walk_then_fix_simulation).held
    assert controller.calls[-1][1].forward == pytest.approx(command)


@needs_mujoco_in_ci
def test_a_held_base_stays_held_while_commanded_slower_than_it_walks(
    walk_then_fix_simulation,
):
    controller = _controller(walk_then_fix_simulation)
    _command_walking(
        walk_then_fix_simulation,
        _walking(walk_then_fix_simulation).walking_share
        * controller.pace.slowest_speed
        / 2,
    )

    walk_then_fix_simulation.step_simulation(timedelta(seconds=0.1))

    assert _walking(walk_then_fix_simulation).held


@needs_mujoco_in_ci
def test_a_released_base_stands_while_commanded_slower_than_it_walks(
    walk_then_fix_simulation,
):
    controller = _controller(walk_then_fix_simulation)
    _command_walking(walk_then_fix_simulation, controller.pace.slowest_speed)
    walk_then_fix_simulation.step_simulation(
        timedelta(seconds=_walking(walk_then_fix_simulation).landing_duration + 0.05)
    )
    _command_walking(
        walk_then_fix_simulation,
        _walking(walk_then_fix_simulation).walking_share
        * controller.pace.slowest_speed
        / 2,
    )

    walk_then_fix_simulation.step_simulation(timedelta(seconds=0.05))

    assert controller.calls[-1][1].to_array() == pytest.approx([0.0, 0.0, 0.0])


@needs_mujoco_in_ci
def test_a_held_base_lends_its_height_to_the_world(walk_then_fix_simulation):
    """
    The drive carries no height, so the world takes the height the base was held at from
    the physics.
    """
    controller = _controller(walk_then_fix_simulation)
    walking = _walking(walk_then_fix_simulation)
    _command_walking(walk_then_fix_simulation, controller.pace.slowest_speed)
    walk_then_fix_simulation.step_simulation(timedelta(seconds=0.2))
    _command_walking(walk_then_fix_simulation, 0.0)

    walking.hold()

    pelvis = walking.drive.connection.child
    assert pelvis.global_pose.to_np()[:3, 3] == pytest.approx(
        _pelvis_position(walk_then_fix_simulation), abs=1e-4
    )


@needs_mujoco_in_ci
def test_a_released_base_is_held_once_it_stood_still_long_enough(
    walk_then_fix_simulation,
):
    """
    Without gravity, the released base floats still on the spot.
    """
    walk_then_fix_simulation.simulator._mj_model.opt.gravity[:] = 0.0
    controller = _controller(walk_then_fix_simulation)
    controller.leg_target = controller.standing_posture["leg"]
    walking = _walking(walk_then_fix_simulation)
    _command_walking(walk_then_fix_simulation, controller.pace.slowest_speed)
    walk_then_fix_simulation.step_simulation(timedelta(milliseconds=20))
    _command_walking(walk_then_fix_simulation, 0.0)
    data = walk_then_fix_simulation.simulator._mj_data
    data.qvel[:] = 0.0
    held_at = []
    while not walking.hold_when_still():
        walk_then_fix_simulation.step_simulation(timedelta(milliseconds=10))
        held_at.append(data.time)

    assert walking.held
    assert len(held_at) == pytest.approx(walking.standstill_duration / 0.01, abs=2)


@needs_mujoco_in_ci
def test_a_fixed_base_is_held_where_it_starts_and_never_walks(fixed_simulation):
    controller = _controller(fixed_simulation)
    started_at = _pelvis_position(fixed_simulation)
    _command_walking(fixed_simulation, controller.pace.fastest_speed)

    fixed_simulation.step_simulation(timedelta(seconds=0.5))

    assert _walking(fixed_simulation).held
    assert controller.calls == []
    assert _pelvis_position(fixed_simulation) == pytest.approx(started_at, abs=1e-3)


@needs_mujoco_in_ci
def test_the_legs_of_a_fixed_base_follow_the_world(fixed_simulation):
    world = fixed_simulation.world
    leg = world.get_connection_by_name("leg").raw_dof
    world.state[leg.id].position = -0.4
    world.notify_state_change()

    fixed_simulation.step_simulation(timedelta(seconds=1.0))

    assert fixed_simulation.walking_actuator_names == set()
    assert world.state[leg.id].position == pytest.approx(-0.4, abs=0.02)


@needs_mujoco_in_ci
def test_a_base_is_held_lifted_clear_of_the_floor_it_stood_on(drive_on_a_leg):
    world = drive_on_a_leg._world
    with world.modify_world():
        floor = Body(
            name=PrefixedName("floor"),
            collision=ShapeCollection([Box(scale=Scale(4.0, 4.0, 0.1))]),
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=floor,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.61, reference_frame=world.root
                ),
            )
        )
    simulation = MujocoSim(
        world=world, headless=True, walking_mode=PhysicalBaseMode.WALK_THEN_FIX
    )
    simulation.start_stepped_simulation()
    model, data = simulation.simulator._mj_model, simulation.simulator._mj_data
    floor_clearance = _walking(simulation).floor_clearance

    def touches_within(margin: float) -> bool:
        # a contact counts within the margins of both its geoms
        model.geom_margin[:] = margin / 2
        mujoco.mj_forward(model, data)
        return data.ncon > 0

    assert _walking(simulation).held
    assert (
        touches_within(0.9 * floor_clearance),
        touches_within(1.1 * floor_clearance),
    ) == (
        False,
        True,
    )
    simulation.stop_simulation()


@needs_mujoco_in_ci
def test_a_released_base_stands_at_the_height_it_was_built_at_in_the_world(
    walk_then_fix_simulation,
):
    walking = _walking(walk_then_fix_simulation)
    pelvis = walking.drive.connection.child
    built_height = walking.localization_when_built[2, 3]

    walking.release()

    assert pelvis.global_pose.to_np()[2, 3] == pytest.approx(built_height)


@needs_mujoco_in_ci
def test_a_released_base_is_let_go_only_once_lowered_back_onto_its_legs(
    walk_then_fix_simulation,
):
    walking = _walking(walk_then_fix_simulation)
    model = walk_then_fix_simulation.simulator._mj_model
    data = walk_then_fix_simulation.simulator._mj_data
    weld = model.equality(walking.floating_base.hold_name).id
    walking.release()

    walk_then_fix_simulation.step_simulation(
        timedelta(seconds=walking.landing_duration / 2)
    )
    welded_while_landing = bool(data.eq_active[weld])
    walk_then_fix_simulation.step_simulation(
        timedelta(seconds=walking.landing_duration)
    )

    assert (welded_while_landing, bool(data.eq_active[weld])) == (True, False)
