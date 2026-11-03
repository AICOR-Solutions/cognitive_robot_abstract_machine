import json
import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
from typing_extensions import Any, Callable, Iterator

import pytest

from cramph.executor import NoPacing
from semantic_digital_twin.adapters.ros import rospy
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from giskardpy.middleware.ros2.command_publishing import CommandPublisher
from giskardpy.middleware.ros2.client_presence import ClientWatchdog, HeartbeatPresence
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.exceptions import (
    GiskardWorldUpdateNotReceivedError,
    WorldModelModifiedDuringMotionError,
)
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from semantic_digital_twin.input_synchronization import WorldStateInputs
from giskardpy.middleware.ros2.world_updates import (
    ClientWorldUpdates,
    IncomingWorldUpdates,
    OutgoingWorldUpdates,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.monitors import CountSimulationTimeSeconds
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

# %% mimics of the synchronizers


@dataclass
class BufferingSynchronizerMimic:
    """Stands in for a synchronizer that buffers what it receives."""

    buffered_model_modification: bool = False
    """Whether a model modification is waiting in the buffer."""

    applied_state_update_batches: int = 0
    """How often the buffer was drained up to the next model modification."""

    applied_message_batches: int = 0
    """How often the whole buffer was applied."""

    caught_up: bool = False
    """The answer this mimic gives about every position it is asked about."""

    asked_about_positions: list[StreamPosition] = field(default_factory=list)
    """Every position this mimic was asked about."""

    @property
    def has_buffered_model_modification(self) -> bool:
        return self.buffered_model_modification

    def apply_missed_state_updates(self) -> None:
        self.applied_state_update_batches += 1

    def apply_missed_messages(self) -> None:
        self.applied_message_batches += 1
        self.buffered_model_modification = False

    def has_applied(self, position: StreamPosition) -> bool:
        self.asked_about_positions.append(position)
        return self.caught_up


@dataclass
class PublishingSynchronizerMimic:
    """Stands in for the synchronizer of a client: it publishes the changes of
    that world and reports how far it caught up with another publisher."""

    published_sequence_number: int = 0
    """Position of the change this world published last."""

    origin: MetaData = field(
        default_factory=lambda: MetaData(node_name="client", process_id=0)
    )
    """The publisher the positions of this world belong to."""

    applied_sequence_number: int = 0
    """The position this world caught up with, for every publisher it is asked
    about."""

    @property
    def latest_published_position(self) -> StreamPosition:
        return StreamPosition(
            origin=self.origin, sequence_number=self.published_sequence_number
        )

    def has_applied(self, position: StreamPosition) -> bool:
        return self.applied_sequence_number >= position.sequence_number


@dataclass
class ReloadingSynchronizerMimic:
    """Stands in for a synchronizer that remembers a request to replace the
    whole world."""

    pending_reload: bool = False
    """Whether a reload is waiting to be applied."""

    applied_reloads: int = 0
    """How often a pending reload was applied."""

    @property
    def has_pending_reload(self) -> bool:
        return self.pending_reload

    def apply_pending_reload(self) -> None:
        if not self.pending_reload:
            return
        self.applied_reloads += 1
        self.pending_reload = False


# %% what may be applied and when


class TestPendingModelChange:
    """Both a modification of the model and a request to replace it are changes
    of the structure of the world, so a motion compiled against that structure
    has to end before either can be applied."""

    def test_a_buffered_modification_is_a_pending_model_modification(self):
        world_updates = IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic(
                buffered_model_modification=True
            )
        )

        assert world_updates.has_pending_model_modification
        assert not world_updates.has_pending_reload

    def test_a_pending_reload_is_no_pending_model_modification(self):
        world_updates = IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic(),
            model_reload_synchronizer=ReloadingSynchronizerMimic(pending_reload=True),
        )

        assert world_updates.has_pending_reload
        assert not world_updates.has_pending_model_modification

    def test_no_reload_is_pending_without_a_reload_synchronizer(self):
        world_updates = IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic()
        )

        assert not world_updates.has_pending_reload

    def test_applying_the_model_modifications_leaves_a_pending_reload_waiting(self):
        world_synchronizer = BufferingSynchronizerMimic(
            buffered_model_modification=True
        )
        reload_synchronizer = ReloadingSynchronizerMimic(pending_reload=True)
        world_updates = IncomingWorldUpdates(
            world_synchronizer=world_synchronizer,
            model_reload_synchronizer=reload_synchronizer,
        )

        world_updates.apply_model_modifications()

        assert world_synchronizer.applied_message_batches == 1
        assert reload_synchronizer.applied_reloads == 0

    def test_applying_everything_also_applies_a_pending_reload(self):
        reload_synchronizer = ReloadingSynchronizerMimic(pending_reload=True)
        world_updates = IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic(),
            model_reload_synchronizer=reload_synchronizer,
        )

        world_updates.apply_all()

        assert reload_synchronizer.applied_reloads == 1
        assert not world_updates.has_pending_reload

    def test_applying_everything_works_without_a_reload_synchronizer(self):
        world_synchronizer = BufferingSynchronizerMimic(
            buffered_model_modification=True
        )
        world_updates = IncomingWorldUpdates(world_synchronizer=world_synchronizer)

        world_updates.apply_all()

        assert world_synchronizer.applied_message_batches == 1


# %% catching up with another process


class TestCatchingUp:
    """Whoever waits for a change of another process asks the world updates, so
    it has to pass the question on to the synchronizer that received it."""

    def test_the_position_to_catch_up_with_reaches_the_synchronizer(self):
        world_synchronizer = BufferingSynchronizerMimic(caught_up=True)
        world_updates = IncomingWorldUpdates(world_synchronizer=world_synchronizer)
        position = StreamPosition(
            origin=MetaData(node_name="publisher", process_id=1), sequence_number=7
        )

        assert world_updates.has_applied(position)
        assert world_synchronizer.asked_about_positions == [position]

    def test_a_position_that_was_not_reached_is_reported_as_such(self):
        world_updates = IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic(caught_up=False)
        )

        assert not world_updates.has_applied(
            StreamPosition(
                origin=MetaData(node_name="publisher", process_id=1), sequence_number=7
            )
        )


# %% the world of a client around a goal


class TestClientWorldUpdates:
    """A goal is built against a world the client may just have changed, and
    running it changes that world again, so both sides are told what to catch
    up with."""

    def test_a_world_that_published_nothing_requires_nothing(self):
        world_updates = ClientWorldUpdates(
            world_synchronizer=PublishingSynchronizerMimic()
        )

        assert world_updates.required_position() is None

    def test_a_goal_requires_the_last_change_of_this_world(self):
        synchronizer = PublishingSynchronizerMimic(published_sequence_number=4)
        world_updates = ClientWorldUpdates(world_synchronizer=synchronizer)

        assert world_updates.required_position() == StreamPosition(
            origin=synchronizer.origin, sequence_number=4
        )

    def test_a_result_without_changes_is_not_waited_for(self):
        world_updates = ClientWorldUpdates(
            world_synchronizer=PublishingSynchronizerMimic(), timeout=0.0
        )

        world_updates.wait_for_the_changes_of_a_goal({})

    def test_the_changes_of_a_goal_are_waited_for(self):
        synchronizer = PublishingSynchronizerMimic(applied_sequence_number=9)
        world_updates = ClientWorldUpdates(
            world_synchronizer=synchronizer, timeout=0.0, poll_interval=0.0
        )

        world_updates.wait_for_the_changes_of_a_goal(
            {
                "published_position": to_json(
                    StreamPosition(
                        origin=MetaData(node_name="giskard", process_id=1),
                        sequence_number=9,
                    )
                )
            }
        )

    def test_changes_that_never_arrive_are_reported(self):
        world_updates = ClientWorldUpdates(
            world_synchronizer=PublishingSynchronizerMimic(applied_sequence_number=8),
            timeout=0.05,
            poll_interval=0.01,
        )

        with pytest.raises(GiskardWorldUpdateNotReceivedError) as raised:
            world_updates.wait_for_the_changes_of_a_goal(
                {
                    "published_position": to_json(
                        StreamPosition(
                            origin=MetaData(node_name="giskard", process_id=1),
                            sequence_number=9,
                        )
                    )
                }
            )

        assert raised.value.awaited_sequence_number == 9


# %% against a real synchronizer


@dataclass
class GoalQueueStub:
    """Stands in for the action server of a control loop that is never
    canceled."""

    action_name: str = "world_updates"
    """Name reported in cancel exceptions."""

    goal_id: int = 0
    """Id of the running goal."""

    feedback_messages: list[Any] = field(default_factory=list)
    """Every feedback message that was published."""

    def is_cancel_requested(self) -> bool:
        return False

    def send_feedback(self, message: Any) -> None:
        self.feedback_messages.append(message)


@dataclass
class ControlLoopFixture:
    """A control loop running on a world that a second process publishes
    updates to."""

    control_loop: ControlLoop
    controlled_world: World
    controlled_synchronizer: WorldSynchronizer
    remote_world: World
    remote_synchronizer: WorldSynchronizer
    world_updates: IncomingWorldUpdates

    @property
    def moving_connection(self) -> PrismaticConnection:
        """The connection whose position both worlds share."""
        return self.controlled_world.get_connections_by_type(PrismaticConnection)[0]

    def wait_until_buffered(self, count: int) -> bool:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if len(self.controlled_synchronizer.missed_messages) >= count:
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self.controlled_synchronizer.close()
        self.remote_synchronizer.close()


def add_moving_connection(world: World) -> None:
    """Add a single prismatic connection between two bodies to the given world.

    The degree of freedom is limited because the compiled motion
    statechart needs velocity limits to decide whether the world has
    settled.
    """
    with world.modify_world():
        parent_body = Body(name=PrefixedName("parent"))
        child_body = Body(name=PrefixedName("child"))
        world.add_body(parent_body)
        world.add_body(child_body)
        world.add_connection(
            PrismaticConnection.create_with_dofs(
                world=world,
                parent=parent_body,
                child=child_body,
                axis=Vector3.X(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=-1.0, upper_position=1.0, maximum_speed=1.0
                ),
            )
        )


def control_loop_running(
    build_motion: Callable[[Statechart, World], None],
) -> ControlLoopFixture:
    """
    :param build_motion: Adds the motion the control loop runs to a statechart, given
        the world it runs on.
    :return: A control loop running that motion on a world a second process updates.
    """
    controlled_world = World(name="controlled")
    remote_world = World(name="remote")
    controlled_synchronizer = WorldSynchronizer(
        node=rospy.get_node(), _world=controlled_world
    )
    remote_synchronizer = WorldSynchronizer(node=rospy.get_node(), _world=remote_world)
    time.sleep(0.3)

    # Build the structure on one side and let it propagate, so both worlds refer to the
    # same entities; the delta protocol cannot bridge two independently built worlds.
    add_moving_connection(remote_world)
    deadline = time.time() + 5.0
    while (
        len(controlled_world.kinematic_structure_entities) < 2
        and time.time() < deadline
    ):
        time.sleep(0.02)
    assert len(controlled_world.kinematic_structure_entities) == 2
    controlled_synchronizer.defer_incoming_updates = True

    executor = StatechartExecutor(
        context=StatechartContext(world=controlled_world),
        pacer=NoPacing(),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )
    motion_statechart = Statechart(context=executor.context)
    build_motion(motion_statechart, controlled_world)
    executor.compile(motion_statechart)
    action_server = GoalQueueStub()
    world_updates = IncomingWorldUpdates(world_synchronizer=controlled_synchronizer)
    return ControlLoopFixture(
        control_loop=ControlLoop(
            executor=executor,
            action_server=action_server,
            client_watchdog=ClientWatchdog(
                presence=HeartbeatPresence(node=rospy.get_node())
            ),
            feedback_publisher=ActionFeedbackPublisher(
                executor=executor,
                action_server=action_server,
                outgoing_world_updates=OutgoingWorldUpdates(
                    world_synchronizer=controlled_synchronizer
                ),
            ),
            inputs=WorldStateInputs(world=controlled_world),
            cycle_counter=CycleCounter(),
            world_updates=world_updates,
        ),
        controlled_world=controlled_world,
        controlled_synchronizer=controlled_synchronizer,
        remote_world=remote_world,
        remote_synchronizer=remote_synchronizer,
        world_updates=world_updates,
    )


def waiting_for_a_long_time(statechart: Statechart, world: World) -> None:
    """Adds a motion that moves nothing and does not end by itself."""
    statechart.add_node(counter := CountSimulationTimeSeconds(seconds=1000.0))
    statechart.add_node(EndMotion.when_true(counter))


class DrivenConnection(float, Enum):
    """Where :func:`driving_the_connection` moves the shared connection."""

    GOAL_POSITION = 0.9
    """Far enough for the motion to still run after a few control cycles."""

def driving_the_connection(statechart: Statechart, world: World) -> None:
    """Adds a motion that moves the shared connection to its goal position and
    then ends."""
    connection = world.get_connections_by_type(PrismaticConnection)[0]
    statechart.add_node(
        joints := JointPositionList(
            goal_state=JointState.from_mapping(
                {connection: DrivenConnection.GOAL_POSITION}
            )
        )
    )
    statechart.add_node(EndMotion.when_true(joints))


@pytest.fixture()
def control_loop(init_rospy) -> Iterator[ControlLoopFixture]:
    fixture = control_loop_running(waiting_for_a_long_time)
    yield fixture
    fixture.close()


@pytest.fixture()
def driving_control_loop(init_rospy) -> Iterator[ControlLoopFixture]:
    fixture = control_loop_running(driving_the_connection)
    yield fixture
    fixture.close()


@dataclass(eq=False)
class SpeedWhenTheModelChanged(ModelChangeCallback):
    """Records how fast a degree of freedom moved whenever the world model
    changed."""

    degree_of_freedom_id: UUID = field(kw_only=True)
    """The degree of freedom whose speed is recorded."""

    speeds: list[float] = field(default_factory=list, init=False)
    """The recorded speeds, in order."""

    def on_model_change(self, **kwargs) -> None:
        self.speeds.append(abs(self._world.state[self.degree_of_freedom_id].velocity))


def add_body_on_six_degrees_of_freedom(world: World, name: str) -> Connection6DoF:
    """
    :return: The connection of a new body that can be moved anywhere, the way a tracked
        person is modelled.
    """
    with world.modify_world():
        body = Body(name=PrefixedName(name))
        world.add_kinematic_structure_entity(body)
        connection = Connection6DoF.create_with_dofs(
            world=world, parent=world.root, child=body
        )
        world.add_connection(connection)
    return connection


def run_cycles_until(
    control_loop: ControlLoop, reached: Callable[[], bool], timeout: float = 10.0
) -> None:
    """Run control cycles until ``reached`` holds, leaving updates of other
    processes time to arrive.

    :raises AssertionError: If it does not hold within ``timeout``
        seconds.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if reached():
            return
        control_loop.run_cycle()
    assert reached(), f"not reached within {timeout} s"


class TestRealWorldUpdatesDuringAMotion:
    """Against a real synchronizer: the world a motion was compiled against
    only changes where the control loop allows it to."""

    def test_a_state_update_reaches_the_running_motion(
        self, control_loop: ControlLoopFixture
    ):
        remote_connection = control_loop.remote_world.get_connections_by_type(
            PrismaticConnection
        )[0]
        control_loop.remote_world.state[remote_connection.dof.id].position = 0.42
        control_loop.remote_world.notify_state_change()
        assert control_loop.wait_until_buffered(1)

        control_loop.control_loop.run_cycle()

        assert control_loop.moving_connection.position == pytest.approx(
            0.42, abs=1e-9
        ), "the state of the other process never reached the running motion"

    def test_a_world_that_is_being_modified_terminates_the_motion(
        self, control_loop: ControlLoopFixture
    ):
        """An open modification is a model change that already started, so the
        motion has to end for the same reason as for one that is still waiting
        in the buffer."""
        with control_loop.controlled_world.modify_world():
            with pytest.raises(WorldModelModifiedDuringMotionError):
                control_loop.control_loop.run_cycle()

    def test_a_model_modification_reaches_the_running_motion(
        self, control_loop: ControlLoopFixture
    ):
        with control_loop.remote_world.modify_world():
            late_body = Body(name=PrefixedName("late_body"))
            control_loop.remote_world.add_body(late_body)
            control_loop.remote_world.add_connection(
                FixedConnection(parent=control_loop.remote_world.root, child=late_body)
            )
        assert control_loop.wait_until_buffered(1)
        entities_before = len(
            control_loop.controlled_world.kinematic_structure_entities
        )

        control_loop.control_loop.run_cycle()

        assert (
            len(control_loop.controlled_world.kinematic_structure_entities)
            == entities_before + 1
        )
        assert not control_loop.control_loop.executor.statechart.is_ended()

    def test_a_person_can_come_move_and_leave_during_the_motion(
        self, control_loop: ControlLoopFixture
    ):
        """A person tracked by another process appears, moves and disappears
        while the motion runs, which changes the structure of the world
        twice."""
        controlled_world = control_loop.controlled_world
        entities_before = len(controlled_world.kinematic_structure_entities)
        person_connection = add_body_on_six_degrees_of_freedom(
            control_loop.remote_world, "person"
        )
        assert control_loop.wait_until_buffered(1)
        control_loop.control_loop.run_cycle()
        person_in_motion = controlled_world.get_kinematic_structure_entity_by_name(
            "person"
        )

        person_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5, reference_frame=control_loop.remote_world.root
        )
        control_loop.remote_world.notify_state_change()
        run_cycles_until(
            control_loop.control_loop,
            lambda: float(person_in_motion.global_pose.x) == pytest.approx(1.5),
        )

        with control_loop.remote_world.modify_world():
            control_loop.remote_world.remove_kinematic_structure_entity(
                person_connection.child
            )
        run_cycles_until(
            control_loop.control_loop,
            lambda: (
                len(controlled_world.kinematic_structure_entities) == entities_before
            ),
        )

        assert not control_loop.control_loop.executor.statechart.is_ended()

    def test_a_moving_robot_comes_to_rest_before_a_model_modification_applies(
        self, driving_control_loop: ControlLoopFixture
    ):
        """The controller was compiled against the old structure, so it stops
        following the motion and brings the robot to rest before anything
        changes underneath it."""
        fixture = driving_control_loop
        speed_at_rest = fixture.control_loop.executor.extensions[0].speed_at_rest
        dof_id = fixture.moving_connection.dof.id
        run_cycles_until(
            fixture.control_loop,
            lambda: (
                abs(fixture.controlled_world.state[dof_id].velocity) > speed_at_rest
            ),
        )
        speeds = SpeedWhenTheModelChanged(
            _world=fixture.controlled_world, degree_of_freedom_id=dof_id
        )
        add_body_on_six_degrees_of_freedom(fixture.remote_world, "person")
        assert fixture.wait_until_buffered(1)

        run_cycles_until(fixture.control_loop, lambda: bool(speeds.speeds))

        assert speeds.speeds[0] <= speed_at_rest

    def test_the_motion_goes_on_after_a_model_modification(
        self, driving_control_loop: ControlLoopFixture
    ):
        fixture = driving_control_loop
        run_cycles_until(
            fixture.control_loop,
            lambda: (
                fixture.moving_connection.position > DrivenConnection.GOAL_POSITION / 3
            ),
        )
        add_body_on_six_degrees_of_freedom(fixture.remote_world, "person")
        assert fixture.wait_until_buffered(1)

        fixture.control_loop.run()

        assert fixture.moving_connection.position == pytest.approx(
            DrivenConnection.GOAL_POSITION.value, abs=0.01
        )

    def test_a_reload_of_the_world_terminates_the_motion_before_it_is_applied(
        self, control_loop: ControlLoopFixture
    ):
        """A reload replaces every entity the motion refers to, which compiling
        again cannot follow."""
        reload_synchronizer = ReloadingSynchronizerMimic(pending_reload=True)
        control_loop.world_updates.model_reload_synchronizer = reload_synchronizer

        with pytest.raises(WorldModelModifiedDuringMotionError):
            control_loop.control_loop.run_cycle()

        assert reload_synchronizer.applied_reloads == 0


# %% a model change the motion makes itself


@dataclass
class CommandPublisherRecordingCalls(CommandPublisher):
    """Stands in for a publisher to the robot, recording which of its methods
    ran, in order."""

    calls: list[str] = field(default_factory=list)
    """The methods that ran, each by its name."""

    def publish(self) -> None:
        self.calls.append(self.publish.__name__)

    def stop(self) -> None:
        self.calls.append(self.stop.__name__)


@pytest.fixture()
def local_control_loop(init_rospy) -> ControlLoop:
    """
    :return: A control loop running a long motion on a world no other process updates.
    """
    world = World(name="local")
    add_moving_connection(world)
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        pacer=NoPacing(),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )
    motion_statechart = Statechart(context=executor.context)
    motion_statechart.add_node(counter := CountSimulationTimeSeconds(seconds=1000.0))
    motion_statechart.add_node(EndMotion.when_true(counter))
    executor.compile(motion_statechart)
    action_server = GoalQueueStub()
    return ControlLoop(
        executor=executor,
        action_server=action_server,
        client_watchdog=ClientWatchdog(
            presence=HeartbeatPresence(node=rospy.get_node())
        ),
        feedback_publisher=ActionFeedbackPublisher(
            executor=executor,
            action_server=action_server,
            outgoing_world_updates=OutgoingWorldUpdates(
                world_synchronizer=PublishingSynchronizerMimic()
            ),
        ),
        inputs=WorldStateInputs(world=world),
        cycle_counter=CycleCounter(),
        world_updates=IncomingWorldUpdates(
            world_synchronizer=BufferingSynchronizerMimic()
        ),
    )


class TestModelChangesOfTheMotionItself:
    def test_commands_keep_being_published_while_the_motion_is_built_again(
        self, local_control_loop: ControlLoop
    ) -> None:
        """The controller slows the robot down while the motion statechart is
        built again after the motion changed the model, so the command
        publishers are not stopped and a command is published in that cycle
        too."""
        publisher = CommandPublisherRecordingCalls()
        local_control_loop.command_publishers.append(publisher)
        world = local_control_loop.world
        with world.modify_world():
            carried_body = Body(name=PrefixedName("carried_body"))
            world.add_body(carried_body)
            world.add_connection(FixedConnection(parent=world.root, child=carried_body))

        local_control_loop.run_cycle()

        assert publisher.calls == [CommandPublisherRecordingCalls.publish.__name__]
