from __future__ import annotations

from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_statechart.monitors.sensor_monitors import PassageClear
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ...semantic_digital_twin_test.test_sensors.test_lidar import (
    FAR_WALL_DISTANCE,
    forward_beam_pattern,
    simulated_lidar,
    world_with_walls,
)

# %% helpers

PASSAGE_DISTANCE = 1.0
"""
Distance from the lidar to the passage it looks through.
"""


def passage_monitor(mount: Body, passage_x: float) -> PassageClear:
    """
    :return: A monitor watching, through a forward-looking lidar on the given body, a
        passage that crosses the lidar's x axis at the given distance.
    """
    return PassageClear(
        lidar=simulated_lidar(mount, forward_beam_pattern()),
        passage_start=Point3(passage_x, -0.5, 0, reference_frame=mount),
        passage_end=Point3(passage_x, 0.5, 0, reference_frame=mount),
    )


def ticked_executor(world: World, monitor: PassageClear) -> StatechartExecutor:
    """
    :return: An executor that compiled a statechart holding only the monitor, ticked once.
    """
    executor = StatechartExecutor(context=StatechartContext(world=world))
    statechart = Statechart(context=executor.context)
    statechart.add_node(monitor)
    executor.compile(statechart)
    executor.tick()
    return executor


# %% passage clear


def test_a_passage_with_nothing_in_it_is_clear():
    world, mount = world_with_walls(FAR_WALL_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)

    ticked_executor(world, monitor)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_a_passage_with_a_wall_in_it_is_not_clear():
    world, mount = world_with_walls(PASSAGE_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)

    ticked_executor(world, monitor)

    assert monitor.observation_state == ObservationStateValues.FALSE


def test_a_passage_no_beam_crosses_is_unknown():
    world, mount = world_with_walls(FAR_WALL_DISTANCE)
    monitor = passage_monitor(mount, -PASSAGE_DISTANCE)

    ticked_executor(world, monitor)

    assert monitor.observation_state == ObservationStateValues.UNKNOWN


def test_a_passage_seen_clear_once_stays_clear_when_asked_to():
    world, mount = world_with_walls(2 * PASSAGE_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)
    monitor.stays_clear_once_seen = True
    executor = ticked_executor(world, monitor)
    monitor.passage_start.x = 2 * PASSAGE_DISTANCE
    monitor.passage_end.x = 2 * PASSAGE_DISTANCE

    executor.tick()

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_a_passage_seen_clear_once_is_judged_anew_by_default():
    world, mount = world_with_walls(2 * PASSAGE_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)
    executor = ticked_executor(world, monitor)
    monitor.passage_start.x = 2 * PASSAGE_DISTANCE
    monitor.passage_end.x = 2 * PASSAGE_DISTANCE

    executor.tick()

    assert monitor.observation_state == ObservationStateValues.FALSE


def test_a_restarted_passage_monitor_judges_the_passage_anew():
    world, mount = world_with_walls(2 * PASSAGE_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)
    monitor.stays_clear_once_seen = True
    executor = ticked_executor(world, monitor)
    monitor.passage_start.x = 2 * PASSAGE_DISTANCE
    monitor.passage_end.x = 2 * PASSAGE_DISTANCE

    monitor.on_start()
    executor.tick()

    assert monitor.observation_state == ObservationStateValues.FALSE


def test_a_clear_passage_never_seen_blocked_is_not_clear_when_asked_to():
    world, mount = world_with_walls(FAR_WALL_DISTANCE)
    monitor = passage_monitor(mount, PASSAGE_DISTANCE)
    monitor.clear_only_after_blocked = True

    ticked_executor(world, monitor)

    assert monitor.observation_state == ObservationStateValues.FALSE


def test_a_passage_seen_blocked_and_then_clear_is_clear_when_asked_to():
    world, mount = world_with_walls(2 * PASSAGE_DISTANCE)
    monitor = passage_monitor(mount, 2 * PASSAGE_DISTANCE)
    monitor.clear_only_after_blocked = True
    executor = ticked_executor(world, monitor)
    monitor.passage_start.x = PASSAGE_DISTANCE
    monitor.passage_end.x = PASSAGE_DISTANCE

    executor.tick()

    assert monitor.observation_state == ObservationStateValues.TRUE
