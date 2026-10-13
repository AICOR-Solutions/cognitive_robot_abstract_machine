from __future__ import annotations

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.sensor_monitors import PassageClear
from semantic_digital_twin.spatial_types.spatial_types import Point3
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


def ticked_executor(world: World, monitor: PassageClear) -> Executor:
    """
    :return: An executor that compiled a statechart holding only the monitor, ticked once.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(monitor)
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=motion_statechart)
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
