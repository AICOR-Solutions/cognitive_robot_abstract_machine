from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.adapters.sensors.lidar import Lidar
from semantic_digital_twin.spatial_types.spatial_types import Point3

# %% lidar monitors


@dataclass(eq=False, repr=False)
class PassageClear(MotionStatechartNode):
    """
    Observes True while a lidar sees nothing blocking a passage, a line segment in its
    scan plane, False while something stands in it, and Unknown while none of its beams
    cross the passage.
    """

    lidar: Lidar = field(kw_only=True)
    """
    The lidar that looks through the passage.
    """

    passage_start: Point3 = field(kw_only=True)
    """
    One side of the passage.
    """

    passage_end: Point3 = field(kw_only=True)
    """
    The other side of the passage.
    """

    tolerance: float = field(default=0.05, kw_only=True)
    """
    How far, in meters, a beam has to travel past the passage for it to count as having
    passed through.
    """

    stays_clear_once_seen: bool = field(default=False, kw_only=True)
    """
    Whether the passage keeps counting as clear once it was seen clear, such as for a
    robot driving into it, whose lidar no longer looks through the passage once inside.
    """

    seen_clear: bool = field(default=False, init=False)
    """
    Whether the passage was seen clear since this monitor started.
    """

    def on_start(self, context: MotionStatechartContext) -> None:
        self.seen_clear = False

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self.stays_clear_once_seen and self.seen_clear:
            return ObservationStateValues.TRUE
        reading = self.lidar.get_lidar_reading()
        passage_distances = reading.distances_to_segment(
            self.passage_start, self.passage_end
        )
        crossing_beams = np.isfinite(passage_distances)
        if not crossing_beams.any():
            return ObservationStateValues.UNKNOWN
        if np.all(
            reading.ranges[crossing_beams]
            > passage_distances[crossing_beams] + self.tolerance
        ):
            self.seen_clear = True
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
