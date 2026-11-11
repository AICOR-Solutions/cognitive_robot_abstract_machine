from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cramph.data_types import ObservationStateValues
from cramph.node import EndedByOwner, StatechartNode
from semantic_digital_twin.adapters.sensors.lidar import Lidar
from krrood_spatial_math.spatial_types import Point3

# %% lidar monitors


@dataclass(eq=False, repr=False, kw_only=True)
class PassageClear(EndedByOwner, StatechartNode):
    """
    Observes True while a lidar sees nothing blocking a passage, a line segment in its
    scan plane, False while something stands in it, and Unknown while none of its beams
    cross the passage.

    With :attr:`clear_only_after_blocked`, it also observes False while the passage is
    clear but was not seen blocked yet.
    """

    lidar: Lidar
    """
    The lidar that looks through the passage.
    """

    passage_start: Point3
    """
    One side of the passage.
    """

    passage_end: Point3
    """
    The other side of the passage.
    """

    tolerance: float = 0.05
    """
    How far, in meters, a beam has to travel past the passage for it to count as having
    passed through.
    """

    stays_clear_once_seen: bool = False
    """
    Whether the passage keeps counting as clear once it was seen clear, such as for a
    robot driving into it, whose lidar no longer looks through the passage once inside.
    """

    clear_only_after_blocked: bool = False
    """
    Whether the passage counts as clear only once it was seen blocked since this monitor
    started, such as elevator doors that have to close and open again.
    """

    seen_clear: bool = field(default=False, init=False)
    """
    Whether the passage was seen clear since this monitor started.
    """

    seen_blocked: bool = field(default=False, init=False)
    """
    Whether the passage was seen blocked since this monitor started.
    """

    def on_start(self) -> None:
        self.seen_clear = False
        self.seen_blocked = False

    def on_tick(self) -> ObservationStateValues:
        if self.stays_clear_once_seen and self.seen_clear:
            return ObservationStateValues.TRUE
        reading = self.lidar.get_lidar_reading()
        passage_distances = reading.distances_to_segment(
            self.passage_start, self.passage_end
        )
        crossing_beams = np.isfinite(passage_distances)
        if not crossing_beams.any():
            return ObservationStateValues.UNKNOWN
        if not np.all(
            reading.ranges[crossing_beams]
            > passage_distances[crossing_beams] + self.tolerance
        ):
            self.seen_blocked = True
            return ObservationStateValues.FALSE
        if self.clear_only_after_blocked and not self.seen_blocked:
            return ObservationStateValues.FALSE
        self.seen_clear = True
        return ObservationStateValues.TRUE
