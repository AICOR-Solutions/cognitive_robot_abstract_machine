from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.scan_pattern import ScanPattern

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types.spatial_types import Point3
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@dataclass
class LidarReading:
    """
    One sweep of a lidar, laid out like ``sensor_msgs/LaserScan``.
    """

    reference_frame: KinematicStructureEntity
    """
    The frame the sweep was measured in.
    """

    scan_pattern: ScanPattern
    """
    The directions the sweep covered and the distances it could measure.
    """

    ranges: npt.NDArray[np.float64]
    """
    The distance each beam travelled before it hit a surface, in meters, ordered like the
    beams of :attr:`scan_pattern`.

    A beam that hit nothing within the scan pattern's range measures ``math.inf``.
    """

    def distances_to_segment(
        self, start: Point3, end: Point3
    ) -> npt.NDArray[np.float64]:
        """
        Measures how far each beam travels before it crosses a line segment in the scan
        plane.

        :param start: One end of the segment.
        :param end: The other end of the segment.
        :return: The distance along every beam, ordered like the beams of
            :attr:`scan_pattern`, to where it crosses the segment, and ``numpy.inf`` for
            beams that miss it.

        .. note:: The segment is projected onto the scan plane, the xy plane of
            :attr:`reference_frame`.
        """
        world = self.reference_frame._world
        frame_P_start = world.transform(start, self.reference_frame).to_np()[:2]
        frame_P_end = world.transform(end, self.reference_frame).to_np()[:2]
        frame_V_segment = frame_P_end - frame_P_start
        beam_directions = self.scan_pattern.beam_directions[:, :2]

        determinants = (
            frame_V_segment[0] * beam_directions[:, 1]
            - frame_V_segment[1] * beam_directions[:, 0]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            beam_distances = (
                frame_V_segment[0] * frame_P_start[1]
                - frame_V_segment[1] * frame_P_start[0]
            ) / determinants
            segment_fractions = (
                beam_directions[:, 0] * frame_P_start[1]
                - beam_directions[:, 1] * frame_P_start[0]
            ) / determinants
        crosses = (
            (determinants != 0)
            & (beam_distances > 0)
            & (segment_fractions >= 0)
            & (segment_fractions <= 1)
        )
        return np.where(crosses, beam_distances, np.inf)
