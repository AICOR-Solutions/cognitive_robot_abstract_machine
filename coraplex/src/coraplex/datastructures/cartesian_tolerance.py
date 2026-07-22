from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CartesianTolerance:
    """
    How precisely and how fast a Cartesian goal must be reached.

    Bundles the position threshold and the linear speed cap so a motion can be tuned
    from loose-and-fast to tight-and-slow as a single value instead of separate flags.
    """

    linear_threshold: float = 0.01
    """
    Position error (m) at which the goal counts as reached.
    """

    reference_linear_velocity: float = 0.2
    """
    Cap on the tool center point linear speed (m/s).
    """
