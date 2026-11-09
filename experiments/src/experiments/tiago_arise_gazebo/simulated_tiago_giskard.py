"""
The Giskard that drives the Tiago of the TIAGo ARISE Gazebo simulator.

Run it as ``python -m experiments.tiago_arise_gazebo.simulated_tiago_giskard`` once the
simulator publishes the robot description.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago import tiago_velocity

# %% launching


@dataclass
class SimulatedTiagoGiskard:
    """
    The Giskard the Tiago launcher
    :mod:`~giskardpy.middleware.ros2.scripts.iai_robots.tiago.tiago_velocity` builds,
    with its drive allowed to move at the velocity limits given here.
    """

    translation_velocity_limit: float = 0.5
    """
    The highest speed the drive may move the base at, in meters per second.
    """

    rotation_velocity_limit: float = 0.6
    """
    The highest speed the drive may turn the base at, in radians per second.
    """

    def create_giskard(self) -> Giskard:
        """
        :return: A new Giskard for the Tiago whose drive moves within this launcher's
            velocity limits.
        """
        giskard = tiago_velocity.create_giskard()
        giskard.world_config.translation_velocity_limit = (
            self.translation_velocity_limit
        )
        giskard.world_config.rotation_velocity_limit = self.rotation_velocity_limit
        return giskard


def main() -> None:
    """
    Run the Giskard of :class:`SimulatedTiagoGiskard` as a ros node until ROS shuts
    down.
    """
    Giskard.launch(SimulatedTiagoGiskard().create_giskard)


if __name__ == "__main__":
    main()
