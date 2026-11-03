#!/usr/bin/env python
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoVelocityInterface,
    WorldWithTiagoConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that commands the Tiago, described by what the robot publishes on
    ``/robot_description``.

    .. note:: The ros node has to be initialized first, because the robot description is
        read from its topic.
    """
    return Giskard(
        world_config=WorldWithTiagoConfig(urdf=get_robot_description()),
        robot_interface_config=TiagoVelocityInterface(),
        server_config=GiskardServerConfig(debug_mode=False),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
