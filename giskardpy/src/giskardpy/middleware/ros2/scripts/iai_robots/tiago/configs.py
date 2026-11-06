from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithMobileRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass
class WorldWithTiagoConfig(WorldWithMobileRobot):
    """
    A world containing only the Tiago, which drives on a differential drive.
    """

    robot_class: type[Tiago] = field(kw_only=True, default=Tiago, init=False)
    """
    The robot class that is applied to the parsed description.
    """

    drive_class: type[DifferentialDrive] = field(
        kw_only=True, default=DifferentialDrive, init=False
    )
    """
    The kind of connection between the odometry body and the robot.
    """


@dataclass
class TiagoVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the drive of the Tiago through its velocity topic.
    """

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        self.sync_robot_parts()

        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.add_drive_velocity_command_topic(
            command_topic="/cmd_vel", drive=diff_drive
        )
