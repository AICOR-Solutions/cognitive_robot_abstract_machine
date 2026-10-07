from dataclasses import dataclass

from giskardpy.motion_statechart.goals.open_close import Open, Close
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.robots.robot_parts import Arm


@dataclass
class OpeningMotion(BaseMotion):
    """
    Designator for opening container.
    """

    object_part: Body
    """
    Object designator for the drawer handle
    """
    arm: Arm
    """
    Arm that should be used.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        connection = self.object_part.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        upper_limit = float(connection.dof.limits.upper.position)
        return Open(
            tip_link=tip,
            environment_link=self.object_part,
            goal_joint_state=upper_limit,
        )


@dataclass
class ClosingMotion(BaseMotion):
    """
    Designator for closing a container.
    """

    object_part: Body
    """
    Object designator for the drawer handle.
    """

    arm: Arm
    """
    Arm that should be used.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        return Close(
            tip_link=tip, environment_link=self.object_part, goal_joint_state=0.01
        )
