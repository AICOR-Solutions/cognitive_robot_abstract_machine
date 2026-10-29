from __future__ import annotations

from dataclasses import dataclass

from std_msgs.msg import String

from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.misc import PublishMessageMotion

# %% speaking aloud


@dataclass
class SayAction(ActionDescription):
    """
    Says a sentence aloud on the robot's text to speech topic.

    The robot stays silent if it cannot speak, because it declares no
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.text_to_speech_topic`,
    or if the plan's context has no ROS node to publish the sentence with.
    """

    text: str
    """
    The sentence the robot says.
    """

    @property
    def _action_plan(self) -> PlanNode:
        if self.robot.text_to_speech_topic is None or self.context.ros_node is None:
            return sequential([])
        return execute_single(
            PublishMessageMotion(
                topic_name=self.robot.text_to_speech_topic,
                message=String(data=self.text),
            )
        )
