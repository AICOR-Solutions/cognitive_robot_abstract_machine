from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

from rosidl_runtime_py.utilities import get_action
from typing_extensions import Any

from coraplex.robot_plans.actions.base import SequentialAction
from cramph.node import StatechartNode
from giskardpy.motion_statechart.ros2_nodes.await_action_result import (
    AwaitActionResult,
)

logger = logging.getLogger(__name__)

# %% the say skill


class SaySkillInterface(StrEnum):
    """
    Where the type of the ROS4HRI ``say`` skill is found.
    """

    PACKAGE = "communication_skills"
    ACTION_TYPE = "communication_skills/action/Say"


# %% speaking aloud


@dataclass(eq=False, repr=False, kw_only=True)
class Speak(AwaitActionResult):
    """
    Says a sentence through the ROS4HRI ``say`` skill and waits until it ended.

    Stays silent with a warning where the skill's package is not installed.
    """

    text: str
    """
    The sentence to say.
    """

    def find_action_type(self) -> type | None:
        if find_spec(SaySkillInterface.PACKAGE) is None:
            logger.warning(
                f"{SaySkillInterface.PACKAGE} is not installed, not saying: {self.text}"
            )
            return None
        return get_action(SaySkillInterface.ACTION_TYPE)

    def build_action_goal(self) -> Any:
        return get_action(SaySkillInterface.ACTION_TYPE).Goal(input=self.text)


@dataclass(eq=False, repr=False, kw_only=True)
class SayAction(SequentialAction):
    """
    Says a sentence aloud through the robot's text to speech action and waits until the
    sentence ended.

    The robot stays silent if it cannot speak, because it declares no
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.text_to_speech_action`,
    or if the plan runs where there is no ROS node or no ``communication_skills`` package
    to send the sentence with.
    """

    text: str
    """
    The sentence the robot says.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [Speak(text=self.text, action_name=self.robot.text_to_speech_action)]
