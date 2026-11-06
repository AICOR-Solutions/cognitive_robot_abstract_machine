import pytest

from coraplex.robot_plans.actions.core.speech import SaySkillInterface
from semantic_digital_twin.robots.tiago import Tiago

from .speech_listener import SpeechListener

# %% what the robot says


@pytest.fixture
def tiago_speech(rclpy_node):
    pytest.importorskip(SaySkillInterface.PACKAGE)
    listener = SpeechListener(node=rclpy_node, action_name=Tiago.text_to_speech_action)
    yield listener
    listener.close()
