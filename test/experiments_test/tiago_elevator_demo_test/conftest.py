import pytest

from semantic_digital_twin.robots.tiago import Tiago

from .scripted_social_navigation import ScriptedSocialNavigation
from .speech_listener import SpeechListener

# %% the social navigation module


@pytest.fixture
def social_navigation(rclpy_node, request):
    module = ScriptedSocialNavigation(node=rclpy_node, answer=request.param)
    yield module
    module.close()


# %% what the robot says


@pytest.fixture
def tiago_speech(rclpy_node):
    listener = SpeechListener(node=rclpy_node, topic_name=Tiago.text_to_speech_topic)
    yield listener
    listener.close()
