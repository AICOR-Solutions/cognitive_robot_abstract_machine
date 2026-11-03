import pytest

from semantic_digital_twin.robots.tiago import Tiago

from .speech_listener import SpeechListener

# %% what the robot says


@pytest.fixture
def tiago_speech(rclpy_node):
    listener = SpeechListener(node=rclpy_node, topic_name=Tiago.text_to_speech_topic)
    yield listener
    listener.close()
