from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tiago import Tiago, TiagoTopic

# %% which robots can speak


def test_tiago_speaks_the_text_it_reads_from_its_text_to_speech_topic():
    assert Tiago.text_to_speech_topic == TiagoTopic.TEXT_TO_SPEECH


def test_a_robot_without_a_speaker_declares_no_text_to_speech_topic():
    assert PR2.text_to_speech_topic is None
