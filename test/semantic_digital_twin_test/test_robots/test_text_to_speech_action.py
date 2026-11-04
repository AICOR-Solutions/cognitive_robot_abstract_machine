from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tiago import Tiago, TiagoAction

# %% which robots can speak


def test_tiago_speaks_through_its_text_to_speech_action():
    assert Tiago.text_to_speech_action == TiagoAction.TEXT_TO_SPEECH


def test_a_robot_without_a_speaker_declares_no_text_to_speech_action():
    assert PR2.text_to_speech_action is None
