import threading

from std_msgs.msg import String

from coraplex.robot_plans.actions.core.speech import SayAction, SpeakOnStart

from ...plan_running import run_plan

# %% saying a sentence


def test_a_robot_that_can_speak_says_the_sentence_on_its_text_to_speech_topic(
    tiago_apartment_context, rclpy_node
):
    _, robot, extensions = tiago_apartment_context
    action = SayAction(text="Hello")
    received_messages = []
    message_received = threading.Event()

    def receive(message: String) -> None:
        received_messages.append(message)
        message_received.set()

    rclpy_node.create_subscription(String, robot.text_to_speech_topic, receive, 10)

    run_plan(action, extensions, ros_node=rclpy_node)

    assert message_received.wait(timeout=5.0)
    assert received_messages == [String(data=action.text)]


def test_a_robot_that_cannot_speak_stays_silent(pr2_apartment_context, rclpy_node):
    _, _, extensions = pr2_apartment_context
    action = SayAction(text="Hello")

    run_plan(action, extensions, ros_node=rclpy_node)

    [speaking] = action.nodes
    assert isinstance(speaking, SpeakOnStart)
    assert speaking._publisher is None


def test_a_robot_that_can_speak_stays_silent_without_a_ros_node(
    tiago_apartment_context,
):
    _, _, extensions = tiago_apartment_context
    action = SayAction(text="Hello")

    run_plan(action, extensions)

    [speaking] = action.nodes
    assert speaking._publisher is None
