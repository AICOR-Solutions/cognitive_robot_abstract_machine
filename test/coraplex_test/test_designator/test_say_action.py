import threading

from std_msgs.msg import String

from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.speech import SayAction

# %% saying a sentence


def test_a_robot_that_can_speak_says_the_sentence_on_its_text_to_speech_topic(
    tiago_apartment_context, rclpy_node
):
    _, robot, context = tiago_apartment_context
    context.ros_node = rclpy_node
    action = SayAction("Hello")
    received_messages = []
    message_received = threading.Event()

    def receive(message: String) -> None:
        received_messages.append(message)
        message_received.set()

    rclpy_node.create_subscription(String, robot.text_to_speech_topic, receive, 10)

    with kinematically_simulated_robot:
        execute_single(action, context=context).perform()

    assert message_received.wait(timeout=5.0)
    assert received_messages == [String(data=action.text)]


def test_a_robot_that_cannot_speak_stays_silent(pr2_apartment_context):
    _, _, context = pr2_apartment_context

    with kinematically_simulated_robot:
        node = execute_single(SayAction("Hello"), context=context)
        node.perform()

    assert not any(isinstance(plan_node, MotionNode) for plan_node in node.plan.nodes)


def test_a_robot_that_can_speak_stays_silent_without_a_ros_node(
    tiago_apartment_context,
):
    _, _, context = tiago_apartment_context

    with kinematically_simulated_robot:
        node = execute_single(SayAction("Hello"), context=context)
        node.perform()

    assert not any(isinstance(plan_node, MotionNode) for plan_node in node.plan.nodes)
