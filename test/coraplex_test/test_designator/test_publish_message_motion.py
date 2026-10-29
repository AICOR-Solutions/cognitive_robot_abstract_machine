import threading

from std_msgs.msg import String

from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.misc import PublishMessageMotion
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import PublishOnStart
from semantic_digital_twin.robots.tiago import TiagoTopic

# %% publishing a message from a plan


def test_publish_message_motion_publishes_its_message_on_its_topic(
    pr2_apartment_context,
):
    """
    The motion's chart publishes exactly the message it was given on the topic it names.
    """
    _, _, context = pr2_apartment_context
    motion = PublishMessageMotion(
        topic_name=TiagoTopic.TEXT_TO_SPEECH, message=String(data="Hello")
    )
    execute_single(motion, context=context)

    motion_chart = motion.motion_chart

    assert isinstance(motion_chart, PublishOnStart)
    assert motion_chart.topic_name == motion.topic_name
    assert motion_chart.msg == motion.message


def test_performed_publish_message_motion_reaches_a_subscriber(
    pr2_apartment_context, rclpy_node
):
    """
    Performing the motion in a plan sends its message to whoever listens on its topic.
    """
    _, _, context = pr2_apartment_context
    context.ros_node = rclpy_node
    motion = PublishMessageMotion(
        topic_name=TiagoTopic.TEXT_TO_SPEECH, message=String(data="Hello")
    )
    received_messages = []
    message_received = threading.Event()

    def receive(message: String) -> None:
        received_messages.append(message)
        message_received.set()

    rclpy_node.create_subscription(String, motion.topic_name, receive, 10)

    with kinematically_simulated_robot:
        execute_single(motion, context=context).perform()

    assert message_received.wait(timeout=5.0)
    assert received_messages == [motion.message]
