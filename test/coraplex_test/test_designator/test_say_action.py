import time
from datetime import timedelta

import pytest
from json_msgs.action import JsonAction
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from typing_extensions import Any

from coraplex.robot_plans.actions.core import speech
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.speech import (
    SayAction,
    SaySkillInterface,
    Speak,
)
from cramph.composites import Sequence
from giskardpy.motion_statechart.ros2_nodes.await_action_result import (
    AwaitActionResult,
)
from krrood_spatial_math.spatial_types import Pose

from ...experiments_test.tiago_elevator_demo_test.speech_listener import SpeechListener
from ...plan_running import run_plan

# %% saying a sentence


def test_a_robot_that_can_speak_says_the_sentence_through_its_text_to_speech_action(
    tiago_apartment_context, rclpy_node
):
    pytest.importorskip(SaySkillInterface.PACKAGE)
    _, robot, extensions = tiago_apartment_context
    action = SayAction(text="Hello")
    listener = SpeechListener(node=rclpy_node, action_name=robot.text_to_speech_action)

    run_plan(action, extensions, ros_node=rclpy_node)

    assert listener.wait_for_sentences(1) == [action.text]
    listener.close()


def test_a_robot_that_cannot_speak_stays_silent(pr2_apartment_context, rclpy_node):
    _, _, extensions = pr2_apartment_context
    action = SayAction(text="Hello")

    run_plan(action, extensions, ros_node=rclpy_node)

    [speaking] = action.nodes
    assert isinstance(speaking, Speak)
    assert speaking._action_client is None


def test_a_robot_that_can_speak_stays_silent_without_a_ros_node(
    tiago_apartment_context,
):
    _, _, extensions = tiago_apartment_context
    action = SayAction(text="Hello")

    run_plan(action, extensions)

    [speaking] = action.nodes
    assert speaking._action_client is None


def test_a_robot_that_can_speak_stays_silent_without_the_package_of_the_action_type(
    tiago_apartment_context, rclpy_node, monkeypatch
):
    _, _, extensions = tiago_apartment_context
    monkeypatch.setattr(speech, "find_spec", lambda _: None)
    action = SayAction(text="Hello")

    run_plan(action, extensions, ros_node=rclpy_node)

    [speaking] = action.nodes
    assert speaking._action_client is None


# %% waiting between two motions


class GreetingGoal(AwaitActionResult):
    """
    Sends a goal of an action type every checkout has.
    """

    def find_action_type(self) -> type:
        return JsonAction

    def build_action_goal(self) -> Any:
        return JsonAction.Goal(goal="Hello")


def test_a_long_wait_before_a_motion_does_not_count_as_a_stalled_motion(
    tiago_apartment_context, rclpy_node
):
    _, robot, extensions = tiago_apartment_context
    action_name = "/greeting"
    # Far beyond the stall monitor's timeout in simulated time, whose ticks run unpaced.
    wait_in_seconds = 1.0
    goals_ended = []

    def execute(goal_handle: ServerGoalHandle) -> JsonAction.Result:
        time.sleep(wait_in_seconds)
        goals_ended.append(goal_handle.request)
        goal_handle.succeed()
        return JsonAction.Result()

    server = ActionServer(rclpy_node, JsonAction, action_name, execute)

    run_plan(
        Sequence(
            nodes=[
                GreetingGoal(action_name=action_name),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        x=robot.root.global_transform.to_np()[0, 3] + 0.1,
                        y=robot.root.global_transform.to_np()[1, 3],
                        reference_frame=robot._world.root,
                    )
                ),
            ]
        ),
        extensions,
        ros_node=rclpy_node,
        # The unpaced ticks of the wait would use up the default budget of simulated time.
        simulation_time_per_action=timedelta(hours=1),
    )

    assert len(goals_ended) == 1
    server.destroy()
