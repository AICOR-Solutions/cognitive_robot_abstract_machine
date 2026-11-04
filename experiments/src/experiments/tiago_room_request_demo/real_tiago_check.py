"""
Tries the room request demonstration's parts on the real Tiago, one stage at a time.

Source ROS and the workspace holding ``communication_skills``, then run the stages in
order with ``python -m experiments.tiago_room_request_demo.real_tiago_check <stage>``:

- ``preflight`` lists what is served and published on the robot, and changes nothing.
- ``say`` speaks a sentence through the skill, then through Giskard.
- ``people`` prints the people the robot tracks.
- ``ask`` asks a person whether the robot may board, and prints the answer.
- ``intent`` waits to be told a room, asks to confirm it, and prints it without driving.
- ``drive`` runs the whole demonstration.

.. warning:: Only ``drive`` moves the robot. Keep a person at the emergency stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum

from action_msgs.msg import GoalStatus
from communication_skills.action import Say
from json_msgs.action import JsonAction
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from std_skills.msg import Result as SkillResult
from typing_extensions import Any, Type

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.robot_plans.actions.core.speech import SayAction
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from experiments.ra_fetch_elevator_people.elevator_people import Ros4HriTopic
from experiments.tiago_arise_gazebo.elevator_drive import SimulatorTopic
from experiments.tiago_room_request_demo.boarding_permission import BoardingQuestion
from experiments.tiago_room_request_demo.demo import TiagoRoomRequestDemonstration
from experiments.tiago_room_request_demo.persons_at_elevator import PersonsAtElevator
from experiments.tiago_room_request_demo.room_request import (
    Ros4HriInteractionTopic,
    RoomRequests,
)
from experiments.tiago_room_request_demo.yes_no_question import (
    Ros4HriVoiceTopic,
    YesNoQuestionAsker,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeBuilding
from semantic_digital_twin.robots.tiago import Tiago, TiagoTopic
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator

# %% the stages


class Stage(StrEnum):
    """
    What can be tried on the robot.
    """

    PREFLIGHT = "preflight"
    SAY = "say"
    PEOPLE = "people"
    ASK = "ask"
    INTENT = "intent"
    DRIVE = "drive"


class GiskardAction(StrEnum):
    """
    Names of the actions Giskard serves.
    """

    COMMAND = "giskard/command"


class DriveConfirmation(StrEnum):
    """
    What has to be typed before the robot may move.
    """

    PHRASE = "DRIVE"


# %% what the stages need from the robot


@dataclass
class RequiredAction:
    """
    An action that has to be served on the robot.
    """

    action_name: str
    """
    The name of the action.
    """

    action_type: Type
    """
    The type of the action.
    """

    needed_by: list[Stage]
    """
    The stages that cannot run without it.
    """


@dataclass
class RequiredTopic:
    """
    A topic something on the robot has to publish.
    """

    topic_name: str
    """
    The name of the topic.
    """

    needed_by: list[Stage]
    """
    The stages that cannot run without it.
    """


def actions_of_the_demonstration() -> list[RequiredAction]:
    """
    :return: The actions the stages call.
    """
    return [
        RequiredAction(
            Tiago.text_to_speech_action,
            Say,
            [Stage.SAY, Stage.ASK, Stage.INTENT, Stage.DRIVE],
        ),
        RequiredAction(GiskardAction.COMMAND, JsonAction, [Stage.SAY, Stage.DRIVE]),
    ]


def topics_of_the_demonstration() -> list[RequiredTopic]:
    """
    :return: The topics the stages read.
    """
    return [
        RequiredTopic(
            Ros4HriTopic.TRACKED_PERSONS, [Stage.PEOPLE, Stage.ASK, Stage.DRIVE]
        ),
        RequiredTopic(
            Ros4HriVoiceTopic.ANONYMOUS_SPEAKER_SPEECH,
            [Stage.ASK, Stage.INTENT, Stage.DRIVE],
        ),
        RequiredTopic(Ros4HriInteractionTopic.INTENTS, [Stage.INTENT, Stage.DRIVE]),
        RequiredTopic(TiagoTopic.LIDAR, [Stage.DRIVE]),
        RequiredTopic(SimulatorTopic.ELEVATOR_JOINT_STATES, [Stage.DRIVE]),
    ]


@dataclass
class PreflightFinding:
    """
    Whether one thing a stage needs is on the robot.
    """

    description: str
    """
    The action or topic that was looked for.
    """

    found: bool
    """
    Whether it is served or published.
    """

    needed_by: list[Stage]
    """
    The stages that cannot run without it.
    """

    def __str__(self) -> str:
        status = "ok     " if self.found else "MISSING"
        return f"{status} {self.description} (needed by: {', '.join(self.needed_by)})"


@dataclass
class RealTiagoPreflight:
    """
    Looks for what the stages need on the robot, without sending it anything.
    """

    node: Node
    """
    The node the robot is looked at from, which an executor has to spin elsewhere.
    """

    actions: list[RequiredAction] = field(default_factory=actions_of_the_demonstration)
    """
    The actions looked for.
    """

    topics: list[RequiredTopic] = field(default_factory=topics_of_the_demonstration)
    """
    The topics looked for.
    """

    discovery_time: float = 3.0
    """
    Seconds given to discover what runs on the robot.
    """

    def find(self) -> list[PreflightFinding]:
        """
        :return: What was found and what is missing.
        """
        clients = [
            ActionClient(self.node, action.action_type, action.action_name)
            for action in self.actions
        ]
        time.sleep(self.discovery_time)
        findings = [
            PreflightFinding(
                f"action {action.action_name}",
                client.server_is_ready(),
                action.needed_by,
            )
            for action, client in zip(self.actions, clients)
        ]
        for client in clients:
            client.destroy()
        return findings + [
            PreflightFinding(
                f"topic {topic.topic_name}",
                self.node.count_publishers(topic.topic_name) > 0,
                topic.needed_by,
            )
            for topic in self.topics
        ]


# %% trying the stages


@dataclass
class RealTiagoCheck:
    """
    Runs the stages against the robot.
    """

    session: RobotDemonstrationRosSession
    """
    The ROS session the stages run in.
    """

    server_timeout: float = 10.0
    """
    Seconds to wait for an action to be served.
    """

    speech_timeout: float = 60.0
    """
    Seconds a sentence may take.
    """

    observation_time: float = 3.0
    """
    Seconds to listen to the people the robot tracks.
    """

    @property
    def node(self) -> Node:
        """
        :return: The node the stages call the robot from.
        """
        return self.session.node

    def preflight(self) -> bool:
        """
        Print what the stages need from the robot.

        :return: Whether all of it is there.
        """
        findings = RealTiagoPreflight(self.node).find()
        for finding in findings:
            print(finding)
        return all(finding.found for finding in findings)

    def say(self, sentence: str) -> bool:
        """
        Speak ``sentence`` directly through the skill, then through Giskard.

        :return: Whether the sentence was spoken and Giskard's chart lasted for at least
            half as long as the sentence, which shows that it waited.
        """
        skill_seconds = self.say_through_skill(sentence)
        if skill_seconds is None:
            print("The skill did not speak the sentence.")
            return False
        print(f"The skill took {skill_seconds:.1f} s.")
        giskard_seconds = self._say_through_giskard(sentence)
        print(f"SayAction through Giskard took {giskard_seconds:.1f} s.")
        if giskard_seconds < skill_seconds / 2:
            print(
                "SayAction did not wait. Look in Giskard's log for a warning about "
                f"{Tiago.text_to_speech_action}."
            )
            return False
        return True

    def say_through_skill(self, sentence: str) -> float | None:
        """
        :return: The seconds the skill took to speak ``sentence``, None if it failed.
        """
        client = ActionClient(self.node, Say, Tiago.text_to_speech_action)
        if not client.wait_for_server(timeout_sec=self.server_timeout):
            return None
        start = time.monotonic()
        goal_handle = self._wait_for(client.send_goal_async(Say.Goal(input=sentence)))
        if goal_handle is None or not goal_handle.accepted:
            return None
        response = self._wait_for(goal_handle.get_result_async())
        if (
            response is None
            or response.status != GoalStatus.STATUS_SUCCEEDED
            or response.result.result.error_code != SkillResult.ROS_ENOERR
        ):
            return None
        return time.monotonic() - start

    def _wait_for(self, future: Future) -> Any | None:
        """
        :return: What ``future`` resolved to, None if it did not within
            :attr:`speech_timeout`.
        """
        deadline = time.monotonic() + self.speech_timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _say_through_giskard(self, sentence: str) -> float:
        """
        :return: The seconds a plan holding nothing but :class:`SayAction` took on
            Giskard.
        """
        world = self.session.fetch_world()
        WorldSynchronizer(_world=world, node=self.node)
        executor = RealRobotPlanExecutor(
            StatechartContext.with_extensions(
                world,
                [RobotAccess(world.get_semantic_annotations_by_type(Tiago)[0])],
            ),
            ros_node=self.node,
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(SayAction(text=sentence))
        start = time.monotonic()
        executor.compile(statechart)
        executor.execute()
        executor.close()
        return time.monotonic() - start

    def _persons_at_elevator(self) -> PersonsAtElevator:
        """
        :return: A listener to the people the robot tracks, around an office kept out of
            the controller's world.
        """
        world = AriseOfficeBuilding().create_world()
        return PersonsAtElevator(
            node=self.node,
            topic_name=Ros4HriTopic.TRACKED_PERSONS,
            elevator=world.get_semantic_annotations_by_type(Elevator)[0],
        )

    def people(self) -> bool:
        """
        Print the people the robot tracks and where they are.

        :return: Whether anybody is tracked.
        """
        persons = self._persons_at_elevator()
        time.sleep(self.observation_time)
        positions = persons.positions()
        for position in positions:
            print(f"{position.person_id}: {position.head_position.to_np()[:3]}")
        persons.close()
        return len(positions) > 0

    def ask(self, person_id: str | None) -> bool:
        """
        Ask whether the robot may board, and print the answer.

        :param person_id: Who to ask, the first person tracked if None, and anybody if
            nobody is tracked.
        :return: Whether the person agreed.
        """
        if person_id is None:
            persons = self._persons_at_elevator()
            time.sleep(self.observation_time)
            tracked = persons.latest_message.ids if persons.latest_message else []
            person_id = tracked[0] if tracked else ""
            persons.close()
        print(
            f"Asking {person_id or 'anybody'}. Say yes or no once the robot finished."
        )
        asker = YesNoQuestionAsker(self.node, Tiago.text_to_speech_action)
        may_board = asker.ask_until_answered(
            BoardingQuestion.ASKING_TO_BOARD, person_id
        )
        asker.close()
        print(f"May the robot board: {may_board}")
        return may_board

    def intent(self, timeout: float) -> bool:
        """
        Wait to be told a room and confirm it, and print it without driving there.

        :param timeout: Seconds to wait at most.
        :return: Always true, as not hearing a room raises.
        :raises NoRoomRequestedError: If no room of the office was requested in time.
        """
        office = AriseOfficeBuilding()
        world = office.create_world()
        room_requests = RoomRequests(
            node=self.node,
            world=world,
            office=office,
            text_to_speech_action=Tiago.text_to_speech_action,
        )
        print("Say the whole name of a room, then answer the robot with yes or no.")
        requested_room = room_requests.wait_for_confirmed_room(timeout)
        print(f"Heard '{requested_room.spoken_name}': {requested_room.room.name.name}")
        room_requests.close()
        return True

    def drive(self) -> bool:
        """
        Run the whole demonstration, if everything it needs is there and the operator
        confirms.

        :return: Whether the demonstration ran.
        :raises TargetRoomNotReachedError: If the robot did not reach the requested
            room.
        """
        if not self.preflight():
            print("Not driving: something the demonstration needs is missing.")
            return False
        print(
            "This moves the robot and adds the ARISE office to Giskard's world.\n"
            "- The office model has to match the building the robot stands in.\n"
            "- The robot does not avoid people or anything else not in the model.\n"
            "- Keep a person at the emergency stop."
        )
        answer = input(f"Type {DriveConfirmation.PHRASE} to go on: ")
        if answer != DriveConfirmation.PHRASE:
            print("Not driving.")
            return False
        TiagoRoomRequestDemonstration(
            used_robot=Tiago, executor_type=RealRobotPlanExecutor
        ).run()
        return True


# %% running a stage


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """
    :param arguments: The command line, the one of this process if None.
    :return: The stage to run and its options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("stage", type=Stage, choices=list(Stage))
    parser.add_argument(
        "--sentence",
        default="Hello, I am checking that I can speak.",
        help="what the say stage speaks",
    )
    parser.add_argument("--person-id", help="who the ask stage asks")
    parser.add_argument(
        "--intent-timeout",
        type=float,
        default=120.0,
        help="seconds the intent stage waits",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """
    Run one stage.

    :param arguments: The command line, the one of this process if None.
    :return: The exit code, zero if the stage passed.
    """
    options = parse_arguments(arguments)
    with RobotDemonstrationRosSession.start("real_tiago_check") as session:
        check = RealTiagoCheck(session)
        match options.stage:
            case Stage.PREFLIGHT:
                passed = check.preflight()
            case Stage.SAY:
                passed = check.say(options.sentence)
            case Stage.PEOPLE:
                passed = check.people()
            case Stage.ASK:
                passed = check.ask(options.person_id)
            case Stage.INTENT:
                passed = check.intent(options.intent_timeout)
            case Stage.DRIVE:
                passed = check.drive()
    print(f"{options.stage}: {'passed' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
