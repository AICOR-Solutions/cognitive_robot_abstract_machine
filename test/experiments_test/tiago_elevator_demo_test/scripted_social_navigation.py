import threading
from dataclasses import dataclass, field
from enum import StrEnum

from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node

from experiments.tiago_elevator_waiting_demo.social_navigation import (
    SocialNavigationAction,
)
from ra_fetch_interfaces.action import BoardElevator

# %% a social navigation module with a fixed answer


class BoardingAnswer(StrEnum):
    """
    How the scripted social navigation module answers a boarding goal.
    """

    SUCCEED = "succeed"
    ABORT = "abort"
    REJECT = "reject"
    KEEP_THE_BASE = "keep_the_base"


@dataclass
class ScriptedSocialNavigation:
    """
    Serves the boarding action, answering every goal the same way without moving the
    robot.
    """

    node: Node
    """
    The node the action is served on.
    """

    answer: BoardingAnswer
    """
    How every goal is answered.
    """

    goals: list[BoardElevator.Goal] = field(default_factory=list, init=False)
    """
    The goals received, in order.
    """

    released: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set to let a module that keeps the base give it back.
    """

    server: ActionServer = field(init=False)
    """
    The served action.
    """

    def __post_init__(self) -> None:
        self.server = ActionServer(
            self.node,
            BoardElevator,
            SocialNavigationAction.BOARD_ELEVATOR,
            execute_callback=self.execute,
            goal_callback=self.decide,
        )

    def decide(self, goal: BoardElevator.Goal) -> GoalResponse:
        if self.answer is BoardingAnswer.REJECT:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self, goal_handle) -> BoardElevator.Result:
        self.goals.append(goal_handle.request)
        if self.answer is BoardingAnswer.KEEP_THE_BASE:
            self.released.wait()
        if self.answer is BoardingAnswer.SUCCEED:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return BoardElevator.Result(reason=self.answer)

    def close(self) -> None:
        self.released.set()
        self.server.destroy()
