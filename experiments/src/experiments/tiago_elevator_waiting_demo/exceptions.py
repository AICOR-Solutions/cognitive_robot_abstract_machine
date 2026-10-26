from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from experiments.tiago_elevator_waiting_demo.demo import BoardingResult

# %% boarding through the social navigation module


@dataclass
class BoardingTimedOutError(DataclassException):
    """
    Raised when the social navigation module did not give the base back in time.
    """

    timeout: float
    """
    Seconds the plan waited for the boarding result.
    """

    def error_message(self) -> str:
        return (
            f"The social navigation module sent no boarding result within "
            f"{self.timeout} s."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the social navigation module runs on the same ROS domain and "
            "listens to the boarding request topic."
        )


@dataclass
class BoardingFailedError(DataclassException):
    """
    Raised when the social navigation module gave the base back without the robot having
    boarded.
    """

    result: BoardingResult
    """
    The result the social navigation module sent.
    """

    def error_message(self) -> str:
        return f"The robot did not board the elevator: {self.result.reason}."

    def suggest_correction(self) -> str:
        return "Ask the people around the elevator to make room, or wait for the cabin to empty."
