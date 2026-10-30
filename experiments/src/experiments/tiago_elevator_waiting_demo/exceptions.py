from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% boarding through the social navigation module


@dataclass
class BoardingTimedOutError(DataclassException):
    """
    Raised when the social navigation module did not give the base back in time.
    """

    timeout: float
    """
    Seconds the plan waited for the social navigation module.
    """

    def error_message(self) -> str:
        return (
            f"The social navigation module did not give the base back within "
            f"{self.timeout} s."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the social navigation module runs on the same ROS domain and "
            "serves the boarding action."
        )


@dataclass
class BoardingFailedError(DataclassException):
    """
    Raised when the social navigation module gave the base back without the robot
    standing inside the cabin.
    """

    elevator_name: str
    """
    The name of the elevator the robot was to board.
    """

    def error_message(self) -> str:
        return f"The robot did not board the elevator {self.elevator_name}."

    def suggest_correction(self) -> str:
        return (
            "Read the social navigation module's reason in the log; ask the people "
            "around the elevator to make room, or wait for the cabin to empty."
        )
