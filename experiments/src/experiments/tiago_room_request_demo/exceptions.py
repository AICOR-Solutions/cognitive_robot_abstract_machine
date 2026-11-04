from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% asking to board


@dataclass
class BoardingPermissionNotGrantedError(DataclassException):
    """
    Raised when the people at the elevator refused the robot every time it asked to
    board.
    """

    maximum_attempts: int
    """
    How many times the robot asked before it gave up.
    """

    def error_message(self) -> str:
        return (
            f"The robot was not allowed to board the elevator in "
            f"{self.maximum_attempts} attempts."
        )

    def suggest_correction(self) -> str:
        return (
            "Allow the robot more attempts, or check that the speech recognition hears "
            "a yes or no from the people at the elevator."
        )


@dataclass
class ElevatorCabinTimedOutError(DataclassException):
    """
    Raised when the elevator's cabin neither arrived nor left in time while the robot
    waited to board.
    """

    timeout: float
    """
    Seconds the robot waited for the cabin.
    """

    def error_message(self) -> str:
        return f"The elevator's cabin did not arrive or leave within {self.timeout} s."

    def suggest_correction(self) -> str:
        return "Check that the elevator runs and that the world follows its cabin."


@dataclass
class SaySkillNotServedError(DataclassException):
    """
    Raised when the robot has somebody to ask and the ``say`` skill is not served.
    """

    action_name: str
    """
    The name the skill was expected under.
    """

    timeout: float
    """
    Seconds the robot waited for the skill.
    """

    def error_message(self) -> str:
        return f"The action {self.action_name} was not served within {self.timeout} s."

    def suggest_correction(self) -> str:
        return (
            "Check that the communication hub runs on the robot's ROS domain and serves "
            "the say skill."
        )


# %% being told where to go


@dataclass
class NoRoomRequestedError(DataclassException):
    """
    Raised when nobody told the robot a room it knows in time.
    """

    timeout: float
    """
    Seconds the robot waited for a room.
    """

    def error_message(self) -> str:
        return f"No known room was requested within {self.timeout} s."

    def suggest_correction(self) -> str:
        return (
            "Check that the chatbot publishes intents on the robot's ROS domain and that "
            "they name the whole name of a room of the office, such as "
            "'east office on the first floor'."
        )
