import threading
import time
from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
)

# %% the cabin leaving and coming back


@dataclass
class NextElevator:
    """
    Sends the cabin to another level and back, the way the people at the elevator ride
    off and the next cabin arrives.
    """

    elevator: Elevator
    """
    The elevator whose cabin leaves and comes back.
    """

    home_level: Level
    """
    The level the cabin comes back to.
    """

    away_level: Level
    """
    The level the cabin leaves to.
    """

    seconds_away: float = 0.5
    """
    Seconds the cabin stays away, enough for a refusal to reach the plan through ROS.
    """

    return_times: list[float] = field(default_factory=list, init=False)
    """
    When the cabin came back from each round, on the monotonic clock.
    """

    _rounds: list[threading.Thread] = field(default_factory=list, init=False)
    """
    The rounds started so far.
    """

    def start_round(self) -> None:
        """
        Let the cabin leave and come back while the caller goes on.
        """
        cabin_round = threading.Thread(target=self._drive_round)
        self._rounds.append(cabin_round)
        cabin_round.start()

    def _drive_round(self) -> None:
        """
        Let the cabin leave and come back.
        """
        self.elevator.drive_to_floor(self.away_level)
        time.sleep(self.seconds_away)
        self.elevator.drive_to_floor(self.home_level)
        self.return_times.append(time.monotonic())

    def wait_for_rounds(self) -> None:
        """
        Wait until the cabin is back from every round.
        """
        for cabin_round in self._rounds:
            cabin_round.join()
