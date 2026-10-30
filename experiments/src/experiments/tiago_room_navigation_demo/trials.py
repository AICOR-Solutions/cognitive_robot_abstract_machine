"""
Repeated trials of :class:`TiagoRoomNavigationDemonstration` to target rooms sampled
from the rooms annotated in the ARISE office, with an overview of how many succeeded and
why the others failed.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import Type

from coraplex.datastructures.enums import ExecutionType
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.demo import (
    TiagoRoomNavigationDemonstration,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Room,
)

# %% outcomes


@dataclass
class RoomNavigationTrialResult:
    """
    The outcome of one drive to a target room.
    """

    target_room_name: str
    """
    The name of the room the robot was sent to.
    """

    failure: Exception | None = None
    """
    What made the trial fail, or None if the robot reached the room.
    """

    @property
    def succeeded(self) -> bool:
        """
        Whether the robot reached the target room.
        """
        return self.failure is None


@dataclass
class RoomNavigationTrialSummary:
    """
    The outcomes of a series of drives to target rooms.
    """

    results: list[RoomNavigationTrialResult] = field(default_factory=list)
    """
    The outcome of each trial, in the order the trials ran.
    """

    @property
    def number_of_successes(self) -> int:
        """
        How many trials reached their target room.
        """
        return sum(result.succeeded for result in self.results)

    @property
    def number_of_failures(self) -> int:
        """
        How many trials did not reach their target room.
        """
        return len(self.results) - self.number_of_successes

    @property
    def failures_by_type(self) -> Counter[str]:
        """
        How many trials failed with each type of failure, by the type's name.
        """
        return Counter(
            type(result.failure).__name__
            for result in self.results
            if not result.succeeded
        )

    @property
    def trials_by_target_room(self) -> dict[str, list[RoomNavigationTrialResult]]:
        """
        The trials to each target room, by the room's name.
        """
        trials = {}
        for result in self.results:
            trials.setdefault(result.target_room_name, []).append(result)
        return trials

    def report(self) -> str:
        """
        :return: A human-readable overview of how many trials succeeded, per target
            room and in total, and why the others failed.
        """
        lines = [
            f"Room navigation: {len(self.results)} trials, "
            f"{self.number_of_successes} succeeded, {self.number_of_failures} failed.",
            "Per target room:",
        ]
        for room_name, trials in sorted(self.trials_by_target_room.items()):
            successes = sum(trial.succeeded for trial in trials)
            lines.append(f"  {room_name}: {successes}/{len(trials)} succeeded")
        if self.number_of_failures:
            lines.append("Failures by type:")
            for failure_type, count in self.failures_by_type.most_common():
                lines.append(f"  {failure_type}: {count}")
            lines.append("Failed trials:")
            for number, result in enumerate(self.results, start=1):
                if result.succeeded:
                    continue
                message_lines = str(result.failure).splitlines() or [""]
                lines.append(
                    f"  #{number} to {result.target_room_name}: "
                    f"{type(result.failure).__name__}: {message_lines[0]}"
                )
        return "\n".join(lines)


# %% trials


@dataclass
class RoomNavigationTrials:
    """
    Sends the robot from the room it starts in to target rooms sampled from the rooms
    annotated in the office, each trial in a fresh world.
    """

    number_of_trials: int = 50
    """
    How many drives to perform.
    """

    seed: int | None = None
    """
    Seed of the sampling of the target rooms, or None to sample them differently each
    time.
    """

    used_robot: Type[AbstractRobot] = Tiago
    """
    The robot that drives.
    """

    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED
    """
    Whether the robot is driven for real or simulated.
    """

    def target_room_names(self) -> list[str]:
        """
        :return: The names of the :attr:`number_of_trials` target rooms, sampled
            uniformly from the rooms of the office except the one the robot starts in.
        """
        world = TiagoElevatorRideDemonstration(
            used_robot=self.used_robot
        ).build_simulated_world()
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        start_floor = Floor.under(robot.root)
        candidates = sorted(
            room.name.name
            for room in world.get_semantic_annotations_by_type(Room)
            if room.floor is not start_floor
        )
        sampler = random.Random(self.seed)
        return [sampler.choice(candidates) for _ in range(self.number_of_trials)]

    def run(self) -> RoomNavigationTrialSummary:
        """
        Perform every trial, carrying on after a failed one.

        :return: The outcomes of the trials.
        """
        return RoomNavigationTrialSummary(
            [self.perform_trial(name) for name in self.target_room_names()]
        )

    def perform_trial(self, target_room_name: str) -> RoomNavigationTrialResult:
        """
        Drive the robot to one room in a fresh world.

        .. note:: Any exception the trial raises fails the trial instead of the series.

        :param target_room_name: The name of the room to drive to.
        :return: The outcome of the drive.
        """
        demonstration = TiagoRoomNavigationDemonstration(
            used_robot=self.used_robot,
            execution_type=self.execution_type,
            target_room_name=target_room_name,
        )
        try:
            demonstration.run()
        except Exception as failure:
            logging.getLogger(__name__).exception(
                f"The trial to {target_room_name} failed."
            )
            return RoomNavigationTrialResult(target_room_name, failure)
        return RoomNavigationTrialResult(target_room_name)


def main(
    number_of_trials: int = 50,
    seed: int | None = None,
    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED,
) -> None:
    """
    Run the trials and print their overview.

    :param number_of_trials: How many drives to perform.
    :param seed: Seed of the sampling of the target rooms.
    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    summary = RoomNavigationTrials(
        number_of_trials=number_of_trials, seed=seed, execution_type=execution_type
    ).run()
    print(summary.report())


if __name__ == "__main__":
    main()
