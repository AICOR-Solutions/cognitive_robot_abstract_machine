"""
How Tiago asks people for help while it carries an intent out: at its tray, and when a
step fails, after which it plans anew from where it stands.

The drives are left out, so only what Tiago asks and plans is tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from speech2plan_msgs.action import CarryOutIntent
from std_skills.msg import Result as SkillResult

from coraplex.plans.executors import PlanExecutor
from experiments.intent_execution.demo import HelpRequest
from experiments.intent_execution.plan_builder import (
    DrivingStep,
    IntentRole,
    PlannableIntent,
    TrayRequest,
)
from krrood.exceptions import DataclassException
from semantic_digital_twin.world import World

from .conftest import Office, OfficeRoom

# %% drives that are only counted


@dataclass
class BlockedWay(DataclassException):
    """
    Why a drive the tests let fail did not arrive.
    """

    room: str
    """
    The name of the room the drive headed to.
    """

    def error_message(self) -> str:
        return f"The way to {self.room} is blocked."

    def suggest_correction(self) -> str:
        return "Clear the way."


@dataclass
class CountedDrives:
    """
    Stands in for Tiago's drives, failing the first ones it is told to fail.
    """

    failing: int = 0
    """
    How many of the drives to fail, the first ones.
    """

    failing_to: str | None = None
    """
    The name of the only room whose drives fail, or None for any room.
    """

    driven: list[str] = field(default_factory=list)
    """
    The rooms of the drives that arrived, in order.
    """

    def __call__(
        self, world: World, executor: PlanExecutor, step: DrivingStep
    ) -> BlockedWay | None:
        """
        Fail the drive of ``step`` if it is one to fail, else count it as arrived.

        :return: Why the drive failed, or None when it arrived.
        """
        room = step.destination.room.name.name
        if self.failing and self.failing_to in (None, room):
            self.failing -= 1
            return BlockedWay(room=room)
        self.driven.append(room)
        return None


def drive_with(
    office: Office, monkeypatch: pytest.MonkeyPatch, drives: CountedDrives
) -> CountedDrives:
    """
    Let ``drives`` stand in for the drives of the robot of ``office``.

    :return:``drives``.
    """
    monkeypatch.setattr(office.demonstration, "drive", drives)
    return drives


def bring(office: Office) -> CarryOutIntent.Result:
    """
    :return: How bringing insulin to the room upstairs ended.
    """
    return office.carry_out(
        PlannableIntent.BRING_OBJECT,
        {IntentRole.OBJECT: "insulin", IntentRole.GOAL: OfficeRoom.UPSTAIRS},
    )


def try_again_after_failing_to_reach(office: Office, room_name: str) -> str:
    """
    :return: What Tiago asks once its drive to the room of ``room_name`` failed.
    """
    drive = office.plan_builder.driving_to(office.room(room_name))
    return HelpRequest.TRY_AGAIN.verbalize(drive)


# %% at the tray


def test_a_delivery_goes_on_once_people_say_they_handled_the_tray(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drives = drive_with(office, monkeypatch, CountedDrives())

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_ENOERR
    assert office.people.texts == [
        TrayRequest.LOAD.verbalize("insulin"),
        TrayRequest.UNLOAD.verbalize("insulin"),
    ]
    assert drives.driven == [office.demonstration.pickup_room, OfficeRoom.UPSTAIRS]


def test_a_tray_nobody_loads_is_asked_about_again_then_given_up(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drives = drive_with(office, monkeypatch, CountedDrives())
    office.people.answers = [False]

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_EOTHER
    assert (
        office.people.texts
        == [TrayRequest.LOAD.verbalize("insulin")]
        * office.demonstration.maximum_tray_requests
    )
    assert outcome.records == []
    assert drives.driven == [office.demonstration.pickup_room]


def test_a_second_request_at_the_tray_may_be_answered(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_with(office, monkeypatch, CountedDrives())
    office.people.answers = [False, True]

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_ENOERR
    assert len(outcome.records) == 2


# %% when a step fails


def test_a_failed_drive_is_tried_again_once_people_say_so(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drives = drive_with(office, monkeypatch, CountedDrives(failing=1))

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_ENOERR
    pickup = office.demonstration.pickup_room
    assert office.people.texts[0] == try_again_after_failing_to_reach(office, pickup)
    assert drives.driven == [pickup, OfficeRoom.UPSTAIRS]


def test_an_item_on_the_tray_is_not_collected_again_after_a_failure(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drives = drive_with(
        office, monkeypatch, CountedDrives(failing=1, failing_to=OfficeRoom.UPSTAIRS)
    )

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_ENOERR
    assert office.people.texts == [
        TrayRequest.LOAD.verbalize("insulin"),
        try_again_after_failing_to_reach(office, OfficeRoom.UPSTAIRS),
        TrayRequest.UNLOAD.verbalize("insulin"),
    ]
    assert drives.driven == [office.demonstration.pickup_room, OfficeRoom.UPSTAIRS]


def test_people_saying_no_end_the_task(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_with(office, monkeypatch, CountedDrives(failing=1))
    office.people.answers = [False]

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_EOTHER
    assert len(office.people.asked) == 1


def test_a_step_failing_every_time_is_given_up_after_the_last_retry(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_with(office, monkeypatch, CountedDrives(failing=100))

    outcome = bring(office)

    assert outcome.result.error_code == SkillResult.ROS_EOTHER
    assert len(office.people.asked) == office.demonstration.maximum_retries


def test_without_anyone_to_ask_a_failed_step_ends_the_task(
    office: Office, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_with(office, monkeypatch, CountedDrives(failing=1))
    office.demonstration.help_skill_path = "/skill/served_by_nobody"
    office.demonstration.help_server_timeout = (
        office.demonstration.help_server_timeout / 5
    )

    outcome = office.carry_out(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: OfficeRoom.UPSTAIRS}
    )

    assert outcome.result.error_code == SkillResult.ROS_EOTHER
    assert office.people.asked == []
