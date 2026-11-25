"""
How the plan of each intent is built, and what it does when performed in the IAI office.
"""

from __future__ import annotations

import pytest

from experiments.intent_execution.exceptions import (
    EntityInNoRoom,
    IntentNotPlannable,
    RoleNotGrounded,
)
from experiments.intent_execution.plan_builder import (
    DrivingStep,
    IntentRole,
    PlannableIntent,
    TrayRequest,
    TrayStep,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Level,
    Table,
)
from semantic_digital_twin.world_description.geometry import Scale

from .conftest import Office, OfficeRoom

# %% planning


def test_an_intent_no_plan_is_built_for_is_not_planned(office: Office) -> None:
    with pytest.raises(IntentNotPlannable):
        office.plan_builder.plan_for("__intent_greet__", {})


def test_a_bring_naming_no_item_is_not_planned(office: Office) -> None:
    with pytest.raises(RoleNotGrounded):
        office.plan_builder.plan_for(
            PlannableIntent.BRING_OBJECT, {IntentRole.GOAL: OfficeRoom.UPSTAIRS}
        )


def test_a_bring_naming_no_goal_is_not_planned(office: Office) -> None:
    with pytest.raises(RoleNotGrounded):
        office.plan_builder.plan_for(
            PlannableIntent.BRING_OBJECT, {IntentRole.OBJECT: "insulin"}
        )


def test_an_entity_over_no_room_is_in_no_room(office: Office) -> None:
    outside = Table.create_with_new_body_in_world(
        name="table_outside",
        world=office.world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=50.0),
        scale=Scale(0.5, 0.5, 0.75),
    )

    with pytest.raises(EntityInNoRoom):
        office.plan_builder.room_of(outside)


def test_a_move_to_a_room_drives_there(office: Office) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)

    [drive] = office.plan_builder.plan_for(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: room.name.name}
    ).steps

    assert isinstance(drive, DrivingStep)
    assert drive.destination.room is room
    assert drive.destination.spoken_name == office.spoken_name_of(room)


def test_a_move_to_a_room_named_with_its_prefix_drives_there(office: Office) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)

    [drive] = office.plan_builder.plan_for(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: str(room.name)}
    ).steps

    assert drive.destination.room is room


def test_a_move_to_a_room_named_as_people_speak_of_it_drives_there(
    office: Office,
) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)

    [drive] = office.plan_builder.plan_for(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: office.spoken_name_of(room)}
    ).steps

    assert drive.destination.room is room


def test_heading_to_a_room_on_another_storey_sends_the_elevator_there(
    office: Office,
) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)
    [drive] = office.plan_builder.plan_for(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: room.name.name}
    ).steps

    office.demonstration.head_to(drive.destination, office.world)

    assert office.demonstration.elevator_operator.floor is Level.under(room.floor.root)


def test_a_bring_collects_the_item_in_the_pickup_room_and_delivers_it(
    office: Office,
) -> None:
    pickup = office.room(office.demonstration.pickup_room)
    goal = office.room(OfficeRoom.UPSTAIRS)

    to_pickup, loading, to_goal, unloading = office.plan_builder.plan_for(
        PlannableIntent.BRING_OBJECT,
        {IntentRole.OBJECT: "insulin", IntentRole.GOAL: goal.name.name},
    ).steps

    assert isinstance(to_pickup, DrivingStep) and isinstance(to_goal, DrivingStep)
    assert to_pickup.destination.room is pickup
    assert to_goal.destination.room is goal
    assert isinstance(loading, TrayStep) and isinstance(unloading, TrayStep)
    assert (loading.room, loading.unloading) == (pickup, False)
    assert (unloading.room, unloading.unloading) == (goal, True)
    assert unloading.record == f"delivered the insulin to {goal.name}"


def test_the_robot_asks_for_the_item_by_the_speakers_word(office: Office) -> None:
    _, loading, _, unloading = office.plan_builder.plan_for(
        PlannableIntent.BRING_OBJECT,
        {IntentRole.OBJECT: "blood samples", IntentRole.GOAL: OfficeRoom.UPSTAIRS},
    ).steps

    assert loading.request == TrayRequest.LOAD.verbalize("blood samples")
    assert unloading.request == TrayRequest.UNLOAD.verbalize("blood samples")


def test_a_bring_with_the_item_collected_goes_straight_to_the_goal(
    office: Office,
) -> None:
    goal = office.room(OfficeRoom.UPSTAIRS)

    to_goal, unloading = office.plan_builder.plan_for(
        PlannableIntent.BRING_OBJECT,
        {IntentRole.OBJECT: "insulin", IntentRole.GOAL: goal.name.name},
        collected=True,
    ).steps

    assert to_goal.destination.room is goal
    assert unloading.unloading


# %% performing the plans


def test_a_robot_sent_to_a_room_two_storeys_up_by_its_spoken_name_stands_in_it(
    office: Office,
) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)

    office.carry_out_without_error(
        PlannableIntent.MOVE_TO, {IntentRole.GOAL: office.spoken_name_of(room)}
    )

    assert Floor.under(office.robot.root) is room.floor


def test_a_delivered_item_leaves_the_robot_in_the_goal_room(office: Office) -> None:
    room = office.room(OfficeRoom.UPSTAIRS)

    outcome = office.carry_out_without_error(
        PlannableIntent.BRING_OBJECT,
        {IntentRole.OBJECT: "insulin", IntentRole.GOAL: room.name.name},
    )

    assert Floor.under(office.robot.root) is room.floor
    assert office.people.texts == [
        TrayRequest.LOAD.verbalize("insulin"),
        TrayRequest.UNLOAD.verbalize("insulin"),
    ]
    assert outcome.records == [
        f"collected the insulin in {office.room(office.demonstration.pickup_room).name}",
        f"delivered the insulin to {room.name}",
    ]
