"""
Tests of where a hand with more than two fingers holds an object, and of the goals that
put that point on a grasp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from typing_extensions import Self

from krrood.ormatic.utils import classproperty

from krrood_spatial_math.spatial_types import (
    Point3,
    Pose,
    Vector3,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasEndEffector,
    HasFingers,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    EndEffector,
    Finger,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% a hand with a thumb and two fingers


@dataclass(eq=False)
class FingerOfTheHand(Finger, ABC):
    """
    A finger of the three-fingered hand, from the palm to the tip link its subclass
    names.
    """

    @classproperty
    @abstractmethod
    def tip_link_name(cls) -> str:
        """
        The name of the link at the end of this finger.
        """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, "palm"),
            tip=world.get_body_in_branch_by_name(robot_root, cls.tip_link_name),
        )

    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []


@dataclass(eq=False)
class ThumbOfTheHand(FingerOfTheHand):
    """
    The thumb, standing alone on one side of the palm.
    """

    @classproperty
    def tip_link_name(cls) -> str:
        return "thumb_tip"


@dataclass(eq=False)
class IndexFingerOfTheHand(FingerOfTheHand):
    """
    One of the two fingers opposing the thumb.
    """

    @classproperty
    def tip_link_name(cls) -> str:
        return "index_tip"


@dataclass(eq=False)
class MiddleFingerOfTheHand(FingerOfTheHand):
    """
    The other finger opposing the thumb.
    """

    @classproperty
    def tip_link_name(cls) -> str:
        return "middle_tip"


@dataclass(eq=False)
class ThreeFingeredHand(
    EndEffector,
    HasFingers[ThumbOfTheHand, IndexFingerOfTheHand, MiddleFingerOfTheHand],
):
    """
    A hand approaching along its palm's normal, its fingers closing along the palm's x
    axis, the thumb curling further than the fingers when it closes.
    """

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, "palm"),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "tool_frame"),
        )

    def setup_hardware_interfaces(self) -> None:
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
        thumb, index, middle = self.active_connections
        return [
            JointState.from_mapping(
                name=PrefixedName("open", prefix=self.name.name),
                mapping={thumb: 0.0, index: 0.0, middle: 0.0},
                state_type=GripperState.OPEN,
            ),
            JointState.from_mapping(
                name=PrefixedName("close", prefix=self.name.name),
                mapping={thumb: 0.8, index: -0.4, middle: -0.4},
                state_type=GripperState.CLOSE,
            ),
        ]


@dataclass(eq=False)
class HandOnItsOwn(AbstractRobot, HasEndEffector[ThreeFingeredHand]):
    """
    A robot that is nothing but the three-fingered hand.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return str(
            Path(files("semantic_digital_twin")).parent.parent
            / "resources"
            / "urdf"
            / "three_fingered_hand.urdf"
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "palm"

    def _setup_collision_rules(self) -> None:
        pass


@pytest.fixture
def hand() -> ThreeFingeredHand:
    world = URDFParser.from_file(HandOnItsOwn.get_ros_file_path()).create_world()
    return HandOnItsOwn.from_world(world).end_effector


# %% where the hand holds an object


def tool_P_link_origin(hand: ThreeFingeredHand, link: Body) -> np.ndarray:
    """
    :return: Where a link's origin lies in the hand's tool frame, as the hand stands.
    """
    return hand._world.compute_forward_kinematics_np(hand.tool_frame, link)[:3, 3]


def test_a_hand_holds_an_object_midway_between_its_thumb_and_its_other_fingers(
    hand: ThreeFingeredHand,
):
    """
    Closed, the thumb meets the other fingers around a point between its tip and the
    middle of theirs; the tips' boxes are centred on their links' origins.
    """
    world = hand._world
    with world.reset_state_context():
        hand.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)
        thumb = tool_P_link_origin(hand, hand.thumb.tip)
        others = np.mean(
            [
                tool_P_link_origin(hand, finger.tip)
                for finger in hand.fingers
                if finger is not hand.thumb
            ],
            axis=0,
        )

    assert hand.tool_P_grasp_when_closed.to_np()[:3] == pytest.approx(
        (thumb + others) / 2
    )


def test_the_grasp_point_is_read_with_the_hand_closed_whatever_it_stands_in(
    hand: ThreeFingeredHand,
):
    world = hand._world
    hand.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    standing_open = hand.tool_P_grasp_when_closed.to_np()
    del hand.tool_P_grasp_when_closed
    hand.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)

    assert hand.tool_P_grasp_when_closed.to_np() == pytest.approx(standing_open)


# %% reaching and reading back a grasp


def test_the_tool_frame_goal_puts_the_grasp_point_on_the_grasp(
    hand: ThreeFingeredHand,
):
    world = hand._world
    grasp = Pose.from_xyz_rpy(
        x=0.4, y=-0.1, z=0.3, roll=0.2, pitch=1.1, reference_frame=world.root
    )

    goal = hand.tool_frame_goal_for_grasp(grasp)

    reached = goal.homogeneous_matrix @ Point3(
        *hand.tool_P_grasp_when_closed.to_np()[:3], reference_frame=hand.tool_frame
    )
    assert reached.to_np()[:3] == pytest.approx(grasp.to_np()[:3, 3])
    assert goal.to_np()[:3, :3] == pytest.approx(
        hand.tool_frame_goal(grasp).to_np()[:3, :3]
    )


def test_a_held_body_is_read_back_at_the_grasp_it_was_taken_by(
    hand: ThreeFingeredHand,
):
    """
    A body attached where the goal for a grasp puts the tool frame is held by that very
    grasp.
    """
    world = hand._world
    held = Body(
        name=PrefixedName("held"),
        collision=ShapeCollection([Box(scale=Scale(0.03, 0.03, 0.03))]),
    )
    body_T_grasp = Pose.from_xyz_rpy(x=0.01, z=0.015, pitch=np.pi / 2)
    body_T_tool = hand.tool_frame_goal_for_grasp(body_T_grasp)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=hand.tool_frame,
                child=held,
                parent_T_connection_expression=body_T_tool.homogeneous_matrix.inverse(),
            )
        )

    assert hand.held_body_T_grasp.to_np() == pytest.approx(
        body_T_grasp.to_np(), abs=1e-9
    )
