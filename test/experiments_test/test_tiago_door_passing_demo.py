from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from coraplex.execution_environment import kinematically_simulated_robot
from experiments.tiago_apartment_door_demo.demo import TiagoDoorPassingDemonstration
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% recording how far the robot gets through the doorway


@dataclass(eq=False)
class DoorwayProgressRecorder(StateChangeCallback):
    """
    Records the closest the robot came to the far side of a door, measured along the
    door frame's x-axis, which points from the far side into the room.
    """

    robot_root: Body = field(kw_only=True)
    """
    The body whose position is recorded.
    """

    door_frame: Body = field(kw_only=True)
    """
    The frame the position is measured in.
    """

    smallest_distance_along_door_axis: float = field(init=False, default=np.inf)
    """
    The smallest x the robot reached in :attr:`door_frame`.
    """

    def on_state_change(self, **kwargs) -> None:
        distance = float(
            self._world.transform(self.robot_root.global_transform, self.door_frame).x
        )
        self.smallest_distance_along_door_axis = min(
            self.smallest_distance_along_door_axis, distance
        )


# %% the scene


@pytest.fixture
def demonstration() -> TiagoDoorPassingDemonstration:
    return TiagoDoorPassingDemonstration(used_robot=Tiago)


@pytest.fixture
def populated_world(apartment_meshes, demonstration) -> World:
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return world


def test_the_scene_counts_as_populated_once_the_door_is_spawned(
    apartment_meshes, demonstration
):
    world = World.create_with_root_body()
    assert not demonstration.is_scene_populated(world)

    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_the_door_sits_in_the_wall_opening(demonstration, populated_world):
    door = populated_world.get_semantic_annotation_by_name(demonstration.door_name)

    np.testing.assert_allclose(
        door.entry_way.root.global_transform.to_np(),
        demonstration.world_T_door.to_np(),
        atol=1e-9,
    )


def test_the_door_leaf_is_open(demonstration, populated_world):
    door = populated_world.get_semantic_annotation_by_name(demonstration.door_name)

    assert door.movable_joint.position == demonstration.door_scale.y


# %% passing the door


def test_the_tiago_passes_the_door_and_comes_back(demonstration, populated_world):
    robot = populated_world.get_semantic_annotations_by_type(Tiago)[0]
    door = populated_world.get_semantic_annotation_by_name(demonstration.door_name)
    start = robot.root.global_transform.to_np()
    recorder = DoorwayProgressRecorder(
        _world=populated_world, robot_root=robot.root, door_frame=door.root
    )
    plan = demonstration.build_plan(demonstration.build_context(populated_world))

    with kinematically_simulated_robot:
        plan.perform()
    recorder.stop()

    assert recorder.smallest_distance_along_door_axis == pytest.approx(
        -demonstration.distance_from_door, abs=0.05
    )
    np.testing.assert_allclose(
        robot.root.global_transform.to_np()[:3, 3], start[:3, 3], atol=0.05
    )
