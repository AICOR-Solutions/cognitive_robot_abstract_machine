from __future__ import annotations

import pytest

from experiments.tiago_arise_gazebo.simulated_tiago_giskard import SimulatedTiagoGiskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago import tiago_velocity

# %% fixtures


@pytest.fixture
def robot_description(monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Stands in for the robot description Tiago publishes, which the launcher reads.
    """
    description = "<robot name='tiago'/>"
    monkeypatch.setattr(tiago_velocity, "get_robot_description", lambda: description)
    return description


# %% the drive's velocity limits


def test_the_simulated_tiago_drives_within_the_limits_it_is_given(
    robot_description: str,
) -> None:
    launcher = SimulatedTiagoGiskard()

    world_config = launcher.create_giskard().world_config

    assert (
        world_config.translation_velocity_limit,
        world_config.rotation_velocity_limit,
    ) == (launcher.translation_velocity_limit, launcher.rotation_velocity_limit)


def test_the_simulated_tiago_is_otherwise_the_tiago_giskard_launches(
    robot_description: str,
) -> None:
    giskard = SimulatedTiagoGiskard().create_giskard()

    assert giskard.world_config.urdf == robot_description
    assert type(giskard.robot_interface_config) is type(
        tiago_velocity.create_giskard().robot_interface_config
    )
