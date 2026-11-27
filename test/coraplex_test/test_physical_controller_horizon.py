"""
Tests of the controller a plan performed in physics is driven by: its prediction horizon
covers the braking of every joint of the robot.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from typing_extensions import Iterator

from giskardpy.qp.dof_limits import DegreeOfFreedomLimitProfiler
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.derivatives import DerivativeMap

from .conftest import physically_simulated_executor

SLOW_BRAKING_ACCELERATION = 15.0
"""
An acceleration limit, in radians per second squared, under which a joint of Tracy
brakes from its velocity limit in more steps than the physical defaults leave.
"""

# %% the prediction horizon


def test_the_horizon_is_the_physical_default_where_every_joint_brakes_within_it(
    tracy_context,
):
    world, robot, extensions = tracy_context

    configuration = physically_simulated_executor(
        extensions
    ).create_qp_controller_config()

    assert (
        configuration.prediction_horizon
        == QPControllerConfig.create_with_physical_simulation_defaults().prediction_horizon
    )


@pytest.fixture
def slowly_braking_joint(tracy_context) -> Iterator[DegreeOfFreedom]:
    """
    :return: A joint of Tracy limited to :data:`SLOW_BRAKING_ACCELERATION`, its own
        limits restored after the test.
    """
    world, robot, extensions = tracy_context
    [degree_of_freedom, *_] = robot.degrees_of_freedom_with_hardware_interface
    lower = deepcopy(degree_of_freedom.limits.lower)
    upper = deepcopy(degree_of_freedom.limits.upper)
    degree_of_freedom._overwrite_dof_limits(
        new_lower_limits=DerivativeMap(None, None, -SLOW_BRAKING_ACCELERATION, None),
        new_upper_limits=DerivativeMap(None, None, SLOW_BRAKING_ACCELERATION, None),
    )
    yield degree_of_freedom
    degree_of_freedom.limits.lower = lower
    degree_of_freedom.limits.upper = upper


def test_the_horizon_covers_the_braking_of_a_joint_with_a_low_acceleration_limit(
    tracy_context, slowly_braking_joint
):
    world, robot, extensions = tracy_context
    defaults = QPControllerConfig.create_with_physical_simulation_defaults()
    braking = DegreeOfFreedomLimitProfiler(qp_controller_config=defaults).braking(
        degree_of_freedom=slowly_braking_joint
    )

    configuration = physically_simulated_executor(
        extensions
    ).create_qp_controller_config()

    assert braking.number_of_steps > defaults.control_horizon
    assert configuration.control_horizon == braking.number_of_steps
    assert configuration.control_time_step == defaults.control_time_step
