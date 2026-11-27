"""
Picking a cube up off Tracy's table, which the cube may touch until it is lifted off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.plan_transformations import LetOnlyTheObjectTouchTheGripper
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body

from ..test_plan_validation import CubeOnTheTable, cube_on_the_table  # noqa: F401
from ...plan_running import statechart_of

# %% recording what collision avoidance checks a held cube against


@dataclass
class TicksCheckingTheHeldCubeAgainst(TickingExecutorExtension):
    """
    Records the ticks at whose start a held cube is checked for collision with a body.
    """

    cube: Body
    """
    The cube picked up.
    """

    end_effector: EndEffector
    """
    The gripper that picks the cube up.
    """

    other_body: Body
    """
    The body the held cube is checked against.
    """

    ticks: list[int] = field(default_factory=list)
    """
    The ticks recorded.
    """

    def before_tick(self, executor: StatechartExecutor) -> None:
        if self.cube not in self.end_effector.held_bodies:
            return
        pair = {self.cube, self.other_body}
        collision_matrix = executor.context.world.collision_manager.collision_matrix
        if any(
            set(check.bodies()) == pair for check in collision_matrix.collision_checks
        ):
            self.ticks.append(executor.tick_count)


# %% lifting


def test_a_pick_lets_the_cube_touch_the_table_while_it_lifts_it_off(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    end_effector = scene.tracy.left_arm.end_effector
    checked_against_the_table = TicksCheckingTheHeldCubeAgainst(
        cube=scene.cube, end_effector=end_effector, other_body=scene.tracy.table.root
    )
    executor = KinematicallySimulatedPlanExecutor(
        context=StatechartContext.with_extensions(
            world=scene.world, extensions=scene.extensions
        ),
        collision_avoidance=True,
        plan_adaptations=[LetOnlyTheObjectTouchTheGripper()],
        extensions=[checked_against_the_table],
    )
    grasp_at_the_top = GraspCandidate(
        scene.top_down_grasp().graspable,
        Pose.from_xyz_rpy(
            z=scene.edge_length / 2, pitch=np.pi / 2, reference_frame=scene.cube
        ),
    )
    executor.compile(
        statechart=statechart_of(
            executor,
            PickUpAction(grasp_candidate=grasp_at_the_top, arm=scene.tracy.left_arm),
        )
    )

    executor.execute()

    assert scene.cube in end_effector.held_bodies
    assert checked_against_the_table.ticks == []
