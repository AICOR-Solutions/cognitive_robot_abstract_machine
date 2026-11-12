"""
Checking a plan of Tracy's before it runs: its steps are tried one after another on a
copy of the world, a pick the robot cannot reach fails as unreachable, an arm left open
is the one that reaches, and a detection is answered as the plan builder's source rule
says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from coraplex.datastructures.enums import DetectionGrounding
from coraplex.exceptions import NothingPerceived, UnidentifiedDetections
from coraplex.perception import (
    Detection,
    PerceptionInterface,
    PerceptionQuery,
    WorldPerception,
)
from coraplex.plans.failures import (
    MotionViolatedCollisionAvoidance,
    MotionWouldBreakSafetyLimit,
    PlanFailure,
    RobotInCollision,
    TargetLeftFieldOfView,
)
from coraplex.plans.validation import FailureCause, PlanValidation, ValidationVerdict
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import StatementNode
from cramph.context import ContextExtension
from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    SafetyConstraintViolated,
)
from giskardpy.motion_statechart.monitors.safety_monitors import (
    SafetyEnvelope,
    WorkspaceBox,
)
from krrood.entity_query_language.factories import a, variable
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Arm, Camera
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% a cube on Tracy's table


@dataclass
class CubeOnTheTable:
    """
    Tracy, its arms parked, and one cube on its table.
    """

    world: World
    """
    The world holding Tracy and the cube.
    """

    tracy: Tracy
    """
    The robot.
    """

    cube: Body
    """
    The cube.
    """

    extensions: list[ContextExtension]
    """
    The context extensions of Tracy's plans.
    """

    edge_length: float = 0.05
    """
    Edge length of the cube, in metres.
    """

    def put_cube_at(self, x: float, y: float) -> None:
        """
        :param x: Where the cube's centre stands in the world frame, in metres.
        :param y: Where the cube's centre stands in the world frame, in metres.
        """
        self.cube.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x,
                y=y,
                z=self.tracy.table.top_z + self.edge_length / 2,
                reference_frame=self.world.root,
            )
        )

    def validation(self) -> PlanValidation:
        """
        :return: A validation of a plan of Tracy in the world, perceiving from the world
            model.
        """
        return PlanValidation(
            world=self.world, extensions=self.extensions, perception=WorldPerception()
        )

    def top_down_grasp(self) -> GraspCandidate:
        """
        :return: A grasp reaching down onto the cube, centred over its origin.
        """
        [graspable] = [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root is self.cube
        ]
        return GraspCandidate(
            graspable, Pose.from_xyz_rpy(pitch=np.pi / 2, reference_frame=self.cube)
        )

    def table_region(self) -> VolumetricBoundingBox:
        """
        :return: The region above Tracy's table, holding the cube.
        """
        top = self.tracy.table.top_z
        return VolumetricBoundingBox(
            origin=HomogeneousTransformationMatrix(reference_frame=self.world.root),
            min_x=0.3,
            min_y=-0.4,
            min_z=top - 0.05,
            max_x=0.9,
            max_y=0.4,
            max_z=top + 0.2,
        )


@pytest.fixture
def cube_on_the_table(tracy_context) -> CubeOnTheTable:
    """
    :return: Tracy with its arms parked and a graspable cube in reach of both arms.
    """
    world, tracy, extensions = tracy_context
    for arm in tracy.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    scene = CubeOnTheTable(
        world=world,
        tracy=tracy,
        cube=Body(
            name=PrefixedName("cube"),
            visual=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
            collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        ),
        extensions=extensions,
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=scene.cube
            )
        )
        world.add_semantic_annotation(HasGraspCandidates(root=scene.cube))
    scene.put_cube_at(0.55, 0.1)
    return scene


def joint_positions(world: World) -> np.ndarray:
    """
    :return: The position of every degree of freedom of the world.
    """
    return world.state.positions.copy()


# %% causes


@pytest.mark.parametrize(
    "failure, cause",
    [
        (
            MotionViolatedCollisionAvoidance(
                CollisionViolatedError(violated_collisions=[])
            ),
            FailureCause.COLLISION,
        ),
        (RobotInCollision(), FailureCause.COLLISION),
        (NothingPerceived(HasGraspCandidates), FailureCause.NOT_PERCEIVED),
        (UnidentifiedDetections(HasGraspCandidates, 2), FailureCause.AMBIGUOUS),
        (
            TargetLeftFieldOfView(
                camera=Camera(
                    root=Body(name=PrefixedName("optical_frame")),
                    camera_model=FieldOfViewCameraModel(
                        angular_extent=FieldOfView(
                            horizontal_angle=1.0, vertical_angle=0.8
                        )
                    ),
                ),
                target=Body(name=PrefixedName("cube")),
            ),
            FailureCause.OUT_OF_VIEW,
        ),
        (
            MotionWouldBreakSafetyLimit(
                SafetyConstraintViolated(
                    constraint="Left arm workspace", violation="1.0 mm past"
                )
            ),
            FailureCause.SAFETY_LIMIT,
        ),
        (PlanFailure(), FailureCause.UNSUCCESSFUL),
    ],
)
def test_a_failure_names_the_cause_a_person_reads(failure, cause):
    assert FailureCause.from_failure(failure) is cause


# %% motions


def test_a_pick_in_reach_is_valid_and_leaves_the_world_where_it_was(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    before = joint_positions(scene.world)
    validation = scene.validation()
    copy = validation.copied_world

    check = validation.check(
        PickUpAction(
            grasp=copy.rebind_world_entities(scene.top_down_grasp()),
            arm=copy.rebind_world_entities(scene.tracy.left_arm),
        )
    )

    assert check.verdict is ValidationVerdict.VALID
    assert check.cause is None
    np.testing.assert_array_equal(joint_positions(scene.world), before)
    assert scene.cube.parent_connection.parent is scene.world.root


def test_a_pick_out_of_reach_fails_as_unreachable(cube_on_the_table: CubeOnTheTable):
    scene = cube_on_the_table
    scene.put_cube_at(1.5, 0.1)
    validation = scene.validation()
    copy = validation.copied_world

    check = validation.check(
        PickUpAction(
            grasp=copy.rebind_world_entities(scene.top_down_grasp()),
            arm=copy.rebind_world_entities(scene.tracy.left_arm),
        )
    )

    assert check.verdict is ValidationVerdict.FAILED
    assert check.cause is FailureCause.UNREACHABLE


def test_a_step_starts_where_the_step_before_it_ended(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = scene.validation()
    copy = validation.copied_world
    grasp = copy.rebind_world_entities(scene.top_down_grasp())
    validation.check(
        PickUpAction(grasp=grasp, arm=copy.rebind_world_entities(scene.tracy.left_arm))
    )

    check = validation.check(
        PlaceAction(
            object_designator=grasp.graspable,
            target_location=Pose.from_xyz_rpy(
                x=0.6,
                y=-0.1,
                z=scene.tracy.table.top_z + 0.025,
                reference_frame=copy.root,
            ),
        )
    )

    assert check.verdict is ValidationVerdict.VALID


def test_a_step_given_as_an_action_runs_as_itself(cube_on_the_table: CubeOnTheTable):
    scene = cube_on_the_table
    validation = scene.validation()
    tracy = validation.copied_world.rebind_world_entities(scene.tracy)
    park = ParkArmsAction(arms=list(tracy.all_arms))

    check = validation.check(park)

    assert check.verdict is ValidationVerdict.VALID
    assert check.grounded_action is park


def workspace_below(
    validation: PlanValidation, scene: CubeOnTheTable, height: float
) -> SafetyEnvelope:
    """
    :return: The left arm's tool held below a height above Tracy's table, in the copy.
    """
    copy = validation.copied_world
    tool = copy.rebind_world_entities(scene.tracy.left_arm).end_effector.tool_frame
    return SafetyEnvelope(
        constraints=[
            WorkspaceBox(
                constraint="Left arm workspace",
                tool=tool,
                minimum=Point3(-5.0, -5.0, 0.0, reference_frame=copy.root),
                maximum=Point3(
                    5.0,
                    5.0,
                    scene.tracy.table.top_z + height,
                    reference_frame=copy.root,
                ),
            )
        ]
    )


def test_a_step_whose_tool_leaves_a_workspace_box_fails_as_breaking_a_safety_limit(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = scene.validation()
    copy = validation.copied_world

    check = validation.check(
        PickUpAction(
            grasp=copy.rebind_world_entities(scene.top_down_grasp()),
            arm=copy.rebind_world_entities(scene.tracy.left_arm),
        ),
        safety=workspace_below(validation, scene, 0.02),
    )

    assert check.verdict is ValidationVerdict.FAILED
    assert check.cause is FailureCause.SAFETY_LIMIT
    assert "Left arm workspace" in check.failure.error_message()


def test_a_step_whose_tool_stays_inside_a_workspace_box_is_valid(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = scene.validation()
    copy = validation.copied_world

    check = validation.check(
        PickUpAction(
            grasp=copy.rebind_world_entities(scene.top_down_grasp()),
            arm=copy.rebind_world_entities(scene.tracy.left_arm),
        ),
        safety=workspace_below(validation, scene, 2.0),
    )

    assert check.verdict is ValidationVerdict.VALID


# %% choosing the arm


def any_arm_picking(validation: PlanValidation, scene: CubeOnTheTable) -> StatementNode:
    """
    :return: A pick of the cube by whichever of Tracy's arms reaches it, in the copy.
    """
    copy = validation.copied_world
    tracy = copy.rebind_world_entities(scene.tracy)
    return StatementNode(
        statement=a(PickUpAction)(
            arm=variable(Arm, domain=tracy.all_arms),
            grasp=copy.rebind_world_entities(scene.top_down_grasp()),
        )
    )


def test_an_arm_left_open_is_the_one_that_reaches(cube_on_the_table: CubeOnTheTable):
    scene = cube_on_the_table
    scene.put_cube_at(0.55, -1.2)
    validation = scene.validation()

    check = validation.check(any_arm_picking(validation, scene))

    assert check.verdict is ValidationVerdict.VALID
    assert check.grounded_action.arm is validation.copied_world.rebind_world_entities(
        scene.tracy.right_arm
    )


def test_an_arm_left_open_that_no_arm_reaches_fails_as_unreachable(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    scene.put_cube_at(1.5, 0.1)
    validation = scene.validation()

    check = validation.check(any_arm_picking(validation, scene))

    assert check.verdict is ValidationVerdict.FAILED
    assert check.cause is FailureCause.UNREACHABLE
    assert len(check.candidate_failures) == len(scene.tracy.all_arms)


# %% detecting


def cube_query(
    validation: PlanValidation, scene: CubeOnTheTable, region: VolumetricBoundingBox
) -> PerceptionQuery:
    """
    :return: A query for a graspable object in the region, in the copy.
    """
    copy = validation.copied_world
    return PerceptionQuery(
        HasGraspCandidates,
        copy.rebind_world_entities(region),
        copy.rebind_world_entities(scene.tracy),
        copy,
    )


def test_a_detection_the_camera_sees_is_valid_and_finds_the_cube(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = scene.validation()

    check = validation.check_detection(
        cube_query(validation, scene, scene.table_region())
    )

    assert check.verdict is ValidationVerdict.VALID
    assert check.found == [validation.copied_world.rebind_world_entities(scene.cube)]


def test_a_detection_out_of_the_cameras_view_is_assumed(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    behind = VolumetricBoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=scene.world.root),
        min_x=-3.0,
        min_y=-0.5,
        min_z=0.0,
        max_x=-2.0,
        max_y=0.5,
        max_z=1.0,
    )
    scene.put_cube_at(-2.5, 0.0)
    validation = scene.validation()

    check = validation.check_detection(cube_query(validation, scene, behind))

    assert check.verdict is ValidationVerdict.ASSUMED
    assert check.grounding is DetectionGrounding.ASSUMED


def test_a_detection_finding_nothing_fails_as_not_perceived(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    scene.put_cube_at(1.5, 0.1)
    validation = scene.validation()

    check = validation.check_detection(
        cube_query(validation, scene, scene.table_region())
    )

    assert check.verdict is ValidationVerdict.FAILED
    assert check.cause is FailureCause.NOT_PERCEIVED


# %% a cube seen inside the table


@dataclass
class PerceptionSeeingTheCubeSunk(PerceptionInterface):
    """
    Reports the cube upright where it stands, but its centre lower by a depth, as a
    camera whose calibration is a few millimetres off sees a cube on the table.
    """

    scene: CubeOnTheTable
    """
    The scene whose cube is seen.
    """

    depth: float
    """
    How far below its resting height the cube's centre is reported, in metres.
    """

    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        resting = query.world.rebind_world_entities(self.scene.cube).global_pose
        return [
            Detection(
                semantic_annotation=HasGraspCandidates,
                pose=Pose.from_xyz_rpy(
                    x=float(resting.x),
                    y=float(resting.y),
                    z=float(resting.z) - self.depth,
                    reference_frame=query.world.root,
                ),
            )
        ]


def sunk_cube_validation(
    scene: CubeOnTheTable, depth: float, collision_avoidance: bool = False
) -> PlanValidation:
    """
    :return: A validation of a plan of Tracy perceiving the cube `depth` metres lower
        than it rests, avoiding collisions as a run does where `collision_avoidance`.
    """
    return PlanValidation(
        world=scene.world,
        extensions=scene.extensions,
        perception=PerceptionSeeingTheCubeSunk(scene=scene, depth=depth),
        collision_avoidance=collision_avoidance,
    )


def lowest_height(body: Body, world: World) -> float:
    """
    :return: The height of the body's lowest point in `world`.
    """
    return float(
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
    )


def test_a_cube_seen_a_few_millimetres_inside_the_table_moves_clear_once_held(
    cube_on_the_table: CubeOnTheTable,
):
    """
    A pick puts its cube under the tool frame once the gripper closed; a cube the
    detection left inside the table is then part of the robot inside its table, which
    collision avoidance stops the next motion for, as on the real Tracy on 2026-11-12.
    """
    scene = cube_on_the_table
    validation = sunk_cube_validation(scene, depth=0.004, collision_avoidance=True)
    copy = validation.copied_world
    tracy = copy.rebind_world_entities(scene.tracy)
    validation.check_detection(cube_query(validation, scene, scene.table_region()))
    copy.move_branch(
        copy.rebind_world_entities(scene.cube), tracy.left_arm.end_effector.tool_frame
    )

    check = validation.check(ParkArmsAction(arms=list(tracy.all_arms)))

    assert check.verdict is ValidationVerdict.VALID, check.failure


def test_a_cube_seen_a_few_millimetres_inside_the_table_is_raised_onto_it(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = sunk_cube_validation(scene, depth=0.004)

    check = validation.check_detection(
        cube_query(validation, scene, scene.table_region())
    )

    assert check.verdict is ValidationVerdict.VALID
    [cube] = check.found
    clearance = Detection(HasGraspCandidates, Pose()).clearance_above_surface
    assert lowest_height(cube, validation.copied_world) == pytest.approx(
        scene.tracy.table.top_z + clearance, abs=1e-6
    )


def test_a_cube_seen_above_the_table_stays_where_it_was_seen(
    cube_on_the_table: CubeOnTheTable,
):
    scene = cube_on_the_table
    validation = sunk_cube_validation(scene, depth=-0.003)

    check = validation.check_detection(
        cube_query(validation, scene, scene.table_region())
    )

    [cube] = check.found
    assert lowest_height(cube, validation.copied_world) == pytest.approx(
        scene.tracy.table.top_z + 0.003, abs=1e-6
    )


def test_a_cube_seen_deep_inside_the_table_is_raised_onto_it(
    cube_on_the_table: CubeOnTheTable,
):
    """
    A cube cannot be inside the table, however deep the camera reports it there.
    """
    scene = cube_on_the_table
    validation = sunk_cube_validation(scene, depth=0.04)

    check = validation.check_detection(
        cube_query(validation, scene, scene.table_region())
    )

    assert check.verdict is ValidationVerdict.VALID
    [cube] = check.found
    clearance = Detection(HasGraspCandidates, Pose()).clearance_above_surface
    assert lowest_height(cube, validation.copied_world) == pytest.approx(
        scene.tracy.table.top_z + clearance, abs=1e-6
    )
