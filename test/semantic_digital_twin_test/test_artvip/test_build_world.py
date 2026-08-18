import math

import numpy as np
import pytest

from semantic_digital_twin.adapters.artvip_dataset.exceptions import (
    ArtVipJointMissingChildBodyError,
    ArtVipUnsupportedJointTypeError,
)
from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.world_entity import Body

from .usd_stages import (
    PXR_AVAILABLE,
    build_single_joint_stage,
    build_stage_with_joint_missing_body1,
    build_stage_with_mesh_targeted_body0,
    build_stage_with_scaled_mesh,
)

if PXR_AVAILABLE:
    from pxr import Gf

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


def test_build_world_connects_a_revolute_joint():
    stage = build_single_joint_stage("RevoluteJoint")
    loader = ArtVipDatasetLoader()
    world = loader._build_world(stage, ArtVipCategory.SMALL_FURNITURE, "test_object")

    assert len(world.bodies) == 2  # root + child
    [connection] = world.connections
    assert isinstance(connection, RevoluteConnection)


def test_build_world_raises_on_an_unsupported_joint_type_instead_of_building_an_incomplete_world():
    # Before the fix, an unrecognized joint type name was silently skipped like any
    # other non-joint prim, so a stage using only unsupported joints built a World with
    # no bodies but the root - not an error, just quietly missing every link.
    stage = build_single_joint_stage("SphericalJoint")
    loader = ArtVipDatasetLoader()

    with pytest.raises(ArtVipUnsupportedJointTypeError) as excinfo:
        loader._build_world(stage, ArtVipCategory.SMALL_FURNITURE, "test_object")

    assert excinfo.value.joint_type == "PhysicsSphericalJoint"


def test_build_world_swaps_inverted_revolute_limits():
    # Seen on a real ArtVIP object (one blade of a pair of scissors): the USD data
    # itself authors lower > upper. Before the fix this reached
    # DegreeOfFreedomLimits with lower > upper and broke the world's modification
    # history instead of building.
    stage = build_single_joint_stage("RevoluteJoint", lower_limit=30.0, upper_limit=0.0)
    loader = ArtVipDatasetLoader()
    world = loader._build_world(stage, ArtVipCategory.SMALL_FURNITURE, "test_object")

    [connection] = world.connections
    assert connection.raw_dof.limits.lower.position == pytest.approx(0.0)
    assert connection.raw_dof.limits.upper.position == pytest.approx(math.radians(30.0))


def test_build_world_resolves_a_mesh_targeted_body0_to_its_enclosing_link():
    # Before the fix, body0 pointing directly at a link's mesh prim (rather than its
    # enclosing Xform) created a second, disconnected Body for the same physical link -
    # reproduced on real ArtVIP basket objects, which built a World with two roots.
    stage = build_stage_with_mesh_targeted_body0()
    loader = ArtVipDatasetLoader()
    world = loader._build_world(stage, ArtVipCategory.SMALL_FURNITURE, "test_object")

    assert len(world.bodies) == 3  # root + carcass + handle
    assert world.root is not None


def test_build_world_raises_on_a_joint_with_no_body1_target():
    # Before the fix, a joint's body1 relationship having no target - unlike body0,
    # this has no "object's own frame" meaning - crashed on an unguarded
    # body1_targets[0] with an IndexError instead of a clear, actionable exception.
    stage = build_stage_with_joint_missing_body1()
    loader = ArtVipDatasetLoader()

    with pytest.raises(ArtVipJointMissingChildBodyError) as excinfo:
        loader._build_world(stage, ArtVipCategory.SMALL_FURNITURE, "test_object")

    assert excinfo.value.joint_path == "/object/joint"


def test_create_mesh_shape_applies_a_non_uniform_scale():
    # Before the fix, the mesh's local-to-link transform was decomposed into a
    # translation and rotation only (Gf.Transform's own rotation/translation
    # accessors), silently dropping any scale - so a mesh under a scaled Xform (seen on
    # real ArtVIP scene props, e.g. a fruit platter's leaf) rendered at its raw,
    # unscaled size instead.
    stage = build_stage_with_scaled_mesh(scale=(2.0, 3.0, 4.0))
    mesh_prim = stage.GetPrimAtPath("/object/scaled/mesh")
    body = Body(name=PrefixedName("test_body"))

    shape = ArtVipDatasetLoader._create_mesh_shape(mesh_prim, Gf.Matrix4d(1), body)

    vertices = shape.unscaled_mesh.vertices
    np.testing.assert_allclose(vertices.max(axis=0), [2.0, 3.0, 0.0], atol=1e-5)
