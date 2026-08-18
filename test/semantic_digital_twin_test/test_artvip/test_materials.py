import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from .usd_stages import (
    PXR_AVAILABLE,
    build_stage_with_textured_mesh,
)

if PXR_AVAILABLE:
    from pxr import Gf, UsdGeom

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


@pytest.fixture
def texture_file(tmp_path):
    path = tmp_path / "wood.png"
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(path)
    return str(path)


def test_diffuse_texture_path_resolves_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    resolved = ArtVipDatasetLoader._diffuse_texture_path(mesh_prim)

    assert resolved == texture_file


def test_diffuse_texture_path_is_none_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    resolved = ArtVipDatasetLoader._diffuse_texture_path(unbound_mesh_prim)

    assert resolved is None


def test_uv_coordinates_reads_the_per_point_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    uv = ArtVipDatasetLoader._uv_coordinates(mesh_prim)

    np.testing.assert_array_equal(uv, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_uv_coordinates_is_none_without_an_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    uv = ArtVipDatasetLoader._uv_coordinates(unbound_mesh_prim)

    assert uv is None


def test_create_mesh_shape_applies_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")
    body = Body(name=PrefixedName("test_body"))

    shape = ArtVipDatasetLoader._create_mesh_shape(mesh_prim, Gf.Matrix4d(1), body)

    mesh = shape.unscaled_mesh
    assert mesh.visual.kind == "texture"
    assert mesh.visual.uv is not None


def test_create_mesh_shape_has_no_texture_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh = UsdGeom.Mesh.Define(stage, "/object/unbound_mesh")
    unbound_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    unbound_mesh.CreateFaceVertexCountsAttr([3])
    unbound_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    body = Body(name=PrefixedName("test_body"))

    shape = ArtVipDatasetLoader._create_mesh_shape(
        unbound_mesh.GetPrim(), Gf.Matrix4d(1), body
    )

    assert shape.unscaled_mesh.visual.kind != "texture"
