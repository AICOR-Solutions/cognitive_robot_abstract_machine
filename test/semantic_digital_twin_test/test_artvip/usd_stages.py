from __future__ import annotations

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False


def _define_link(stage: Usd.Stage, path: str) -> None:
    """
    Define a link ``Xform`` at ``path`` with a single quad mesh, so it has visual
    geometry the way every real ArtVIP link does.
    """
    UsdGeom.Xform.Define(stage, path)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])


def build_single_joint_stage(
    joint_type: str,
    *,
    axis: str = "Z",
    local_pos0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot1: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    lower_limit: float = -90.0,
    upper_limit: float = 0.0,
) -> Usd.Stage:
    """
    A minimal in-memory stage with a root link ("carcass") and a child link ("child")
    connected by one joint of ``joint_type``, in the same shape
    ``ArtVipDatasetLoader._build_world`` reads: ``Xform`` links each holding one
    ``Mesh``, and a joint prim with body0/body1 relationships and
    localPos/localRot/axis/limit attributes.

    :param joint_type: A ``UsdPhysics`` joint type name, e.g. ``"RevoluteJoint"``,
        ``"PrismaticJoint"``, ``"FixedJoint"``, or ``"SphericalJoint"`` for an
        unsupported-type test.
    :param axis: The joint's local-frame axis token.
    :param local_pos0: The joint frame's translation relative to body0.
    :param local_rot0: The joint frame's rotation (w, x, y, z) relative to body0.
    :param local_pos1: The joint frame's translation relative to body1.
    :param local_rot1: The joint frame's rotation (w, x, y, z) relative to body1.
    :param lower_limit: The joint's lower limit (degrees for Revolute, meters for
        Prismatic; ignored for Fixed/Spherical).
    :param upper_limit: The joint's upper limit.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    _define_link(stage, "/object/child")

    joint_class = getattr(UsdPhysics, joint_type)
    joint = joint_class.Define(stage, "/object/joint")
    # body0 is left with no targets: the USD convention for "the object's own root
    # frame", so the built connection's parent is the object's root body.
    joint.CreateBody1Rel().SetTargets(["/object/child"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(*local_rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(*local_rot1))
    if joint_type in ("RevoluteJoint", "PrismaticJoint"):
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(lower_limit)
        joint.CreateUpperLimitAttr(upper_limit)

    return stage


def build_stage_with_joint_missing_body1() -> Usd.Stage:
    """
    A minimal in-memory stage with a single ``FixedJoint`` whose ``body1`` relationship
    has no target - unlike ``body0``, an unset ``body1`` has no "object's own frame"
    meaning for ``ArtVipDatasetLoader._connect_joint``, since every joint is expected to
    connect a link into the object.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    joint = UsdPhysics.FixedJoint.Define(stage, "/object/joint")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return stage


def build_stage_with_mesh_targeted_body0() -> Usd.Stage:
    """
    Reproduces a layout seen on real ArtVIP basket objects: one link ("carcass") is
    connected to the root by a fixed joint targeting its enclosing Xform, and a second
    joint's body0 targets that same link's Mesh prim directly instead of the Xform. Both
    should resolve to the same link body, not create a second, disconnected one.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    _define_link(stage, "/object/carcass")
    _define_link(stage, "/object/handle")

    fixed = UsdPhysics.FixedJoint.Define(stage, "/object/carcass/joint")
    fixed.CreateBody1Rel().SetTargets(["/object/carcass"])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/object/handle/joint")
    hinge.CreateBody0Rel().SetTargets(["/object/carcass/mesh"])
    hinge.CreateBody1Rel().SetTargets(["/object/handle"])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLowerLimitAttr(-30.0)
    hinge.CreateUpperLimitAttr(30.0)
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return stage


def build_stage_with_textured_mesh(texture_file_path: str) -> Usd.Stage:
    """
    A minimal in-memory stage with a single quad mesh, per-point ``st`` UV
    coordinates, and a material whose ``diffuseColor`` is driven by a texture read
    from ``texture_file_path`` - the layout
    ``ArtVipDatasetLoader._diffuse_texture_path``/``_uv_coordinates`` read.

    :param texture_file_path: Path to the texture image the material's
        ``UsdUVTexture`` node reads.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/object/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    material = UsdShade.Material.Define(stage, "/object/material")
    pbr_shader = UsdShade.Shader.Define(stage, "/object/material/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        pbr_shader.ConnectableAPI(), "surface"
    )

    texture_shader = UsdShade.Shader.Define(stage, "/object/material/diffuseTexture")
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file_path)
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb"
    )

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    return stage


def build_stage_with_scaled_mesh(scale: tuple[float, float, float]) -> Usd.Stage:
    """
    A minimal in-memory stage with a quad mesh under a child ``Xform`` authored with
    ``scale`` - the shape of decorative props seen on real ArtVIP scene furniture (e.g.
    a fruit platter's leaf), whose local-to-world transform carries scale on top of its
    translation and rotation.

    :param scale: The child ``Xform``'s authored scale.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    scaled = UsdGeom.Xform.Define(stage, "/object/scaled")
    scaled.AddScaleOp().Set(Gf.Vec3f(*scale))
    mesh = UsdGeom.Mesh.Define(stage, "/object/scaled/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    return stage
