from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from huggingface_hub import HfApi, hf_hub_download
from typing_extensions import Callable, Optional

from semantic_digital_twin.adapters.artvip_dataset.exceptions import (
    ArtVipJointMissingChildBodyError,
    ArtVipMainStageFileAmbiguousError,
    ArtVipObjectNotFoundError,
    ArtVipUnsupportedJointTypeError,
)
from semantic_digital_twin.adapters.artvip_dataset.schema import (
    ArtVipCategory,
    ArtVipObject,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    logger.warning(
        "usd-core is required for ArtVIP dataset loading. "
        "Please install it using 'pip install usd-core'"
    )

_JOINT_CONNECTION_CLASSES: dict[str, type[Connection]] = {
    "PhysicsFixedJoint": FixedConnection,
    "PhysicsRevoluteJoint": RevoluteConnection,
    "PhysicsPrismaticJoint": PrismaticConnection,
}
"""
Maps a USD physics joint prim's type name to the matching Connection class.
"""

_AXIS_VECTORS: dict[str, Vector3] = {
    "X": Vector3.X(),
    "Y": Vector3.Y(),
    "Z": Vector3.Z(),
}
"""
Maps a UsdPhysics joint's local-frame axis token to the matching unit vector.
"""


def _universal_scene_description_quaternion_to_homogeneous_transformation_matrix(
    position: Gf.Vec3d, rotation: Gf.Quatf, **kwargs
) -> HomogeneousTransformationMatrix:
    """
    Build a transform from a USD position and rotation.

    :param position: The translation.
    :param rotation: The rotation.
    :param kwargs: Forwarded to ``HomogeneousTransformationMatrix.from_xyz_quaternion``
        (typically ``reference_frame``/``child_frame``).
    :return: The built transform.
    """
    imaginary = rotation.GetImaginary()
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=position[0],
        pos_y=position[1],
        pos_z=position[2],
        quat_x=imaginary[0],
        quat_y=imaginary[1],
        quat_z=imaginary[2],
        quat_w=rotation.GetReal(),
        **kwargs,
    )


@dataclass
class ArtVipDatasetLoader:
    """
    Loader for professionally modelled, articulated digital-twin objects from the ArtVIP
    dataset (https://x-humanoid-artvip.github.io/), including a dedicated IKEA furniture
    category.

    ArtVIP's meshes are clean, hand-authored CAD, decomposed into rigid links connected
    by real USD Physics joints
    (``UsdPhysics.FixedJoint``/``RevoluteJoint``/``PrismaticJoint``), each carrying an
    authored axis, position, and limits. This loader downloads one object's files from
    its Hugging Face repository, opens its main USD stage, and builds a
    :class:`~semantic_digital_twin.world.World` with one Body per link and a matching
    Connection per joint.

    .. note::
        Requires the ``usd-core`` package (``pxr``) to read the object's USD stage.

    .. note::
        Every rigid link is assumed to appear as the child (``body1``) of exactly one
        joint in the object's stage - true of every ArtVIP object inspected so far, since
        the joint graph is how the asset defines its own rigid body structure. A link
        that does not would be created (as another joint's parent) but never connected,
        and so would not appear in the built world.
    """

    directory: Path = field(default_factory=lambda: Path.home() / "artvip-dataset")
    """
    The directory object files are downloaded to.
    """

    token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    """
    The Hugging Face access token used to download the dataset.

    The dataset is public, so this is only needed to raise Hugging Face's anonymous-
    access rate limit.
    """

    repository_id: str = "x-humanoid-robomind/ArtVIP"
    """
    The Hugging Face dataset repository ID.
    """

    _repository_files: Optional[tuple[str, ...]] = field(
        default=None, init=False, repr=False
    )
    """
    Every file path in the dataset repository, cached here after the first listing so
    that loading many objects (e.g. sweeping the whole catalog) only lists the
    repository's ~5000+ files once instead of once per :meth:`available_objects`/
    :meth:`load` call.
    """

    def available_objects(self, category: ArtVipCategory) -> tuple[str, ...]:
        """
        :param category: The category to list objects of.
        :return: Every object name in ``category`` - the path (relative to the
            category, ``/``-separated) of each directory that directly contains a
            top-level USD stage file. Most objects are one path segment deep; some
            categories nest an extra subcategory level (e.g. a name of
            ``"refrigerator/fridge_01"``), in which case the returned name includes
            that segment too.
        """
        prefix = f"Articulated_objects/{category.value}/"
        return self._object_names(self._list_repository_files(), prefix)

    @staticmethod
    def _object_names(files: tuple[str, ...], prefix: str) -> tuple[str, ...]:
        """
        :param files: Every file path in the dataset repository.
        :param prefix: The category's path prefix, e.g.
            ``"Articulated_objects/small_furniture/"``.
        :return: The name of every object directory directly under ``prefix``, found by
            looking for a top-level (not nested under a ``resource`` directory) ``.usd``
            file rather than assuming a fixed nesting depth or file-naming convention.
        """
        names = set()
        for file_path in files:
            if not file_path.startswith(prefix):
                continue
            segments = file_path[len(prefix) :].split("/")
            if len(segments) < 2 or "resource" in segments[:-1]:
                continue
            if not segments[-1].endswith(".usd"):
                continue
            names.add("/".join(segments[:-1]))
        return tuple(sorted(names))

    def _list_repository_files(self) -> tuple[str, ...]:
        """
        :return: Every file path in the dataset repository, listing the repository only
            on the first call and reusing that listing afterward.
        """
        if self._repository_files is None:
            self._repository_files = tuple(
                HfApi(token=self.token).list_repo_files(
                    self.repository_id, repo_type="dataset"
                )
            )
        return self._repository_files

    def load(self, category: ArtVipCategory, name: str) -> ArtVipObject:
        """
        Load one object as a World, with joints for its links.

        :param category: The object's category.
        :param name: The object's folder name, e.g.
            ``"EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm"``.
        :return: The loaded object.
        :raises ArtVipObjectNotFoundError: if no dataset entry matches category and name.
        """
        main_usd_path = self._download_object_if_not_exists(category, name)
        stage = Usd.Stage.Open(str(main_usd_path))
        world = self._build_world(stage, category, name)
        return ArtVipObject(world=world, category=category, name=name)

    def _download_object_if_not_exists(
        self, category: ArtVipCategory, name: str
    ) -> Path:
        """
        Download every file of one object if not already present, preserving the
        dataset's own relative file layout so the object's USD stage's relative
        references resolve.

        :param category: The object's category.
        :param name: The object's folder name.
        :return: The path to the object's main USD file.
        :raises ArtVipObjectNotFoundError: if no dataset entry matches category and
            name.
        :raises ArtVipMainStageFileAmbiguousError: if the object's directory does not
            contain exactly one top-level USD file to treat as its main stage.
        """
        prefix = f"Articulated_objects/{category.value}/{name}/"
        object_files = [
            file_path
            for file_path in self._list_repository_files()
            if file_path.startswith(prefix)
        ]
        if not object_files:
            raise ArtVipObjectNotFoundError(category=category, name=name)

        main_relative_path = self._main_stage_file(
            object_files, prefix, category=category, name=name
        )

        main_usd_path = None
        for file_path in object_files:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=self.repository_id,
                    repo_type="dataset",
                    filename=file_path,
                    token=self.token,
                    local_dir=self.directory,
                )
            )
            if file_path == main_relative_path:
                main_usd_path = downloaded_path

        logger.info(
            f"Downloaded ArtVIP object {category.value}/{name} to {self.directory}"
        )
        return main_usd_path

    @staticmethod
    def _main_stage_file(
        object_files: list[str],
        prefix: str,
        *,
        category: ArtVipCategory,
        name: str,
    ) -> str:
        """
        Pick out an object's main USD stage file: the one ``.usd`` file directly inside
        its directory (not one of its ``resource/`` reference files). Objects do not
        agree on a single naming convention - some use ``model_<name>.usd``, others
        reuse the object's own name or an unrelated one (e.g.
        ``cabinet_1/cabinet_1.usd``, ``Collected_AirFryer1/AirFryer1.usd``) - so the
        directory position, not the file name, is what identifies it.

        :param object_files: Every file path under the object's directory.
        :param prefix: The object's directory path prefix.
        :param category: The object's category, for the error if this is ambiguous.
        :param name: The object's name, for the error if this is ambiguous.
        :return: The main USD file's path.
        :raises ArtVipMainStageFileAmbiguousError: if there is not exactly one such
            file.
        """
        candidates = [
            file_path
            for file_path in object_files
            if "/" not in file_path[len(prefix) :] and file_path.endswith(".usd")
        ]
        if len(candidates) != 1:
            raise ArtVipMainStageFileAmbiguousError(
                category=category, name=name, candidates=tuple(candidates)
            )
        return candidates[0]

    def _build_world(
        self, stage: Usd.Stage, category: ArtVipCategory, name: str
    ) -> World:
        """
        Build a World with one Body per rigid link in ``stage``'s physics joint graph,
        connected as described in ``ArtVipObject.world``.

        :param stage: The object's opened USD stage.
        :param category: The object's category, used to name and annotate its root body.
        :param name: The object's name, used to name and annotate its root body.
        :return: The built world.
        """
        world = World()
        object_root = Body(name=PrefixedName(name=name, prefix=category.value))
        with world.modify_world():
            world.add_body(object_root)
            world.add_semantic_annotation(
                NaturalLanguageWithTypeDescription(
                    root=object_root, description=name, type_description=category.value
                )
            )

        link_bodies: dict[str, Body] = {}

        def link_body(prim_path: Sdf.Path) -> Body:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.GetTypeName() == "Mesh":
                # Most joints target the link's enclosing Xform, whose subtree holds
                # its mesh(es); some instead target a link's mesh prim directly (seen
                # on a joint that shares a link with another joint that does target the
                # Xform). Resolving both to the same parent Xform avoids treating one
                # link as two disconnected bodies - the mesh prim itself has no
                # children of its own to be a link's subtree, and was never any other
                # joint's body1.
                prim = prim.GetParent()
            path_string = str(prim.GetPath())
            if path_string not in link_bodies:
                link_bodies[path_string] = self._create_link_body(stage, prim, name)
            return link_bodies[path_string]

        with world.modify_world():
            for joint_prim in stage.Traverse():
                if not joint_prim.IsA(UsdPhysics.Joint):
                    continue
                connection_class = _JOINT_CONNECTION_CLASSES.get(
                    joint_prim.GetTypeName()
                )
                if connection_class is None:
                    raise ArtVipUnsupportedJointTypeError(
                        category=category,
                        name=name,
                        joint_path=str(joint_prim.GetPath()),
                        joint_type=joint_prim.GetTypeName(),
                    )
                self._connect_joint(
                    world,
                    joint_prim,
                    connection_class,
                    object_root,
                    link_body,
                    category=category,
                    name=name,
                )

        return world

    @staticmethod
    def _connect_joint(
        world: World,
        joint_prim: Usd.Prim,
        connection_class: type[Connection],
        object_root: Body,
        link_body: Callable[[Sdf.Path], Body],
        *,
        category: ArtVipCategory,
        name: str,
    ) -> None:
        """
        Add the Connection ``joint_prim`` describes to ``world``.

        :param world: The world to add the connection to.
        :param joint_prim: The USD physics joint prim (a Fixed/Revolute/PrismaticJoint).
        :param connection_class: The Connection class matching the joint's type.
        :param object_root: The object's root body, used as the connection's parent if
            the joint's ``body0`` relationship has no target (the USD convention for
            "the object's own frame").
        :param link_body: Callable resolving a USD prim path to its (created-on-first-
            use) link Body.
        :param category: The object's category, used only to report a missing body1.
        :param name: The object's name, used only to report a missing body1.
        :raises ArtVipJointMissingChildBodyError: if the joint's ``body1`` relationship
            has no target - unlike ``body0``, an unset ``body1`` has no "object's own
            frame" meaning for this loader, so there is no link for the joint to
            connect.
        """
        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body1_targets:
            raise ArtVipJointMissingChildBodyError(
                category=category,
                name=name,
                joint_path=str(joint_prim.GetPath()),
            )
        parent = link_body(body0_targets[0]) if body0_targets else object_root
        child = link_body(body1_targets[0])

        parent_T_connection = _universal_scene_description_quaternion_to_homogeneous_transformation_matrix(
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
            reference_frame=parent,
        )
        connection_T_child = _universal_scene_description_quaternion_to_homogeneous_transformation_matrix(
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
            child_frame=child,
        )

        if connection_class is FixedConnection:
            connection = FixedConnection.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                parent_T_connection_expression=parent_T_connection,
                connection_T_child_expression=connection_T_child,
            )
        else:
            axis_joint = (
                UsdPhysics.RevoluteJoint(joint_prim)
                if connection_class is RevoluteConnection
                else UsdPhysics.PrismaticJoint(joint_prim)
            )
            axis = _AXIS_VECTORS[axis_joint.GetAxisAttr().Get()]
            lower = axis_joint.GetLowerLimitAttr().Get()
            upper = axis_joint.GetUpperLimitAttr().Get()
            if connection_class is RevoluteConnection:
                lower, upper = math.radians(lower), math.radians(upper)
            if lower > upper:
                # Seen authored this way on a mirrored part (e.g. one blade of a pair
                # of scissors): the pair's joints share one axis convention, so the
                # mirrored joint's authored "lower"/"upper" swap relative to it even
                # though both describe the same-sized range of motion. The DOF's own
                # lower/upper are just its two extremes, so swapping the values (not
                # negating them) keeps the authored range of motion intact.
                lower, upper = upper, lower
            connection = connection_class.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                parent_T_connection_expression=parent_T_connection,
                connection_T_child_expression=connection_T_child,
                axis=axis,
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=lower),
                    upper=DerivativeMap(position=upper),
                ),
            )
        world.add_connection(connection)

    @classmethod
    def _create_link_body(
        cls, stage: Usd.Stage, link_prim: Usd.Prim, object_name: str
    ) -> Body:
        """
        Create the Body for one rigid link, with a Shape for each mesh in the link's USD
        subtree, positioned by that mesh's transform relative to the link.

        :param stage: The object's opened USD stage.
        :param link_prim: The link's root USD prim.
        :param object_name: The object's name, used as the body's name prefix.
        :return: The created body, not yet added to a world.
        """
        body = Body(name=PrefixedName(name=link_prim.GetName(), prefix=object_name))
        link_to_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        shapes = [
            cls._create_mesh_shape(mesh_prim, link_to_world, body)
            for mesh_prim in Usd.PrimRange(link_prim)
            if mesh_prim.GetTypeName() == "Mesh"
        ]
        shape_collection = ShapeCollection(shapes, reference_frame=body)
        body.visual = shape_collection
        body.collision = shape_collection
        return body

    @staticmethod
    def _create_mesh_shape(
        mesh_prim: Usd.Prim, link_to_world: Gf.Matrix4d, body: Body
    ) -> Mesh:
        """
        Create the Mesh shape for one USD mesh prim, positioned relative to its link.

        :param mesh_prim: The USD mesh prim.
        :param link_to_world: The enclosing link's local-to-world transform.
        :param body: The link's body, used as the shape's reference frame.
        :return: The created mesh shape.
        """
        mesh_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        mesh_to_link = mesh_to_world * link_to_world.GetInverse()

        mesh_geometry = UsdGeom.Mesh(mesh_prim)
        local_vertices = np.array(mesh_geometry.GetPointsAttr().Get())
        vertices = ArtVipDatasetLoader._transform_points(local_vertices, mesh_to_link)
        faces = ArtVipDatasetLoader._triangulate(
            mesh_geometry.GetFaceVertexCountsAttr().Get(),
            mesh_geometry.GetFaceVertexIndicesAttr().Get(),
        )
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        texture_file_path = ArtVipDatasetLoader._diffuse_texture_path(mesh_prim)
        uv_per_point = ArtVipDatasetLoader._uv_coordinates(mesh_prim)
        if texture_file_path is None or uv_per_point is None:
            uv = None
        else:
            uv = uv_per_point[faces.reshape(-1)]

        return Mesh.from_trimesh(
            mesh=trimesh_mesh,
            origin=HomogeneousTransformationMatrix(reference_frame=body),
            uv=uv,
            texture_file_path=texture_file_path if uv is not None else None,
        )

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: Gf.Matrix4d) -> np.ndarray:
        """
        Apply a USD transform to an array of points.

        Applied directly to the raw vertex positions rather than split into a rotation
        and translation for the shape's origin, since an ArtVIP mesh's local-to-link
        transform can carry a non-uniform scale (seen on several ArtVIP scene props):
        decomposing a general affine transform into translation/rotation/scale is ill-
        posed in the presence of scale or shear, while applying the matrix to the points
        themselves is exact regardless.

        :param points: An ``(n, 3)`` array of points in the transform's source frame.
        :param matrix: The transform to apply.
        :return: An ``(n, 3)`` array of the transformed points.
        """
        points_homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        return (points_homogeneous @ np.array(matrix))[:, :3]

    @staticmethod
    def _diffuse_texture_path(mesh_prim: Usd.Prim) -> Optional[str]:
        """
        Resolve the file path of the diffuse texture bound to a mesh prim's material.

        :param mesh_prim: The mesh prim to look up.
        :return: The resolved path to the diffuse texture image, or ``None`` if the prim
            has no bound material, its surface shader has no ``diffuseColor`` input, or
            that input is not connected to a texture (e.g. a flat colour).
        """
        material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        if not material:
            return None

        surface_source = material.GetSurfaceOutput().GetConnectedSource()
        if surface_source is None:
            return None
        surface_shader = UsdShade.Shader(surface_source[0])

        diffuse_input = surface_shader.GetInput("diffuseColor")
        diffuse_source = diffuse_input.GetConnectedSource() if diffuse_input else None
        if diffuse_source is None:
            return None
        texture_shader = UsdShade.Shader(diffuse_source[0])

        file_input = texture_shader.GetInput("file")
        asset_path = file_input.Get() if file_input else None
        if asset_path is None or not asset_path.resolvedPath:
            return None
        return asset_path.resolvedPath

    @staticmethod
    def _uv_coordinates(mesh_prim: Usd.Prim) -> Optional[np.ndarray]:
        """
        Read a mesh prim's per-point UV coordinates from its ``st`` primvar.

        :param mesh_prim: The mesh prim to look up.
        :return: An ``(n_points, 2)`` array of UV coordinates, or ``None`` if the prim
            has no ``st`` primvar, or its interpolation is not per-point
            (``vertex``/``varying``) - the only layout observed in ArtVIP data.
        """
        primvar = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
        if not primvar.IsDefined():
            return None
        if primvar.GetInterpolation() not in (
            UsdGeom.Tokens.vertex,
            UsdGeom.Tokens.varying,
        ):
            return None
        values = primvar.Get()
        if not values:
            return None
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _triangulate(face_vertex_counts, face_vertex_indices) -> np.ndarray:
        """
        Fan-triangulate a USD mesh's polygonal faces.

        :param face_vertex_counts: The number of vertices of each face.
        :param face_vertex_indices: The faces' vertex indices, flattened in
            ``face_vertex_counts`` order.
        :return: An ``(n, 3)`` array of triangle vertex indices.
        """
        triangles = []
        cursor = 0
        for count in face_vertex_counts:
            face = face_vertex_indices[cursor : cursor + count]
            for i in range(1, count - 1):
                triangles.append((face[0], face[i], face[i + 1]))
            cursor += count
        return np.array(triangles, dtype=np.int64)
