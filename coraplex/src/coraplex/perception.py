from __future__ import annotations

from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Event

import numpy as np

from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner, FailsOnObservingFalse
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from giskardpy.motion_statechart.ros_context import RosContextExtension
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.reasoning.robot_predicates import VisibleTo
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
    IsPerceivable,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
)
from krrood_spatial_math.spatial_types import Pose, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.color_names import ColorName
from semantic_digital_twin.world_description.geometry import (
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    KinematicStructureEntity,
)
from typing_extensions import Any, Self, TYPE_CHECKING

from coraplex.datastructures.enums import DetectionGrounding, PerceptionSource
from coraplex.exceptions import (
    AmbiguousDetection,
    PerceivedObjectInsideSurface,
    PerceivedObjectNotInWorld,
    PerceptionSourceDidNotAnswer,
    PerceptionSourceFailed,
    PerceptionSourceUnavailable,
    UnidentifiedDetections,
)

if TYPE_CHECKING:
    from robokudo_msgs.msg import ObjectDesignator

ROBOKUDO_QUERY_ACTION_NAME = "robokudo/query"
"""
Name of the action RoboKudo answers perception queries on.
"""

# %% queries


@dataclass
class PerceptionQuery(SubclassJSONSerializer):
    """
    What to look for and where.

    Travels to whichever process answers it, so the world and the robot are carried by
    reference and resolved against the world on the receiving side.
    """

    semantic_annotation: type[SemanticAnnotation]
    """
    The semantic annotation for which to perceive.
    """

    region: VolumetricBoundingBox
    """
    The region in which the object should be detected.
    """

    robot: AbstractRobot
    """
    ' Robot annotation of the robot that should perceive the object.
    """

    world: World
    """
    The world in which the object should be detected.
    """

    trust_detected_orientation: bool = True
    """
    Whether to also apply the perception source's detected orientation.

    When False, only the detected position is applied to the world; the object's
    existing orientation is kept. See :meth:`Detection.apply_to`.
    """

    color: ColorName | None = None
    """
    The color the object is seen in, None for any color.
    """

    attributes: list[str] = field(default_factory=list)
    """
    Further qualities the object has, such as ``wooden``, in the words of the perception
    source; only a perception source answers them, the world model does not.
    """

    def candidates(self) -> list[Body]:
        """
        Answer this query from the world model alone, whether or not the robot's camera
        sees the bodies.

        :return: The bodies of the queried annotation that lie inside the region and
            show the queried color. A body several matching annotations describe is one
            object, so it is reported once.
        """
        bodies = dict.fromkeys(
            body
            for semantic_annotation in self.world.get_semantic_annotations_by_type(
                self.semantic_annotation
            )
            for body in semantic_annotation.bodies
        )
        return [
            body
            for body in bodies
            if self.region.contains(body.global_transform.position)
            and (self.color is None or self.color in body.visual.color_names())
        ]

    def from_world(self) -> list[Body]:
        """
        Answer this query from the world model alone.

        :return: The :meth:`candidates` that are visible to the robot's camera.
        """
        robot_camera = self.robot.get_default_camera()
        return [
            body
            for body in self.candidates()
            if VisibleTo(entity=body, camera=robot_camera)()
        ]

    def is_region_in_view(self) -> bool:
        """
        :return: Whether the centre of the region lies in the field of view of the
            robot's camera now, so the camera can look for the object there.
        """
        return self.robot.get_default_camera().is_in_field_of_view(self.region.center)

    def to_json(self, **kwargs) -> dict[str, Any]:
        result = super().to_json(**kwargs)
        result["semantic_annotation"] = to_json(self.semantic_annotation, **kwargs)
        result["region"] = to_json(self.region, **kwargs)
        result["robot_id"] = to_json(self.robot.id, **kwargs)
        result["trust_detected_orientation"] = self.trust_detected_orientation
        result["color"] = to_json(self.color, **kwargs)
        result["attributes"] = to_json(self.attributes, **kwargs)
        return result

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return cls(
            semantic_annotation=from_json(data["semantic_annotation"], **kwargs),
            region=from_json(data["region"], **kwargs),
            robot=tracker.get(from_json(data["robot_id"])),
            world=kwargs["world"],
            trust_detected_orientation=data.get("trust_detected_orientation", True),
            color=from_json(data.get("color"), **kwargs),
            attributes=from_json(data.get("attributes", []), **kwargs),
        )


# %% detections


@dataclass
class Detection:
    """
    One object reported by a perception source.
    """

    semantic_annotation: type[SemanticAnnotation]
    """
    The annotation the perception source identified the object as.
    """

    pose: Pose
    """
    Where the perception source saw the object.
    """

    color: ColorName | None = None
    """
    The color the perception source saw the object in, None when it named none.
    """

    size: Scale | None = None
    """
    The extent of the box around the object in the frame of :attr:`pose`, None when the
    perception source measured none.
    """

    maximum_raise_onto_surface: float = 0.01
    """
    How far inside, in metres, the supporting surface it stands on this detection may
    put an object that is then raised onto the surface; one it puts deeper inside is
    refused, see :class:`~coraplex.exceptions.PerceivedObjectInsideSurface`.
    """

    resting_tolerance: float = 1e-6
    """
    How far, in metres, an object may reach into the surface it stands on and still
    count as resting on it, so that one seen exactly on the surface, up to rounding, is
    left where it was seen.
    """

    clearance_above_surface: float = 0.001
    """
    How far, in metres, an object raised onto the surface it stands on is put above
    it: an object exactly on the surface touches it, which collision avoidance counts
    as a collision once the object is held.
    """

    def apply_to(
        self,
        world: World,
        trust_orientation: bool = True,
        region: VolumetricBoundingBox | None = None,
    ) -> list[IsPerceivable]:
        """
        Write this detection into ``world``, moving the perceived body to where it was
        seen and giving it the color it was seen in unless it shows that color already.
        A detection the world holds no body for becomes a new body, see :meth:`add_to`.

        :param world: The world holding the annotations this detection refers to.
        :param trust_orientation: Whether to also apply the detected orientation. When
            False, only the detected position is applied and the body keeps its existing
            orientation; useful while a perception source's orientation estimate is not
            yet reliable enough for grasp planning.
        :param region: The region the object was looked for in, see
            :meth:`resolve_annotations`.
        :return: The annotations that were updated.
        :raises PerceivedObjectNotInWorld: If the world holds no such annotation and
            none can be added for the detection.
        :raises AmbiguousDetection: If the annotation describes more than one body.
        """
        matching_annotations = self.resolve_annotations(world, region) or [
            self.add_to(world)
        ]
        self.write_into(matching_annotations[0].root, world, trust_orientation)
        return matching_annotations

    def add_to(self, world: World) -> HasRootBody:
        """
        Add the object this detection saw to ``world`` as a new free body: a box of
        :attr:`size` annotated with the detected annotation, at the detected pose.

        :param world: The world that holds no body for this detection.
        :return: The new body's annotation.
        :raises PerceivedObjectNotInWorld: If the detection measured no size, or its
            annotation cannot be created around a body of its own.
        """
        if self.size is None or not issubclass(self.semantic_annotation, HasRootBody):
            raise PerceivedObjectNotInWorld(self.semantic_annotation)
        return self.semantic_annotation.create_with_new_body_in_world(
            name=f"{self.semantic_annotation.__name__.lower()}_{len(world.bodies)}",
            world=world,
            world_root_T_self=world.transform(self.pose, world.root).homogeneous_matrix,
            parent_connection_specification=Connection6DoFSpecification(),
            scale=self.size,
        )

    def write_into(self, body: Body, world: World, trust_orientation: bool) -> None:
        """
        Move ``body`` to where this detection saw it, raised to
        :attr:`clearance_above_surface` above the supporting surface it stands on where
        the detection puts it partly inside, and give it the detected color unless it
        shows that color already.

        :param body: The body that was seen.
        :param world: The world holding the body.
        :param trust_orientation: Whether to also apply the detected orientation, see
            :meth:`apply_to`.
        :raises PerceivedObjectInsideSurface: When the detection puts the body deeper
            inside the surface it stands on than :attr:`maximum_raise_onto_surface`.
        """
        origin = self.perceived_origin(body, world, trust_orientation)
        inside = self.depth_inside_surface(body, world, trust_orientation)
        if inside is not None:
            if not inside.is_raisable:
                raise PerceivedObjectInsideSurface(
                    object_name=body.name.name,
                    surface_name=inside.surface.root.name.name,
                    depth=inside.depth,
                    maximum_raise=inside.maximum_raise,
                )
            origin = world.transform(
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=inside.depth + self.clearance_above_surface,
                    reference_frame=world.root,
                )
                @ world.transform(origin, world.root),
                origin.reference_frame,
            )
        body.parent_connection.origin = origin
        if self.color is not None and self.color not in body.visual.color_names():
            with world.modify_world():
                body.update_visual_color(self.color.representative_color)

    def perceived_origin(
        self, body: Body, world: World, trust_orientation: bool
    ) -> HomogeneousTransformationMatrix:
        """
        :param body: The body that was seen.
        :param world: The world holding the body.
        :param trust_orientation: Whether to also apply the detected orientation, see
            :meth:`apply_to`.
        :return: The origin of ``body``'s parent connection that puts it where this
            detection saw it: upright, see :meth:`_with_z_up_and_x_towards`, or in its
            present orientation when the detected one is not trusted.
        """
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]

        # Foundation Pose may return a upside down pose. this detects and fixes it by rotation around x by 180 degree
        detected_global_pose = world.transform(self.pose, world.root).homogeneous_matrix
        detected_global_pose = self._with_z_up_and_x_towards(
            root_T_frame=detected_global_pose,
            root_P_target=robot.root.global_transform.position,
        )

        if trust_orientation:
            return detected_global_pose
        parent_T_object = world.transform(
            detected_global_pose, body.parent_connection.parent
        )
        parent_origin = body.parent_connection.origin
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=parent_T_object.position,
            rotation_matrix=parent_origin.rotation_matrix,
            reference_frame=body.parent_connection.parent,
        )

    def depth_inside_surface(
        self, body: Body, world: World, trust_orientation: bool = True
    ) -> DepthInsideSurface | None:
        """
        :param body: The body that was seen.
        :param world: The world holding the body and the surfaces it may stand on.
        :param trust_orientation: Whether the detected orientation is applied, see
            :meth:`apply_to`.
        :return: How far ``body``, where this detection puts it, lies inside the
            highest supporting surface it stands on: a surface under the body's middle
            whose top lies below that middle and above the body's lowest point, by more
            than :attr:`resting_tolerance`; None when the body lies inside none, or has
            no collision shapes to lie inside one with.
        """
        if not body.collision.shapes:
            return None
        root_T_body = world.transform(
            self.perceived_origin(body, world, trust_orientation), world.root
        ).to_np()
        heights = (
            body.collision.combined_mesh.vertices @ root_T_body[2, :3]
            + root_T_body[2, 3]
        )
        lowest = float(np.min(heights))
        middle_x, middle_y, middle_z = root_T_body[:3, 3]
        tops = [
            (float(box.max_z), surface)
            for surface in world.get_semantic_annotations_by_type(HasSupportingSurface)
            if surface.root is not body
            for box in surface.root.collision.as_bounding_box_collection_in_frame(
                world.root
            ).bounding_boxes
            if box.min_x <= middle_x <= box.max_x
            and box.min_y <= middle_y <= box.max_y
            and lowest + self.resting_tolerance < box.max_z <= middle_z
        ]
        if not tops:
            return None
        top, surface = max(tops, key=lambda found: found[0])
        return DepthInsideSurface(
            surface=surface,
            depth=top - lowest,
            maximum_raise=self.maximum_raise_onto_surface,
        )

    @staticmethod
    def _with_z_up_and_x_towards(
        root_T_frame: HomogeneousTransformationMatrix, root_P_target: Point3
    ) -> HomogeneousTransformationMatrix:
        """
        Return ``root_T_frame`` rotated by 180° around its own axes so that its z axis
        points along the world z axis and its x axis points towards ``root_P_target`` as
        closely as those rotations allow.

        The position is left untouched, and every axis stays on the axis it started on.
        :param root_T_frame:``HomogeneousTransformationMatrix`` which will be rotated
        :param root_P_target:``Point3`` towards which root_T_frame x axis will point
        :return: rotated transformation matrix
        """
        root_V_target = root_P_target - root_T_frame.position

        flip_z = root_T_frame[2, 2] < 0
        flip_x = root_T_frame.rotation_matrix.x_vector().dot(root_V_target) > 0

        return root_T_frame @ HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=np.pi if flip_z and not flip_x else 0,
            pitch=np.pi if flip_z and flip_x else 0,
            yaw=np.pi if flip_x and not flip_z else 0,
        )

    def resolve_annotations(
        self, world: World, region: VolumetricBoundingBox | None = None
    ) -> list[IsPerceivable]:
        """
        Find the annotations in ``world`` that this detection identified.

        Several annotations may describe the same body, which is not ambiguous; only an
        annotation describing more than one body is. Given a region, only the bodies the
        world holds inside it are candidates, and of those the ones showing
        :attr:`color`, unless none does.

        :param world: The world to search.
        :param region: The region the object was looked for in.
        :return: The matching annotations, all sharing one root body; empty when the
            world holds no instance of the annotation there.
        :raises AmbiguousDetection: If the matching annotations sit on different bodies.
        """
        candidates = [
            annotation
            for annotation in world.get_semantic_annotations_by_type(
                self.semantic_annotation
            )
            if region is None
            or region.contains(annotation.root.global_transform.position)
        ]
        if not candidates:
            return []
        candidates = [
            annotation
            for annotation in candidates
            if self.color in annotation.root.visual.color_names()
        ] or candidates
        bodies = {annotation.root for annotation in candidates}
        if len(bodies) > 1:
            raise AmbiguousDetection(self.semantic_annotation, len(bodies))
        return candidates


@dataclass
class DepthInsideSurface:
    """
    How far a detection puts an object inside the supporting surface it stands on, such
    as a cube a camera whose calibration is a few millimetres off sees sunk into the
    table.
    """

    surface: HasSupportingSurface
    """
    The surface the object stands on.
    """

    depth: float
    """
    How far, in metres, the object's lowest point lies below the surface's top.
    """

    maximum_raise: float
    """
    How far, in metres, the object may be raised onto the surface, see
    :attr:`Detection.maximum_raise_onto_surface`.
    """

    @property
    def is_raisable(self) -> bool:
        """
        Whether the object lies shallow enough inside the surface to be raised onto it.
        """
        return self.depth <= self.maximum_raise


@dataclass
class DetectedScene:
    """
    The objects a perception source saw in a region at once, each detection a different
    object.
    """

    detections: list[Detection]
    """
    What the perception source saw.
    """

    region: VolumetricBoundingBox
    """
    The region the objects were looked for in.
    """

    def apply_to(self, world: World, trust_orientation: bool = True) -> list[Body]:
        """
        Write every detection into ``world``: each moves the body nearest to it that
        carries its annotation, lies in the region, shows its color and was taken by no
        earlier detection, or becomes a new body, see :meth:`Detection.add_to`.

        :param world: The world the objects are written into.
        :param trust_orientation: Whether to also apply the detected orientations, see
            :meth:`Detection.apply_to`.
        :return: The body of each detection, in the order of :attr:`detections`.
        :raises PerceivedObjectNotInWorld: If a detection matches no body and none can
            be added for it.
        """
        bodies: list[Body] = []
        for detection in self.detections:
            body = self.nearest_free_body(detection, world, bodies)
            if body is None:
                body = detection.add_to(world).root
            detection.write_into(body, world, trust_orientation)
            bodies.append(body)
        return bodies

    def nearest_free_body(
        self, detection: Detection, world: World, taken: list[Body]
    ) -> Body | None:
        """
        :param detection: One detection of the scene.
        :param world: The world the objects are written into.
        :param taken: The bodies earlier detections of the scene moved.
        :return: The body nearest to the detection that carries its annotation, lies in
            the region, shows its color and is not taken, None when there is none.
        """
        detected_position = world.transform(detection.pose, world.root).position
        candidates = [
            annotation.root
            for annotation in world.get_semantic_annotations_by_type(
                detection.semantic_annotation
            )
            if annotation.root not in taken
            and self.region.contains(annotation.root.global_transform.position)
            and (
                detection.color is None
                or detection.color in annotation.root.visual.color_names()
            )
        ]
        return min(
            candidates,
            key=lambda body: float(
                body.global_transform.position.euclidean_distance(detected_position)
            ),
            default=None,
        )


# %% perception sources


class PerceptionInterface(ABC):
    """
    A source of detections.
    """

    @abstractmethod
    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        """
        Report every object this source sees for a perception query.

        :param query: What to look for and where.
        :return: One detection per object seen, empty when the source saw nothing.
        """

    def detect(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> Detection | None:
        """
        Answer a perception query with the one object it asked about.

        :param query: What to look for and where.
        :param accept_first_if_multiple: Whether several candidates may be resolved by
            taking the first one. When False, several candidates raise
            :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen
            between.
        :return: The object this source saw, None when it saw nothing.
        :raises UnidentifiedDetections: If several candidates were seen and none of them
            may be picked.
        """
        detections = self.look_for(query)
        if not detections:
            return None
        if len(detections) > 1 and not accept_first_if_multiple:
            raise UnidentifiedDetections(query.semantic_annotation, len(detections))
        return detections[0]

    @classproperty
    def default_answer_timeout(cls) -> timedelta:
        """
        :return: How long a source answering in another process is waited for by
            default.
        """
        return timedelta(seconds=30)

    @staticmethod
    def for_source(
        source: PerceptionSource,
        ros_node: Node | None = None,
        answer_timeout: timedelta | None = None,
    ) -> PerceptionInterface:
        """
        :param source: The kind of source that answers the queries.
        :param ros_node: Node a RoboKudo pipeline is reached through.
        :param answer_timeout: How long a RoboKudo pipeline is waited for, None for
            :attr:`default_answer_timeout`.
        :return: The source to answer queries with.
        """
        match source:
            case PerceptionSource.WORLD_MODEL:
                return WorldPerception()
            case PerceptionSource.ROBOKUDO:
                return RoboKudoPerception(
                    ros_node=ros_node,
                    answer_timeout=(
                        PerceptionInterface.default_answer_timeout
                        if answer_timeout is None
                        else answer_timeout
                    ),
                )


@dataclass
class WorldPerception(PerceptionInterface):
    """
    Source that reads the objects straight out of the world model.

    Reports what a perfect sensor would see: every body of the queried annotation that
    lies inside the region and is visible to the robot's camera, at the pose the world
    already holds.
    """

    reports_unseen_bodies: bool = False
    """
    Whether it also reports the bodies the robot's camera does not see now, as the world
    model alone believes them to be.
    """

    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        annotation_by_body = {}
        for annotation in query.world.get_semantic_annotations_by_type(
            query.semantic_annotation
        ):
            annotation_by_body.setdefault(annotation.root, type(annotation))
        bodies = (
            query.candidates() if self.reports_unseen_bodies else query.from_world()
        )
        return [
            Detection(
                semantic_annotation=annotation_by_body[body],
                pose=body.global_pose,
                color=query.color,
            )
            for body in bodies
            if body in annotation_by_body
        ]


@dataclass
class GroundedDetections:
    """
    The objects a detection reports while a plan is checked, and where they come from.
    """

    detections: list[Detection]
    """
    The objects reported.
    """

    grounding: DetectionGrounding
    """
    Whether they were perceived now or assumed from the world model.
    """


@dataclass
class ValidationPerception:
    """
    Answers perception queries while a plan is checked, before it runs: through the
    robot's perception where its camera sees the region now, else from the world model,
    whose answer is assumed until the plan runs and perceives it.
    """

    live: PerceptionInterface
    """
    The robot's perception.
    """

    def look_for(self, query: PerceptionQuery) -> GroundedDetections:
        """
        :param query: What to look for and where.
        :return: Every object reported for the query, and where they come from.
        """
        if query.is_region_in_view():
            return GroundedDetections(
                self.live.look_for(query), DetectionGrounding.PERCEIVED
            )
        return GroundedDetections(
            WorldPerception(reports_unseen_bodies=True).look_for(query),
            DetectionGrounding.ASSUMED,
        )


@dataclass
class QueryAnswerWait:
    """
    The wait for the answer to one query sent over an action, bounded by one deadline
    for every step of it, keeping what the action server last reported as feedback.
    """

    deadline: float
    """
    The time, on :func:`time.monotonic`'s clock, the answer is waited for until.
    """

    latest_feedback: str = ""
    """
    What the action server last reported while working on the query, empty before it
    reported anything.
    """

    def record_feedback(self, feedback_message: Any) -> None:
        """
        :param feedback_message: A feedback message of the query, whose ``feedback``
            text is kept.
        """
        self.latest_feedback = feedback_message.feedback.feedback

    def until_done(self, future: Future) -> bool:
        """
        :param future: A step of the query, completed by the node's executor.
        :return: Whether it completed before the deadline.
        """
        done = Event()
        future.add_done_callback(lambda _: done.set())
        return done.wait(max(0.0, self.deadline - time.monotonic()))


@dataclass
class RoboKudoPerception(PerceptionInterface):
    """
    Source that asks a running RoboKudo pipeline over its query action, reporting the
    objects it sees inside the queried region.
    """

    ros_node: Node
    """
    Node the action client is created on.
    """

    action_name: str = ROBOKUDO_QUERY_ACTION_NAME
    """
    Action RoboKudo answers queries on.
    """

    server_timeout: timedelta = field(default=timedelta(seconds=10))
    """
    How long to wait for the action server before giving up.
    """

    answer_timeout: timedelta = field(
        default_factory=lambda: PerceptionInterface.default_answer_timeout
    )
    """
    How long a query the action server took up may take to be answered.
    """

    def look_for(self, query: PerceptionQuery) -> list[Detection]:
        """
        :raises PerceptionSourceUnavailable: If no action server serves
            :attr:`action_name` within :attr:`server_timeout`.
        :raises PerceptionSourceDidNotAnswer: If the query is not answered within
            :attr:`answer_timeout`; the query is canceled then.
        :raises PerceptionSourceFailed: If the pipeline aborts the query, naming why.
        """
        # RoboKudo's messages only exist where its pipeline is installed, so they are
        # imported here rather than at module level: reading the world in simulation must
        # not depend on them.
        from robokudo_msgs.action import Query
        from robokudo_msgs.msg import ObjectDesignator

        client = ActionClient(self.ros_node, Query, self.action_name)
        if not client.wait_for_server(timeout_sec=self.server_timeout.total_seconds()):
            raise PerceptionSourceUnavailable(self.action_name)

        goal = Query.Goal(
            obj=ObjectDesignator(
                type=query.semantic_annotation.__name__.lower(),
                color=[] if query.color is None else [query.color.value],
                attribute=query.attributes,
            )
        )
        wait = QueryAnswerWait(
            deadline=time.monotonic() + self.answer_timeout.total_seconds()
        )
        sent = client.send_goal_async(goal, feedback_callback=wait.record_feedback)
        if not wait.until_done(sent):
            raise self._unanswered(wait)
        goal_handle = sent.result()
        answered = goal_handle.get_result_async()
        if not wait.until_done(answered):
            goal_handle.cancel_goal_async()
            raise self._unanswered(wait)
        if answered.result().status == GoalStatus.STATUS_ABORTED:
            raise PerceptionSourceFailed(
                self.action_name, answered.result().result.text_result
            )
        result = answered.result().result
        detections = [
            self._to_detection(designator, query)
            for designator in result.res
            if designator.pose and self._can_be_requested_object(designator, query)
        ]
        return [
            detection
            for detection in detections
            if query.region.contains(detection.pose.position)
        ]

    def _unanswered(self, wait: QueryAnswerWait) -> PerceptionSourceDidNotAnswer:
        """
        :param wait: The wait for the query that was not answered.
        :return: The failure naming what the pipeline last reported waiting for.
        """
        return PerceptionSourceDidNotAnswer(
            self.action_name, self.answer_timeout, wait.latest_feedback
        )

    @staticmethod
    def _can_be_requested_object(
        designator: ObjectDesignator, query: PerceptionQuery
    ) -> bool:
        """
        Whether a reported designator could be the object the query asked for.

        A pipeline that only localizes reports no class label, and every object it found
        is then a candidate. One that classifies lets the objects that were not asked
        for be discarded here, and one that names colors those of another color.

        :param designator: The designator the pipeline reported.
        :param query: The query it was answering.
        :return: True if the designator is a candidate for the requested object.
        """
        if (
            query.color is not None
            and designator.color
            and query.color.value not in designator.color
        ):
            return False
        if not designator.type:
            return True
        return (
            designator.type.strip().lower()
            == query.semantic_annotation.__name__.lower()
        )

    def _to_detection(
        self, designator: ObjectDesignator, query: PerceptionQuery
    ) -> Detection:
        """
        Convert one RoboKudo object designator into a detection of the queried
        annotation, which :meth:`_can_be_requested_object` narrowed it to, in the first,
        most dominant, color reported that has a :class:`ColorName` and of the first
        shape size reported.

        :param designator: The designator RoboKudo reported.
        :param query: The query it was answering, supplying the annotation and the
            frame.
        :return: The detection carrying the queried annotation and the reported pose.
        """
        world = query.world
        pose_stamped = designator.pose[0]
        extent = designator.shape_size[0].dimensions if designator.shape_size else None
        return Detection(
            semantic_annotation=query.semantic_annotation,
            size=None if extent is None else Scale(extent.x, extent.y, extent.z),
            color=next(
                (ColorName(color) for color in designator.color if color in ColorName),
                None,
            ),
            pose=Pose.from_xyz_quaternion(
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
                reference_frame=self._reported_frame(
                    world, pose_stamped.header.frame_id
                ),
            ),
        )

    @staticmethod
    def _reported_frame(world: World, frame_id: str) -> KinematicStructureEntity:
        """
        The entity a frame RoboKudo reports names.

        A frame named as the world's root is the root, as the world's root is published
        on tf under its name, even where the robot's description, merged with its own
        names, has a root link named alike.

        :param world: The world the detection is expressed in.
        :param frame_id: The frame of the reported pose.
        :return: The entity the frame names in the world.
        """
        if frame_id == str(world.root.name):
            return world.root
        return world.get_kinematic_structure_entity_by_name(frame_id)


# %% perceiving inside the motion chart


@dataclass(eq=False, repr=False, kw_only=True)
class PerceptionTask(EndedByOwner, FailsOnObservingFalse, StatechartNode):
    """
    Statechart node that answers a perception query and writes what it saw into the
    world.

    The node moves nothing, so it adds no motion constraints and its owner decides when
    it succeeded, like a task of a motion. The query is answered on the node's first tick,
    so the whole detection takes one tick however long the source needs to reply, and the
    node then observes ``TRUE`` so the surrounding sequence continues. A source that saw
    nothing makes it observe ``FALSE``, which fails it, so a surrounding
    :class:`~cramph.composites.TryInOrder` can go and look somewhere else.

    .. warning:: That tick blocks until the source replies, which on the real robot holds
        up the control loop for as long as the pipeline takes to answer.
    """

    query: PerceptionQuery
    """
    What to look for and where.
    """

    answered_by: PerceptionSource
    """
    Which kind of source answers the query.

    Carried by the node rather than read from the context, because on the real robot the
    chart is answered in the controller's process, whose context does not say how the
    plan is executed.
    """

    answer_timeout: timedelta = field(
        default_factory=lambda: PerceptionInterface.default_answer_timeout
    )
    """
    How long a source answering in another process is waited for.
    """

    perception_source: PerceptionInterface | None = field(init=False, default=None)
    """
    The source answering the query, resolved during :meth:`set_up`.
    """

    _observation: ObservationStateValues | None = field(
        init=False, default=None, repr=False
    )
    """
    The observation this node keeps once the query has been answered, None before that.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    def set_up(self) -> None:
        super().set_up()
        self.perception_source = PerceptionInterface.for_source(
            self.answered_by,
            self.context.require_extension(RosContextExtension).ros_node,
            self.answer_timeout,
        )

    def on_start(self) -> None:
        self._observation = None

    def on_tick(self) -> ObservationStateValues | None:
        if self._observation is None:
            self._observation = self._answer_query()
        return self._observation

    def _answer_query(self) -> ObservationStateValues:
        """
        Answer the query and write what was seen into the world.

        :return: True once the object was seen, False when the source saw nothing.
        """
        detection = self.perception_source.detect(
            self.query, self.accept_first_if_multiple
        )
        if detection is None:
            return ObservationStateValues.FALSE
        detection.apply_to(
            self.query.world,
            trust_orientation=self.query.trust_detected_orientation,
            region=self.query.region,
        )
        return ObservationStateValues.TRUE
