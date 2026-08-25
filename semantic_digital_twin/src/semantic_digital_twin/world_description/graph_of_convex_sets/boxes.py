from __future__ import annotations

import itertools
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from operator import or_

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
from random_events.interval import Bound, closed, Interval
from random_events.plotting import EventPlotter
from random_events.product_algebra import Event
from random_events.product_algebra import SimpleEvent
from rtree import index
from typing_extensions import (
    Generic,
    List,
    Optional,
    Dict,
    Sequence,
    Self,
    Type,
    TypeVar,
)

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    CanBehaveLikeAVariable,
    MappedVariable,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    chained_logic,
)
from krrood.symbol_graph.helpers import get_field_type_endpoint
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    BoundingBox,
    BoundingBox2D,
    Bounds,
    Color,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    GraphOfConvexSets,
    PointT,
    SearchSpaceT,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    AmbiguousSelectedVariableError,
    MissingFloatLikeFieldError,
    UnconnectedGraphError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    BoundingBoxCollection2D,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
)

logger = logging.getLogger(__name__)

BoxT = TypeVar("BoxT")
"""
The bounding-box type a :class:`GraphOfBoundingBoxes` subclass decomposes free space
into -- :class:`BoundingBox` for a volumetric decomposition,
:class:`BoundingBox2D` for a planar one.
"""


@dataclass
class BoundingBoxAdjacency(Generic[BoxT]):
    """
    Edge payload connecting two adjacent bounding boxes in a
    :class:`GraphOfBoundingBoxes`.
    """

    intersection: BoxT
    """
    The region where the two adjacent boxes overlap or touch.
    """

    distance: float
    """
    Euclidean distance between the centers of the two adjacent boxes.

    Used as the edge cost for shortest-path search, so that the search minimizes
    travelled distance instead of the number of boxes crossed.
    """


@dataclass(frozen=True)
class PathQuery(Generic[PointT]):
    """
    A start and a goal to plan a path between.
    """

    start: PointT
    """
    Where the path begins.
    """

    goal: PointT
    """
    Where the path ends.
    """


@dataclass
class GraphOfBoundingBoxes(
    Generic[BoxT, PointT, SearchSpaceT],
    GraphOfConvexSets[PointT, SearchSpaceT],
):
    """
    Abstract base for graphs of convex sets whose nodes are axis-aligned bounding boxes.

    Free space is decomposed into an exact, exhaustive partition of boxes via the
    `random_events` product algebra (obstacles subtracted from the search space). Every
    node is a box; every edge represents the adjacency between two boxes. Concrete
    subclasses differ in how many dimensions that decomposition happens in --
    :class:`VolumetricGraphOfBoundingBoxes` partitions all three,
    :class:`PlanarGraphOfBoundingBoxes` partitions the floor plane only.
    """

    graph: rx.PyGraph[BoxT, BoundingBoxAdjacency[BoxT]] = field(
        default_factory=lambda: rx.PyGraph(multigraph=False)
    )
    """
    The connectivity graph of the convex sets.
    """

    box_to_index_map: Dict[BoxT, int] = field(default_factory=dict)
    """
    A mapping from bounding boxes to their indices in the graph.
    """

    @abstractmethod
    def _default_search_space(self) -> SearchSpaceT:
        raise NotImplementedError

    def create_subgraph(self, nodes: Sequence[int]) -> Self:
        """
        Create a subgraph of the current graph containing only the given nodes.

        :param nodes: The nodes to include in the subgraph.
        :return: The subgraph.
        """
        subgraph = type(self)(self.world, self.search_space)
        subgraph.graph = self.graph.subgraph(nodes)
        subgraph.box_to_index_map = {
            box: index for box, index in self.box_to_index_map.items() if index in nodes
        }
        return subgraph

    def add_node(self, box: BoxT):
        self.box_to_index_map[box] = self.graph.add_node(box)

    def calculate_connectivity(self, tolerance: float = 0.001):
        """
        Calculate the connectivity of the graph by checking for intersections between
        the bounding boxes of the nodes. This uses an R-tree for efficient spatial
        indexing and intersection queries. Each edge is weighted by the Euclidean
        distance between the centers of the two boxes it connects, for use by
        :meth:`path_from_to`.

        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        """

        def _overlap(a_min, a_max, b_min, b_max) -> bool:
            return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))

        node_list = list(self.graph.nodes())
        if not node_list:
            return

        # BoundingBox.x_interval/y_interval/z_interval (and their planar counterparts)
        # recompute symbolic arithmetic on every access, so every node's bounds are read
        # as plain floats exactly once here rather than once per pair below.
        bounds_list = [node.to_array_bounds() for node in node_list]
        dimensionality = len(bounds_list[0].lower)
        centers = [(bounds.lower + bounds.upper) / 2 for bounds in bounds_list]
        expanded = [
            tuple(bounds.lower - tolerance) + tuple(bounds.upper + tolerance)
            for bounds in bounds_list
        ]

        prop = index.Property()
        prop.dimension = dimensionality
        rtree_idx = index.Index(properties=prop)
        for i, box_expansion in enumerate(expanded):
            rtree_idx.insert(i, box_expansion)

        # Query & link, skip self-loops and symmetric pairs
        for i, bounds_i in enumerate(bounds_list):
            for j in rtree_idx.intersection(expanded[i]):
                if j <= i:  # symmetry → skip
                    continue
                bounds_j = bounds_list[j]
                if not _overlap(
                    bounds_i.lower, bounds_i.upper, bounds_j.lower, bounds_j.upper
                ):
                    continue  # no true overlap
                lower = np.maximum(bounds_i.lower, bounds_j.lower)
                upper = np.minimum(bounds_i.upper, bounds_j.upper)
                box = type(node_list[i]).from_array_bounds(
                    lower,
                    upper,
                    HomogeneousTransformationMatrix(reference_frame=self.world.root),
                )
                distance = float(np.linalg.norm(centers[i] - centers[j]))

                # Map from the local list positions back to the graph node indices
                u = self.box_to_index_map[node_list[i]]
                v = self.box_to_index_map[node_list[j]]

                self.graph.add_edge(u, v, BoundingBoxAdjacency(box, distance))

    def draw(self):
        import rustworkx.visualization

        rustworkx.visualization.mpl_draw(self.graph)
        plt.show()

    def plot_free_space(self) -> List[go.BaseTraceType]:
        """
        Plot the free space of the environment in blue.

        :return: A list of traces that can be put into a plotly figure.
        """
        return EventPlotter(self.free_space_event).plot(color="blue")

    def plot_and_show_free_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_free_space()).show()

    def plot_occupied_space(self) -> List[go.BaseTraceType]:
        """
        Plot the occupied space of the environment in red.

        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )
        occupied_space = ~free_space & self.search_space.event
        return EventPlotter(occupied_space).plot(color="red")

    def plot_and_show_occupied_space(self) -> None:
        import plotly.graph_objects as go

        go.Figure(self.plot_occupied_space()).show()

    def node_of_point(self, point: PointT) -> Optional[BoxT]:
        """
        Find the node that contains a point.

        :return: The node that contains the point or None if no node contains the point.
        """
        for node in self.graph.nodes():
            if node.contains(point):
                return node
        return None

    def path_from_to(self, start: PointT, goal: PointT) -> Optional[List[PointT]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        .. note::
            Uses a single-source Dijkstra search, weighted by the Euclidean distance
            between adjacent boxes' centers, rather than enumerating all shortest paths
            and picking the first one. Free-space decompositions with thousands of
            nodes routinely have an exponential number of equally-short (by hop count)
            paths, which makes enumerating all of them intractable; finding the one
            that minimizes travelled distance is not.

        .. note::
            The resulting waypoints are shortcut afterwards: any waypoint that a
            straight line can bypass without leaving free space is dropped. See
            :meth:`_shortcut_waypoints`.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to or None if no path
            exists.
        """
        # get poses from params
        start_node = self.node_of_point(start)
        goal_node = self.node_of_point(goal)

        # validate if the poses are part of the graph
        if start_node is None:
            raise PointOccupiedError(start)
        if goal_node is None:
            raise PointOccupiedError(goal)

        if start_node == goal_node:
            return [start, goal]

        start_index = self.box_to_index_map[start_node]
        goal_index = self.box_to_index_map[goal_node]

        paths = rx.dijkstra_shortest_paths(
            self.graph,
            start_index,
            target=goal_index,
            weight_fn=lambda adjacency: adjacency.distance,
        )

        # if it is not possible to find a path
        if goal_index not in paths:
            return None

        path = paths[goal_index]

        # build the path
        reference_frame = self.search_space.reference_frame
        waypoints = [self.world.transform(start, reference_frame)]

        for source, target in zip(path, path[1:]):
            intersection = self.graph.get_edge_data(source, target).intersection
            waypoints.append(intersection.center)

        waypoints.append(self.world.transform(goal, reference_frame))
        waypoints = self._shortcut_waypoints(waypoints)

        result = [start]
        result.extend(waypoints[1:-1])
        result.append(goal)
        return result

    def _shortcut_waypoints(self, waypoints: List[PointT]) -> List[PointT]:
        """
        Drop waypoints that a straight line can bypass without leaving free space.

        Greedily extends the current anchor waypoint forward as far as a straight
        line to it stays collision-free, then commits the farthest waypoint still
        visible from it and continues from there (classic "string pulling"). Each
        waypoint is tested against the current anchor at most once, so this is
        linear in the number of waypoints rather than quadratic.

        :param waypoints: The waypoints of a path, in the search space's reference
            frame.
        :return: The shortcut waypoints.
        """
        if len(waypoints) <= 2:
            return list(waypoints)

        node_bounds = [node.to_array_bounds() for node in self.graph.nodes()]

        result = [waypoints[0]]
        anchor_index = 0
        for index in range(2, len(waypoints)):
            if not self._segment_is_collision_free(
                waypoints[anchor_index], waypoints[index], node_bounds
            ):
                result.append(waypoints[index - 1])
                anchor_index = index - 1
        result.append(waypoints[-1])
        return result

    def _segment_is_collision_free(
        self, start: PointT, end: PointT, node_bounds: List[Bounds[np.ndarray]]
    ) -> bool:
        """
        Check whether a straight-line segment stays entirely within free space.

        :param start: The segment's start point, in the search space's reference frame.
        :param end: The segment's end point, in the search space's reference frame.
        :param node_bounds: The graph's nodes' bounds, in the same order
            :meth:`_shortcut_waypoints` collected them.
        :return: True if the segment never leaves the union of the graph's bounding-box
            nodes.
        """
        dimensionality = len(node_bounds[0].lower)
        coordinates = np.asarray(start.to_np()[:dimensionality], dtype=float)
        end_coordinates = np.asarray(end.to_np()[:dimensionality], dtype=float)
        deltas = end_coordinates - coordinates
        covered_intervals = [
            interval
            for bounds in node_bounds
            if (interval := bounds.clip_segment(coordinates, deltas)) is not None
        ]
        if not covered_intervals:
            return False
        covered = Interval.from_simple_sets(*covered_intervals).make_disjoint()
        return (closed(0.0, 1.0) - covered).is_empty()

    @property
    def free_space_event(self) -> Event:
        return Event.from_simple_sets(
            *[node.simple_event for node in self.graph.nodes()]
        )

    def constrain_to_free_space(self, variable: MappedVariable) -> OR:
        """
        Add a where condition to ``variable``'s query, restricting it to lie within
        this graph's free space.

        The free space is a union of boxes, so the condition is an ``OR`` over one
        ``AND`` per box, each conjoining a lower and an upper bound per coordinate.
        Which coordinates that is -- x, y for a planar graph; x, y, z for a
        volumetric one -- follows from :attr:`free_space_event` alone, so this one
        implementation serves every :class:`GraphOfBoundingBoxes` subclass.

        ``variable`` is rerooted onto its query's own ``selected_variable`` before
        its fields are read: a query's own type does not resolve (what it selects
        lives on ``selected_variable`` instead), and a condition built directly
        against the selection is exactly what evaluates correctly per query result
        without needing the query's own rerooting machinery to intervene.

        :param variable: The eql variable to constrain, e.g. a Pose2D- or
            Point3-typed attribute of a query, or the query itself.
        :return: The condition that was added.
        :raises MissingFloatLikeFieldError: If ``variable`` has no float-like field
            for one of this graph's spatial coordinates.
        """
        chain_root = (
            variable._chain_root_ if isinstance(variable, MappedVariable) else variable
        )
        selected_variable = GraphOfBoundingBoxes._selected_variable_of(chain_root)
        if selected_variable is not None:
            variable = (
                variable._reroot_on_(selected_variable, chain_root)
                if isinstance(variable, MappedVariable)
                else selected_variable
            )

        free_space_event = self.free_space_event
        fields = {
            spatial_variable.name: self._floatlike_field(
                variable, spatial_variable.name
            )
            for spatial_variable in free_space_event.variables
        }

        simple_event_conditions = []
        for simple_event in free_space_event.simple_sets:
            axes = [simple_event[name].simple_sets for name in fields]
            for combination in itertools.product(*axes):
                bounds = []
                for field_name, simple_interval in zip(fields, combination):
                    field = fields[field_name]
                    bounds.append(
                        field >= simple_interval.lower
                        if simple_interval.left == Bound.CLOSED
                        else field > simple_interval.lower
                    )
                    bounds.append(
                        field <= simple_interval.upper
                        if simple_interval.right == Bound.CLOSED
                        else field < simple_interval.upper
                    )
                simple_event_conditions.append(chained_logic(AND, *bounds))

        condition = chained_logic(OR, *simple_event_conditions)
        chain_root.where(condition)
        return condition

    @staticmethod
    def _floatlike_field(variable: MappedVariable, field_name: str) -> MappedVariable:
        """
        :param variable: The eql variable to read the field from.
        :param field_name: The name of the field.
        :return: The field, accessed symbolically on ``variable``.
        :raises MissingFloatLikeFieldError: If the field does not exist or is not
            float-like.
        """
        field = variable._get_mapped_variable_(Attribute, field_name)
        resolved_type = GraphOfBoundingBoxes._resolved_type_of(field)
        if resolved_type is not float:
            raise MissingFloatLikeFieldError(variable, field_name, resolved_type)
        return field

    @staticmethod
    def _resolved_type_of(field: MappedVariable) -> Optional[Type]:
        """
        Resolve the domain type a mapping chain's leaf field holds.

        ``field._type_`` itself does not resolve when the chain is rooted at a query
        rather than at an already-typed variable, since a query's own ``_type_`` is
        unset -- what it selects lives on its ``selected_variable`` instead. Walking
        the chain's own access path against that seed type sidesteps the gap.

        :param field: The field to resolve the type of.
        :return: The field's domain type, or None if it does not exist.
        """
        root = field._chain_root_
        selected_variable = GraphOfBoundingBoxes._selected_variable_of(root)
        owner_type = (
            selected_variable._type_
            if selected_variable is not None
            else root.__dict__.get("_type_")
        )
        for step in field._access_path_:
            owner_type = get_field_type_endpoint(owner_type, step._attribute_name_)
        return owner_type

    @staticmethod
    def _selected_variable_of(root) -> Optional[CanBehaveLikeAVariable]:
        """
        :param root: The chain root of an eql variable.
        :return: ``root.selected_variable`` if ``root`` is a query with a real
            ``selected_variable`` property, or None otherwise.

        .. note::
            Checked via the class rather than ``hasattr(root, "selected_variable")``:
            every eql variable answers any attribute name through ``__getattr__``, by
            fabricating a new symbolic attribute rather than raising, so probing the
            instance can never tell a real property from one that does not exist.

        :raises AmbiguousSelectedVariableError: If ``root`` selects more than one
            variable, so ``root.selected_variable`` would silently pick the first one
            rather than the one the caller actually meant to constrain.
        """
        if not hasattr(type(root), "selected_variable"):
            return None
        # ``root._selected_variables_`` is a genuine dataclass field of ``root``, always
        # set by ``__init__``, so reading it never falls through to ``__getattr__`` the
        # way probing an arbitrary name would.
        selected_variable_count = len(root._selected_variables_)
        if selected_variable_count != 1:
            raise AmbiguousSelectedVariableError(root, selected_variable_count)
        return root.selected_variable


@dataclass
class VolumetricGraphOfBoundingBoxes(
    GraphOfBoundingBoxes[BoundingBox, Point3, BoundingBoxCollection]
):
    """
    A graph of convex sets whose nodes are axis-aligned bounding boxes, partitioning
    free space in all three dimensions.
    """

    def _default_search_space(self) -> BoundingBoxCollection:
        """
        :return: A search space spanning the entire three-dimensional space around
            ``self.world.root``.
        """
        return BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=-np.inf,
                    min_y=-np.inf,
                    min_z=-np.inf,
                    max_x=np.inf,
                    max_y=np.inf,
                    max_z=np.inf,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=self.world.root
                    ),
                )
            ],
            reference_frame=self.world.root,
        )

    @classmethod
    def obstacles_from_semantic_annotations(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles in a list of semantic annotations.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation to create the
            connectivity graph from.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: An event representing the obstacles in the search space.
        """
        bloated_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )
        return cls.obstacles_from_bounding_boxes(bloated_obstacles, search_space.event)

    @classmethod
    def obstacles_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles from a list of bounding boxes.

        :param bounding_boxes: The list of bounding boxes to create the event from.
        :param search_space_event: The search space event to limit the event to.
        :return: An event representing the obstacles in the search space, or None if no
            obstacles are found.
        """
        events = (
            bb.simple_event.as_composite_set() & search_space_event
            for bb in bounding_boxes
        )
        events = (event for event in events if not event.is_empty())

        try:
            return reduce(or_, events)
        except TypeError:
            logger.warning(
                "No obstacles found in the given semantic annotations. Returning None."
            )
            return None

    @classmethod
    def free_space_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
    ) -> Event:
        """
        Compute the free space by subtracting each obstacle bounding box from the search
        space incrementally (subtract_disjoint), avoiding complement in the full ambient
        space and the costly union-then-complement pipeline.

        This is 40-50× faster than
        ``~obstacles_from_bounding_boxes(...) & search_space_event``
        because:
        - The subtraction stays bounded inside search_space_event at every step.
        - No make_disjoint() calls are needed (disjointness is maintained by construction).
        - The intermediate obstacle union is never materialised.

        :param bounding_boxes: The obstacle bounding boxes to subtract.
        :param search_space_event: The search space; the result is always a subset of this.
        :return: The free space as a disjoint Event.
        """
        free_space = search_space_event
        for bounding_box in bounding_boxes:
            obstacle = bounding_box.simple_event.as_composite_set()
            obstacle_in_search = obstacle & search_space_event
            if not obstacle_in_search.is_empty():
                free_space = free_space.subtract_disjoint(obstacle_in_search)
            if free_space.is_empty():
                break
        return free_space

    @classmethod
    def free_space_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Self:
        """
        Create a connectivity graph from the free space in the belief state of the
        robot.

        :param search_space: The search space for the connectivity graph.
        :param semantic_obstacle_annotation: The semantic annotation containing the
            obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        bloated_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )

        search_event = search_space.event

        start_time = time.time_ns()
        # compute free space via bounded incremental subtraction (avoids complement in ℝ³)
        free_space = cls.free_space_from_bounding_boxes(bloated_obstacles, search_event)
        logger.info(
            f"Free space calculated in {(time.time_ns() - start_time) / 1e6} ms"
        )

        # create a connectivity graph from the free space and calculate the edges
        result = cls(
            search_space=search_space, world=semantic_obstacle_annotation._world
        )
        [
            result.add_node(bounding_box)
            for bounding_box in BoundingBoxCollection.from_event(
                reference_frame=search_space.reference_frame,
                event=free_space,
            )
        ]

        start_time = time.time_ns()
        result.calculate_connectivity(tolerance)
        logger.info(
            f"Connectivity calculated in {(time.time_ns() - start_time) / 1e6} ms"
        )

        return result

    @classmethod
    def free_space_from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
    ) -> Self:
        """
        Create a connectivity graph from the free space in the belief state of the
        robot.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :return: The connectivity graph.
        """
        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return cls.free_space_from_semantic_annotation(
            search_space=search_space,
            semantic_obstacle_annotation=semantic_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
        )

    @classmethod
    def obstacles_from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        bloat_obstacles: float = 0.0,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles in the belief state of the robot.

        :param world: The belief state.
        :param search_space: The search space for the connectivity graph.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :return: An event representing the obstacles in the search space.
        """
        view = SemanticEnvironmentAnnotation(root=world.root, _world=world)

        return cls.obstacles_from_semantic_annotations(
            search_space=search_space,
            semantic_obstacle_annotation=view,
            bloat_obstacles=bloat_obstacles,
        )

    def create_as_region(
        self,
        name: Optional[PrefixedName] = None,
        color: Optional[Color] = None,
    ) -> Region:
        """
        Spawn the GCS as a region (world_entity) connected with a fixed connection with
        the root of the GCS search space. The geometry should be all boxes extracted
        from its free space.

        :param name: The name of the region.
        :param color: The color of the region. Defaults to a translucent green.
        :return: The region.
        """
        if name is None:
            name = PrefixedName("gcs_region")
        if color is None:
            color = Color(0.5, 1.0, 0.5, 0.5)

        bbox_collection = BoundingBoxCollection(
            shapes=list(self.graph.nodes()),
            reference_frame=self.search_space.reference_frame,
        )

        shapes = bbox_collection.as_shapes()
        shapes.dye_shapes(color)
        region = Region.from_shape_collection(name, shapes)

        with self.world.modify_world():
            self.world.add_region(region)

            self.world.add_connection(
                FixedConnection(
                    parent=self.search_space.reference_frame,
                    child=region,
                )
            )
        return region


@dataclass
class PlanarGraphOfBoundingBoxes(
    GraphOfBoundingBoxes[BoundingBox2D, Pose2D, BoundingBoxCollection2D]
):
    """
    A graph of convex sets whose nodes are axis-aligned bounding boxes, partitioning
    free space on a single navigable plane.

    Built for base navigation: the z-axis never enters the decomposition, so a query
    only has to answer whether an x,y footprint is free -- and an obstacle blocks a
    footprint at every height, since the robot has to fit through the entire column of
    space above it, not just its floor-level silhouette.
    """

    def _default_search_space(self) -> BoundingBoxCollection2D:
        """
        :return: A search space spanning the entire two-dimensional plane around
            ``self.world.root``.
        """
        return BoundingBoxCollection2D(
            shapes=[
                BoundingBox2D(
                    min_x=-np.inf,
                    min_y=-np.inf,
                    max_x=np.inf,
                    max_y=np.inf,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=self.world.root
                    ),
                )
            ],
            reference_frame=self.world.root,
        )

    @classmethod
    def obstacles_from_semantic_annotations(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles' floor footprint in a list of
        semantic annotations.

        :param search_space: The three-dimensional search space for the connectivity
            graph -- its height range bounds which obstacles count as blocking.
        :param semantic_obstacle_annotation: The semantic annotation to create the
            connectivity graph from.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: An event representing the obstacles' floor footprint in the search
            space.
        """
        bloated_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )
        return cls.obstacles_from_bounding_boxes(bloated_obstacles, search_space.event)

    @classmethod
    def obstacles_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles' floor footprint from a list of
        bounding boxes.

        :param bounding_boxes: The list of bounding boxes to create the event from.
        :param search_space_event: The three-dimensional search space event to limit
            the event to before flattening onto the floor plane.
        :return: An event representing the obstacles' floor footprint in the search
            space, or None if no obstacles are found.
        """
        search_space_event = search_space_event.marginal(SpatialVariables.xy)
        events = (
            bb.simple_event.as_composite_set().marginal(SpatialVariables.xy)
            & search_space_event
            for bb in bounding_boxes
        )
        events = (event for event in events if not event.is_empty())

        try:
            return reduce(or_, events)
        except TypeError:
            logger.warning(
                "No obstacles found in the given semantic annotations. Returning None."
            )
            return None

    @classmethod
    def free_space_from_bounding_boxes(
        cls,
        bounding_boxes: BoundingBoxCollection,
        search_space_event: Event,
    ) -> Event:
        """
        Compute the floor-plan free space by subtracting each obstacle's floor
        footprint from the search space's floor footprint incrementally
        (subtract_disjoint), the same way :meth:`VolumetricGraphOfBoundingBoxes.free_space_from_bounding_boxes`
        does in three dimensions, but marginalized onto the x,y plane first: an
        obstacle blocks a footprint at every height, since the robot has to fit
        through the entire column of space above it, not just its floor-level
        silhouette.

        :param bounding_boxes: The obstacle bounding boxes to subtract.
        :param search_space_event: The three-dimensional search space; the result is
            always a subset of its floor footprint.
        :return: The floor-plan free space as a disjoint, two-dimensional Event.
        """
        free_space = search_space_event.marginal(SpatialVariables.xy)
        for bounding_box in bounding_boxes:
            obstacle = bounding_box.simple_event.as_composite_set().marginal(
                SpatialVariables.xy
            )
            obstacle_in_search = obstacle & free_space
            if not obstacle_in_search.is_empty():
                free_space = free_space.subtract_disjoint(obstacle_in_search)
            if free_space.is_empty():
                break
        return free_space

    @classmethod
    def obstacles_from_world(
        cls,
        world: World,
        search_space: BoundingBoxCollection,
        bloat_obstacles: float = 0.0,
    ) -> Optional[Event]:
        """
        Create an event representing the obstacles' floor footprint in the belief
        state of the robot.

        :param world: The belief state.
        :param search_space: The three-dimensional search space for the connectivity
            graph.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :return: An event representing the obstacles' floor footprint in the search
            space.
        """
        view = SemanticEnvironmentAnnotation(root=world.root, _world=world)

        return cls.obstacles_from_semantic_annotations(
            search_space=search_space,
            semantic_obstacle_annotation=view,
            bloat_obstacles=bloat_obstacles,
        )

    @classmethod
    def navigation_map_from_semantic_annotation(
        cls,
        search_space: BoundingBoxCollection,
        semantic_obstacle_annotation: SemanticAnnotation,
        semantic_wall_annotation: Optional[SemanticAnnotation] = None,
        tolerance=0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param search_space: The three-dimensional search space for the connectivity
            graph -- its height range bounds which obstacles count as blocking, since
            the robot has to fit through the entire space, not just the floor-level
            obstacles. The graph's own :attr:`search_space` is this volume's floor
            footprint.
        :param semantic_obstacle_annotation: The semantic annotation containing the
            obstacles.
        :param semantic_wall_annotation: An optional semantic annotation containing
            walls to be considered as obstacles.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :return: The connectivity graph. If no obstacles are found, an empty graph is
            returned.
        """
        world = search_space.reference_frame._world
        floor_search_space = BoundingBoxCollection2D.from_event(
            search_space.reference_frame,
            search_space.event.marginal(SpatialVariables.xy),
        )

        nav_obstacles = cls._build_bloated_obstacle_collection(
            search_space,
            semantic_obstacle_annotation,
            semantic_wall_annotation,
            bloat_obstacles,
            bloat_walls,
        )

        if not nav_obstacles:
            return cls(world=world, search_space=floor_search_space)

        free_space = cls.free_space_from_bounding_boxes(
            nav_obstacles, search_space.event
        )

        # create a connectivity graph from the free space and calculate the edges
        result = cls(world=world, search_space=floor_search_space)
        free_space_boxes = BoundingBoxCollection2D.from_event(
            search_space.reference_frame, free_space
        )
        [result.add_node(bounding_box) for bounding_box in free_space_boxes]
        result.calculate_connectivity(tolerance)

        return result

    @classmethod
    def navigation_map_from_world(
        cls,
        world: World,
        tolerance=0.001,
        search_space: Optional[BoundingBoxCollection] = None,
        bloat_obstacles: float = 0.0,
    ) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for
        navigation. The resulting GCS describes the paths for navigation, meaning that
        changing the z-axis position is not possible. Furthermore, it is taken into
        account that the robot has to fit through the entire space and not just through
        the floor level obstacles.

        :param world: The belief state.
        :param search_space: The three-dimensional search space for the connectivity
            graph.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :return: The connectivity graph.
        """
        semantic_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return cls.navigation_map_from_semantic_annotation(
            search_space,
            semantic_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
        )

    def create_as_region(
        self,
        slab_height: float,
        name: Optional[PrefixedName] = None,
        color: Optional[Color] = None,
    ) -> Region:
        """
        Spawn the GCS as a region (world_entity) connected with a fixed connection with
        the root of the GCS search space. Since a floor-plan box has no z-extent of its
        own to spawn, each one is extruded into a thin 3D slab of ``slab_height``,
        centered on the plane the boxes were built on.

        :param slab_height: The thickness of the spawned slab.
        :param name: The name of the region.
        :param color: The color of the region. Defaults to a translucent green.
        :return: The region.
        """
        if name is None:
            name = PrefixedName("gcs_region")
        if color is None:
            color = Color(0.5, 1.0, 0.5, 0.5)

        half_height = slab_height / 2
        bbox_collection = BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    box.min_x,
                    box.min_y,
                    -half_height,
                    box.max_x,
                    box.max_y,
                    half_height,
                    box.origin,
                )
                for box in self.graph.nodes()
            ],
            reference_frame=self.search_space.reference_frame,
        )

        shapes = bbox_collection.as_shapes()
        shapes.dye_shapes(color)
        region = Region.from_shape_collection(name, shapes)

        with self.world.modify_world():
            self.world.add_region(region)

            self.world.add_connection(
                FixedConnection(
                    parent=self.search_space.reference_frame,
                    child=region,
                )
            )
        return region


def hardest_path_query(
    graph: GraphOfBoundingBoxes[BoxT, PointT, SearchSpaceT],
) -> PathQuery[PointT]:
    """
    Pick the query that is hardest to answer: the two convex set centers whose shortest
    path through the graph is the longest one it holds.

    Distance is measured along the graph rather than straight-line, so the query lands
    on the pair the environment actually forces a detour between instead of on whichever
    two sets happen to sit in opposite corners of an open room. Centers of convex sets
    are free by construction, and a pair connected by a path is solvable by definition.
    Ties are broken by coordinate rather than by the order the graph happens to hold its
    nodes in, so that the same graph always yields the same query.

    :param graph: The graph to query.
    :return: The query.
    :raises UnconnectedGraphError: If no two convex sets are connected.
    """
    path_lengths = rx.all_pairs_dijkstra_path_lengths(
        graph.graph, edge_cost_fn=lambda adjacency: adjacency.distance
    )
    # BoundingBox.center recomputes symbolic arithmetic on every access, and the
    # tie-break below reads one per pair, so every center is resolved to floats once
    # here instead.
    coordinates = {
        index: _coordinates_of(graph, index) for index in graph.graph.node_indices()
    }
    # Each connected pair once, so which end is named start never depends on which
    # direction the search happened to report first.
    connected_pairs = [
        (source, target)
        for source, targets in path_lengths.items()
        for target in targets
        if source < target
    ]
    if not connected_pairs:
        raise UnconnectedGraphError(graph.graph.num_nodes())

    # Ties are broken on the pair's coordinates rather than on its graph indices, which
    # depend on the order the world happened to yield its obstacles in.
    most_distant_pair = max(
        connected_pairs,
        key=lambda pair: (
            path_lengths[pair[0]][pair[1]],
            *sorted((coordinates[pair[0]], coordinates[pair[1]])),
        ),
    )
    start_index, goal_index = sorted(
        most_distant_pair, key=lambda index: coordinates[index]
    )
    return PathQuery(
        start=graph.graph[start_index].center, goal=graph.graph[goal_index].center
    )


def _coordinates_of(
    graph: GraphOfBoundingBoxes[BoxT, PointT, SearchSpaceT], index: int
) -> tuple[float, float, float]:
    """
    :param graph: The graph holding the convex set.
    :param index: The index of the convex set in that graph.
    :return: The center of the convex set, as plain floats to compare by. The third
        coordinate reads as 0 for a planar graph, whose centers have no z.
    """
    center = graph.graph[index].center
    return float(center.x), float(center.y), float(center.z)


def navigation_map_at_target(
    target: Body,
    search_range_x: float = 2.0,
    search_range_y: float = 2.0,
    max_height: float = 2.0,
    bloat_obstacles: float = 0.02,
) -> PlanarGraphOfBoundingBoxes:
    """
    Create a navigation map around the target.

    The navigation map is a Graph of Convex Sets that represents the navigable space
    around the target. The search space is constructed as a box around the target with
    the specified search ranges in the x and y directions.

    :param target: The target around which the navigation map is created.
    :param search_range_x: The search range in the x-direction.
    :param search_range_y: The search range in the y-direction.
    :param max_height: The maximum height of the navigation map from the floor.
    :param bloat_obstacles: The amount to bloat obstacles in the navigation map.
    :return: The navigation map as a Graph of Convex Sets.
    """
    search_space = BoundingBoxCollection.from_simple_event(
        reference_frame=target,
        simple_event=SimpleEvent.from_data(
            {
                SpatialVariables.x.value: closed(
                    -search_range_x / 2, search_range_x / 2
                ),
                SpatialVariables.y.value: closed(
                    -search_range_y / 2, search_range_y / 2
                ),
                SpatialVariables.z.value: closed(
                    -target.global_pose.z, max_height - target.global_pose.z
                ),
            }
        ),
    )

    return PlanarGraphOfBoundingBoxes.navigation_map_from_world(
        world=target._world, search_space=search_space, bloat_obstacles=bloat_obstacles
    )
