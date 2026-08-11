"""
Renders a three-panel figure explaining what a Graph of Convex Sets (GCS) does with an
environment.

The panels answer three questions in order, all drawn as a top-down view of the same
x-y extent so they can be read against each other:

  * **Environment** -- what the planner is given: the obstacle footprints of a world and
    the search space bounding them.

  * **Graph of convex sets** -- what the planner builds: the exact partition of free
    space into axis-aligned convex sets, and the adjacency graph over them.

  * **Optimal path** -- what the planner returns: the minimum-distance sequence of
    waypoints between the two convex sets the environment forces the longest detour
    between.

The scene is built by :class:`NavigationScene` and drawn by
:class:`GraphOfConvexSetsFigure`.
"""

from __future__ import annotations

import enum
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing_extensions import ClassVar, Iterable, List, Optional, Sequence, Self

from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kActionableScenes,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    EmptyFreeSpaceError,
    UnconnectedGraphError,
    UnreachableGoalError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% palette


@dataclass(frozen=True)
class FigurePalette:
    """
    The colors a figure draws with, one field per role rather than per hue, so the same
    entity keeps its color across all panels.
    """

    surface: str
    """
    The color the figure and its panels are painted on, and the color separating
    adjacent fills from each other.
    """

    text_primary: str
    """
    The color of titles and axis labels.
    """

    text_secondary: str
    """
    The color of panel subtitles, tick labels and the search-space outline.
    """

    obstacle: str
    """
    The color of obstacle footprints.
    """

    obstacle_edge: str
    """
    A darker step of :attr:`obstacle`, separating adjacent obstacles from each other.
    """

    convex_set: str
    """
    The color filling a convex set of free space.
    """

    convex_set_edge: str
    """
    A darker step of :attr:`convex_set`, separating neighbouring sets from each other.
    """

    receded_convex_set: str
    """
    A lighter step of :attr:`convex_set`, used where the sets are context rather than
    subject.
    """

    adjacency: str
    """
    The color of the adjacency graph's nodes and edges.
    """

    path: str
    """
    The color of the optimal path.
    """

    start: str
    """
    The color of the start marker.
    """

    goal: str
    """
    The color of the goal marker.
    """

    @classmethod
    def light(cls) -> Self:
        """
        :return: The palette stepped for a light surface.
        """
        return cls(
            surface="#fcfcfb",
            text_primary="#0b0b0b",
            text_secondary="#52514e",
            obstacle="#c9c8c2",
            obstacle_edge="#a8a7a1",
            convex_set="#b7d3f6",
            convex_set_edge="#86b6ef",
            receded_convex_set="#cde2fb",
            adjacency="#184f95",
            path="#eb6834",
            start="#1baf7a",
            goal="#e34948",
        )

    @classmethod
    def dark(cls) -> Self:
        """
        :return: The palette stepped for a dark surface, chosen against that surface
            rather than inverted from :meth:`light`.
        """
        return cls(
            surface="#1a1a19",
            text_primary="#ffffff",
            text_secondary="#c3c2b7",
            obstacle="#4d4c48",
            obstacle_edge="#6b6a65",
            convex_set="#1c5cab",
            convex_set_edge="#3987e5",
            receded_convex_set="#123a6b",
            adjacency="#86b6ef",
            path="#d95926",
            start="#199e70",
            goal="#e66767",
        )


class Theme(enum.Enum):
    """
    The surface a figure is rendered for.
    """

    LIGHT = "light"
    DARK = "dark"

    @property
    def palette(self) -> FigurePalette:
        """
        :return: The palette stepped for this theme's surface.
        """
        return FigurePalette.light() if self is Theme.LIGHT else FigurePalette.dark()


# %% the scene a figure draws


@dataclass(eq=False)
class NavigationObstacleAnnotation(SemanticAnnotation):
    """
    The bodies a navigating robot has to travel around, as an annotation a graph of
    convex sets can be built from.
    """

    obstacles: List[Body] = field(default_factory=list)
    """
    The obstacle bodies.
    """


def sage10k_environment_name(scene_url: str) -> str:
    """
    :param scene_url: URL of a Sage10k scene.
    :return: The label a figure of that scene is titled with, taken from the curated
        scene's name where the URL is one of :class:`Sage10kActionableScenes` and from
        the URL's filename otherwise.
    """
    curated = {str(scene): scene.name.lower() for scene in Sage10kActionableScenes}
    if scene_url in curated:
        return curated[scene_url]
    return Path(urlparse(scene_url).path).stem


@dataclass(frozen=True)
class Footprint:
    """
    The x-y projection of a bounding box, which is the shape a top-down panel draws for
    it.
    """

    min_x: float
    """
    The lower x bound, in the search space's reference frame.
    """

    min_y: float
    """
    The lower y bound, in the search space's reference frame.
    """

    max_x: float
    """
    The upper x bound, in the search space's reference frame.
    """

    max_y: float
    """
    The upper y bound, in the search space's reference frame.
    """

    @classmethod
    def of(cls, bounding_box: BoundingBox) -> Self:
        """
        :param bounding_box: The bounding box to project.
        :return: The projection of that box onto the x-y plane.
        """
        bounds = bounding_box.to_array_bounds()
        return cls(
            min_x=float(bounds.lower[0]),
            min_y=float(bounds.lower[1]),
            max_x=float(bounds.upper[0]),
            max_y=float(bounds.upper[1]),
        )

    @property
    def width(self) -> float:
        """
        :return: The extent along x.
        """
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """
        :return: The extent along y.
        """
        return self.max_y - self.min_y

    def as_rectangle(
        self, facecolor: str, edgecolor: str, linewidth: float, zorder: float
    ) -> Rectangle:
        """
        :param facecolor: The color filling the rectangle.
        :param edgecolor: The color of its outline.
        :param linewidth: The width of its outline in points.
        :param zorder: The drawing order of the rectangle.
        :return: A matplotlib rectangle covering this footprint.
        """
        return Rectangle(
            (self.min_x, self.min_y),
            self.width,
            self.height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )


@dataclass(frozen=True)
class NavigationPath:
    """
    A solved path through a graph of convex sets.
    """

    waypoints: List[Point3]
    """
    The points to navigate to, starting at the query's start and ending at its goal.
    """

    @property
    def length(self) -> float:
        """
        :return: The distance travelled along the path.
        """
        return sum(
            float(current.euclidean_distance(following))
            for current, following in zip(self.waypoints, self.waypoints[1:])
        )

    @property
    def vertical_travel(self) -> float:
        """
        :return: The height gained and lost along the path, which is zero for a path
            planned on a floor plan.
        """
        return sum(
            abs(float(following.z) - float(current.z))
            for current, following in zip(self.waypoints, self.waypoints[1:])
        )

    @property
    def coordinates(self) -> np.ndarray:
        """
        :return: The waypoints as an array of x-y rows, in the order they are visited.
        """
        return np.array(
            [[float(waypoint.x), float(waypoint.y)] for waypoint in self.waypoints]
        )

    @property
    def spatial_coordinates(self) -> np.ndarray:
        """
        :return: The waypoints as an array of x-y-z rows, in the order they are visited.
        """
        return np.array(
            [
                [float(waypoint.x), float(waypoint.y), float(waypoint.z)]
                for waypoint in self.waypoints
            ]
        )


@dataclass(frozen=True)
class ConvexSetAdjacency:
    """
    One edge of the graph, drawn the way the planner crosses it: from the center of one
    convex set, through the portal the two sets share, to the center of the other.
    """

    source_center: Point3
    """
    The center of the convex set the edge starts at.
    """

    portal_center: Point3
    """
    The center of the region where the two convex sets overlap.
    """

    target_center: Point3
    """
    The center of the convex set the edge ends at.
    """

    @property
    def spatial_coordinates(self) -> np.ndarray:
        """
        :return: The edge as an array of x-y-z rows, from source through portal to
            target.
        """
        return np.array(
            [
                [float(point.x), float(point.y), float(point.z)]
                for point in (
                    self.source_center,
                    self.portal_center,
                    self.target_center,
                )
            ]
        )

    @property
    def coordinates(self) -> np.ndarray:
        """
        :return: The edge as an array of x-y rows, from source through portal to target.
        """
        return np.array(
            [
                [float(point.x), float(point.y)]
                for point in (
                    self.source_center,
                    self.portal_center,
                    self.target_center,
                )
            ]
        )


@dataclass(frozen=True)
class NavigationQuery:
    """
    A start and a goal to plan between.
    """

    start: Point3
    """
    Where the path begins.
    """

    goal: Point3
    """
    Where the path ends.
    """

    @classmethod
    def between_most_distant_convex_sets(cls, graph: GraphOfBoundingBoxes) -> Self:
        """
        Pick the query that is hardest to answer: the two convex set centers whose
        shortest path through the graph is the longest one it holds.

        Distance is measured along the graph rather than straight-line, so the query
        lands on the pair the environment actually forces a detour between instead of on
        whichever two sets happen to sit in opposite corners of an open room. Centers of
        convex sets are free by construction, and a pair connected by a path is solvable
        by definition. Ties are broken by coordinate rather than by the order the graph
        happens to hold its nodes in, so that the same environment always yields the
        same query.

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
            index: cls._coordinates_of(graph, index)
            for index in graph.graph.node_indices()
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

        # Ties are broken on the pair's coordinates rather than on its graph indices,
        # which depend on the order the world happened to yield its obstacles in.
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
        return cls(
            start=graph.graph[start_index].center, goal=graph.graph[goal_index].center
        )

    @staticmethod
    def _coordinates_of(
        graph: GraphOfBoundingBoxes, index: int
    ) -> tuple[float, float, float]:
        """
        :param graph: The graph holding the convex set.
        :param index: The index of the convex set in that graph.
        :return: The center of the convex set, as plain floats to compare by.
        """
        center = graph.graph[index].center
        return float(center.x), float(center.y), float(center.z)


@dataclass(frozen=True)
class EnvironmentGeometry:
    """
    A world's collision geometry, split into the ground it rests on and the bodies
    standing on that ground.

    Geometry lying within a step height of the lowest point of the world is ground. What
    the split is for depends on the decomposition: a floor plan has to leave the ground
    out of its obstacles, while a volumetric decomposition treats it as an obstacle like
    any other. Both derive the volume to plan in from it.
    """

    world: World
    """
    The world the geometry belongs to.
    """

    ground_bodies: List[Body]
    """
    The bodies forming the ground.
    """

    standing_bodies: List[Body]
    """
    The bodies standing on the ground.
    """

    ground: BoundingBoxCollection
    """
    The bounding boxes of :attr:`ground_bodies`, in the same order.
    """

    covering_box: BoundingBox
    """
    The minimal box covering all of the world's collision geometry, ground included.
    """

    floor_level: float
    """
    The height, in the world's root frame, below which nothing is navigable.
    """

    @classmethod
    def of(cls, world: World, step_height: float, floor_level: float) -> Self:
        """
        :param world: The world to split.
        :param step_height: Height above the lowest point of the world below which
            geometry counts as ground.
        :param floor_level: The height below which nothing is navigable.
        :return: The split geometry.
        """
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        bodies = annotation.bodies_with_collision
        boxes = cls._bounding_boxes_of(bodies, world)
        ground_level = min(float(box.to_array_bounds().lower[2]) for box in boxes)

        is_ground = [
            float(box.to_array_bounds().upper[2]) <= ground_level + step_height
            for box in boxes
        ]
        ground_bodies = [body for body, ground in zip(bodies, is_ground) if ground]
        return cls(
            world=world,
            ground_bodies=ground_bodies,
            standing_bodies=[
                body for body, ground in zip(bodies, is_ground) if not ground
            ],
            ground=cls._bounding_boxes_of(ground_bodies, world),
            covering_box=boxes.bounding_box(),
            floor_level=floor_level,
        )

    def bounding_boxes_of(self, bodies: List[Body]) -> BoundingBoxCollection:
        """
        :param bodies: The bodies to measure, one bounding box each.
        :return: The bounding boxes, in the world's root frame and in the order the
            bodies were given in.
        """
        return self._bounding_boxes_of(bodies, self.world)

    def navigable_volume(self) -> BoundingBoxCollection:
        """
        Derive the volume to plan in from the ground: a robot may stand wherever there
        is floor under it, and nowhere else.

        Every ground footprint is raised to the full height of the world, so obstacles
        standing on it are subtracted over their whole height. A world without ground
        geometry falls back to the box covering it, which is the tightest volume that
        can be justified without knowing where its floor is. Either way the volume stops
        at :attr:`floor_level`.

        :return: The volume to plan in.
        """
        covering_bounds = self.covering_box.to_array_bounds()
        lower_z = max(self.floor_level, float(covering_bounds.lower[2]))
        upper_z = float(covering_bounds.upper[2])
        footprints = self.ground if self.ground else [self.covering_box]
        return BoundingBoxCollection(
            [
                BoundingBox(
                    min_x=box.min_x,
                    min_y=box.min_y,
                    min_z=lower_z,
                    max_x=box.max_x,
                    max_y=box.max_y,
                    max_z=upper_z,
                    origin=box.origin,
                )
                for box in footprints
            ],
            self.covering_box.origin.reference_frame,
        )

    @staticmethod
    def _bounding_boxes_of(bodies: List[Body], world: World) -> BoundingBoxCollection:
        """
        :param bodies: The bodies to measure, one bounding box each.
        :param world: The world the bodies belong to.
        :return: The bounding boxes, in the world's root frame and in the order the
            bodies were given in.
        """
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        return BoundingBoxCollection(
            [
                body.collision.as_bounding_box_collection_at_origin(
                    origin
                ).bounding_box()
                for body in bodies
            ],
            world.root,
        )


@dataclass(frozen=True)
class FreeSpaceDecomposition(ABC):
    """
    How a world's free space is turned into a graph of convex sets, which decides both
    what counts as an obstacle and how many dimensions a path may move in.
    """

    def graph_of(
        self, geometry: EnvironmentGeometry, clearance: float
    ) -> GraphOfBoundingBoxes:
        """
        :param geometry: The world's geometry, already split into ground and the bodies
            standing on it.
        :param clearance: Amount the obstacles are bloated by.
        :return: The graph of convex sets covering the world's free space.
        """
        return self.decompose(
            geometry.navigable_volume(),
            NavigationObstacleAnnotation(
                obstacles=self.obstacle_bodies(geometry), _world=geometry.world
            ),
            clearance,
        )

    @abstractmethod
    def obstacle_bodies(self, geometry: EnvironmentGeometry) -> List[Body]:
        """
        :param geometry: The world's geometry, already split into ground and the bodies
            standing on it.
        :return: The bodies to subtract from the search space.
        """
        raise NotImplementedError

    @abstractmethod
    def decompose(
        self,
        search_space: BoundingBoxCollection,
        obstacles: SemanticAnnotation,
        clearance: float,
    ) -> GraphOfBoundingBoxes:
        """
        :param search_space: The volume to decompose.
        :param obstacles: The obstacles to subtract from it.
        :param clearance: Amount the obstacles are bloated by.
        :return: The graph of convex sets covering what is left.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class FloorPlanDecomposition(FreeSpaceDecomposition):
    """
    Free space as a floor plan: obstacles are projected onto the floor plane, so a path
    moves in x and y at a fixed height.

    The ground is left out of the obstacles, because projecting a floor slab onto the
    floor plane would close the whole environment.
    """

    def obstacle_bodies(self, geometry: EnvironmentGeometry) -> List[Body]:
        return geometry.standing_bodies

    def decompose(
        self,
        search_space: BoundingBoxCollection,
        obstacles: SemanticAnnotation,
        clearance: float,
    ) -> GraphOfBoundingBoxes:
        return GraphOfBoundingBoxes.navigation_map_from_semantic_annotation(
            search_space, obstacles, bloat_obstacles=clearance
        )


@dataclass(frozen=True)
class VolumetricDecomposition(FreeSpaceDecomposition):
    """
    Free space as a three-dimensional partition: a path may change height, passing over
    what it cannot pass beside.

    The ground is an obstacle here like any other body, since nothing may pass through
    it.
    """

    def obstacle_bodies(self, geometry: EnvironmentGeometry) -> List[Body]:
        return geometry.ground_bodies + geometry.standing_bodies

    def decompose(
        self,
        search_space: BoundingBoxCollection,
        obstacles: SemanticAnnotation,
        clearance: float,
    ) -> GraphOfBoundingBoxes:
        return GraphOfBoundingBoxes.free_space_from_semantic_annotation(
            search_space, obstacles, bloat_obstacles=clearance
        )


@dataclass
class NavigationScene:
    """
    Everything one figure draws: an environment, the graph of convex sets covering its
    free space, and one query solved over that graph.
    """

    environment_name: str
    """
    The label the figure is titled with.
    """

    search_space: BoundingBoxCollection
    """
    The volume the graph of convex sets was built in, and the extent every panel is
    framed to.
    """

    obstacles: BoundingBoxCollection
    """
    The bounding boxes of the environment's collision geometry, unbloated, so that the
    clearance the planner keeps from them stays visible.
    """

    graph_of_convex_sets: GraphOfBoundingBoxes
    """
    The decomposition of free space into convex sets, and their adjacency graph.
    """

    query: NavigationQuery
    """
    The start and goal the path was planned between.
    """

    path: NavigationPath
    """
    The minimum-distance path the graph of convex sets returned for :attr:`query`.
    """

    DEFAULT_CLEARANCE: ClassVar[float] = 0.2
    """
    Default amount obstacles are bloated by, standing in for the radius of the robot
    that has to fit past them.
    """

    STEP_HEIGHT: ClassVar[float] = 0.05
    """
    Height above the ground below which geometry is driven over rather than around, so
    that floor slabs and thresholds do not close the whole environment.
    """

    FLOOR_LEVEL: ClassVar[float] = 0.0
    """
    Default height below which nothing is navigable.

    The environments drawn here put their floor plane on the world frame's ``z = 0``, so
    a body modelled reaching below it is a modelling error rather than space to plan in.
    Pass a different level for a world whose floor is somewhere else; the volume is
    never lowered past the world's own geometry, so a scene standing well above zero
    stays tight to itself.
    """

    @classmethod
    def from_world(
        cls,
        world: World,
        environment_name: str,
        clearance: float = DEFAULT_CLEARANCE,
        decomposition: Optional[FreeSpaceDecomposition] = None,
        floor_level: float = FLOOR_LEVEL,
    ) -> Self:
        """
        Build the scene of a world, planning between the two convex sets it forces the
        longest detour between.

        :param world: The world to decompose.
        :param environment_name: The label the figure is titled with.
        :param clearance: Amount obstacles are bloated by while building the graph.
        :param decomposition: How free space is decomposed; defaults to a floor plan.
        :param floor_level: The height below which nothing is navigable.
        :return: The scene.
        :raises EmptyFreeSpaceError: If the world leaves no free space to plan in.
        :raises UnreachableGoalError: If the graph contains no path for its own query.
        """
        decomposition = decomposition or FloorPlanDecomposition()
        geometry = EnvironmentGeometry.of(world, cls.STEP_HEIGHT, floor_level)
        graph = decomposition.graph_of(geometry, clearance)
        if not graph.graph.num_nodes():
            raise EmptyFreeSpaceError(environment_name)

        query = NavigationQuery.between_most_distant_convex_sets(graph)
        waypoints = graph.path_from_to(query.start, query.goal)
        if waypoints is None:
            raise UnreachableGoalError(query.start, query.goal)

        return cls(
            environment_name=environment_name,
            search_space=geometry.navigable_volume(),
            obstacles=geometry.bounding_boxes_of(
                decomposition.obstacle_bodies(geometry)
            ),
            graph_of_convex_sets=graph,
            query=query,
            path=NavigationPath(waypoints),
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        clearance: float = DEFAULT_CLEARANCE,
        decomposition: Optional[FreeSpaceDecomposition] = None,
        floor_level: float = FLOOR_LEVEL,
    ) -> Self:
        """
        Build the scene of a URDF environment.

        :param urdf_path: Path of the ``.urdf`` file to load.
        :param clearance: Amount obstacles are bloated by while building the graph.
        :param decomposition: How free space is decomposed; defaults to a floor plan.
        :param floor_level: The height below which nothing is navigable.
        :return: The scene.
        """
        return cls.from_world(
            URDFParser.from_file(str(urdf_path)).parse(),
            environment_name=urdf_path.stem,
            clearance=clearance,
            decomposition=decomposition,
            floor_level=floor_level,
        )

    @classmethod
    def from_sage10k_scene(
        cls,
        loader: Sage10kDatasetLoader,
        scene_url: str,
        clearance: float = DEFAULT_CLEARANCE,
        decomposition: Optional[FreeSpaceDecomposition] = None,
        floor_level: float = FLOOR_LEVEL,
    ) -> Self:
        """
        Build the scene of a Sage10k scene, downloading and caching it on first use.

        :param loader: The loader holding the directory scenes are cached in.
        :param scene_url: URL of the Sage10k scene to draw.
        :param clearance: Amount obstacles are bloated by while building the graph.
        :param decomposition: How free space is decomposed; defaults to a floor plan.
        :param floor_level: The height below which nothing is navigable.
        :return: The scene.
        """
        return cls.from_world(
            loader.create_scene(scene_url=scene_url).create_world(),
            environment_name=sage10k_environment_name(scene_url),
            clearance=clearance,
            decomposition=decomposition,
            floor_level=floor_level,
        )

    @property
    def extent(self) -> Footprint:
        """
        :return: The x-y extent every panel is framed to.
        """
        return Footprint.of(self.search_space.bounding_box())

    @property
    def convex_sets(self) -> List[BoundingBox]:
        """
        :return: The convex sets partitioning free space.
        """
        return list(self.graph_of_convex_sets.graph.nodes())

    @property
    def adjacencies(self) -> List[ConvexSetAdjacency]:
        """
        :return: One adjacency per edge of the graph.
        """
        graph = self.graph_of_convex_sets.graph
        return [
            ConvexSetAdjacency(
                source_center=graph[source].center,
                portal_center=adjacency.intersection.center,
                target_center=graph[target].center,
            )
            for source, target, adjacency in graph.weighted_edge_list()
        ]


# %% the layers a panel is drawn from


@dataclass(frozen=True)
class LegendEntry:
    """
    One row of the figure's shared legend.
    """

    handle: Artist
    """
    The sample mark drawn beside the label.
    """

    label: str
    """
    The name of the entity the mark stands for.
    """


@dataclass(frozen=True)
class SceneLayer(ABC):
    """
    One kind of mark a panel draws, so that panels differ in what they compose rather
    than in how any one entity is drawn.
    """

    @abstractmethod
    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        """
        Draw this layer's marks.

        :param axes: The panel to draw into.
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: The legend entries this layer contributes.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SearchSpaceLayer(SceneLayer):
    """
    Draws the outline of every part of the volume the graph of convex sets was built in,
    which in a world with floor geometry is one outline per floor.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        outlines = [
            Footprint.of(part).as_rectangle(
                facecolor="none",
                edgecolor=palette.text_secondary,
                linewidth=0.8,
                zorder=10,
            )
            for part in scene.search_space
        ]
        for outline in outlines:
            outline.set_linestyle((0, (4, 3)))
            axes.add_patch(outline)
        return [LegendEntry(outlines[0], "search space")]


@dataclass(frozen=True)
class ObstacleLayer(SceneLayer):
    """
    Draws the environment's obstacle footprints, outlined in a darker tint of their own
    fill so that abutting obstacles stay countable.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        rectangles = [
            Footprint.of(obstacle).as_rectangle(
                facecolor=palette.obstacle,
                edgecolor=palette.obstacle_edge,
                linewidth=0.4,
                zorder=2,
            )
            for obstacle in scene.obstacles
        ]
        for rectangle in rectangles:
            axes.add_patch(rectangle)
        return [LegendEntry(rectangles[0], "obstacle")] if rectangles else []


class ConvexSetEmphasis(enum.Enum):
    """
    Whether the convex sets are the subject of a panel or its context.
    """

    SUBJECT = enum.auto()
    CONTEXT = enum.auto()

    def fill(self, palette: FigurePalette) -> str:
        """
        :param palette: The colors of the figure.
        :return: The fill the sets take at this emphasis.
        """
        return (
            palette.convex_set
            if self is ConvexSetEmphasis.SUBJECT
            else palette.receded_convex_set
        )

    def edge(self, palette: FigurePalette) -> str:
        """
        :param palette: The colors of the figure.
        :return: The color separating neighbouring sets at this emphasis.
        """
        return (
            palette.convex_set_edge
            if self is ConvexSetEmphasis.SUBJECT
            else palette.convex_set
        )


@dataclass(frozen=True)
class ConvexSetsLayer(SceneLayer):
    """
    Draws the convex sets partitioning free space, outlined in a darker tint of their
    own fill so that the partition is visible without slicing free space into stripes
    where the sets are narrow.
    """

    emphasis: ConvexSetEmphasis = ConvexSetEmphasis.SUBJECT
    """
    Whether the sets are the panel's subject or the background its subject sits on.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        rectangles = [
            Footprint.of(convex_set).as_rectangle(
                facecolor=self.emphasis.fill(palette),
                edgecolor=self.emphasis.edge(palette),
                linewidth=0.4,
                zorder=3,
            )
            for convex_set in scene.convex_sets
        ]
        for rectangle in rectangles:
            axes.add_patch(rectangle)
        if self.emphasis is ConvexSetEmphasis.CONTEXT:
            return []
        return [LegendEntry(rectangles[0], "convex set")] if rectangles else []


@dataclass(frozen=True)
class AdjacencyLayer(SceneLayer):
    """
    Draws the graph over the convex sets: a node at every set's center and an edge
    between every pair of adjacent sets.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        for adjacency in scene.adjacencies:
            coordinates = adjacency.coordinates
            axes.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=palette.adjacency,
                linewidth=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4,
            )
        centers = np.array(
            [
                [float(convex_set.center.x), float(convex_set.center.y)]
                for convex_set in scene.convex_sets
            ]
        )
        axes.plot(
            centers[:, 0],
            centers[:, 1],
            linestyle="none",
            marker="o",
            markersize=3.0,
            color=palette.adjacency,
            markeredgecolor=palette.surface,
            markeredgewidth=0.5,
            zorder=5,
        )
        sample = Line2D(
            [],
            [],
            color=palette.adjacency,
            linewidth=1.0,
            marker="o",
            markersize=3.0,
        )
        return [LegendEntry(sample, "adjacency graph")]


@dataclass(frozen=True)
class PathLayer(SceneLayer):
    """
    Draws the optimal path and the waypoints it turns at.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        coordinates = scene.path.coordinates
        line = axes.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=palette.path,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6,
        )
        axes.plot(
            coordinates[1:-1, 0],
            coordinates[1:-1, 1],
            linestyle="none",
            marker="o",
            markersize=4.0,
            color=palette.path,
            markeredgecolor=palette.surface,
            markeredgewidth=0.6,
            zorder=7,
        )
        return [LegendEntry(line[0], "optimal path")]


@dataclass(frozen=True)
class EndpointsLayer(SceneLayer):
    """
    Draws the start and the goal of the query, each labelled beside its marker so that
    neither is identified by color alone.
    """

    LABEL_OFFSET: ClassVar[float] = 0.18
    """
    Distance in meters between an endpoint's marker and its label.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        return [
            self._draw_endpoint(
                axes, scene.query.start, "start", "o", palette.start, palette
            ),
            self._draw_endpoint(
                axes, scene.query.goal, "goal", "s", palette.goal, palette
            ),
        ]

    def _draw_endpoint(
        self,
        axes: Axes,
        endpoint: Point3,
        label: str,
        marker: str,
        color: str,
        palette: FigurePalette,
    ) -> LegendEntry:
        """
        :param axes: The panel to draw into.
        :param endpoint: The point to mark.
        :param label: The name written beside the marker.
        :param marker: The matplotlib marker to draw.
        :param color: The color of the marker.
        :param palette: The colors of the figure.
        :return: The legend entry of this endpoint.
        """
        x, y = float(endpoint.x), float(endpoint.y)
        drawn = axes.plot(
            [x],
            [y],
            linestyle="none",
            marker=marker,
            markersize=8.0,
            color=color,
            markeredgecolor=palette.surface,
            markeredgewidth=1.0,
            zorder=8,
        )
        axes.annotate(
            label,
            (x, y),
            xytext=(self.LABEL_OFFSET, self.LABEL_OFFSET),
            textcoords="offset fontsize",
            color=palette.text_primary,
            fontsize=8.0,
            zorder=9,
        )
        return LegendEntry(drawn[0], label)


# %% the panels of the figure


@dataclass(frozen=True)
class ScenePanel(ABC):
    """
    One sub-plot of the figure, defined by the layers it composes and the statistic it
    reports about the scene.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The panel's title.
        """
        raise NotImplementedError

    @abstractmethod
    def subtitle(self, scene: NavigationScene) -> str:
        """
        :param scene: The scene the panel draws.
        :return: The line under the title, quantifying what the panel shows.
        """
        raise NotImplementedError

    @abstractmethod
    def layers(self) -> Sequence[SceneLayer]:
        """
        :return: The layers to draw, in the order they are stacked.
        """
        raise NotImplementedError

    def draw(
        self,
        axes: Axes,
        scene: NavigationScene,
        palette: FigurePalette,
        panel_label: str,
    ) -> Sequence[LegendEntry]:
        """
        Draw the whole panel.

        :param axes: The sub-plot to draw into.
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :param panel_label: The enumerating label the title is prefixed with.
        :return: The legend entries the panel's layers contribute.
        """
        self._frame(axes, scene, palette)
        entries = [
            entry
            for layer in self.layers()
            for entry in layer.draw(axes, scene, palette)
        ]
        axes.set_title(
            f"{panel_label} {self.name}",
            loc="left",
            pad=14.0,
            fontsize=11.0,
            fontweight="semibold",
            color=palette.text_primary,
        )
        axes.text(
            0.0,
            1.01,
            self.subtitle(scene),
            transform=axes.transAxes,
            fontsize=8.5,
            color=palette.text_secondary,
        )
        return entries

    def _frame(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> None:
        """
        Give the panel the metric frame every panel shares, so the three read against
        each other.

        :param axes: The sub-plot to frame.
        :param scene: The scene whose extent the panel is framed to.
        :param palette: The colors to draw with.
        """
        extent = scene.extent
        margin = 0.02 * max(extent.width, extent.height)
        axes.set_xlim(extent.min_x - margin, extent.max_x + margin)
        axes.set_ylim(extent.min_y - margin, extent.max_y + margin)
        axes.set_aspect("equal")
        axes.set_facecolor(palette.surface)
        axes.set_xlabel("x [m]", fontsize=9.0, color=palette.text_secondary)
        axes.set_ylabel("y [m]", fontsize=9.0, color=palette.text_secondary)
        axes.tick_params(
            labelsize=8.0, colors=palette.text_secondary, length=3.0, width=0.6
        )
        for spine in axes.spines.values():
            spine.set_visible(False)


@dataclass(frozen=True)
class EnvironmentPanel(ScenePanel):
    """
    Shows what the planner is given: the obstacles of the environment and the search
    space bounding them.
    """

    @property
    def name(self) -> str:
        return "Environment"

    def subtitle(self, scene: NavigationScene) -> str:
        extent = scene.extent
        return (
            f"{len(scene.obstacles)} obstacle footprints in "
            f"{extent.width:.1f} × {extent.height:.1f} m"
        )

    def layers(self) -> Sequence[SceneLayer]:
        return (SearchSpaceLayer(), ObstacleLayer())


@dataclass(frozen=True)
class ConvexSetsPanel(ScenePanel):
    """
    Shows what the planner builds: free space partitioned into convex sets, and the
    graph connecting the sets that touch.
    """

    @property
    def name(self) -> str:
        return "Graph of convex sets"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{len(scene.convex_sets)} convex sets, "
            f"{len(scene.adjacencies)} adjacencies"
        )

    def layers(self) -> Sequence[SceneLayer]:
        return (ObstacleLayer(), ConvexSetsLayer(), AdjacencyLayer())


@dataclass(frozen=True)
class OptimalPathPanel(ScenePanel):
    """
    Shows what the planner returns: the minimum-distance path between the queried start
    and goal, threading through the convex sets.
    """

    @property
    def name(self) -> str:
        return "Optimal path"

    def subtitle(self, scene: NavigationScene) -> str:
        return f"{scene.path.length:.2f} m over {len(scene.path.waypoints)} waypoints"

    def layers(self) -> Sequence[SceneLayer]:
        return (
            ObstacleLayer(),
            ConvexSetsLayer(ConvexSetEmphasis.CONTEXT),
            PathLayer(),
            EndpointsLayer(),
        )


# %% the figure


@dataclass(frozen=True)
class PanelArrangement:
    """
    How the panels are tiled, chosen so that a wide environment is stacked and a compact
    one is placed side by side.
    """

    rows: int
    """
    Number of panel rows.
    """

    columns: int
    """
    Number of panel columns.
    """

    WIDE_ASPECT_RATIO: ClassVar[float] = 1.4
    """
    Width-to-height ratio above which an environment is considered wide.
    """

    MAXIMUM_WIDTH_INCHES: ClassVar[float] = 10.0
    """
    Largest figure width the arrangement will produce.
    """

    PREFERRED_INCHES_PER_METER: ClassVar[float] = 0.5
    """
    Scale used unless it would exceed :attr:`MAXIMUM_WIDTH_INCHES`.
    """

    TITLE_INCHES_PER_ROW: ClassVar[float] = 0.75
    """
    Vertical space reserved per row for its panel's title and axis labels.
    """

    LEGEND_INCHES: ClassVar[float] = 1.0
    """
    Vertical space reserved for the figure title and the shared legend.
    """

    LEGEND_ENTRY_INCHES: ClassVar[float] = 1.6
    """
    Width one legend entry takes, used to decide how many fit on a row.
    """

    @classmethod
    def for_extent(cls, extent: Footprint, panel_count: int) -> Self:
        """
        :param extent: The x-y extent one panel spans.
        :param panel_count: Number of panels to tile.
        :return: The arrangement to tile them with.
        """
        if extent.width > cls.WIDE_ASPECT_RATIO * extent.height:
            return cls(rows=panel_count, columns=1)
        return cls(rows=1, columns=panel_count)

    def figure_size(self, extent: Footprint) -> tuple[float, float]:
        """
        :param extent: The x-y extent one panel spans.
        :return: The figure's width and height in inches, scaled so that a meter is the
            same length in every panel.
        """
        scale = min(
            self.PREFERRED_INCHES_PER_METER,
            self.MAXIMUM_WIDTH_INCHES / (self.columns * extent.width),
        )
        width = self.columns * extent.width * scale
        height = (
            self.rows * (extent.height * scale + self.TITLE_INCHES_PER_ROW)
            + self.LEGEND_INCHES
        )
        return width, height

    def legend_columns(self, extent: Footprint, entry_count: int) -> int:
        """
        Wrap the legend so that a narrow figure does not run it off the page, and spread
        the entries evenly over however many rows that takes rather than leaving a last
        row with one entry in it.

        :param extent: The x-y extent one panel spans.
        :param entry_count: Number of legend entries to lay out.
        :return: How many entries to put on one legend row.
        """
        width, _ = self.figure_size(extent)
        entries_that_fit = max(1, int(width / self.LEGEND_ENTRY_INCHES))
        rows = math.ceil(entry_count / entries_that_fit)
        return math.ceil(entry_count / rows)


@dataclass
class GraphOfConvexSetsFigure:
    """
    The three-panel figure of a scene: its environment, its graph of convex sets, and
    the optimal path over that graph.
    """

    scene: NavigationScene
    """
    The scene every panel draws.
    """

    theme: Theme = Theme.LIGHT
    """
    The surface the figure is rendered for.
    """

    panels: tuple[ScenePanel, ...] = field(
        default_factory=lambda: (
            EnvironmentPanel(),
            ConvexSetsPanel(),
            OptimalPathPanel(),
        )
    )
    """
    The panels to draw, left to right or top to bottom.
    """

    DOTS_PER_INCH: ClassVar[int] = 300
    """
    Resolution of the rendered raster image.
    """

    def render(self) -> Figure:
        """
        Draw the figure.

        :return: The rendered figure, which the caller owns and has to close.
        """
        palette = self.theme.palette
        arrangement = PanelArrangement.for_extent(self.scene.extent, len(self.panels))
        with plt.rc_context(self._rc_parameters(palette)):
            figure, axes_grid = plt.subplots(
                arrangement.rows,
                arrangement.columns,
                figsize=arrangement.figure_size(self.scene.extent),
                layout="constrained",
            )
            entries = self._draw_panels(np.atleast_1d(axes_grid).ravel(), palette)
            figure.suptitle(
                f"Graph of convex sets navigation in {self.scene.environment_name}",
                fontsize=13.0,
                fontweight="semibold",
                color=palette.text_primary,
                x=0.02,
                horizontalalignment="left",
            )
            figure.legend(
                handles=[entry.handle for entry in entries],
                labels=[entry.label for entry in entries],
                loc="outside lower left",
                ncols=arrangement.legend_columns(self.scene.extent, len(entries)),
                frameon=False,
                fontsize=9.0,
                labelcolor=palette.text_secondary,
            )
        return figure

    def save(self, output_directory: Path) -> List[Path]:
        """
        Render the figure and write it as both a vector and a raster file.

        :param output_directory: Directory the files are written to; created if missing.
        :return: The written paths.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"graph_of_convex_sets_{self.scene.environment_name}_{self.theme.value}"
        figure = self.render()
        written = []
        for suffix in (".pdf", ".png"):
            path = output_directory / f"{stem}{suffix}"
            figure.savefig(path, dpi=self.DOTS_PER_INCH)
            written.append(path)
        plt.close(figure)
        return written

    def _draw_panels(
        self, all_axes: Iterable[Axes], palette: FigurePalette
    ) -> List[LegendEntry]:
        """
        Draw every panel and collect one legend entry per distinct label.

        :param all_axes: The sub-plots, in the order the panels are drawn into them.
        :param palette: The colors to draw with.
        :return: The legend entries of the whole figure.
        """
        entries: List[LegendEntry] = []
        seen_labels = set()
        for panel, axes, label in zip(self.panels, all_axes, self._panel_labels()):
            for entry in panel.draw(axes, self.scene, palette, label):
                if entry.label in seen_labels:
                    continue
                seen_labels.add(entry.label)
                entries.append(entry)
        return entries

    def _panel_labels(self) -> List[str]:
        """
        :return: The enumerating labels the panel titles are prefixed with.
        """
        return [
            f"({letter})"
            for letter in itertools.islice(
                (chr(ordinal) for ordinal in range(ord("a"), ord("z") + 1)),
                len(self.panels),
            )
        ]

    def _rc_parameters(self, palette: FigurePalette) -> dict[str, object]:
        """
        :param palette: The colors to draw with.
        :return: The matplotlib settings the figure is rendered under, applied in a
            context so that no global state is changed.
        """
        return {
            "figure.facecolor": palette.surface,
            "savefig.facecolor": palette.surface,
            "axes.facecolor": palette.surface,
            "text.color": palette.text_primary,
            "axes.labelcolor": palette.text_secondary,
            "axes.grid": False,
            "font.size": 9.0,
        }
