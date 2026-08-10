"""
Renders the three-panel Graph of Convex Sets figure as a volume rather than a floor
plan.

The panels answer the same three questions as the two-dimensional figure -- what the
planner is given, what it builds, what it returns -- but over a
:class:`~experiments.graph_of_convex_sets_figure.VolumetricDecomposition`, so free space
is partitioned in all three dimensions and a path may change height to pass over what it
cannot pass beside.

Drawn with plotly, which the graph of convex sets already uses for its own
three-dimensional plots. Every run writes an interactive page next to the static image,
since a single camera angle hides whatever it projects behind something else.

Run this module as a script to write the figure of a URDF environment
(``--environment``) or of one or more Sage10k scenes (``--sage10k``) to disk.
"""

from __future__ import annotations

import argparse
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing_extensions import ClassVar, Iterable, List, Sequence, Self

from experiments.graph_of_convex_sets_figure import (
    FigurePalette,
    NavigationScene,
    Theme,
    VolumetricDecomposition,
    urdf_directory,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kActionableScenes,
)
from semantic_digital_twin.world_description.geometry import BoundingBox, Bounds

# %% turning boxes into drawable geometry


@dataclass(frozen=True)
class BoxGeometry:
    """
    A collection of axis-aligned boxes expressed as the arrays plotly draws from, so
    that any number of boxes costs one trace rather than one trace each.
    """

    corners: np.ndarray
    """
    The boxes' vertices, eight rows of x-y-z per box.
    """

    FACES: ClassVar[tuple[tuple[int, int, int], ...]] = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    """
    The two triangles of each of a box's six faces, as indices into its eight vertices.
    """

    EDGES: ClassVar[tuple[tuple[int, int], ...]] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    """
    The twelve edges of a box, as index pairs into its eight vertices.
    """

    @classmethod
    def of(cls, boxes: Iterable[BoundingBox]) -> Self:
        """
        :param boxes: The boxes to express.
        :return: Their geometry.
        """
        corners = []
        for box in boxes:
            bounds = box.to_array_bounds()
            lower, upper = bounds.lower, bounds.upper
            corners.extend(
                [
                    [lower[0], lower[1], lower[2]],
                    [upper[0], lower[1], lower[2]],
                    [upper[0], upper[1], lower[2]],
                    [lower[0], upper[1], lower[2]],
                    [lower[0], lower[1], upper[2]],
                    [upper[0], lower[1], upper[2]],
                    [upper[0], upper[1], upper[2]],
                    [lower[0], upper[1], upper[2]],
                ]
            )
        return cls(corners=np.array(corners).reshape(-1, 8, 3))

    @property
    def box_count(self) -> int:
        """
        :return: How many boxes the geometry holds.
        """
        return len(self.corners)

    def as_mesh(self, color: str, opacity: float, name: str) -> go.Mesh3d:
        """
        :param color: The color the boxes are filled with.
        :param opacity: How opaque the fill is.
        :param name: The name the trace carries into the legend.
        :return: One solid mesh covering every box.
        """
        vertices = self.corners.reshape(-1, 3)
        offsets = np.arange(self.box_count).repeat(len(self.FACES)) * 8
        faces = np.tile(np.array(self.FACES), (self.box_count, 1)) + offsets[:, None]
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
            hoverinfo="skip",
            name=name,
            legendgroup=name,
        )

    def as_wireframe(self, color: str, width: float, name: str) -> go.Scatter3d:
        """
        :param color: The color of the edges.
        :param width: The width of the edges in pixels.
        :param name: The name the trace carries into the legend.
        :return: One line trace covering every box's edges, which is what makes boxes
            behind other boxes readable at all.
        """
        starts = self.corners[:, [edge[0] for edge in self.EDGES], :]
        ends = self.corners[:, [edge[1] for edge in self.EDGES], :]
        segments = np.full((self.box_count * len(self.EDGES), 3, 3), np.nan)
        segments[:, 0, :] = starts.reshape(-1, 3)
        segments[:, 1, :] = ends.reshape(-1, 3)
        points = segments.reshape(-1, 3)
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            name=name,
            legendgroup=name,
        )


def polyline_trace(
    polylines: Sequence[np.ndarray], color: str, width: float, name: str
) -> go.Scatter3d:
    """
    Join separate polylines into a single trace, separated by gaps.

    :param polylines: The polylines, each an array of x-y-z rows.
    :param color: The color of the lines.
    :param width: The width of the lines in pixels.
    :param name: The name the trace carries into the legend.
    :return: The trace.
    """
    separator = np.full((1, 3), np.nan)
    points = np.vstack([row for polyline in polylines for row in (polyline, separator)])
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
    )


# %% the layers a panel is drawn from


@dataclass(frozen=True)
class VolumeLayer(ABC):
    """
    One kind of geometry a panel draws, so that panels differ in what they compose
    rather than in how any one entity is drawn.
    """

    @abstractmethod
    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        """
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: The traces to add to the panel.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SearchSpaceVolumeLayer(VolumeLayer):
    """
    Draws the volume the graph of convex sets was built in, as a wireframe so that
    everything inside it stays visible.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        geometry = BoxGeometry.of(scene.search_space)
        return [geometry.as_wireframe(palette.text_secondary, 2.0, "search space")]


class ObstacleEmphasis(enum.Enum):
    """
    Whether the obstacles are the subject of a panel or the context something else is
    read against.

    Neither is opaque: a room is seen from outside, so solid walls would hide every
    thing the figure is about. Both steps let what is behind them show through, and the
    wireframe edges are what keep the geometry readable.
    """

    SUBJECT = enum.auto()
    CONTEXT = enum.auto()

    @property
    def opacity(self) -> float:
        """
        :return: How opaque the obstacles are at this emphasis.
        """
        return 0.45 if self is ObstacleEmphasis.SUBJECT else 0.2


@dataclass(frozen=True)
class ObstacleVolumeLayer(VolumeLayer):
    """
    Draws the environment's collision geometry as boxes.
    """

    emphasis: ObstacleEmphasis = ObstacleEmphasis.SUBJECT
    """
    Whether the obstacles are the panel's subject or the context its subject sits in.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        geometry = BoxGeometry.of(scene.obstacles)
        return [
            geometry.as_mesh(palette.obstacle, self.emphasis.opacity, "obstacle"),
            geometry.as_wireframe(palette.obstacle_edge, 1.0, "obstacle"),
        ]


@dataclass(frozen=True)
class ConvexSetVolumeLayer(VolumeLayer):
    """
    Draws the convex sets partitioning free space, translucent enough to see the sets
    behind them and wireframed so the partition itself is legible.
    """

    FILL_OPACITY: ClassVar[float] = 0.08
    """
    How opaque a convex set's fill is; low enough that a stack of them does not turn
    into a solid block.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        geometry = BoxGeometry.of(scene.convex_sets)
        return [
            geometry.as_mesh(palette.convex_set, self.FILL_OPACITY, "convex set"),
            geometry.as_wireframe(palette.convex_set_edge, 1.0, "convex set"),
        ]


@dataclass(frozen=True)
class AdjacencyVolumeLayer(VolumeLayer):
    """
    Draws the graph over the convex sets: a node at every set's center and an edge
    through the portal every pair of adjacent sets shares.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        centers = np.array(
            [
                [
                    float(convex_set.center.x),
                    float(convex_set.center.y),
                    float(convex_set.center.z),
                ]
                for convex_set in scene.convex_sets
            ]
        )
        return [
            polyline_trace(
                [adjacency.spatial_coordinates for adjacency in scene.adjacencies],
                palette.adjacency,
                2.0,
                "adjacency graph",
            ),
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=palette.adjacency),
                hoverinfo="skip",
                name="adjacency graph",
                legendgroup="adjacency graph",
            ),
        ]


@dataclass(frozen=True)
class PathVolumeLayer(VolumeLayer):
    """
    Draws the optimal path and the waypoints it turns at.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        coordinates = scene.path.spatial_coordinates
        return [
            go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode="lines+markers",
                line=dict(color=palette.path, width=8.0),
                marker=dict(size=4.0, color=palette.path),
                hoverinfo="skip",
                name="optimal path",
                legendgroup="optimal path",
            )
        ]


@dataclass(frozen=True)
class EndpointsVolumeLayer(VolumeLayer):
    """
    Draws the start and the goal of the query, each labelled beside its marker so that
    neither is identified by color alone.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        return [
            self._endpoint_trace(scene.query.start, "start", "circle", palette.start),
            self._endpoint_trace(scene.query.goal, "goal", "square", palette.goal),
        ]

    @staticmethod
    def _endpoint_trace(endpoint, label: str, symbol: str, color: str) -> go.Scatter3d:
        """
        :param endpoint: The point to mark.
        :param label: The name written beside the marker.
        :param symbol: The plotly marker symbol to draw.
        :param color: The color of the marker.
        :return: The trace marking the endpoint.
        """
        return go.Scatter3d(
            x=[float(endpoint.x)],
            y=[float(endpoint.y)],
            z=[float(endpoint.z)],
            mode="markers+text",
            marker=dict(size=7.0, color=color, symbol=symbol),
            text=[label],
            textposition="top center",
            hoverinfo="skip",
            name=label,
            legendgroup=label,
        )


# %% the panels of the figure


@dataclass(frozen=True)
class VolumePanel(ABC):
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
    def layers(self) -> Sequence[VolumeLayer]:
        """
        :return: The layers to draw, in the order they are stacked.
        """
        raise NotImplementedError

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> List[go.BaseTraceType]:
        """
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: Every trace the panel's layers contribute.
        """
        return [
            trace for layer in self.layers() for trace in layer.traces(scene, palette)
        ]


@dataclass(frozen=True)
class EnvironmentVolumePanel(VolumePanel):
    """
    Shows what the planner is given: the collision geometry of a world and the volume
    bounding it.
    """

    @property
    def name(self) -> str:
        return "Environment"

    def subtitle(self, scene: NavigationScene) -> str:
        bounds = scene.search_space.bounding_box().to_array_bounds()
        extent = bounds.upper - bounds.lower
        return (
            f"{len(scene.obstacles)} obstacle boxes in "
            f"{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} m"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (SearchSpaceVolumeLayer(), ObstacleVolumeLayer())


@dataclass(frozen=True)
class ConvexSetsVolumePanel(VolumePanel):
    """
    Shows what the planner builds: free space partitioned into convex sets in all three
    dimensions, and the graph connecting the sets that touch.
    """

    @property
    def name(self) -> str:
        return "Graph of convex sets"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{len(scene.convex_sets)} convex sets, "
            f"{len(scene.adjacencies)} adjacencies"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (ConvexSetVolumeLayer(), AdjacencyVolumeLayer())


@dataclass(frozen=True)
class OptimalPathVolumePanel(VolumePanel):
    """
    Shows what the planner returns: the minimum-distance path between the queried start
    and goal, including the height it changes on the way.

    The convex sets are left out here; translucent boxes filling the whole volume would
    hide the one line the panel is about.
    """

    @property
    def name(self) -> str:
        return "Optimal path"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{scene.path.length:.2f} m over {len(scene.path.waypoints)} waypoints, "
            f"{scene.path.vertical_travel:.2f} m of it vertical"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (
            ObstacleVolumeLayer(ObstacleEmphasis.CONTEXT),
            PathVolumeLayer(),
            EndpointsVolumeLayer(),
        )


# %% the figure


@dataclass(frozen=True)
class SceneCamera:
    """
    Where every panel is looked at from, shared by all of them so the three read against
    each other.
    """

    eye_x: float = 1.25
    """
    The viewpoint's x coordinate, in units of the scene's own size.
    """

    eye_y: float = -1.25
    """
    The viewpoint's y coordinate, in units of the scene's own size.
    """

    eye_z: float = 1.15
    """
    The viewpoint's z coordinate, in units of the scene's own size.
    """

    def as_plotly(self) -> dict[str, object]:
        """
        :return: The camera as plotly expects it, with z up.
        """
        return dict(
            eye=dict(x=self.eye_x, y=self.eye_y, z=self.eye_z),
            up=dict(x=0.0, y=0.0, z=1.0),
        )


@dataclass
class GraphOfConvexSetsVolumeFigure:
    """
    The three-panel volume figure of a scene: its environment, its graph of convex sets,
    and the optimal path over that graph.
    """

    scene: NavigationScene
    """
    The scene every panel draws.
    """

    theme: Theme = Theme.LIGHT
    """
    The surface the figure is rendered for.
    """

    camera: SceneCamera = field(default_factory=SceneCamera)
    """
    Where the panels are looked at from.
    """

    panels: Sequence[VolumePanel] = field(
        default_factory=lambda: (
            EnvironmentVolumePanel(),
            ConvexSetsVolumePanel(),
            OptimalPathVolumePanel(),
        )
    )
    """
    The panels to draw, left to right.
    """

    image_formats: Sequence[str] = (".pdf", ".png")
    """
    The static formats written beside the interactive page.

    Rendering these drives a headless browser, so pass an empty sequence where only the
    page is wanted.
    """

    PANEL_WIDTH_PIXELS: ClassVar[int] = 620
    """
    Width one panel is rendered at.
    """

    HEIGHT_PIXELS: ClassVar[int] = 520
    """
    Height the figure is rendered at.
    """

    IMAGE_SCALE: ClassVar[int] = 2
    """
    Factor the raster image is rendered at above the nominal size.
    """

    def render(self) -> go.Figure:
        """
        Draw the figure.

        :return: The rendered figure.
        """
        palette = self.theme.palette
        figure = make_subplots(
            rows=1,
            cols=len(self.panels),
            specs=[[{"type": "scene"}] * len(self.panels)],
            horizontal_spacing=0.02,
        )
        self._add_panels(figure, palette)
        self._style(figure, palette)
        return figure

    def save(self, output_directory: Path) -> List[Path]:
        """
        Render the figure and write it as an interactive page, a vector file and a
        raster file.

        :param output_directory: Directory the files are written to; created if missing.
        :return: The written paths.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"graph_of_convex_sets_volume_{self.scene.environment_name}_"
            f"{self.theme.value}"
        )
        figure = self.render()

        written = [output_directory / f"{stem}.html"]
        figure.write_html(str(written[0]), include_plotlyjs=True)
        for suffix in self.image_formats:
            path = output_directory / f"{stem}{suffix}"
            figure.write_image(
                str(path),
                width=self.PANEL_WIDTH_PIXELS * len(self.panels),
                height=self.HEIGHT_PIXELS,
                scale=self.IMAGE_SCALE,
            )
            written.append(path)
        return written

    def _add_panels(self, figure: go.Figure, palette: FigurePalette) -> None:
        """
        Add every panel's traces, showing each legend entry once across the figure.

        :param figure: The figure to add to.
        :param palette: The colors to draw with.
        """
        shown_groups = set()
        for column, panel in enumerate(self.panels, start=1):
            for trace in panel.traces(self.scene, palette):
                trace.showlegend = trace.legendgroup not in shown_groups
                shown_groups.add(trace.legendgroup)
                figure.add_trace(trace, row=1, col=column)

    def _style(self, figure: go.Figure, palette: FigurePalette) -> None:
        """
        Give the figure its titles, its shared camera and its theme.

        :param figure: The figure to style.
        :param palette: The colors to draw with.
        """
        figure.update_layout(
            title=dict(
                text=(
                    "Graph of convex sets navigation in "
                    f"{self.scene.environment_name}"
                ),
                x=0.01,
                font=dict(size=20, color=palette.text_primary),
            ),
            paper_bgcolor=palette.surface,
            font=dict(color=palette.text_primary, size=12),
            legend=dict(
                orientation="h",
                x=0.0,
                y=-0.02,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=palette.text_secondary),
            ),
            margin=dict(l=0, r=0, t=110, b=10),
            annotations=self._panel_titles(palette),
        )
        # Every panel is pinned to the same box rather than auto-scaled to its own
        # traces, so a metre is the same length in all three and they can be read
        # against each other.
        bounds = self.scene.search_space.bounding_box().to_array_bounds()
        for index in range(1, len(self.panels) + 1):
            figure.update_layout(
                {
                    f"scene{index}": dict(
                        aspectmode="data",
                        camera=self.camera.as_plotly(),
                        xaxis=self._axis("x [m]", palette, bounds, 0),
                        yaxis=self._axis("y [m]", palette, bounds, 1),
                        zaxis=self._axis("z [m]", palette, bounds, 2),
                    )
                }
            )

    def _panel_titles(self, palette: FigurePalette) -> List[dict[str, object]]:
        """
        :param palette: The colors to draw with.
        :return: One annotation per panel, holding its label, its name and the statistic
            it reports.
        """
        width = 1.0 / len(self.panels)
        return [
            dict(
                text=(
                    f"<b>({label}) {panel.name}</b><br>"
                    f'<span style="color:{palette.text_secondary}">'
                    f"{panel.subtitle(self.scene)}</span>"
                ),
                x=index * width,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=14, color=palette.text_primary),
            )
            for index, (label, panel) in enumerate(zip("abc", self.panels))
        ]

    @staticmethod
    def _axis(
        title: str, palette: FigurePalette, bounds: Bounds[np.ndarray], axis: int
    ) -> dict[str, object]:
        """
        :param title: The axis label.
        :param palette: The colors to draw with.
        :param bounds: The corners of the box every panel is framed to.
        :param axis: Which of the three axes this is.
        :return: The styling and range shared by this axis across every panel.
        """
        return dict(
            title=dict(text=title, font=dict(size=11, color=palette.text_secondary)),
            range=[float(bounds.lower[axis]), float(bounds.upper[axis])],
            backgroundcolor=palette.surface,
            gridcolor=palette.obstacle_edge,
            zerolinecolor=palette.obstacle_edge,
            tickfont=dict(size=9, color=palette.text_secondary),
        )


# %% command line


DEFAULT_VOLUME_CLEARANCE = 0.05
"""
Amount obstacles are bloated by when decomposing a volume.

Smaller than the floor plan figure's clearance: a volumetric partition is cut in three
dimensions, and bloating obstacles by a mobile base's radius closes the gaps above and
between them that give the partition its structure.
"""


def _parse_arguments(available_environments: Sequence[str]) -> argparse.Namespace:
    """
    :param available_environments: The URDF environment names that can be selected.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    environment_group = parser.add_mutually_exclusive_group()
    environment_group.add_argument(
        "--environment",
        default="kitchen",
        choices=available_environments,
        help="Name of the URDF environment in semantic_digital_twin/resources/urdf to "
        "draw.",
    )
    environment_group.add_argument(
        "--sage10k",
        nargs="+",
        choices=[scene.name.lower() for scene in Sage10kActionableScenes],
        metavar="SCENE",
        help="Names of curated Sage10k scenes to draw, one figure each. Each scene is "
        "downloaded and cached on first use, which takes minutes. Available: "
        f"{', '.join(scene.name.lower() for scene in Sage10kActionableScenes)}.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=DEFAULT_VOLUME_CLEARANCE,
        help="Amount in meters obstacles are bloated by. Smaller than the floor plan "
        "figure's default, since a volume is subdivided in three dimensions and a wide "
        "clearance closes the gaps that make its structure visible.",
    )
    parser.add_argument(
        "--floor-level",
        type=float,
        default=NavigationScene.FLOOR_LEVEL,
        help="Height in meters below which nothing is navigable. Defaults to the world "
        "frame's zero, which is the floor plane in these environments; geometry "
        "modelled below it is a modelling error rather than space to plan in.",
    )
    parser.add_argument(
        "--theme",
        type=Theme,
        default=Theme.LIGHT,
        choices=list(Theme),
        metavar="{light,dark}",
        help="The surface the figure is rendered for.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path.cwd(),
        help="Directory the figure is written to.",
    )
    return parser.parse_args()


def _selected_scenes(
    arguments: argparse.Namespace, environment_paths: dict[str, Path]
) -> Iterable[NavigationScene]:
    """
    Build the scenes the parsed arguments select, lazily, so that a figure is written
    before the next Sage10k scene is downloaded.

    :param arguments: The parsed command line.
    :param environment_paths: The available URDF environments, by name.
    :return: The selected scenes.
    """
    decomposition = VolumetricDecomposition()
    if arguments.sage10k:
        loader = Sage10kDatasetLoader()
        return (
            NavigationScene.from_sage10k_scene(
                loader,
                str(Sage10kActionableScenes[name.upper()]),
                clearance=arguments.clearance,
                decomposition=decomposition,
                floor_level=arguments.floor_level,
            )
            for name in arguments.sage10k
        )

    return [
        NavigationScene.from_urdf(
            environment_paths[arguments.environment],
            clearance=arguments.clearance,
            decomposition=decomposition,
            floor_level=arguments.floor_level,
        )
    ]


def main() -> None:
    """
    Draw the three-panel volume figure of every selected environment and write it to
    disk.
    """
    environment_paths = {
        path.stem: path for path in sorted(urdf_directory().glob("*.urdf"))
    }
    arguments = _parse_arguments(sorted(environment_paths))

    for scene in _selected_scenes(arguments, environment_paths):
        for path in GraphOfConvexSetsVolumeFigure(scene, theme=arguments.theme).save(
            arguments.output_directory
        ):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
