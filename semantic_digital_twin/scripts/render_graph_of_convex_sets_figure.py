"""
Writes the figure of a URDF environment (``--environment``) or of one or more Sage10k
scenes (``--sage10k``) rendered by
:class:`~semantic_digital_twin.world_description.graph_of_convex_sets.figure.GraphOfConvexSetsFigure`
to disk.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from typing_extensions import Iterable, Sequence

from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kActionableScenes,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
    hardest_path_query,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    UnreachableGoalError,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.figure import (
    GraphOfConvexSetsFigure,
    NavigationPath,
    NavigationScene,
    Theme,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

DEFAULT_CLEARANCE = 0.2
"""
Default amount obstacles are bloated by, standing in for the radius of the robot that
has to fit past them.
"""

DEFAULT_FLOOR_LEVEL = 0.0
"""
Default height in meters below which nothing is navigable.
"""


def urdf_directory() -> Path:
    """
    :return: The directory holding the URDF environments this script can draw.
    """
    return Path(files("semantic_digital_twin")).parent.parent / "resources" / "urdf"


def default_output_directory() -> Path:
    """
    :return: The directory figures are written to unless ``--output-directory`` is
        given.
    """
    return Path(files("semantic_digital_twin")).parent.parent / "resources" / "output"


def search_space_of(world: World, floor_level: float) -> BoundingBoxCollection:
    """
    :param world: The world to cover.
    :param floor_level: The height below which nothing is navigable.
    :return: The single box covering the world's collision geometry, clipped to
        :attr:`floor_level` at the bottom.
    """
    covering_box = obstacle_boxes_of(world).bounding_box()
    lower_z = max(floor_level, float(covering_box.to_array_bounds().lower[2]))
    return BoundingBoxCollection([replace(covering_box, min_z=lower_z)], world.root)


def obstacle_boxes_of(world: World) -> BoundingBoxCollection:
    """
    :param world: The world to measure.
    :return: The bounding box of every body with collision geometry, unbloated.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
    return annotation.as_bounding_box_collection_at_origin(origin)


def navigation_scene_of(
    world: World, environment_name: str, clearance: float, floor_level: float
) -> NavigationScene:
    """
    :param world: The world to build the scene of.
    :param environment_name: The label the figure is titled with.
    :param clearance: Amount obstacles are bloated by while building the graph.
    :param floor_level: The height below which nothing is navigable.
    :return: The scene, planned between the two convex sets the environment forces the
        longest detour between.
    :raises UnreachableGoalError: If the graph contains no path for its own query.
    """
    search_space = search_space_of(world, floor_level)
    graph = GraphOfBoundingBoxes.navigation_map_from_world(
        world=world, search_space=search_space, bloat_obstacles=clearance
    )
    query = hardest_path_query(graph)
    waypoints = graph.path_from_to(query.start, query.goal)
    if waypoints is None:
        raise UnreachableGoalError(query.start, query.goal)
    return NavigationScene(
        graph_of_convex_sets=graph,
        environment_name=environment_name,
        path=NavigationPath(waypoints),
        obstacles=obstacle_boxes_of(world),
    )


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
    environment_group.add_argument(
        "--sage10k-url",
        nargs="+",
        metavar="URL",
        help="URLs of Sage10k scenes to draw, one figure each, for scenes outside the "
        "curated set.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=DEFAULT_CLEARANCE,
        help="Amount in meters obstacles are bloated by, standing in for the radius of "
        "the robot that has to fit past them.",
    )
    parser.add_argument(
        "--floor-level",
        type=float,
        default=DEFAULT_FLOOR_LEVEL,
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
        default=default_output_directory(),
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
    if arguments.sage10k or arguments.sage10k_url:
        loader = Sage10kDatasetLoader()
        scene_urls = arguments.sage10k_url or [
            str(Sage10kActionableScenes[name.upper()]) for name in arguments.sage10k
        ]
        return (
            navigation_scene_of(
                loader.create_scene(scene_url=scene_url).create_world(),
                loader.environment_name(scene_url),
                clearance=arguments.clearance,
                floor_level=arguments.floor_level,
            )
            for scene_url in scene_urls
        )

    urdf_path = environment_paths[arguments.environment]
    return [
        navigation_scene_of(
            URDFParser.from_file(str(urdf_path)).parse(),
            urdf_path.stem,
            clearance=arguments.clearance,
            floor_level=arguments.floor_level,
        )
    ]


def main() -> None:
    """
    Draw the three-panel figure of every selected environment and write it to disk.
    """
    environment_paths = {
        path.stem: path for path in sorted(urdf_directory().glob("*.urdf"))
    }
    arguments = _parse_arguments(sorted(environment_paths))

    for scene in _selected_scenes(arguments, environment_paths):
        for path in GraphOfConvexSetsFigure(scene, theme=arguments.theme).save(
            arguments.output_directory
        ):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
