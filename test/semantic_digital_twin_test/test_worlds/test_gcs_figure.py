"""
Tests for the three-panel Graph of Convex Sets figure.

The scenes are built from a purpose-made room world rather than from a URDF, so every
obstacle position the assertions depend on is stated here.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass
from matplotlib.colors import to_rgba

from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kActionableScenes,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, BoundingBox, Scale
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    EmptyFreeSpaceError,
    UnconnectedGraphError,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.figure import (
    ConvexSetAdjacency,
    ConvexSetsPanel,
    EnvironmentGeometry,
    EnvironmentPanel,
    FloorPlanDecomposition,
    Footprint,
    GraphOfConvexSetsFigure,
    NavigationPath,
    NavigationQuery,
    NavigationScene,
    OptimalPathPanel,
    PanelArrangement,
    Theme,
    VolumetricDecomposition,
    sage10k_environment_name,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% the world every scene in this module is built from

ROOM_HALF_EXTENT = 3.0
"""
Half the side length of the square room, so its walls span -3 m to 3 m in x and y.
"""

WALL_THICKNESS = 0.2
"""
Thickness of each of the room's four walls.
"""

PILLAR_HALF_EXTENT = 0.5
"""
Half the side length of the pillar standing in the middle of the room.
"""

ROOM_HEIGHT = 2.0
"""
Height of the room's walls and of its pillar.
"""

CLEARANCE = 0.2
"""
Clearance the scenes under test are built with.
"""


def add_box_body(world: World, name: str, scale: Scale, x: float, y: float) -> Body:
    """
    Add a box-shaped obstacle standing on the floor of a world.

    :param world: The world to add the obstacle to.
    :param name: The name of the body carrying the obstacle.
    :param scale: The extent of the box.
    :param x: The x coordinate of the box's center.
    :param y: The y coordinate of the box's center.
    :return: The added body.
    """
    body = Body(name=PrefixedName(name))
    world.add_connection(
        FixedConnection.create_with_dofs(
            world,
            world.root,
            body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x, y, ROOM_HEIGHT / 2, reference_frame=world.root
            ),
        )
    )
    body.collision.append(Box(scale=scale))
    return body


@pytest.fixture
def room_world() -> World:
    """
    A square room walled on all four sides, with a single square pillar in its middle.

    The pillar sits on the line between any two opposite corners, so a path across the
    room has to detour around it.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

        wall_center = ROOM_HALF_EXTENT - WALL_THICKNESS / 2
        wall_length = 2 * ROOM_HALF_EXTENT
        horizontal_wall = Scale(wall_length, WALL_THICKNESS, ROOM_HEIGHT)
        vertical_wall = Scale(WALL_THICKNESS, wall_length, ROOM_HEIGHT)

        add_box_body(world, "south_wall", horizontal_wall, 0.0, -wall_center)
        add_box_body(world, "north_wall", horizontal_wall, 0.0, wall_center)
        add_box_body(world, "west_wall", vertical_wall, -wall_center, 0.0)
        add_box_body(world, "east_wall", vertical_wall, wall_center, 0.0)
        add_box_body(
            world,
            "pillar",
            Scale(2 * PILLAR_HALF_EXTENT, 2 * PILLAR_HALF_EXTENT, ROOM_HEIGHT),
            0.0,
            0.0,
        )
    return world


@pytest.fixture
def room_scene(room_world: World) -> NavigationScene:
    """
    The scene of the room world, planned with :data:`CLEARANCE`.
    """
    return NavigationScene.from_world(room_world, "room", clearance=CLEARANCE)


FLOOR_THICKNESS = 0.02
"""
Thickness of the floor slabs, thin enough to count as ground under
:attr:`NavigationScene.STEP_HEIGHT`.
"""

FLOOR_FOOTPRINTS = ((-3.0, -2.0, 0.0, 2.0), (1.0, -2.0, 4.0, 2.0))
"""
The x-y bounds of the two floors of :func:`two_floor_world`, separated by a gap no floor
covers.
"""


@pytest.fixture
def two_floor_world() -> World:
    """
    Two floor slabs separated by a gap, with one waist-high obstacle standing on the
    first of them.

    This mirrors how a dataset scene is laid out: a floor per room, and furniture placed
    on top of it.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

        for index, (min_x, min_y, max_x, max_y) in enumerate(FLOOR_FOOTPRINTS):
            floor = Body(name=PrefixedName(f"floor_{index}"))
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world,
                    world.root,
                    floor,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        (min_x + max_x) / 2,
                        (min_y + max_y) / 2,
                        -FLOOR_THICKNESS / 2,
                        reference_frame=world.root,
                    ),
                )
            )
            floor.collision.append(
                Box(scale=Scale(max_x - min_x, max_y - min_y, FLOOR_THICKNESS))
            )

        add_box_body(world, "cabinet", Scale(1.0, 1.0, ROOM_HEIGHT), -1.5, 0.0)
    return world


@dataclass
class BoxRowFixture:
    """
    A graph of three convex sets laid out in a row along x, whose farthest pair is known
    by construction.
    """

    graph: GraphOfBoundingBoxes
    boxes: list[BoundingBox]


@pytest.fixture
def box_row() -> BoxRowFixture:
    """
    Three unit boxes touching along x, spanning x from 0 m to 3 m.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = GraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    boxes = [
        BoundingBox(offset, 0.0, 0.0, offset + 1.0, 1.0, 1.0, origin)
        for offset in (0.0, 1.0, 2.0)
    ]
    for box in boxes:
        graph.add_node(box)
    graph.calculate_connectivity()
    return BoxRowFixture(graph, boxes)


# %% the pieces a scene is made of


def test_footprint_projects_a_bounding_box_including_its_origin():
    """
    A footprint is the box's x-y extent in the frame its origin is expressed in, not the
    extent relative to that origin.
    """
    origin = HomogeneousTransformationMatrix.from_xyz_rpy(2.0, -1.0, 5.0)
    footprint = Footprint.of(BoundingBox(-0.5, -0.25, 0.0, 0.5, 0.25, 1.0, origin))

    assert (footprint.min_x, footprint.max_x) == (1.5, 2.5)
    assert (footprint.min_y, footprint.max_y) == (-1.25, -0.75)
    assert (footprint.width, footprint.height) == (1.0, 0.5)


def test_navigation_path_length_sums_its_segments():
    """
    The reported length is the distance travelled, not the distance between the
    endpoints.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    corner_path = NavigationPath(
        [
            Point3(0.0, 0.0, 0.0, reference_frame=world.root),
            Point3(3.0, 0.0, 0.0, reference_frame=world.root),
            Point3(3.0, 4.0, 0.0, reference_frame=world.root),
        ]
    )

    assert corner_path.length == pytest.approx(7.0)


def test_navigation_path_reports_how_much_of_it_is_vertical():
    """
    The height a path gains and loses is counted whichever way it goes, and is zero for
    a path that stays level.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    climbing_path = NavigationPath(
        [
            Point3(0.0, 0.0, 0.0, reference_frame=world.root),
            Point3(0.0, 0.0, 2.0, reference_frame=world.root),
            Point3(1.0, 0.0, 1.5, reference_frame=world.root),
        ]
    )
    level_path = NavigationPath(
        [
            Point3(0.0, 0.0, 1.0, reference_frame=world.root),
            Point3(5.0, 0.0, 1.0, reference_frame=world.root),
        ]
    )

    assert climbing_path.vertical_travel == pytest.approx(2.5)
    assert level_path.vertical_travel == pytest.approx(0.0)


def test_a_graph_with_no_connected_pair_has_nothing_to_plan_between():
    """
    Two convex sets that do not touch leave no query to pose, which is reported rather
    than failing inside the search for the most distant pair.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = GraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    for offset in (0.0, 10.0):
        graph.add_node(BoundingBox(offset, 0.0, 0.0, offset + 1.0, 1.0, 1.0, origin))
    graph.calculate_connectivity()

    with pytest.raises(UnconnectedGraphError):
        NavigationQuery.between_most_distant_convex_sets(graph)


def test_adjacency_is_drawn_through_the_portal_the_sets_share(
    box_row: BoxRowFixture,
):
    """
    An edge is drawn from one set's center through the overlap the two sets share, so it
    never cuts a corner the planner cannot cut.
    """
    adjacency = ConvexSetAdjacency(
        source_center=box_row.boxes[0].center,
        portal_center=box_row.graph.graph.get_edge_data(
            box_row.graph.box_to_index_map[box_row.boxes[0]],
            box_row.graph.box_to_index_map[box_row.boxes[1]],
        ).intersection.center,
        target_center=box_row.boxes[1].center,
    )

    assert adjacency.coordinates.tolist() == [[0.5, 0.5], [1.0, 0.5], [1.5, 0.5]]


def test_query_spans_the_graph_in_coordinate_order(box_row: BoxRowFixture):
    """
    The query connects the two ends of the longest shortest path, oriented so that the
    same graph always produces the same start, whatever order its nodes are held in.
    """
    query = NavigationQuery.between_most_distant_convex_sets(box_row.graph)

    assert (float(query.start.x), float(query.start.y)) == (0.5, 0.5)
    assert (float(query.goal.x), float(query.goal.y)) == (2.5, 0.5)


def test_query_measures_distance_along_the_graph_not_across_it():
    """
    Two sets facing each other across a barrier are far apart to travel between but
    close together in space, and it is the travelling distance that decides the query.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = GraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    # A U of unit boxes: the two tips face each other one meter apart, but reaching one
    # tip from the other means walking around the whole U.
    corner_offsets = [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
        (2.0, 2.0),
        (2.0, 1.0),
        (2.0, 0.0),
    ]
    for offset_x, offset_y in corner_offsets:
        graph.add_node(
            BoundingBox(
                offset_x, offset_y, 0.0, offset_x + 1.0, offset_y + 1.0, 1.0, origin
            )
        )
    graph.calculate_connectivity()

    query = NavigationQuery.between_most_distant_convex_sets(graph)

    assert (float(query.start.x), float(query.start.y)) == (0.5, 0.5)
    assert (float(query.goal.x), float(query.goal.y)) == (2.5, 0.5)
    assert float(query.start.euclidean_distance(query.goal)) == pytest.approx(2.0)


# %% the scene


def test_a_world_without_ground_geometry_is_searched_in_its_covering_box(
    room_scene: NavigationScene,
):
    """
    With no floor to derive a navigable volume from, the search space falls back to the
    box covering the world, which hugs the outer walls.
    """
    covering_box = Footprint.of(room_scene.search_space.bounding_box())

    assert len(room_scene.search_space) == 1
    assert (covering_box.min_x, covering_box.max_x) == pytest.approx(
        (-ROOM_HALF_EXTENT, ROOM_HALF_EXTENT)
    )
    assert (covering_box.min_y, covering_box.max_y) == pytest.approx(
        (-ROOM_HALF_EXTENT, ROOM_HALF_EXTENT)
    )


def test_ground_geometry_is_travelled_on_rather_than_around(two_floor_world: World):
    """
    A floor slab is not an obstacle: counting it as one would project onto the whole
    room and leave nothing to plan in.

    What stands on it still is.
    """
    geometry = EnvironmentGeometry.of(
        two_floor_world, NavigationScene.STEP_HEIGHT, NavigationScene.FLOOR_LEVEL
    )

    obstacles = FloorPlanDecomposition().obstacle_bodies(geometry)

    assert [str(body.name.name) for body in obstacles] == ["cabinet"]
    assert len(geometry.ground) == len(FLOOR_FOOTPRINTS)


def test_a_volumetric_decomposition_keeps_the_ground_as_an_obstacle(
    two_floor_world: World,
):
    """
    A three-dimensional partition has to subtract the floor like any other body, since a
    path may change height but never pass through the ground.
    """
    geometry = EnvironmentGeometry.of(
        two_floor_world, NavigationScene.STEP_HEIGHT, NavigationScene.FLOOR_LEVEL
    )

    obstacles = VolumetricDecomposition().obstacle_bodies(geometry)

    assert sorted(str(body.name.name) for body in obstacles) == [
        "cabinet",
        "floor_0",
        "floor_1",
    ]


def test_the_navigable_volume_is_the_floor_raised_to_the_full_height(
    two_floor_world: World,
):
    """
    A robot may stand where there is floor under it and nowhere else, so the search
    space follows the floors rather than covering the gap between them.

    It starts at the floor level rather than at the underside of the slab, and spans the
    world's full height so obstacles are subtracted over all of theirs.
    """
    geometry = EnvironmentGeometry.of(
        two_floor_world, NavigationScene.STEP_HEIGHT, NavigationScene.FLOOR_LEVEL
    )

    volume = geometry.navigable_volume()
    footprints = sorted(
        (footprint.min_x, footprint.min_y, footprint.max_x, footprint.max_y)
        for footprint in map(Footprint.of, volume)
    )
    lower_heights = [float(box.to_array_bounds().lower[2]) for box in volume]
    upper_heights = [float(box.to_array_bounds().upper[2]) for box in volume]

    assert np.allclose(footprints, sorted(FLOOR_FOOTPRINTS))
    assert lower_heights == pytest.approx([NavigationScene.FLOOR_LEVEL] * len(volume))
    assert upper_heights == pytest.approx([ROOM_HEIGHT] * len(volume))


def test_geometry_reaching_below_the_floor_does_not_lower_the_navigable_volume(
    two_floor_world: World,
):
    """
    A body modelled reaching through the floor is a modelling error, not space to plan
    in, so the volume still starts at the floor.
    """
    with two_floor_world.modify_world():
        add_box_body(two_floor_world, "sunken_table", Scale(1.0, 1.0, 4.0), 2.0, 0.0)

    geometry = EnvironmentGeometry.of(
        two_floor_world, NavigationScene.STEP_HEIGHT, NavigationScene.FLOOR_LEVEL
    )

    sunken_bottom = min(
        float(box.to_array_bounds().lower[2])
        for box in geometry.bounding_boxes_of(geometry.standing_bodies)
    )
    volume_bottom = min(
        float(box.to_array_bounds().lower[2]) for box in geometry.navigable_volume()
    )

    assert sunken_bottom < NavigationScene.FLOOR_LEVEL
    assert volume_bottom == pytest.approx(NavigationScene.FLOOR_LEVEL)


def test_a_world_standing_above_the_floor_stays_tight_to_itself(room_world: World):
    """
    The floor level is a bound, not a target: a world whose geometry begins above it is
    not given empty space underneath.
    """
    geometry = EnvironmentGeometry.of(room_world, NavigationScene.STEP_HEIGHT, -5.0)

    volume_bottom = min(
        float(box.to_array_bounds().lower[2]) for box in geometry.navigable_volume()
    )

    assert volume_bottom == pytest.approx(0.0)


def test_curated_sage10k_scenes_are_named_after_the_scene():
    """
    A curated scene is titled with its name, and any other scene with the filename of
    its URL.
    """
    assert sage10k_environment_name(str(Sage10kActionableScenes.GYM)) == "gym"
    assert (
        sage10k_environment_name(
            "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/"
            "20251213_020526_layout_84b703fb.zip"
        )
        == "20251213_020526_layout_84b703fb"
    )


def test_convex_sets_keep_the_clearance_away_from_every_obstacle(
    room_scene: NavigationScene,
):
    """
    No convex set reaches into an obstacle bloated by the clearance, which is what makes
    a path through the sets passable for a robot of that radius.
    """
    bloated_obstacles = [
        Footprint.of(obstacle.bloat(CLEARANCE, CLEARANCE, 0.0))
        for obstacle in room_scene.obstacles
    ]

    overlaps = [
        (convex_set, obstacle)
        for convex_set in map(Footprint.of, room_scene.convex_sets)
        for obstacle in bloated_obstacles
        if convex_set.min_x < obstacle.max_x
        and obstacle.min_x < convex_set.max_x
        and convex_set.min_y < obstacle.max_y
        and obstacle.min_y < convex_set.max_y
    ]

    assert overlaps == []


def test_path_detours_around_the_pillar(room_scene: NavigationScene):
    """
    The path runs from the query's start to its goal without entering the pillar, and is
    therefore longer than the straight line between them.
    """
    pillar_footprint = Footprint(
        min_x=-PILLAR_HALF_EXTENT - CLEARANCE,
        min_y=-PILLAR_HALF_EXTENT - CLEARANCE,
        max_x=PILLAR_HALF_EXTENT + CLEARANCE,
        max_y=PILLAR_HALF_EXTENT + CLEARANCE,
    )
    waypoints = room_scene.path.waypoints

    assert waypoints[0] is room_scene.query.start
    assert waypoints[-1] is room_scene.query.goal
    assert not [
        waypoint
        for waypoint in waypoints
        if pillar_footprint.min_x < float(waypoint.x) < pillar_footprint.max_x
        and pillar_footprint.min_y < float(waypoint.y) < pillar_footprint.max_y
    ]
    assert room_scene.path.length > float(
        room_scene.query.start.euclidean_distance(room_scene.query.goal)
    )


def test_an_environment_without_free_space_is_reported(room_world: World):
    """
    A clearance wide enough to close the room leaves nothing to plan in, and that is
    reported rather than drawn as an empty figure.
    """
    with pytest.raises(EmptyFreeSpaceError):
        NavigationScene.from_world(room_world, "room", clearance=ROOM_HALF_EXTENT)


# %% the figure


def test_wide_environments_are_stacked_and_compact_ones_are_placed_side_by_side():
    """
    The arrangement follows the environment's aspect ratio, so panels never end up as
    thin slivers.
    """
    wide = Footprint(min_x=0.0, min_y=0.0, max_x=20.0, max_y=5.0)
    compact = Footprint(min_x=0.0, min_y=0.0, max_x=6.0, max_y=6.0)

    assert PanelArrangement.for_extent(wide, 3) == PanelArrangement(rows=3, columns=1)
    assert PanelArrangement.for_extent(compact, 3) == PanelArrangement(
        rows=1, columns=3
    )


def test_a_meter_is_the_same_length_in_every_panel():
    """
    Panels are framed to a shared extent and sized from a single scale, so the three can
    be read against each other.
    """
    extent = Footprint(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0)
    arrangement = PanelArrangement(rows=1, columns=3)

    width, _ = arrangement.figure_size(extent)

    assert width == pytest.approx(3 * 4.0 * arrangement.PREFERRED_INCHES_PER_METER)


def test_a_legend_too_wide_for_the_figure_wraps_into_even_rows():
    """
    A stacked figure is too narrow to lay its legend out on one row, and wrapping it
    spreads the entries rather than leaving a row with a single entry in it.
    """
    narrow = Footprint(min_x=0.0, min_y=0.0, max_x=12.0, max_y=8.0)
    arrangement = PanelArrangement(rows=3, columns=1)

    width, _ = arrangement.figure_size(narrow)
    entries_that_fit = int(width / arrangement.LEGEND_ENTRY_INCHES)

    assert entries_that_fit == 3
    assert arrangement.legend_columns(narrow, 7) == 3
    assert arrangement.legend_columns(narrow, 4) == 2


def test_a_legend_that_fits_stays_on_one_row():
    """
    A legend with room for all of its entries is laid out in a single row.
    """
    wide = Footprint(min_x=0.0, min_y=0.0, max_x=6.0, max_y=6.0)
    arrangement = PanelArrangement(rows=1, columns=3)

    width, _ = arrangement.figure_size(wide)
    entry_count = int(width / arrangement.LEGEND_ENTRY_INCHES)

    assert arrangement.legend_columns(wide, entry_count) == entry_count


def test_figure_draws_one_titled_panel_per_stage(room_scene: NavigationScene):
    """
    The figure walks through what the planner is given, what it builds and what it
    returns, each panel labelled and quantified.
    """
    figure = GraphOfConvexSetsFigure(room_scene)

    rendered = figure.render()
    titles = [axes.get_title(loc="left") for axes in rendered.axes]

    assert titles == [
        f"(a) {EnvironmentPanel().name}",
        f"(b) {ConvexSetsPanel().name}",
        f"(c) {OptimalPathPanel().name}",
    ]
    assert ConvexSetsPanel().subtitle(room_scene) == (
        f"{len(room_scene.convex_sets)} convex sets, "
        f"{len(room_scene.adjacencies)} adjacencies"
    )


def test_figure_is_painted_on_its_theme_surface(room_scene: NavigationScene):
    """
    A theme's palette is applied to the figure itself, not only to the marks in it.
    """
    rendered = GraphOfConvexSetsFigure(room_scene, theme=Theme.DARK).render()

    assert rendered.get_facecolor() == pytest.approx(
        to_rgba(Theme.DARK.palette.surface)
    )


def test_figure_is_saved_as_vector_and_raster(room_scene, tmp_path):
    """
    Saving produces both a file to typeset and a file to look at, named after the
    environment and the theme.
    """
    written = GraphOfConvexSetsFigure(room_scene).save(tmp_path / "figures")

    assert [path.name for path in written] == [
        "graph_of_convex_sets_room_light.pdf",
        "graph_of_convex_sets_room_light.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)
