from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError
from semantic_digital_twin.spatial_types import Point3


@dataclass
class EmptyFreeSpaceError(UsageError):
    """
    Raised when an environment leaves no free space to plan in, so there is no graph of
    convex sets to draw.
    """

    environment_name: str
    """
    The label of the environment that has no free space.
    """

    def error_message(self) -> str:
        return (
            f"The environment {self.environment_name} has no free space in its search "
            "space."
        )

    def suggest_correction(self) -> str:
        return (
            "reduce the clearance obstacles are bloated by, or check that the "
            "environment's search space covers its obstacles."
        )


@dataclass
class UnreachableGoalError(UsageError):
    """
    Raised when the graph of convex sets contains no path between the queried start and
    goal.
    """

    start: Point3
    """
    Where the queried path was supposed to begin.
    """

    goal: Point3
    """
    Where the queried path was supposed to end.
    """

    def error_message(self) -> str:
        return f"No path connects {self.start} to {self.goal}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnconnectedGraphError(UsageError):
    """
    Raised when no two convex sets of a graph are connected, so there is no pair to pose
    a query between.
    """

    convex_set_count: int
    """
    The number of convex sets in the graph, none of which are connected to each other.
    """

    def error_message(self) -> str:
        return (
            f"None of the {self.convex_set_count} convex sets are connected to each "
            "other, so there is nothing to plan between."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnboundedSearchSpaceError(UsageError):
    """
    Raised when a
    :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.polygons.GraphOfConvexPolygons`
    is built with a search space that is not a single, finite bounding box.

    IRIS grows regions within a bounded convex domain; unlike
    :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.boxes.GraphOfBoundingBoxes`,
    which can decompose an unbounded or multi-box search space via the product algebra,
    Drake's ``Iris`` function requires exactly one finite ``HPolyhedron`` domain.
    """

    def error_message(self) -> str:
        return (
            "GraphOfConvexPolygons requires a search space consisting of exactly one "
            "finite bounding box."
        )

    def suggest_correction(self) -> str:
        return "pass an explicit, finite search_space with a single bounding box."
