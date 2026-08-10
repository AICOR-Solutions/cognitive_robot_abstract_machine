import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import WorldStateArrayReplacedError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


def create_compiled_executor(world: World) -> Executor:
    """
    An executor with a minimal compiled motion statechart in the given world.
    """
    motion_statechart = MotionStatechart()
    node = ConstTrueNode()
    motion_statechart.add_node(node)
    motion_statechart.add_node(EndMotion.when_true(node))

    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=motion_statechart)
    return executor


def test_tick_reports_replaced_world_state_array():
    """
    Adding a degree of freedom after compiling replaces the world state array, which
    leaves the compiled updaters reading a detached copy, so ticking must report it.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    executor = create_compiled_executor(world)

    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=Body(name=PrefixedName("added"))
            )
        )

    with pytest.raises(WorldStateArrayReplacedError):
        executor.tick()


def test_tick_accepts_unchanged_world_state_array():
    """
    A model change that keeps every degree of freedom leaves the state array in place,
    so ticking proceeds.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        new_parent = Body(name=PrefixedName("new_parent"))
        free_child = Body(name=PrefixedName("free_child"))
        for body in [root, new_parent, free_child]:
            world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=new_parent)
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=free_child)
        )
    executor = create_compiled_executor(world)

    with world.modify_world():
        world.move_branch(free_child, new_parent)

    executor.tick()
