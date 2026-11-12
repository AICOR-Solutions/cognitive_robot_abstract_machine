"""
An analysis engine answering queries on RoboKudo's query action from within the calling
process, so that it perceives the very world that process holds.
"""

from __future__ import annotations

# %% Imports
import threading
import time
from dataclasses import dataclass, field
from threading import Event, ExceptHookArgs, Thread

import rclpy
from py_trees.blackboard import Blackboard
from py_trees_ros.trees import BehaviourTree
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.world import World
from typing_extensions import Any, Callable

import robokudo.world
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import QueryServiceAlreadyRunning, QueryServiceThreadFailed
from robokudo.garden import grow_tree
from robokudo.identifier import BBIdentifier
from robokudo.utils.error_handling import raise_to_blackboard
from robokudo.utils.tree import setup_with_descendants_rk

# %% Query service


@dataclass(eq=False)
class QueryService:
    """
    Runs an analysis engine that answers queries on RoboKudo's query action, ticking it
    in a thread of the calling process against :attr:`world`.

    A process runs at most one at a time, since RoboKudo keeps its state in globals.
    """

    analysis_engine: AnalysisEngineInterface
    """
    The analysis engine answering the queries; its pipeline holds a
    :class:`~robokudo.annotators.query.QueryAnnotator`.
    """

    world: World
    """
    The world the analysis engine perceives and reports poses in.
    """

    tick_rate: float = 20.0
    """
    How often the analysis engine is ticked, in Hz.
    """

    thread_failed: Event = field(default_factory=Event)
    """
    Set once a thread of the service ended with an exception; the service answers no
    more queries from then on.
    """

    thread_failure: QueryServiceThreadFailed | None = field(default=None, init=False)
    """
    Why a thread of the service ended, once one did, else None.
    """

    _stopped: Event = field(default_factory=Event, init=False, repr=False)
    """
    Set once the analysis engine is to stop ticking.
    """

    _node: Node | None = field(default=None, init=False, repr=False)
    """
    The node the analysis engine runs on while it runs, else None.
    """

    _executor: MultiThreadedExecutor | None = field(
        default=None, init=False, repr=False
    )
    """
    Spins the nodes of the analysis engine while it runs, else None.
    """

    _threads: list[Thread] = field(default_factory=list, init=False, repr=False)
    """
    The threads spinning the nodes and ticking the analysis engine while it runs.
    """

    _previous_excepthook: Callable[[ExceptHookArgs], Any] | None = field(
        default=None, init=False, repr=False
    )
    """
    The process's handler of exceptions ending a thread before the service started.
    """

    def start(self) -> None:
        """
        Build the analysis engine and answer queries until :meth:`stop`.

        :raises QueryServiceAlreadyRunning: If this service runs already.
        """
        if self._node is not None:
            raise QueryServiceAlreadyRunning(self.analysis_engine.name())
        robokudo.world.set_world(self.world)
        robokudo.world.init_world_entity_tracker_from_world(self.world)
        self._node = rclpy.create_node(PACKAGE_NAME)
        ROSNodeRegistry().register(self._node)
        tree = grow_tree(
            self.analysis_engine.implementation(), node=self._node, include_gui=False
        )
        setup_with_descendants_rk(tree)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor.add_node(Blackboard().get(BBIdentifier.QUERY_SERVER))
        self._stopped.clear()
        self._previous_excepthook = threading.excepthook
        threading.excepthook = self._record_thread_failure
        self._threads = [
            Thread(target=self._executor.spin, daemon=True, name="robokudo_spin"),
            Thread(target=self._tick, args=(tree,), daemon=True, name="robokudo_tick"),
        ]
        for thread in self._threads:
            thread.start()

    def _tick(self, tree: BehaviourTree) -> None:
        """
        Tick the analysis engine at :attr:`tick_rate` until :attr:`_stopped` is set.

        :param tree: The grown analysis engine.
        """
        period = 1.0 / self.tick_rate
        while not self._stopped.is_set():
            started = time.monotonic()
            tree.tick()
            self._stopped.wait(max(0.0, period - (time.monotonic() - started)))

    def _record_thread_failure(self, arguments: ExceptHookArgs) -> None:
        """
        Handle an exception ending a thread of the process: when it ends a thread of the
        service that was not asked to stop, keep it as :attr:`thread_failure`, abort the
        query in progress with it and set :attr:`thread_failed`. The previous handler
        reports it either way.

        :param arguments: The exception and the thread it ended.
        """
        self._previous_excepthook(arguments)
        if arguments.thread not in self._threads or self._stopped.is_set():
            return
        self.thread_failure = QueryServiceThreadFailed(
            thread_name=arguments.thread.name, reason=arguments.exc_value
        )
        raise_to_blackboard(self.thread_failure)
        self.thread_failed.set()

    def stop(self) -> None:
        """
        Stop answering queries and release the nodes; does nothing when not running.
        """
        if self._node is None:
            return
        self._stopped.set()
        Blackboard().get(BBIdentifier.QUERY_SERVER).stop_answering()
        self._executor.shutdown()
        for thread in self._threads:
            thread.join()
        threading.excepthook = self._previous_excepthook
        query_server = Blackboard().get(BBIdentifier.QUERY_SERVER)
        Blackboard().unset(BBIdentifier.QUERY_SERVER)
        query_server.destroy_node()
        ROSNodeRegistry().clear(self._node)
        self._node.destroy_node()
        self._node = None
        self._executor = None
        self._threads = []
