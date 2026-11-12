import json
from dataclasses import dataclass, field
from threading import Event
from time import time, sleep

from rclpy.node import Node
from rclpy.service import Service
from std_srvs.srv import Trigger

from semantic_digital_twin.adapters.ros.exceptions import WorldServiceDidNotAnswer
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node


@dataclass
class FetchWorldServer(HasROS2Node):
    """
    A ros service that allows other processes to fetch the entire world modification
    list from this world.

    The modification list is sent via a JSON string message.
    """

    world: World
    """
    The world to fetch modifications from.
    """

    service_name: str = "/semantic_digital_twin/fetch_world"
    """
    The name of the service.
    """

    service: Service | None = field(default=None, init=False)
    """
    The ROS service object.
    """

    def __post_init__(self):
        """
        Initialize the ROS service.
        """
        self.service = self.node.create_service(
            Trigger, self.service_name, self.service_callback
        )

    def service_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Handle service requests to fetch the world modification list.

        :param request: The service request (empty for Trigger service).
        :param response: The service response containing success status and message.
        :return: The populated response.
        """
        # Return a payload that contains both the modification blocks and the current state snapshot
        modifications_json = self.get_payload_as_json()
        response.success = True
        response.message = modifications_json
        return response

    def get_modifications_as_json(self) -> str:
        """
        Serialize all world modification blocks to JSON.

        :return: JSON string containing all modification blocks.
        """
        modifications_list = [
            block.to_json()
            for block in self.world.get_world_model_manager().model_modification_blocks
        ]
        return json.dumps(modifications_list)

    def get_payload_as_json(self) -> str:
        """
        Serialize both the world modification blocks and a snapshot of the current
        state.

        The returned JSON has the structure:
        {
            "modifications": [ ... ],
            "state": {
                "ids": [ UUIDJson, ... ],
                "states": [ float, ... ]
            }
        }

        This ensures a receiver can rebuild the model from the modification blocks
        and then apply the most recent state values.
        """
        # Build snapshot object and serialize through its serializer
        snapshot = WorldModelSnapshot(
            modifications=list(
                self.world.get_world_model_manager().model_modification_blocks
            ),
            ids=list(self.world.state.keys()),
            states=list(self.world.state.positions),
        )

        return json.dumps(snapshot.to_json())

    def close(self):
        """
        Clean up the service.
        """
        if self.service is not None:
            self.node.destroy_service(self.service)
            self.service = None


@dataclass
class NoServiceFoundError(Exception):
    service_suffix: str


def fetch_world_from_service(
    node: Node,
    service_suffix: str = "fetch_world",
    timeout_seconds: float = 10.0,
    request_interval: float = 5.0,
) -> World:
    """
    Fetch a world from any WorldFetcher Service.

    This method discovers all available WorldFetcher Services in the ROS2 network and
    picks the first one to get all world modification blocks from it.

    :param node: The ROS2 node to use for communication; it is spun by another thread.
    :param service_suffix: The suffix (last part behind '/') of the WorldFetcher
        services to look for.
    :param timeout_seconds: Maximum time to wait for service availability and response.
    :param request_interval: How long, in seconds, a request is waited for before it
        is sent again, as a request can be lost, such as with a server ending.
    :return: The fetched modification blocks.
    :raises WorldServiceDidNotAnswer: When no request was answered in time.
    """
    deadline = time() + timeout_seconds
    while time() < deadline:
        available_services = node.get_service_names_and_types()
        matching_services = [
            name
            for name, srv_type in available_services
            if name.endswith(service_suffix) and srv_type == ["std_srvs/srv/Trigger"]
        ]

        if matching_services:
            break

        sleep(0.1)
    else:
        raise NoServiceFoundError(service_suffix)

    chosen_service = matching_services[0]
    client = node.create_client(Trigger, chosen_service)
    answered = Event()
    requests = []
    while client.wait_for_service(timeout_sec=max(0.0, deadline - time())):
        request = client.call_async(Trigger.Request())
        request.add_done_callback(lambda done: answered.set())
        requests.append(request)
        if answered.wait(max(0.0, min(request_interval, deadline - time()))):
            break
        if time() >= deadline:
            break
    for request in requests:
        if not request.done():
            client.remove_pending_request(request)
    node.destroy_client(client)
    responses = [request.result() for request in requests if request.done()]
    if not responses:
        raise WorldServiceDidNotAnswer(
            service=chosen_service, timeout_seconds=timeout_seconds
        )
    response = responses[0]

    # New format is an object {"modifications": [...], "state": {...}}.
    world = World()
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    kwargs = tracker.create_kwargs()

    payload = json.loads(response.message)
    WorldModelSnapshot.apply_to_json_snapshot_to_world(world, payload, **kwargs)

    return world
