"""
Localize the cubes on Tracy's table through Tracy's real camera, as a process of its own
answering RoboKudo's query action until it receives SIGINT or SIGTERM.

Run it as ``python -m robokudo.scripts.perceive_tracy_cubes --settings <file>``.
"""

from __future__ import annotations

# %% Imports
import argparse
import signal
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.wait_for_message import wait_for_message
from std_msgs.msg import String
from typing_extensions import Any

from robokudo.defs import PACKAGE_NAME
from robokudo.descriptors.analysis_engines.tracy_cubes import (
    AnalysisEngine,
    TracyCubesSettings,
)
from robokudo.query_service import QueryService
from robokudo.utils.logging_configuration import configure_logging
from robokudo.utils.module_loader import ModuleLoader
from semantic_digital_twin.adapters.urdf import URDFParser

# %% Process


@dataclass
class TracyCubesPerceptionProcess:
    """
    Answers queries for the cubes on Tracy's table with what Tracy's real camera sees,
    reporting poses in the frames of Tracy's published description.
    """

    settings: TracyCubesSettings
    """
    The thresholds and the real camera's topics and frames.
    """

    robot_description_topic: str = "/robot_description"
    """
    The topic Tracy's description is published on, the one its motions are planned with.
    """

    _stopped: Event = field(default_factory=Event, init=False, repr=False)
    """
    Set once the process is asked to stop.
    """

    def run(self) -> None:
        """
        Answer queries until SIGINT or SIGTERM arrives; before the first query can be
        answered, either signal ends the process at once.
        """
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        configure_logging(
            logging_config_file_name=str(
                ModuleLoader.get_module_path(PACKAGE_NAME) / "logging_levels.yaml"
            )
        )
        service = QueryService(
            analysis_engine=AnalysisEngine(
                camera=self.settings.real_camera, settings=self.settings
            ),
            # Without a prefix, the world names Tracy's frames as tf and Giskard do.
            world=URDFParser(
                urdf=self.read_robot_description(), prefix=""
            ).create_world(),
        )
        service.start()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_number, self.stop)
        self._stopped.wait()
        service.stop()
        rclpy.shutdown()

    def read_robot_description(self) -> str:
        """
        :return: Tracy's description as published, once it is.
        """
        node = rclpy.create_node("tracy_cubes_robot_description_reader")
        _, message = wait_for_message(
            String,
            node,
            self.robot_description_topic,
            qos_profile=QoSProfile(
                depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
            ),
        )
        node.destroy_node()
        return message.data

    def stop(self, signal_number: int, frame: Any) -> None:
        """
        Let :meth:`run` stop answering queries.

        :param signal_number: The signal that arrived.
        :param frame: The stack frame the signal interrupted.
        """
        self._stopped.set()


# %% Command line


def main(arguments: list[str] | None = None) -> None:
    """
    Run :class:`TracyCubesPerceptionProcess` with the settings file named on the command
    line, else the one the environment names, else the defaults.

    :param arguments: The command line arguments, by default those of the process.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="YAML file with the thresholds and the camera's topics and frames.",
    )
    settings_file = parser.parse_args(arguments).settings
    TracyCubesPerceptionProcess(
        settings=(
            TracyCubesSettings.from_environment()
            if settings_file is None
            else TracyCubesSettings.from_file(settings_file)
        )
    ).run()


if __name__ == "__main__":
    main()
