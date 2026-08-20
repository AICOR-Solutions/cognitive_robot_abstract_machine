"""
The ORM interfaces the packages of this repository generate with ORMatic.

The interfaces are generated rather than written, so the repository ignores them instead
of tracking them: a fresh checkout carries no database mapping at all, and nothing can
be persisted or turned into a data access object until they have been generated once.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from typing_extensions import Sequence

from cognitive_robot_abstract_machine.exceptions import (
    MissingOrmGeneratorError,
    OrmGenerationFailedError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""
Root of the checkout this package is installed from.
"""

INTERFACE_FILE_NAME = "ormatic_interface.py"
"""
Name every package's generator writes its interface to.
"""

PROGRESS_DESCRIPTION = "Building ORM interfaces"
"""
What the progress bar of a build calls itself.
"""

# %% a single package's interface


@dataclass
class OrmInterface:
    """
    The ORM interface a single package generates.
    """

    package_name: str
    """
    Name of the package, which is also the name of its source folder and module.
    """

    repository_root: Path
    """
    Root of the checkout the package lives in.
    """

    @property
    def generator(self) -> Path:
        """
        The script that generates this interface.
        """
        return self.repository_root / self.package_name / "scripts" / "generate_orm.py"

    @property
    def path(self) -> Path:
        """
        The generated interface file.
        """
        return (
            self.repository_root
            / self.package_name
            / "src"
            / self.package_name
            / "orm"
            / INTERFACE_FILE_NAME
        )

    @property
    def is_generated(self) -> bool:
        """
        Whether this checkout holds the interface.
        """
        return self.path.exists()

    def remove(self) -> None:
        """
        Delete the interface, so that a stale version cannot be imported while the new
        one is generated.
        """
        self.path.unlink(missing_ok=True)

    def generate(self, show_generator_output: bool = False) -> None:
        """
        Run this package's generator in a subprocess.

        A generator logs its way through a whole class hierarchy, which buries the
        progress of a build, so what it writes is kept for the failure report instead of
        reaching the terminal.

        :param show_generator_output: Whether to let the generator write to the terminal
            rather than into the report of a failure.
        :raises MissingOrmGeneratorError: If the package has no generator.
        :raises OrmGenerationFailedError: If the generator exits without having built
            the interface.
        """
        if not self.generator.exists():
            raise MissingOrmGeneratorError(self.package_name, self.generator)
        result = subprocess.run(
            [sys.executable, str(self.generator)],
            cwd=self.generator.parent,
            capture_output=not show_generator_output,
            text=True,
        )
        if result.returncode != 0:
            raise OrmGenerationFailedError(
                self.package_name, self.reported_output(result)
            )

    @staticmethod
    def reported_output(result: subprocess.CompletedProcess) -> str:
        """
        Collect what a finished generator wrote, for a failure to report.

        :param result: The finished generator run.
        :return: Everything it wrote, empty when it wrote to the terminal instead.
        """
        return "".join(stream for stream in (result.stdout, result.stderr) if stream)


# %% every interface of the repository


@dataclass
class WorkspaceOrmInterfaces:
    """
    The ORM interfaces of a checkout, as one unit.
    """

    interfaces: Sequence[OrmInterface]
    """
    The interfaces ordered by dependency: each generator imports the already generated
    interfaces of the packages listed before it.
    """

    @property
    def are_generated(self) -> bool:
        """
        Whether every interface holds generated content.
        """
        return all(interface.is_generated for interface in self.interfaces)

    def regenerate(self, show_generator_output: bool = False) -> None:
        """
        Build every interface anew, from an empty state and in dependency order.

        ..note:: This takes about a minute and a half, since every package's generator
            introspects its whole class hierarchy.

        :param show_generator_output: Whether to let the generators write to the
            terminal. Their logging and the progress bar cannot share it, so asking for
            one leaves out the other.
        """
        for interface in self.interfaces:
            interface.remove()

        progress = tqdm(
            self.interfaces,
            desc=PROGRESS_DESCRIPTION,
            unit="interface",
            disable=show_generator_output,
        )
        for interface in progress:
            progress.set_postfix_str(interface.package_name)
            interface.generate(show_generator_output=show_generator_output)

    def ensure_generated(self, show_generator_output: bool = False) -> bool:
        """
        Leave the checkout with interfaces it can persist objects through.

        A checkout that is missing one is built whole rather than in part: a generator
        reads the interfaces of the packages before it, so the one that is missing
        decides nothing about which of the others are still valid.

        :param show_generator_output: Whether to let the generators write to the
            terminal rather than reporting progress.
        :return: Whether they had to be built.
        """
        if self.are_generated:
            return False
        self.regenerate(show_generator_output=show_generator_output)
        return True


WORKSPACE_ORM_INTERFACES = WorkspaceOrmInterfaces(
    tuple(
        OrmInterface(package_name, REPOSITORY_ROOT)
        for package_name in (
            "semantic_digital_twin",
            "giskardpy",
            "coraplex",
            "segmind",
            "experiments",
        )
    )
)
"""
The ORM interfaces of this repository.
"""
