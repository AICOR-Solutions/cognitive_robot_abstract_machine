"""
Tests for building the ORM interfaces a checkout needs before it can persist objects.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typing_extensions import List, Optional, Set, Tuple

from cognitive_robot_abstract_machine import orm_interfaces
from cognitive_robot_abstract_machine.exceptions import (
    MissingOrmGeneratorError,
    OrmGenerationFailedError,
)
from cognitive_robot_abstract_machine.orm_interfaces import (
    INTERFACE_FILE_NAME,
    OrmInterface,
    REPOSITORY_ROOT,
    WORKSPACE_ORM_INTERFACES,
    WorkspaceOrmInterfaces,
)

from .dataset import failing_generate_orm, generate_orm

# %% a checkout of packages that generate an interface

PACKAGE_NAMES: Tuple[str, ...] = ("upstream", "downstream")
"""
Packages of the checkout under test, in dependency order.
"""

STALE_INTERFACE_CONTENT = "# interface of a previous run\n"
"""
Content the interfaces of the checkout hold before it is regenerated.
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """
    A git checkout of two packages whose interfaces hold content of a previous run.
    """
    for package_name in PACKAGE_NAMES:
        package_root = tmp_path / package_name
        (package_root / "scripts").mkdir(parents=True)
        shutil.copy(
            Path(generate_orm.__file__),
            package_root / "scripts" / "generate_orm.py",
        )
        interface = generate_orm.interface_of(package_root)
        interface.parent.mkdir(parents=True)
        interface.write_text(STALE_INTERFACE_CONTENT, encoding="utf-8")

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "--all"], cwd=tmp_path, check=True, capture_output=True
    )
    return tmp_path


@pytest.fixture
def workspace(checkout: Path) -> WorkspaceOrmInterfaces:
    """
    The ORM interfaces of the checkout under test.
    """
    return WorkspaceOrmInterfaces(
        tuple(OrmInterface(package_name, checkout) for package_name in PACKAGE_NAMES)
    )


def tracked_interfaces(repository_root: Path) -> Set[str]:
    """
    Read which ORM interfaces of a checkout git tracks.

    :param repository_root: Root of the checkout.
    :return: The repository-relative paths of the tracked interfaces.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--", f"*/{INTERFACE_FILE_NAME}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(listing.stdout.splitlines())


def git_ignores(repository_root: Path, path: Path) -> bool:
    """
    Ask git whether a checkout's ignore rules cover a path.

    :param repository_root: Root of the checkout.
    :param path: Path to ask about.
    :return: Whether git ignores it.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=repository_root,
        ).returncode
        == 0
    )


# %% telling a generated checkout from a fresh one


def test_workspace_is_not_generated_while_one_interface_is_missing(
    workspace: WorkspaceOrmInterfaces,
):
    assert workspace.are_generated

    workspace.interfaces[-1].remove()

    assert workspace.interfaces[0].is_generated
    assert not workspace.interfaces[-1].is_generated
    assert not workspace.are_generated


# %% regeneration


def test_regeneration_runs_the_generators_in_dependency_order(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert [record.package_name for record in records] == list(PACKAGE_NAMES)


def test_regeneration_clears_every_interface_before_generating_any(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert records[0].generated_packages == []
    assert records[1].generated_packages == [PACKAGE_NAMES[0]]


def test_regeneration_fills_every_interface(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()

    assert workspace.are_generated
    for interface in workspace.interfaces:
        assert interface.path.read_text(
            encoding="utf-8"
        ) == generate_orm.interface_content(interface.package_name)


# %% building only a checkout that needs it


def test_a_checkout_that_already_has_its_interfaces_is_left_alone(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    assert not workspace.ensure_generated()

    assert generate_orm.read_generation_log(checkout) == []


def test_a_checkout_missing_one_interface_has_every_one_built(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    workspace.interfaces[-1].remove()

    assert workspace.ensure_generated()

    records = generate_orm.read_generation_log(checkout)
    assert [record.package_name for record in records] == list(PACKAGE_NAMES)


# %% incomplete checkouts


def test_missing_generator_names_its_package(workspace: WorkspaceOrmInterfaces):
    incomplete = workspace.interfaces[-1]
    incomplete.generator.unlink()

    with pytest.raises(MissingOrmGeneratorError) as error:
        workspace.regenerate()

    assert error.value.package_name == incomplete.package_name
    assert error.value.path == incomplete.generator


# %% this repository


def test_every_workspace_package_has_a_generator():
    without_generator = [
        interface.package_name
        for interface in WORKSPACE_ORM_INTERFACES.interfaces
        if not interface.generator.exists()
    ]
    assert without_generator == []


def test_this_repository_tracks_no_generated_interface():
    assert tracked_interfaces(REPOSITORY_ROOT) == set()


def test_this_repository_ignores_every_generated_interface():
    not_ignored = [
        interface.package_name
        for interface in WORKSPACE_ORM_INTERFACES.interfaces
        if not git_ignores(REPOSITORY_ROOT, interface.path)
    ]
    assert not_ignored == []


def test_this_repository_ignores_a_generated_interface_outside_a_workspace_package():
    krrood_test_dataset_interface = (
        REPOSITORY_ROOT / "test" / "krrood_test" / "dataset" / INTERFACE_FILE_NAME
    )

    assert git_ignores(REPOSITORY_ROOT, krrood_test_dataset_interface)


# %% what a build lets through to the terminal


@pytest.fixture
def failing_workspace(checkout: Path) -> WorkspaceOrmInterfaces:
    """
    The interfaces of a checkout whose first generator fails.
    """
    shutil.copy(
        Path(failing_generate_orm.__file__),
        checkout / PACKAGE_NAMES[0] / "scripts" / "generate_orm.py",
    )
    return WorkspaceOrmInterfaces(
        tuple(OrmInterface(package_name, checkout) for package_name in PACKAGE_NAMES)
    )


@dataclass
class RecordedProgress:
    """
    A stand-in for the progress bar, remembering how it was asked to report.
    """

    wrapped: List[OrmInterface] = field(default_factory=list)
    """
    The interfaces the bar was given to iterate.
    """

    disabled: Optional[bool] = None
    """
    Whether the bar was asked to keep quiet.
    """

    labelled: List[str] = field(default_factory=list)
    """
    The labels the bar was asked to show, one per interface it reported on.
    """

    def __call__(self, interfaces, **keywords) -> RecordedProgress:
        self.wrapped = list(interfaces)
        self.disabled = keywords.get("disable")
        return self

    def __iter__(self):
        return iter(self.wrapped)

    def set_postfix_str(self, text: str) -> None:
        """
        Record the label a real bar would render beside itself.

        :param text: The label.
        """
        self.labelled.append(text)


@pytest.fixture
def recorded_progress(monkeypatch) -> RecordedProgress:
    """
    Replace the progress bar with one that records how it was used.
    """
    progress = RecordedProgress()
    monkeypatch.setattr(orm_interfaces, "tqdm", progress)
    return progress


def test_a_quiet_build_keeps_the_generator_output_off_the_terminal(
    workspace: WorkspaceOrmInterfaces, capfd
):
    workspace.regenerate()

    assert generate_orm.PROGRESS_LINE not in capfd.readouterr().out


def test_a_build_showing_generator_output_lets_it_through(
    workspace: WorkspaceOrmInterfaces, capfd
):
    workspace.regenerate(show_generator_output=True)

    assert capfd.readouterr().out.count(generate_orm.PROGRESS_LINE) == len(
        PACKAGE_NAMES
    )


def test_a_failing_generator_reports_what_it_wrote(
    failing_workspace: WorkspaceOrmInterfaces,
):
    with pytest.raises(OrmGenerationFailedError) as failure:
        failing_workspace.regenerate()

    assert failing_generate_orm.DIAGNOSTIC in str(failure.value)
    assert failure.value.package_name == PACKAGE_NAMES[0]


def test_progress_is_reported_for_every_interface(
    workspace: WorkspaceOrmInterfaces, recorded_progress: RecordedProgress
):
    workspace.regenerate()

    assert recorded_progress.wrapped == list(workspace.interfaces)
    assert recorded_progress.labelled == list(PACKAGE_NAMES)
    assert recorded_progress.disabled is False


def test_progress_stays_out_of_the_way_of_the_generator_output(
    workspace: WorkspaceOrmInterfaces, recorded_progress: RecordedProgress
):
    workspace.regenerate(show_generator_output=True)

    assert recorded_progress.disabled is True
