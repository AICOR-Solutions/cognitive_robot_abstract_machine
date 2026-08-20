"""
Building the ORM interfaces a test run needs.

Every package's ``ormatic_interface.py`` is generated rather than tracked, so a checkout
holds none until something builds them.
"""

from __future__ import annotations

import os
from enum import StrEnum

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES


class PytestEnvironmentVariable(StrEnum):
    """
    Environment variables pytest sets for the processes of a run.
    """

    XDIST_WORKER = "PYTEST_XDIST_WORKER"
    """
    Names the xdist worker a process is; absent in the controller.
    """


def regenerate_orm_interfaces() -> bool:
    """
    Build the ORM interfaces of this checkout, unless a worker is asking.

    The controller imports the root conftest before it starts any worker, so building
    there leaves the interfaces on disk by the time a worker imports a mapped
    datastructure. Letting the workers build too would set several processes writing the
    same files at once, each paying for the whole build again.

    :return: Whether this process built them.
    """
    if os.environ.get(PytestEnvironmentVariable.XDIST_WORKER):
        return False
    WORKSPACE_ORM_INTERFACES.regenerate()
    return True
