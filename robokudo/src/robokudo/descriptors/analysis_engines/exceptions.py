"""
Exceptions raised while configuring analysis engines.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from robokudo.exceptions import RoboKudoError

# %% Settings


@dataclass
class UnknownSettingsSection(RoboKudoError, ValueError):
    """
    Raised when a settings file holds sections its analysis engine does not know.
    """

    unknown_sections: list[str]
    """
    The sections the analysis engine does not know.
    """

    known_sections: list[str]
    """
    The sections the analysis engine reads.
    """

    def error_message(self) -> str:
        return f"The settings file holds unknown sections {self.unknown_sections}."

    def suggest_correction(self) -> str:
        return f"name only the sections {self.known_sections}."
