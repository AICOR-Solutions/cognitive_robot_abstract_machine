from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory


@dataclass
class ArtVipObjectNotFoundError(DataclassException, LookupError):
    """
    Raised when no ArtVIP dataset entry matches a requested category and object name.
    """

    category: ArtVipCategory
    """The category that was searched."""

    name: str
    """
    The object name that could not be found.
    """

    def error_message(self) -> str:
        return (
            f"No ArtVIP object named '{self.name}' found in category "
            f"'{self.category.value}'."
        )

    def suggest_correction(self) -> str:
        return "Call ArtVipDatasetLoader.available_objects(category) for the names available in that category."


@dataclass
class ArtVipMainStageFileAmbiguousError(DataclassException, LookupError):
    """
    Raised when an ArtVIP object's directory does not contain exactly one top-level USD
    file to open as its main stage - either none (a folder present without its stage,
    or fully nested under a further subcategory) or more than one (an object whose main
    file cannot be picked out unambiguously).
    """

    category: ArtVipCategory
    """The object's category."""

    name: str
    """
    The object name whose directory was searched.
    """

    candidates: tuple[str, ...]
    """The top-level ``.usd`` file paths found directly in the object's directory."""

    def error_message(self) -> str:
        return (
            f"ArtVIP object '{self.name}' in category '{self.category.value}' has "
            f"{len(self.candidates)} top-level USD files, not exactly one: "
            f"{self.candidates}."
        )

    def suggest_correction(self) -> str:
        return (
            "Inspect the object's directory on the Hugging Face repository to "
            "identify its actual main stage file."
        )


@dataclass
class ArtVipJointMissingChildBodyError(DataclassException, LookupError):
    """
    Raised when an ArtVIP object's USD stage contains a physics joint whose ``body1``
    relationship (the joint's child link) has no target.

    Unlike ``body0``, where an unset target is the USD convention for "the object's own
    root frame", ``body1`` has no such meaning for this loader - every joint is expected
    to connect a link into the object, so a joint with no child leaves nothing for this
    loader to build a Connection to.
    """

    category: ArtVipCategory
    """The object's category."""

    name: str
    """
    The object's name.
    """

    joint_path: str
    """The joint prim's stage path."""

    def error_message(self) -> str:
        return (
            f"ArtVIP object '{self.name}' in category '{self.category.value}' has a "
            f"joint at '{self.joint_path}' with no body1 target."
        )

    def suggest_correction(self) -> str:
        return "Inspect the joint prim's body1 relationship on the object's USD stage."


@dataclass
class ArtVipUnsupportedJointTypeError(DataclassException, LookupError):
    """
    Raised when an ArtVIP object's USD stage contains a physics joint of a type this
    loader does not build a Connection for (only Fixed/Revolute/PrismaticJoint are
    handled).

    Silently skipping it would build a World missing whatever link that joint's body1 is
    the only connection to.
    """

    category: ArtVipCategory
    """The object's category."""

    name: str
    """
    The object's name.
    """

    joint_path: str
    """The unsupported joint prim's stage path."""

    joint_type: str
    """
    The unsupported joint prim's USD type name.
    """

    def error_message(self) -> str:
        return (
            f"ArtVIP object '{self.name}' in category '{self.category.value}' has a "
            f"joint of unsupported type '{self.joint_type}' at '{self.joint_path}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Add a Connection class for this joint type to "
            "_JOINT_CONNECTION_CLASSES and handle its schema in _connect_joint."
        )
