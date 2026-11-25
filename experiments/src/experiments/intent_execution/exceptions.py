from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException

# %% planning an intent


@dataclass
class IntentNotPlannable(DataclassException):
    """
    Raised when no plan is built for an intent.
    """

    intent: str = field(kw_only=True)
    """
    The name of the intent.
    """

    plannable: list[str] = field(kw_only=True)
    """
    The names of the intents a plan is built for.
    """

    def error_message(self) -> str:
        return f"No plan is built for {self.intent}."

    def suggest_correction(self) -> str:
        return f"Send one of the plannable intents: {', '.join(self.plannable)}."


@dataclass
class RoleNotGrounded(DataclassException):
    """
    Raised when an intent names no world entity for a role its plan needs.
    """

    role: str = field(kw_only=True)
    """
    The role the intent left open.
    """

    intent: str = field(kw_only=True)
    """
    The name of the intent.
    """

    def error_message(self) -> str:
        return (
            f"{self.intent} needs an entity for '{self.role}', the intent names none."
        )

    def suggest_correction(self) -> str:
        return (
            f"Publish the intent only once '{self.role}' names an entity of the world."
        )


@dataclass
class EntityInNoRoom(DataclassException):
    """
    Raised when an intent names an entity that stands over the floor of no room, so the
    robot cannot drive to it.
    """

    entity: str = field(kw_only=True)
    """
    The name of the entity.
    """

    def error_message(self) -> str:
        return f"{self.entity} stands over the floor of no room."

    def suggest_correction(self) -> str:
        return "Ask for something inside a room, or annotate the room it is in."


# %% asking people


@dataclass
class HelpSkillNotServedError(DataclassException):
    """
    Raised when the robot has people to ask and Speech2Plan's help skill is not served.
    """

    skill_path: str = field(kw_only=True)
    """
    Where the skill was expected.
    """

    timeout: float = field(kw_only=True)
    """
    Seconds the robot waited for the skill.
    """

    def error_message(self) -> str:
        return f"The skill {self.skill_path} was not served within {self.timeout} s."

    def suggest_correction(self) -> str:
        return "Start Speech2Plan's intent node, which serves the help skill."
