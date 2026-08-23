from dataclasses import dataclass

from typing_extensions import Any, List, Type

import random_events.variable
from krrood.entity_query_language.core.causal import CausalRole
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import ConditionType
from krrood.exceptions import DataclassException, InputError


@dataclass
class WhereExpressionIsFirstOrder(DataclassException):
    """
    Raised when a quantified `Where` expression is asked to be translated into a random
    event, since a product algebra is propositional and constrains a fixed set of
    variables instead of the objects a quantifier ranges over.
    """

    where_expression: ConditionType
    """
    The quantified expression that has no random event representation.
    """

    def error_message(self) -> str:
        return (
            f"The where expression {self.where_expression} quantifies over a variable, "
            f"which no fixed set of random event variables represents."
        )

    def suggest_correction(self) -> str:
        return (
            "State the condition over the attributes that are parameterized instead of "
            "quantifying, or evaluate the quantified condition on the query results."
        )


@dataclass
class WhereExpressionHasNoRandomEventRepresentation(DataclassException):
    """
    Raised when a part of a `Where` expression constrains something that no random event
    variable stands for, for example a comparison between two variables.
    """

    where_expression: ConditionType
    """
    The expression that has no random event representation.
    """

    def error_message(self) -> str:
        return (
            f"The where expression {self.where_expression} is neither a logical operator "
            f"nor a comparison between a variable and a literal, so it constrains no "
            f"random event variable."
        )

    def suggest_correction(self) -> str:
        return (
            "Compare a variable against a literal value, and combine such comparisons "
            "with and_, or_ and not_ only."
        )


@dataclass
class EmptyVariableDomain(InputError):
    variable: Variable

    def error_message(self) -> str:
        return f"The domain of the variable {self.variable} is empty. Domains must be non-empty for the variable to be valid."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidEllipsis(InputError):
    type_: Type

    def error_message(self) -> str:
        return f"Ellipsis is not allowed for type {self.type_}. Ellipsis are only allowed for the leaf objects (random events compatible types, see `random_events.variable.Variable.compatible_types`)."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DoRequiresCausalCircuitModel(DataclassException):
    """
    Raised when a match has a ``cause()`` intervention but the model registry resolved a
    model that is not a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.

    ``cause()`` needs a registered causal graph to know what to cut when intervening; a
    plain (non-causal) probabilistic model has no such graph.
    """

    resolved_model: Any

    def error_message(self) -> str:
        return (
            f"A cause() intervention was used, but the model registry returned "
            f"{type(self.resolved_model).__name__}, not a CausalCircuit. cause() needs a "
            f"registered causal graph to know what to cut when intervening."
        )

    def suggest_correction(self) -> str:
        return "Use a model registry that returns a CausalCircuit for this class (see CausalCircuitRegistry)."


@dataclass
class MultipleCauseOrEffectVariablesNotSupported(DataclassException):
    """
    Raised when a query has more than one ``cause()`` intervention target, or more than
    one effect variable declared via ``causes_effect(...)``.

    v1 causal search supports exactly one cause variable and one effect variable per
    query -- joint interventions and multi-effect causal queries are not yet supported,
    matching :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.backdoor_adjustment`'s
    single-cause-variable, single-effect-variable signature.
    """

    role: CausalRole
    """
    Which role -- cause or effect -- has too many variables.
    """

    variables: List[random_events.variable.Variable]
    """
    The variables found in that role.
    """

    _construct_by_role = {
        CausalRole.CAUSE: "cause()",
        CausalRole.EFFECT: "causes_effect(...)",
    }
    """
    The EQL construct that declares each role, for :meth:`suggest_correction`.
    """

    def error_message(self) -> str:
        return (
            f"Found {len(self.variables)} {self.role.value} variables "
            f"({[v.name for v in self.variables]}), but a causal search supports "
            f"exactly one."
        )

    def suggest_correction(self) -> str:
        return f"Use exactly one {self._construct_by_role[self.role]} per query."
