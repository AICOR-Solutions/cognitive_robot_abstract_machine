"""
Constructs for expressing Pearl-style causal (``do()``) queries in the Entity Query
Language.

Kept physically separate from ``core/variable.py`` and
``operators/core_logical_operators.py`` (the modules
:class:`Cause`/:class:`CausesEffect` otherwise resemble) so causal-specific code has its
own, easily reviewable surface, mirroring how ``probabilistic_model`` isolates its own
causal code under a ``causal`` subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any, Iterable, List

import random_events.variable
from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    UnaryExpression,
)
from krrood.entity_query_language.core.helpers import (
    is_equality_literal_comparator_or_conjunction,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresEqualityComparator,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)


@dataclass(eq=False, repr=False)
class Cause(Literal):
    """
    Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as
    a ``do()``-intervention target searched for by the query, rather than an observed
    value.

    ``arm=cause()`` means: find the value of ``arm`` whose intervention (Pearl's
    ``do(arm=value)``) best explains the effect declared via
    :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`. Always wraps
    ``Ellipsis`` -- there is no pinned-value form; pin a value with a plain assignment
    (``arm=0.3``) instead.
    """

    _value_: Any = field(default=Ellipsis, init=False)


@dataclass(eq=False, repr=False)
class CausesEffect(LogicalOperator, UnaryExpression):
    """
    Tags a condition as the effect side of a causal query.

    Evaluates transparently -- the same truth value as its wrapped condition -- under
    every backend, so filtering behaves identically whether or not a condition is
    wrapped in :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`.
    Only :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`
    additionally reads it, to find which variable(s) a :class:`Cause` search should
    optimize for.
    """

    def __post_init__(self):
        super().__post_init__()
        if not is_equality_literal_comparator_or_conjunction(self._child_):
            raise CausesEffectRequiresEqualityComparator(self._child_)

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for child_result in self._evaluate_child_as_condition_(self._child_, sources):
            yield self._build_operation_result_with_truth_(
                child_result.is_true, child_result.bindings, child_result
            )


@dataclass
class CauseEffectVariables:
    """
    The cause candidates and effect variable a ``cause()`` search resolves to.
    """

    cause_variables: List[random_events.variable.Variable]
    """
    The variable(s) a ``cause()`` intervention is searched over.

    When there is more than one, each is tried independently and the one whose
    intervention best explains the effect becomes the primary cause -- there is no
    joint, multi-variable intervention.
    """

    effect_variable: random_events.variable.Variable
    """
    The variable a ``causes_effect(...)`` condition declares as the effect.
    """


@dataclass
class ScoredIntervention:
    """
    One cause candidate's best-region search result, scored for comparison against the
    other candidates when a query has more than one ``cause()`` field.
    """

    cause_variable: random_events.variable.Variable
    """
    The candidate cause variable this result is for.
    """

    effect_probability_given_region: float
    """
    The interventional probability that the effect holds, given this variable is
    restricted to its best region -- how *reliably* this candidate's best value produces
    the effect.

    Comparable across candidates regardless of how much of each candidate's own domain
    that region happens to cover: the higher one is the better explanation.
    """

    narrowed_circuit: Any
    """
    The interventional joint, truncated to the effect condition and this candidate's
    best region -- what the query samples from if this candidate turns out to be the
    primary cause.
    """
