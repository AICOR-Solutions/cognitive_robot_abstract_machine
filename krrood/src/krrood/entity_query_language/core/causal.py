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
from enum import StrEnum
from typing_extensions import Any, Iterable

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    UnaryExpression,
)
from krrood.entity_query_language.core.helpers import (
    is_literal_comparator_or_conjunction,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresLiteralComparator,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)


class CausalRole(StrEnum):
    """
    The role a variable plays in a causal query.
    """

    CAUSE = "cause"
    """
    An intervention target, declared with ``cause()``.
    """

    EFFECT = "effect"
    """
    A declared effect, declared with ``causes_effect(...)``.
    """


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

    _value_: Any = field(default=Ellipsis, init=False, kw_only=True)


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
        if not is_literal_comparator_or_conjunction(self._child_):
            raise CausesEffectRequiresLiteralComparator(self._child_)

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for child_result in self._evaluate_child_as_condition_(self._child_, sources):
            yield self._build_operation_result_with_truth_(
                child_result.is_true, child_result.bindings, child_result
            )
