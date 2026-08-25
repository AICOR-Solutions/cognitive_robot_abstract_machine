"""
Constructs for expressing Pearl-style causal (``do()``) queries in the Entity Query
Language.

Kept in its own module rather than folded into ``operators/core_logical_operators.py``
or ``core/variable.py`` (the modules :class:`Cause`/:class:`CausesEffect` otherwise
resemble) so causal-specific code has its own, easily reviewable surface, mirroring how
``probabilistic_model`` isolates its own causal code under a ``causal`` subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Iterable, List

import random_events.variable
from krrood.entity_query_language.core.base_expressions import (
    BinaryExpression,
    OperationResult,
    UnaryExpression,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresEqualityComparator,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        ProbabilisticCircuit,
    )


@dataclass(eq=False, repr=False)
class Cause(Literal):
    """
    Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as
    a ``do()``-intervention target searched for by the query, rather than an observed
    value.

    ``arm=CAUSE`` means: find the value of ``arm`` whose intervention (Pearl's
    ``do(arm=value)``) best explains the effect declared via
    :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`. Always wraps
    ``Ellipsis`` -- there is no pinned-value form; pin a value with a plain assignment
    (``arm=0.3``) instead.
    """

    _value_: Any = field(default=Ellipsis, init=False)


class CauseSentinel:
    """
    Type of :data:`CAUSE`.

    A *sentinel* here means a fixed marker value whose only job is to be recognised and
    swapped out later, the same role ``Ellipsis`` (``...``) already plays for a plain
    underspecified field.

    A distinct type -- not ``Cause`` itself -- because unlike a plain literal kwarg
    (``arm=0.3``), whose ``Literal`` wrapper :meth:`AttributeMatch.assigned_variable
    <krrood.entity_query_language.query.match.AttributeMatch.assigned_variable>` builds
    fresh, on the spot, once the attribute it belongs to is known, a bare
    ``cause()``/``CAUSE`` is a fully-built value the *caller* supplies before any
    attribute is known. If ``CAUSE`` were itself a ``Cause`` instance, every field
    marked with it would share that one object; the second field's type backfill (see
    :meth:`AttributeMatch.assigned_variable
    <krrood.entity_query_language.query.match.AttributeMatch.assigned_variable>`) would
    then silently overwrite the first's. :meth:`~krrood.entity_query_language.query.match.Match.resolve`
    avoids this by replacing each occurrence of the sentinel with its own fresh
    :class:`Cause` when it walks the kwargs.
    """

    def __repr__(self) -> str:
        return "CAUSE"


CAUSE = CauseSentinel()
"""
Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as a
``do()``-intervention target, without the parentheses of
:func:`~krrood.entity_query_language.factories.cause` (``arm=CAUSE`` instead of
``arm=cause()``).

The two are fully equivalent at runtime; ``cause()`` exists only so
this marker reads the same way as the query's other field-marking factories, which are
all calls (e.g. :func:`~krrood.entity_query_language.factories.set_of`) -- pick
whichever spelling reads better in a given query.
"""


@dataclass(eq=False, repr=False)
class Confounder(Literal):
    """
    Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument
    as a variable to adjust for when searching a :class:`Cause` intervention -- Pearl's
    backdoor-criterion adjustment set Z in
    ``P(effect | do(cause=v)) = sum_z P(effect | cause=v, Z=z) * P(Z=z)``.

    ``season=CONFOUNDER`` means: season is a common cause of the searched
    :class:`Cause` and the declared effect, and must be summed back out rather than
    left baked into the correlation between them. Always wraps ``Ellipsis``, the same
    as :class:`Cause`.
    """

    _value_: Any = field(default=Ellipsis, init=False)


class ConfounderSentinel:
    """
    Type of :data:`CONFOUNDER` -- the :class:`Confounder` counterpart of
    :class:`CauseSentinel`, for the same reason: each field it marks needs its own fresh
    :class:`Confounder` rather than sharing one instance.
    """

    def __repr__(self) -> str:
        return "CONFOUNDER"


CONFOUNDER = ConfounderSentinel()
"""
Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as a
variable to adjust for when searching a :class:`Cause` intervention -- see
:class:`Confounder`.
"""


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
        if (
            not isinstance(self._child_, BinaryExpression)
            or not self._child_._is_equality_literal_comparator_or_conjunction_()
        ):
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

    When there is more than one, each is tried independently -- there is no joint,
    multi-variable intervention -- and the one with the highest
    :attr:`~krrood.entity_query_language.operators.causal.ScoredIntervention.effect_probability_given_region`
    becomes the primary cause: the candidate whose own best region gives the highest
    ``P(effect | do(cause in best_region))``.
    """

    effect_variable: random_events.variable.Variable
    """
    The variable a ``causes_effect(...)`` condition declares as the effect.
    """

    confounder_variables: List[random_events.variable.Variable]
    """
    Variables assigned a ``CONFOUNDER`` marker: Pearl's backdoor-criterion adjustment
    set, summed back out of each cause candidate's interventional probability so it is
    not left baked into the correlation between cause and effect.

    Empty for a query with no confounders declared -- the interventional search then
    falls back to an empty adjustment set, exactly as before ``CONFOUNDER`` existed.
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

    narrowed_circuit: ProbabilisticCircuit
    """
    The interventional joint, truncated to the effect condition and this candidate's
    best region -- what the query samples from if this candidate turns out to be the
    primary cause.
    """
