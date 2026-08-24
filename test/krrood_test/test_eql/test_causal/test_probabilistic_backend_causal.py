import math
from dataclasses import dataclass
from enum import Enum, auto

import pytest
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from random_events.interval import closed
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    NoCauseVariablesForRanking,
    NoCausesEffectConditionForCause,
)
from krrood.entity_query_language.factories import an, cause
from krrood.parametrization.exceptions import (
    DoRequiresCausalCircuitModel,
    MultipleEffectVariablesNotSupported,
)
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters


class Outcome(Enum):
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class Pick:
    arm: float
    success: Outcome


def _build_two_region_causal_circuit() -> tuple:
    """
    A causal circuit where `arm`'s intervention region determines `success`'s value:
    two equal-weight mixture components, `arm`/`success` co-varying together.

        Low:  arm in [0, 1], success = FAILURE (deterministically)
        High: arm in [2, 3], success = SUCCESS (deterministically)

    Ground truth: interventionally forcing `arm` into the high region is what makes
    `success` equal SUCCESS -- the query
    `an(Pick)(arm=cause(), ...).causes_effect(success == Outcome.SUCCESS)` should
    therefore only ever return instances with `arm` in `[2, 3]`.
    """
    arm = Continuous("Pick.arm")
    success = Symbolic("Pick.success", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for arm_range, outcome in [((0, 1), Outcome.FAILURE), ((2, 3), Outcome.SUCCESS)]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=arm, interval=closed(*arm_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=success,
                    probabilities=MissingDict(float, {hash(outcome): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph([arm], [success]),
        [arm],
        [success],
    )
    return causal_circuit, arm, success


@dataclass
class TwoCauseCandidatesCircuit:
    """
    A causal circuit with two cause candidates of unequal explanatory strength for the
    same effect, and the variables/circuit needed to query and verify it.
    """

    causal_circuit: CausalCircuit
    decisive_cause: Continuous
    """
    `decisive_cause`'s region alone almost perfectly determines `success`.
    """

    uninformative_cause: Continuous
    """
    `uninformative_cause` has the same distribution regardless of `success`.
    """

    success: Symbolic


def _build_two_cause_candidates_circuit() -> TwoCauseCandidatesCircuit:
    """
    Two equal-weight mixture components over three variables:

        Low:  decisive in [0, 1], uninformative in [0, 2], success = FAILURE
        High: decisive in [2, 3], uninformative in [0, 2], success = SUCCESS

    `decisive` separates the components perfectly (its regions [0, 1] / [2, 3] each
    occur with exactly one outcome), so restricting it to its best region ([2, 3])
    makes `P(success == SUCCESS | do(decisive in [2, 3]))` near-certain (~1.0).
    `uninformative` has the identical range [0, 2] in both components, so its own
    support forms a single, whole-domain region compatible with either outcome --
    restricting it to that region changes nothing, so
    `P(success == SUCCESS | do(uninformative in [0, 2]))` stays at the prior, 0.5.
    `decisive` should therefore always win as the primary cause.
    """
    decisive = Continuous("Pick.arm")
    uninformative = Continuous("Pick.grip")
    success = Symbolic("Pick.success", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for decisive_range, outcome in [
        ((0, 1), Outcome.FAILURE),
        ((2, 3), Outcome.SUCCESS),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=decisive, interval=closed(*decisive_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=uninformative, interval=closed(0, 2).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=success,
                    probabilities=MissingDict(float, {hash(outcome): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph(
            [decisive, uninformative], [success]
        ),
        [decisive, uninformative],
        [success],
    )
    return TwoCauseCandidatesCircuit(causal_circuit, decisive, uninformative, success)


# %% error handling


def test_raises_when_model_registry_does_not_resolve_a_causal_circuit():
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)

    backend = ProbabilisticBackend(model_registry=FullyFactorizedRegistry())
    with pytest.raises(DoRequiresCausalCircuitModel):
        list(match.evaluate(backend=backend))


def test_raises_when_cause_has_no_causes_effect_condition():
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit})
    )
    with pytest.raises(NoCausesEffectConditionForCause):
        list(match.evaluate(backend=backend))


# %% end-to-end correctness


def test_results_satisfy_the_causes_effect_condition():
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=10,
    )
    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    for result in results:
        assert result.success == Outcome.SUCCESS


def test_results_land_in_the_intervention_region_that_causes_the_effect():
    """
    `arm` must land specifically in [2, 3] -- the region whose intervention causes
    `success == SUCCESS` -- not merely anywhere in `arm`'s domain.

    This is the
    cause/effect *correlation* backdoor_adjustment's per-branch ProductUnit structure
    is supposed to preserve; it regressed to marginal-only correctness when its region
    extraction collapsed disjoint branches into one, and is covered directly against
    `CausalCircuit` in `probabilistic_model_test/test_causal/test_causal_circuit.py`
    (`BestDisjointRegionTestCase`).
    """
    causal_circuit, _, _ = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=10,
    )
    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    for result in results:
        assert 2.0 <= result.arm <= 3.0


def test_pipeline_reproduces_directly_computed_backdoor_adjustment_and_best_region():
    """
    The EQL pipeline's interventional branch is glue over
    `CausalCircuit.backdoor_adjustment` and `CausalCircuit._best_disjoint_region` (both
    untouched, already-tested primitives) -- this checks it wires them together
    correctly by reproducing the same region a direct, hand-written call to those
    primitives selects.
    """
    causal_circuit, arm, success = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)

    parameters = UnderspecifiedParameters(match)
    [cause_variable] = parameters.search_cause_variables
    [effect_variable] = parameters.effect_variables_from_causes_effect
    assert cause_variable == arm
    assert effect_variable == success

    # the reference computation: exactly what the plan's design describes as the
    # interventional branch, called directly against the causal circuit
    interventional = causal_circuit.backdoor_adjustment(cause_variable, effect_variable)
    effect_truncated, _ = interventional.truncated(
        parameters.truncation_assignments_from_where_conditions
    )
    expected_best_region = causal_circuit._best_disjoint_region(
        cause_variable, effect_truncated
    )
    expected_narrowed, _ = effect_truncated.truncated(
        expected_best_region.fill_missing_variables_pure(effect_truncated.variables)
    )

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=1,
    )
    scored = backend._score_intervention(
        causal_circuit,
        cause_variable,
        effect_variable,
        parameters.truncation_assignments_from_where_conditions,
    )

    assert scored.cause_variable == cause_variable
    assert scored.narrowed_circuit.probability(
        expected_best_region.fill_missing_variables_pure(
            scored.narrowed_circuit.variables
        )
    ) == pytest.approx(1.0)
    assert expected_narrowed.probability(
        expected_best_region.fill_missing_variables_pure(expected_narrowed.variables)
    ) == pytest.approx(1.0)


# %% multiple cause() candidates


def test_multiple_cause_candidates_selects_the_decisive_one_as_primary():
    """
    With two `cause()` candidates of unequal explanatory strength for the same effect,
    the primary cause the query narrows to must be the decisive one -- not because joint
    intervention was computed (it wasn't; `backdoor_adjustment` cannot), but because
    each candidate was searched independently and the decisive one scored higher (see
    `_build_two_cause_candidates_circuit`).
    """
    circuit = _build_two_cause_candidates_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)

    parameters = UnderspecifiedParameters(match)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: circuit.causal_circuit})
    )
    primary = backend._resolve_primary_intervention(
        circuit.causal_circuit,
        [circuit.decisive_cause, circuit.uninformative_cause],
        circuit.success,
        parameters.truncation_assignments_from_where_conditions,
        match,
    )

    assert primary.cause_variable == circuit.decisive_cause
    assert primary.effect_probability_given_region == pytest.approx(1.0, abs=0.02)


def test_the_uninformative_candidate_scores_lower_than_the_decisive_one():
    circuit = _build_two_cause_candidates_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)
    parameters = UnderspecifiedParameters(match)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: circuit.causal_circuit})
    )

    decisive_score = backend._score_intervention(
        circuit.causal_circuit,
        circuit.decisive_cause,
        circuit.success,
        parameters.truncation_assignments_from_where_conditions,
    )
    uninformative_score = backend._score_intervention(
        circuit.causal_circuit,
        circuit.uninformative_cause,
        circuit.success,
        parameters.truncation_assignments_from_where_conditions,
    )

    assert (
        decisive_score.effect_probability_given_region
        > uninformative_score.effect_probability_given_region
    )


def test_multiple_effect_variables_are_rejected():
    causal_circuit, arm, success = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(
        match.variable.success == Outcome.SUCCESS, match.variable.arm == 2.5
    )

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit})
    )
    with pytest.raises(MultipleEffectVariablesNotSupported) as excinfo:
        list(match.evaluate(backend=backend))
    assert set(excinfo.value.variables) == {success, arm}


# %% rank_causes


@dataclass
class PickAttempt:
    arm: float
    grip: float
    success: Outcome


def _build_pick_attempt_ranking_circuit() -> TwoCauseCandidatesCircuit:
    """
    The same two-candidate structure as `_build_two_cause_candidates_circuit`
    (`decisive` separates the outcome perfectly, `uninformative` does not), built for
    :class:`PickAttempt` instead of :class:`Pick` so `rank_causes` tests can query both
    candidates through a real match without touching `Pick`'s own widely-reused fixture.
    """
    decisive = Continuous("PickAttempt.arm")
    uninformative = Continuous("PickAttempt.grip")
    success = Symbolic("PickAttempt.success", domain=Set.from_iterable(Outcome))
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for decisive_range, outcome in [
        ((0, 1), Outcome.FAILURE),
        ((2, 3), Outcome.SUCCESS),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=decisive, interval=closed(*decisive_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=uninformative, interval=closed(0, 2).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=success,
                    probabilities=MissingDict(float, {hash(outcome): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph(
            [decisive, uninformative], [success]
        ),
        [decisive, uninformative],
        [success],
    )
    return TwoCauseCandidatesCircuit(causal_circuit, decisive, uninformative, success)


def test_rank_causes_returns_every_candidate_ranked_highest_first():
    circuit = _build_pick_attempt_ranking_circuit()
    match = an(PickAttempt)(arm=cause(), grip=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({PickAttempt: circuit.causal_circuit})
    )

    ranking = backend.rank_causes(match)

    assert [scored.cause_variable for scored in ranking] == [
        circuit.decisive_cause,
        circuit.uninformative_cause,
    ]
    assert ranking[0].effect_probability_given_region == pytest.approx(1.0, abs=0.02)
    assert ranking[1].effect_probability_given_region == pytest.approx(0.5, abs=0.05)


def test_rank_causes_does_not_change_the_result_of_evaluate():
    """
    `rank_causes` is a read alongside the existing search, not a replacement for it --
    `evaluate()`'s primary-cause selection for the same multi-`cause()` match must be
    unaffected by calling `rank_causes` on it.
    """
    circuit = _build_pick_attempt_ranking_circuit()
    match = an(PickAttempt)(arm=cause(), grip=cause(), success=...)
    match.causes_effect(match.variable.success == Outcome.SUCCESS)
    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({PickAttempt: circuit.causal_circuit}),
        number_of_samples=10,
    )

    ranking = backend.rank_causes(match)
    results = list(match.evaluate(backend=backend))

    assert ranking[0].cause_variable == circuit.decisive_cause
    assert len(results) == 10
    assert all(2.0 <= result.arm <= 3.0 for result in results)


def test_rank_causes_rejects_a_match_with_no_cause_fields():
    match = an(PickAttempt)(arm=..., grip=..., success=...)
    backend = ProbabilisticBackend(model_registry=CausalCircuitRegistry({}))
    with pytest.raises(NoCauseVariablesForRanking):
        backend.rank_causes(match)
