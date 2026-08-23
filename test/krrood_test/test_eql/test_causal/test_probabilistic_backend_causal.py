import math
from dataclasses import dataclass

import pytest
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
from random_events.interval import closed
from random_events.variable import Continuous

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.core.causal import CausalRole
from krrood.entity_query_language.exceptions import NoCausesEffectConditionForCause
from krrood.entity_query_language.factories import an, cause
from krrood.parametrization.exceptions import (
    DoRequiresCausalCircuitModel,
    MultipleCauseOrEffectVariablesNotSupported,
)
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters


@dataclass
class Pick:
    arm: float
    success: float


def _build_two_region_causal_circuit() -> tuple:
    """
    A causal circuit where `arm`'s intervention region determines `success`'s region:
    two equal-weight mixture components, `arm`/`success` co-varying together.

        Low:  arm in [0, 1], success in [0, 1]
        High: arm in [2, 3], success in [9, 10]

    Ground truth: interventionally forcing `arm` into the high region is what makes
    `success` land above 5 -- the query `an(Pick)(arm=cause(), ...).causes_effect(success > 5)`
    should therefore only ever return instances with `success` in `[9, 10]`.
    """
    arm = Continuous("Pick.arm")
    success = Continuous("Pick.success")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for arm_range, success_range in [((0, 1), (0, 1)), ((2, 3), (9, 10))]:
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
                UniformDistribution(
                    variable=success, interval=closed(*success_range).simple_sets[0]
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


# %% error handling


def test_raises_when_model_registry_does_not_resolve_a_causal_circuit():
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success > 5.0)

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
    match.causes_effect(match.variable.success > 5.0)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=10,
    )
    results = list(match.evaluate(backend=backend))

    assert len(results) == 10
    for result in results:
        assert result.success > 5.0


def test_pipeline_reproduces_directly_computed_backdoor_adjustment_and_best_region():
    """
    The EQL pipeline's interventional branch is glue over
    `CausalCircuit.backdoor_adjustment` and `CausalCircuit._best_region` (both
    untouched, already-tested primitives) -- this checks it wires them together
    correctly by reproducing the same region a direct, hand-written call to those
    primitives selects.
    """
    causal_circuit, arm, success = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=...)
    match.causes_effect(match.variable.success > 5.0)

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
    expected_best_region = causal_circuit._best_region(cause_variable, effect_truncated)
    expected_narrowed, _ = effect_truncated.truncated(
        expected_best_region.fill_missing_variables_pure(effect_truncated.variables)
    )

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit}),
        number_of_samples=1,
    )
    truncated = backend._narrow_to_best_intervention_region(
        causal_circuit,
        cause_variable,
        effect_truncated,
        match,
    )

    assert truncated.probability(
        expected_best_region.fill_missing_variables_pure(truncated.variables)
    ) == pytest.approx(1.0)
    assert expected_narrowed.probability(
        expected_best_region.fill_missing_variables_pure(expected_narrowed.variables)
    ) == pytest.approx(1.0)


def test_two_cause_variables_are_rejected():
    causal_circuit, arm, success = _build_two_region_causal_circuit()
    match = an(Pick)(arm=cause(), success=cause())
    match.causes_effect(match.variable.success > 5.0)

    backend = ProbabilisticBackend(
        model_registry=CausalCircuitRegistry({Pick: causal_circuit})
    )
    with pytest.raises(MultipleCauseOrEffectVariablesNotSupported) as excinfo:
        list(match.evaluate(backend=backend))
    assert excinfo.value.role == CausalRole.CAUSE
