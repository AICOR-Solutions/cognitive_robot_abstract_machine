---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Causal (`do()`) Queries

Ice cream sales and drowning incidents both rise in summer. Looking at the data alone,
more ice cream sold comes with more drownings, but banning ice cream would not save
anyone; warm weather drives both. *Observing* a value tells you what tends to come with
it; *setting* a value tells you what actually follows from it. The first is ordinary
probabilistic conditioning (`X=`); Pearl calls the second "intervention" (`do(X)`) and
argued the two are different questions whenever two fields share a hidden common cause
-- conditioning picks up the confounding correlation, intervention cuts it.

{py:func}`~krrood.entity_query_language.factories.cause` and
{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` route a query
through
{py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`'s
backdoor-adjustment machinery instead of plain conditioning, when evaluated with
{py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`.

### Seeing the difference

A minimal model makes the distinction concrete. `ice_cream_sales` and
`drowning_incidents` are both driven by a hidden season that pushes both high or both
low together (a warm-season branch, a cold-season branch), with no direct link between
the two:

```python
import math
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit, ProductUnit, SumUnit, leaf,
)
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit, MarginalDeterminismTreeNode,
)

ice_cream_sales = Continuous("ice_cream_sales")
drowning_incidents = Continuous("drowning_incidents")

circuit = ProbabilisticCircuit()
root = SumUnit(probabilistic_circuit=circuit)
for ice_cream_range, drowning_range in [((0, 1), (0, 1)), ((9, 10), (9, 10))]:
    branch = ProductUnit(probabilistic_circuit=circuit)
    branch.add_subcircuit(leaf(
        UniformDistribution(variable=ice_cream_sales, interval=closed(*ice_cream_range).simple_sets[0]),
        circuit,
    ))
    branch.add_subcircuit(leaf(
        UniformDistribution(variable=drowning_incidents, interval=closed(*drowning_range).simple_sets[0]),
        circuit,
    ))
    root.add_subcircuit(branch, math.log(0.5))

causal_circuit = CausalCircuit.from_probabilistic_circuit(
    circuit,
    MarginalDeterminismTreeNode.from_causal_graph([ice_cream_sales], [drowning_incidents]),
    [ice_cream_sales],
    [drowning_incidents],
)

high_ice_cream = SimpleEvent.from_data({ice_cream_sales: closed(9, 10).simple_sets[0]}).as_composite_set()
high_drowning = SimpleEvent.from_data({drowning_incidents: closed(9, 10).simple_sets[0]}).as_composite_set()

# Conditioning: observe high ice cream sales, ask about drowning incidents.
conditioned, _ = circuit.truncated(high_ice_cream.fill_missing_variables_pure(circuit.variables))
conditioned.probability(high_drowning.fill_missing_variables_pure(conditioned.variables))
# 1.0 -- conditioning picks up the confound: high sales imply the warm season, which
# implies drownings.

# Intervention: set ice cream sales high without touching the season, ask the same question.
interventional = causal_circuit.backdoor_adjustment(ice_cream_sales, drowning_incidents)
interventional.probability(high_drowning.fill_missing_variables_pure(interventional.variables))
# 0.5 -- unchanged from the unconditioned baseline: selling more ice cream does not
# cause drownings.
```

Conditioning and intervention give different answers on the exact same model -- that is
the distinction `cause()` and `causes_effect()` exist to make queryable.

---

## Declaring a causal query

`cause()` and `causes_effect()` are always used together: `cause()` marks the field to
search an intervention over, `causes_effect()` declares the condition that intervention
should explain.

```python
from krrood.entity_query_language.factories import an, cause
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

pick = (match := an(Pick)(arm=cause(), success=...)).causes_effect(
    match.variable.success == Status.SUCCESS
)
verbalize_expression(pick)
# 'Generate a Pick and predict its arm and success values where what causes its
#  success to be SUCCESS'
```

### `cause()`

`cause()` takes no arguments: it always means *find the value of this field whose
intervention (`do(arm=value)`) best explains the declared effect*. To pin a known value
instead -- an ordinary conditioning assignment, not a causal one -- use a plain literal
kwarg (`arm=0.3`).

```{important}
`cause()` needs a declared effect to search *for* -- see `causes_effect()` below.
Using `cause()` with no `causes_effect(...)` condition anywhere in the query raises
{py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`.
Declare exactly one effect per query;
{py:class}`~krrood.parametrization.exceptions.MultipleEffectVariablesNotSupported`
raises otherwise -- there is no multi-effect form of the underlying interventional
computation to route several through. Multiple `cause()` fields are fine: each candidate
is searched independently and the one that best explains the effect becomes the primary
cause -- see
{py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
below to see every candidate's score, not just the primary one.
```

### `causes_effect()`

{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` is sugar for
`.where(...)`: it accepts one equality comparator (`attribute == value`) or several
combined with `and_`, declaring exactly one effect variable per query. It filters
results **identically** to an ordinary `.where()` under every backend, including
selective ones, so `causes_effect(...)` never changes what a query would otherwise
select -- only `ProbabilisticBackend` additionally reads it, to know which variable to
compute `P(effect | do(cause))` over.

---

## Backend behaviour

### `ProbabilisticBackend`

Requires a model registry that resolves a
{py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
for the queried class -- {py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`
maps classes directly to pre-built causal circuits:

```python
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.model_registries import CausalCircuitRegistry

backend = ProbabilisticBackend(
    model_registry=CausalCircuitRegistry({Pick: pick_causal_circuit}),
)
results = list(pick.evaluate(backend=backend))
assert all(r.success == Status.SUCCESS for r in results)
```

If the registry resolves anything other than a `CausalCircuit` for a query containing
`cause()`, the backend raises
{py:class}`~krrood.parametrization.exceptions.DoRequiresCausalCircuitModel` --
`cause()` needs a registered causal graph to know what to cut when intervening; there
is no fallback to plain conditioning. Confounder adjustment is configured on however
the registered `CausalCircuit` itself was built, not from the query.

#### Ranking every candidate

`evaluate()` only ever narrows to the single best-scoring `cause()` candidate. When more
than one field is plausibly responsible -- a pick failure that could trace back to
either `arm` or `grip` -- and how the candidates compare matters, not just which one
wins,
{py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
returns every candidate's score instead:

```python
match = an(Pick)(arm=cause(), grip=cause(), success=...)
match.causes_effect(match.variable.success == Status.SUCCESS)

ranking = backend.rank_causes(match)
for scored in ranking:
    print(scored.cause_variable.name, scored.effect_probability_given_region)
# arm   0.94
# grip  0.61
```

Every candidate is still searched independently, exactly as `evaluate()` already does
internally -- `rank_causes` runs the same per-candidate search, it just keeps every
result instead of discarding all but the best. It is still not a joint intervention:
this ranks how well each candidate explains the effect on its own, not how well
intervening on several of them together would.

### Selective backends and `EntityQueryLanguageGenerativeBackend`

Neither has a notion of a causal graph. Rather than failing the whole query, they
**warn** (via `krrood.logger`) and treat `cause()` exactly like an ordinary
unspecified field (`...`) -- a selective backend naturally selects nothing (nothing
equals `cause()`'s wrapped `Ellipsis`), and the generative backend enumerates it if the
field is an enum, or raises the same
{py:class}`~krrood.entity_query_language.exceptions.UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration`
a bare `...` on a non-enum field already raises. Pass
`raise_on_unresolvable_cause=True` to a backend's constructor to fail loudly instead --
useful in tests that want to catch accidental `cause()` misuse against a non-causal
backend.

---

## API Reference

- {py:func}`~krrood.entity_query_language.factories.cause`
- {py:class}`~krrood.entity_query_language.core.causal.Cause`
- {py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect`
- {py:class}`~krrood.entity_query_language.core.causal.CausesEffect`
- {py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`
- {py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
- {py:class}`~krrood.parametrization.exceptions.DoRequiresCausalCircuitModel`
- {py:class}`~krrood.parametrization.exceptions.MultipleEffectVariablesNotSupported`
- {py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`
- {py:meth}`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes`
- {py:class}`~krrood.entity_query_language.exceptions.NoCauseVariablesForRanking`
