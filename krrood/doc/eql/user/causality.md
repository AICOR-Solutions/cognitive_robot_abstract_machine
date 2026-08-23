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
it; *setting* a value tells you what actually follows from it. Pearl calls the first
"conditioning" (`X=`) and the second "intervention" (`do(X)`) -- whenever two fields
share a hidden common cause, the two questions have different answers, because
conditioning picks up the confounding correlation and intervention cuts it.

{py:func}`~krrood.entity_query_language.factories.cause` and
{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` route a query
through
{py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`'s
backdoor-adjustment machinery instead of plain conditioning, when evaluated with
{py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`.

---

## Declaring a causal query

`cause()` and `causes_effect()` are always used together: `cause()` marks the field to
search an intervention over, `causes_effect()` declares the condition that intervention
should explain.

```python
from krrood.entity_query_language.factories import an, cause

pick = (match := an(Pick)(arm=cause(), success=...)).causes_effect(
    match.variable.success == Status.SUCCESS
)
```

reads as: *"find the arm value whose intervention best causes success"*.

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
cause.
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
