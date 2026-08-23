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

Ordinary underspecified fields (see {doc}`underspecified`) either *condition* the
probabilistic model on a value or leave it *free*, sampling from whatever the model's
own correlations imply. Neither answers Pearl's causal question: "if we *set* this
field to some value, what happens to another field?" ("do(X)") is a different query
from "given that we *observed* this field at some value, what else do we know?"
("X="), whenever the two fields share a hidden common cause -- conditioning picks up
that confounding correlation, an intervention cuts it.

{py:func}`~krrood.entity_query_language.factories.cause` and
{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` route a query
through {py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`'s
backdoor-adjustment machinery instead of plain conditioning, when evaluated with
{py:class}`~krrood.entity_query_language.backends.ProbabilisticBackend`.

---

## `cause()` — search for an intervention

```python
from krrood.entity_query_language.factories import an, cause

query = an(Pick)(arm=cause(), success=...)
```

`cause()` takes no arguments. It always means: *find the value of this field whose
intervention (`do(arm=value)`) best explains the declared effect* -- there is no
pinned-value form (`cause(0.3)` is a `TypeError`); pin a value with a plain literal
kwarg instead (`arm=0.3`), which is an ordinary conditioning assignment, not a causal
one.

```{important}
`cause()` needs a declared effect to search *for* -- see `causes_effect()` below.
Using `cause()` with no `causes_effect(...)` condition anywhere in the query raises
{py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`.
```

## `causes_effect()` — declare the effect

```python
query.causes_effect(query.variable.success == Status.SUCCESS)
```

{py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect` is sugar for
`.where(...)`: it accepts one literal comparator (`attribute == value`, or `>`, `<`,
...) or several combined with `and_`. It marks the wrapped condition as the causal
effect a `cause()` search should optimize -- but it filters results **identically** to
an ordinary `.where()` under every backend, including selective ones, so
`causes_effect(...)` never changes what a query would otherwise select. Only
`ProbabilisticBackend` additionally reads it, to know which variable to compute
`P(effect | do(cause))` over.

```python
pick = an(Pick)(arm=cause(), success=...)
pick.causes_effect(pick.variable.success == Status.SUCCESS)
```

reads as: *"find the arm value whose intervention best causes success"*.

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
is no fallback to plain conditioning.

### Selective backends and `EntityQueryLanguageGenerativeBackend`

Neither has a notion of a causal graph. Rather than failing the whole query, they
**warn** (via `krrood.logger`) and treat `cause()` exactly like an ordinary
unspecified field (`...`) -- a selective backend naturally selects nothing (nothing
equals `cause()`'s wrapped `Ellipsis`), and the generative backend enumerates it if the
field is an enum, or raises the same
{py:class}`~krrood.entity_query_language.exceptions.UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration`
a bare `...` on a non-enum field already raises. Pass `crash_on_unresolvable_cause=True`
to a backend's constructor to fail loudly instead -- useful in tests that want to catch
accidental `cause()` misuse against a non-causal backend.

---

## v1 scope

- **One cause variable, one effect variable per query.** Two `cause()` fields, or a
  `causes_effect(...)` conjunction spanning more than one distinct effect variable,
  raise {py:class}`~krrood.parametrization.exceptions.MultipleCauseOrEffectVariablesNotSupported`
  (joint interventions need a multi-cause-variable `backdoor_adjustment` overload that
  does not exist yet).
- **No query-side adjustment-set specification.** `cause()` always calls
  `backdoor_adjustment` with an empty adjustment set, matching
  `CausalCircuit`'s own documented "use empty adjustment sets for independent
  randomised training data" case. Confounder adjustment is configured on however the
  registered `CausalCircuit` was built, not from the query.
- **No pinned-value intervention** (`do(X=x)` for a specific `x`) -- only the search
  form. Pin a value with an ordinary literal kwarg for plain conditioning instead.

---

## API Reference

- {py:func}`~krrood.entity_query_language.factories.cause`
- {py:class}`~krrood.entity_query_language.core.causal.Cause`
- {py:meth}`~krrood.entity_query_language.query.match.Match.causes_effect`
- {py:class}`~krrood.entity_query_language.core.causal.CausesEffect`
- {py:class}`~krrood.parametrization.model_registries.CausalCircuitRegistry`
- {py:class}`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
- {py:class}`~krrood.parametrization.exceptions.DoRequiresCausalCircuitModel`
- {py:class}`~krrood.parametrization.exceptions.MultipleCauseOrEffectVariablesNotSupported`
- {py:class}`~krrood.entity_query_language.exceptions.NoCausesEffectConditionForCause`
