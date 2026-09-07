---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter:causal-reasoning)=
# Causal Reasoning

This chapter summarizes the causal-inference literature this package builds on, and shows
how `probabilistic_model.probabilistic_circuit.causal` and
`probabilistic_model.probabilistic_circuit.relational` implement it. It assumes the
probabilistic-circuit background from {ref}`chapter:queries` and the definitions in
{cite}`choi2020probabilistic`.

## Interventions and the backdoor-adjustment formula

A causal query asks what happens to one variable when another is *forced* to a value,
rather than merely *observed* to have it. Pearl's do-operator {cite}`pearl2009causality`
writes this as $P(Y \mid do(X = x))$: the distribution of effect $Y$ once cause $X$ is set
to $x$ by intervention, cutting $X$ off from whatever normally determines it. This differs
from conditioning, $P(Y \mid X = x)$, whenever $X$ and $Y$ share a common cause: observing
$X = x$ also tells you something about that common cause, which then leaks into what you
infer about $Y$; intervening on $X$ does not.

````{prf:definition} Backdoor Criterion
:label: def-backdoor-criterion

A set of variables $Z$ satisfies the backdoor criterion relative to an ordered pair
$(X, Y)$ if:

1. no node in $Z$ is a descendant of $X$, and
2. $Z$ blocks every path between $X$ and $Y$ that contains an arrow into $X$.

If $Z$ satisfies the backdoor criterion, the causal effect of $X$ on $Y$ is identifiable
from observational data alone, and is given by

$$P(Y \mid do(X = x)) = \sum_z P(Y \mid X = x, Z = z) \, P(Z = z).$$
````

With an empty adjustment set (no confounding to correct for -- the case for independently
randomized training data), this collapses to $P(Y \mid do(X = x)) = P(Y \mid X = x)$: the
interventional and observational distributions coincide.

## Tractable causal inference on probabilistic circuits

Evaluating the backdoor formula requires computing $P(Y \mid X = x, Z = z)$ for every value
$z$ in the adjustment set's domain, and summing. On an arbitrary joint distribution this can
be as expensive as the domain of $Z$ is large. {cite}`wang2023compositional` shows that on a
*structured-decomposable* probabilistic circuit -- one where every product unit's children
partition the scope along a shared, recursively fixed variable tree (a vtree) -- backdoor
adjustment is tractable in the size of the circuit whenever the circuit satisfies one
additional structural property for the relevant variables: **support determinism**.

````{prf:definition} Support Determinism
:label: def-support-determinism

A sum unit is support-deterministic over a variable set $V$ if, for every pair of its
children, the children's marginal supports on $V$ are disjoint: no assignment to $V$ has
positive probability under more than one child.

A circuit is support-deterministic over $V$ if every sum unit that splits on any variable
in $V$ is support-deterministic over $V$.
````

Support determinism over the cause and adjustment variables means each sum unit's children
already partition the world by the value those variables take -- so answering "what is
$P(Y \mid X = x, Z = z)$ under this branch" never requires mixing across branches, and the
backdoor sum becomes a single weighted pass over the circuit rather than one circuit
evaluation per $(x, z)$ pair. {cite}`wang2023compositional` formalizes the circuit family
that guarantees this (*md-vtrees*, a generalization of probabilistic sentential decision
diagrams) and derives the first polytime algorithm for backdoor adjustment on such circuits.
This package does not require an md-vtree-typed circuit outright; instead it *verifies* the
support-determinism property directly on whatever structured-decomposable circuit it is
given, which is the property the polytime algorithm actually depends on.

### Implementation: `CausalCircuit`

`probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit` wraps a
fitted `ProbabilisticCircuit` together with:

- `causal_variables` / `effect_variables` -- the registered $X$ and $Y$ of
  {prf:ref}`def-backdoor-criterion`.
- `marginal_determinism_tree`, a `MarginalDeterminismTreeNode` -- a binary tree over the
  cause variables, built by `from_causal_graph` recursively bisecting the (priority-ordered)
  cause list. Each node records the variables in its subtree and a `query_set` (by default
  its own highest-priority variable); `all_query_sets()` flattens every node's `query_set`
  in pre-order, and their union is exactly the set of cause variables that need to satisfy
  {prf:ref}`def-support-determinism`. `find_node_for_variable` additionally lets
  `diagnose_failure` walk the tree to recover the priority a variable was registered with.
- `verify_support_determinism()`, which takes that union of query variables and checks
  {prf:ref}`def-support-determinism` for each of them against every sum unit in the circuit,
  raising `SupportDeterminismVerificationResult` with every violation found if any check
  fails. This is a validity *check*, not a construction step: it does not repair a circuit
  that fails it.
- `backdoor_adjustment(cause_variable, effect_variable, adjustment_variables)`, which
  implements the formula in {prf:ref}`def-backdoor-criterion` directly: with no adjustment
  set it truncates the circuit to each cause region and reads off the effect marginal; with
  one, it additionally weights each cause-region branch by $P(Z = z)$ and sums, using the
  region structure `verify_support_determinism` already confirmed is disjoint.

## Relational causal reasoning

The circuits above assume a fixed variable set, decided once and for all when the circuit
is fitted. That does not hold for a scene with an unknown number of objects: how many
objects there are is itself a fact you might want to intervene on or condition on, but no
single flat circuit has "object count" as one of its variables ahead of time -- the variable
set depends on which scene is being described.

`RelationalProbabilisticCircuit` ({cite}`nath2015rspn`, "Relational Sum-Product Networks",
extended here onto circuits with the query system `probabilistic_model.probabilistic_circuit.relational`
bridges into `krrood`) resolves this by *grounding*: given a query describing one concrete
scene, it stamps out one instance of a fitted template circuit per object the query
mentions, and combines the instances into a single circuit over that scene's actual
variables. Where the query leaves some *aggregation statistic* undetermined -- e.g. it asks
about "a room with some chairs" without saying how many -- grounding must decide how many
child instances to stamp out and how to combine them, which requires resolving the
statistic to a concrete value or distribution over values (the "undetermined latent") before
grounding the exchangeable part.

Prior to this package, that undetermined latent was resolved by marginalizing it out
immediately: grounding blended over every possible object count and discarded the
count itself, leaving no variable behind to register as a cause or an effect. That is fine
for prediction (the blended answer is what you want, if you don't care why), but it removes
exactly the variable a causal query about it would need. `GroundingMode` gives grounding two
alternative representations that *retain* the latent as a variable of the grounded circuit
instead:

- `GroundingMode.SAMPLED`: draw `monte_carlo_sample_count` values of the latent from the
  conditioned circuit, ground one child instance per distinct sampled value, and retain each
  sampled value as an additional point-valued (Dirac) leaf mounted alongside its instance.
- `GroundingMode.EXACT`: instead of sampling, enumerate the latent's own exact partition as
  already fitted by the class circuit's `JointProbabilityTree` {cite}`nyga2023joint` -- the
  branches of `circuit.marginal(undetermined_latents)`'s root sum unit -- and ground one
  instance per branch, retaining that branch's own region rather than a single point.

### Why retaining the latent this way preserves support determinism

This is the structural argument `ExchangeablePartGrounder.attach_monte_carlo_mixture` and
`attach_exact_partition_mixture` rely on, and the reason grounding can hand a causal circuit
a retained latent without breaking {prf:ref}`def-support-determinism`:

````{prf:theorem} Grounded latent retention is support-deterministic
:label: thm-retained-latent-disjoint

Let $L$ be an undetermined latent and $\{a_1, \dots, a_k\}$ a finite set of values (or
regions) of $L$ such that the $a_i$ are pairwise disjoint as subsets of $L$'s domain. Let
$C_1, \dots, C_k$ be circuits, each grounded by conditioning on a determined statistics and
one $a_i$, and mounted so that $C_i$ additionally carries $L \in a_i$ as part of its scope
(as a Dirac leaf at $a_i$, for a sampled point; as the branch region itself, for an exact
partition branch). Then the sum unit $S = \sum_i w_i \cdot C_i$ is support-deterministic over
$L$.
````

*Sketch.* Support determinism over $L$ requires every pair of $S$'s children to have
disjoint marginal support on $L$. $C_i$'s marginal support on $L$ is exactly $a_i$ -- for the
Dirac case because a point mass has no support outside its point, and for the exact-partition
case because $a_i$ is one branch of `circuit.marginal(undetermined_latents)`'s root sum unit.
Since the $a_i$ are pairwise disjoint by hypothesis, so are the $C_i$'s marginal supports on
$L$. $\blacksquare$

The hypothesis -- that the $a_i$ are actually pairwise disjoint -- is where the two modes
differ, and it is not free for the exact-partition case:

- **`SAMPLED`** deduplicates its Monte-Carlo draws before grounding
  (`_sample_undetermined_latents`), so the $a_i$ are distinct points -- trivially pairwise
  disjoint regardless of what the underlying distribution over $L$ looks like. This is why
  `SAMPLED` always succeeds: disjointness of single points needs no assumption about the
  fitted model.
- **`EXACT`** instead needs the fitted `JointProbabilityTree` {cite}`nyga2023joint` to have
  *actually split* on $L$: a JPT is only guaranteed deterministic
  ({cite}`choi2020probabilistic`) over variables it chose to split on, and does not retain
  which those were after fitting. $L$ being one of the tree's *targets* rather than a
  *feature* it split on can leave `circuit.marginal(undetermined_latents)`'s root as a
  single, undifferentiated branch (trivially "one region", not a genuine partition) or, in
  principle, with overlapping branches -- either of which would violate the theorem's
  hypothesis while still superficially looking like "a sum unit over $L$".
  `_undetermined_latents_partition_disjointly` checks pairwise disjointness on the
  marginalized circuit directly, rather than assuming it from how the tree was fit. If the
  check fails, the precondition of {prf:ref}`thm-retained-latent-disjoint` does not hold, and
  `attach_exact_partition_mixture` raises `UndeterminedLatentsNotPartitionedError` rather than
  silently grounding every branch from the same representative point (which would discard
  $L$'s correlation with the rest of the circuit while still looking, structurally, like a
  valid disjoint partition). Grounding catches exactly this and falls back to `SAMPLED`,
  logging a warning that the template needs refitting with $L$ as a feature for `EXACT` to
  apply.

The theorem only gives *structural* determinism (the property `verify_support_determinism`
checks); it says nothing about whether the mixture weights $w_i$ are the right ones. Getting
those right is a separate, per-mounting-node computation
(`_node_local_latent_log_likelihoods` for `SAMPLED`, `_node_local_branch_log_probabilities`
for `EXACT`): two product nodes that mount the same exchangeable relation can correlate $L$
differently with the rest of the scene, so each node's weights over the same global set of
branches are computed from that node's own local marginal, not shared globally.

### `RelationalCausalCircuit`

`probabilistic_model.probabilistic_circuit.relational.causal.RelationalCausalCircuit` is the
factory that composes the two halves: `ground()` grounds a `RelationalProbabilisticCircuit`
for a query under a chosen `GroundingMode`, then `from_grounded_circuit()` resolves the
requested cause/effect/adjustment variables (by `Variable` or by a dotted access-path string,
via `resolve_variable`), builds the `MarginalDeterminismTreeNode`, and returns a verified
`CausalCircuit` -- the same object the flat, non-relational path produces, so
`backdoor_adjustment` and `diagnose_failure` work identically regardless of whether the
circuit came from a fixed template or was grounded per-query.

Registering more than one relational adjustment variable at once is the one place this can
get expensive: `CausalCircuit`'s adjustment-region extraction takes a Cartesian product
across the adjustment variables' leaf regions, and a `GroundingMode.EXACT` variable's region
count can grow with the training data (unlike `SAMPLED`, capped by
`monte_carlo_sample_count`). `RelationalCausalCircuit.adjustment_region_count_warning_threshold`
warns when that product is large, as a diagnostic against discovering the cost only at query
time -- not a guarantee, since it is read off however the adjustment variables actually got
grounded.

## Literature position

| Work | Relational | Causal | Substrate | Tractability mechanism |
|---|---|---|---|---|
| {cite}`nath2015rspn` (RSPN) | Yes | No | Sum-product network | Templated grounding |
| {cite}`wang2023compositional` (md-vtrees) | No | Yes | Structured-decomposable circuit | Support determinism $\to$ polytime backdoor adjustment |
| {cite}`luttermann2024lifted` (Lifted Causal Inference) | Yes | Yes | Parametric factor graphs | Domain-lifted: polynomial in domain size for a bounded-logvar fragment, non-grounding queries only |
| This package | Yes | Yes | RSPN grounded onto a structured-decomposable circuit | Cause/effect registration: free (additive to circuit size). Adjustment registration: guarded against Cartesian-product blowup |

No prior work was found combining md-vtree-style circuit causal inference with relational
grounding directly; {cite}`luttermann2024lifted` is the closest bridge, but on a different
substrate (parametric factor graphs, not circuits) with its own, differently scoped
tractability class (lifted over object symmetry, rather than exact per-query grounding).
This package stays fully grounded per query rather than lifted: it answers exact causal
queries at RSPN's existing per-query grounding granularity, not population-scale inference
over many symmetric objects at once.

## Open question: does the interventional distribution match reality?

Everything above establishes that grounding-then-registering a relational cause is
*structurally* sound -- it produces a valid, support-deterministic `CausalCircuit`, so
`backdoor_adjustment` runs and returns something. It does not establish that the resulting
interventional distribution is numerically close to the true causal effect in a physical
system the model was fit on, nor how the two `GroundingMode`s compare in accuracy or latency
at realistic leaf counts. That is an empirical question this chapter deliberately leaves
open: the natural check is grounding on simulated data with a known ground truth (e.g. a
box-stacking simulation where the true stack-success rate at each box count is directly
measurable) and comparing it against `backdoor_adjustment`'s prediction, since only
simulation offers that ground truth directly.

```{bibliography}
```
