from dataclasses import dataclass

import pytest

from krrood.entity_query_language.backends import EntityQueryLanguageBackend
from krrood.entity_query_language.core.causal import CausesEffect
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresLiteralComparator,
)
from krrood.entity_query_language.factories import an, and_, not_
from krrood.entity_query_language.operators.core_logical_operators import AND


@dataclass
class Pick:
    arm: float
    status: str


# %% construction


def test_causes_effect_accepts_a_literal_comparator():
    arm = an(Pick)(arm=..., status="idle").variable
    CausesEffect(arm.status == "SUCCESS")  # does not raise


def test_causes_effect_accepts_a_conjunction_of_literal_comparators():
    arm = an(Pick)(arm=..., status="idle").variable
    CausesEffect(and_(arm.status == "SUCCESS", arm.arm > 0.0))  # does not raise


def test_causes_effect_rejects_a_comparison_between_two_attributes():
    a = an(Pick)(arm=..., status="idle").variable
    b = an(Pick)(arm=..., status="idle").variable
    with pytest.raises(CausesEffectRequiresLiteralComparator):
        CausesEffect(a.arm > b.arm)


# %% transparent evaluation


def test_causes_effect_selects_the_same_instances_as_a_plain_where():
    matching = Pick(0.3, "SUCCESS")
    non_matching = Pick(0.7, "FAILURE")

    plain = an(Pick)().from_([matching, non_matching])
    plain.where(plain.variable.status == "SUCCESS")

    wrapped = an(Pick)().from_([matching, non_matching])
    wrapped.causes_effect(wrapped.variable.status == "SUCCESS")

    backend = EntityQueryLanguageBackend()
    assert list(plain.evaluate(backend=backend)) == list(
        wrapped.evaluate(backend=backend)
    )


def test_causes_effect_negation_still_selects_the_complement():
    matching = Pick(0.3, "SUCCESS")
    non_matching = Pick(0.7, "FAILURE")

    wrapped = an(Pick)().from_([matching, non_matching])
    wrapped.where(not_(CausesEffect(wrapped.variable.status == "SUCCESS")))

    backend = EntityQueryLanguageBackend()
    assert list(wrapped.evaluate(backend=backend)) == [non_matching]


# %% Match.causes_effect sugar


def test_match_causes_effect_is_sugar_for_where_with_causes_effect():
    match = an(Pick)(arm=..., status=...)
    match.causes_effect(match.variable.status == "SUCCESS")
    [condition] = match._where_conditions_
    assert isinstance(condition, CausesEffect)


def test_match_causes_effect_ands_multiple_conditions():
    match = an(Pick)(arm=..., status=...)
    match.causes_effect(match.variable.status == "SUCCESS", match.variable.arm > 0.0)
    [condition] = match._where_conditions_
    assert isinstance(condition, CausesEffect)
    assert isinstance(condition._child_, AND)
