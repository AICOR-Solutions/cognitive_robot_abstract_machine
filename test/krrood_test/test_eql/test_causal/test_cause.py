from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.causal import Cause
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import an, cause

# %% construction


def test_cause_wraps_ellipsis():
    assert cause()._value_ is Ellipsis


def test_cause_is_a_literal():
    assert isinstance(cause(), Literal)


def test_cause_rejects_a_positional_argument():
    with pytest.raises(TypeError):
        cause(0.3)


def test_two_cause_instances_are_distinct_objects():
    # Each `cause()` call is a fresh marker, since `Cause._type_` is backfilled
    # per-attribute-match in place (see `AttributeMatch.assigned_variable`); sharing one
    # instance across two attributes would let the second backfill silently overwrite it.
    assert cause() is not cause()


# %% flowing through Match unmodified (no new Match/AttributeMatch branch is needed)


@dataclass
class Pick:
    arm: float
    grasped: bool


def test_cause_flows_through_match_as_the_assigned_variable():
    match = an(Pick)(arm=cause(), grasped=True)
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Pick.arm"
    ]
    assert isinstance(attribute_match.assigned_variable, Cause)


def test_cause_backfills_its_type_from_the_attribute_it_is_assigned_to():
    match = an(Pick)(arm=cause(), grasped=True)
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Pick.arm"
    ]
    assert attribute_match.assigned_variable._type_ is float


def test_match_marks_a_cause_attribute_as_present():
    match = an(Pick)(arm=cause(), grasped=True)
    assert match.has_cause_attributes is True


def test_match_without_cause_reports_no_cause_attributes():
    match = an(Pick)(arm=0.3, grasped=True)
    assert match.has_cause_attributes is False
