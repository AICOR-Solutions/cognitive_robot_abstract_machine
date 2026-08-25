from dataclasses import dataclass

from krrood.entity_query_language.operators.causal import Cause, CauseSentinel
from krrood.entity_query_language.factories import a, cause

# %% construction


def test_cause_sentinel_is_not_itself_a_cause_instance():
    # cause must stay a distinct type so each attribute it marks gets its own fresh
    # Cause() during match resolution -- see CauseSentinel's docstring for why.
    assert not isinstance(cause, Cause)
    assert isinstance(cause, CauseSentinel)


# %% flowing through Match (converted to a fresh Cause() per attribute on resolution)


@dataclass
class Pick:
    arm: float
    grasped: bool


def _cause_attribute_match(match):
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Pick.arm"
    ]
    return attribute_match


def test_cause_flows_through_match_as_the_assigned_variable():
    match = a(Pick)(arm=cause, grasped=True)
    assert isinstance(_cause_attribute_match(match).assigned_variable, Cause)


def test_cause_backfills_its_type_from_the_attribute_it_is_assigned_to():
    match = a(Pick)(arm=cause, grasped=True)
    assert _cause_attribute_match(match).assigned_variable._type_ is float


def test_match_marks_a_cause_attribute_as_present():
    match = a(Pick)(arm=cause, grasped=True)
    assert match.has_cause_attributes is True


def test_match_without_cause_reports_no_cause_attributes():
    match = a(Pick)(arm=0.3, grasped=True)
    assert match.has_cause_attributes is False


def test_two_cause_marked_attributes_resolve_to_distinct_objects():
    # Sharing one Cause() across two attributes would let the second attribute's
    # type-backfill (see AttributeMatch.assigned_variable) silently overwrite the
    # first's -- each attribute must resolve to its own instance instead.
    match = a(Pick)(arm=cause, grasped=cause)
    [arm_match, grasped_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path
        in ("Pick.arm", "Pick.grasped")
    ]
    assert arm_match.assigned_variable is not grasped_match.assigned_variable
