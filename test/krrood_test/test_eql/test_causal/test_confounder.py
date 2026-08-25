from dataclasses import dataclass

import pytest

from krrood.entity_query_language.operators.causal import (
    CONFOUNDER,
    Confounder,
    ConfounderSentinel,
)
from krrood.entity_query_language.factories import a, confounder

# %% construction


def test_confounder_rejects_a_positional_argument():
    with pytest.raises(TypeError):
        confounder(0.3)


def test_confounder_call_returns_the_confounder_sentinel():
    assert confounder() is CONFOUNDER


def test_confounder_sentinel_is_not_itself_a_confounder_instance():
    # CONFOUNDER must stay a distinct type so each attribute it marks gets its own
    # fresh Confounder() during match resolution -- see ConfounderSentinel's
    # docstring for why.
    assert not isinstance(CONFOUNDER, Confounder)
    assert isinstance(CONFOUNDER, ConfounderSentinel)


# %% flowing through Match (converted to a fresh Confounder() per attribute on
# resolution)


@dataclass
class Trial:
    treatment: float
    season: str


def _confounder_attribute_match(match):
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Trial.season"
    ]
    return attribute_match


@pytest.mark.parametrize("mark_season_as_confounder", [confounder(), CONFOUNDER])
def test_confounder_flows_through_match_as_the_assigned_variable(
    mark_season_as_confounder,
):
    match = a(Trial)(treatment=0.3, season=mark_season_as_confounder)
    assert isinstance(_confounder_attribute_match(match).assigned_variable, Confounder)


@pytest.mark.parametrize("mark_season_as_confounder", [confounder(), CONFOUNDER])
def test_confounder_backfills_its_type_from_the_attribute_it_is_assigned_to(
    mark_season_as_confounder,
):
    match = a(Trial)(treatment=0.3, season=mark_season_as_confounder)
    assert _confounder_attribute_match(match).assigned_variable._type_ is str


def test_two_confounder_marked_attributes_resolve_to_distinct_objects():
    # Sharing one Confounder() across two attributes would let the second
    # attribute's type-backfill (see AttributeMatch.assigned_variable) silently
    # overwrite the first's -- each attribute must resolve to its own instance
    # instead.
    match = a(Trial)(treatment=CONFOUNDER, season=CONFOUNDER)
    [treatment_match, season_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path
        in ("Trial.treatment", "Trial.season")
    ]
    assert treatment_match.assigned_variable is not season_match.assigned_variable
