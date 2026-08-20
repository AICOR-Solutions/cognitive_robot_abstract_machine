"""
A condition may be written with its attribute chain rooted at the query itself
(``query.where(query.size > 1)``) rather than at the variable the query selects.

These tests pin that such a condition filters the row being tested, instead of ranging
over the query's own results.
"""

import pytest

from krrood.entity_query_language.exceptions import AmbiguousQueryAttribute
from krrood.entity_query_language.factories import (
    an,
    entity,
    flat_variable,
    set_of,
    variable,
)

from ...dataset.semantic_world_like_classes import Body, Cabinet, Handle

# %% query-rooted conditions filter


def test_query_rooted_condition_filters(handles_and_containers_world):
    bodies = handles_and_containers_world.bodies
    query = entity(variable(Body, domain=bodies))
    query.where(query.size > 1)

    assert query.tolist() == [body for body in bodies if body.size > 1]


def test_query_rooted_condition_matches_the_variable_rooted_spelling(
    handles_and_containers_world,
):
    bodies = handles_and_containers_world.bodies
    body = variable(Body, domain=bodies)
    variable_rooted = entity(body).where(body.size > 1)

    query_rooted = entity(variable(Body, domain=bodies))
    query_rooted.where(query_rooted.size > 1)

    assert query_rooted.tolist() == variable_rooted.tolist()


def test_query_rooted_condition_does_not_multiply_results(
    handles_and_containers_world,
):
    bodies = handles_and_containers_world.bodies
    query = entity(variable(Body, domain=bodies))
    query.where(query.size >= 1)

    assert query.tolist() == list(bodies)


def test_match_condition_rooted_at_the_lowered_query_filters(
    handles_and_containers_world,
):
    bodies = handles_and_containers_world.bodies
    match = an(Body)().from_(bodies)
    match.where(match.expression.size > 1)

    assert match.tolist() == [body for body in bodies if body.size > 1]


def test_query_rooted_condition_through_a_flattened_attribute_filters(
    handles_and_containers_world,
):
    """
    Every mapping a chain can be built from has to survive re-rooting, including the
    flattening one, whose values are reached by iterating rather than by an operator
    symbolic expressions trace.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    handle_name = cabinets[0].drawers[0].handle.name

    query = entity(variable(Cabinet, domain=cabinets))
    query.where(flat_variable(query.drawers).handle.name == handle_name)

    assert query.tolist() == [
        cabinet
        for cabinet in cabinets
        if any(drawer.handle.name == handle_name for drawer in cabinet.drawers)
    ]


# %% flattenings keep their own identity across re-rooting


def test_query_rooted_condition_keeps_two_flattenings_independent(
    handles_and_containers_world,
):
    """
    Two flattenings of the same attribute are two iteration variables, so they range
    over the elements independently of each other.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    handle_name = cabinet.drawers[0].handle.name
    container_name = cabinet.drawers[1].container.name

    subject = variable(Cabinet, domain=cabinets)
    variable_rooted = entity(subject).where(
        flat_variable(subject.drawers).handle.name == handle_name,
        flat_variable(subject.drawers).container.name == container_name,
    )

    query = entity(variable(Cabinet, domain=cabinets))
    query.where(
        flat_variable(query.drawers).handle.name == handle_name,
        flat_variable(query.drawers).container.name == container_name,
    )

    assert query.tolist() == variable_rooted.tolist()


def test_query_rooted_condition_keeps_one_flattening_shared(
    handles_and_containers_world,
):
    """
    One flattening used by several conditions is one iteration variable, so every
    condition follows the same element.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    excluded_handle_name = cabinet.drawers[0].handle.name
    container_name = cabinet.drawers[0].container.name

    subject = variable(Cabinet, domain=cabinets)
    subject_drawer = flat_variable(subject.drawers)
    variable_rooted = entity(subject).where(
        subject_drawer.handle.name != excluded_handle_name,
        subject_drawer.container.name == container_name,
    )

    query = entity(variable(Cabinet, domain=cabinets))
    query_drawer = flat_variable(query.drawers)
    query.where(
        query_drawer.handle.name != excluded_handle_name,
        query_drawer.container.name == container_name,
    )

    assert query.tolist() == variable_rooted.tolist()


# %% conditions that must keep their uncorrelated meaning


def test_condition_rooted_at_another_query_stays_a_subquery(
    handles_and_containers_world,
):
    bodies = handles_and_containers_world.bodies
    handles = entity(variable(Handle, domain=bodies))
    body = variable(Body, domain=bodies)
    query = entity(body).where(body.name == handles.name)

    assert query.tolist() == [body for body in bodies if isinstance(body, Handle)]


# %% attributes a query cannot give a subject


def test_attribute_of_a_multi_variable_query_is_rejected(
    handles_and_containers_world,
):
    bodies = handles_and_containers_world.bodies
    query = set_of(variable(Body, domain=bodies), variable(Handle, domain=bodies))

    with pytest.raises(AmbiguousQueryAttribute):
        query.where(query.size > 1)
