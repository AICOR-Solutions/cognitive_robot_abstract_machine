"""
:meth:`MappedVariable.apply_mapping_on_external_root` follows a chain from a value
outside query evaluation, which is how features are read off an instance.

These tests pin what it does when a step along the way maps one value to several.
"""

import pytest

from krrood.entity_query_language.exceptions import MultipleValuesAlongAccessPath
from krrood.entity_query_language.factories import flat_variable, variable

from ...dataset.semantic_world_like_classes import Cabinet

# %% following a chain of one-to-one mappings


def test_chain_of_attributes_reaches_its_value(handles_and_containers_world):
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = cabinets[0]
    chain = variable(Cabinet, domain=cabinets).container.name

    assert chain.apply_mapping_on_external_root(cabinet) == cabinet.container.name


# %% a step that maps one value to several


def test_chain_through_a_flattened_attribute_has_no_single_value(
    handles_and_containers_world,
):
    """
    Flattening a collection leaves the rest of the chain with an element per item rather
    than one value, which the walk reports instead of silently following the first.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    chain = flat_variable(variable(Cabinet, domain=cabinets).drawers).handle.name

    with pytest.raises(MultipleValuesAlongAccessPath):
        chain.apply_mapping_on_external_root(cabinet)


def test_chain_through_a_single_element_collection_reaches_its_value(
    handles_and_containers_world,
):
    """
    A flattening that yields one element leaves the chain with one value, so it is
    followed like any other step.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) == 1)
    chain = flat_variable(variable(Cabinet, domain=cabinets).drawers).handle.name

    assert (
        chain.apply_mapping_on_external_root(cabinet) == cabinet.drawers[0].handle.name
    )
