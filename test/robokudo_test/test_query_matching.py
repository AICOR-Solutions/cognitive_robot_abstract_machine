"""
Matching the object hypotheses of a pipeline against the object a query asks for.
"""

from __future__ import annotations

# %% Imports
import pytest

from robokudo.types.annotation import Classification, SemanticColor
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.query import ObjectHypothesisQueryMatcher

designator_messages = pytest.importorskip("robokudo_msgs.msg")

# %% hypotheses


def hypothesis_in(color: str, classname: str | None = None) -> ObjectHypothesis:
    """
    :param color: The semantic color annotated on the hypothesis.
    :param classname: The class it is classified as, None for an unclassified one.
    :return: The hypothesis.
    """
    hypothesis = ObjectHypothesis()
    color_annotation = SemanticColor()
    color_annotation.color = color
    hypothesis.annotations.append(color_annotation)
    if classname is not None:
        classification = Classification()
        classification.classname = classname
        hypothesis.annotations.append(classification)
    return hypothesis


# %% matching types of unclassified hypotheses


def test_an_unclassified_hypothesis_does_not_match_a_type_by_default():
    requested = designator_messages.ObjectDesignator(type="milk")

    assert not ObjectHypothesisQueryMatcher().matches(hypothesis_in("red"), requested)


def test_an_unclassified_hypothesis_may_match_every_type():
    requested = designator_messages.ObjectDesignator(type="milk", color=["red"])
    matcher = ObjectHypothesisQueryMatcher(unclassified_objects_match_every_type=True)

    assert matcher.matches(hypothesis_in("red"), requested)


def test_an_unclassified_hypothesis_matching_every_type_is_still_matched_by_color():
    requested = designator_messages.ObjectDesignator(type="milk", color=["red"])
    matcher = ObjectHypothesisQueryMatcher(unclassified_objects_match_every_type=True)

    assert not matcher.matches(hypothesis_in("blue"), requested)


def test_a_classified_hypothesis_is_still_matched_by_its_type():
    requested = designator_messages.ObjectDesignator(type="milk")
    matcher = ObjectHypothesisQueryMatcher(unclassified_objects_match_every_type=True)

    assert not matcher.matches(hypothesis_in("red", classname="cup"), requested)
