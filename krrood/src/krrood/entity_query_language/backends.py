import enum
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from types import NoneType
from typing import Iterable, TypeVar

from sqlalchemy.orm import sessionmaker
from typing_extensions import ClassVar, Dict, Optional

from krrood import logger
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.causal import CausalRole, Cause
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateCause,
    NoCausesEffectConditionForCause,
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import entity, set_of, variable
from krrood.entity_query_language.query.match import Match, AttributeMatch
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql

try:
    from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
        CausalCircuit,
    )
    from krrood.parametrization.exceptions import (
        DoRequiresCausalCircuitModel,
        MultipleCauseOrEffectVariablesNotSupported,
    )
    from krrood.parametrization.model_registries import (
        ModelRegistry,
        FullyFactorizedRegistry,
    )
    from krrood.parametrization.parameterizer import (
        UnderspecifiedParameters,
    )
except ImportError as e:
    logger.debug(f"Couldn't import probabilistic model needed classes: {e}")
    CausalCircuit = NoneType
    DoRequiresCausalCircuitModel = NoneType
    MultipleCauseOrEffectVariablesNotSupported = NoneType
    ModelRegistry = NoneType
    FullyFactorizedRegistry = NoneType
    UnderspecifiedParameters = NoneType

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.

    Query backends are objects that answer queries by different means.
    """

    opening_directive: ClassVar[Optional[Directive]] = None
    """
    The opening verb a verbalization uses when this backend evaluates the expression
    (``None`` keeps the query-type default).

    A backend declares its own performative so the verbalization layer never inspects
    concrete backend types.
    """

    crash_on_unresolvable_cause: bool = field(default=False, kw_only=True)
    """
    Whether to raise instead of warning when an expression contains a `Cause`
    (`cause()`) intervention this backend cannot resolve causally.

    Defaults to ``False``: the `Cause` is then treated as an ordinary unspecified field
    (a warning is logged explaining why) rather than failing the query. Set ``True`` to
    fail loudly instead -- for example in tests that want to catch accidental `cause()`
    misuse against a non-causal backend. Read only by :class:`SelectiveBackend` and
    :class:`EntityQueryLanguageGenerativeBackend`; :class:`ProbabilisticBackend` always
    raises when it cannot resolve a causal model, regardless of this flag.
    """

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """

    def _warn_or_raise_on_unresolved_cause_(self, expression: Evaluable) -> None:
        """
        Warn (or, if :attr:`crash_on_unresolvable_cause` is set, raise) when
        *expression* is a :class:`~krrood.entity_query_language.query.match.Match`
        containing a `Cause` this backend has no causal graph to resolve.

        :param expression: The expression about to be evaluated.
        """
        if not (isinstance(expression, Match) and expression.has_cause_attributes):
            return
        if self.crash_on_unresolvable_cause:
            raise BackendCannotEvaluateCause(expression, backend_type=type(self))
        logger.warning(
            f"{expression} contains a cause() intervention, which {type(self).__name__} "
            f"cannot evaluate causally; treating it as an ordinary unspecified field."
        )


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.

    These can take any query as input.
    """

    opening_directive: ClassVar[Optional[Directive]] = Directive.FIND
    """
    Selecting from existing data reads as *"Find …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, Match) and expression.has_ellipsis_attributes:
            raise SelectiveBackendCannotResolveEllipsisMatch(expression)
        self._warn_or_raise_on_unresolved_cause_(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Evaluable) -> Iterable[T]: ...


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.

    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    opening_directive: ClassVar[Optional[Directive]] = Directive.GENERATE
    """
    Generating new elements reads as *"Generate …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def _evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A backend that selects elements in this python process.

    This is just ordinary EQL: each expression evaluates itself natively (queries and matches both select over their domains).
    Constructing new instances is the job of a :class:`GenerativeBackend`.
    """

    def _evaluate(self, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass
class EntityQueryLanguageGenerativeBackend(GenerativeBackend):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match's ``where`` conditions.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        self._warn_or_raise_on_unresolved_cause_(expression)
        variables: Dict[str, Variable] = {}
        for attribute_match in expression.matches_with_variables:
            self._check_attribute_match_is_suitable_for_generation(attribute_match)
            variables[attribute_match.name_from_variable_access_path] = (
                self._convert_attribute_match_to_variable(attribute_match)
            )

        expression.variable._update_domain_(
            self._generate_raw_results(expression, variables)
        )

        filtered_results = entity(expression.variable)._quantify_(
            expression._quantifier_type_
        )
        if expression._where_conditions_:
            filtered_results = filtered_results.where(*expression._where_conditions_)
        yield from filtered_results._evaluate_natively_()

    @staticmethod
    def _check_attribute_match_is_suitable_for_generation(
        attribute_match: AttributeMatch,
    ) -> None:
        """
        Raise if an assignment in the match cannot be used to generate solutions.

        :param attribute_match: The attribute match to check.
        :raises UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration: If a
            non-enum leaf is left fully unspecified (``...`` or ``cause()``), which
            deterministic generation cannot enumerate (use the
            :class:`ProbabilisticBackend` instead).
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            raise UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration(
                attribute_match
            )

    @staticmethod
    def _convert_attribute_match_to_variable(
        attribute_match: AttributeMatch,
    ) -> Selectable:
        """
        Convert an attribute match into a variable to enumerate, handling ellipsis (and,
        identically, ``cause()``) assignments for enum fields and concrete values.

        :param attribute_match: The attribute match to convert.
        :return: A variable (or symbolic expression) representing the attribute match.
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            return variable(
                attribute_match.assigned_variable._type_,
                list(attribute_match.assigned_variable._type_),
            )
        if isinstance(attribute_match.assigned_value, SymbolicExpression):
            return attribute_match.assigned_value
        return variable(
            type(attribute_match.assigned_value),
            [attribute_match.assigned_value],
        )

    def _generate_raw_results(
        self, expression: Match[T], variables: Dict[str, Variable]
    ) -> Iterable[T]:
        """
        Construct instances from the given match and enumerable variables.

        :param expression: The match expression to construct instances from.
        :param variables: The variables to enumerate, keyed by access- path name.
        :return: A generator yielding an instance per variable combination.
        """
        all_combinations = set_of(*variables.values())
        for combination in all_combinations._evaluate_natively_():
            for variable_name, value in zip(variables, combination.values()):
                mapped_variable = expression._get_mapped_variable_by_name(variable_name)
                mapped_variable._value_ = value
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model
    registry.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic
    models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.

    This is only used if the query does not specify a limit.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        model = self.model_registry.get_model(parameters)

        if parameters.search_cause_variables:
            cause_variable, effect_variable = self._resolve_cause_and_effect_variables(
                parameters, expression
            )
            if not isinstance(model, CausalCircuit):
                raise DoRequiresCausalCircuitModel(model)
            # compute the interventional joint P(cause, effect | do(cause)) instead of
            # conditioning on the literal assignments (there are none to condition on: a
            # search cause() variable is registered like a free field, not a value)
            conditioned = model.backdoor_adjustment(cause_variable, effect_variable)
        else:
            # apply conditions from literal assignments to underspecified variables
            conditioned, _ = model.conditional(
                parameters.conditioning_assignments_from_literal_values
            )

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements (this is also where the
        # causes_effect(...) condition -- transparent to the translator -- narrows the
        # interventional joint down to the declared effect)
        if parameters.truncation_assignments_from_where_conditions:
            truncated, _ = conditioned.truncated(
                parameters.truncation_assignments_from_where_conditions
            )
        else:
            truncated = conditioned

        if parameters.search_cause_variables:
            # search over the cause variable's regions for the one with the highest
            # remaining probability now that the joint is narrowed to the effect: the
            # intervention that best explains it
            truncated = self._narrow_to_best_intervention_region(
                model, cause_variable, truncated, expression
            )

        # apply conditions from variable assignments to underspecified variables
        if parameters.truncation_assignments_from_krrood_variables:
            complete_event = parameters.truncation_assignments_from_krrood_variables[0]
            complete_event.fill_missing_variables(parameters.variables.values())
            for event in parameters.truncation_assignments_from_krrood_variables[1:]:
                complete_event = complete_event.intersection_with(event)
            truncated, _ = conditioned.truncated(complete_event, singleton_allowed=True)

            if truncated is None:
                raise NoSolutionFound(expression.expression)

        number_of_samples = expression.expression._limit_ or self.number_of_samples

        # sample and sort by log likelihood
        samples = truncated.sample(number_of_samples)
        log_likelihoods = truncated.log_likelihood(samples)
        samples = samples[log_likelihoods.argsort()[::-1]]

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.construct_instance_from_model_sample(
                truncated.variables, sample
            )
            yield instance

    @staticmethod
    def _resolve_cause_and_effect_variables(
        parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> tuple:
        """
        Resolve the single cause and single effect variable a ``cause()`` search
        optimizes for, per this backend's v1 single-cause/single-effect scope.

        :param parameters: The parameters extracted from *expression*.
        :param expression: The match being evaluated.
        :raises NoCausesEffectConditionForCause: If no ``causes_effect(...)`` condition
            declared an effect.
        :raises MultipleCauseOrEffectVariablesNotSupported: If more than one cause or
            effect variable was found.
        :return: The ``(cause_variable, effect_variable)`` pair.
        """
        if not parameters.effect_variables_from_causes_effect:
            raise NoCausesEffectConditionForCause(expression.expression)
        if len(parameters.search_cause_variables) > 1:
            raise MultipleCauseOrEffectVariablesNotSupported(
                CausalRole.CAUSE, parameters.search_cause_variables
            )
        if len(parameters.effect_variables_from_causes_effect) > 1:
            raise MultipleCauseOrEffectVariablesNotSupported(
                CausalRole.EFFECT, parameters.effect_variables_from_causes_effect
            )
        [cause_variable] = parameters.search_cause_variables
        [effect_variable] = parameters.effect_variables_from_causes_effect
        return cause_variable, effect_variable

    @staticmethod
    def _narrow_to_best_intervention_region(
        model: CausalCircuit,
        cause_variable,
        truncated,
        expression: Match[T],
    ):
        """
        Further truncate *truncated* -- the interventional joint, already truncated to
        the declared effect condition -- to the ``cause_variable`` region with the
        highest remaining probability: the intervention that best explains the effect.

        Ranking candidate regions on the effect-truncated circuit (rather than the raw
        interventional joint) is what makes this a search *for the effect*, not merely
        for the most likely intervention overall: truncation renormalizes by the
        (region-independent) probability of the effect condition, so the relative
        ranking of regions is exactly the ranking by joint intervention-and-effect mass.

        :param model: The causal circuit `truncated` was computed from.
        :param cause_variable: The cause variable to search regions of.
        :param truncated: The interventional joint, truncated to the effect condition.
        :param expression: The match being evaluated, for error reporting.
        :raises NoSolutionFound: If no cause region remains, or the best region has zero
            probability.
        :return: `truncated`, further truncated to the best cause region.
        """
        # `_best_region` is private: `causal_circuit.py` is a stable dependency this glue
        # code does not modify (see doc/eql/user/causality.md), and it is the same
        # region-search `diagnose_failure` already uses internally for `recommended_region` --
        # there is no public equivalent to call instead.
        best_region = model._best_region(cause_variable, truncated)
        if best_region is None:
            raise NoSolutionFound(expression.expression)
        narrowed, _ = truncated.truncated(
            best_region.fill_missing_variables_pure(truncated.variables)
        )
        if narrowed is None:
            raise NoSolutionFound(expression.expression)
        return narrowed
