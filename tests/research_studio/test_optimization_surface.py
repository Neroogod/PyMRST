"""M4.4 objectives, normalized constraints, KS and gradient evidence."""

import numpy as np
import pytest

from pymrst.research_studio.optimization import (
    ConstraintDefinition, ObjectiveDefinition, OptimizationProblem,
    evaluate_with_gradient, ks_aggregate,
)


def _objective(x, context):
    return context["price"]*x[0]-0.5*np.dot(x, x)


def _gradient(x, context):
    return _objective(x, context), np.array([context["price"]-x[0], -x[1]])


def _pressure(x, context):
    return np.array([x[0]+1.0, x[0]+2.0, x[1]+1.5])


def test_adjoint_provider_matches_on_demand_finite_difference_and_keeps_evidence():
    problem = OptimizationProblem(
        ObjectiveDefinition("economic value", _objective), ("rate", "bhp"),
        ((-10.0, 10.0), (-10.0, 10.0)),
        constraints=(ConstraintDefinition(
            "pressure", _pressure, scale=2.0, limit=3.0, rho=20.0),),
        metadata={"case": "manufactured"})
    result = evaluate_with_gradient(
        problem, [2.0, -1.0], {"price": 4.0}, gradient_owner=_gradient,
        evidence={"run_id": "run_gradient"})
    np.testing.assert_allclose(result.gradient, [2.0, 1.0])
    assert result.maximum_fd_error < 1e-9
    assert result.evidence["run_id"] == "run_gradient"
    assert result.constraints["pressure"] >= max((np.array([3, 4, .5])-3)/2)


def test_ks_is_conservative_and_gap_obeys_log_n_over_rho():
    values = np.array([-2.0, 0.1, 0.4, 0.3])
    aggregate = ks_aggregate(values, 30.0)
    assert aggregate >= np.max(values)
    assert aggregate-np.max(values) <= np.log(values.size)/30.0
    assert ks_aggregate(values, 100.0) < aggregate


def test_bounds_and_gradient_value_disagreement_are_refused():
    problem = OptimizationProblem(ObjectiveDefinition("o", _objective), ("x", "y"),
                                  ((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="outside"):
        evaluate_with_gradient(problem, [2, 0], {"price": 1})
    with pytest.raises(ValueError, match="objective differs"):
        evaluate_with_gradient(problem, [.5, .5], {"price": 1},
                               gradient_owner=lambda x, c: (99, np.ones(2)))
