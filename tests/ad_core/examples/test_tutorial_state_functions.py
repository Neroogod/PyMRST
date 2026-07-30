import numpy as np

from pymrst.modules.ad_core.examples import (
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
    build_tutorial_state_function_group,
    run_conceptual_ad,
)
from pymrst.modules.ad_core.statefunctions import State, StateFunctionGrouping
from pymrst.utils.adi import init_variables_adi, value


def _group():
    return build_tutorial_state_function_group()


def test_conceptual_graph_numeric_and_cache():
    group = _group()
    state = State(
        a=np.array([2.0, 3.0]), b=np.array([4.0, 5.0]),
        x=np.array([6.0, 7.0]), y=np.array([8.0, 9.0]),
    )
    result = group.get(None, state, "G")
    assert np.array_equal(result, state.a * state.b + state.x * state.y)
    assert set(state.containers["tutorial"]) == {"A", "B", "X", "Y",
                                                  "AB", "XY", "G"}
    assert group.get(None, state, "G") is result


def test_conceptual_graph_propagates_ad_jacobians():
    a, b, x, y = init_variables_adi(
        np.array([2.0]), np.array([3.0]), np.array([4.0]), np.array([5.0])
    )
    result = _group().get(None, State(a=a, b=b, x=x, y=y), "G")
    assert np.allclose(value(result), 26.0)
    derivatives = [jac.toarray().item() for jac in result.jac]
    assert np.allclose(derivatives, [3.0, 2.0, 5.0, 4.0])


def test_complete_conceptual_example_matches_backends_and_derivatives():
    result = run_conceptual_ad(
        a=[2.0, 3.0], b=[4.0, 5.0], x=[6.0, 7.0], y=[8.0, 9.0]
    )

    expected = np.array([56.0, 78.0])
    assert np.array_equal(result.numeric_value, expected)
    assert np.array_equal(result.sparse_value, expected)
    assert np.array_equal(result.diagonal_value, expected)
    for sparse, diagonal in zip(
        result.sparse_derivatives, result.diagonal_derivatives
    ):
        assert np.array_equal(sparse, diagonal)
    assert np.array_equal(np.diag(result.sparse_derivatives[0]), [4.0, 5.0])
    assert np.array_equal(np.diag(result.sparse_derivatives[1]), [2.0, 3.0])
    assert np.array_equal(np.diag(result.sparse_derivatives[2]), [8.0, 9.0])
    assert np.array_equal(np.diag(result.sparse_derivatives[3]), [6.0, 7.0])
    assert result.evaluation_order[-1] == "G"
    assert set(result.cached_properties) == {"A", "B", "X", "Y", "AB", "XY", "G"}
