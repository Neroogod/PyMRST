import numpy as np

from pymrst.modules.ad_core.examples import (
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
)
from pymrst.modules.ad_core.statefunctions import State, StateFunctionGrouping
from pymrst.utils.adi import init_variables_adi, value


def _group():
    group = StateFunctionGrouping("tutorial")
    group.set_state_function("A", TutorialNumber("a"))
    group.set_state_function("B", TutorialNumber("b"))
    group.set_state_function("X", TutorialNumber("x"))
    group.set_state_function("Y", TutorialNumber("y"))
    group.set_state_function("AB", TutorialProduct("A", "B"))
    group.set_state_function("XY", TutorialProduct("X", "Y"))
    group.set_state_function("G", TutorialAdd("AB", "XY"))
    return group


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
