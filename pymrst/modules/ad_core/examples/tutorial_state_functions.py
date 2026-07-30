"""Book II conceptual state-function tutorial components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_diagonal,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    State,
    StateFunction,
    StateFunctionGrouping,
)
from pymrst.utils.adi import init_variables_adi, value


class TutorialNumber(StateFunction):
    """Leaf node that retrieves one primary variable from the state."""

    def __init__(self, state_field: str):
        super().__init__(label=str(state_field))
        self.stateField = str(state_field)
        self.depends_on(self.stateField, STATE)

    def evaluate_on_domain(self, model, state):
        return state[self.stateField]


class _TutorialBinary(StateFunction):
    def __init__(self, left: str, right: str):
        super().__init__()
        self.leftNumber = str(left)
        self.rightNumber = str(right)
        self.depends_on([self.leftNumber, self.rightNumber])

    def operands(self, state):
        return (
            self.get_evaluated(state, self.leftNumber),
            self.get_evaluated(state, self.rightNumber),
        )


class TutorialAdd(_TutorialBinary):
    """Add two evaluated state-function dependencies."""

    def evaluate_on_domain(self, model, state):
        left, right = self.operands(state)
        return left + right


class TutorialProduct(_TutorialBinary):
    """Element-wise product of two evaluated dependencies."""

    def evaluate_on_domain(self, model, state):
        left, right = self.operands(state)
        return left * right


@dataclass(frozen=True)
class ConceptualADResult:
    """Outputs of the complete conceptual state-function example."""

    numeric_value: np.ndarray
    sparse_value: np.ndarray
    diagonal_value: np.ndarray
    sparse_derivatives: tuple[np.ndarray, ...]
    diagonal_derivatives: tuple[np.ndarray, ...]
    evaluation_order: tuple[str, ...]
    cached_properties: tuple[str, ...]


def build_tutorial_state_function_group(
    storage_name: str = "tutorial",
) -> StateFunctionGrouping:
    """Build the ``G = A*B + X*Y`` dependency graph used by MRST Book II."""

    group = StateFunctionGrouping(storage_name)
    group.set_state_function("A", TutorialNumber("a"))
    group.set_state_function("B", TutorialNumber("b"))
    group.set_state_function("X", TutorialNumber("x"))
    group.set_state_function("Y", TutorialNumber("y"))
    group.set_state_function("AB", TutorialProduct("A", "B"))
    group.set_state_function("XY", TutorialProduct("X", "Y"))
    group.set_state_function("G", TutorialAdd("AB", "XY"))
    return group


def run_conceptual_ad(
    a=(3.0,),
    b=(5.0,),
    x=(7.0,),
    y=(2.0,),
) -> ConceptualADResult:
    """Run MRST ``showConceptualAD``/``exampleStateFunctionsBook`` numerics.

    The plotting-only portion is omitted; the returned dependency order,
    populated cache, values, and Jacobian blocks are the testable outputs.
    """

    inputs = tuple(np.atleast_1d(np.asarray(v, dtype=float)) for v in (a, b, x, y))
    sizes = {array.size for array in inputs}
    if len(sizes) != 1:
        raise ValueError("a, b, x, and y must have equal lengths")
    group = build_tutorial_state_function_group()

    numeric_state = State(zip(("a", "b", "x", "y"), inputs))
    numeric = group.get(None, numeric_state, "G")

    sparse_variables = init_variables_adi(*inputs)
    sparse_state = State(zip(("a", "b", "x", "y"), sparse_variables))
    sparse = group.get(None, sparse_state, "G")

    diagonal_variables = init_variables_diagonal(*inputs)
    diagonal_state = State(zip(("a", "b", "x", "y"), diagonal_variables))
    diagonal = group.get(None, diagonal_state, "G")

    return ConceptualADResult(
        numeric_value=np.asarray(numeric),
        sparse_value=np.asarray(value(sparse)),
        diagonal_value=np.asarray(value(diagonal)),
        sparse_derivatives=tuple(j.toarray() for j in sparse.jac),
        diagonal_derivatives=tuple(j.to_sparse().toarray() for j in diagonal.jac),
        evaluation_order=tuple(group.evaluation_order()),
        cached_properties=tuple(numeric_state.containers[group.struct_name]),
    )


showConceptualAD = run_conceptual_ad
exampleStateFunctionsBook = run_conceptual_ad

__all__ = [
    "TutorialNumber",
    "TutorialAdd",
    "TutorialProduct",
    "ConceptualADResult",
    "build_tutorial_state_function_group",
    "run_conceptual_ad",
    "showConceptualAD",
    "exampleStateFunctionsBook",
]
