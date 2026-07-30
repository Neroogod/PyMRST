"""Book II conceptual state-function tutorial components."""

from __future__ import annotations

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


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


__all__ = ["TutorialNumber", "TutorialAdd", "TutorialProduct"]
