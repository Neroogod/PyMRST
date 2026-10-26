"""MRST ``ad-core/examples/book-ii/utils/TutorialAdd.m``.

A state function that is the sum of two state functions of the same
grouping, saying so.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class TutorialAdd(StateFunction):
    """MRST `TutorialAdd`: the sum of `left` and `right`."""

    def __init__(self, left: str, right: str):
        super().__init__()
        self.leftNumber, self.rightNumber = left, right
        self.depends_on([left, right])

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        l, r = self.get_evaluated_dependencies(state, self.leftNumber, self.rightNumber)
        print(f"Adding {self.leftNumber} and {self.rightNumber}.")
        return l + r

    evaluateOnDomain = evaluate_on_domain


__all__ = ["TutorialAdd"]


if __name__ == "__main__":
    import contextlib
    import io

    from pymrst.modules.ad_core.examples.book_ii.utils.tutorial_number import TutorialNumber
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_core.statefunctions.state_function_grouping import (
        StateFunctionGrouping,
    )

    def test_combines_its_dependencies_and_says_so():
        group = StateFunctionGrouping("mystorage")
        group.set_state_function("A", TutorialNumber("a"))
        group.set_state_function("B", TutorialNumber("b"))
        group.set_state_function("C", TutorialAdd("A", "B"))
        state = group.init_state_function_container(State(a=3.0, b=5.0))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v = group.get(None, state, "C")
        l, r = 3.0, 5.0
        assert v == l + r
        assert out.getvalue().splitlines()[-1] == "Adding A and B."
        print("ok test_combines_its_dependencies_and_says_so")

    test_combines_its_dependencies_and_says_so()
    print("\nAll tests passed!")
