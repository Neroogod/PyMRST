"""MRST ``ad-core/examples/book-ii/utils/TutorialNumber.m``.

A state function that reads one field of the state, saying so.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.state_function import STATE, StateFunction


class TutorialNumber(StateFunction):
    """MRST `TutorialNumber`: the state field `n`."""

    def __init__(self, n: str):
        super().__init__()
        self.stateField = n
        self.depends_on(n, STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        print(f"Retrieving {self.stateField} from state.")
        return state[self.stateField]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["TutorialNumber"]


if __name__ == "__main__":
    import contextlib
    import io

    from pymrst.modules.ad_core.statefunctions.state_function import State

    def test_reads_the_field_and_says_so():
        fn = TutorialNumber("a")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert fn.evaluate_on_domain(None, State(a=3)) == 3
        assert out.getvalue() == "Retrieving a from state.\n"
        assert [(d.name, d.grouping) for d in fn.dependencies] == [("a", STATE)]
        print("ok test_reads_the_field_and_says_so")

    test_reads_the_field_and_says_so()
    print("\nAll tests passed!")
