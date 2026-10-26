"""MRST ``ad-core/examples/book-ii/exampleStateFunctionsBook.m``.

State functions and their caching: a grouping stored as `state.mystorage`
that computes `G = A*B + X*Y` from the state fields `a`, `b`, `x` and `y`
(`TutorialNumber`, `TutorialProduct`, `TutorialAdd`, each saying what it
does). The grouping is displayed; the first `get` of `G` evaluates every
function, the second only reads the cache; with `XY` already evaluated, `G`
evaluates only the rest. Then the dependencies are checked, and a function
`F = X*Z` is added whose dependency `Z` does not exist, which the check
reports. The output is MRST's, MATLAB's displays of the cache and of `G` and
`ok` included.

MRST also plots the grouping's graph (`plotStateFunctionGroupings`) and
prints it as TikZ (`printStateFunctionGroupingTikz`); PyMRST does not port
those two figure tools.

`exampleStateFunctionsBook` in `tutorial_state_functions.py` is PyMRST's
earlier version, kept for its callers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pymrst.modules.ad_core.examples.book_ii.utils.tutorial_add import TutorialAdd
from pymrst.modules.ad_core.examples.book_ii.utils.tutorial_number import TutorialNumber
from pymrst.modules.ad_core.examples.book_ii.utils.tutorial_product import TutorialProduct
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.statefunctions.state_function_grouping import (
    StateFunctionGrouping,
)


def example_state_functions_book() -> SimpleNamespace:
    """MRST `exampleStateFunctionsBook`."""
    group = tutorial_group()
    group.disp()
    state0 = State(a=3.0, b=5.0, x=7.0, y=2.0)
    state = group.init_state_function_container(State(state0))
    _disp_handle_struct(state.containers["mystorage"])
    print("First call:")
    G = group.get(None, state, "G")
    _disp_handle_struct(state.containers["mystorage"])
    print("Second call:")
    G = group.get(None, state, "G")
    _echo("G", G)
    state = group.init_state_function_container(State(state0))
    group.get(None, state, "XY")
    G = group.get(None, state, "G")
    group.check_dependencies(verbose=True)
    group = group.set_state_function("F", TutorialProduct("X", "Z"))
    ok = group.check_dependencies(nargout=1, verbose=True)
    _echo("ok", ok)
    return SimpleNamespace(group=group, state=state, G=G, ok=ok)


def tutorial_group() -> StateFunctionGrouping:
    """`G = A*B + X*Y`, stored as `state.mystorage`."""
    group = StateFunctionGrouping("mystorage")
    group = group.set_state_function("A", TutorialNumber("a"))
    group = group.set_state_function("B", TutorialNumber("b"))
    group = group.set_state_function("X", TutorialNumber("x"))
    group = group.set_state_function("Y", TutorialNumber("y"))
    group = group.set_state_function("AB", TutorialProduct("A", "B"))
    group = group.set_state_function("XY", TutorialProduct("X", "Y"))
    group = group.set_state_function("G", TutorialAdd("AB", "XY"))
    return group


def _matlab_scalar(v: Any) -> str:
    return "[]" if v is None else f"{float(v):g}"


def _disp_handle_struct(container: dict) -> None:
    """MATLAB's `disp` of the cache, a `HandleStruct`."""
    width = max(len(k) for k in container)
    print("HandleStruct with fields:")
    for key, v in container.items():
        print(f"    {key:>{width}}: {_matlab_scalar(v)}")
    print()


def _echo(name: str, v: Any) -> None:
    """MATLAB's display of `name = v` without a semicolon."""
    if isinstance(v, bool):
        print(f"\n{name} =\n\n  logical\n\n   {int(v)}\n")
    else:
        print(f"\n{name} =\n\n    {_matlab_scalar(v)}\n")


exampleStateFunctionsBook = example_state_functions_book

__all__ = ["exampleStateFunctionsBook", "example_state_functions_book", "tutorial_group"]


if __name__ == "__main__":
    import contextlib
    import io

    def test_values_caching_and_dependency_check():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = example_state_functions_book()
        text = out.getvalue()
        assert result.G == 29.0 and result.ok is False
        first, second = text.split("First call:")[1].split("Second call:")
        assert first.count("Retrieving") == 4 and "Retrieving" not in second.split("G =")[0]
        assert "Unmet internal dependency for F in group StateFunctionGrouping: " \
               "Did not find Z in own group." in text
        print("ok test_values_caching_and_dependency_check")

    test_values_caching_and_dependency_check()
    print("\nAll tests passed!")
