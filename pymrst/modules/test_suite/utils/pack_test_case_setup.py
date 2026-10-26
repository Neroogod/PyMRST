"""MRST ``test-suite/utils/packTestCaseSetup.m``.

The setup a test-case function returns: its `name`, `description`,
`options`, `state0`, `model`, `schedule`, `plotOptions` (name-value pairs)
and `extra`, all empty unless given.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pymrst.modules.ad_core.ad_oo.physical_model import merge_options


def pack_test_case_setup(name: str, **varargin: Any) -> SimpleNamespace:
    """MRST `setup = packTestCaseSetup(name, ...)`."""
    setup = SimpleNamespace(name=name, description=None, options=None, state0=None,
                            model=None, schedule=None, plotOptions=[], extra=None)
    return merge_options(setup, varargin)


packTestCaseSetup = pack_test_case_setup

__all__ = ["packTestCaseSetup", "pack_test_case_setup"]


if __name__ == "__main__":
    def test_fields_default_to_empty_and_can_be_set():
        setup = pack_test_case_setup("case", description="d", plotoptions=["View", [0, 90]])
        assert setup.name == "case" and setup.description == "d"
        assert setup.plotOptions == ["View", [0, 90]] and setup.model is None
        try:
            pack_test_case_setup("case", bogus=1)
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")
        print("ok test_fields_default_to_empty_and_can_be_set")

    test_fields_default_to_empty_and_can_be_set()
    print("\nAll tests passed!")
