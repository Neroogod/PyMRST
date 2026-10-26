"""MRST ``test-suite/utils/processTestCaseInput.m``.

The common input handling of a test-case function: a leading `False` asks
only for the setup's name, description and options (`fullSetup` false);
the remaining options are merged into the function's own `options`, and any
it does not know are kept in `options['extra']`, with a warning, to be passed
on (to the setup function it calls, for instance). Options are matched
case-insensitively, as by MRST's `merge_options`.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

from pymrst.modules.test_suite.utils.pack_test_case_setup import pack_test_case_setup


def process_test_case_input(name: str, options: Dict[str, Any], description: str,
                            *args: Any, **varargin: Any
                            ) -> Tuple[Dict[str, Any], bool, Optional[Any]]:
    """MRST `[options, fullSetup, setup] = processTestCaseInput(...)`."""
    full_setup = True
    if args and isinstance(args[0], bool):
        full_setup = args[0]
    options = dict(options)
    names = {k.lower(): k for k in options}
    extra: Dict[str, Any] = {}
    for key, value in varargin.items():
        if key.lower() in names:
            options[names[key.lower()]] = value
        else:
            extra[key] = value
    options["extra"] = extra
    setup = None if full_setup else pack_test_case_setup(name, description=description,
                                                         options=options)
    if extra:
        plural = "s" if len(extra) > 1 else ""
        listed = "".join(f"\n * {k}" for k in extra)
        warnings.warn(f"Option{plural} {listed} \n not part of default test case options. "
                      f"Storing to options.extra")
    return options, full_setup, setup


processTestCaseInput = process_test_case_input

__all__ = ["processTestCaseInput", "process_test_case_input"]


if __name__ == "__main__":
    def test_known_options_merge_and_the_rest_are_extra():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt, full, setup = process_test_case_input("c", {"layers": []}, "d",
                                                       LAYERS=1, J=[1, 2])
        assert opt["layers"] == 1 and opt["extra"] == {"J": [1, 2]} and full and setup is None
        assert "Storing to options.extra" in str(caught[0].message)
        print("ok test_known_options_merge_and_the_rest_are_extra")

    def test_leading_false_returns_a_light_setup():
        opt, full, setup = process_test_case_input("c", {"layers": []}, "d", False)
        assert not full and setup.name == "c" and setup.options is opt
        print("ok test_leading_false_returns_a_light_setup")

    test_known_options_merge_and_the_rest_are_extra()
    test_leading_false_returns_a_light_setup()
    print("\nAll tests passed!")
