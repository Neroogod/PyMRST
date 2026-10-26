"""EGG ad-blackoil example layout.

`setup_egg` is re-exported for callers that had it here, but it lives in
`pymrst.modules.test_suite.dataset_functions.egg` - which is where MRST keeps
`setupEGG.m`, and the reason `eggExample.m` opens with
`mrstModule add ... test-suite`.
"""

from pymrst.modules.ad_blackoil.examples.egg.egg_example import (
    EGGResult, eggExample, egg_example, run_egg_example, well_sol_from_state,
)
from pymrst.modules.test_suite.dataset_functions.egg import setup_egg

__all__ = [
    "EGGResult", "eggExample", "egg_example", "run_egg_example", "setup_egg",
    "well_sol_from_state",
]
