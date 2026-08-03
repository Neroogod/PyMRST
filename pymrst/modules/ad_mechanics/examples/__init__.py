"""MRST-shaped ad-mechanics example landing modules."""

from pymrst.modules.ad_mechanics.examples.mandel import (
    MandelCase,
    consolidation_coefficient,
    mandel_pressure,
    mandel_roots,
    poro_params,
)
from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
from pymrst.modules.ad_mechanics.examples.run_adjoint_example import (
    run_adjoint_example,
)
from pymrst.modules.ad_mechanics.examples.run_all_2d_cases import (
    run_all_2d_cases,
)
from pymrst.modules.ad_mechanics.examples.run_all_norne_examples import (
    run_all_norne_examples,
)
from pymrst.modules.ad_mechanics.examples.run_norne_example import (
    run_norne_example,
)
from pymrst.modules.ad_mechanics.examples.terzaghi import (
    run_terzaghi,
)

__all__ = [
    "MandelCase",
    "consolidation_coefficient",
    "mandel_pressure",
    "mandel_roots",
    "poro_params",
    "run_2d_case",
    "run_adjoint_example",
    "run_all_2d_cases",
    "run_all_norne_examples",
    "run_norne_example",
    "run_terzaghi",
]
