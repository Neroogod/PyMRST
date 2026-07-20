"""MRST-shaped incompressible-fluid helpers.

This mirrors `modules/incomp/fluid/incompressible/` while delegating to the
validated PyMRST implementation in `pymrst.modules.incomp.fluid`.
"""

from pymrst.modules.incomp.fluid import Fluid, init_simple_fluid, init_single_fluid

init_corey_fluid = init_simple_fluid
init_simple_fluid_pc = init_simple_fluid
init_simple_fluid_jfunc = init_simple_fluid

__all__ = [
    "Fluid",
    "init_corey_fluid",
    "init_simple_fluid",
    "init_simple_fluid_jfunc",
    "init_simple_fluid_pc",
    "init_single_fluid",
]
