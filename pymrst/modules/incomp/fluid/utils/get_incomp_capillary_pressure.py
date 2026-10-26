"""MRST ``incomp/fluid/utils/getIncompCapillaryPressure.m``: the capillary
pressure the incompressible solvers use - `fluid.pc(state)` for an
incompressible fluid, `fluid.pcOW(sW)` for an AD fluid - or empty when the
fluid has none."""

from typing import Any, Optional

import numpy as np


def get_incomp_capillary_pressure(state: Any, fluid: Any) -> Optional[np.ndarray]:
    """MRST `pc = getIncompCapillaryPressure(state, fluid)`; `None` for MRST's `[]`."""
    from pymrst.utils.adi import value
    if hasattr(fluid, "properties"):
        pc_fn = getattr(fluid, "pc", None)
        return None if pc_fn is None else np.asarray(pc_fn(state), dtype=float)
    pc_fn = getattr(fluid, "pcOW", None)
    if pc_fn is None:
        return None
    s = state["s"] if isinstance(state, dict) else state.s
    return np.asarray(value(pc_fn(np.atleast_2d(np.asarray(s, dtype=float))[:, 0])), dtype=float)


getIncompCapillaryPressure = get_incomp_capillary_pressure
__all__ = ["get_incomp_capillary_pressure", "getIncompCapillaryPressure"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.incomp.fluid import init_simple_fluid

    def test_no_capillary_pressure():
        state = SimpleNamespace(s=np.ones((2, 2)))
        assert get_incomp_capillary_pressure(state, init_simple_fluid([1, 1], [1, 1])) is None
        print("ok test_no_capillary_pressure")

    def test_ad_fluid_pcow():
        fluid = SimpleNamespace(pcOW=lambda s: 2*s)
        pc = get_incomp_capillary_pressure({"s": np.array([[0.25, 0.75]])}, fluid)
        assert np.allclose(pc, [0.5])
        print("ok test_ad_fluid_pcow")

    test_no_capillary_pressure()
    test_ad_fluid_pcow()
    print("\nAll tests passed!")
