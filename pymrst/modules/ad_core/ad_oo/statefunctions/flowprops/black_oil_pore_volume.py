"""MRST ``ad-core/statefunctions/flowprops/BlackOilPoreVolume.m``.

Pore volume with rock compressibility: the static pore volume times the
fluid's `pvMultR(p)`.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.pore_volume import (
    PoreVolume,
)
from pymrst.modules.ad_core.statefunctions.state_function import STATE


class BlackOilPoreVolume(PoreVolume):
    """`Phi(p) = Phi_0 * pvMultR(p)`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, regions)
        self.depends_on(["pressure"], STATE)
        assert getattr(model.fluid, "pvMultR", None) is not None, (
            "pvMultR missing from fluid.")

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        pv = super().evaluate_on_domain(model, state)
        p = model.get_prop(state, "pressure")
        return pv*self.evaluate_fluid(model, "pvMultR", p)

    evaluateOnDomain = evaluate_on_domain


__all__ = ["BlackOilPoreVolume"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_pore_volume_grows_with_pressure():
        model = toy_model("W", nx=2, c_r=1e-9)
        state = toy_state(model, pressure=np.array([0.0, 1e8]))
        pv = BlackOilPoreVolume(model).evaluate_on_domain(model, state)
        np.testing.assert_allclose(pv, 0.3*np.array([1.0, 1.1]))
        print("ok test_pore_volume_grows_with_pressure")

    def test_a_fluid_without_rock_compressibility_is_refused():
        try:
            BlackOilPoreVolume(toy_model("W", nx=2))
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_a_fluid_without_rock_compressibility_is_refused")

    test_pore_volume_grows_with_pressure()
    test_a_fluid_without_rock_compressibility_is_refused()
    print("\nAll tests passed!")
