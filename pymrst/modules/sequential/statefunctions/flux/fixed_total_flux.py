"""MRST ``sequential/statefunctions/flux/FixedTotalFlux.m``.

The total volumetric flux over the interior faces, fixed from the pressure
step: the sum over phases of `state.flux` (or another field).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import STATE, StateFunction


class FixedTotalFlux(StateFunction):
    """MRST `FixedTotalFlux`."""

    def __init__(self, model: Any, fluxfield: str = "flux", **options: Any):
        super().__init__(model)
        self.fluxfield = fluxfield          # Name of field in state where static flux is stored
        self.depends_on([fluxfield], STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> np.ndarray:
        internal = np.asarray(model.operators.internalConn, dtype=bool)
        flux = np.asarray(state[self.fluxfield], dtype=float)
        if flux.ndim == 1:
            flux = flux[:, None]           # MATLAB's `initResSol` flux is a column
        return flux[internal, :].sum(axis=1)

    evaluateOnDomain = evaluate_on_domain


__all__ = ["FixedTotalFlux"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_it_sums_the_stored_phase_fluxes():
        vT = T.parent.get_prop(T.ad_state, "TotalFlux")
        internal = np.asarray(T.parent.operators.internalConn, dtype=bool)
        np.testing.assert_allclose(vT, np.asarray(T.state["flux"])[internal].sum(axis=1))
        assert np.all(vT > 0), "the flood moves left to right"
        print("ok test_it_sums_the_stored_phase_fluxes")

    test_it_sums_the_stored_phase_fluxes()
    print("\nAll tests passed!")
