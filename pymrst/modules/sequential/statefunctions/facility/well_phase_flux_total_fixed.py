"""MRST ``sequential/statefunctions/facility/WellPhaseFluxTotalFixed.m``.

Well phase fluxes for transport at a fixed total well flux: each
perforation's total flux (from the well solutions' `flux`, fixed by the
pressure step) split by the fractional flows of the perforated cells'
mobilities. Where the flux goes into the reservoir (`-dp > 0`), the split is
the well's injection composition `compi` instead - mixed by
`crossFlowMixture` when some perforations cross-flow (injection into a
producer or production from an injector).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class WellPhaseFluxTotalFixed(StateFunction):
    """MRST `WellPhaseFluxTotalFixed`."""

    def __init__(self, model: Any, *args: Any, **options: Any):
        super().__init__(model, *args, **options)
        self.depends_on(["FacilityWellMapping", "PressureGradient", "WellIndex"])
        self.depends_on(["Mobility"], "FlowPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        from pymrst.modules.ad_core.utils.cross_flow_mixture import cross_flow_mixture
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        mob = model.ReservoirModel.get_props(state, "Mobility")
        nph = len(mob)
        q_ph: List[Any] = [None]*nph
        W = mapping.W
        if W is None or len(W) == 0:
            return q_ph
        dp, wi = self.get_evaluated_dependencies(state, "PressureGradient", "WellIndex")
        active = np.atleast_1d(np.asarray(mapping.active, dtype=np.int64))
        ws = [state["wellSol"][k] for k in active]
        qT = np.vstack([np.atleast_2d(np.asarray(w.flux, dtype=float)) for w in ws]).sum(axis=1)
        cells = np.asarray(mapping.cells, dtype=np.int64)
        mobw = [m[cells] for m in mob]
        mobt = 0
        for m in mobw:
            mobt = mobt + m
        perf2well = np.asarray(mapping.perf2well, dtype=np.int64)
        isInjector = np.asarray(mapping.isInjector, dtype=bool)[perf2well]
        vTdp = np.asarray(value(-dp), dtype=float).ravel()
        injection = vTdp > 0
        production = ~injection & (vTdp != 0)
        crossflow = (injection & ~isInjector) | (production & isInjector)
        f = [m/mobt for m in mobw]
        if np.any(injection):
            compi = np.vstack([np.asarray(w.compi, dtype=float).ravel() for w in W])
            if np.any(crossflow):
                if getattr(model, "verbose", 0) > 1:
                    print(f"Crossflow occuring in {int(np.sum(crossflow))} perforations")
                q_wb = np.column_stack([np.asarray(value(m), dtype=float) for m in mobw])*vTdp[:, None]
                compi = cross_flow_mixture(q_wb, compi, mapping)
            compi_perf = compi[perf2well, :]
            for i in range(nph):
                f[i] = _assign(f[i], injection, compi_perf[injection, i])
        for i in range(nph):
            q_ph[i] = f[i]*qT
        return q_ph

    evaluateOnDomain = evaluate_on_domain


def _assign(x: Any, mask: np.ndarray, values: np.ndarray) -> Any:
    """MATLAB `x(mask) = values` for an AD or numeric `x`."""
    if hasattr(x, "jac"):
        from copy import copy
        x = copy(x)
        x[mask] = values
        return x
    x = np.array(x, dtype=float, copy=True)
    x[mask] = values
    return x


__all__ = ["WellPhaseFluxTotalFixed"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _function(dp, flux, is_injector, compi):
        fn = WellPhaseFluxTotalFixed.__new__(WellPhaseFluxTotalFixed)
        mapping = SimpleNamespace(W=[SimpleNamespace(compi=np.asarray(compi, dtype=float))],
                                  active=np.array([0]), cells=np.array([0, 1]),
                                  perf2well=np.array([0, 0]), isInjector=np.array([is_injector]))
        values = {"FacilityWellMapping": mapping, "PressureGradient": np.asarray(dp, dtype=float),
                  "WellIndex": np.ones(2)}
        fn.get_evaluated_dependencies = lambda state, *names: (
            values[names[0]] if len(names) == 1 else [values[n] for n in names])
        mob = [np.array([1.0, 3.0]), np.array([3.0, 1.0])]
        model = SimpleNamespace(ReservoirModel=SimpleNamespace(get_props=lambda s, n: mob),
                                verbose=0)
        state = {"wellSol": [SimpleNamespace(flux=np.asarray(flux, dtype=float))]}
        return fn.evaluate_on_domain(model, state)

    def test_production_splits_by_fractional_flow():
        q = _function([1.0, 1.0], [[-2.0], [-4.0]], False, [0, 1])
        np.testing.assert_allclose(q[0], [-0.5, -3.0])
        np.testing.assert_allclose(q[1], [-1.5, -1.0])
        print("ok test_production_splits_by_fractional_flow")

    def test_injection_takes_the_well_composition():
        q = _function([-1.0, -1.0], [[2.0], [4.0]], True, [1, 0])
        np.testing.assert_allclose(q[0], [2.0, 4.0])
        np.testing.assert_allclose(q[1], [0.0, 0.0])
        print("ok test_injection_takes_the_well_composition")

    test_production_splits_by_fractional_flow()
    test_injection_takes_the_well_composition()
    print("\nAll tests passed!")
