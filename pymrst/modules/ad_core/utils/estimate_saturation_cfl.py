"""MRST ``ad-core/utils/estimateSaturationCFL.m``.

The saturation CFL number of each cell for a step `dt`: `dt/pv` times the
rate at which saturation leaves it. Each interior face carries
`|vT|*|df/ds|`, the total flux times the fractional-flow wave speed
(`getFractionalFlowMagnitude`) taken upstream of the total flux, or the
larger of the two neighbours where the phases flow in opposite directions
(cross flow). A cell sums the faces it is upstream of (downstream with
`useInflow`), and cross-flow faces count for both neighbours. With
`forces`, the total well flux in each perforated cell (from the well
solutions' `flux`) is added, scaled by that cell's wave speed. Fluxes come
from `state.flux` if present, otherwise from the `PhaseFlux` state function.

`estimate_saturation_cfl_ad` (in `cfl.py`) is PyMRST's earlier version for
the legacy models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.utils.get_fractional_flow_magnitude import (
    get_fractional_flow_magnitude,
)
from pymrst.utils.adi import value


def estimate_saturation_cfl(model: Any, state: Any, dt: Any, *, forces: Any = None,
                            useInflow: bool = False) -> np.ndarray:
    """MRST `cfl = estimateSaturationCFL(model, state, dt, ...)`."""
    internal = np.asarray(model.operators.internalConn, dtype=bool)
    if "flux" in state:
        v = np.asarray(state["flux"], dtype=float)[internal, :]
    else:
        v = np.column_stack([np.asarray(value(x), dtype=float)
                             for x in model.get_props(state, "PhaseFlux")])
    pv = np.asarray(model.operators.pv, dtype=float).ravel()
    F, Q = get_fractional_flow_magnitude(model, state)
    vT = v.sum(axis=1)
    xflow = ~(np.all(v >= 0, axis=1) | np.all(v <= 0, axis=1))
    N = np.asarray(model.operators.N, dtype=np.int64)
    l, r = N[:, 0], N[:, 1]
    flag = vT > 0
    df_face = _upstream(F, flag, xflow, l, r)
    rate_face = np.abs(vT)*np.abs(df_face)
    if Q is not None:
        T = np.asarray(model.operators.T, dtype=float)
        cap_face = _upstream(Q, flag, xflow, l, r)
        rate_face = rate_face + 2*T*cap_face
    nc = model.G.cells.num
    f = ~flag if useInflow else flag
    rate_cell = (np.bincount(l, rate_face*(f | xflow), minlength=nc)
                 + np.bincount(r, rate_face*(~f | xflow), minlength=nc))
    if forces is not None:
        well_sol = state.get("wellSol") if hasattr(state, "get") else None
        if well_sol and all(hasattr(w, "flux") for w in well_sol):     # Wells
            model = model.validate_model(forces)
            mapping = model.FacilityModel.get_prop(state, "FacilityWellMapping")
            active = np.atleast_1d(np.asarray(mapping.active, dtype=np.int64))
            flux = [np.atleast_2d(np.asarray(well_sol[k].flux, dtype=float)) for k in active]
            wflux = np.vstack(flux).sum(axis=1) if flux else np.zeros(0)
            wc = np.asarray(mapping.cells, dtype=np.int64).ravel()
            np.add.at(rate_cell, wc, np.abs(wflux)*F[wc])
    return (dt/pv)*rate_cell


def _upstream(F: np.ndarray, flag: np.ndarray, xflow: np.ndarray, l: np.ndarray,
              r: np.ndarray) -> np.ndarray:
    """MRST `faceUpstr(flag, F)`, with the larger magnitude of the two
    neighbours on cross-flow faces."""
    df_face = np.where(flag, F[l], F[r])
    df_face[xflow] = np.maximum(np.abs(F[l[xflow]]), np.abs(F[r[xflow]]))
    return df_face


estimateSaturationCFL = estimate_saturation_cfl

__all__ = ["estimateSaturationCFL", "estimate_saturation_cfl"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.solvers.init_res_sol import init_res_sol

    def _model(n=4):
        G = compute_geometry(cart_grid([n], [float(n)]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.5), fluid, gas=False)
        return model.validate_model(), G

    def test_a_uniform_flux_gives_dt_q_over_pv():
        """kr linear, equal viscosities: df/ds = 1, so cfl = dt*q/pv in every
        cell that sends flux on (the last cell has no outgoing interior
        face)."""
        model, G = _model()
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        flux = np.zeros((G.faces.num, 2))
        internal = np.asarray(model.operators.internalConn, dtype=bool)
        flux[internal] = [1e-5, 1e-5]
        state["flux"] = flux
        cfl = estimate_saturation_cfl(model, state, 10.0)
        pv = np.asarray(model.operators.pv)
        np.testing.assert_allclose(cfl[:3], 10*2e-5/pv[:3])
        assert cfl[3] == 0
        inflow = estimate_saturation_cfl(model, state, 10.0, useInflow=True)
        assert inflow[0] == 0 and inflow[3] > 0
        print("ok test_a_uniform_flux_gives_dt_q_over_pv")

    def test_cross_flow_counts_for_both_neighbours():
        model, G = _model(2)
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        flux = np.zeros((G.faces.num, 2))
        internal = np.asarray(model.operators.internalConn, dtype=bool)
        flux[internal] = [2e-5, -1e-5]
        state["flux"] = flux
        cfl = estimate_saturation_cfl(model, state, 1.0)
        pv = np.asarray(model.operators.pv)
        np.testing.assert_allclose(cfl, 1e-5/pv)
        print("ok test_cross_flow_counts_for_both_neighbours")

    test_a_uniform_flux_gives_dt_q_over_pv()
    test_cross_flow_counts_for_both_neighbours()
    print("\nAll tests passed!")
