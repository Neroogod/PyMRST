"""MRST ``ad-core/utils/estimateCompositionCFL.m``.

The composition CFL number of each cell and component for a step `dt`:
`dt` times the rate at which the component's mass leaves the cell relative
to the mass there. Each interior face carries the component's total flux
(`ComponentTotalFlux`) over the upstream cell's component mass
(`ComponentTotalMass`, 1 where the component is under 1e-7 of the cell's
mass; on cross-flow faces MRST takes the larger neighbour for the first
component only - its linear indexing of the mass matrix), summed over the faces a
cell is upstream of (downstream with `useInflow`). Non-finite values are
set to 0. `forces` is accepted, as in MRST, and not used.

`estimate_composition_cfl_ad` (in `cfl.py`) is PyMRST's earlier version
for the legacy models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.utils.adi import value


def estimate_composition_cfl(model: Any, state: Any, dt: Any, *, forces: Any = None,
                             useInflow: bool = False) -> np.ndarray:
    """MRST `cfl = estimateCompositionCFL(model, state, dt, ...)`."""
    internal = np.asarray(model.operators.internalConn, dtype=bool)
    v = np.asarray(state["flux"], dtype=float)[internal, :]
    vT = v.sum(axis=1)
    xflow = ~(np.all(v >= 0, axis=1) | np.all(v <= 0, axis=1))
    N = np.asarray(model.operators.N, dtype=np.int64)
    l, r = N[:, 0], N[:, 1]
    flag = vT > 0
    q = _columns(model.get_prop(state, "ComponentTotalFlux"))
    compMass = _columns(model.get_prop(state, "ComponentTotalMass"))
    totMass = compMass.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = compMass/totMass[:, None] < 1e-7
    compMass[bad] = 1
    massFace = np.where(flag[:, None], compMass[l], compMass[r])
    # MRST's `df_face(xflow) = max(abs(F(l(xflow))), abs(F(r(xflow))))` on the
    # matrix of component masses: linear indexing reads and writes the first
    # column only, so the other components keep their upstream value.
    massFace[xflow, 0] = np.maximum(np.abs(compMass[l[xflow], 0]), np.abs(compMass[r[xflow], 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_face = np.abs(q)/massFace
    nc = model.G.cells.num
    ncomp = model.get_number_of_components()
    cfl = np.zeros((nc, ncomp))
    f = ~flag if useInflow else flag
    for i in range(ncomp):
        rate_cell = (np.bincount(l, rate_face[:, i]*(f | xflow), minlength=nc)
                     + np.bincount(r, rate_face[:, i]*(~f | xflow), minlength=nc))
        cfl[:, i] = dt*rate_cell
    cfl[~np.isfinite(cfl)] = 0
    return cfl


def _columns(values: Any) -> np.ndarray:
    """MRST's `value(x')` of a cell array of columns: an n-by-k matrix."""
    return np.column_stack([np.asarray(value(x), dtype=float).ravel() for x in values])


estimateCompositionCFL = estimate_composition_cfl

__all__ = ["estimateCompositionCFL", "estimate_composition_cfl"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.solvers.init_res_sol import init_res_sol

    def test_component_flux_over_upstream_mass():
        """Pressure falling left to right drives both components right: the
        component flux over the upstream cell's mass, times dt, in the two
        cells with an outgoing face."""
        G = compute_geometry(cart_grid([3], [3.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.5), fluid, gas=False)
        model = model.validate_model()
        state = init_res_sol(G, 1e7, np.array([0.5, 0.5]))
        state.pressure = np.array([3e7, 2e7, 1e7])
        state = model.validate_state(state)
        flux = np.zeros((G.faces.num, 2))
        internal = np.asarray(model.operators.internalConn, dtype=bool)
        flux[internal] = [1e-5, 1e-5]
        state["flux"] = flux
        cfl = estimate_composition_cfl(model, state, 10.0)
        assert cfl.shape == (3, 2)
        assert np.all(cfl[:2] > 0) and np.all(cfl[2] == 0)
        np.testing.assert_allclose(cfl[0, 0], cfl[1, 0])
        print("ok test_component_flux_over_upstream_mass")

    test_component_flux_over_upstream_mass()
    def test_cross_flow_faces_take_the_larger_neighbour_for_the_first_component_only():
        """MRST's `upstream` indexes the n-by-ncomp mass matrix linearly, so on
        a cross-flow face only component 1 takes the larger neighbour's mass;
        component 2 keeps the upstream cell's."""
        G = compute_geometry(cart_grid([2], [2.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, np.array([0.1, 0.5])), fluid,
                                     gas=False).validate_model()
        state = init_res_sol(G, 1e7, np.array([0.5, 0.5]))
        state.s = np.array([[0.2, 0.8], [0.5, 0.5]])   # the downstream cell holds more of both
        state = model.validate_state(state)
        flux = np.zeros((G.faces.num, 2))
        internal = np.asarray(model.operators.internalConn, dtype=bool)
        flux[internal] = [2e-5, -1e-5]          # water right, oil left: cross-flow, total right
        state["flux"] = flux
        q = np.column_stack([np.asarray(value(x)) for x in model.get_prop(state, "ComponentTotalFlux")])
        m = np.column_stack([np.asarray(value(x)) for x in model.get_prop(state, "ComponentTotalMass")])
        cfl = estimate_composition_cfl(model, state, 1.0)
        # Both cells see the cross-flow face. Water: the larger of the two masses.
        np.testing.assert_allclose(cfl[0, 0], abs(q[0, 0])/max(m[0, 0], m[1, 0]))
        # Oil: the upstream (left, total flux > 0) cell's mass, not the larger one.
        assert m[1, 1] > m[0, 1]
        np.testing.assert_allclose(cfl[0, 1], abs(q[0, 1])/m[0, 1])
        print("ok test_cross_flow_faces_take_the_larger_neighbour_for_the_first_component_only")

    test_cross_flow_faces_take_the_larger_neighbour_for_the_first_component_only()
    print("\nAll tests passed!")
