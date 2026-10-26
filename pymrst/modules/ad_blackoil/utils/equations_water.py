"""MRST ``ad-blackoil/utils/equationsWater.m``.

The single-phase water equations: accumulation with pressure-dependent pore
volume and shrinkage factor, upstream-weighted Darcy flux with gravity,
boundary conditions and sources, and the wells. Returns MRST's
`(problem, state)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.utils.adi import value


def equations_water(state0: Any, state: Any, model: Any, dt: float,
                    driving_forces: Any, *, Verbose: bool = False,
                    reverseMode: bool = False, resOnly: bool = False,
                    iteration: float = -1, **_: Any):
    s = model.operators
    state = copy_state(state)
    G = model.G
    f = model.fluid
    p, well_sol = model.get_props(state, "pressure", "wellsol")
    p0, well_sol0 = model.get_props(state0, "pressure", "wellSol")
    well_vars, well_var_names, well_map = \
        model.FacilityModel.get_all_primary_variables(well_sol)
    if not resOnly:
        if not reverseMode:
            p, *well_vars = model.AutoDiffBackend.init_variables_ad(p, *well_vars)
        else:
            # MRST seeds `sW0` here, which a single-phase model does not have,
            # so reverse mode errors in MATLAB as well.
            raise NotImplementedError(
                "equationsWater: MRST's reverse-mode branch references sW0, "
                "which the single-phase model does not define")
    primary_vars = ["pressure"] + list(well_var_names)
    gdz = s.grad(np.asarray(G.cells.centroids, dtype=float)) @ model.get_gravity_vector()
    tr_mult = f.tranMultR(p) if getattr(f, "tranMultR", None) is not None else 1.0
    pv_mult = pv_mult0 = 1.0
    if getattr(f, "pvMultR", None) is not None:
        pv_mult = f.pvMultR(p)
        pv_mult0 = f.pvMultR(p0)
    trans_mult = f.transMult(p) if getattr(f, "transMult", None) is not None else 1.0
    trans = s.T*trans_mult
    b_w = f.bW(p)
    rho_w = b_w*f.rhoWS
    rho_wf = s.avg(rho_w)
    mob_w = tr_mult/f.muW(p)
    dp_w = s.grad(p) - rho_wf*gdz
    upcw = np.asarray(value(dp_w)) <= 0
    v_w = -s.upw(upcw, mob_w)*trans*dp_w
    b_wv_w = s.upw(upcw, b_w)*v_w
    if model.outputFluxes:
        state = model.store_fluxes(state, v_w, None, None)
    if model.extraStateOutput:
        state = model.storebfactors(state, b_w, None, None)
        state = model.store_mobilities(state, mob_w, None, None)
        state = model.store_upstream_indices(state, upcw, None, None)
    names = ["water"]
    types = ["cell"]
    eqs = [(s.pv/dt)*(pv_mult*b_w - pv_mult0*f.bW(p0)) + s.div(b_wv_w)]
    s_w = np.ones(G.cells.num)
    eqs, state, _ = model.add_boundary_conditions_and_sources(
        eqs, names, types, state, [p], [s_w], [mob_w], [rho_w], [], [],
        driving_forces)
    opt = {"iteration": iteration, "reverseMode": reverseMode,
           "resOnly": resOnly}
    eqs, names, types, state["wellSol"], _ = model.insert_well_equations(
        eqs, names, types, well_sol0, well_sol, well_vars, well_map, p, [mob_w],
        [rho_w], [], [], dt, opt)
    problem = model.setup_linearized_problem(eqs, types, names, primary_vars,
                                             state, dt)
    return problem, state


equationsWater = equations_water

__all__ = ["equations_water", "equationsWater"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.water_model import WaterModel
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock

    def _model():
        grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        fluid = init_simple_adi_fluid(phases="W", mu=[1e-3], rho=[1000.0],
                                      n=[1.0], c=[1e-9])
        return WaterModel(grid, make_rock(grid, 1e-13, 0.3), fluid).validate_model()

    def _state(p):
        return State(pressure=np.asarray(p, dtype=float), wellSol=[])

    def test_a_uniform_state_at_rest_has_no_residual():
        with isolated_gravity("off"):
            model = _model()
            problem, _ = equations_water(_state([1e7]*3), _state([1e7]*3), model,
                                         1.0, model.get_valid_driving_forces())
        np.testing.assert_allclose(value(problem.equations[0]), 0.0, atol=1e-15)
        assert problem.primary_variables == ["pressure"]
        print("ok test_a_uniform_state_at_rest_has_no_residual")

    def test_compression_and_flux_enter_the_residual():
        with isolated_gravity("off"):
            model = _model()
            problem, _ = equations_water(_state([1e7]*3), _state([2e7, 1e7, 1e7]),
                                         model, 10.0,
                                         model.get_valid_driving_forces())
        r = value(problem.equations[0])
        assert r[0] > 0 and r[1] < 0, r
        J = problem.equations[0].jac[0].toarray()
        assert J[0, 0] > 0 and J[0, 1] < 0
        print("ok test_compression_and_flux_enter_the_residual")

    test_a_uniform_state_at_rest_has_no_residual()
    test_compression_and_flux_enter_the_residual()
    print("\nAll tests passed!")
