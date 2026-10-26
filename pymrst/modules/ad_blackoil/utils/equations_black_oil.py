"""MRST ``ad-blackoil/utils/equationsBlackOil.m``.

The three-phase black-oil equations with dissolved gas (`rs`) and vaporised
oil (`rv`). The hydrocarbon unknown `x` switches meaning cell by cell - `rs`
in undersaturated oil, `rv` in undersaturated gas, `sG` where both phases
are present - and `calculateHydrocarbonsFromStatusBO` recovers `sG`, `rs` and
`rv` from it. Accumulation, upstream-weighted fluxes with the dissolved
carry-over, boundary conditions, sources, aquifers and wells are assembled
into MRST's `(problem, state)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.models.aquifers.aquifer_model import add_aquifers_contribution


def equations_black_oil(state0: Any, state: Any, model: Any, dt: float,
                        driving_forces: Any, *, Verbose: bool = False,
                        reverseMode: bool = False, resOnly: bool = False,
                        iteration: float = -1, **_: Any):
    s = model.operators
    state = copy_state(state)
    state0 = copy_state(state0)
    p, s_g, rs, rv, well_sol = model.get_props(state, "pressure", "gas", "rs",
                                              "rv", "wellSol")
    p0, s_g0, rs0, rv0, well_sol0 = model.get_props(state0, "pressure", "gas",
                                                   "rs", "rv", "wellSol")
    if model.water:
        s_w = model.get_prop(state, "water")
        s_w0 = model.get_prop(state0, "water")
    else:
        s_w = s_w0 = 0.0
    well_vars, well_var_names, well_map = \
        model.FacilityModel.get_all_primary_variables(well_sol)
    st = model.get_cell_status_vo(state, 1 - s_w - s_g, s_w, s_g)
    st0 = model.get_cell_status_vo(state0, 1 - s_w0 - s_g0, s_w0, s_g0)
    if model.disgas or model.vapoil:
        x = _mask(st[0])*rs + _mask(st[1])*rv + _mask(st[2])*s_g
        gvar = "x"
    else:
        x = s_g
        gvar = "sG"
    backend = model.AutoDiffBackend
    if not resOnly:
        if not reverseMode:
            if model.water:
                p, s_w, x, *well_vars = backend.init_variables_ad(p, s_w, x, *well_vars)
            else:
                p, x, *well_vars = backend.init_variables_ad(p, x, *well_vars)
        else:
            well_vars0, _, _ = model.FacilityModel.get_all_primary_variables(well_sol0)
            x0 = _mask(st0[0])*rs0 + _mask(st0[1])*rv0 + _mask(st0[2])*s_g0
            if model.water:
                p0, s_w0, x0, *_ = backend.init_variables_ad(p0, s_w0, x0, *well_vars0)
            else:
                p0, x0, *_ = backend.init_variables_ad(p0, x0, *well_vars0)
            s_g0, rs0, rv0, _, _ = model.calculate_hydrocarbons_from_status_bo(
                st0, 1 - s_w0, x0, rs0, rv0, p0)
    if model.water:
        primary_vars = ["pressure", "sW", gvar] + list(well_var_names)
    else:
        primary_vars = ["pressure", gvar] + list(well_var_names)
    if not reverseMode:
        s_g, rs, rv, _, _ = model.calculate_hydrocarbons_from_status_bo(
            st, 1 - s_w, x, rs, rv, p)
    s_o = 1 - s_w - s_g
    s_o0 = 1 - s_w0 - s_g0
    if model.water:
        sat, sat0 = [s_w, s_o, s_g], [s_w0, s_o0, s_g0]
    else:
        sat, sat0 = [s_o, s_g], [s_o0, s_g0]
    state = model.set_props(state, ["s", "pressure", "rs", "rv"], [sat, p, rs, rv])
    state0 = model.set_props(state0, ["s", "pressure", "rs", "rv"],
                             [sat0, p0, rs0, rv0])
    state = model.init_state_function_containers(state)
    b, pv = model.get_props(state, "ShrinkageFactors", "PoreVolume")
    b0, pv0 = model.get_props(state0, "ShrinkageFactors", "PoreVolume")
    phase_flux, flags = model.get_props(state, "PhaseFlux", "PhaseUpwindFlag")
    b_w, b_o, b_g = b
    b_w0, b_o0, b_g0 = b0
    v_w, v_o, v_g = phase_flux
    upcw, upco, upcg = flags
    pressures, mob, rho = model.get_props(state, "PhasePressures", "Mobility",
                                          "Density")
    if model.outputFluxes:
        state = model.store_fluxes(state, v_w, v_o, v_g)
    if model.extraStateOutput:
        state = model.storebfactors(state, b_w, b_o, b_g)
        state = model.store_mobilities(state, *mob)
        state = model.store_upstream_indices(state, upcw, upco, upcg)
    b_ov_o = s.upw(upco, b_o)*v_o
    b_gv_g = s.upw(upcg, b_g)*v_g
    b_wv_w = s.upw(upcw, b_w)*v_w
    water = (1/dt)*(pv*b_w*s_w - pv0*b_w0*s_w0)
    div_water = s.div(b_wv_w)
    if model.vapoil:
        rvb_gv_g = s.upw(upcg, rv)*b_gv_g
        oil = (1/dt)*(pv*(b_o*s_o + rv*b_g*s_g) - pv0*(b_o0*s_o0 + rv0*b_g0*s_g0))
        div_oil = s.div(b_ov_o + rvb_gv_g)
    else:
        oil = (1/dt)*(pv*b_o*s_o - pv0*b_o0*s_o0)
        div_oil = s.div(b_ov_o)
    if model.disgas:
        rsb_ov_o = s.upw(upco, rs)*b_ov_o
        gas = (1/dt)*(pv*(b_g*s_g + rs*b_o*s_o) - pv0*(b_g0*s_g0 + rs0*b_o0*s_o0))
        div_gas = s.div(b_gv_g + rsb_ov_o)
    else:
        gas = (1/dt)*(pv*b_g*s_g - pv0*b_g0*s_g0)
        div_gas = s.div(b_gv_g)
    eqs = [water, oil, gas]
    div_terms = [div_water, div_oil, div_gas]
    names = ["water", "oil", "gas"]
    types = ["cell", "cell", "cell"]
    dissolved = model.get_dissolution_matrix(rs, rv)
    eqs, state, _ = model.add_boundary_conditions_and_sources(
        eqs, names, types, state, pressures, sat, mob, rho, dissolved, [],
        driving_forces)
    if model.AquiferModel is not None:
        eqs = add_aquifers_contribution(model.AquiferModel, eqs, names, state, dt)
    opt = {"iteration": iteration, "reverseMode": reverseMode, "resOnly": resOnly}
    eqs, names, types, state["wellSol"], _ = model.insert_well_equations(
        eqs, names, types, well_sol0, well_sol, well_vars, well_map, p, mob, rho,
        dissolved, [], dt, opt)
    for i, div in enumerate(div_terms):
        eqs[i] = eqs[i] + div
    problem = model.setup_linearized_problem(eqs, types, names, primary_vars,
                                             state, dt)
    return problem, state


def _mask(st: Any) -> Any:
    """A status mask as 0/1 floats, for multiplying AD values."""
    return np.asarray(st, dtype=float)


equationsBlackOil = equations_black_oil

__all__ = ["equations_black_oil", "equationsBlackOil"]


if __name__ == "__main__":
    def test_the_function_is_mrsts_name_too():
        assert equationsBlackOil is equations_black_oil
        print("ok test_the_function_is_mrsts_name_too")

    test_the_function_is_mrsts_name_too()
    print("\nAll tests passed!")
