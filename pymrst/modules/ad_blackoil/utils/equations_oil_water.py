"""MRST ``ad-blackoil/utils/equationsOilWater.m``.

The two-phase oil-water equations: water and oil accumulation with
pressure-dependent pore volume and shrinkage, upstream-weighted fluxes from
the flow discretisation, boundary conditions, sources, aquifers and wells.
Returns MRST's `(problem, state)`.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.models.aquifers.aquifer_model import add_aquifers_contribution


def equations_oil_water(state0: Any, state: Any, model: Any, dt: float,
                        driving_forces: Any, *, Verbose: bool = False,
                        reverseMode: bool = False, resOnly: bool = False,
                        iteration: float = -1, **_: Any):
    s = model.operators
    state = copy_state(state)
    state0 = copy_state(state0)
    p, s_w, well_sol = model.get_props(state, "pressure", "water", "wellsol")
    p0, s_w0, well_sol0 = model.get_props(state0, "pressure", "water", "wellSol")
    well_vars, well_var_names, well_map = \
        model.FacilityModel.get_all_primary_variables(well_sol)
    if not resOnly and not reverseMode:
        p, s_w, *well_vars = model.AutoDiffBackend.init_variables_ad(p, s_w,
                                                                     *well_vars)
    primary_vars = ["pressure", "sW"] + list(well_var_names)
    s_o = 1 - s_w
    s_o0 = 1 - s_w0
    sat, sat0 = [s_w, s_o], [s_w0, s_o0]
    state = model.set_props(state, ["s", "pressure"], [sat, p])
    state0 = model.set_props(state0, ["s", "pressure"], [sat0, p0])
    state = model.init_state_function_containers(state)
    b, pv = model.get_props(state, "ShrinkageFactors", "PoreVolume")
    b0, pv0 = model.get_props(state0, "ShrinkageFactors", "PoreVolume")
    phase_flux, flags = model.get_props(state, "PhaseFlux", "PhaseUpwindFlag")
    b_w, b_o = b
    b_w0, b_o0 = b0
    v_w, v_o = phase_flux
    upcw, upco = flags
    pressures, mob, rho = model.get_props(state, "PhasePressures", "Mobility",
                                          "Density")
    if model.outputFluxes:
        state = model.store_fluxes(state, v_w, v_o, None)
    if model.extraStateOutput:
        state = model.storebfactors(state, b_w, b_o, None)
        state = model.store_mobilities(state, *mob, None)
        state = model.store_upstream_indices(state, upcw, upco, None)
    b_ov_o = s.upw(upco, b_o)*v_o
    b_wv_w = s.upw(upcw, b_w)*v_w
    water = (1/dt)*(pv*b_w*s_w - pv0*b_w0*s_w0)
    oil = (1/dt)*(pv*b_o*s_o - pv0*b_o0*s_o0)
    eqs = [water, oil]
    names = ["water", "oil"]
    types = ["cell", "cell"]
    eqs, state, _ = model.add_boundary_conditions_and_sources(
        eqs, names, types, state, pressures, sat, mob, rho, [], [], driving_forces)
    if model.AquiferModel is not None:
        eqs = add_aquifers_contribution(model.AquiferModel, eqs, names, state, dt)
    opt = {"iteration": iteration, "reverseMode": reverseMode, "resOnly": resOnly}
    eqs, names, types, state["wellSol"], _ = model.insert_well_equations(
        eqs, names, types, well_sol0, well_sol, well_vars, well_map, p, mob, rho,
        [], [], dt, opt)
    eqs[0] = eqs[0] + s.div(b_wv_w)
    eqs[1] = eqs[1] + s.div(b_ov_o)
    problem = model.setup_linearized_problem(eqs, types, names, primary_vars,
                                             state, dt)
    return problem, state


equationsOilWater = equations_oil_water

__all__ = ["equations_oil_water", "equationsOilWater"]


if __name__ == "__main__":
    def test_the_function_is_mrsts_name_too():
        assert equationsOilWater is equations_oil_water
        print("ok test_the_function_is_mrsts_name_too")

    test_the_function_is_mrsts_name_too()
    print("\nAll tests passed!")
