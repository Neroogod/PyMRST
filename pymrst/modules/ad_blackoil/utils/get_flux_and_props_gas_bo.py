"""MRST ``ad-blackoil/utils/getFluxAndPropsGas_BO.m``.

Gas phase flux and properties for the black-oil equations: gas pressure from
the oil pressure plus oil-gas capillary pressure, PVT properties at that
pressure (depending on vaporised oil `rv` when the model has it), and the
upstream-weighted flux.

Capillary pressure is evaluated per saturation region through the model's
`RelativePermeability` state function, as in MRST. The gas PVT is not region
dispatched here, which also follows MRST. Without property groupings MRST
falls back to `rock.regions.saturations` (sic); see the water function for why
that spelling is kept.
"""

from typing import Any, Optional, Tuple
import warnings

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.utils.split_face_cell_value import split_face_cell_value
from pymrst.utils.adi import value


def get_flux_and_props_gas_bo(model: Any, p_o: Any, s_g: Any, kr_g: Any,
                              transmissibility: Any, gdz: Any,
                              rv: Optional[Any] = None,
                              is_saturated: Optional[Any] = None) -> Tuple:
    """Return `(vG, bG, mobG, rhoG, pG, upcg, dpG, muG)` as MRST does."""
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    ops = model.operators
    fp_sat = _saturation_function(model)

    pc_og = 0.0
    if getattr(fluid, "pc_og", None) is not None and s_g is not None:
        pc_og = fp_sat.evaluate_function_on_domain_with_arguments(fluid.pc_og, s_g)
    p_g = p_o + pc_og

    vapoil = bool(getattr(model, "vapoil", False))
    if rv is None:
        if vapoil:
            raise ValueError("RS and saturated flag must be supplied for vapoil model")
        rv = 0.0
    if vapoil:
        b_g = fluid.b_g(p_g, rv, is_saturated)
        mu_g = fluid.mu_g(p_g, rv, is_saturated)
    else:
        b_g = fluid.b_g(p_g)
        mu_g = fluid.mu_g(p_g)
    if np.any(np.asarray(value(b_g)) < 0):
        warnings.warn("Negative gas compressibility present!", RuntimeWarning,
                      stacklevel=2)

    rho_s = np.asarray(fluid.rhoS, dtype=float).ravel()
    rho_g = b_g*(rv*rho_s[1] + rho_s[2])
    rho_gf = ops.avg(rho_g)
    dp_g = ops.grad(p_g) - rho_gf*gdz
    upcg = np.asarray(value(dp_g)) <= 0
    grid = model.G if hasattr(model, "G") else model.g
    sizes = (ops.T.size, grid.cells.num)
    kr_gf, kr_g = split_face_cell_value(ops, upcg, kr_g, sizes)
    mu_gf, mu_g = split_face_cell_value(ops, upcg, mu_g, sizes)
    mob_g = kr_g/mu_g
    v_g = -(kr_gf/mu_gf)*transmissibility*dp_g
    return v_g, b_g, mob_g, rho_g, p_g, upcg, dp_g, mu_g


def _saturation_function(model: Any) -> StateFunction:
    """MRST's `fp_sat`: the model's `RelativePermeability`, or a bare one."""
    flow = getattr(model, "FlowPropertyFunctions", None)
    if flow is not None:
        return flow.get_state_function("RelativePermeability")
    regions = getattr(getattr(model, "rock", None), "regions", None)
    saturations = (regions.get("saturations") if isinstance(regions, dict)
                   else getattr(regions, "saturations", None))
    return StateFunction(model, regions=saturations)


getFluxAndPropsGas_BO = get_flux_and_props_gas_bo

__all__ = ["get_flux_and_props_gas_bo", "getFluxAndPropsGas_BO"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi

    def _model(vapoil: bool, regions=None, pc_og=lambda sg: 3.0*sg):
        grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        rock = make_rock(grid, 1.0, 0.2)
        if regions is not None:
            rock.regions = {"saturations": np.asarray(regions)}
        if vapoil:
            pvt = dict(b_g=lambda p, rv, sat: 4.0 + 0.04*p + 0.05*rv + 0.06*sat,
                       mu_g=lambda p, rv, sat: 0.5 + 0.003*p + 0.02*rv)
        else:
            pvt = dict(b_g=lambda p: 4.0 + 0.04*p, mu_g=lambda p: 0.5 + 0.003*p)
        fluid = SimpleNamespace(rhoS=np.array([1000.0, 700.0, 2.0]),
                                pc_og=pc_og, **pvt)
        return SimpleNamespace(G=grid, rock=rock, fluid=fluid, vapoil=vapoil,
                               operators=setup_operators_tpfa(grid, rock))

    P = np.array([10.0, 12.0, 11.0])
    SG = np.array([0.1, 0.15, 0.2])
    RV = np.array([0.01, 0.02, 0.03])
    SAT = np.array([False, True, True])

    def test_gas_pressure_is_oil_pressure_plus_capillary_pressure():
        out = get_flux_and_props_gas_bo(_model(False), P, SG, SG,
                                        np.ones(2), np.zeros(2))
        np.testing.assert_allclose(out[4], P + 3.0*SG)
        print("ok test_gas_pressure_is_oil_pressure_plus_capillary_pressure")

    def test_wet_gas_density_carries_its_vaporised_oil():
        model = _model(True)
        out = get_flux_and_props_gas_bo(model, P, SG, SG, model.operators.T,
                                        np.zeros(2), RV, SAT)
        p_g = P + 3.0*SG
        b_g = 4.0 + 0.04*p_g + 0.05*RV + 0.06*SAT
        np.testing.assert_allclose(out[1], b_g)
        np.testing.assert_allclose(out[3], b_g*(RV*700.0 + 2.0))
        print("ok test_wet_gas_density_carries_its_vaporised_oil")

    def test_wet_gas_without_rv_is_refused():
        try:
            get_flux_and_props_gas_bo(_model(True), P, SG, SG,
                                      np.ones(2), np.zeros(2))
            raise SystemExit("should have raised")
        except ValueError:
            pass
        print("ok test_wet_gas_without_rv_is_refused")

    def test_dry_gas_density_is_surface_density_times_shrinkage():
        model = _model(False)
        out = get_flux_and_props_gas_bo(model, P, SG, SG, model.operators.T,
                                        np.zeros(2))
        np.testing.assert_allclose(out[3], out[1]*2.0)
        print("ok test_dry_gas_density_is_surface_density_times_shrinkage")

    def test_capillary_pressure_is_evaluated_per_saturation_region():
        model = _model(False, regions=[2, 1, 2],
                       pc_og=[lambda sg: 3.0*sg, lambda sg: 10.0*sg])
        out = get_flux_and_props_gas_bo(model, P, np.full(3, 0.1), np.ones(3),
                                        model.operators.T, np.zeros(2))
        np.testing.assert_allclose(out[4], P + [1.0, 0.3, 1.0])
        print("ok test_capillary_pressure_is_evaluated_per_saturation_region")

    def test_derivatives_survive():
        model = _model(True)
        p, sg, rv = init_variables_adi(P, SG, RV)
        out = get_flux_and_props_gas_bo(model, p, sg, sg, model.operators.T,
                                        np.zeros(2), rv, SAT)
        assert all(len(out[i].jac) == 3 for i in (0, 1, 2, 3, 4, 6, 7))
        print("ok test_derivatives_survive")

    test_gas_pressure_is_oil_pressure_plus_capillary_pressure()
    test_wet_gas_density_carries_its_vaporised_oil()
    test_wet_gas_without_rv_is_refused()
    test_dry_gas_density_is_surface_density_times_shrinkage()
    test_capillary_pressure_is_evaluated_per_saturation_region()
    test_derivatives_survive()
    print("\nAll tests passed!")
