"""MRST ``ad-blackoil/utils/getFluxAndPropsWater_BO.m``.

Water phase flux and properties for the black-oil equations: water pressure
from the oil pressure and oil-water capillary pressure, PVT properties at that
pressure, the gravity-corrected potential difference, and the upstream-weighted
flux.

Capillary pressure is evaluated per saturation region and the PVT functions
per PVT region, through the model's `RelativePermeability` and `Density` state
functions - exactly as MRST does it. A model that carries no property
groupings falls back to a bare `StateFunction`; MRST then reads
`rock.regions.saturations` (sic), which is not the name MRST stores regions
under, so in practice that fallback is single-region. The spelling is kept so
the fallback behaves as MRST's does.
"""

from typing import Any, Tuple
import warnings

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.utils.split_face_cell_value import split_face_cell_value
from pymrst.utils.adi import value


def get_flux_and_props_water_bo(model: Any, p_o: Any, s_w: Any, kr_w: Any,
                                transmissibility: Any, gdz: Any) -> Tuple:
    """Return `(vW, bW, mobW, rhoW, pW, upcw, dpW, muW)` as MRST does."""
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    ops = model.operators
    fp_sat, fp_pvt = _region_functions(model)

    pc_ow = 0.0
    if getattr(fluid, "pc_ow", None) is not None and s_w is not None:
        pc_ow = fp_sat.evaluate_function_on_domain_with_arguments(fluid.pc_ow, s_w)
    p_w = p_o - pc_ow

    mu_w = fp_pvt.evaluate_function_on_domain_with_arguments(fluid.mu_w, p_w)
    b_w = fp_pvt.evaluate_function_on_domain_with_arguments(fluid.b_w, p_w)
    rho_w = b_w*_surface_density(fluid, 0)
    rho_wf = ops.avg(rho_w)
    dp_w = ops.grad(p_w) - rho_wf*gdz

    upcw = np.asarray(value(dp_w)) <= 0
    sizes = _face_cell_sizes(model)
    kr_wf, kr_w = split_face_cell_value(ops, upcw, kr_w, sizes)
    mu_wf, mu_w = split_face_cell_value(ops, upcw, mu_w, sizes)
    mob_w = kr_w/mu_w
    v_w = -(kr_wf/mu_wf)*transmissibility*dp_w

    if np.any(np.asarray(value(b_w)) < 0):
        warnings.warn("Negative water compressibility present!", RuntimeWarning,
                      stacklevel=2)
    return v_w, b_w, mob_w, rho_w, p_w, upcw, dp_w, mu_w


def _region_functions(model: Any) -> Tuple[StateFunction, StateFunction]:
    """MRST's `fp_sat`/`fp_pvt`: the model's own region-aware functions."""
    flow = getattr(model, "FlowPropertyFunctions", None)
    pvt = getattr(model, "PVTPropertyFunctions", None)
    if flow is not None and pvt is not None:
        return (flow.get_state_function("RelativePermeability"),
                pvt.get_state_function("Density"))
    regions = getattr(getattr(model, "rock", None), "regions", None)
    saturations = (regions.get("saturations") if isinstance(regions, dict)
                   else getattr(regions, "saturations", None))
    return StateFunction(model, regions=saturations), StateFunction(model)


def _face_cell_sizes(model: Any) -> Tuple[int, int]:
    """MRST `splitFaceCellValue`'s `[numel(s.T), G.cells.num]`."""
    grid = model.G if hasattr(model, "G") else model.g
    return model.operators.T.size, grid.cells.num


def _surface_density(fluid: Any, phase: int) -> float:
    """MRST `fluid.rhoWS`: PyMRST fluids carry `rhoS` in W, O, G order."""
    return np.asarray(fluid.rhoS, dtype=float).ravel()[phase]


getFluxAndPropsWater_BO = get_flux_and_props_water_bo

__all__ = ["get_flux_and_props_water_bo", "getFluxAndPropsWater_BO"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi

    def _model(regions=None, pc_ow=lambda sw: 2.0*sw):
        grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        rock = make_rock(grid, 1.0, 0.2)
        if regions is not None:
            rock.regions = {"saturations": np.asarray(regions)}
        fluid = SimpleNamespace(rhoS=np.array([1000.0, 700.0, 2.0]),
                                pc_ow=pc_ow,
                                b_w=lambda p: 1.0 + 0.01*p,
                                mu_w=lambda p: 2.0 + 0.001*p)
        return SimpleNamespace(G=grid, rock=rock, fluid=fluid,
                               operators=setup_operators_tpfa(grid, rock))

    def test_water_pressure_is_oil_pressure_less_capillary_pressure():
        model = _model()
        p = np.array([10.0, 12.0, 11.0])
        sw = np.array([0.2, 0.3, 0.25])
        out = get_flux_and_props_water_bo(model, p, sw, sw,
                                          model.operators.T, np.zeros(2))
        np.testing.assert_allclose(out[4], p - 2.0*sw)
        np.testing.assert_allclose(out[1], 1.0 + 0.01*(p - 2.0*sw))
        print("ok test_water_pressure_is_oil_pressure_less_capillary_pressure")

    def test_flux_is_upstream_weighted_against_the_potential():
        model = _model()
        p = np.array([10.0, 12.0, 11.0])
        sw = np.array([0.2, 0.3, 0.25])
        v, _, _, rho, pw, up, dp, _ = get_flux_and_props_water_bo(
            model, p, sw, sw, model.operators.T, np.zeros(2))
        np.testing.assert_allclose(dp, np.diff(pw))
        np.testing.assert_array_equal(up, dp <= 0)
        # Face 0 flows from cell 1 down to cell 0, so cell 1 is upstream.
        mu = 2.0 + 0.001*pw
        expected = -(sw[1]/mu[1])*model.operators.T[0]*dp[0]
        np.testing.assert_allclose(v[0], expected)
        print("ok test_flux_is_upstream_weighted_against_the_potential")

    def test_derivatives_survive_every_output():
        model = _model()
        p, sw = init_variables_adi(np.array([10.0, 12.0, 11.0]),
                                   np.array([0.2, 0.3, 0.25]))
        out = get_flux_and_props_water_bo(model, p, sw, sw,
                                          model.operators.T, np.zeros(2))
        for index in (0, 1, 2, 3, 4, 6, 7):
            assert len(out[index].jac) == 2, index
        print("ok test_derivatives_survive_every_output")

    def test_capillary_pressure_is_evaluated_per_saturation_region():
        model = _model(regions=[1, 2, 1],
                       pc_ow=[lambda sw: 2.0*sw, lambda sw: 5.0*sw])
        p = np.full(3, 10.0)
        sw = np.full(3, 0.5)
        out = get_flux_and_props_water_bo(model, p, sw, sw,
                                          model.operators.T, np.zeros(2))
        np.testing.assert_allclose(out[4], [9.0, 7.5, 9.0])
        print("ok test_capillary_pressure_is_evaluated_per_saturation_region")

    def test_absent_capillary_pressure_leaves_water_at_oil_pressure():
        model = _model(pc_ow=None)
        p = np.array([10.0, 12.0, 11.0])
        out = get_flux_and_props_water_bo(model, p, np.full(3, 0.3), np.ones(3),
                                          model.operators.T, np.zeros(2))
        np.testing.assert_allclose(out[4], p)
        print("ok test_absent_capillary_pressure_leaves_water_at_oil_pressure")

    def test_negative_shrinkage_warns():
        model = _model()
        model.fluid.b_w = lambda p: -np.ones_like(np.asarray(value(p)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_flux_and_props_water_bo(model, np.ones(3), np.full(3, 0.2),
                                        np.ones(3), model.operators.T,
                                        np.zeros(2))
        assert any("Negative water" in str(w.message) for w in caught)
        print("ok test_negative_shrinkage_warns")

    test_water_pressure_is_oil_pressure_less_capillary_pressure()
    test_flux_is_upstream_weighted_against_the_potential()
    test_derivatives_survive_every_output()
    test_capillary_pressure_is_evaluated_per_saturation_region()
    test_absent_capillary_pressure_leaves_water_at_oil_pressure()
    test_negative_shrinkage_warns()
    print("\nAll tests passed!")
