"""MRST ``ad-blackoil/utils/getFluxAndPropsOil_BO.m``.

Oil phase flux and properties for the black-oil equations. With dissolved gas
the oil's shrinkage factor, viscosity and density depend on `rs` and on whether
the cell is saturated; without it they depend on pressure alone, and the
viscosity may come from the tabulated product `bO*muO` (`BOxmuO`). Unlike the
water and gas functions, MRST evaluates the oil PVT here without region
dispatch, and so does this port.
"""

from typing import Any, Optional, Tuple
import warnings

import numpy as np

from pymrst.modules.ad_core.utils.split_face_cell_value import split_face_cell_value
from pymrst.utils.adi import value


def get_flux_and_props_oil_bo(model: Any, p: Any, s_o: Any, kr_o: Any,
                              transmissibility: Any, gdz: Any,
                              rs: Optional[Any] = None,
                              is_saturated: Optional[Any] = None) -> Tuple:
    """Return `(vO, bO, mobO, rhoO, p, upco, dpO, muO)` as MRST does."""
    disgas = bool(getattr(model, "disgas", False))
    if rs is None:
        if disgas:
            raise ValueError("RS and saturated flag must be supplied for disgas model")
        rs = 0.0
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    ops = model.operators
    rho_s = np.asarray(fluid.rhoS, dtype=float).ravel()

    if disgas:
        b_o = fluid.b_o(p, rs, is_saturated)
        mu_o = fluid.mu_o(p, rs, is_saturated)
        rho_o = b_o*(rs*rho_s[2] + rho_s[1])
    else:
        b_o = fluid.b_o(p)
        bo_x_mu_o = getattr(fluid, "bo_x_mu_o", getattr(fluid, "BOxmuO", None))
        mu_o = bo_x_mu_o(p)*b_o if bo_x_mu_o is not None else fluid.mu_o(p)
        rho_o = b_o*rho_s[1]

    if np.any(np.asarray(value(b_o)) < 0):
        warnings.warn("Negative oil compressibility present!", RuntimeWarning,
                      stacklevel=2)

    rho_of = ops.avg(rho_o)
    dp_o = ops.grad(p) - rho_of*gdz
    upco = np.asarray(value(dp_o)) <= 0
    grid = model.G if hasattr(model, "G") else model.g
    sizes = (ops.T.size, grid.cells.num)
    kr_of, kr_o = split_face_cell_value(ops, upco, kr_o, sizes)
    mu_of, mu_o = split_face_cell_value(ops, upco, mu_o, sizes)
    mob_o = kr_o/mu_o
    v_o = -(kr_of/mu_of)*transmissibility*dp_o
    return v_o, b_o, mob_o, rho_o, p, upco, dp_o, mu_o


getFluxAndPropsOil_BO = get_flux_and_props_oil_bo

__all__ = ["get_flux_and_props_oil_bo", "getFluxAndPropsOil_BO"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi

    def _model(disgas: bool, **fluid_fields):
        grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        rock = make_rock(grid, 1.0, 0.2)
        fluid = SimpleNamespace(rhoS=np.array([1000.0, 700.0, 2.0]), **fluid_fields)
        return SimpleNamespace(G=grid, rock=rock, fluid=fluid, disgas=disgas,
                               operators=setup_operators_tpfa(grid, rock))

    def _live_model():
        return _model(True,
                      b_o=lambda p, rs, sat: 1.2 + 0.02*p + 0.03*rs + 0.04*sat,
                      mu_o=lambda p, rs, sat: 3.0 + 0.002*p + 0.01*rs)

    P = np.array([10.0, 12.0, 11.0])
    RS = np.array([2.0, 2.5, 3.0])
    SAT = np.array([True, False, True])

    def test_live_oil_density_carries_its_dissolved_gas():
        model = _live_model()
        out = get_flux_and_props_oil_bo(model, P, None, np.ones(3),
                                        model.operators.T, np.zeros(2), RS, SAT)
        b_o = 1.2 + 0.02*P + 0.03*RS + 0.04*SAT
        np.testing.assert_allclose(out[1], b_o)
        np.testing.assert_allclose(out[3], b_o*(RS*2.0 + 700.0))
        print("ok test_live_oil_density_carries_its_dissolved_gas")

    def test_live_oil_without_rs_is_refused():
        try:
            get_flux_and_props_oil_bo(_live_model(), P, None, np.ones(3),
                                      np.ones(2), np.zeros(2))
            raise SystemExit("should have raised")
        except ValueError:
            pass
        print("ok test_live_oil_without_rs_is_refused")

    def test_dead_oil_viscosity_comes_from_the_bo_mu_product():
        model = _model(False, b_o=lambda p: 1.0 + 0.01*p,
                       BOxmuO=lambda p: 4.0 + 0.0*p)
        out = get_flux_and_props_oil_bo(model, P, None, np.ones(3),
                                        model.operators.T, np.zeros(2))
        b_o = 1.0 + 0.01*P
        np.testing.assert_allclose(out[7], 4.0*b_o)
        np.testing.assert_allclose(out[3], 700.0*b_o)
        print("ok test_dead_oil_viscosity_comes_from_the_bo_mu_product")

    def test_dead_oil_without_the_product_uses_mu_o():
        model = _model(False, b_o=lambda p: 1.0 + 0.0*p,
                       mu_o=lambda p: 5.0 + 0.0*p)
        out = get_flux_and_props_oil_bo(model, P, None, np.ones(3),
                                        model.operators.T, np.zeros(2))
        np.testing.assert_allclose(out[7], 5.0)
        print("ok test_dead_oil_without_the_product_uses_mu_o")

    def test_flux_points_down_the_potential_and_keeps_derivatives():
        model = _live_model()
        p, rs = init_variables_adi(P, RS)
        v, _, mob, _, _, up, dp, _ = get_flux_and_props_oil_bo(
            model, p, None, np.ones(3), model.operators.T, np.zeros(2), rs, SAT)
        np.testing.assert_array_equal(up, value(dp) <= 0)
        assert np.all(value(v)*value(dp) <= 0)
        assert len(v.jac) == 2 and len(mob.jac) == 2
        print("ok test_flux_points_down_the_potential_and_keeps_derivatives")

    test_live_oil_density_carries_its_dissolved_gas()
    test_live_oil_without_rs_is_refused()
    test_dead_oil_viscosity_comes_from_the_bo_mu_product()
    test_dead_oil_without_the_product_uses_mu_o()
    test_flux_points_down_the_potential_and_keeps_derivatives()
    print("\nAll tests passed!")
