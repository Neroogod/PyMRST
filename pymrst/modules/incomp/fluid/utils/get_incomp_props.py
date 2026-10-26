"""MRST ``incomp/fluid/utils/getIncompProps.m``: densities, relative
permeabilities and viscosities for the incompressible solvers, from either
an incompressible fluid (one with `properties`) or an AD fluid evaluated at
the mean pressure."""

from typing import Any, Tuple

import numpy as np


def get_incomp_props(state: Any, fluid: Any, nargout: int = 3) -> Tuple[np.ndarray, ...]:
    """MRST `[rho, kr, mu, dkr] = getIncompProps(state, fluid)`; `nargout`
    of 4 also returns `dkr`."""
    if hasattr(fluid, "properties"):
        return _props_legacy(state, fluid, nargout)
    return _props_ad(state, fluid, nargout)


def _props_legacy(state: Any, fluid: Any, nargout: int) -> Tuple[np.ndarray, ...]:
    mu, rho = fluid.properties(state)
    s = fluid.saturation(state)
    kr = np.atleast_2d(fluid.relperm(s))
    out = (np.asarray(rho, dtype=float).ravel(), kr, np.asarray(mu, dtype=float).ravel())
    if nargout < 4:
        return out
    return out + (np.atleast_2d(fluid.d_relperm(s)),)


def _props_ad(state: Any, fluid: Any, nargout: int) -> Tuple[np.ndarray, ...]:
    from pymrst.utils.adi import value
    from pymrst.utils.units import ATM
    p = _get(state, "pressure")
    p_avg = np.mean(p) if p is not None else 1*ATM
    rho = np.array([float(fluid.rhoWS*value(fluid.bW(p_avg))),
                    float(fluid.rhoOS*value(fluid.bO(p_avg)))])
    mu = np.column_stack([np.asarray(value(fluid.muW(p)), dtype=float),
                          np.asarray(value(fluid.muO(p)), dtype=float)])
    s = np.atleast_2d(np.asarray(_get(state, "s"), dtype=float))[:, 0]
    so = 1 - s
    krO_fn = fluid.krOW if hasattr(fluid, "krOW") else fluid.krO
    kr = np.column_stack([value(fluid.krW(s)), value(krO_fn(so))])
    if nargout < 4:
        return rho, kr, mu
    from pymrst.utils.adi import init_variables_adi
    s_ad = init_variables_adi(s)[0]
    krW, krO = fluid.krW(s_ad), krO_fn(1 - s_ad)
    dkr = np.column_stack([krW.jac[0].diagonal(), krO.jac[0].diagonal()])
    return rho, kr, mu, dkr


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


getIncompProps = get_incomp_props
__all__ = ["get_incomp_props", "getIncompProps"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.incomp.fluid import init_simple_fluid

    def test_incompressible_fluid():
        fluid = init_simple_fluid([1e-3, 5e-3], [1000.0, 700.0], [2.0, 2.0])
        state = SimpleNamespace(s=np.array([[0.5, 0.5]]), pressure=np.zeros(1))
        rho, kr, mu, dkr = get_incomp_props(state, fluid, nargout=4)
        assert np.allclose(rho, [1000, 700]) and np.allclose(mu, [1e-3, 5e-3])
        assert np.allclose(kr, [[0.25, 0.25]]) and np.allclose(dkr, [[1.0, 1.0]])
        print("ok test_incompressible_fluid")

    def test_ad_fluid_uses_the_mean_pressure_for_density():
        from pymrst.modules.ad_props import init_simple_adi_fluid
        fluid = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000, 700], n=[2, 2], phases="wo")
        state = {"pressure": np.array([1e7, 2e7]), "s": np.array([[0.5, 0.5], [1.0, 0.0]])}
        rho, kr, mu = get_incomp_props(state, fluid)
        assert np.allclose(rho, [1000, 700]) and mu.shape == (2, 2)
        assert np.allclose(kr, [[0.25, 0.25], [1.0, 0.0]])
        print("ok test_ad_fluid_uses_the_mean_pressure_for_density")

    test_incompressible_fluid()
    test_ad_fluid_uses_the_mean_pressure_for_density()
    print("\nAll tests passed!")
