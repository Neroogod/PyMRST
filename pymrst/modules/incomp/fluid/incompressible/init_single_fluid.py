"""MRST ``incomp/fluid/incompressible/initSingleFluid.m``: one incompressible
phase of constant viscosity `mu` and density `rho`.

There is no relative-permeability effect in a single phase: `relperm`
returns ones whatever the saturation, its derivatives zeros. As in MRST the
fluid also carries the phase-named fields - `rhoWS`, `bW`, `krW`, `muW` for
the default phase `'W'` - with the other two phases' functions returning
empty arrays.

`init_single_fluid_ad`, PyMRST's earlier version, is kept for its callers;
its relative permeability is the saturation itself.
"""

import numpy as np

from pymrst.modules.incomp.fluid import Fluid, init_single_fluid_ad

_PHASES = ("W", "O", "G")


def init_single_fluid(mu: float, rho: float, phase: str = "W") -> Fluid:
    """MRST `fluid = initSingleFluid('mu', mu, 'rho', rho, 'phase', phase)`."""
    assert np.size(mu) == 1 and np.size(rho) == 1, (
        "Function 'initSingleFluid' is supported for single phase only")
    first = phase[0].upper()
    if first not in _PHASES:
        raise ValueError(f"Phase name '{phase}' is unsupported.  Phase name must be one of "
                         "'W', 'O', or 'G'.")
    mu, rho = float(np.ravel(mu)[0]), float(np.ravel(rho)[0])
    fluid = Fluid(mu=np.array([mu]), rho=np.array([rho]), relperm=_relperm, d_relperm=_d_relperm)
    phases = [first] + [p for p in _PHASES if p != first]
    for k, name in enumerate(phases):
        setattr(fluid, f"rho{name}S", rho if k == 0 else np.zeros(0))
        setattr(fluid, f"b{name}", _ones_like if k == 0 else _missing)
        setattr(fluid, f"kr{name}", _ones_like if k == 0 else _missing)
        setattr(fluid, f"mu{name}", (lambda p, m=mu: m + 0*np.asarray(p, dtype=float))
                if k == 0 else _missing)
    return fluid


def _relperm(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    s = s.reshape(-1, 1) if s.ndim < 2 else s
    assert s.shape[1] == 1
    return np.ones(s.shape)


def _d_relperm(s: np.ndarray) -> np.ndarray:
    return np.zeros(_relperm(s).shape)


def _ones_like(x: np.ndarray) -> np.ndarray:
    return 1 + 0*np.asarray(x, dtype=float)


def _missing(x: np.ndarray) -> np.ndarray:
    return np.zeros(0)


initSingleFluid = init_single_fluid
__all__ = ["Fluid", "initSingleFluid", "init_single_fluid", "init_single_fluid_ad"]


if __name__ == "__main__":
    from pymrst.utils.units import CENTI, POISE

    def test_no_relative_permeability_effect():
        f = init_single_fluid(1*CENTI*POISE, 1000.0)
        s = np.array([[0.0], [0.3], [1.0]])
        assert np.array_equal(f.relperm(s), np.ones((3, 1)))
        assert np.array_equal(f.d_relperm(s), np.zeros((3, 1)))
        assert np.allclose(f.mobility(s), 1/(1*CENTI*POISE))
        print("ok test_no_relative_permeability_effect")

    def test_properties_and_phase_fields():
        f = init_single_fluid(2e-3, 800.0, phase="oil")
        mu, rho = f.properties()
        assert mu[0] == 2e-3 and rho[0] == 800.0 and f.rhoOS == 800.0
        assert np.allclose(f.muO(np.zeros(2)), 2e-3) and f.krW(np.ones(2)).size == 0
        print("ok test_properties_and_phase_fields")

    def test_unknown_phase_raises():
        try:
            init_single_fluid(1.0, 1.0, phase="X")
        except ValueError:
            print("ok test_unknown_phase_raises")
            return
        raise AssertionError("expected ValueError")

    test_no_relative_permeability_effect()
    test_properties_and_phase_fields()
    test_unknown_phase_raises()
    print("\nAll tests passed!")
