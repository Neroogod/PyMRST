"""MRST ``incomp/computeIncompWellPressureDrop.m``: the hydrostatic pressure
drop from a well's reference depth to each perforation. An injector's
mixture density comes from its injected composition; a producer's is the
WI-weighted mean of the fractional-flow-weighted phase densities in its
cells."""

from typing import Any

import warnings

import numpy as np

_LAST_COUNTS: dict = {}


def compute_incomp_well_pressure_drop(W: Any, mob: np.ndarray, rho: np.ndarray, g: float) -> np.ndarray:
    """MRST `dp = computeIncompWellPressureDrop(W, mob, rho, g)`."""
    mob = np.atleast_2d(np.asarray(mob, dtype=float))
    rho = np.asarray(rho, dtype=float).ravel()
    np_ = rho.size
    compi = np.atleast_1d(np.asarray(W.compi, dtype=float)).ravel()
    _check_input(compi.size, np_, mob.shape[1])
    if _is_injector(W):
        i = slice(0, min(np_, compi.size))
        rho_mix = rho[i] @ compi[i]
    else:
        i = slice(0, min(np_, mob.shape[1]))
        mobw = mob[np.asarray(W.cells, dtype=np.int64)]
        f = mobw/mobw.sum(axis=1, keepdims=True)
        WI = np.asarray(W.WI, dtype=float).ravel()
        rho_mix = np.sum(WI*(f[:, i] @ rho[i]))/np.sum(WI)
    return g*(np.asarray(W.dZ, dtype=float).ravel()*rho_mix)


def _check_input(ni: int, np_: int, nm: int) -> None:
    """MRST's `check_input`: warn, once per change in the counts, when the
    injected composition, densities and mobilities disagree on the phases."""
    counts = {"NI": ni, "NP": np_, "NM": nm}
    changed = counts != _LAST_COUNTS
    _LAST_COUNTS.update(counts)
    if changed and (ni != np_ or nm != np_):
        n = min(ni, nm, np_)
        warnings.warn(f"Inconsistent Number of Phases.  Using {n} Phase{'' if n == 1 else 's'} "
                      f"(=min([{ni}, {np_}, {nm}])).")


def _is_injector(W: Any) -> bool:
    sign = getattr(W, "sign", None)
    by_sign = sign is not None and np.size(sign) > 0 and bool(np.all(np.asarray(sign) > 0))
    return by_sign or (str(W.type).lower() == "rate" and float(np.sum(W.val)) > 0)


computeIncompWellPressureDrop = compute_incomp_well_pressure_drop
__all__ = ["compute_incomp_well_pressure_drop", "computeIncompWellPressureDrop"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_injector_uses_its_composition():
        W = SimpleNamespace(sign=1, type="rate", val=1.0, compi=np.array([1.0, 0.0]),
                            cells=np.array([0, 1]), WI=np.ones(2), dZ=np.array([1.0, 2.0]))
        dp = compute_incomp_well_pressure_drop(W, np.ones((2, 2)), np.array([1000.0, 800.0]), 10.0)
        assert np.allclose(dp, [1e4, 2e4])
        print("ok test_injector_uses_its_composition")

    def test_producer_uses_fractional_flow_and_wi():
        W = SimpleNamespace(sign=-1, type="bhp", val=1.0, compi=np.array([1.0, 0.0]),
                            cells=np.array([0, 1]), WI=np.array([1.0, 3.0]), dZ=np.array([1.0, 1.0]))
        mob = np.array([[1.0, 0.0], [0.0, 1.0]])
        dp = compute_incomp_well_pressure_drop(W, mob, np.array([1000.0, 800.0]), 1.0)
        assert np.allclose(dp, (1*1000 + 3*800)/4)
        print("ok test_producer_uses_fractional_flow_and_wi")

    test_injector_uses_its_composition()
    test_producer_uses_fractional_flow_and_wi()
    print("\nAll tests passed!")
