"""MRST ``diagnostics/utils/computeFandPhiFromDist.m``: flow capacity `F` and
storage capacity `Phi` from residence-time distributions.

For each well pair (column of `RTD.t` and `RTD.values`) the distribution is
integrated with the midpoint rule: `F` is the cumulative flux and `Phi` the
cumulative volume, scaled by the injector flux and the pair's allocation and
normalised - `F` by the allocation and `Phi` by its end value, or, for an RTD
from `estimateRTD` (with a warning that the result is inaccurate), `F` by
its end value and `Phi` by the pair volume. Both start at 0 and end at 1.
With `sum`, the pairs are added into one curve. `RTD.pairIx` is 0-based.
"""

import warnings
from typing import Any, Tuple

import numpy as np


def compute_f_and_phi_from_dist(RTD: Any, *, sum: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[F, Phi] = computeFandPhiFromDist(RTD, 'sum', sum)`."""
    if str(_get(RTD, "creator")).lower() == "estimatertd":
        warnings.warn("F-Phi diagrams computed from estimated RTD are highly inaccurate. "
                      "Proceed with caution\n")
        normalize_alloc, normalize_vol = True, False
    else:
        normalize_alloc, normalize_vol = False, True
    t_all = _columns(_get(RTD, "t"))
    values = _columns(_get(RTD, "values"))
    if sum:
        t = t_all[:, :1].copy()
        vals = values.sum(axis=1, keepdims=True)
        vol = np.atleast_1d(np.sum(_get(RTD, "volumes")))
        alloc = np.atleast_1d(np.sum(_get(RTD, "allocations")))
        injflux = np.atleast_1d(np.sum(_get(RTD, "injectorFlux")))
    else:
        t = t_all.copy()
        vals = values.copy()
        vol = np.ravel(_get(RTD, "volumes")).astype(float)
        alloc = np.ravel(_get(RTD, "allocations")).astype(float)
        pair_ix = np.atleast_2d(np.asarray(_get(RTD, "pairIx"), dtype=np.int64))
        injflux = np.ravel(_get(RTD, "injectorFlux")).astype(float)[pair_ix[:, 0]]
    t[np.isnan(t)] = 0
    vals[np.isnan(vals)] = 0
    mt, mvals = 0.5*t[:-1] + 0.5*t[1:], vals[1:]
    dt = np.diff(t, axis=0)
    F = np.cumsum(dt*mvals, axis=0)*injflux
    F = F/F[-1] if normalize_alloc else F/alloc
    Phi = np.cumsum(dt*mvals*mt, axis=0)*alloc
    Phi = Phi/Phi[-1] if normalize_vol else Phi/vol
    one, zero = np.ones((1, vol.size)), np.zeros((1, vol.size))
    return np.vstack([zero, F, one]), np.vstack([zero, Phi, one])


def _columns(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim < 2 else x


def _get(rtd: Any, name: str) -> Any:
    return rtd[name] if isinstance(rtd, dict) else getattr(rtd, name)


computeFandPhiFromDist = compute_f_and_phi_from_dist
__all__ = ["compute_f_and_phi_from_dist", "computeFandPhiFromDist"]


if __name__ == "__main__":
    def _rtd(**extra):
        t = np.linspace(0, 10, 41)[:, None]*np.array([1.0, 2.0])
        rtd = {"creator": "computeRTD", "t": t, "values": np.exp(-t)*[1.0, 0.5],
               "volumes": [3.0, 4.0], "allocations": [1.0, 2.0], "injectorFlux": [5.0, 6.0],
               "pairIx": np.array([[0, 0], [1, 0]])}
        rtd.update(extra)
        return rtd

    def test_curves_run_from_zero_to_one_per_pair():
        F, Phi = compute_f_and_phi_from_dist(_rtd())
        assert F.shape == Phi.shape == (42, 2)
        assert np.all(F[0] == 0) and np.all(F[-1] == 1) and np.all(Phi[-1] == 1)
        assert np.allclose(Phi[-2], 1.0)
        print("ok test_curves_run_from_zero_to_one_per_pair")

    def test_flux_is_scaled_by_injector_flux_over_allocation():
        F, _ = compute_f_and_phi_from_dist(_rtd())
        t = np.linspace(0, 10, 41)
        expected = np.cumsum(np.diff(t)*np.exp(-t[1:]))*5.0/1.0
        assert np.allclose(F[1:-1, 0], expected)
        print("ok test_flux_is_scaled_by_injector_flux_over_allocation")

    def test_sum_gives_one_curve_and_estimated_rtd_warns():
        t = np.linspace(0, 10, 41)[:, None]*np.ones(2)
        F, Phi = compute_f_and_phi_from_dist(_rtd(t=t, values=np.exp(-t)), sum=True)
        assert F.shape == (42, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            F, _ = compute_f_and_phi_from_dist(_rtd(creator="estimateRTD"))
        assert caught and np.allclose(F[-2], 1.0)
        print("ok test_sum_gives_one_curve_and_estimated_rtd_warns")

    test_curves_run_from_zero_to_one_per_pair()
    test_flux_is_scaled_by_injector_flux_over_allocation()
    test_sum_gives_one_curve_and_estimated_rtd_warns()
    print("\nAll tests passed!")
