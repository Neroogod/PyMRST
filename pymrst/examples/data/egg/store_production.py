"""Accumulate well production history.

Port of MRST ``examples/data/Egg/Egg_Model_Data_Files_v2/MRST/StoreProduction.m``.

MRST's version is a script that appends to a `wellData` struct array in the
caller's workspace on every timestep. The port is a small class instead, since
mutating a caller's variables is not something Python should imitate.

The interesting part is the rate split. A producer's total flux is divided into
water and oil by the fractional flow evaluated at the mean saturation over the
well's cells:

    f_w = 1 / (1 + (kr_o/mu_o) / (kr_w/mu_w))

An injector is recorded entirely as water. MRST notes that the mean-saturation
step is strictly valid only in 2-D; the same caveat applies here, and the
docstring keeps it rather than quietly generalising.

Rates are converted to volume per day, matching MRST's `*86400`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

SECONDS_PER_DAY = 86400.0


@dataclass
class WellHistory:
    """Bottom-hole pressure and phase rates for one well, over time."""

    name: str
    bhp: list[float] = field(default_factory=lambda: [0.0])
    water: list[float] = field(default_factory=lambda: [0.0])
    oil: list[float] = field(default_factory=lambda: [0.0])

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"bhp": np.asarray(self.bhp, dtype=float),
                "water": np.asarray(self.water, dtype=float),
                "oil": np.asarray(self.oil, dtype=float)}


class ProductionRecorder:
    """Collect per-well production as a simulation advances.

    Call ``record`` once per accepted timestep. MRST seeds every series with a
    leading zero, which is kept so the histories line up with its output.
    """

    def __init__(self, wells: Sequence[Any]):
        self.wells = list(wells)
        self.history = [WellHistory(name=getattr(w, "name", f"W{i + 1}"))
                        for i, w in enumerate(self.wells)]

    def record(self, state: Any, fluid: Any) -> None:
        """Append one timestep for every well."""
        for index, well in enumerate(self.wells):
            entry = self.history[index]
            entry.bhp.append(float(_well_pressure(state, index)))
            rate = float(np.sum(_well_flux(state, index)))
            if rate < 0.0:
                f_w = _fractional_flow(state, fluid, well)
                entry.water.append(f_w * rate * SECONDS_PER_DAY)
                entry.oil.append((1.0 - f_w) * rate * SECONDS_PER_DAY)
            else:
                # An injector's stream is water by definition here, so no
                # fractional-flow split applies.
                entry.water.append(rate * SECONDS_PER_DAY)
                entry.oil.append(0.0)

    def as_dict(self) -> dict[str, dict[str, np.ndarray]]:
        return {h.name: h.as_arrays() for h in self.history}


def _fractional_flow(state, fluid, well) -> float:
    """Water fractional flow at the mean saturation over a well's cells.

    Returns 0.0 when the mobilities are degenerate, matching MRST's
    ``if isnan(fw), fw = 0; end``.
    """
    cells = np.asarray(well.cells, dtype=np.int64)
    s = np.asarray(state.s, dtype=float)
    s_mean = float(np.mean(s[cells, 0])) if s.ndim > 1 else float(np.mean(s[cells]))

    kr = np.atleast_1d(np.asarray(fluid.relperm(s_mean), dtype=float)).ravel()
    mu = np.atleast_1d(np.asarray(fluid.properties(s_mean), dtype=float)).ravel()
    if kr.size < 2 or mu.size < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mob_w, mob_o = kr[0] / mu[0], kr[1] / mu[1]
        f_w = 1.0 / (1.0 + mob_o / mob_w)
    return 0.0 if not np.isfinite(f_w) else float(f_w)


def _well_pressure(state, index: int) -> float:
    sol = getattr(state, "wellSol", None)
    if sol is not None and len(sol) > index:
        return float(getattr(sol[index], "pressure", np.asarray(sol[index]).ravel()[0]))
    bhp = getattr(state, "bhp", None)
    return float(np.asarray(bhp).ravel()[index]) if bhp is not None else 0.0


def _well_flux(state, index: int) -> np.ndarray:
    sol = getattr(state, "wellSol", None)
    if sol is not None and len(sol) > index:
        return np.atleast_1d(np.asarray(getattr(sol[index], "flux", 0.0), dtype=float))
    return np.zeros(1)


__all__ = ["ProductionRecorder", "WellHistory"]


if __name__ == "__main__":
    from types import SimpleNamespace

    class _Fluid:
        """Quadratic Corey relperms with a 1:5 viscosity ratio."""

        def relperm(self, s):
            s = np.asarray(s, dtype=float)
            return np.array([s ** 2, (1.0 - s) ** 2])

        def properties(self, s):
            return np.array([1e-3, 5e-3])

    def _state(sw, flux):
        n = 4
        s = np.column_stack([np.full(n, sw), np.full(n, 1.0 - sw)])
        wells = [SimpleNamespace(pressure=200e5, flux=np.array([flux]))]
        return SimpleNamespace(s=s, wellSol=wells)

    def _wells():
        return [SimpleNamespace(name="P1", cells=np.arange(4))]

    def test_history_starts_at_zero_like_mrst():
        rec = ProductionRecorder(_wells())
        assert rec.history[0].bhp == [0.0]
        assert rec.history[0].water == [0.0] and rec.history[0].oil == [0.0]
        print("ok test_history_starts_at_zero_like_mrst")

    def test_injection_is_recorded_entirely_as_water():
        """A positive rate is injection, and no fractional-flow split applies."""
        rec = ProductionRecorder(_wells())
        rec.record(_state(0.5, +1.0), _Fluid())
        assert np.isclose(rec.history[0].water[-1], SECONDS_PER_DAY)
        assert rec.history[0].oil[-1] == 0.0
        print("ok test_injection_is_recorded_entirely_as_water")

    def test_production_splits_by_fractional_flow():
        """At sw = 0.5 the mobility ratio makes water the larger share.

        krw/muw = 0.25/1e-3 = 250; kro/muo = 0.25/5e-3 = 50, so
        f_w = 1/(1 + 50/250) = 0.8333.
        """
        rec = ProductionRecorder(_wells())
        rec.record(_state(0.5, -1.0), _Fluid())
        total = rec.history[0].water[-1] + rec.history[0].oil[-1]
        assert np.isclose(total, -SECONDS_PER_DAY), total
        f_w = rec.history[0].water[-1] / total
        assert np.isclose(f_w, 1.0 / (1.0 + 50.0 / 250.0)), f_w
        print(f"ok test_production_splits_by_fractional_flow (f_w = {f_w:.4f})")

    def test_dry_oil_produces_no_water():
        """At sw = 0, krw is zero, so the split must give all oil.

        The mobility ratio is infinite here, which is the case MRST guards with
        `if isnan(fw), fw = 0`.
        """
        rec = ProductionRecorder(_wells())
        rec.record(_state(0.0, -1.0), _Fluid())
        assert np.isclose(rec.history[0].water[-1], 0.0), rec.history[0].water[-1]
        assert np.isclose(rec.history[0].oil[-1], -SECONDS_PER_DAY)
        print("ok test_dry_oil_produces_no_water")

    def test_rates_accumulate_over_steps():
        rec = ProductionRecorder(_wells())
        for _ in range(3):
            rec.record(_state(0.5, -1.0), _Fluid())
        out = rec.as_dict()["P1"]
        assert out["water"].size == 4, out["water"].size   # seed plus three steps
        assert np.all(out["water"][1:] < 0.0)
        print("ok test_rates_accumulate_over_steps")

    def test_bhp_is_captured():
        rec = ProductionRecorder(_wells())
        rec.record(_state(0.5, -1.0), _Fluid())
        assert np.isclose(rec.history[0].bhp[-1], 200e5)
        print("ok test_bhp_is_captured")

    test_history_starts_at_zero_like_mrst()
    test_injection_is_recorded_entirely_as_water()
    test_production_splits_by_fractional_flow()
    test_dry_oil_produces_no_water()
    test_rates_accumulate_over_steps()
    test_bhp_is_captured()
    print("\nAll tests passed!")
