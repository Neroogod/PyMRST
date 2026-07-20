"""MRST-style gas relative-permeability hysteresis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.utils.adi import value


@dataclass
class KilloughGasHysteresis:
    """Killough non-wetting gas scanning curve used by MRST ``EHYSTR``."""

    sg_max_primary: float
    sg_min_primary: float
    krg_drainage: callable
    krg_imbibition: callable
    sg_tmax: float | None = None
    min_sat: float = 0.05
    modifier: float = 0.1
    tol: float = 1e-3

    def __post_init__(self):
        if self.sg_tmax is None:
            self.sg_tmax = self.sg_max_primary

    def __call__(self, sg, sg_max):
        sg_v = np.asarray(value(sg), dtype=float)
        sg_i = np.asarray(value(sg_max), dtype=float)
        out = self.krg_drainage(sg)
        imb = (sg_v + self.tol < sg_i) & (sg_i > self.min_sat + self.sg_min_primary)
        if not np.any(imb):
            return out
        out_v = np.asarray(value(out), dtype=float).copy()
        sgiv = sg_i[imb]
        sgv = sg_v[imb]
        sgmx = self.sg_max_primary
        sgmn = self.sg_min_primary
        sgtmx = self.sg_tmax
        A = 1.0 + self.modifier * (sgmx - sgiv)
        C = 1.0 / (sgtmx - sgmn) - 1.0 / (sgmx - sgmn)
        sgt = sgmn + (sgiv - sgmn) / (A + C * (sgiv - sgmn))
        snorm = sgtmx + (sgv - sgt) * (sgmx - sgtmx) / (sgiv - sgt)
        scan = self.krg_imbibition(snorm) * self.krg_drainage(sgiv) / self.krg_drainage(sgmx)
        drain = self.krg_drainage(sgv)
        out_v[imb] = np.minimum(np.asarray(value(scan), float), np.asarray(value(drain), float))
        out_v[out_v < 0.0] = 0.0
        return out_v


def update_saturation_maximum(state, phase: str = "g"):
    """Update MRST hysteresis state variable ``sMax`` for a phase."""
    key = f"s{phase.upper()}"
    max_key = f"s{phase.upper()}Max"
    current = np.asarray(value(state[key]), dtype=float)
    old = np.asarray(state.get(max_key, current), dtype=float)
    state[max_key] = np.maximum(old, current)
    return state


__all__ = ["KilloughGasHysteresis", "update_saturation_maximum"]


if __name__ == "__main__":
    def test_killough_scan_is_below_drainage():
        krd = lambda s: np.asarray(s, dtype=float) ** 2
        kri = lambda s: 0.7 * np.asarray(s, dtype=float) ** 2
        h = KilloughGasHysteresis(1.0, 0.0, krd, kri, sg_tmax=1.0)
        sg = np.array([0.2, 0.6])
        smax = np.array([0.8, 0.6])
        out = h(sg, smax)
        assert out[0] <= krd(sg)[0]
        assert np.isclose(out[1], krd(sg)[1])
        print("ok test_killough_scan_is_below_drainage")

    test_killough_scan_is_below_drainage()
