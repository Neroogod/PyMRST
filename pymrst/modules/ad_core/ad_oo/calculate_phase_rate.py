"""MRST ``ad-core/utils/calculatePhaseRate.m``.

Perforation phase rates `q_alpha = lambda_alpha T dp`. Producing perforations
take each phase at its own mobility; injecting ones take the total mobility
split by the well's injection composition - corrected for whatever
crossflowing perforations bring into the wellbore.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from pymrst.modules.ad_core.utils.cross_flow_mixture import cross_flow_mixture
from pymrst.utils.adi import value


def calculate_phase_rate(Tdp: Any, mobw: Sequence[Any], mapping: Any,
                         allow_cross_flow: bool, model: Any) -> List[Any]:
    """`q_alpha` per perforation; `mapping.perf2well` is 0-based."""
    perf2well = np.asarray(mapping.perf2well, dtype=np.int64)
    is_injector = np.asarray(mapping.isInjector, dtype=bool)[perf2well]
    W = mapping.W
    v_tdp = np.asarray(value(Tdp), dtype=float)
    injection = v_tdp > 0
    production = ~injection & (v_tdp != 0)
    crossflow = (injection & ~is_injector) | (production & is_injector)
    crossflow = crossflow & bool(allow_cross_flow)
    nph = len(mobw)
    mobw = list(mobw)
    if np.any(injection):
        compi = np.vstack([np.asarray(w.compi, dtype=float).ravel() for w in W])
        if np.any(crossflow):
            if getattr(model, "verbose", 0) and model.verbose > 1:
                print(f"Crossflow occuring in {int(crossflow.sum())} perforations")
            q_wb = np.column_stack([np.asarray(value(m), dtype=float) for m in mobw])*v_tdp[:, None]
            compi = cross_flow_mixture(q_wb, compi, mapping)
        compi_perf = compi[perf2well, :]
        rows = np.flatnonzero(injection)
        mobt = 0.0
        for i in range(nph):
            mobt = mobt + mobw[i][rows]
        for i in range(nph):
            updated = mobw[i] + 0.0
            updated[rows] = mobt*compi_perf[rows, i]
            mobw[i] = updated
    return [mobw[i]*Tdp for i in range(nph)]


calculatePhaseRate = calculate_phase_rate

__all__ = ["calculate_phase_rate", "calculatePhaseRate"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _map(sign):
        return SimpleNamespace(perf2well=np.array([0, 0]),
                               isInjector=np.array([sign > 0]),
                               W=[SimpleNamespace(compi=np.array([1.0, 0.0]))])

    def test_production_takes_each_phase_at_its_mobility():
        q = calculate_phase_rate(np.array([-1.0, -2.0]),
                                 [np.array([1.0, 1.0]), np.array([3.0, 3.0])],
                                 _map(-1), True, None)
        np.testing.assert_allclose(q[0], [-1.0, -2.0])
        np.testing.assert_allclose(q[1], [-3.0, -6.0])
        print("ok test_production_takes_each_phase_at_its_mobility")

    def test_injection_carries_the_declared_stream_at_total_mobility():
        q = calculate_phase_rate(np.array([1.0, 2.0]),
                                 [np.array([1.0, 1.0]), np.array([3.0, 3.0])],
                                 _map(1), True, None)
        np.testing.assert_allclose(q[0], [4.0, 8.0])
        np.testing.assert_allclose(q[1], [0.0, 0.0])
        print("ok test_injection_carries_the_declared_stream_at_total_mobility")

    test_production_takes_each_phase_at_its_mobility()
    test_injection_carries_the_declared_stream_at_total_mobility()
    print("\nAll tests passed!")
