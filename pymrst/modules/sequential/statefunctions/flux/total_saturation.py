"""MRST ``sequential/statefunctions/flux/TotalSaturation.m``.

The total saturation `sT`, a primary variable of the transport model's
total-saturation formulation, read from the state.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import STATE, StateFunction


class TotalSaturation(StateFunction):
    """MRST `TotalSaturation`: `state.sT`."""

    def __init__(self, model: Any, **options: Any):
        super().__init__(model)
        self.depends_on("s", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        return state["sT"]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["TotalSaturation"]


if __name__ == "__main__":
    import numpy as np

    def test_it_reads_the_total_saturation():
        sT = np.array([1.0, 0.9])
        assert TotalSaturation(None).evaluate_on_domain(None, {"sT": sT}) is sT
        print("ok test_it_reads_the_total_saturation")

    test_it_reads_the_total_saturation()
    print("\nAll tests passed!")
