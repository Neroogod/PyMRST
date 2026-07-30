"""Compact end-to-end counterpart of MRST's conference ``mrst_demo``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_core.examples.ad_buckley_leverett_1d import (
    run_ad_buckley_leverett_1d,
)


@dataclass(frozen=True)
class MRSTDemoResult:
    cell_count: int
    state_count: int
    inlet_saturation: float
    outlet_saturation: float
    swept_cells: int
    monotonicity_error: float


def mrst_demo(*, timesteps: int = 10) -> MRSTDemoResult:
    """Build and simulate a complete two-phase AD displacement problem."""

    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    grid, states = run_ad_buckley_leverett_1d(n=timesteps)
    saturation = np.asarray(states[-1]["sW"], dtype=float)
    return MRSTDemoResult(
        cell_count=int(grid.cells.num),
        state_count=len(states),
        inlet_saturation=float(saturation[0]),
        outlet_saturation=float(saturation[-1]),
        swept_cells=int(np.count_nonzero(saturation > 1e-3)),
        monotonicity_error=float(
            np.max(np.maximum(np.diff(saturation), 0.0), initial=0.0)
        ),
    )


__all__ = ["MRSTDemoResult", "mrst_demo"]
