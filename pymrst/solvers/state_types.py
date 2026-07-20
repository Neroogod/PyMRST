"""Reservoir and well solution state containers."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class WellSol:
    """Per-well solution: connection fluxes and bottom-hole pressure."""

    flux: np.ndarray
    pressure: float
    cdp: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class State:
    """Reservoir state: cell pressures, face fluxes, and saturations."""

    pressure: np.ndarray
    flux: np.ndarray
    s: np.ndarray
    facePressure: Optional[np.ndarray] = None
    wellSol: List[WellSol] = field(default_factory=list)


if __name__ == "__main__":

    def test_state_containers() -> None:
        state = State(pressure=np.ones(2), flux=np.zeros(3), s=np.ones((2, 1)))
        assert state.pressure.shape == (2,)
        assert state.flux.shape == (3,)
        assert state.wellSol == []
        print("ok test_state_containers")

    test_state_containers()
    print("\nAll tests passed!")

