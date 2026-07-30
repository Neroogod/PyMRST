"""Data structures for MRST `params/wells_and_bc` ports."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BoundaryCondition:
    """External boundary conditions; empty arrays mean all no-flow."""

    face: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    type: List[str] = field(default_factory=list)
    value: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # Overall mole fractions for compositional inflow (MRST `bc.components`).
    components: Optional[np.ndarray] = None
    # Optional properties of the fluid outside the domain.  MRST uses these
    # for pressure and reservoir-flux boundaries; when absent, the inside
    # density/mobility is mirrored.
    mob: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    # Optional injected dissolution ratios, either (phase, phase) or
    # (boundary, phase, phase).
    dissolution: Optional[np.ndarray] = None


@dataclass
class Source:
    """Explicit volumetric cell sources, injection positive."""

    cell: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # Overall mole fractions for compositional inflow (MRST-like extension).
    components: Optional[np.ndarray] = None
    dissolution: Optional[np.ndarray] = None


@dataclass
class Well:
    """One well with perforations, control, and Peaceman indices."""

    cells: np.ndarray
    type: str
    val: float
    r: np.ndarray
    dir: str
    WI: np.ndarray
    dZ: np.ndarray
    name: str
    compi: np.ndarray
    refDepth: float
    sign: int = 0
    status: bool = True
    # MRST `W.cstatus`: one active/open flag per perforation. `None` means all
    # completions are active, matching legacy PyMRST wells.
    cstatus: Optional[np.ndarray] = None
    # Overall mole fractions injected (MRST `W.components`). When set, the
    # injection mass split per component comes from here; otherwise `compi`
    # alone defines a single-phase injection stream.
    components: Optional[np.ndarray] = None
    # Optional MRST facility extras. `vfp_table` closes THP/VFP-controlled
    # wells through a tabulated BHP; `multisegment` stores explicit segment
    # topology for workflows that need more detail than a perforation column.
    vfp_table: Optional[Any] = None
    vfp_index: Optional[int] = None
    thp: Optional[float] = None
    multisegment: Optional[Any] = None
    # Operating limits (MRST's W.lims). A well runs on one control until it
    # violates a limit, then switches to the limit it hit - which is how real
    # schedules are written: "inject at 100 m3/day, but never above 300 bar".
    # Keys: 'bhp', 'rate' for injectors; 'bhp', 'orat', 'lrat', 'grat',
    # 'wrat', 'vrat', 'resv_history' for producers. Absent keys are never
    # violated.
    lims: Optional[Dict[str, float]] = None


if __name__ == "__main__":

    def test_empty_boundary_condition() -> None:
        bc = BoundaryCondition()
        assert bc.face.size == 0
        assert bc.type == []
        print("ok test_empty_boundary_condition")

    def test_empty_source() -> None:
        src = Source()
        assert src.cell.size == 0
        assert src.sat.shape == (0, 0)
        print("ok test_empty_source")

    test_empty_boundary_condition()
    test_empty_source()
    print("\nAll tests passed!")
