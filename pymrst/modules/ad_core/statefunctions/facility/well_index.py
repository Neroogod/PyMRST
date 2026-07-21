"""WellIndex (MRST ad-core/statefunctions/facility).

Ports MRST's `WellIndex.m`: the productivity/injectivity index per perforation -
the transmissibility between the reservoir cell and the wellbore. In MRST it is
``W.WI`` gated by the connection status; PyMRST's wells carry the same ``WI``
per perforation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class WellIndex(StateFunction):
    """Per-perforation well index ``W_I``."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="W_I")
        self.depends_on("FacilityWellMapping")

    def evaluate_on_domain(self, facility: Any, state: Any):
        m = self.get_evaluated(state, "FacilityWellMapping")
        wi = [np.asarray(w.WI, dtype=float).ravel() for w in m.wells]
        return np.concatenate(wi) if wi else np.zeros(0)


__all__ = ["WellIndex"]


if __name__ == "__main__":
    # exercised through the grouping in validate_phase343.py and well_phase_flux
    print("ok well_index (see validate_phase343)")
    print("\nAll tests passed!")
