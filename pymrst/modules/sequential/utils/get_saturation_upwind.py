"""MRST ``sequential/utils/getSaturationUpwind.m``.

Upwind flags for transport, viscous and gravity, by type: 'potential'
(phase-potential upwinding for both), 'hybrid'/'hybrid_combined'
(`hybridUpwind`) or 'static' (`state.upstreamFlag`).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np


def get_saturation_upwind(upwtype: str, state: Any, G: Sequence[Any], vT: Any, T: Any,
                          K: Sequence[Any], upstr: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[flag_v, flag_g] = getSaturationUpwind(upwtype, state, G, vT, T, K, upstr)`."""
    kind = upwtype.lower()
    if kind == "potential":
        from pymrst.modules.sequential.utils.multiphase_upwind_indices import (
            multiphase_upwind_indices,
        )
        flag, _, _ = multiphase_upwind_indices(G, vT, T, K, upstr)
        return flag, flag
    if kind in ("hybrid", "hybrid_combined"):
        from pymrst.modules.sequential.utils.hybrid_upwind import hybrid_upwind
        return hybrid_upwind(G, vT, T, K, upstr)
    if kind == "static":
        flag = np.asarray(state["upstreamFlag"])
        return flag, flag
    raise ValueError(f"Unknown upwind type: '{upwtype}'")


getSaturationUpwind = get_saturation_upwind

__all__ = ["getSaturationUpwind", "get_saturation_upwind"]


if __name__ == "__main__":
    def test_the_static_flags_come_from_the_state():
        flag = np.array([[True, False]])
        fv, fg = get_saturation_upwind("static", {"upstreamFlag": flag}, None, None, None,
                                       None, None)
        assert fv is fg and fv.tolist() == [[True, False]]
        try:
            get_saturation_upwind("other", {}, None, None, None, None, None)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown type must be refused")
        print("ok test_the_static_flags_come_from_the_state")

    test_the_static_flags_come_from_the_state()
    print("\nAll tests passed!")
