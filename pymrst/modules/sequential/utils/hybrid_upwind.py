"""MRST ``sequential/utils/hybridUpwind.m``.

Hybrid upwinding: the viscous part follows the total flux, and the gravity
part is phase-potential upwinded with no total flux.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np

from pymrst.utils.adi import value


def hybrid_upwind(G: Sequence[Any], vT: Any, T: Any, K: Sequence[Any],
                  upstr: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[flag_v, flag_g] = hybridUpwind(G, vT, T, K, upstr)`."""
    from pymrst.modules.sequential.utils.multiphase_upwind_indices import (
        multiphase_upwind_indices,
    )
    vT = np.asarray(value(vT), dtype=float).ravel()
    flag_v = np.tile((vT > 0)[:, None], (1, len(G)))
    flag_g, _, _ = multiphase_upwind_indices(G, 0*vT, T, K, upstr)
    return flag_v, flag_g


hybridUpwind = hybrid_upwind

__all__ = ["hybridUpwind", "hybrid_upwind"]


if __name__ == "__main__":
    def test_viscous_flags_follow_the_total_flux():
        upstr = lambda flag, x: np.where(flag, x[0], x[1])
        flag_v, flag_g = hybrid_upwind([np.zeros(1), np.zeros(1)], np.array([-1.0]),
                                       np.array([1.0]), [np.ones(2), np.ones(2)], upstr)
        assert flag_v.tolist() == [[False, False]] and flag_g.shape == (1, 2)
        print("ok test_viscous_flags_follow_the_total_flux")

    test_viscous_flags_follow_the_total_flux()
    print("\nAll tests passed!")
