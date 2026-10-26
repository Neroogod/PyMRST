"""MRST ``diagnostics/utils/computeFandPhi.m``: the flow-capacity /
storage-capacity (F-Phi) diagram.

Cells are sorted by total travel time (forward plus backward time of flight,
MATLAB's stable sort); `Phi` is their cumulative pore volume and `F` the
cumulative flux `pv/tau` each carries under incompressible flow, both
normalised and starting at 0. Given a residence-time distribution instead,
the curves come from `computeFandPhiFromDist`.

`compute_f_phi`, below, is PyMRST's earlier version, kept for its callers:
it takes the total time of flight directly and guards zero times.
"""

from typing import Any, Tuple

import numpy as np


def compute_f_and_phi(arg1: Any, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[F, Phi] = computeFandPhi(pv, tof)` or `computeFandPhi(RTD, ...)`."""
    if isinstance(arg1, dict) or hasattr(arg1, "creator"):
        from pymrst.modules.diagnostics.utils.compute_f_and_phi_from_dist import (
            compute_f_and_phi_from_dist,
        )
        return compute_f_and_phi_from_dist(arg1, *args, **kwargs)
    pv = np.asarray(arg1, dtype=float).ravel()
    tof = np.asarray(args[0], dtype=float)
    assert tof.ndim == 2 and tof.shape[1] == 2, "Tof input must have two columns."
    t = tof.sum(axis=1)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    v = pv[order]
    Phi = np.cumsum(v)
    vt = Phi[-1]
    Phi = np.concatenate([[0.0], Phi/vt])
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = v/ts
        ff = np.cumsum(flux)
        ft = ff[-1]
        F = np.concatenate([[0.0], ff/ft])
    return F, Phi


computeFandPhi = compute_f_and_phi


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def compute_f_phi(
    pv: np.ndarray,
    tof_total: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flow capacity ``F`` against storage capacity ``Phi``."""
    order = np.argsort(tof_total)
    t = tof_total[order]
    v = pv[order]
    phi = np.cumsum(v)
    phi = np.concatenate([[0.0], phi / phi[-1]])
    flux = v / np.maximum(t, 1e-300)
    ff = np.cumsum(flux)
    f = np.concatenate([[0.0], ff / ff[-1]])
    return f, phi


__all__ = ["computeFandPhi", "compute_f_and_phi", "compute_f_phi"]


if __name__ == "__main__":
    def test_homogeneous_travel_times_give_the_diagonal():
        """Equal pore volumes and equal travel times: F = Phi, a Lorenz of 0."""
        F, Phi = compute_f_and_phi(np.ones(4), np.full((4, 2), 0.5))
        assert np.allclose(F, Phi) and np.allclose(F, [0, 0.25, 0.5, 0.75, 1.0])
        print("ok test_homogeneous_travel_times_give_the_diagonal")

    def test_fast_cells_come_first():
        pv = np.array([1.0, 1.0])
        F, Phi = compute_f_and_phi(pv, np.array([[3.0, 1.0], [0.5, 0.5]]))
        # Cell 1 (total 1) before cell 0 (total 4): flux 1 then 0.25.
        assert np.allclose(F, [0, 0.8, 1.0]) and np.allclose(Phi, [0, 0.5, 1.0])
        print("ok test_fast_cells_come_first")

    def test_ties_keep_cell_order():
        """MATLAB's sort is stable; equal travel times keep their cell order."""
        pv = np.array([1.0, 2.0, 3.0])
        F, Phi = compute_f_and_phi(pv, np.ones((3, 2)))
        assert np.allclose(Phi, [0, 1/6, 3/6, 1.0])
        print("ok test_ties_keep_cell_order")

    def test_a_distribution_goes_to_from_dist():
        t = np.linspace(0, 1, 5)[:, None]
        F, _ = compute_f_and_phi({"creator": "computeRTD", "t": t, "values": np.ones_like(t),
                                  "volumes": [1.0], "allocations": [1.0], "injectorFlux": [1.0],
                                  "pairIx": np.array([[0, 0]])})
        assert F.shape == (6, 1)
        print("ok test_a_distribution_goes_to_from_dist")

    test_homogeneous_travel_times_give_the_diagonal()
    test_fast_cells_come_first()
    test_ties_keep_cell_order()
    test_a_distribution_goes_to_from_dist()
    print("\nAll tests passed!")
