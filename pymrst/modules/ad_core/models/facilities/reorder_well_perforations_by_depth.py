"""Reorder well perforations by depth (MRST ad-core/models/facilities).

Ports MRST's `reorderWellPerforationsByDepth.m` (AC14): sort each well's
perforations from shallow to deep, permuting the per-perforation arrays
(`cells`, `WI`, `dZ`, `cstatus`) together so the well's connection list is in
depth order. A multisegment well's wellbore column is accumulated segment by
segment down the well, so its perforations must be depth-ordered for the
hydrostatic column and the segment topology to line up; this is the utility
that guarantees it.

The reorder is a stable sort on ``dZ`` (the perforation depth relative to the
bhp datum) with the same permutation applied to every per-perforation field, so
each perforation's ``(cell, WI, dZ, cstatus)`` tuple stays together.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def reorder_well_perforations_by_depth(wells: Sequence[Any],
                                       active: Optional[Sequence[bool]] = None
                                       ) -> Sequence[Any]:
    """Sort each active well's perforations shallow-to-deep, in place.

    ``active`` optionally selects which wells to reorder (default all). Each
    well's ``cells``, ``WI``, ``dZ`` and ``cstatus`` are permuted by the same
    depth-ordering. Returns the (mutated) well list.
    """
    if active is None:
        active = [True] * len(wells)
    for i, w in enumerate(wells):
        if not active[i]:
            continue
        dz = np.asarray(w.dZ, dtype=float).ravel()
        order = np.argsort(dz, kind="stable")
        w.dZ = dz[order]
        w.cells = np.asarray(w.cells).ravel()[order]
        wi = np.asarray(w.WI, dtype=float).ravel()
        w.WI = wi[order] if wi.size == order.size else w.WI
        cstat = getattr(w, "cstatus", None)
        if cstat is not None:
            cstat = np.asarray(cstat).ravel()
            if cstat.size == order.size:
                w.cstatus = cstat[order]
    return wells


__all__ = ["reorder_well_perforations_by_depth"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params import add_well, make_rock

    def _well(cells, dz):
        g = compute_geometry(cart_grid([3, 1, 3], [30.0, 10.0, 30.0]))
        rock = make_rock(g, 1e-13, 0.25)
        w = add_well(None, g, rock, np.asarray(cells), well_type="bhp",
                     val=1e7, compi=np.array([1.0, 0.0]), sign=1)[0]
        w.dZ = np.asarray(dz, float)
        w.WI = np.arange(1.0, len(cells) + 1.0)          # distinct WIs to track
        w.cstatus = np.ones(len(cells), bool)
        return w

    def test_perforations_end_depth_sorted():
        """After reorder, dZ is ascending and cells/WI move with it."""
        w = _well([6, 0, 3], dz=[25.0, 5.0, 15.0])       # out of order
        reorder_well_perforations_by_depth([w])
        assert np.all(np.diff(w.dZ) >= 0), "dZ must be ascending"
        # the perforation that was at dZ=5 (cell 0, WI 2) is now first
        assert w.dZ[0] == 5.0 and w.cells[0] == 0 and w.WI[0] == 2.0
        assert w.dZ[-1] == 25.0 and w.cells[-1] == 6 and w.WI[-1] == 1.0
        print("ok test_perforations_end_depth_sorted")

    def test_tuple_stays_together():
        """Each (cell, WI, dZ) tuple is permuted consistently."""
        w = _well([6, 0, 3], dz=[25.0, 5.0, 15.0])
        pairs_before = set(zip(w.cells.tolist(), w.WI.tolist(),
                               w.dZ.tolist()))
        reorder_well_perforations_by_depth([w])
        pairs_after = set(zip(w.cells.tolist(), w.WI.tolist(), w.dZ.tolist()))
        assert pairs_before == pairs_after, "a field was permuted independently"
        print("ok test_tuple_stays_together")

    def test_already_sorted_is_unchanged():
        w = _well([0, 3, 6], dz=[5.0, 15.0, 25.0])
        reorder_well_perforations_by_depth([w])
        assert np.array_equal(w.cells, [0, 3, 6])
        print("ok test_already_sorted_is_unchanged")

    def test_active_mask_skips_wells():
        w = _well([6, 0, 3], dz=[25.0, 5.0, 15.0])
        reorder_well_perforations_by_depth([w], active=[False])
        assert np.array_equal(w.dZ, [25.0, 5.0, 15.0]), "inactive well touched"
        print("ok test_active_mask_skips_wells")

    test_perforations_end_depth_sorted()
    test_tuple_stays_together()
    test_already_sorted_is_unchanged()
    test_active_mask_skips_wells()
    print("\nAll tests passed!")
