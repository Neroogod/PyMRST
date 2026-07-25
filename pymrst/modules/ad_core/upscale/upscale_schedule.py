"""Coarsen a schedule's wells onto a coarse grid (MRST `upscaleSchedule`).

A schedule points its wells at fine cells; on the coarse grid those perforations
have to be re-expressed in coarse blocks. Several fine perforations can collapse
into one coarse block, so the well's cell list becomes the *unique* coarse blocks
it touches, and the collapsed connections' well indices are combined. Parallel
flow paths into the same block add, so the default here sums the well index
(`sum`) - the conductance-preserving choice; `harmonic` and `mean` are available
too. MRST's default is `recompute` (a fresh `computeWellIndex` on the coarse
grid); that needs a coarse well-index routine for an arbitrary perforation
direction, so this port defaults to `sum`, which is well-defined from the fine
well index alone, and notes the difference.

Boundary-condition and source coarsening (the `handleBC`/`handleSRC` branches)
attach here when a consuming case needs them; the well path is what schedules in
this port use.
"""

import copy
from typing import Any, List, Tuple

import numpy as np

_METHODS = ("sum", "harmonic", "mean")


def _unique_stable(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unique values in first-appearance order, plus a map from each input to
    its position in that unique list (MRST `uniqueStable`)."""
    _, first_idx = np.unique(a, return_index=True)
    order = np.sort(first_idx)
    uniq = a[order]
    pos = {v: i for i, v in enumerate(uniq)}
    new_map = np.array([pos[v] for v in a], dtype=np.int64)
    return uniq, new_map


def upscale_well(coarse_model: Any, well: Any, method: str = "sum") -> Any:
    """Remap a single well onto the coarse grid, combining its well indices."""
    if method not in _METHODS:
        raise ValueError(f"unknown wellUpscaleMethod {method!r}; "
                         f"use one of {_METHODS}")
    partition = np.asarray(coarse_model.partition, dtype=np.int64)
    pc = partition[np.asarray(well.cells, dtype=np.int64)]
    coarse_cells, new_map = _unique_stable(pc)
    nc = coarse_cells.size
    counts = np.bincount(new_map, minlength=nc)

    wi = np.asarray(well.WI, dtype=float).ravel()
    if method == "sum":
        wi_c = np.bincount(new_map, weights=wi, minlength=nc)
    elif method == "mean":
        wi_c = np.bincount(new_map, weights=wi, minlength=nc) / counts
    else:  # harmonic
        wi_c = 1.0 / (np.bincount(new_map, weights=1.0 / wi, minlength=nc) / counts)

    wc = copy.copy(well)
    wc.cells = coarse_cells
    wc.WI = wi_c
    wc.cstatus = np.ones(nc, dtype=bool)
    # first-encountered perforation direction/radius, as MRST does
    _, first_ind = np.unique(pc, return_index=True)
    first_ind = np.sort(first_ind)
    if np.ndim(well.dir) and np.size(well.dir) > 1:
        wc.dir = np.asarray(well.dir)[first_ind]
    if np.ndim(well.r) and np.size(well.r) > 1:
        wc.r = np.asarray(well.r)[first_ind]
    z = coarse_model.G.cells.centroids[coarse_cells, 2] \
        if coarse_model.G.cells.centroids.shape[1] > 2 else np.zeros(nc)
    wc.dZ = z - well.refDepth
    wc.fperf = new_map
    return wc


def upscale_schedule(coarse_model: Any, schedule: Any, method: str = "sum") -> Any:
    """Return a copy of `schedule` with every control's wells coarsened."""
    coarse = copy.copy(schedule)
    controls = []
    for ctrl in schedule.controls:
        c = copy.copy(ctrl)
        if getattr(ctrl, "W", None):
            c.W = [upscale_well(coarse_model, w, method) for w in ctrl.W]
        controls.append(c)
    coarse.controls = controls
    return coarse


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
        partition_cart_grid,
    )
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _coarse():
        g = compute_geometry(cart_grid([4, 4, 1], [400.0, 400.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
        model = GenericBlackOilModel(g, rock, fluid, [])
        part = partition_cart_grid([4, 4, 1], [2, 2, 1])       # 2x2 blocks
        return model, upscale_model_tpfa(model, part), g, rock

    def test_perforations_collapse_to_unique_coarse_blocks():
        """A well perforating cells 0,1,4,5 - all in coarse block 0 of a 4x4 ->
        2x2 partition - collapses to a single coarse perforation."""
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1, 4, 5]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well(cm, w)
        assert wc.cells.size == 1, wc.cells
        assert wc.cells[0] == cm.partition[0]
        print("ok test_perforations_collapse_to_unique_coarse_blocks")

    def test_well_index_sums_over_collapsed_perforations():
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1, 4, 5]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well(cm, w, method="sum")
        assert np.isclose(wc.WI[0], np.sum(w.WI)), (wc.WI, w.WI)
        # cstatus and dZ have the coarse length
        assert wc.cstatus.size == 1 and wc.dZ.size == 1
        print("ok test_well_index_sums_over_collapsed_perforations")

    def test_two_blocks_stay_separate():
        """A well crossing two coarse blocks keeps two coarse perforations."""
        model, cm, g, rock = _coarse()
        # cell 0 -> block 0, cell 2 -> block 1 (columns 0-1 vs 2-3)
        w = add_well(None, g, rock, np.array([0, 2]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well(cm, w)
        assert wc.cells.size == 2
        assert set(wc.cells.tolist()) == {cm.partition[0], cm.partition[2]}
        print("ok test_two_blocks_stay_separate")

    def test_harmonic_and_mean_methods():
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wi = np.asarray(w.WI, float)
        wm = upscale_well(cm, w, method="mean")
        wh = upscale_well(cm, w, method="harmonic")
        assert np.isclose(wm.WI[0], wi.mean())
        assert np.isclose(wh.WI[0], 1.0 / np.mean(1.0 / wi))
        print("ok test_harmonic_and_mean_methods")

    test_perforations_collapse_to_unique_coarse_blocks()
    test_well_index_sums_over_collapsed_perforations()
    test_two_blocks_stay_separate()
    test_harmonic_and_mean_methods()
    print("\nAll tests passed!")
