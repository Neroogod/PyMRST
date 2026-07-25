"""Coarsen a reservoir state onto a coarse grid (MRST `upscaleState`).

When a fine model is upscaled to a coarse one, its state has to come along: each
coarse block takes a single pressure and saturation that stand in for its fine
cells. The right average is *pore-volume weighted* - a block's coarse saturation
is the total fluid volume of its fine cells divided by the block pore volume, and
its coarse pressure is the pore-volume-weighted fine pressure - so the coarse
state holds the same fluid in place as the fine one. Dissolved-gas ratios `rs`
and `rv` are weighted by the phase pore volume they live in (`sO*pv`, `sG*pv`),
matching MRST.

The compositional branch of MRST's `upscaleState` (coarsening EOS mole fractions
and re-flashing) is not ported - there is no compositional upscaling consumer in
this port - and coarse-face flux coarsening needs the coarse grid's fine-face
connectivity, which is noted where it would attach.
"""

from typing import Dict, Optional

import numpy as np


def coarsen_state(partition: np.ndarray, pv_fine: np.ndarray,
                  pv_coarse: np.ndarray, state: Dict,
                  oil_col: int = 1, gas_col: int = 2) -> Dict:
    """Pore-volume-weighted coarsening of a black-oil state.

    `partition` maps each fine cell to its coarse block (0-based). `state` holds
    `pressure` (n_fine,) and `s` (n_fine, nph); optional `rs`, `rv`, `T`.
    Returns the coarse state. `oil_col`/`gas_col` index the oil and gas
    saturation columns for the `rs`/`rv` weighting.
    """
    p = np.asarray(partition, dtype=np.int64).ravel()
    pvf = np.asarray(pv_fine, dtype=float).ravel()
    pvc = np.asarray(pv_coarse, dtype=float).ravel()
    nc = pvc.size

    s = np.atleast_2d(np.asarray(state["s"], dtype=float))
    if s.shape[0] == 1 and pvf.size > 1:
        s = s.T
    nph = s.shape[1]

    out: Dict = {}
    s_c = np.zeros((nc, nph))
    for i in range(nph):
        s_c[:, i] = np.bincount(p, weights=s[:, i] * pvf, minlength=nc) / pvc
    out["s"] = s_c

    pressure = np.asarray(state["pressure"], dtype=float).ravel()
    out["pressure"] = np.bincount(p, weights=pvf * pressure, minlength=nc) / pvc

    if "rs" in state and state["rs"] is not None:
        so_f = s[:, oil_col]
        so_c = s_c[:, oil_col]
        rs = np.asarray(state["rs"], dtype=float).ravel()
        num = np.bincount(p, weights=so_f * rs * pvf, minlength=nc)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs_c = num / (so_c * pvc)
        rs_c[~np.isfinite(rs_c)] = 0.0
        out["rs"] = rs_c

    if "rv" in state and state["rv"] is not None:
        sg_f = s[:, gas_col]
        sg_c = s_c[:, gas_col]
        rv = np.asarray(state["rv"], dtype=float).ravel()
        num = np.bincount(p, weights=sg_f * rv * pvf, minlength=nc)
        with np.errstate(divide="ignore", invalid="ignore"):
            rv_c = num / (sg_c * pvc)
        rv_c[~np.isfinite(rv_c)] = 0.0
        out["rv"] = rv_c

    if "T" in state and state["T"] is not None:
        counts = np.bincount(p, minlength=nc)
        T = np.asarray(state["T"], dtype=float).ravel()
        out["T"] = np.bincount(p, weights=T, minlength=nc) / counts

    return out


def upscale_state(coarse_model, model, state: Dict,
                  oil_col: int = 1, gas_col: int = 2) -> Dict:
    """MRST-signature wrapper: read the partition and pore volumes from the
    models and coarsen `state`. The coarse model must carry `partition` (set by
    `upscale_model_tpfa`) and its `operators.pv`.
    """
    partition = getattr(coarse_model, "partition", None)
    if partition is None:
        partition = coarse_model.G.partition
    return coarsen_state(partition, model.operators.pv,
                         coarse_model.operators.pv, state, oil_col, gas_col)


if __name__ == "__main__":
    def test_saturation_is_pore_volume_weighted():
        """Two fine cells into one block: the coarse saturation is the
        pv-weighted mean, not the plain mean."""
        partition = np.array([0, 0])
        pvf = np.array([1.0, 3.0])            # second cell holds 3x the volume
        pvc = np.array([4.0])
        state = {"pressure": np.array([100.0, 200.0]),
                 "s": np.array([[0.2, 0.8], [0.6, 0.4]])}
        out = coarsen_state(partition, pvf, pvc, state)
        # sW = (0.2*1 + 0.6*3)/4 = 0.5 ; sO = (0.8*1 + 0.4*3)/4 = 0.5
        assert np.allclose(out["s"], [[0.5, 0.5]]), out["s"]
        # pressure = (100*1 + 200*3)/4 = 175
        assert np.allclose(out["pressure"], [175.0])
        print("ok test_saturation_is_pore_volume_weighted")

    def test_saturations_sum_to_one_per_block():
        rng = np.random.default_rng(0)
        partition = np.array([0, 0, 1, 1, 1])
        pvf = rng.uniform(0.5, 2.0, 5)
        s = rng.uniform(size=(5, 3))
        s /= s.sum(axis=1, keepdims=True)
        pvc = np.bincount(partition, weights=pvf)
        out = coarsen_state(partition, pvf, pvc, {"pressure": np.zeros(5), "s": s})
        assert np.allclose(out["s"].sum(axis=1), 1.0), out["s"].sum(axis=1)
        print("ok test_saturations_sum_to_one_per_block")

    def test_rs_weighted_by_oil_pore_volume():
        """rs coarsens weighted by oil pore volume sO*pv; a block with no oil
        gets rs = 0, not a divide-by-zero."""
        partition = np.array([0, 0, 1, 1])
        pvf = np.array([1.0, 1.0, 1.0, 1.0])
        pvc = np.array([2.0, 2.0])
        s = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0],   # block 0 has oil
                      [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # block 1 no oil
        state = {"pressure": np.zeros(4), "s": s,
                 "rs": np.array([100.0, 140.0, 50.0, 50.0])}
        out = coarsen_state(partition, pvf, pvc, state)
        # block 0: (0.5*100*1 + 0.5*140*1)/(0.5*2) = 120
        assert np.isclose(out["rs"][0], 120.0), out["rs"]
        assert out["rs"][1] == 0.0, "no-oil block must give rs=0, not inf"
        print("ok test_rs_weighted_by_oil_pore_volume")

    def test_total_pore_volume_of_a_phase_is_conserved():
        """The whole point: coarse s*pv per phase equals the summed fine s*pv."""
        rng = np.random.default_rng(1)
        partition = np.array([0, 1, 0, 1, 0])
        pvf = rng.uniform(0.5, 2.0, 5)
        s = rng.uniform(size=(5, 2))
        pvc = np.bincount(partition, weights=pvf)
        out = coarsen_state(partition, pvf, pvc, {"pressure": np.zeros(5), "s": s})
        fine_phase_pv = np.array([np.bincount(partition, weights=s[:, i] * pvf)
                                  for i in range(2)]).T
        coarse_phase_pv = out["s"] * pvc[:, None]
        assert np.allclose(coarse_phase_pv, fine_phase_pv)
        print("ok test_total_pore_volume_of_a_phase_is_conserved")

    test_saturation_is_pore_volume_weighted()
    test_saturations_sum_to_one_per_block()
    test_rs_weighted_by_oil_pore_volume()
    test_total_pore_volume_of_a_phase_is_conserved()
    print("\nAll tests passed!")
