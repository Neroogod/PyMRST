"""MRST ``ad-core/upscale/upscaleState.m``: a fine model's state moved onto
the coarse grid of an upscaled model.

Saturations and pressure are pore-volume weighted with the two models'
`operators.pv`; `rs` (`rv`) is weighted by the oil (gas) pore volume when the
fine model has dissolved gas (vaporized oil); temperature is the plain block
mean; face fluxes are summed over each coarse face with the fine-to-coarse
orientation; cached `FlowProps` are dropped. For a compositional state the
coarse moles are formed from the fine phase compositions, the flash fields
removed and the coarse state re-flashed, as MRST does.

`upscale_state_ad` and `coarsen_state`, below, are PyMRST's earlier version,
kept for their callers. Its notes follow.

Coarsen a reservoir state onto a coarse grid (MRST `upscaleState`).

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

from typing import Any, Dict, Optional

import numpy as np

_FLASH_FIELDS = ("L", "x", "y", "K", "Z_L", "Z_V", "mixing", "flag", "eos")


def upscale_state(coarsemodel: Any, model: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """MRST `state = upscaleState(coarsemodel, model, state)`."""
    p = np.asarray(coarsemodel.G.partition, dtype=np.int64).ravel()
    CG = coarsemodel.G
    nc = CG.cells.num
    pvc = np.asarray(coarsemodel.operators.pv, dtype=float).ravel()
    pvf = np.asarray(model.operators.pv, dtype=float).ravel()
    counts = np.bincount(p, minlength=nc)
    state_f = state
    state = dict(state)

    sf = np.asarray(state_f["s"], dtype=float)
    pvs = sf*pvf[:, None]
    state["s"] = np.column_stack([np.bincount(p, pvs[:, i], minlength=nc)/pvc
                                  for i in range(sf.shape[1])])
    if "components" in state:
        state = _upscale_components(model, state_f, state, p, pvf, pvc)
    with np.errstate(divide="ignore", invalid="ignore"):
        if getattr(model, "disgas", False) and "rs" in state:
            sO = np.asarray(model.get_prop(state_f, "sO"), dtype=float)
            sO_c = np.asarray(coarsemodel.get_prop(state, "sO"), dtype=float)
            rs = np.bincount(p, sO*np.asarray(state["rs"], dtype=float)*pvf, minlength=nc)/(sO_c*pvc)
            rs[~np.isfinite(rs)] = 0
            state["rs"] = rs
        if getattr(model, "vapoil", False) and "rv" in state:
            sG = np.asarray(model.get_prop(state_f, "sG"), dtype=float)
            sG_c = np.asarray(coarsemodel.get_prop(state, "sG"), dtype=float)
            rv = np.bincount(p, sG*np.asarray(state["rv"], dtype=float)*pvf, minlength=nc)/(sG_c*pvc)
            rv[~np.isfinite(rv)] = 0
            state["rv"] = rv
    pressure = np.asarray(state["pressure"], dtype=float).ravel()
    state["pressure"] = np.bincount(p, pvf*pressure, minlength=nc)/pvc
    if "T" in state:
        state["T"] = np.bincount(p, np.asarray(state["T"], dtype=float).ravel(), minlength=nc)/counts
    if "flux" in state:
        state["flux"] = _coarsen_flux(CG, np.asarray(state["flux"], dtype=float))
    state.pop("FlowProps", None)
    if "components" in state:
        state = model.computeFlash(state, np.inf)
    return state


def _coarsen_flux(CG: Any, flux: np.ndarray) -> np.ndarray:
    """MRST: fine fluxes summed over each coarse face, oriented by `fineToCoarseSign`."""
    from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign
    cg = CG.coarse_grid
    cfsign = fine_to_coarse_sign(cg)
    nfc = CG.faces.num
    cfacesno = np.repeat(np.arange(nfc), np.diff(np.asarray(cg.conn_pos)))
    flux = flux.reshape(flux.shape[0], -1)
    return np.column_stack([np.bincount(cfacesno, flux[cg.fconn, i]*cfsign, minlength=nfc)
                            for i in range(flux.shape[1])])


def _upscale_components(model: Any, state_f: Dict[str, Any], state: Dict[str, Any],
                        p: np.ndarray, pvf: np.ndarray, pvc: np.ndarray) -> Dict[str, Any]:
    """MRST's compositional branch: coarse overall composition from the fine moles."""
    prop = model.PropertyModel
    rhoL = prop.computeMolarDensity(model.EOSModel, state["pressure"], state_f["x"],
                                    state_f["Z_L"], state["T"], True)
    rhoV = prop.computeMolarDensity(model.EOSModel, state["pressure"], state_f["y"],
                                    state_f["Z_V"], state["T"], False)
    sL, sV = model.get_props(state_f, "so", "sg")
    N = (np.asarray(rhoL)[:, None]*np.asarray(state_f["x"])*np.asarray(sL)[:, None]
         + np.asarray(rhoV)[:, None]*np.asarray(state_f["y"])*np.asarray(sV)[:, None])
    N_f = N*pvf[:, None]
    nc = pvc.size
    N_c = np.column_stack([np.bincount(p, N_f[:, i], minlength=nc)/pvc
                           for i in range(N_f.shape[1])])
    state["components"] = N_c/N_c.sum(axis=1, keepdims=True)
    for field in _FLASH_FIELDS:
        state.pop(field, None)
    return state


upscaleState = upscale_state


# ---------------------------------------------------------------------------
# PyMRST's earlier version.




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


#: PyMRST's per-phase saturation keys, in MRST's phase order.
PHASE_KEYS = ("sW", "sO", "sG")


def upscale_state_ad(coarse_model, model, state: Dict,
                  oil_col: int = 1, gas_col: int = 2) -> Dict:
    """MRST-signature wrapper: read the partition and pore volumes from the
    models and coarsen `state`. The coarse model must carry `partition` (set by
    `upscale_model_tpfa`) and its `operators.pv`.

    MRST keeps saturations in one `state.s` matrix; PyMRST's models read and
    write them as separate `sW`/`sO`/`sG` arrays. `coarsen_state` below is the
    ported arithmetic and works in the matrix form, so the translation happens
    here - in both directions. Without it this function could neither be given
    a PyMRST state nor return one the models could run, which is why it had
    never been used against a real model.
    """
    partition = getattr(coarse_model, "partition", None)
    if partition is None:
        partition = coarse_model.G.partition

    keys = [key for key in PHASE_KEYS if key in state]
    matrix_form = dict(state)
    if "s" not in state:
        if not keys:
            raise KeyError(
                "the state carries no saturations (s, or sW/sO/sG); there is "
                "nothing to coarsen")
        matrix_form["s"] = np.column_stack(
            [np.asarray(state[key], dtype=float).ravel() for key in keys])

    coarse = coarsen_state(partition, model.operators.pv,
                           coarse_model.operators.pv, matrix_form,
                           oil_col, gas_col)
    if "s" not in state and keys:
        # Give it back in the shape it arrived in, so the coarse model can run.
        columns = coarse.pop("s")
        for index, key in enumerate(keys):
            coarse[key] = columns[:, index]
        try:
            from pymrst.modules.ad_core import State
            coarse = State(coarse)
        except Exception:
            pass
    return coarse



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

    def test_faithful_state_on_a_coarse_model():
        """MRST's `upscaleState`: pv-weighted saturations and pressure, coarse
        face fluxes summed with orientation, `FlowProps` dropped, the fine
        state left alone."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
        from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa
        from pymrst.modules.ad_props import init_simple_adi_fluid
        from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import partition_cart_grid
        from pymrst.params.rock import make_rock
        g = compute_geometry(cart_grid([4, 2, 1]))
        rock = make_rock(g, 1e-13, np.linspace(0.1, 0.4, 8))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800], n=[2, 2])
        model = GenericBlackOilModel(g, rock, fluid, gas=False).setup_operators()
        cm = upscale_model_tpfa(model, partition_cart_grid([4, 2, 1], [2, 1, 1]))
        sw = np.linspace(0, 1, 8)
        flux = np.zeros((g.faces.num, 2))
        flux[np.all(g.faces.neighbors >= 0, axis=1)] = 1.0
        state = {"pressure": np.arange(8.0), "s": np.column_stack([sw, 1 - sw]),
                 "flux": flux, "FlowProps": {"x": 1}}
        out = upscale_state(cm, model, state)
        pvf, part = model.operators.pv, cm.G.partition
        for b in range(2):
            m = part == b
            assert np.isclose(out["s"][b, 0], np.sum(pvf[m]*sw[m])/np.sum(pvf[m]))
            assert np.isclose(out["pressure"][b], np.sum(pvf[m]*np.arange(8.0)[m])/np.sum(pvf[m]))
        interior = np.flatnonzero(np.all(cm.G.faces.neighbors >= 0, axis=1))
        assert interior.size == 1 and np.allclose(np.abs(out["flux"][interior]), 2.0)
        assert "FlowProps" not in out and "FlowProps" in state and state["s"].shape == (8, 2)
        print("ok test_faithful_state_on_a_coarse_model")

    test_faithful_state_on_a_coarse_model()
    test_saturation_is_pore_volume_weighted()
    test_saturations_sum_to_one_per_block()
    test_rs_weighted_by_oil_pore_volume()
    test_total_pore_volume_of_a_phase_is_conserved()
    print("\nAll tests passed!")
