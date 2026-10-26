"""MRST ``ad-blackoil/utils/computeFlashBlackOil.m``.

After a Newton update, decide which hydrocarbon phases are present in each
cell, let a phase appear or vanish, clip negative saturations by rescaling the
others, and pin `rs` / `rv` to their saturated values where the other phase is
present.

`compute_flash_black_oil` is MRST's function on the AD-OO state (`s` matrix,
model property access, `RsMax`/`RvMax` from the PVT grouping), taking the
previous step's status masks. `compute_flash_black_oil_ad` is PyMRST's earlier
version for the legacy state layout (`sW`/`sO`/`sG` keys), kept for the
models built on it.
"""

import numpy as np

from pymrst.utils.adi import value

ETOL: float = np.sqrt(np.finfo(float).eps)


def compute_flash_black_oil_ad(state, state0, model):
    """Update saturations, dissolved ratios, and status after a Newton step.

    This is the Python owner for MRST's `computeFlashBlackOil`: it clips
    negative phase saturations, handles gas/oil phase appearance, caps `rs`
    and `rv` against their saturation curves, and writes the hydrocarbon
    status `oil_present + 2*gas_present`.

    **One deliberate departure from MATLAB.** The saturations are renormalized
    before being written back; MRST writes them as the clipping cascade leaves
    them. On a well-formed state the two agree exactly - each clip divides by
    ``1 - s_negative``, which preserves a unit sum - so the normalization only
    bites on input that did not sum to one, where MRST propagates the error and
    this silently repairs it. Kept because a saturation triple that does not
    sum to one is not a state any downstream property can read, but recorded
    because it means this function cannot be used to detect that upstream fault.
    """
    f = model.raw_fluid
    sw, so, sg = state["sW"], state["sO"], state["sG"]
    rs = state["rs"]
    st1_prev = state0["status"] == 1
    st2_prev = state0["status"] == 2
    wat_only = sw > 1.0 - ETOL

    if not getattr(model, "disgas", True):
        # MRST's `~disgas` branch, which this port omitted. Without dissolved
        # gas there is no saturation curve to cross: the previous `rs` *is* the
        # bound, and gas counts as present everywhere. The omission was only
        # reachable through `disgas=False, vapoil=True` - the caller in
        # `update_state_black_oil_generic` skips the flash when both are false -
        # and there it reported status 1 where MRST reports 3, because
        # `gasPresent` contributes the 2.
        rs_sat0 = np.asarray(value(state0["rs"]), dtype=float)
        rs_sat = rs_sat0
        gas_present = np.ones_like(wat_only, dtype=bool)
    else:
        rs_sat = value(f.rs_sat(state["pressure"]))
        rs_sat0 = value(f.rs_sat(state0["pressure"]))

        gas_present = ((sg > 0) | (rs == 0)) & ~st1_prev
        gas_present |= wat_only
        gas_present |= (sg < 0) & (state0["sG"] > ETOL)
        crossed = (
            (rs > rs_sat * (1 + ETOL))
            & st1_prev
            & (state0["rs"] > rs_sat0 * (1 - ETOL))
        )
        sg[crossed] = 0.0
        gas_present |= crossed

    neg = sg < 0
    sw[neg] /= 1.0 - sg[neg]
    so[neg] /= 1.0 - sg[neg]
    sg[neg] = 0.0

    if model.vapoil:
        rv = state["rv"]
        rv_sat = value(f.rv_sat(state["pressure"]))
        rv_sat0 = value(f.rv_sat(state0["pressure"]))

        oil_present = ((so > 0) | (rv == 0)) & ~st2_prev
        oil_present |= wat_only
        oil_present |= (so < 0) & (state0["sO"] > ETOL)
        crossed_o = (
            (rv > rv_sat * (1 + ETOL))
            & st2_prev
            & (state0["rv"] > rv_sat0 * (1 - ETOL))
        )
        so[crossed_o] = 0.0
        oil_present |= crossed_o
    else:
        oil_present = np.ones_like(gas_present)

    neg = so < 0
    sw[neg] /= 1.0 - so[neg]
    sg[neg] /= 1.0 - so[neg]
    so[neg] = 0.0

    neg = sw < 0
    so[neg] /= 1.0 - sw[neg]
    sg[neg] /= 1.0 - sw[neg]
    sw[neg] = 0.0

    rs[gas_present] = rs_sat[gas_present]
    rs[~gas_present] = np.minimum(rs_sat[~gas_present], rs[~gas_present])
    state["rs"] = np.maximum(rs, 0.0)
    if model.vapoil:
        rv[oil_present] = rv_sat[oil_present]
        state["rv"] = np.maximum(rv, 0.0)

    state["status"] = (
        oil_present.astype(np.int64) + 2 * gas_present.astype(np.int64)
    )

    s = np.column_stack([sw, so, sg])
    s = s / s.sum(axis=1, keepdims=True)
    state["sW"], state["sO"], state["sG"] = s[:, 0], s[:, 1], s[:, 2]
    return state


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
def compute_flash_black_oil(state, state0, model, status):
    """MRST `computeFlashBlackOil(state, state0, model, status)`.

    `status` is `[st1, st2, st3]` from `getCellStatusVO` on the previous
    iterate. Returns the state with saturations, `rs`, `rv` and the integer
    `status = oilPresent + 2*gasPresent` updated.
    """
    disgas, vapoil = bool(model.disgas), bool(model.vapoil)
    so = np.array(value(model.get_prop(state, "so")), dtype=float, copy=True)
    so0 = np.asarray(value(model.get_prop(state0, "so")), dtype=float)
    sw = (np.array(value(model.get_prop(state, "sw")), dtype=float, copy=True)
          if model.water else np.zeros(model.G.cells.num))
    sg = np.array(value(model.get_prop(state, "sg")), dtype=float, copy=True)
    sg0 = np.asarray(value(model.get_prop(state0, "sg")), dtype=float)
    rs = np.array(value(model.get_prop(state, "rs")), dtype=float, copy=True)
    rs0 = np.asarray(value(model.get_prop(state0, "rs")), dtype=float)
    rv = np.array(value(model.get_prop(state, "rv")), dtype=float, copy=True)
    rv0 = np.asarray(value(model.get_prop(state0, "rv")), dtype=float)
    wat_only = sw > 1 - ETOL
    # Without dissolution MRST uses a scalar `true` mask, so a scalar `rs`
    # (or `rv`) stays scalar through `rs(gasPresent) = rsSat(gasPresent)`.
    if not disgas:
        rs_sat0 = rs0
        rs_sat = rs_sat0
        gas_present = True
    else:
        st1 = np.broadcast_to(np.asarray(status[0], dtype=bool), sg.shape)
        pvt = model.PVTPropertyFunctions
        rs_sat0 = np.asarray(value(pvt.get(model, state0, "RsMax")), dtype=float)
        rs_sat = np.asarray(value(pvt.get(model, state, "RsMax")), dtype=float)
        gas_present = (((sg > 0) | (rs == 0)) & ~st1) | wat_only
        gas_present = gas_present | ((sg < 0) & (sg0 > ETOL))
        ix2 = (rs > rs_sat*(1 + ETOL)) & st1 & (rs0 > rs_sat0*(1 - ETOL))
        sg[ix2] = 0
        gas_present = gas_present | ix2
    ix = sg < 0
    sw[ix] = sw[ix]/(1 - sg[ix])
    so[ix] = so[ix]/(1 - sg[ix])
    sg[ix] = 0
    if not vapoil:
        oil_present = True
        rv_sat0 = rv0
        rv_sat = rv_sat0
    else:
        st2 = np.broadcast_to(np.asarray(status[1], dtype=bool), sg.shape)
        rv_sat0 = np.asarray(value(model.get_prop(state0, "RvMax")), dtype=float)
        rv_sat = np.asarray(value(model.get_prop(state, "RvMax")), dtype=float)
        oil_present = (((so > 0) | (rv == 0)) & ~st2) | wat_only
        oil_present = oil_present | ((so < 0) & (so0 > ETOL))
        ix2 = (rv > rv_sat*(1 + ETOL)) & st2 & (rv0 > rv_sat0*(1 - ETOL))
        so[ix2] = 0
        oil_present = oil_present | ix2
    ix = so < 0
    sw[ix] = sw[ix]/(1 - so[ix])
    sg[ix] = sg[ix]/(1 - so[ix])
    so[ix] = 0
    ix = sw < 0
    so[ix] = so[ix]/(1 - sw[ix])
    sg[ix] = sg[ix]/(1 - sw[ix])
    sw[ix] = 0
    rs = _set_saturated(rs, rs_sat, gas_present, undersaturated_min=True)
    rv = _set_saturated(rv, rv_sat, oil_present, undersaturated_min=False)
    if model.water:
        state = model.set_prop(state, "sw", sw)
    state = model.set_prop(state, "so", so)
    state = model.set_prop(state, "sg", sg)
    state = model.set_prop(state, "rs", np.maximum(rs, 0))
    state = model.set_prop(state, "rv", np.maximum(rv, 0))
    state["status"] = (np.asarray(oil_present, dtype=np.int64)
                       + 2*np.asarray(gas_present, dtype=np.int64))*np.ones(sg.shape, np.int64)
    return state


def _set_saturated(r, r_sat, present, undersaturated_min: bool):
    """`r(present) = rSat(present)`, and for `rs` also
    `r(~present) = min(rSat(~present), r(~present))`. A scalar `true` mask
    replaces `r` by `rSat` whole."""
    if present is True:
        return np.array(r_sat, dtype=float, copy=True)
    r = np.array(np.broadcast_to(r, present.shape), dtype=float, copy=True)
    r_sat = np.broadcast_to(r_sat, present.shape)
    r[present] = r_sat[present]
    if undersaturated_min:
        r[~present] = np.minimum(r_sat[~present], r[~present])
    return r


computeFlashBlackOil = compute_flash_black_oil

__all__ = ["compute_flash_black_oil", "compute_flash_black_oil_ad",
           "computeFlashBlackOil"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock

    def _live_oil(rs, sg):
        _, _, fluid, _, _ = setup_spe1()
        G = compute_geometry(cart_grid([2, 1, 1]))
        model = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid,
                                     disgas=True).validate_model()
        sw = np.full(2, 0.2)
        state = model.validate_state({"pressure": np.full(2, 200e5), "rs": np.asarray(rs),
                                      "s": np.column_stack([sw, 1 - sw - sg, sg])})
        return model, model.init_state_function_containers(state)

    def test_dry_gas_keeps_a_scalar_rv():
        """Without vaporised oil MRST's mask is a scalar `true`, so `rv = 0`
        stays the scalar it was."""
        with isolated_gravity():
            model, state = _live_oil([50.0, 50.0], np.array([0.0, 0.0]))
            st = model.get_cell_status_vo(state, 1 - 0.2 - 0.0, 0.2, np.zeros(2))
            out = compute_flash_black_oil(state, state, model, st)
        assert np.ndim(out["rv"]) == 0 and float(out["rv"]) == 0.0
        print("ok test_dry_gas_keeps_a_scalar_rv")

    def test_free_gas_saturates_the_oil():
        """A cell with free gas has `rs` set to its saturated value."""
        with isolated_gravity():
            model, state = _live_oil([50.0, 50.0], np.array([0.0, 0.1]))
            st = model.get_cell_status_vo(state, np.array([0.8, 0.7]), np.full(2, 0.2),
                                          np.array([0.0, 0.1]))
            out = compute_flash_black_oil(state, state, model, st)
            rs_sat = np.asarray(value(model.get_prop(out, "RsMax")))
        assert np.isclose(out["rs"][1], rs_sat[1]) and out["rs"][0] == 50.0
        assert out["status"].tolist() == [1, 3]
        print("ok test_free_gas_saturates_the_oil")

    def test_negative_gas_is_removed_and_the_rest_rescaled():
        with isolated_gravity():
            model, state = _live_oil([50.0, 50.0], np.array([0.0, -0.05]))
            st = model.get_cell_status_vo(state, np.array([0.8, 0.85]), np.full(2, 0.2),
                                          np.array([0.0, -0.05]))
            out = compute_flash_black_oil(state, state, model, st)
        s = np.asarray(out["s"])
        assert s[1, 2] == 0 and np.isclose(s[1].sum(), 1.0)
        print("ok test_negative_gas_is_removed_and_the_rest_rescaled")

    test_dry_gas_keeps_a_scalar_rv()
    test_free_gas_saturates_the_oil()
    test_negative_gas_is_removed_and_the_rest_rescaled()
    print("\nAll tests passed!")
