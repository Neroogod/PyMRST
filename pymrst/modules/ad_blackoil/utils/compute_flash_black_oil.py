"""MRST `ad-blackoil/utils/computeFlashBlackOil.m`."""

import numpy as np

from pymrst.utils.adi import value

ETOL: float = np.sqrt(np.finfo(float).eps)


def compute_flash_black_oil(state, state0, model):
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


__all__ = ["compute_flash_black_oil"]
