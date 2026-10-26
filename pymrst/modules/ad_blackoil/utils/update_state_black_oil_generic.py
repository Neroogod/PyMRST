"""MRST ``updateStateBlackOilGeneric.m`` Newton-state update owner."""

import numpy as np

from pymrst.modules.ad_blackoil.utils.compute_flash_black_oil import (
    compute_flash_black_oil_ad,
)


def update_state_black_oil_generic(
    model, state, problem, dx=None, driving_forces=None,
):
    """Apply a packed Newton increment to a black-oil state.

    PyMRST's nonlinear problems already pack variables in the model's primary
    order, so ``problem`` is metadata-only here. For compatibility, callers
    may pass ``(model, state, dx)`` and omit it. Pressure and saturation steps
    are limited, the switched hydrocarbon variable is routed by status, wells
    and multisegment variables are updated, and the black-oil flash completes
    phase appearance/disappearance.
    """
    if dx is None:
        dx, problem = problem, None
    increment = np.asarray(dx, dtype=float).ravel()
    nc = model.G.cells.num
    nw = model.facility.n_wells
    nph = model.n_phases
    new = state.copy()

    cap = model.dp_max_rel*np.abs(state["pressure"])
    new["pressure"] = state["pressure"] + np.clip(increment[:nc], -cap, cap)
    dsw = increment[nc:2*nc]

    if not (model.disgas or model.vapoil):
        new["sW"] = np.clip(
            state["sW"] + np.clip(dsw, -model.ds_max, model.ds_max), 0.0, 1.0
        )
        offset = 2*nc
        if nph > 2:
            # A three-phase model without dissolution carries `sG` as a second
            # independent saturation, seeded by `init_state_ad` right after
            # `sW`. The unpacking here has to use the same layout: reading the
            # well block from `2*nc` instead leaves the bottom-hole pressures
            # frozen at their initial guess while the rates absorb the gas
            # saturation's increment, and Newton never converges.
            dsg = increment[2*nc:3*nc]
            new["sG"] = np.clip(
                state["sG"] + np.clip(dsg, -model.ds_max, model.ds_max),
                0.0, 1.0)
            new["sO"] = 1.0 - new["sW"] - new["sG"]
            offset = 3*nc
    else:
        st1, st2, st3 = model._status_masks(state["status"])
        dx_hc = increment[2*nc:3*nc]
        dsg = np.where(st3, dx_hc, 0.0) - np.where(st2, dsw, 0.0)
        drs_max = float(getattr(model, "drs_max", np.inf))
        drs = np.clip(np.where(st1, dx_hc, 0.0), -drs_max, drs_max)
        new["rs"] = np.maximum(state["rs"] + drs, 0.0)
        if model.vapoil:
            drv = np.clip(np.where(st2, dx_hc, 0.0), -drs_max, drs_max)
            new["rv"] = np.maximum(state["rv"] + drv, 0.0)

        ds = np.column_stack([dsw, -(dsg+dsw), dsg])
        largest = np.max(np.abs(ds), axis=1)
        # Subnormal nonzero increments can make ds_max/largest overflow even
        # though the subsequent minimum correctly selects 1. Silence that
        # harmless intermediate and retain the exact limiter semantics.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scale = np.minimum(
                np.divide(model.ds_max, largest,
                          out=np.ones_like(largest), where=largest > 0),
                1.0,
            )
        ds = ds*scale[:, None]
        saturation = np.column_stack([
            state["sW"], state["sO"], state["sG"],
        ]) + ds
        new["sW"], new["sO"], new["sG"] = saturation.T
        offset = 3*nc

    rate_names = ["qWs", "qOs", "qGs"][:nph]
    for index, name in enumerate(rate_names):
        lo = offset + index*nw
        new[name] = state[name] + increment[lo:lo+nw]
    bhp_lo = offset + nph*nw
    new["bhp"] = state["bhp"] + increment[bhp_lo:bhp_lo+nw]

    cursor = bhp_lo + nw
    if model._ms_well_indices():
        names = ["ms_pN", *["ms_rW", "ms_rO", "ms_rG"][:nph], "ms_vmix"]
        for name in names:
            count = len(state.get(name, []))
            if count:
                updated = state[name] + increment[cursor:cursor+count]
                if name in ("ms_rW", "ms_rO", "ms_rG"):
                    updated = np.clip(updated, 0.0, 1.0)
                new[name] = updated
                cursor += count

    if model.disgas or model.vapoil:
        compute_flash_black_oil_ad(new, state, model)
    model.facility.update_switched_controls(new, rate_names)
    new.clear_cache()
    return new


updateStateBlackOilGeneric = update_state_black_oil_generic

__all__ = ["update_state_black_oil_generic", "updateStateBlackOilGeneric"]
