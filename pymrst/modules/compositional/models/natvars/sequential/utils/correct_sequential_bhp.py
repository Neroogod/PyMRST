"""Port of MRST ``correctSequentialBHP.m``."""

from __future__ import annotations

import numpy as np


def correct_sequential_bhp(modelseq, states_seq, schedule):
    """Correct sequential well BHP estimates from stored flux/mobility data."""
    transport_model = getattr(modelseq, "transportModel", getattr(modelseq, "transport_model", None))
    if transport_model is not None:
        transport_model.extraStateOutput = True
    out = list(states_seq)
    for i, state in enumerate(out):
        control_ix = int(schedule.step.control[i]) - 1 if hasattr(schedule.step, "control") else 0
        control = schedule.control[control_ix] if hasattr(schedule, "control") else schedule
        wells = getattr(control, "W", getattr(control, "wells", ()))
        for j, well in enumerate(wells):
            if str(getattr(well, "type", "")).lower() == "bhp":
                continue
            wellsol = state.wellSol[j]
            flux = np.asarray(getattr(wellsol, "flux"), dtype=float)
            q = flux.sum(axis=1) if flux.ndim == 2 else flux.reshape(-1)
            cells = np.asarray(getattr(well, "cells"), dtype=int).reshape(-1)
            wi = np.asarray(getattr(well, "WI", np.ones_like(cells, dtype=float)), dtype=float).reshape(-1)
            mob = np.asarray(getattr(state, "mob"), dtype=float)
            mob_sum = mob[cells, :].sum(axis=1) if mob.ndim == 2 else mob[cells]
            pressure = np.asarray(state.pressure, dtype=float)[cells]
            cdp = np.asarray(getattr(wellsol, "cdp", np.zeros_like(q)), dtype=float).reshape(-1)
            bhp_perf = q / np.maximum(wi * mob_sum, 1.0e-300) - cdp + pressure
            weights = np.abs(q)
            if weights.sum() > 0.0:
                bhp_new = float(np.sum(bhp_perf * weights) / weights.sum())
            else:
                bhp_new = float(np.mean(bhp_perf))
            if isinstance(wellsol, dict):
                wellsol["bhp"] = bhp_new
            else:
                setattr(wellsol, "bhp", bhp_new)
    return out


def correctSequentialBHP(modelseq, states_seq, schedule):
    """MRST camelCase alias for :func:`correct_sequential_bhp`."""
    return correct_sequential_bhp(modelseq, states_seq, schedule)


__all__ = ["correct_sequential_bhp", "correctSequentialBHP"]
