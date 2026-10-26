"""MRST ``ad-core/models/facilities/GenericFacilityModel.m`` surface."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.units import ATM


class GenericFacilityModel(FacilityModel):
    """Generic facility model with MRST's generic-facility defaults."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = 288.15
        self.pressure = ATM
        self.SeparatorGroup = None
        self.separator_group = None
        self.outputFluxes = True
        self.output_fluxes = True
        self.doPostUpdate = None
        self.do_post_update = None
        self.setWellTargets = False

    def getNumberOfComponents(self) -> int:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is None:
            return self._number_of_components()
        for method in ("getNumberOfComponents", "get_number_of_components"):
            if hasattr(reservoir_model, method):
                return int(getattr(reservoir_model, method)())
        return self._number_of_components()

    get_number_of_components = getNumberOfComponents

    def getNumberOfPhases(self) -> int:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is not None:
            for method in ("getNumberOfPhases", "get_number_of_phases"):
                if hasattr(reservoir_model, method):
                    return int(getattr(reservoir_model, method)())
        return len(self._phase_names())

    get_number_of_phases = getNumberOfPhases

    def getSurfaceConditions(self) -> tuple[float, float]:
        """Return separator/surface pressure and temperature."""
        return float(self.pressure), float(self.T)

    get_surface_conditions = getSurfaceConditions

    def getComponentSources(self, state: Any):
        """Return component source data from the facility discretization."""
        fp = self.FacilityFlowDiscretization
        if fp is None or not hasattr(fp, "get"):
            return {"value": [], "cells": []}
        mapping = fp.get(self, state, "FacilityWellMapping")
        wells = getattr(mapping, "W", None)
        cells = getattr(mapping, "cells", [])
        if wells is None or len(wells) == 0:
            values = []
        else:
            values = fp.get(self, state, "ComponentTotalFlux")
        return {"value": values, "cells": cells}

    get_component_sources = getComponentSources

    # ======================================================================
    # Well control limits.
    #
    # A well is run on one control - a rate, or a bottom-hole pressure - and
    # carries limits on the others. Honouring them is *control switching*: a
    # producer on a rate target whose pressure falls below its minimum stops
    # being a rate well and becomes a BHP well, for that step and until
    # conditions change.
    #
    # Skipping it does not fail. The well simply produces at a rate the
    # reservoir cannot support, drawing the pressure down through the limit
    # and past it, and the run completes with a plausible profile that no
    # facility could have delivered.
    #
    # `FacilityModel._violated_limit` is a second, narrower implementation of
    # the check half, used by the group-control path. The two are kept
    # separate deliberately rather than merged: they differ on the sign
    # convention for `vrat` and on which limits they consider, and unifying
    # them is a behaviour change to the group path that no test currently
    # covers. Recorded here so the duplication is a decision, not an oversight.
    # ======================================================================

    @staticmethod
    def set_missing_limits(lims, modes, default):
        """Fill in the limits a well does not declare (MRST `setMissingLimits`).

        The default differs by well type and that is the whole subtlety: an
        *injector*'s unset limits are `+inf` (nothing is ever above them) and a
        *producer*'s are `-inf` (nothing is ever below them). Getting it
        backwards makes every unset limit violated on the first check, and the
        well switches control immediately to a mode nobody configured.
        """
        filled = dict(lims or {})
        for mode in modes:
            if filled.get(mode) is None:
                filled[mode] = default
        return filled

    setMissingLimits = set_missing_limits

    def apply_well_limits(self, state):
        """Apply control switching to every active well (MRST
        `applyWellLimits`)."""
        well_sols = list(_field(state, "wellSol", []) or [])
        # `WellModels` is the MRST-named mirror and this port leaves it empty;
        # `wells` is the populated list. Indexing the mirror is an `IndexError`
        # on the first active well - the same latent crash `update_resv_controls`
        # carried, and for the same reason: neither had ever been called.
        registry = list(getattr(self, "WellModels", None) or
                        getattr(self, "wells", None) or [])
        for index in np.atleast_1d(self.get_indices_of_active_wells(well_sols)):
            index = int(index)
            if index >= len(well_sols) or index >= len(registry):
                continue
            well_sols[index] = self.apply_well_limits_well_sol(
                registry[index], well_sols[index])
        _set_field(state, "wellSol", well_sols)
        return state

    applyWellLimits = apply_well_limits

    def apply_well_limits_well_sol(self, well, well_sol):
        """Switch one well onto the first limit it violates.

        Two guards come first and both mean "do not touch this well":
        `allow_control_switching` is how a caller pins a well to its control,
        and a shut well has no rates to check.

        The mode it is *currently on* is never a candidate. MRST excludes it
        explicitly, and it has to: a well sitting exactly on its BHP limit
        satisfies `bhp <= lims.bhp` as an equality, so without the exclusion it
        would "violate" its own control and switch to it every step.

        After a switch the well's stored rates are reset to the new target so
        the next Newton solve starts from the control it is actually on -
        leaving the old rates makes the first iteration solve the previous
        control's problem.
        """
        if not _truthy(getattr(well, "allow_control_switching",
                               getattr(well, "allowControlSwitching", True))):
            return well_sol
        W = getattr(well, "W", well)
        if not _truthy(getattr(W, "status", True)):
            return well_sol

        lims = getattr(W, "lims", None)
        if not lims or isinstance(lims, (int, float)):
            return well_sol

        model = self.ReservoirModel
        phases = list(model.get_phase_names())
        rates = {p: float(np.sum(_value_of(_field(well_sol, f"q{p}s", 0.0))))
                 for p in phases}
        qs_t = float(sum(rates.values()))
        bhp = float(np.ravel(_value_of(_field(well_sol, "bhp", 0.0)))[0])

        injector = _truthy(getattr(well, "is_injector", lambda: False)()) \
            if callable(getattr(well, "is_injector", None)) \
            else float(getattr(W, "sign", -1.0)) > 0

        if injector:
            modes = ["bhp", "rate", "vrat"]
            lims = self.set_missing_limits(lims, modes, np.inf)
            # An unset volumetric-rate *lower* bound is -inf, not +inf: it is
            # the one limit on an injector that is checked from below.
            if not np.isfinite(lims["vrat"]):
                lims["vrat"] = -np.inf
            flags = [bhp > lims["bhp"], qs_t > lims["rate"],
                     qs_t < lims["vrat"]]
        else:
            modes = ["bhp", "orat", "lrat", "grat", "wrat", "vrat"]
            lims = self.set_missing_limits(lims, modes, -np.inf)
            if not np.isfinite(lims["vrat"]):
                lims["vrat"] = np.inf
            q_w, q_o, q_g = (rates.get("W", 0.0), rates.get("O", 0.0),
                             rates.get("G", 0.0))
            q_sl = float(np.sum(_value_of(_field(well_sol, "qSs", 0.0))))
            flags = [bhp < lims["bhp"], q_o < lims["orat"],
                     q_w + q_o < lims["lrat"], q_g + q_sl < lims["grat"],
                     q_w < lims["wrat"], qs_t > lims["vrat"]]

        current = str(_field(well_sol, "type", "")).lower()
        switch_mode = None
        for mode, violated in zip(modes, flags):
            if mode.lower() == current:
                continue
            if violated:
                switch_mode = mode
                break
        if switch_mode is None:
            return well_sol

        _set_field(well_sol, "type", switch_mode)
        _set_field(well_sol, "val", lims[switch_mode])

        target = lims[switch_mode]
        if switch_mode == "bhp":
            _set_field(well_sol, "bhp", target)
        elif switch_mode == "rate":
            compi = np.atleast_1d(np.asarray(getattr(W, "compi", []),
                                             dtype=float))
            for i, phase in enumerate(phases):
                share = compi[i] if i < compi.size else 0.0
                _set_field(well_sol, f"q{phase}s", target*share)
        elif switch_mode in ("orat", "wrat", "grat"):
            _set_field(well_sol, f"q{switch_mode[0].upper()}s", target)
        # `lrat` and `vrat` are deliberately left without a rate guess: they
        # constrain a *sum*, and splitting one across the phases would be an
        # invention. MRST does the same and says so.
        return well_sol

    applyWellLimitsWellSol = apply_well_limits_well_sol

    def prepare_report_step(self, state, state0, dt, forces=None):
        """MRST ``GenericFacilityModel.prepareReportstep``.

        The base does `validateModel`; this adds the reservoir-volume
        conversion, which has to happen *once per report step* rather than per
        Newton iteration, because it reads `state0` - the pore volumes and
        dissolved ratios at the start of the step.
        """
        _, state = FacilityModel.prepare_report_step(self, state, state0, dt,
                                                    forces)
        return self.update_resv_controls(state, state0)

    prepareReportstep = prepare_report_step

    def prepare_timestep(self, state, state0, dt, forces=None):
        """MRST ``GenericFacilityModel.prepareTimestep``.

        Two things happen here, and the first is the one that decides whether a
        zero-rate well is solvable at all::

            ctrl = wm.W.type;
            if ~(strcmpi(ctrl,'bhp') || strcmpi(ctrl,'thp'))
                wellSol(wellNo).status = wm.W.val ~= 0;
            end

        A well on any *rate* control with a zero target is **shut**, not open
        and held at zero. MRST never asks Newton to find the pressure at which
        an open well's perforations cancel out; it closes the well, which takes
        the trivially-conditioned shut branch instead. Norne's B-2H is an
        `orat` well with a zero target, and without this it stays open and the
        well block is left near-indeterminate.

        A BHP or THP well is exempt: zero *pressure* is a target, not a
        closure.
        """
        well_sols = list(_field(state, "wellSol", []) or [])
        registry = list(getattr(self, "WellModels", None) or
                        getattr(self, "wells", None) or [])
        for index, well in enumerate(registry):
            W = getattr(well, "W", well)
            control = str(getattr(W, "type", "")).lower()
            if control in ("bhp", "thp"):
                continue
            # Conjoin, never replace. MRST's `status = W.val ~= 0` overwrites
            # a status that has already been through the deck, so it can only
            # ever close a well. Assigning it directly here *re-opens* every
            # well the deck shut that still carries a target - on Norne that is
            # 14 of 36, taking the model from 3 open wells to 17 and making the
            # very first step singular.
            nonzero = float(np.ravel(_value_of(getattr(W, "val", 0.0)))[0]) != 0.0
            open_ = bool(getattr(W, "status", True)) and nonzero
            W.status = open_
            if index < len(well_sols):
                _set_field(well_sols[index], "status", open_)
        _set_field(state, "wellSol", well_sols)

        # MRST also runs `applyWellLimits` here. Enabled 2026-08-08: the
        # shut-well rule above is now established (freeze gate 8/8 with it
        # alone), and Norne's large-timestep failures show the oscillation
        # signature of a well repeatedly overshooting a control limit rather
        # than switching to it - CNV and control residuals swinging back and
        # forth by 3-6 orders of magnitude, never settling, at exactly the
        # symptom control switching exists to prevent.
        state = self.apply_well_limits(state)
        return self, state

    prepareTimestep = prepare_timestep

    def update_resv_controls(self, state, state0):
        """Convert reservoir-volume targets into surface rates.

        A `resv` well is controlled on volume *at reservoir conditions*, which
        is what a voidage-replacement scheme actually cares about, but the well
        equations are written in surface volume. The conversion needs the
        shrinkage factors at the well's own conditions, and those depend on the
        pressure and dissolved ratios in the region it sits in - hence the
        pore-volume-weighted averages, taken over each PVT region separately.

        A `resv_history` well additionally has its target *computed* here from
        its historical surface rates and then becomes an ordinary `resv` well,
        which is why the type is rewritten.

        Returns `(self, state)`, matching MRST's `[model, state]`.
        """
        well_sols = list(_field(state, "wellSol", []) or [])
        active = np.flatnonzero(
            np.atleast_1d(self.get_well_status_mask(well_sols)))
        # MRST keeps its per-well models in `WellModels` and derives `W` from
        # them. This port keeps `wells` as the canonical list and leaves the
        # MRST-named mirror empty, which is precisely why this method could
        # never have run: it indexed an empty list and raised `IndexError` on
        # the first active well. Resolve whichever is actually populated - the
        # loop below already reads through a `.W` wrapper when there is one.
        registry = list(getattr(self, "WellModels", None) or
                        getattr(self, "wells", None) or [])
        wells = [registry[int(i)] for i in active]

        def well_type(w):
            return str(getattr(getattr(w, "W", w), "type", "")).lower()

        is_hist = np.array([well_type(w) == "resv_history" for w in wells])
        is_now = np.array([well_type(w) == "resv" for w in wells])
        is_resv = is_hist | is_now if is_hist.size else np.zeros(0, dtype=bool)
        if not is_resv.any():
            return self, state

        model = self.ReservoirModel
        phases = list(model.get_phase_names())
        selected = np.flatnonzero(is_resv)
        structs = [getattr(wells[int(i)], "W", wells[int(i)])
                   for i in selected]

        compi = np.atleast_2d(np.array(
            [np.asarray(getattr(w, "compi", []), dtype=float) for w in structs]))
        targets = np.array([float(getattr(w, "val", 0.0)) for w in structs])
        qs = targets[:, None]*compi

        disgas = bool(getattr(model, "disgas", False))
        vapoil = bool(getattr(model, "vapoil", False))
        cells = np.array([int(np.atleast_1d(w.cells)[0]) for w in structs])

        pv = np.asarray(_value_of(model.get_prop(state0, "PoreVolume")),
                        dtype=float).ravel()
        if "W" in phases:
            pv = pv*(1.0 - np.asarray(
                _value_of(model.get_prop(state0, "sW")), dtype=float).ravel())

        regions = _pvt_regions(model)
        region_of = regions[cells]
        pressure = np.asarray(_value_of(_field(state0, "pressure")),
                              dtype=float).ravel()
        p_mean = np.zeros(cells.size)
        rs = np.zeros(cells.size)
        rv = np.zeros(cells.size)
        for region in np.unique(region_of):
            rows = region_of == region
            weights = pv*(regions == region)
            total = weights.sum()
            if total <= 0:
                continue
            p_mean[rows] = float((pressure*weights).sum()/total)
            if disgas:
                rs[rows] = float(
                    (np.asarray(_value_of(_field(state0, "rs", 0.0)),
                                dtype=float).ravel()*weights).sum()/total)
            if vapoil:
                rv[rows] = float(
                    (np.asarray(_value_of(_field(state0, "rv", 0.0)),
                                dtype=float).ravel()*weights).sum()/total)

        rs0, rv0 = rs.copy(), rv.copy()
        oix = model.get_phase_index("O") if "O" in phases else None
        gix = model.get_phase_index("G") if "G" in phases else None
        if is_hist[selected].any():
            hist = is_hist[selected]
            if disgas and oix is not None and gix is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = qs[hist, gix]/qs[hist, oix]
                rs[hist] = np.minimum(np.nan_to_num(ratio), rs[hist])
            if vapoil and oix is not None and gix is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = qs[hist, oix]/qs[hist, gix]
                rv[hist] = np.minimum(np.nan_to_num(ratio), rv[hist])

        b = _shrinkage_at(model, cells, p_mean, rs, rv)
        shrink = 1.0 - rs*rv
        shrink0 = 1.0 - rs0*rv0

        n_phase = len(phases)
        factors = np.zeros((cells.size, n_phase))
        new_rates = np.zeros(cells.size)
        if "W" in phases:
            wix = model.get_phase_index("W")
            factors[:, wix] = 1.0/b[wix]
            new_rates += qs[:, wix]/b[wix]
        if oix is not None:
            orat = qs[:, oix].copy()
            factors[:, oix] += 1.0/(b[oix]*shrink0)
            if vapoil and gix is not None:
                orat = orat - rv*qs[:, gix]
                factors[:, gix] -= rv/(b[oix]*shrink0)
            new_rates += orat/(b[oix]*shrink)
        if gix is not None:
            grat = qs[:, gix].copy()
            factors[:, gix] += 1.0/(b[gix]*shrink0)
            if disgas and oix is not None:
                grat = grat - rs*qs[:, oix]
                factors[:, oix] -= rs/(b[gix]*shrink0)
            new_rates += grat/(b[gix]*shrink)

        for row, local in enumerate(selected):
            global_ix = int(active[int(local)])
            if is_hist[int(local)]:
                W = getattr(registry[global_ix], "W", registry[global_ix])
                W.val = float(new_rates[row])
                W.type = "resv"
                if global_ix < len(well_sols):
                    _set_field(well_sols[global_ix], "val",
                               float(new_rates[row]))
                    _set_field(well_sols[global_ix], "type", "resv")
            setattr(registry[global_ix], "ControlDensity", factors[row, :])
        _set_field(state, "wellSol", well_sols)
        return self, state

    updateRESVControls = update_resv_controls


def _pvt_regions(model) -> np.ndarray:
    """The PVT region of each cell, ones when none is declared."""
    density = getattr(getattr(model, "PVTPropertyFunctions", None),
                      "Density", None)
    regions = getattr(density, "regions", None)
    if regions is None:
        return np.ones(int(model.G.cells.num), dtype=int)
    return np.asarray(regions).ravel()


def _shrinkage_at(model, cells, pressure, rs, rv):
    """Shrinkage factors at the wells' own averaged conditions.

    Evaluated on a *subset* model restricted to the well cells, so the region
    dispatch inside the property picks each well's own tables. Evaluating on
    the full grid and indexing afterwards would use the wrong region wherever
    a well sits outside region one.
    """
    # MRST's substate is `s = repmat([1, 0, 0], nc, 1)` exactly. This port's
    # capillary pressure reads the per-phase names rather than the `s` list,
    # so the same saturations are published under both spellings - otherwise
    # `PhasePressures`, which shrinkage depends on whenever the deck has
    # capillary pressure, raises `KeyError: 'sW'`.
    phase_names = list(model.get_phase_names())
    saturations = ([np.ones_like(pressure)]
                   + [np.zeros_like(pressure)]*(len(phase_names) - 1))
    substate = {"pressure": np.asarray(pressure, dtype=float),
                "s": saturations,
                **{f"s{name}": saturations[i]
                   for i, name in enumerate(phase_names)},
                "rs": np.asarray(rs, dtype=float),
                "rv": np.asarray(rv, dtype=float),
                # MRST asks for saturated properties here by setting
                # `useSaturatedFlag`. This port carries the same information in
                # the black-oil switching status, where `gas_present =
                # status >= 2` and `oil_present = status % 2 == 1`, so "both
                # phases present" - saturated - is status 3.
                "status": np.full(np.size(pressure), 3, dtype=int)}
    groupings = model.PVTPropertyFunctions
    if hasattr(groupings, "subset"):
        groupings = groupings.subset(np.asarray(cells, dtype=int))
    # MRST reaches a state function as a property of its grouping and calls it
    # directly. This port registers them by name and resolves dependencies
    # through the grouping, so `.ShrinkageFactors` is an `AttributeError`, and
    # calling `evaluate_on_domain` by hand skips the dependency pass - which
    # matters here because a deck with capillary pressure makes shrinkage
    # depend on `PhasePressures`. Going through `get` evaluates what is needed
    # and caches it, which is what MRST's property access does too.
    shrinkage = getattr(groupings, "ShrinkageFactors", None)
    if shrinkage is not None:
        # A grouping that exposes the function directly, MRST-style. Kept as
        # the first branch because it is the shape the existing tests build.
        if hasattr(shrinkage, "use_saturated_flag"):
            shrinkage.use_saturated_flag = True
        return [np.asarray(_value_of(x), dtype=float).ravel()
                for x in shrinkage.evaluate_on_domain(model, substate)]
    # Deliberately *not* pre-initialising the container: this port's
    # `is_evaluated` is `name in container`, and `initStateFunctionContainer`
    # seeds every name with None - so priming it makes `get` believe the value
    # is already computed and hand back None for all of them.
    substate = State(substate)
    return [np.asarray(_value_of(x), dtype=float).ravel()
            for x in groupings.get(model, substate, "ShrinkageFactors")]


def _field(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_field(obj, name, value) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _value_of(x):
    return getattr(x, "val", x)


def _truthy(x) -> bool:
    return bool(np.all(np.asarray(x))) if x is not None else False


__all__ = ["GenericFacilityModel"]
