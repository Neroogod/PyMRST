"""MRST ``ad-core/utils/convertDeckScheduleToMRST.m``.

`convert_deck_schedule_to_mrst` is the port. It builds each control's wells
with `processWells`, adds multisegment data when COMPSEGS and WELSEGS are
given, and maps each well's `compi` from the deck's water/oil/gas(/solvent)
order onto the model's saturation variables. It then makes the schedule
consistent and applies WPIMULT. The result is a PyMRST `Schedule` (0-based
`step_control`), the form the AD-OO simulator runs.

`convert_deck_schedule_to_mrst_ad`, below, is PyMRST's earlier version, kept
for its callers. Its notes follow.

This ports the structural part of MRST ``convertDeckScheduleToMRST.m``:
step/control conversion, active-phase composition mapping, compositional
defaults, step limiting and consistent well ordering between control periods.
Controls may contain processed ``W`` records or raw ``WELSPECS``/``COMPDAT``
geometry with ``WCON*`` operating records. The raw path reuses PyMRST's shared
MRST record parser so compositional and black-oil schedules have one set of
defaulting and completion rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.utils.schedule import Control, Schedule
from pymrst.params.wells_and_bc import Well


def _get(obj: Any, *names, default=None):
    if isinstance(obj, Mapping):
        for name in names:
            for key in (name, name.lower(), name.upper()):
                if key in obj:
                    return obj[key]
    return default


def _list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel())
    return [value]


def _phase_names(model) -> tuple[str, ...]:
    if hasattr(model, "getPhaseNames"):
        return tuple(str(x).upper()[0] for x in model.getPhaseNames())
    phases = getattr(getattr(model, "fluid", None), "phases", ())
    aliases = {"WATER": "W", "OIL": "O", "GAS": "G", "SOLVENT": "S"}
    return tuple(aliases.get(str(x).upper(), str(x).upper()[0]) for x in phases)


def _well(value, index: int) -> Well:
    if isinstance(value, Well):
        return deepcopy(value)
    if not isinstance(value, Mapping):
        raise TypeError("processed control W entries must be Well objects or mappings")
    data = dict(value)
    cells = np.atleast_1d(np.asarray(_get(data, "cells", "cell", default=[]), dtype=np.int64))
    # Processed Python deck records use zero-based cells. Normalized MATLAB
    # structs may opt into conversion explicitly with ``one_based=True``.
    if bool(_get(data, "one_based", default=False)):
        cells = cells - 1

    def array(*names, default=0.0):
        raw = np.atleast_1d(np.asarray(_get(data, *names, default=default), dtype=float))
        return np.full(cells.size, float(raw[0])) if raw.size == 1 and cells.size > 1 else raw

    components = _get(data, "components")
    cstatus = _get(data, "cstatus")
    return Well(
        cells=cells,
        type=str(_get(data, "type", default="unknown")).lower(),
        val=float(np.asarray(_get(data, "val", default=0.0)).reshape(-1)[0]),
        r=array("r", "radius"),
        dir=str(_get(data, "dir", default="z")).lower(),
        WI=array("WI", "wi"),
        dZ=array("dZ", "dz"),
        name=str(_get(data, "name", default=f"W{index + 1}")),
        compi=np.atleast_1d(np.asarray(_get(data, "compi", default=np.ones(1)), float)),
        refDepth=float(np.asarray(_get(data, "refDepth", "ref_depth", default=0.0)).reshape(-1)[0]),
        sign=int(np.asarray(_get(data, "sign", default=0)).reshape(-1)[0]),
        status=bool(np.asarray(_get(data, "status", default=True)).reshape(-1)[0]),
        cstatus=None if cstatus is None else np.atleast_1d(np.asarray(cstatus, bool)),
        components=None if components is None else np.atleast_1d(np.asarray(components, float)),
        lims=_get(data, "lims"),
    )


def _map_composition(well: Well, phase_names: Sequence[str]) -> Well:
    deck_order = {"W": 0, "O": 1, "G": 2, "S": 3}
    raw = np.pad(np.asarray(well.compi, dtype=float).ravel(), (0, 4))[:4]
    mapped = np.array([raw[deck_order[name]] for name in phase_names], dtype=float)
    total = float(mapped.sum())
    if total <= 0:
        raise ValueError(f"well {well.name!r} has zero composition in active phases")
    well.compi = mapped / total
    return well


def _component_count(model) -> int:
    eos = getattr(model, "EOSModel", None)
    if eos is None:
        return 0
    getter = getattr(eos, "getNumberOfComponents", None)
    return int(getter()) if callable(getter) else 0


def _processed_wells(
    ctrl: Mapping[str, Any], model, base_wells: Sequence[Well]
) -> list[Well]:
    if _get(ctrl, "W") is None:
        raw_keys = (
            "WELSPECS", "COMPDAT", "WCONPROD", "WCONHIST",
            "WCONINJE", "WCONINJ", "WCONINJH", "WELOPEN",
            "WELTARG", "WELLTARG",
        )
        if not any(_get(ctrl, key) is not None for key in raw_keys):
            return deepcopy(list(base_wells))
        grid = getattr(model, "G", None)
        rock = getattr(model, "rock", None)
        if grid is None or rock is None:
            raise ValueError("model.G and model.rock are required for raw well records")
        # These parsers already implement MRST processWells record defaulting,
        # completion expansion, WI construction, and control/limit semantics.
        from pymrst.modules.compositional.facility import (
            apply_deck_well_controls,
            wells_from_deck_completion_records,
        )
        gravity = float(getattr(
            model, "gravity_magnitude", getattr(model, "gravity", 0.0)
        ))
        geometry = None
        if _get(ctrl, "WELSPECS") is not None or _get(ctrl, "COMPDAT") is not None:
            geometry = wells_from_deck_completion_records(
                grid, rock, ctrl, compi=np.array([1.0, 0.0, 0.0]),
                gravity=np.array([0.0, 0.0, gravity]),
            )
        wells = list(base_wells)
        if geometry:
            by_name = {well.name: deepcopy(well) for well in wells}
            by_name.update({well.name: well for well in geometry})
            order = [well.name for well in wells]
            order.extend(well.name for well in geometry if well.name not in order)
            wells = [by_name[name] for name in order]
        wells = apply_deck_well_controls(wells, ctrl, grid=grid)
        phases = _phase_names(model)
        n_hc = _component_count(model)
        for well in wells:
            _map_composition(well, phases)
            if n_hc and well.components is None:
                well.components = np.full(n_hc, 1.0 / n_hc)
        return wells
    phases = _phase_names(model)
    n_hc = _component_count(model)
    wells = []
    for index, item in enumerate(_list(_get(ctrl, "W"))):
        well = _map_composition(_well(item, index), phases)
        if n_hc and well.components is None:
            well.components = np.full(n_hc, 1.0 / n_hc)
        wells.append(well)
    return wells


def convert_deck_schedule_to_mrst_ad(
    model,
    schedule_deck,
    *,
    step_limit: Optional[int] = None,
    ensure_consistent: bool = True,
) -> Schedule:
    """Convert deck-style ``step``/``control`` data into a :class:`Schedule`.

    MATLAB one-based control indices are converted to Python zero-based indices.
    """
    nested = _get(schedule_deck, "SCHEDULE")
    if nested is not None:
        schedule_deck = nested
    step = _get(schedule_deck, "step")
    if step is None and _get(schedule_deck, "step_val") is not None:
        step = {
            "val": _get(schedule_deck, "step_val"),
            "control": _get(schedule_deck, "step_control"),
        }
    if step is None:
        raise KeyError("schedule deck must contain step.val and step.control")
    values = np.asarray(_get(step, "val"), dtype=float).ravel()
    indices = np.asarray(_get(step, "control"), dtype=np.int64).ravel()
    if values.size != indices.size:
        raise ValueError("step.val and step.control must have equal length")

    raw_controls = _list(_get(schedule_deck, "control", "controls", default=[]))
    controls = []
    previous_wells: list[Well] = []
    for ctrl in raw_controls:
        if not isinstance(ctrl, Mapping):
            raise TypeError("each deck control must be a mapping")
        controls.append(Control(
            W=_processed_wells(ctrl, model, previous_wells),
            bc=_get(ctrl, "bc"),
            src=_get(ctrl, "src"),
        ))
        previous_wells = deepcopy(list(controls[-1].W or []))
    if indices.size and indices.min() >= 1 and indices.max() <= len(controls):
        indices = indices - 1
    if step_limit is not None:
        if step_limit < 0:
            raise ValueError("step_limit must be non-negative")
        values, indices = values[:step_limit], indices[:step_limit]
    schedule = Schedule(values, indices, controls)
    if ensure_consistent:
        from pymrst.modules.ad_core.utils.make_schedule_consistent import (
            make_schedule_consistent_ad,
        )
        schedule = make_schedule_consistent_ad(schedule)
    return schedule


#: MRST silences these while building wells; matched against the message.
SKIP_WARNINGS = ("RefDepth:BelowTopConnection", "RepRad:FaceGeomMissing")


def convert_deck_schedule_to_mrst(model, scheduleDeck, *, StepLimit: float = np.inf,
                                  useCpGeometry: Optional[bool] = None,
                                  DepthReorder: bool = False, setDepths: bool = False,
                                  skipWarnings: Sequence[str] = SKIP_WARNINGS,
                                  ReorderStrategy: Sequence[str] = (),
                                  EnsureConsistent: bool = True,
                                  **wellArg) -> Schedule:
    """MRST `convertDeckScheduleToMRST`: the deck's schedule for `model`."""
    import warnings

    from pymrst.modules.ad_core.utils.make_schedule_consistent import (
        make_schedule_consistent,
    )
    from pymrst.modules.deckformat.params.wells_and_bc.process_wells import (
        process_wells,
    )
    G = model.G
    if useCpGeometry is None:
        # MRST: `size(G.cells.faces, 2) > 1`; PyMRST keeps that column in
        # `G.cells.tags`.
        useCpGeometry = getattr(G.cells, "tags", None) is not None
    cell_dims = None
    if _get(scheduleDeck, "RUNSPEC") is not None and _get(scheduleDeck, "SCHEDULE") is not None:
        grid = _get(scheduleDeck, "GRID")
        if useCpGeometry and grid is not None and _get(grid, "COORD") is not None:
            from pymrst.modules.ad_core.utils.compute_cp_geometry import (
                compute_cp_geometry,
            )
            _, _, cell_dims = compute_cp_geometry(G, grid)
            cell_dims = np.asarray(cell_dims)[np.asarray(G.cells.indexMap, dtype=np.int64)]
        elif all(hasattr(G.cells, f) for f in ("DX", "DY", "DZ")):
            cell_dims = np.column_stack([G.cells.DX, G.cells.DY, G.cells.DZ])
        scheduleDeck = _get(scheduleDeck, "SCHEDULE")
    step = _schedule_field(scheduleDeck, "step")
    step_val = np.asarray(_get(step, "val"), dtype=float).ravel()
    step_control = np.asarray(_get(step, "control"), dtype=np.int64).ravel() - 1
    deck_controls = _list(_schedule_field(scheduleDeck, "control"))
    is_compositional = any(c.__name__ == "ThreePhaseCompositionalModel"
                           for c in type(model).__mro__)
    if is_compositional:
        n_hc = model.EOSModel.getNumberOfComponents()
    phase_map = []
    for name in model.get_saturation_var_names():
        index = {"sw": 0, "so": 1, "sg": 2, "ss": 3}.get(name.lower())
        if index is None:
            index = 3
            warnings.warn("Unknown phase, translation directly from deck "
                          "difficult, setting zero")
        phase_map.append(index)
    controls = []
    with warnings.catch_warnings():
        for pattern in skipWarnings:
            warnings.filterwarnings("ignore", message=f".*{pattern}.*")
        for ctrl in deck_controls:
            W = process_wells(G, model.rock, ctrl, outputDefaulted=True,
                              DepthReorder=DepthReorder, cellDims=cell_dims, **wellArg)
            if "COMPSEGS" in ctrl and "WELSEGS" in ctrl:
                from pymrst.modules.ad_core.utils.convert_multi_segment_wells_deck import (
                    convert_multi_segment_wells_deck,
                )
                W = convert_multi_segment_wells_deck(W, G, ctrl)
            for w in W:
                c = np.append(np.asarray(w.compi, dtype=float).ravel(), 0.0)
                if hasattr(w, "solventFrac"):
                    c[3] = c[2]*w.solventFrac
                    c[2] = c[2]*(1 - w.solventFrac)
                ci = c[phase_map]
                w.compi = ci/ci.sum()
                if is_compositional and getattr(w, "components", None) is None:
                    w.components = np.ones(n_hc)/n_hc
                if hasattr(w, "isMS") and w.isMS is None:
                    w.isMS = False
            for w in W:
                if hasattr(w, "solventFrac"):
                    del w.solventFrac
            controls.append(Control(W=W, bc=None, src=None))
    if np.isfinite(StepLimit):
        step_val, step_control = step_val[:int(StepLimit)], step_control[:int(StepLimit)]
    schedule = Schedule(step_val, step_control, controls)
    if EnsureConsistent:
        schedule = make_schedule_consistent(schedule, DepthReorder=DepthReorder,
                                            setDepths=setDepths,
                                            ReorderStrategy=ReorderStrategy, G=G)
        if deck_controls and "WPIMULT" in deck_controls[0]:
            if any(_list(c.get("WPIMULT")) for c in deck_controls):
                from pymrst.utils.gridtools.grid_logical_indices import (
                    grid_logical_indices,
                )
                ijk = [x + 1 for x in grid_logical_indices(G)]          # 1-based
                wi_prev = [np.asarray(w.WI, dtype=float).copy()
                           for w in schedule.controls[0].W]
                wi_raw = [x.copy() for x in wi_prev]
                for k, deck_ctrl in enumerate(deck_controls):
                    W = schedule.controls[k].W
                    W, wi_raw, wi_prev = _apply_wpimult(W, ijk, deck_ctrl["WPIMULT"],
                                                        wi_raw, wi_prev)
                    schedule.controls[k].W = W
    return schedule


def _schedule_field(schedule_deck, name: str):
    """`scheduleDeck.(name)` for a deck `Schedule` object or a mapping."""
    if isinstance(schedule_deck, Mapping):
        return _get(schedule_deck, name)
    return getattr(schedule_deck, name)


def _wpimult_tables(wpimult) -> list:
    """WPIMULT as MRST's cell array of tables (one per keyword occurrence)."""
    tables = _list(wpimult)
    if tables and isinstance(tables[0], (list, tuple)) and tables[0] \
            and isinstance(tables[0][0], str):
        return [tables]
    return tables


def _apply_wpimult(W, ijk, wpimult, wi_raw, wi_now):
    """Scale the well indices of the named perforations, compounding over
    controls the way ECLIPSE's WPIMULT does."""
    names = [w.name for w in W]
    for k, w in enumerate(W):
        current = np.asarray(w.WI, dtype=float)
        new_perf = (wi_raw[k] != current) & (current > 0)
        wi_raw[k] = wi_raw[k].copy()
        wi_now[k] = wi_now[k].copy()
        wi_raw[k][new_perf] = current[new_perf]
        wi_now[k][new_perf] = current[new_perf]
    for table in _wpimult_tables(wpimult):
        for name, val, I, J, K, start, stop in table:
            matches = [k for k, n in enumerate(names) if n == name]
            assert len(matches) == 1
            assert val > 1e-10 and np.isfinite(val)
            k = matches[0]
            wc = np.asarray(W[k].cells, dtype=np.int64)
            n = np.size(W[k].WI)
            start = 1 if start < 1 else start
            stop = n if stop < 1 else stop
            rng = np.arange(1, n + 1)
            active = (rng >= start) & (rng <= stop)
            for index, value in zip(ijk, (I, J, K)):
                if value > 0:
                    active &= index[wc] == value
            wi = wi_now[k].copy()
            wi[active] = wi[active]*val
            wi_now[k] = wi
    for k, w in enumerate(W):
        w.WI = wi_now[k]*np.asarray(w.cstatus, dtype=float)
    return W, wi_raw, wi_now


convertDeckScheduleToMRST = convert_deck_schedule_to_mrst

__all__ = ["convert_deck_schedule_to_mrst", "convert_deck_schedule_to_mrst_ad",
           "convertDeckScheduleToMRST"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock

    def _model(phases=("sw", "so", "sg")):
        G = compute_geometry(cart_grid([3, 3, 2], [30.0, 30.0, 20.0]))
        return SimpleNamespace(G=G, rock=make_rock(G, 1e-13, 0.2),
                               get_saturation_var_names=lambda: list(phases))

    def _deck(wpimult=()):
        nan = float("nan")
        control = {
            "WELSPECS": [["P", "G", 3, 3, nan, "OIL"], ["I", "G", 1, 1, nan, "WATER"]],
            "COMPDAT": [["P", 3, 3, 1, 2, "OPEN", 0, -1.0, 0.2, -1, 0.0, "D", "Z", -1],
                        ["I", 1, 1, 1, 2, "OPEN", 0, -1.0, 0.2, -1, 0.0, "D", "Z", -1]],
            "WCONPROD": [["P", "OPEN", "BHP", 0, 0, 0, 0, 0, 1e7, 0, 0]],
            "WCONINJE": [["I", "WATER", "OPEN", "RATE", 1e-3, nan, nan, nan, 0]],
            "WPIMULT": list(wpimult),
        }
        return {"step": {"val": [86400.0, 86400.0], "control": [1, 1]},
                "control": [control]}

    def test_steps_become_zero_based_and_injectors_come_first():
        schedule = convert_deck_schedule_to_mrst(_model(), _deck())
        assert schedule.step_control.tolist() == [0, 0]
        assert [w.name for w in schedule.controls[0].W] == ["I", "P"]
        print("ok test_steps_become_zero_based_and_injectors_come_first")

    def test_compi_follows_the_models_phases():
        """A two-phase (water, oil) model keeps the deck's water and oil
        fractions, renormalised."""
        schedule = convert_deck_schedule_to_mrst(_model(("sw", "so")), _deck())
        inj = schedule.controls[0].W[0]
        assert np.allclose(inj.compi, [1.0, 0.0])
        print("ok test_compi_follows_the_models_phases")

    def test_wpimult_scales_the_named_wells_index():
        plain = convert_deck_schedule_to_mrst(_model(), _deck())
        scaled = convert_deck_schedule_to_mrst(_model(), _deck([["P", 2.0, 0, 0, 0, 0, 0]]))
        p0, p1 = plain.controls[0].W[1], scaled.controls[0].W[1]
        assert np.allclose(p1.WI, 2*np.asarray(p0.WI))
        assert np.allclose(scaled.controls[0].W[0].WI, plain.controls[0].W[0].WI)
        print("ok test_wpimult_scales_the_named_wells_index")

    test_steps_become_zero_based_and_injectors_come_first()
    test_compi_follows_the_models_phases()
    test_wpimult_scales_the_named_wells_index()
    print("\nAll tests passed!")
