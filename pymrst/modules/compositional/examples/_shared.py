"""Shared routines for runnable compositional example ports."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_core.models.facilities.vfp import VFPTable
from pymrst.modules.compositional.facility import (
    apply_deck_well_controls,
    facility_group_controls_from_deck_control,
    wells_from_deck_completion_records,
)
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.fluids import get_compositional_fluid_case
from pymrst.modules.compositional.model import CompositionalState, OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import (
    BoundaryCondition,
    Source,
    add_bc,
    add_source,
    add_well,
    fluxside,
    pside,
    psideh,
)
from pymrst.utils.units import BARSA, DARCY, MILLI


@dataclass
class SimulationResult:
    """States, iteration counts, and optional MRST-style reports."""

    states: list[CompositionalState]
    iterations: np.ndarray
    reports: list[dict] | None = None

    @property
    def final_state(self) -> CompositionalState:
        return self.states[-1]


@dataclass
class ScheduleControl:
    """One MRST-style schedule control block."""

    wells: list | None = None
    bc: BoundaryCondition | None = None
    src: object | None = None
    separator_group: object | None = None
    group_controls: list | None = None
    vfp_tables_injector: list | None = None
    vfp_tables_producer: list | None = None
    extra: dict | None = None


@dataclass
class ScheduleStep:
    """Report-step sizes and one-based control indices."""

    val: np.ndarray
    control: np.ndarray


@dataclass
class SimpleSchedule:
    """Compact port of MRST ``simpleSchedule`` for compositional examples."""

    step: ScheduleStep
    control: list[ScheduleControl]


@dataclass
class PackedSimulationProblem:
    """Small in-memory analogue of MRST ``packSimulationProblem``.

    MRST's packed problems are disk-backed restartable jobs. The compositional
    example ports only need the structural contract: a named bundle of
    ``state0``, ``model``, ``schedule`` and solver options, later passed to
    ``simulatePackedProblem`` and collected with
    ``getMultiplePackedSimulatorOutputs``.
    """

    state0: CompositionalState
    model: OverallCompositionModel
    schedule: SimpleSchedule
    directory: str
    name: str
    nonlinear_solver: object | None = None
    result: SimulationResult | None = None


def simple_schedule(
    timesteps,
    *,
    bc=None,
    wells=None,
    src=None,
    separator_group=None,
    group_controls=None,
    vfp_tables_injector=None,
    vfp_tables_producer=None,
    control=None,
    step_control=None,
) -> SimpleSchedule:
    """Build a one-control schedule, or accept explicit controls.

    The ``step.control`` array follows MRST's one-based indexing. Solvers convert
    to Python zero-based indices only at the point of use.
    """
    dt = np.atleast_1d(np.asarray(timesteps, dtype=float)).copy()
    if control is None:
        controls = [
            ScheduleControl(
                wells=wells,
                bc=bc,
                src=src,
                separator_group=separator_group,
                group_controls=group_controls,
                vfp_tables_injector=vfp_tables_injector,
                vfp_tables_producer=vfp_tables_producer,
            )
        ]
        ctrl = np.ones(dt.size, dtype=np.int64)
    else:
        known = {
            "wells", "W", "Wells", "bc", "BC", "boundary", "Boundary",
            "boundary_condition", "boundaryCondition", "BoundaryCondition",
            "boundary_conditions", "boundaryConditions", "BoundaryConditions",
            "bc_records", "bcRecords", "BCRecords", "src", "SRC", "source",
            "Source", "sources", "source_term", "sourceTerm", "SourceTerm",
            "source_terms", "sourceTerms", "SourceTerms", "src_terms", "srcTerms",
            "SrcTerms", "src_records", "srcRecords", "SrcRecords",
            "separator_group", "SeparatorGroup", "separatorGroup",
            "group_controls", "groupControls", "GroupControls",
            "vfp_tables_injector", "vfpTablesInjector", "VFPTablesInjector",
            "VFPINJ", "vfpinj", "vfp_tables_producer", "vfpTablesProducer",
            "VFPTablesProducer", "VFPPROD", "vfpprod",
        }
        controls = [
            c if isinstance(c, ScheduleControl) else ScheduleControl(
                wells=_control_get(c, "wells", "W", "Wells"),
                bc=_native_boundary_from_control(c) or _control_get(c, "bc", "BC"),
                src=_native_source_from_control(c) or _control_get(c, "src", "SRC"),
                separator_group=_control_get(c, "separator_group", "separatorGroup", "SeparatorGroup"),
                group_controls=_control_get(c, "group_controls", "groupControls", "GroupControls"),
                vfp_tables_injector=_vfp_tables_from_control(c, producer=False),
                vfp_tables_producer=_vfp_tables_from_control(c, producer=True),
                extra=_extra_without_known_keys(c, known),
            )
            for c in control
        ]
        if step_control is None:
            ctrl = np.ones(dt.size, dtype=np.int64)
        else:
            ctrl = np.asarray(step_control, dtype=np.int64).ravel()
            if ctrl.size != dt.size:
                raise ValueError("step_control must contain one one-based control index per timestep")
            if ctrl.min(initial=1) < 1 or ctrl.max(initial=1) > len(controls):
                raise ValueError("step_control contains an index outside schedule.control")
    return SimpleSchedule(ScheduleStep(dt, ctrl), controls)


def schedule_controls_from_deck_controls(
    base_wells,
    controls,
    *,
    copy_wells: bool = True,
    grid=None,
    rock=None,
    well_type: str = "bhp",
    val: float = 0.0,
    compi=None,
    gravity=None,
    default_radius: float = 0.1,
) -> list[ScheduleControl]:
    """Convert MRST-like ``schedule.control`` blocks to native controls.

    MRST examples often carry well and group updates as deck-style keyword
    fields inside each ``schedule.control`` entry.  This helper preserves that
    structure for Python ports: it applies ``WCON*``/``WELTARG`` records to a
    base set of PyMRST wells, imports group-control rows, and keeps ordinary
    PyMRST fields such as ``bc``, ``src`` and separator controls attached to the
    same control block.
    """
    control_list = list(controls)
    if base_wells is None:
        if grid is None or rock is None:
            raise ValueError("grid and rock are required when base_wells is None")
        for control in control_list:
            if isinstance(control, ScheduleControl):
                wells = control.wells
            else:
                wells = _control_get(control, "wells", "W", "Wells")
                if wells is None and (
                    _control_get(control, "COMPDAT", "compdat") is not None
                    or _control_get(control, "WELSPECS", "welspecs") is not None
                ):
                    wells = wells_from_deck_completion_records(
                        grid,
                        rock,
                        control,
                        well_type=well_type,
                        val=val,
                        compi=compi,
                        gravity=gravity,
                        default_radius=default_radius,
                    )
            if wells is not None:
                base_wells = list(wells)
                break
        else:
            base_wells = []

    out: list[ScheduleControl] = []
    for control in control_list:
        if isinstance(control, ScheduleControl):
            out.append(copy.deepcopy(control) if copy_wells else control)
            continue
        wells = _control_get(control, "wells", "W", "Wells")
        if wells is None:
            wells = apply_deck_well_controls(base_wells, control, copy_wells=copy_wells, grid=grid)
        else:
            wells = copy.deepcopy(list(wells)) if copy_wells else list(wells)

        group_controls = _control_get(control, "group_controls", "groupControls", "GroupControls")
        if group_controls is None:
            group_controls = facility_group_controls_from_deck_control(control)
        else:
            group_controls = copy.deepcopy(list(group_controls)) if copy_wells else list(group_controls)
        vfp_tables_injector, vfp_tables_producer = vfp_tables_from_deck_control(control)

        known = {
            "wells", "W", "Wells", "bc", "BC", "boundary", "Boundary",
            "boundary_condition", "boundaryCondition", "BoundaryCondition",
            "boundary_conditions", "boundaryConditions", "BoundaryConditions",
            "bc_records", "bcRecords", "BCRecords", "src", "SRC", "source",
            "Source", "sources", "source_term", "sourceTerm", "SourceTerm",
            "source_terms", "sourceTerms", "SourceTerms", "src_terms", "srcTerms",
            "SrcTerms", "src_records", "srcRecords", "SrcRecords",
            "separator_group", "SeparatorGroup", "separatorGroup",
            "group_controls", "groupControls", "GroupControls",
            "vfp_tables_injector", "vfpTablesInjector", "VFPTablesInjector",
            "VFPINJ", "vfpinj", "vfp_tables_producer", "vfpTablesProducer",
            "VFPTablesProducer", "VFPPROD", "vfpprod",
        }
        out.append(
            ScheduleControl(
                wells=wells,
                bc=_native_boundary_from_control(control) or _control_get(control, "bc", "BC"),
                src=_native_source_from_control(control) or _control_get(control, "src", "SRC"),
                separator_group=_control_get(
                    control,
                    "separator_group",
                    "separatorGroup",
                    "SeparatorGroup",
                ),
                group_controls=group_controls,
                vfp_tables_injector=vfp_tables_injector,
                vfp_tables_producer=vfp_tables_producer,
                extra=_extra_without_known_keys(control, known),
            )
        )
    return out


def simple_schedule_from_deck_controls(
    timesteps,
    base_wells,
    controls,
    *,
    step_control=None,
    copy_wells: bool = True,
    grid=None,
    rock=None,
    well_type: str = "bhp",
    val: float = 0.0,
    compi=None,
    gravity=None,
    default_radius: float = 0.1,
) -> SimpleSchedule:
    """Build a ``SimpleSchedule`` from MRST-like control dictionaries."""
    return simple_schedule(
        timesteps,
        control=schedule_controls_from_deck_controls(
            base_wells,
            controls,
            copy_wells=copy_wells,
            grid=grid,
            rock=rock,
            well_type=well_type,
            val=val,
            compi=compi,
            gravity=gravity,
            default_radius=default_radius,
        ),
        step_control=step_control,
    )


_DECK_CONTROL_KEYS = {
    "WCONPROD", "WCONINJE", "WCONINJ", "WCONHIST", "WCONINJH",
    "WELTARG", "WELOPEN", "WELSPECS", "COMPDAT",
    "GCONPROD", "GCONINJE", "GRUPTREE", "WGRUPCON", "GECON", "GRUPNET",
    "VFPINJ", "VFPPROD",
}


_FORCING_CONTROL_KEYS = {
    "bc", "BC", "boundary", "Boundary", "boundary_condition",
    "boundaryCondition", "BoundaryCondition", "boundary_conditions",
    "boundaryConditions", "bc_records", "bcRecords", "side", "face_side",
    "faceSide", "FaceSide", "side_name", "sideName", "SideName",
    "pressure_side", "pressureSide", "PressureSide",
    "pside", "pSide", "PSide", "psideh", "pSideH", "PSideH",
    "hydrostatic_side", "hydrostaticSide", "HydrostaticSide",
    "flux_side", "fluxSide", "FluxSide",
    "faces", "face", "type",
    "face_index", "faceIndex", "face_indices", "faceIndices",
    "bc_type", "kind", "value", "values", "val", "vals", "pressure", "pres", "p", "bhp",
    "flux", "rate", "q", "massrate", "mass_rate", "sat", "saturation",
    "saturations", "s", "S", "components", "component", "z",
    "zi", "composition", "compositions", "compi", "comp_i",
    "composition_i", "component_i", "i1", "i2", "range", "Range",
    "search_range", "searchRange", "fluid", "Fluid", "ref_position",
    "refPosition", "ref_pressure", "refPressure", "ref_depth", "refDepth",
    "gravity", "Gravity", "pn", "src", "SRC", "source",
    "Source", "sources", "source_term", "sourceTerm", "SourceTerm",
    "source_terms", "sourceTerms", "src_terms", "srcTerms", "SrcTerms",
    "src_records", "srcRecords", "cells", "cell", "cell_index", "cellIndex",
    "cell_indices", "cellIndices",
}


def _mapping_has_any_key(obj, keys: set[str]) -> bool:
    obj = _struct_to_mapping(obj)
    if not isinstance(obj, Mapping):
        return False
    present = {str(k) for k in obj.keys()}
    present_upper = {k.upper() for k in present}
    return bool(present & keys or present_upper & {k.upper() for k in keys})


def _struct_field_names(obj) -> list[str]:
    if isinstance(obj, Mapping):
        return [str(k) for k in obj.keys()]
    names = getattr(obj, "_fieldnames", None)
    if names is not None:
        return [str(name) for name in names]
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return [str(k) for k in data.keys() if not str(k).startswith("_")]
    return []


def _struct_get(obj, *names, default=None):
    if isinstance(obj, Mapping):
        lower = {str(k).lower(): k for k in obj.keys()}
        for name in names:
            key = lower.get(str(name).lower())
            if key is not None:
                return obj[key]
        return default
    if obj is None:
        return default
    lower = {name.lower(): name for name in _struct_field_names(obj)}
    for name in names:
        actual = lower.get(str(name).lower())
        if actual is not None and hasattr(obj, actual):
            return getattr(obj, actual)
    return default


def _struct_to_mapping(obj):
    if isinstance(obj, Mapping) or obj is None:
        return obj
    names = _struct_field_names(obj)
    if not names:
        return obj
    return {name: getattr(obj, name) for name in names if hasattr(obj, name)}


def _schedule_step_mapping(schedule):
    step = _struct_get(schedule, "step", "STEP", default={})
    step = _struct_to_mapping(step)
    return step if isinstance(step, Mapping) else {}


def _schedule_step_values(schedule):
    step = _schedule_step_mapping(schedule)
    value = (
        _struct_get(step, "val", "values", "dt", default=None)
        if step else None
    )
    if value is None:
        value = _struct_get(
            schedule,
            "timesteps",
            "dt",
            "step_values",
            "stepValues",
            default=None,
        )
    if value is None:
        raise ValueError("MRST-like schedule must provide step.val or timesteps")
    return value


def _schedule_step_controls(schedule):
    step = _schedule_step_mapping(schedule)
    control = _struct_get(step, "control", "controls", default=None) if step else None
    if control is None:
        control = _struct_get(schedule, "step_control", "stepControl", default=None)
    return control


def _schedule_controls(schedule):
    controls = _struct_get(schedule, "control", "controls", "CONTROL", default=None)
    if controls is None:
        raise ValueError("MRST-like schedule must provide a control list")
    if isinstance(controls, (Mapping, ScheduleControl)):
        return [controls]
    controls = _struct_to_mapping(controls)
    if isinstance(controls, Mapping):
        return [controls]
    if isinstance(controls, np.ndarray):
        if controls.size == 0:
            return []
        if controls.dtype == object:
            return [_struct_to_mapping(item) for item in controls.ravel().tolist()]
    return list(controls)


def _strip_forcing_keys(extra: dict | None) -> dict | None:
    if not extra:
        return extra
    filtered = {
        k: v for k, v in extra.items()
        if str(k) not in _FORCING_CONTROL_KEYS
        and str(k).upper() not in {kk.upper() for kk in _FORCING_CONTROL_KEYS}
    }
    return filtered or None


def _extra_without_known_keys(control: Mapping, known: set[str]) -> dict | None:
    """Return metadata after consuming MRST-like keys case-insensitively."""
    known_lower = {str(k).lower() for k in known}
    extra = {
        k: v for k, v in control.items()
        if str(k).lower() not in known_lower
    }
    return extra or None


def simple_schedule_from_mrst_schedule(
    schedule,
    *,
    grid=None,
    base_wells=None,
    rock=None,
    copy_wells: bool = True,
    well_type: str = "bhp",
    val: float = 0.0,
    compi=None,
    gravity=None,
    default_radius: float = 0.1,
    default_bc_sat=1.0,
    default_bc_components=None,
    default_src_sat=1.0,
    default_src_components=None,
) -> SimpleSchedule:
    """Build a native schedule from an MRST-like ``schedule`` mapping.

    Exact MRST example ports usually have the same three fields:
    ``schedule.step.val``, ``schedule.step.control`` and ``schedule.control``.
    This helper accepts that shape directly and delegates the heavy lifting to
    the existing deck, boundary and source converters.  Control indices remain
    MRST one-based.
    """
    if isinstance(schedule, SimpleSchedule):
        return copy.deepcopy(schedule)
    schedule = _struct_to_mapping(schedule)
    if not isinstance(schedule, Mapping):
        raise TypeError("MRST-like schedule must be a mapping, MATLAB-style struct or SimpleSchedule")

    timesteps = _schedule_step_values(schedule)
    step_control = _schedule_step_controls(schedule)
    raw_controls = [_struct_to_mapping(control) for control in _schedule_controls(schedule)]
    needs_deck = base_wells is not None or any(
        _mapping_has_any_key(control, _DECK_CONTROL_KEYS)
        for control in raw_controls
    )

    if needs_deck:
        controls = schedule_controls_from_deck_controls(
            base_wells,
            raw_controls,
            copy_wells=copy_wells,
            grid=grid,
            rock=rock,
            well_type=well_type,
            val=val,
            compi=compi,
            gravity=gravity,
            default_radius=default_radius,
        )
    else:
        controls = simple_schedule(
            timesteps,
            control=raw_controls,
            step_control=step_control,
        ).control

    if grid is not None:
        for raw, control in zip(raw_controls, controls):
            if not isinstance(raw, Mapping):
                continue
            bc = boundary_condition_from_control(
                grid,
                raw,
                default_sat=default_bc_sat,
                default_components=default_bc_components,
            )
            src = source_from_control(
                raw,
                default_sat=default_src_sat,
                default_components=default_src_components,
            )
            if bc is not None:
                control.bc = bc
            if src is not None:
                control.src = src
            control.extra = _strip_forcing_keys(control.extra)

    return simple_schedule(timesteps, control=controls, step_control=step_control)


def _records(obj) -> list:
    if obj is None:
        return []
    obj = _struct_to_mapping(obj)
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return []
        if obj.dtype == object:
            if obj.ndim == 0:
                return _records(obj.item())
            flat = obj.ravel().tolist()
            if any(
                isinstance(
                    _struct_to_mapping(item),
                    (Mapping, ScheduleControl, BoundaryCondition, Source),
                )
                for item in flat
            ):
                return [_struct_to_mapping(item) for item in flat]
            if obj.ndim == 1:
                return [obj.tolist()]
            return [row.tolist() for row in obj]
        if obj.ndim == 1:
            return [obj.tolist()]
        return [row.tolist() for row in obj]
    if isinstance(obj, Mapping):
        return [obj]
    if isinstance(obj, (str, bytes)):
        return [obj]
    return list(obj)


def _record_get(record, names, index=None, default=None):
    record = _struct_to_mapping(record)
    if isinstance(record, Mapping):
        lower = {str(k).lower(): v for k, v in record.items()}
        for name in names:
            key = str(name).lower()
            if key in lower:
                return lower[key]
        return default
    if isinstance(record, (str, bytes)):
        return record if index == 0 else default
    if index is None:
        return default
    try:
        return record[index]
    except (IndexError, TypeError):
        return default


def _control_get(control: Mapping, *names, default=None):
    control = _struct_to_mapping(control)
    if not isinstance(control, Mapping):
        return default
    lower = {str(k).lower(): v for k, v in control.items()}
    for name in names:
        key = str(name).lower()
        if key in lower:
            return lower[key]
    return default


_MISSING = object()


def _native_boundary_from_control(control) -> BoundaryCondition | None:
    value = _control_get(
        control,
        "bc",
        "BC",
        "boundary",
        "Boundary",
        "boundary_condition",
        "boundaryCondition",
        "BoundaryCondition",
        "boundary_conditions",
        "boundaryConditions",
        "BoundaryConditions",
        "bc_records",
        "bcRecords",
        "BCRecords",
        default=None,
    )
    return copy.deepcopy(value) if isinstance(value, BoundaryCondition) else None


def _native_source_from_control(control) -> Source | None:
    value = _control_get(
        control,
        "src",
        "SRC",
        "source",
        "Source",
        "sources",
        "source_term",
        "sourceTerm",
        "SourceTerm",
        "source_terms",
        "sourceTerms",
        "SourceTerms",
        "src_terms",
        "srcTerms",
        "SrcTerms",
        "src_records",
        "srcRecords",
        "SrcRecords",
        default=None,
    )
    return copy.deepcopy(value) if isinstance(value, Source) else None


def _is_vfp_deck_table(obj) -> bool:
    obj = _struct_to_mapping(obj)
    if not isinstance(obj, Mapping):
        return False
    keys = {str(k).upper() for k in obj.keys()}
    return "FLO" in keys and "THP" in keys and ("Q" in keys or "BHP" in keys)


def _as_vfp_table(obj) -> VFPTable:
    if isinstance(obj, VFPTable):
        return obj
    return VFPTable.from_deck(obj)


def _set_sparse_table(tables: list, table_no: int, table) -> None:
    if table_no <= 0:
        raise ValueError("VFP table numbers are one-based and must be positive")
    while len(tables) < table_no:
        tables.append(None)
    tables[table_no - 1] = _as_vfp_table(table)


def _vfp_tables_from_records(records) -> list | None:
    if records is None:
        return None
    if isinstance(records, VFPTable) or _is_vfp_deck_table(records):
        return [_as_vfp_table(records)]

    out: list = []
    if isinstance(records, Mapping):
        for key, table in records.items():
            if table is None:
                continue
            if isinstance(table, (list, tuple, np.ndarray)) and len(table) == 0:
                continue
            _set_sparse_table(out, int(key), table)
        return out

    if isinstance(records, np.ndarray):
        if records.size == 0:
            return []
        records = records.tolist()
    for table_no, table in enumerate(list(records), start=1):
        if table is None:
            continue
        if isinstance(table, (list, tuple, np.ndarray)) and len(table) == 0:
            continue
        _set_sparse_table(out, table_no, table)
    return out


def _vfp_tables_from_control(control, *, producer: bool) -> list | None:
    if not isinstance(control, Mapping):
        return None
    if producer:
        value = _control_get(
            control,
            "vfp_tables_producer",
            "vfpTablesProducer",
            "VFPTablesProducer",
            "VFPPROD",
            "vfpprod",
            default=None,
        )
    else:
        value = _control_get(
            control,
            "vfp_tables_injector",
            "vfpTablesInjector",
            "VFPTablesInjector",
            "VFPINJ",
            "vfpinj",
            default=None,
        )
    return _vfp_tables_from_records(value)


def vfp_tables_from_deck_control(control: Mapping) -> tuple[list | None, list | None]:
    """Convert MRST-like ``VFPINJ``/``VFPPROD`` control fields to tables.

    MRST stores VFP tables by one-based table number inside each schedule
    control.  The returned lists preserve that numbering with ``None``
    placeholders for sparse table numbers, so a well with ``vfp_index = 3``
    still binds to Python index 2.
    """
    return (
        _vfp_tables_from_control(control, producer=False),
        _vfp_tables_from_control(control, producer=True),
    )


def _boundary_control_records(control) -> list:
    if not isinstance(control, Mapping):
        return []
    value = _control_get(
        control,
        "boundary_conditions",
        "boundaryConditions",
        "BoundaryConditions",
        "boundary",
        "Boundary",
        "boundary_condition",
        "boundaryCondition",
        "BoundaryCondition",
        "bc_records",
        "bcRecords",
        "BCRecords",
        "bc",
        "BC",
        default=_MISSING,
    )
    if value is not _MISSING and value is not None and not isinstance(value, BoundaryCondition):
        return _records(value)
    keys = {str(k).lower() for k in control.keys()}
    if keys & {
        "side", "face_side", "faceside", "side_name", "sidename",
        "pressure_side", "pressureside", "pside", "psideh", "hydrostatic_side",
        "hydrostaticside", "flux_side", "fluxside",
        "faces", "face", "face_index", "faceindex",
        "face_indices", "faceindices", "type", "bc_type", "kind",
        "pressure", "pres", "p", "bhp",
    }:
        return [control]
    return []


def _source_control_records(control) -> list:
    if not isinstance(control, Mapping):
        return []
    value = _control_get(
        control,
        "sources",
        "source",
        "Source",
        "source_term",
        "sourceTerm",
        "SourceTerm",
        "source_terms",
        "sourceTerms",
        "SourceTerms",
        "src_terms",
        "srcTerms",
        "SrcTerms",
        "src_records",
        "srcRecords",
        "SrcRecords",
        "src",
        "SRC",
        default=_MISSING,
    )
    if value is not _MISSING and value is not None and not isinstance(value, Source):
        return _records(value)
    keys = {str(k).lower() for k in control.keys()}
    if keys & {"cells", "cell", "cell_index", "cellindex", "cell_indices", "cellindices"}:
        return [control]
    return []


def _broadcast_components(components, count: int) -> np.ndarray:
    if components is None:
        return np.zeros((count, 0), dtype=float)
    arr = np.atleast_2d(np.asarray(components, dtype=float))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, count, axis=0)
    if arr.shape[0] != count:
        raise ValueError("boundary components must have one row or one row per boundary face")
    return arr


_BC_PRESSURE_ALIASES = {"pressure", "pres", "p", "bhp", "dirichlet"}
_BC_FLUX_ALIASES = {"flux", "rate", "q", "neumann", "massrate", "mass_rate"}
_BC_PRESSURE_VALUE_NAMES = ("pressure", "pres", "p", "bhp")
_BC_FLUX_VALUE_NAMES = ("flux", "rate", "q", "massrate", "mass_rate")
_VALUE_FIELD_NAMES = ("value", "values", "val", "vals")
_SATURATION_FIELD_NAMES = ("sat", "saturation", "saturations", "s")
_SIDE_FIELD_NAMES = ("side", "face_side", "faceSide", "FaceSide", "side_name", "sideName", "SideName")
_BC_PRESSURE_SIDE_FIELD_NAMES = ("pressure_side", "pressureSide", "PressureSide", "pside", "pSide", "PSide")
_BC_HYDROSTATIC_SIDE_FIELD_NAMES = (
    "psideh",
    "pSideH",
    "PSideH",
    "hydrostatic_side",
    "hydrostaticSide",
    "HydrostaticSide",
)
_BC_FLUX_SIDE_FIELD_NAMES = ("flux_side", "fluxSide", "FluxSide")
_FACE_FIELD_NAMES = ("faces", "face", "face_index", "faceIndex", "face_indices", "faceIndices")
_CELL_FIELD_NAMES = ("cells", "cell", "cell_index", "cellIndex", "cell_indices", "cellIndices")
_SOURCE_RATE_VALUE_NAMES = ("rate", "value", "values", "val", "vals", "flux", "q", "massrate", "mass_rate")
_BC_HYDROSTATIC_OPTION_NAMES = {
    "fluid", "Fluid", "range", "Range", "search_range", "searchRange",
    "ref_position", "refPosition", "ref_pressure", "refPressure",
    "ref_depth", "refDepth", "gravity", "Gravity", "pn",
}


def _normalize_bc_type(kind, *, default: str = "flux") -> str:
    if kind is None or kind == "":
        kind = default
    value = str(kind).strip().lower()
    if value in _BC_PRESSURE_ALIASES:
        return "pressure"
    if value in _BC_FLUX_ALIASES:
        return "flux"
    raise ValueError(f"unsupported boundary-control type '{value}'")


def _broadcast_bc_types(kind, count: int, *, default: str = "flux") -> list[str]:
    if kind is None or kind == "":
        out = [_normalize_bc_type(default)] * count
    elif isinstance(kind, str):
        out = [_normalize_bc_type(kind)] * count
    else:
        arr = np.asarray(kind, dtype=object).ravel()
        if arr.size == 1:
            out = [_normalize_bc_type(arr[0])] * count
        elif arr.size == count:
            out = [_normalize_bc_type(v) for v in arr]
        else:
            raise ValueError("boundary-control type must be scalar or one entry per boundary face")
    return out


def _broadcast_bc_values(value, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        return np.full(count, arr[0], dtype=float)
    if arr.size != count:
        raise ValueError("boundary value must be scalar or one entry per boundary face")
    return arr


_COMPONENT_FIELD_NAMES = (
    "components",
    "component",
    "z",
    "zi",
    "composition",
    "compositions",
    "compi",
    "comp_i",
    "composition_i",
    "component_i",
)


def source_from_control(
    control,
    *,
    default_sat=1.0,
    default_components=None,
) -> Source | None:
    """Build a native ``Source`` from an MRST-like source-control dictionary."""
    if isinstance(control, Source):
        return copy.deepcopy(control)
    native = _native_source_from_control(control)
    if native is not None:
        return native

    records = _source_control_records(control)
    if not records:
        return None

    src = None
    for record in records:
        record = _struct_to_mapping(record)
        if isinstance(record, Mapping):
            cells = _record_get(record, _CELL_FIELD_NAMES, default=None)
            rate = _record_get(record, _SOURCE_RATE_VALUE_NAMES, default=None)
            sat = _record_get(record, _SATURATION_FIELD_NAMES, default=default_sat)
            components = _record_get(
                record,
                _COMPONENT_FIELD_NAMES,
                default=default_components,
            )
        else:
            cells = _record_get(record, ("cells",), 0, None)
            rate = _record_get(record, ("rate",), 1, None)
            sat = _record_get(record, ("sat",), 2, default_sat)
            components = _record_get(record, ("components",), 3, default_components)
        if cells is None:
            raise ValueError("source-control record must provide cells or cell")
        if rate is None:
            raise ValueError("source-control record must provide rate, value, flux, q or massrate")
        src = add_source(
            src,
            np.asarray(cells, dtype=np.int64),
            rate,
            sat=sat,
            components=components,
        )
    return src


def boundary_condition_from_control(
    grid,
    control,
    *,
    default_sat=1.0,
    default_components=None,
) -> BoundaryCondition | None:
    """Build a compositional ``BoundaryCondition`` from a control dictionary.

    Exact MRST example ports often need controls where the boundary stream or
    boundary type changes by report step.  This helper accepts compact records
    such as ``{"type": "flux", "side": "xmin", "value": q, "components": z}``
    and ``{"type": "pressure", "faces": faces, "value": p}``, returning the
    same native ``BoundaryCondition`` object that ``run_schedule`` already
    knows how to switch.
    """
    if isinstance(control, BoundaryCondition):
        return copy.deepcopy(control)
    native = _native_boundary_from_control(control)
    if native is not None:
        return native

    records = _boundary_control_records(control)
    if not records:
        return None

    bc = None
    component_chunks: list[np.ndarray] = []
    component_width = 0
    for record in records:
        record = _struct_to_mapping(record)
        if isinstance(record, Mapping):
            kind_value = _record_get(record, ("type", "bc_type", "kind"), default="")
            kind = str(kind_value).lower() if isinstance(kind_value, str) else ""
            if not kind:
                keys = {str(k).lower() for k in record.keys()}
                kind = "pressure" if keys & set(_BC_PRESSURE_VALUE_NAMES) else "flux"
            kind = _normalize_bc_type(kind)
            side = _record_get(record, _SIDE_FIELD_NAMES, default=None)
            pressure_side = _record_get(record, _BC_PRESSURE_SIDE_FIELD_NAMES, default=None)
            hydrostatic_side = _record_get(record, _BC_HYDROSTATIC_SIDE_FIELD_NAMES, default=None)
            flux_side = _record_get(record, _BC_FLUX_SIDE_FIELD_NAMES, default=None)
            encoded_kind = None
            encoded_hydrostatic = False
            side_aliases = [
                name for name, value_ in (
                    ("pressureSide", pressure_side),
                    ("psideh", hydrostatic_side),
                    ("fluxSide", flux_side),
                )
                if value_ is not None
            ]
            if len(side_aliases) > 1:
                raise ValueError("boundary-control record cannot provide multiple typed side aliases")
            if hydrostatic_side is not None:
                encoded_kind = "pressure"
                encoded_hydrostatic = True
                if side is None:
                    side = hydrostatic_side
            elif pressure_side is not None:
                encoded_kind = "pressure"
                if side is None:
                    side = pressure_side
            elif flux_side is not None:
                encoded_kind = "flux"
                if side is None:
                    side = flux_side
            if encoded_kind is not None:
                if kind_value in ("", None):
                    kind = encoded_kind
                elif kind != encoded_kind:
                    raise ValueError("boundary-control side alias conflicts with boundary type")
            faces = _record_get(record, _FACE_FIELD_NAMES, default=None)
            value = _record_get(record, _VALUE_FIELD_NAMES, default=None)
            if value is None:
                value = _record_get(record, _BC_PRESSURE_VALUE_NAMES, default=None)
            if value is None:
                value = _record_get(record, _BC_FLUX_VALUE_NAMES, default=None)
            sat = _record_get(record, _SATURATION_FIELD_NAMES, default=default_sat)
            components = _record_get(
                record,
                _COMPONENT_FIELD_NAMES,
                default=default_components,
            )
            i1 = _record_get(record, ("i1",), default=None)
            i2 = _record_get(record, ("i2",), default=None)
            hydrostatic_options = _hydrostatic_options_from_record(record, value)
        else:
            first = str(_record_get(record, ("type",), 0, "")).lower()
            first_is_type = False
            try:
                kind = _normalize_bc_type(first)
                first_is_type = True
            except ValueError:
                kind = ""
            if first_is_type:
                side = _record_get(record, ("side",), 1, None)
                value = _record_get(record, ("value",), 2, None)
                sat = _record_get(record, ("sat",), 3, default_sat)
                components = _record_get(record, ("components",), 4, default_components)
            else:
                side = _record_get(record, ("side",), 0, None)
                kind = _normalize_bc_type(_record_get(record, ("type",), 1, "flux"))
                value = _record_get(record, ("value",), 2, None)
                sat = _record_get(record, ("sat",), 3, default_sat)
                components = _record_get(record, ("components",), 4, default_components)
            faces = None
            i1 = None
            i2 = None
            encoded_hydrostatic = False
            hydrostatic_options = {}

        if value is None and not encoded_hydrostatic:
            raise ValueError("boundary-control record must provide a value, pressure, flux or rate")
        before = 0 if bc is None else bc.face.size
        if side is not None:
            if encoded_hydrostatic:
                if "fluid" not in hydrostatic_options:
                    raise ValueError("psideh boundary-control record must provide a fluid")
                fluid = hydrostatic_options.pop("fluid")
                bc = psideh(bc, grid, str(side), fluid, i1=i1, i2=i2, sat=sat, **hydrostatic_options)
            elif kind == "pressure":
                bc = pside(bc, grid, str(side), value, sat=sat, i1=i1, i2=i2)
            elif kind == "flux":
                bc = fluxside(bc, grid, str(side), value, sat=sat, i1=i1, i2=i2)
        else:
            if faces is None:
                raise ValueError("boundary-control record must provide either side or faces")
            face_arr = np.asarray(faces, dtype=np.int64).ravel()
            types = _broadcast_bc_types(kind_value if kind_value not in ("", None) else kind, face_arr.size)
            values = _broadcast_bc_values(value, face_arr.size)
            sat_rows = np.atleast_2d(np.asarray(sat, dtype=float))
            if sat_rows.shape[0] == 1:
                sat_rows = np.repeat(sat_rows, face_arr.size, axis=0)
            if sat_rows.shape[0] != face_arr.size:
                raise ValueError("boundary saturation must have one row or one row per boundary face")
            comp_rows = _broadcast_components(components, face_arr.size)
            for face, typ, val, sat_row, comp_row in zip(face_arr, types, values, sat_rows, comp_rows):
                bc = add_bc(bc, np.asarray([face], dtype=np.int64), typ, np.asarray([val]), sat=sat_row)
                component_width = max(component_width, comp_row.reshape(1, -1).shape[1])
                component_chunks.append(comp_row.reshape(1, -1))
            continue
        assert bc is not None
        count = bc.face.size - before
        chunk = _broadcast_components(components, count)
        component_width = max(component_width, chunk.shape[1])
        component_chunks.append(chunk)

    if bc is not None and component_width:
        padded = []
        for chunk in component_chunks:
            if chunk.shape[1] == component_width:
                padded.append(chunk)
            elif chunk.shape[1] == 0:
                padded.append(np.zeros((chunk.shape[0], component_width), dtype=float))
            else:
                raise ValueError("boundary component vectors must have a consistent width")
        bc.components = np.vstack(padded)
    return bc


def _hydrostatic_options_from_record(record: Mapping, value) -> dict:
    opts = {}
    fluid = _record_get(record, ("fluid", "Fluid"), default=None)
    if fluid is not None:
        opts["fluid"] = fluid
    for out_name, names in (
        ("search_range", ("search_range", "searchRange", "range", "Range")),
        ("ref_position", ("ref_position", "refPosition")),
        ("ref_depth", ("ref_depth", "refDepth")),
        ("gravity", ("gravity", "Gravity")),
    ):
        found = _record_get(record, names, default=_MISSING)
        if found is not _MISSING:
            opts[out_name] = found
    ref_pressure = _record_get(record, ("ref_pressure", "refPressure", "pn"), default=_MISSING)
    if ref_pressure is not _MISSING:
        opts["ref_pressure"] = ref_pressure
    elif value is not None:
        opts["ref_pressure"] = value
    return opts


def schedule_controls_from_boundary_controls(
    grid,
    controls,
    *,
    default_sat=1.0,
    default_components=None,
) -> list[ScheduleControl]:
    """Convert MRST-like boundary-control dictionaries to schedule controls."""
    out: list[ScheduleControl] = []
    for control in list(controls):
        if isinstance(control, ScheduleControl):
            out.append(copy.deepcopy(control))
            continue
        if isinstance(control, BoundaryCondition):
            out.append(ScheduleControl(bc=copy.deepcopy(control)))
            continue
        if not isinstance(control, Mapping):
            raise TypeError("boundary schedule controls must be mappings, BoundaryCondition or ScheduleControl")
        bc = boundary_condition_from_control(
            grid,
            control,
            default_sat=default_sat,
            default_components=default_components,
        )
        known = {
            "wells", "W", "Wells", "bc", "BC", "boundary", "Boundary",
            "boundary_condition", "boundaryCondition", "BoundaryCondition",
            "boundary_conditions", "boundaryConditions", "BoundaryConditions",
            "bc_records", "bcRecords", "BCRecords", "side", "face_side",
            "faceSide", "FaceSide", "side_name", "sideName", "SideName",
            "pressure_side", "pressureSide", "PressureSide",
            "pside", "pSide", "PSide", "psideh", "pSideH", "PSideH",
            "hydrostatic_side", "hydrostaticSide", "HydrostaticSide",
            "flux_side", "fluxSide", "FluxSide", "faces", "face",
            "face_index", "faceIndex", "face_indices", "faceIndices", "type",
            "bc_type", "kind", "value", "values", "val", "vals", "pressure", "pres", "p",
            "bhp", "flux", "rate", "q", "massrate", "mass_rate",
            "sat", "saturation", "saturations", "s", "S", "components", "component", "z",
            "zi", "composition", "compositions", "compi", "comp_i",
            "composition_i", "component_i", "i1", "i2", "range", "Range",
            "search_range", "searchRange", "fluid", "Fluid", "ref_position",
            "refPosition", "ref_pressure", "refPressure", "ref_depth",
            "refDepth", "gravity", "Gravity", "pn", "src", "SRC",
            "source", "Source", "sources", "source_term", "sourceTerm",
            "SourceTerm", "source_terms", "sourceTerms", "SourceTerms",
            "src_terms", "srcTerms", "SrcTerms", "src_records", "srcRecords",
            "SrcRecords", "cell_index", "cellIndex", "cell_indices", "cellIndices",
            "separator_group", "separatorGroup", "SeparatorGroup",
            "group_controls", "groupControls", "GroupControls",
            "vfp_tables_injector", "vfpTablesInjector", "VFPTablesInjector",
            "VFPINJ", "vfpinj", "vfp_tables_producer", "vfpTablesProducer",
            "VFPTablesProducer", "VFPPROD", "vfpprod",
        }
        out.append(
            ScheduleControl(
                wells=_control_get(control, "wells", "W", "Wells"),
                bc=bc,
                src=_native_source_from_control(control),
                separator_group=_control_get(
                    control,
                    "separator_group",
                    "separatorGroup",
                    "SeparatorGroup",
                ),
                group_controls=_control_get(control, "group_controls", "groupControls", "GroupControls"),
                vfp_tables_injector=_vfp_tables_from_control(control, producer=False),
                vfp_tables_producer=_vfp_tables_from_control(control, producer=True),
                extra=_extra_without_known_keys(control, known),
            )
        )
    return out


def simple_schedule_from_boundary_controls(
    timesteps,
    grid,
    controls,
    *,
    step_control=None,
    default_sat=1.0,
    default_components=None,
) -> SimpleSchedule:
    """Build a ``SimpleSchedule`` from MRST-like boundary controls."""
    return simple_schedule(
        timesteps,
        control=schedule_controls_from_boundary_controls(
            grid,
            controls,
            default_sat=default_sat,
            default_components=default_components,
        ),
        step_control=step_control,
    )


def schedule_controls_from_source_controls(
    controls,
    *,
    default_sat=1.0,
    default_components=None,
) -> list[ScheduleControl]:
    """Convert MRST-like source-control dictionaries to schedule controls."""
    out: list[ScheduleControl] = []
    for control in list(controls):
        if isinstance(control, ScheduleControl):
            out.append(copy.deepcopy(control))
            continue
        if isinstance(control, Source):
            out.append(ScheduleControl(src=copy.deepcopy(control)))
            continue
        if not isinstance(control, Mapping):
            raise TypeError("source schedule controls must be mappings, Source or ScheduleControl")
        src = source_from_control(
            control,
            default_sat=default_sat,
            default_components=default_components,
        )
        known = {
            "wells", "W", "Wells", "bc", "BC", "boundary", "Boundary",
            "boundary_condition", "boundaryCondition", "BoundaryCondition",
            "boundary_conditions", "boundaryConditions", "BoundaryConditions",
            "bc_records", "bcRecords", "BCRecords",
            "src", "SRC", "sources", "source", "Source", "source_term",
            "sourceTerm", "SourceTerm", "source_terms", "sourceTerms",
            "SourceTerms", "src_terms", "srcTerms", "SrcTerms", "src_records",
            "srcRecords", "SrcRecords", "cells", "cell", "cell_index", "cellIndex",
            "cell_indices", "cellIndices",
            "rate", "value", "values", "val", "vals", "flux", "q", "massrate", "mass_rate",
            "sat", "saturation", "saturations", "s", "S",
            "components", "component", "z", "zi", "composition",
            "compositions", "compi", "comp_i", "composition_i",
            "component_i",
            "separator_group", "separatorGroup", "SeparatorGroup",
            "group_controls", "groupControls", "GroupControls",
            "vfp_tables_injector", "vfpTablesInjector", "VFPTablesInjector",
            "VFPINJ", "vfpinj", "vfp_tables_producer", "vfpTablesProducer",
            "VFPTablesProducer", "VFPPROD", "vfpprod",
        }
        out.append(
            ScheduleControl(
                wells=_control_get(control, "wells", "W", "Wells"),
                bc=_native_boundary_from_control(control),
                src=src,
                separator_group=_control_get(
                    control,
                    "separator_group",
                    "separatorGroup",
                    "SeparatorGroup",
                ),
                group_controls=_control_get(control, "group_controls", "groupControls", "GroupControls"),
                vfp_tables_injector=_vfp_tables_from_control(control, producer=False),
                vfp_tables_producer=_vfp_tables_from_control(control, producer=True),
                extra=_extra_without_known_keys(control, known),
            )
        )
    return out


def simple_schedule_from_source_controls(
    timesteps,
    controls,
    *,
    step_control=None,
    default_sat=1.0,
    default_components=None,
) -> SimpleSchedule:
    """Build a ``SimpleSchedule`` from MRST-like source controls."""
    return simple_schedule(
        timesteps,
        control=schedule_controls_from_source_controls(
            controls,
            default_sat=default_sat,
            default_components=default_components,
        ),
        step_control=step_control,
    )


def schedule_controls_from_forcing_controls(
    grid,
    controls,
    *,
    default_bc_sat=1.0,
    default_bc_components=None,
    default_src_sat=1.0,
    default_src_components=None,
) -> list[ScheduleControl]:
    """Convert mixed MRST-like BC/SRC controls to native schedule controls."""
    out: list[ScheduleControl] = []
    for control in list(controls):
        if isinstance(control, ScheduleControl):
            out.append(copy.deepcopy(control))
            continue
        if not isinstance(control, Mapping):
            raise TypeError("forcing schedule controls must be mappings or ScheduleControl")
        bc = boundary_condition_from_control(
            grid,
            control,
            default_sat=default_bc_sat,
            default_components=default_bc_components,
        )
        src = source_from_control(
            control,
            default_sat=default_src_sat,
            default_components=default_src_components,
        )
        known = {
            "wells", "W", "Wells", "bc", "BC", "boundary", "Boundary",
            "boundary_condition", "boundaryCondition", "BoundaryCondition",
            "boundary_conditions", "boundaryConditions", "BoundaryConditions",
            "bc_records", "bcRecords", "BCRecords", "side", "face_side",
            "faceSide", "FaceSide", "side_name", "sideName", "SideName",
            "pressure_side", "pressureSide", "PressureSide",
            "pside", "pSide", "PSide", "psideh", "pSideH", "PSideH",
            "hydrostatic_side", "hydrostaticSide", "HydrostaticSide",
            "flux_side", "fluxSide", "FluxSide",
            "faces", "face", "face_index", "faceIndex", "face_indices",
            "faceIndices", "type", "bc_type", "kind", "pressure",
            "pres", "p", "bhp", "flux", "q", "massrate", "mass_rate",
            "sat", "saturation", "saturations", "s", "S", "components", "component",
            "z", "zi", "composition", "compositions", "compi", "comp_i",
            "composition_i", "component_i", "i1", "i2", "range", "Range",
            "search_range", "searchRange", "fluid", "Fluid", "ref_position",
            "refPosition", "ref_pressure", "refPressure", "ref_depth",
            "refDepth", "gravity", "Gravity", "pn", "src", "SRC", "sources",
            "source", "Source", "source_term", "sourceTerm", "SourceTerm",
            "source_terms", "sourceTerms", "SourceTerms", "src_terms", "srcTerms",
            "SrcTerms", "src_records", "srcRecords", "SrcRecords", "cells",
            "cell", "cell_index", "cellIndex", "cell_indices", "cellIndices",
            "rate", "value", "values", "val", "vals",
            "separator_group", "SeparatorGroup", "separatorGroup",
            "group_controls", "groupControls", "GroupControls",
            "vfp_tables_injector", "vfpTablesInjector", "VFPTablesInjector",
            "VFPINJ", "vfpinj", "vfp_tables_producer", "vfpTablesProducer",
            "VFPTablesProducer", "VFPPROD", "vfpprod",
        }
        out.append(
            ScheduleControl(
                wells=_control_get(control, "wells", "W", "Wells"),
                bc=bc,
                src=src,
                separator_group=_control_get(
                    control,
                    "separator_group",
                    "separatorGroup",
                    "SeparatorGroup",
                ),
                group_controls=_control_get(control, "group_controls", "groupControls", "GroupControls"),
                vfp_tables_injector=_vfp_tables_from_control(control, producer=False),
                vfp_tables_producer=_vfp_tables_from_control(control, producer=True),
                extra=_extra_without_known_keys(control, known),
            )
        )
    return out


def simple_schedule_from_forcing_controls(
    timesteps,
    grid,
    controls,
    *,
    step_control=None,
    default_bc_sat=1.0,
    default_bc_components=None,
    default_src_sat=1.0,
    default_src_components=None,
) -> SimpleSchedule:
    """Build a ``SimpleSchedule`` from mixed MRST-like BC/SRC controls."""
    return simple_schedule(
        timesteps,
        control=schedule_controls_from_forcing_controls(
            grid,
            controls,
            default_bc_sat=default_bc_sat,
            default_bc_components=default_bc_components,
            default_src_sat=default_src_sat,
            default_src_components=default_src_components,
        ),
        step_control=step_control,
    )


def pack_simulation_problem(
    state0: CompositionalState,
    model: OverallCompositionModel,
    schedule: SimpleSchedule,
    directory: str,
    *,
    name: str,
    nonlinear_solver: object | None = None,
    **kwargs,
) -> PackedSimulationProblem:
    """Pack a compositional example problem, MRST-style.

    Extra keyword arguments are accepted for source compatibility with MRST
    scripts; currently only the nonlinear-solver handle is retained.
    """
    return PackedSimulationProblem(
        state0=state0,
        model=model,
        schedule=schedule,
        directory=directory,
        name=name,
        nonlinear_solver=nonlinear_solver or kwargs.get("nonlinearsolver"),
    )


def simulate_packed_problem(problems, *, max_cuts: int = 6, verbose: bool = False):
    """Run one or more packed problems in-place and return the same objects."""
    scalar = isinstance(problems, PackedSimulationProblem)
    items = [problems] if scalar else list(problems)
    for problem in items:
        problem.result = run_schedule(
            problem.model,
            problem.state0,
            problem.schedule,
            max_cuts=max_cuts,
            verbose=verbose,
        )
    return items[0] if scalar else items


def _separator_allocation_summary(diagnostics):
    """Compact report payload for separator allocation diagnostics."""
    if diagnostics is None:
        return None
    producer_indices = list(diagnostics.get("producer_indices", []))
    producer_names = list(diagnostics.get("producer_names", []))
    shares = np.asarray(diagnostics.get("allocation_shares", []), dtype=float).copy()
    totals = np.asarray(diagnostics.get("positive_surface_rates_total", []), dtype=float).copy()
    allocated = np.asarray(diagnostics.get("allocated_surface_rates", []), dtype=float).copy()
    component_stream = diagnostics.get("total_component_mass_stream")
    component_stream = (
        None
        if component_stream is None
        else np.asarray(component_stream, dtype=float).copy()
    )
    return {
        "ProducerIndices": producer_indices,
        "ProducerNames": producer_names,
        "AllocationShares": shares,
        "PositiveSurfaceRatesTotal": totals,
        "AllocatedSurfaceRates": allocated,
        "TotalComponentMassStream": component_stream,
    }


def _make_schedule_report(
    *,
    step: int,
    control: int,
    control_changed: bool,
    timestep: float,
    iterations: int,
    cuts: int = 0,
    mini_steps=None,
    converged: bool = True,
    facility_changed: bool = False,
    well_controls_changed: bool = False,
    boundary_changed: bool = False,
    source_changed: bool = False,
    forcing_changed: bool = False,
    separator_changed: bool = False,
    group_controls_changed: bool = False,
    vfp_injector_tables_changed: bool = False,
    vfp_producer_tables_changed: bool = False,
    well_switches=None,
    group_switches=None,
    end_run_requested: bool = False,
    end_run_groups=None,
    separator_allocation_diagnostics=None,
):
    """Create the MRST-style report dict used by schedule-like helpers."""
    mini_steps = [float(timestep)] if mini_steps is None else [float(dt) for dt in mini_steps]
    well_switches = [] if well_switches is None else copy.deepcopy(well_switches)
    group_switches = [] if group_switches is None else copy.deepcopy(group_switches)
    end_run_groups = [] if end_run_groups is None else copy.deepcopy(end_run_groups)
    separator_allocation_diagnostics = copy.deepcopy(separator_allocation_diagnostics)
    separator_allocation_summary = _separator_allocation_summary(separator_allocation_diagnostics)
    vfp_tables_changed = bool(vfp_injector_tables_changed or vfp_producer_tables_changed)
    facility_changed = bool(
        facility_changed
        or well_switches
        or group_switches
        or well_controls_changed
        or separator_changed
        or group_controls_changed
        or vfp_tables_changed
    )
    return {
        "Step": int(step),
        "Control": int(control),
        "ControlChanged": bool(control_changed),
        "FacilityChanged": facility_changed,
        "WellControlsChanged": bool(well_controls_changed),
        "BoundaryChanged": bool(boundary_changed),
        "SourceChanged": bool(source_changed),
        "ForcingChanged": bool(forcing_changed),
        "SeparatorChanged": bool(separator_changed),
        "GroupControlsChanged": bool(group_controls_changed),
        "VFPInjectorTablesChanged": bool(vfp_injector_tables_changed),
        "VFPProducerTablesChanged": bool(vfp_producer_tables_changed),
        "VFPTablesChanged": vfp_tables_changed,
        "WellSwitches": well_switches,
        "GroupSwitches": group_switches,
        "EndRunRequested": bool(end_run_requested),
        "EndRunGroups": end_run_groups,
        "SeparatorAllocationDiagnostics": separator_allocation_diagnostics,
        "SeparatorAllocationSummary": separator_allocation_summary,
        "Timestep": float(timestep),
        "Iterations": int(iterations),
        "Cuts": int(cuts),
        "MiniSteps": mini_steps,
        "Converged": bool(converged),
    }


def get_multiple_packed_simulator_outputs(problems):
    """Return ``(problems, states, reports, names)`` like MRST's helper."""
    items = [problems] if isinstance(problems, PackedSimulationProblem) else list(problems)
    states = []
    reports = []
    names = []
    for problem in items:
        if problem.result is None:
            simulate_packed_problem(problem)
        assert problem.result is not None
        states.append(problem.result.states)
        if problem.result.reports is None:
            schedule_timesteps = np.asarray(problem.schedule.step.val, dtype=float).ravel()
            schedule_controls = np.asarray(problem.schedule.step.control, dtype=int).ravel()
            schedule_manages_wells = any(c.wells is not None for c in problem.schedule.control)
            schedule_manages_bc = any(c.bc is not None for c in problem.schedule.control)
            schedule_manages_src = any(c.src is not None for c in problem.schedule.control)
            schedule_manages_separator = any(c.separator_group is not None for c in problem.schedule.control)
            schedule_manages_groups = any(c.group_controls is not None for c in problem.schedule.control)
            schedule_manages_vfp_injector = any(c.vfp_tables_injector is not None for c in problem.schedule.control)
            schedule_manages_vfp_producer = any(c.vfp_tables_producer is not None for c in problem.schedule.control)
            fallback_reports = []
            for step_ix, it in enumerate(np.asarray(problem.result.iterations, dtype=int)):
                ctrl_ix = (
                    int(schedule_controls[step_ix])
                    if step_ix < schedule_controls.size
                    else 1
                )
                previous_ctrl_ix = (
                    int(schedule_controls[step_ix - 1])
                    if 0 <= step_ix - 1 < schedule_controls.size
                    else None
                )
                control_changed = bool(step_ix == 0 or (
                    previous_ctrl_ix is not None and ctrl_ix != previous_ctrl_ix
                ))
                fallback_reports.append(_make_schedule_report(
                    step=step_ix + 1,
                    control=ctrl_ix,
                    control_changed=control_changed,
                    well_controls_changed=bool(schedule_manages_wells and control_changed),
                    boundary_changed=bool(schedule_manages_bc and control_changed),
                    source_changed=bool(schedule_manages_src and control_changed),
                    forcing_changed=bool((schedule_manages_bc or schedule_manages_src) and control_changed),
                    separator_changed=bool(schedule_manages_separator and control_changed),
                    group_controls_changed=bool(schedule_manages_groups and control_changed),
                    vfp_injector_tables_changed=bool(schedule_manages_vfp_injector and control_changed),
                    vfp_producer_tables_changed=bool(schedule_manages_vfp_producer and control_changed),
                    timestep=(
                        float(schedule_timesteps[step_ix])
                        if step_ix < schedule_timesteps.size
                        else np.nan
                    ),
                    iterations=int(it),
                    mini_steps=(
                        [float(schedule_timesteps[step_ix])]
                        if step_ix < schedule_timesteps.size
                        else []
                    ),
                ))
            reports.append(fallback_reports)
        else:
            reports.append(copy.deepcopy(problem.result.reports))
        names.append(problem.name)
    return items, states, reports, names


def run_steps(model: OverallCompositionModel, state: CompositionalState, timesteps, bc=None, src=None) -> SimulationResult:
    """Run a sequence of implicit compositional timesteps."""
    states: list[CompositionalState] = []
    iterations = []
    reports = []
    current = state
    for step_ix, dt in enumerate(np.atleast_1d(np.asarray(timesteps, dtype=float))):
        if src is None:
            current, ok, it = model.step(current, float(dt), bc=bc)
        else:
            current, ok, it = model.step(current, float(dt), bc=bc, src=src)
        if not ok:
            raise RuntimeError("compositional example step did not converge")
        states.append(current)
        iterations.append(it)
        reports.append(
            _make_schedule_report(
                step=step_ix + 1,
                control=1,
                control_changed=step_ix == 0,
                boundary_changed=bool(bc is not None and step_ix == 0),
                source_changed=bool(src is not None and step_ix == 0),
                forcing_changed=bool((bc is not None or src is not None) and step_ix == 0),
                separator_allocation_diagnostics=getattr(current, "separator_group_diagnostics", None),
                timestep=float(dt),
                iterations=int(it),
                mini_steps=[float(dt)],
            )
        )
    return SimulationResult(states, np.asarray(iterations, dtype=int), reports)


def run_schedule(
    model: OverallCompositionModel,
    state: CompositionalState,
    schedule: SimpleSchedule,
    *,
    max_cuts: int = 6,
    verbose: bool = False,
    stop_on_end_run: bool = True,
    observer=None,
) -> SimulationResult:
    """Run an MRST-style compositional schedule with timestep cutting.

    Each report step uses the one-based `schedule.step.control` entry just as
    MRST does. If a nonlinear step fails, the report step is retried with
    halved substeps until it succeeds or `max_cuts` is exceeded. If an imported
    ``GECON`` economics row requests end-run on violation, the accepted report
    is emitted and the schedule stops by default.

    ``observer`` is an optional, dependency-free observation hook matching the
    one ``NonLinearSolver`` and the fixed-stress splitting loop carry, with the
    same event names. This loop drives ``model.step`` and owns its own cutting
    and retry, so the canonical solver hook never reaches it and a schedule run
    here produced no events at all. Per-iteration residuals stay out of reach -
    they live inside ``model.step`` - so this reports ministep and cut
    decisions, which is what the loop itself owns. It receives copied scalars
    only, its exceptions are swallowed, and with the default ``None`` this
    function behaves exactly as before.
    """
    def _observe(event, **payload):
        if observer is None:
            return
        try:
            observer(event, payload)
        except Exception:
            pass

    states: list[CompositionalState] = []
    iterations = []
    reports: list[dict] = []
    current = state
    previous_ctrl_ix: int | None = None
    original_wells = copy.deepcopy(list(model.wells))
    original_bc = model.bc
    original_separator_group = copy.deepcopy(getattr(model, "separator_group", None))
    original_group_controls = copy.deepcopy(list(getattr(model, "group_controls", [])))
    original_src = copy.deepcopy(getattr(model, "src", None))
    original_vfp_tables_injector = copy.deepcopy(getattr(model, "vfp_tables_injector", []))
    original_vfp_tables_producer = copy.deepcopy(getattr(model, "vfp_tables_producer", []))
    schedule_manages_wells = any(c.wells is not None for c in schedule.control)
    schedule_manages_bc = any(c.bc is not None for c in schedule.control)
    schedule_manages_src = any(c.src is not None for c in schedule.control)
    schedule_manages_separator = any(c.separator_group is not None for c in schedule.control)
    schedule_manages_groups = any(c.group_controls is not None for c in schedule.control)
    schedule_manages_vfp_injector = any(c.vfp_tables_injector is not None for c in schedule.control)
    schedule_manages_vfp_producer = any(c.vfp_tables_producer is not None for c in schedule.control)
    try:
        for step_ix, (dt_report, ctrl_ix) in enumerate(zip(schedule.step.val, schedule.step.control)):
            ctrl = schedule.control[int(ctrl_ix) - 1]
            needs_vfp_rebind = False
            if ctrl.wells is not None:
                model.wells = copy.deepcopy(list(ctrl.wells))
                needs_vfp_rebind = True
            if ctrl.vfp_tables_injector is not None and hasattr(model, "vfp_tables_injector"):
                model.vfp_tables_injector = copy.deepcopy(list(ctrl.vfp_tables_injector))
                needs_vfp_rebind = True
            if ctrl.vfp_tables_producer is not None and hasattr(model, "vfp_tables_producer"):
                model.vfp_tables_producer = copy.deepcopy(list(ctrl.vfp_tables_producer))
                needs_vfp_rebind = True
            if needs_vfp_rebind and hasattr(model, "_bind_vfp_tables"):
                model._bind_vfp_tables()
            if hasattr(model, "separator_group") and schedule_manages_separator:
                model.separator_group = copy.deepcopy(ctrl.separator_group)
            if hasattr(model, "group_controls") and schedule_manages_groups:
                model.group_controls = (
                    []
                    if ctrl.group_controls is None
                    else copy.deepcopy(list(ctrl.group_controls))
                )
            active_bc = copy.deepcopy(ctrl.bc) if ctrl.bc is not None else None
            active_src = copy.deepcopy(ctrl.src) if ctrl.src is not None else None
            model.bc = active_bc
            if hasattr(model, "src"):
                model.src = active_src
            t_local, it, cuts = 0.0, 0, 0
            mini_steps: list[float] = []
            dt = float(dt_report)
            _observe("report_step_started", report_step=int(step_ix),
                     report_dt=float(dt_report), control=int(ctrl_ix))
            while t_local < float(dt_report) - 1.0e-12 * max(float(dt_report), 1.0):
                dt = min(dt, float(dt_report) - t_local)
                _observe("ministep_started", report_step=int(step_ix),
                         ministep_index=len(mini_steps), trial_dt=float(dt))
                candidate, ok, nit = model.step(current, dt, bc=active_bc, src=active_src)
                it += nit
                _observe("step_accepted" if ok else "step_rejected",
                         report_step=int(step_ix), ministep_index=len(mini_steps),
                         trial_dt=float(dt), iterations=int(nit))
                if ok:
                    current = candidate
                    t_local += dt
                    mini_steps.append(float(dt))
                    dt = min(2.0 * dt, float(dt_report) - t_local) if cuts else dt
                else:
                    cuts += 1
                    if cuts > max_cuts:
                        raise RuntimeError(
                            f"compositional schedule step {step_ix} did not converge after {max_cuts} cuts"
                        )
                    dt *= 0.5
                    _observe("timestep_cut", report_step=int(step_ix),
                             cut=int(cuts), report_dt=float(dt_report),
                             trial_dt=float(dt))
                    if verbose:
                        print(f"  step {step_ix}: cutting dt to {dt:.3e}")
            states.append(current)
            iterations.append(it)
            well_switches = copy.deepcopy(getattr(model, "last_well_switches", []))
            group_switches = copy.deepcopy(getattr(model, "last_group_switches", []))
            end_run_requested = bool(getattr(model, "last_end_run_requested", False))
            end_run_groups = copy.deepcopy(getattr(model, "last_end_run_groups", []))
            control_changed = previous_ctrl_ix is None or int(ctrl_ix) != previous_ctrl_ix
            well_controls_changed = bool(schedule_manages_wells and control_changed)
            boundary_changed = bool(schedule_manages_bc and control_changed)
            source_changed = bool(schedule_manages_src and control_changed)
            forcing_changed = bool(boundary_changed or source_changed)
            separator_changed = bool(schedule_manages_separator and control_changed)
            group_controls_changed = bool(schedule_manages_groups and control_changed)
            vfp_injector_changed = bool(schedule_manages_vfp_injector and control_changed)
            vfp_producer_changed = bool(schedule_manages_vfp_producer and control_changed)
            reports.append(
                _make_schedule_report(
                    step=step_ix + 1,
                    control=ctrl_ix,
                    control_changed=control_changed,
                    facility_changed=bool(well_switches or group_switches),
                    well_controls_changed=well_controls_changed,
                    boundary_changed=boundary_changed,
                    source_changed=source_changed,
                    forcing_changed=forcing_changed,
                    separator_changed=separator_changed,
                    group_controls_changed=group_controls_changed,
                    vfp_injector_tables_changed=vfp_injector_changed,
                    vfp_producer_tables_changed=vfp_producer_changed,
                    well_switches=well_switches,
                    group_switches=group_switches,
                    end_run_requested=end_run_requested,
                    end_run_groups=end_run_groups,
                    separator_allocation_diagnostics=getattr(current, "separator_group_diagnostics", None),
                    timestep=dt_report,
                    iterations=it,
                    cuts=cuts,
                    mini_steps=mini_steps,
                )
            )
            previous_ctrl_ix = int(ctrl_ix)
            if stop_on_end_run and end_run_requested:
                break
    finally:
        model.wells = original_wells
        model.bc = original_bc
        if hasattr(model, "separator_group"):
            model.separator_group = original_separator_group
        if hasattr(model, "group_controls"):
            model.group_controls = original_group_controls
        if hasattr(model, "src"):
            model.src = original_src
        if hasattr(model, "vfp_tables_injector"):
            model.vfp_tables_injector = original_vfp_tables_injector
        if hasattr(model, "vfp_tables_producer"):
            model.vfp_tables_producer = original_vfp_tables_producer
        if hasattr(model, "_bind_vfp_tables"):
            model._bind_vfp_tables()
    return SimulationResult(states, np.asarray(iterations, dtype=int), reports)


def make_line_drive_case(
    nx: int = 8,
    mixture_name: str = "simple",
    rate: float = 5.0e-5,
    producer_bhp: float = 55.0 * BARSA,
    length: float = 100.0,
    permeability_md: float = 100.0,
):
    """Build a small injector-producer line-drive compositional case."""
    grid = compute_geometry(cart_grid([nx, 1, 1], [length, 10.0, 10.0]))
    rock = make_rock(grid, permeability_md * MILLI * DARCY, 0.25)
    mixture, info = get_compositional_fluid_case(mixture_name)
    wells = add_well(
        None,
        grid,
        rock,
        np.array([0]),
        well_type="rate",
        val=rate,
        compi=np.array([0.0, 1.0]),
        sign=1,
        name="Injector",
    )
    wells[0].components = info.injection
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([nx - 1]),
        well_type="bhp",
        val=producer_bhp,
        compi=np.array([0.0, 1.0]),
        sign=-1,
        name="Producer",
    )
    wells[1].components = info.initial
    model = OverallCompositionModel(grid, rock, mixture, temperature=info.T, wells=wells)
    state = model.initial_state(info.pressure, info.initial)
    return grid, rock, mixture, info, wells, model, state


if __name__ == "__main__":
    from pymrst.params.rock import pore_volume
    from pymrst.params.wells_and_bc import add_source
    from pymrst.utils.units import DAY

    def test_schedule_can_switch_sources():
        grid, _, _, info, _, model, _ = make_line_drive_case(nx=4)
        model.wells = []
        state = model.initial_state(info.pressure, info.initial)
        rate = pore_volume(grid, model.rock).sum() / (1000.0 * DAY)
        src_a = add_source(
            None,
            np.array([0]),
            rate,
            sat=np.array([0.0, 1.0]),
            components=np.array([0.9, 0.05, 0.05]),
        )
        src_a = add_source(src_a, np.array([grid.cells.num - 1]), -rate,
                           sat=np.array([0.5, 0.5]))
        src_b = add_source(
            None,
            np.array([0]),
            rate,
            sat=np.array([0.0, 1.0]),
            components=np.array([0.05, 0.9, 0.05]),
        )
        src_b = add_source(src_b, np.array([grid.cells.num - 1]), -rate,
                           sat=np.array([0.5, 0.5]))
        sched = simple_schedule(
            [2.0 * DAY, 2.0 * DAY],
            control=[ScheduleControl(src=src_a), ScheduleControl(src=src_b)],
            step_control=[1, 2],
        )
        out = run_schedule(model, state, sched)
        assert out.iterations.size == 2
        assert model.src is None
        assert sched.control[0].src is src_a and sched.control[1].src is src_b
        assert out.states[0].z[0, 0] > state.z[0, 0]
        assert out.states[1].z[0, 1] > out.states[0].z[0, 1]
        print("ok test_schedule_can_switch_sources")

    def test_schedule_cuts_failed_report_step():
        class DummyModel:
            def __init__(self):
                self.wells = []
                self.bc = None
                self.src = None
                self.separator_group = None
                self.calls = []

            def step(self, state, dt, bc=None, src=None):
                self.calls.append(dt)
                if dt > 0.5:
                    return state, False, 1
                return state + dt, True, 2

        model = DummyModel()
        sched = simple_schedule([1.0])
        out = run_schedule(model, 0.0, sched, max_cuts=2)
        assert np.allclose(model.calls, [1.0, 0.5, 0.5])
        assert np.isclose(out.final_state, 1.0)
        assert out.iterations[0] == 5
        assert out.reports is not None
        assert out.reports[0]["Cuts"] == 1
        assert out.reports[0]["MiniSteps"] == [0.5, 0.5]
        assert out.reports[0]["ControlChanged"] is True
        print("ok test_schedule_cuts_failed_report_step")

    def test_schedule_can_switch_group_controls():
        class DummyModel:
            def __init__(self):
                self.wells = []
                self.bc = None
                self.src = None
                self.separator_group = None
                self.group_controls = []
                self.seen = []

            def step(self, state, dt, bc=None, src=None):
                self.seen.append(list(self.group_controls))
                return state + 1, True, 1

        group_a = {"name": "group-a"}
        group_b = {"name": "group-b"}
        model = DummyModel()
        sched = simple_schedule(
            [1.0, 1.0],
            control=[
                ScheduleControl(group_controls=[group_a]),
                ScheduleControl(group_controls=[group_b]),
            ],
            step_control=[1, 2],
        )
        out = run_schedule(model, 0, sched)
        assert out.final_state == 2
        assert model.group_controls == []
        assert model.seen == [[group_a], [group_b]]
        print("ok test_schedule_can_switch_group_controls")

    def test_simple_schedule_accepts_facility_blocks():
        group = {"name": "G", "lims": {"orat": 1.0}}
        separator = {"pressure": 1.0}
        src = {"cells": [0]}
        sched = simple_schedule(
            [1.0, 2.0],
            src=src,
            separator_group=separator,
            group_controls=[group],
        )
        assert sched.step.control.tolist() == [1, 1]
        assert sched.control[0].src is src
        assert sched.control[0].separator_group is separator
        assert sched.control[0].group_controls == [group]
        print("ok test_simple_schedule_accepts_facility_blocks")

    def test_schedule_copies_facility_blocks():
        class DummyModel:
            def __init__(self):
                self.wells = []
                self.bc = None
                self.src = {"original": True}
                self.separator_group = {"original": True}
                self.group_controls = [{"original": True}]
                self.seen = []

            def step(self, state, dt, bc=None, src=None):
                self.seen.append(
                    (
                        copy.deepcopy(self.src),
                        copy.deepcopy(self.separator_group),
                        copy.deepcopy(self.group_controls),
                    )
                )
                self.src["mutated"] = True
                self.separator_group["mutated"] = True
                self.group_controls[0]["mutated"] = True
                return state + 1, True, 1

        src = {"name": "source-control"}
        separator = {"name": "separator-control"}
        group = {"name": "group-control", "lims": {"orat": 1.0}}
        model = DummyModel()
        sched = simple_schedule(
            [1.0],
            src=src,
            separator_group=separator,
            group_controls=[group],
        )
        out = run_schedule(model, 0, sched)
        assert out.final_state == 1
        assert model.src == {"original": True}
        assert model.separator_group == {"original": True}
        assert model.group_controls == [{"original": True}]
        assert sched.control[0].src == {"name": "source-control"}
        assert sched.control[0].separator_group == {"name": "separator-control"}
        assert sched.control[0].group_controls == [{"name": "group-control", "lims": {"orat": 1.0}}]
        assert model.seen == [(
            {"name": "source-control"},
            {"name": "separator-control"},
            [{"name": "group-control", "lims": {"orat": 1.0}}],
        )]
        print("ok test_schedule_copies_facility_blocks")

    def test_schedule_facility_management_scope():
        class DummyModel:
            def __init__(self):
                self.wells = []
                self.bc = None
                self.src = None
                self.separator_group = {"default": "separator"}
                self.group_controls = [{"default": "group"}]
                self.seen = []

            def step(self, state, dt, bc=None, src=None):
                self.seen.append(
                    (
                        copy.deepcopy(self.separator_group),
                        copy.deepcopy(self.group_controls),
                    )
                )
                return state + 1, True, 1

        model = DummyModel()
        unmanaged = simple_schedule([1.0])
        out = run_schedule(model, 0, unmanaged)
        assert out.final_state == 1
        assert model.seen == [({"default": "separator"}, [{"default": "group"}])]
        assert model.separator_group == {"default": "separator"}
        assert model.group_controls == [{"default": "group"}]

        model.seen = []
        managed = simple_schedule(
            [1.0, 1.0],
            control=[
                ScheduleControl(separator_group={"active": "separator"},
                                group_controls=[{"active": "group"}]),
                ScheduleControl(separator_group=None, group_controls=None),
            ],
            step_control=[1, 2],
        )
        out = run_schedule(model, 0, managed)
        assert out.final_state == 2
        assert model.seen == [
            ({"active": "separator"}, [{"active": "group"}]),
            (None, []),
        ]
        assert model.separator_group == {"default": "separator"}
        assert model.group_controls == [{"default": "group"}]
        print("ok test_schedule_facility_management_scope")

    def test_explicit_schedule_copies_well_controls():
        grid, _, _, info, wells, model, state = make_line_drive_case(nx=4)
        model = model.set_time_discretization("explicit")
        model.max_iterations = 60

        wells_a = copy.deepcopy(wells)
        wells_b = copy.deepcopy(wells)
        wells_a[0].components = np.array([0.9, 0.05, 0.05])
        wells_b[0].components = np.array([0.05, 0.9, 0.05])
        wells_a[0].lims = {"bhp": 1.0 * info.pressure}
        wells_b[0].lims = {"bhp": 1.0 * info.pressure}

        sched = simple_schedule(
            [0.25 * DAY, 0.25 * DAY],
            control=[
                ScheduleControl(wells=wells_a),
                ScheduleControl(wells=wells_b),
            ],
            step_control=[1, 2],
        )
        out = run_schedule(model, state, sched)
        assert out.iterations.size == 2
        assert model.time_discretization == "explicit"
        assert [w.name for w in model.wells] == [w.name for w in wells]
        assert sched.control[0].wells[0].type == "rate"
        assert sched.control[1].wells[0].type == "rate"
        assert np.all(np.isfinite(out.final_state.pressure))
        assert out.final_state.z[0, 1] > 0.0
        print("ok test_explicit_schedule_copies_well_controls")

    def test_packed_problem_helpers_mirror_mrst_workflow():
        grid, _, _, info, wells, model, state = make_line_drive_case(nx=3)
        assert grid.cells.num == 3
        schedule = simple_schedule([1.0 * DAY], wells=wells)
        problem = pack_simulation_problem(
            state,
            model,
            schedule,
            "ExampleSuite",
            name="TinyPackedProblem",
        )
        assert problem.result is None
        simulate_packed_problem(problem)
        problems, states, reports, names = get_multiple_packed_simulator_outputs([problem])
        assert problems[0] is problem
        assert names == ["TinyPackedProblem"]
        assert len(states[0]) == 1
        assert reports[0][0]["Iterations"] == int(problem.result.iterations[0])
        assert reports[0][0]["Control"] == 1
        assert "MiniSteps" in reports[0][0]
        assert np.all(np.isfinite(states[0][-1].pressure))
        print("ok test_packed_problem_helpers_mirror_mrst_workflow")

    test_schedule_can_switch_sources()
    test_schedule_cuts_failed_report_step()
    test_schedule_can_switch_group_controls()
    test_simple_schedule_accepts_facility_blocks()
    test_schedule_copies_facility_blocks()
    test_schedule_facility_management_scope()
    test_explicit_schedule_copies_well_controls()
    test_packed_problem_helpers_mirror_mrst_workflow()
    print("\nAll tests passed!")


__all__ = [
    "PackedSimulationProblem",
    "ScheduleControl",
    "ScheduleStep",
    "SimpleSchedule",
    "SimulationResult",
    "boundary_condition_from_control",
    "get_multiple_packed_simulator_outputs",
    "make_line_drive_case",
    "pack_simulation_problem",
    "run_schedule",
    "run_steps",
    "schedule_controls_from_boundary_controls",
    "schedule_controls_from_deck_controls",
    "schedule_controls_from_forcing_controls",
    "schedule_controls_from_source_controls",
    "simulate_packed_problem",
    "simple_schedule",
    "simple_schedule_from_boundary_controls",
    "simple_schedule_from_deck_controls",
    "simple_schedule_from_forcing_controls",
    "simple_schedule_from_mrst_schedule",
    "simple_schedule_from_source_controls",
    "source_from_control",
    "vfp_tables_from_deck_control",
]
