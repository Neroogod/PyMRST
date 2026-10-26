"""Validated, revisioned model construction over real PyMRST contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.construction import (
    FieldSource, FieldSourceKind, ModelConstructionDocument, ModelDraft, ModelRevision,
)
from .grid_service import GridInspectionService
from .project_service import ProjectService


@dataclass(frozen=True, slots=True)
class ConstructedModelInputs:
    grid: object
    rock: object
    fluid: object
    initial_state: object
    wells: tuple[object, ...]
    schedule: object
    revision: ModelRevision


class ModelConstructionService:
    def __init__(self, database: AppDatabase, projects: ProjectService, grids: GridInspectionService):
        self.database, self.projects, self.grids = database, projects, grids

    @staticmethod
    def _canonical(document: ModelConstructionDocument) -> tuple[str, str]:
        encoded = json.dumps(document.to_dict(), sort_keys=True, separators=(",", ":"))
        return encoded, hashlib.sha256(encoded.encode()).hexdigest()

    def _source(self, snapshot_id: str, source: FieldSource, cells: int) -> np.ndarray:
        if source.kind is FieldSourceKind.UNIFORM:
            return np.full(cells, float(source.value)*float(source.scale))
        value = np.asarray(self.grids.array(snapshot_id, source.array), dtype=float)
        if value.ndim == 0 or value.shape[0] != cells:
            raise ValueError(f"grid array {source.array} does not align with {cells} cells")
        if value.ndim > 1 and int(np.prod(value.shape[1:])) != 1:
            raise ValueError(f"grid array {source.array} is not a scalar cell property")
        return value.reshape(cells)*float(source.scale)

    def validate(self, snapshot_id: str, document: ModelConstructionDocument) -> tuple[dict, ...]:
        snapshot = self.grids.get(snapshot_id); cells = snapshot.cell_count
        issues: list[dict] = []
        def add(code, severity, section, message, **details):
            issues.append({"code": code, "severity": severity, "section": section,
                           "message": message, "details": details})
        def resolve(label, source):
            try: return self._source(snapshot_id, source, cells)
            except (KeyError, TypeError, ValueError) as error:
                add("field.invalid_source", "error", label, str(error)); return None
        poro = resolve("rock", document.rock.porosity)
        if poro is not None and np.any((poro <= 0) | (poro > 1) | ~np.isfinite(poro)):
            add("rock.porosity_range", "error", "rock", "porosity must be finite in (0,1]")
        for name in ("permeability_x", "permeability_y", "permeability_z"):
            source = getattr(document.rock, name)
            if source is not None:
                value = resolve("rock", source)
                if value is not None and np.any((value <= 0) | ~np.isfinite(value)):
                    add("rock.permeability_range", "error", "rock",
                        f"{name} must be finite and positive")
        dimensional = (
            document.rock.permeability_y is not None,
            document.rock.permeability_z is not None,
        )
        valid_tensor = {
            1: {(False, False)},
            2: {(False, False), (True, False), (True, True)},
            3: {(False, False), (True, True)},
        }[snapshot.dimension]
        if dimensional not in valid_tensor:
            add(
                "rock.permeability_basis", "error", "rock",
                f"permeability components are incompatible with a {snapshot.dimension}D grid",
                dimension=snapshot.dimension,
            )
        if document.rock.net_to_gross is not None:
            value = resolve("rock", document.rock.net_to_gross)
            if value is not None and np.any((value < 0) | (value > 1) | ~np.isfinite(value)):
                add("rock.ntg_range", "error", "rock", "net-to-gross must lie in [0,1]")
        fluid = document.fluid
        for label, values in (("viscosity", fluid.viscosity_pa_s), ("density", fluid.density_kg_m3),
                              ("relative-permeability exponent", fluid.relperm_exponent)):
            if any(not np.isfinite(item) or item <= 0 for item in values):
                add("fluid.invalid_property", "error", "fluid", f"{label} values must be finite and positive")
        if any(not np.isfinite(item) or item < 0 for item in fluid.compressibility_pa_inverse):
            add("fluid.invalid_compressibility", "error", "fluid", "compressibilities must be finite and nonnegative")
        if fluid.reference_pressure_pa <= 0 or not np.isfinite(fluid.reference_pressure_pa):
            add("fluid.invalid_reference_pressure", "error", "fluid", "reference pressure must be finite and positive")
        saturation = {}
        if set(document.initial_state.saturations) != set(fluid.phases):
            add("initialization.phase_basis", "error", "initialization",
                "initial saturation fields must exactly match the declared fluid phases",
                expected=list(fluid.phases), received=sorted(document.initial_state.saturations))
        for phase, source in document.initial_state.saturations.items():
            value = resolve("initialization", source); saturation[phase] = value
            if value is not None and np.any((value < 0) | (value > 1) | ~np.isfinite(value)):
                add("initialization.saturation_range", "error", "initialization",
                    f"{phase} saturation must be finite in [0,1]")
        if saturation and all(value is not None for value in saturation.values()):
            total = np.sum(tuple(saturation.values()), axis=0)
            if not np.allclose(total, 1.0, rtol=0, atol=1e-8):
                add("initialization.saturation_sum", "error", "initialization",
                    "phase saturations must sum to one in every active cell",
                    maximum_error=float(np.max(np.abs(total-1))))
        pressure = resolve("initialization", document.initial_state.pressure)
        if pressure is not None and np.any((pressure <= 0) | ~np.isfinite(pressure)):
            add("initialization.pressure_range", "error", "initialization",
                "initial pressure must be finite and positive")
        names = [well.name for well in document.wells]
        if len(names) != len(set(names)):
            add("wells.duplicate_name", "error", "wells", "well names must be unique")
        valid_control = {"bhp", "rate", "vrat", "resv_history", "orat", "wrat", "grat", "lrat"}
        for well in document.wells:
            if not well.cells or min(well.cells, default=0) < 0 or max(well.cells, default=-1) >= cells:
                add("wells.invalid_completion", "error", "wells",
                    f"well {well.name} has empty or out-of-range completions")
            if well.control_type not in valid_control or not np.isfinite(well.target_si):
                add("wells.invalid_control", "error", "wells", f"well {well.name} has an invalid control")
            if well.radius_m <= 0 or well.direction not in {"x", "y", "z"}:
                add("wells.invalid_geometry", "error", "wells", f"well {well.name} has invalid geometry")
            if well.sign not in {-1, 0, 1}:
                add("wells.invalid_sign", "error", "wells",
                    f"well {well.name} sign must be -1, 0 or 1")
            if any(not np.isfinite(float(item)) for item in well.limits.values()):
                add("wells.invalid_limit", "error", "wells",
                    f"well {well.name} has a non-finite operating limit")
            if len(well.composition) != len(fluid.phases) or any(item < 0 for item in well.composition) \
                    or not np.isclose(sum(well.composition), 1.0):
                add("wells.invalid_composition", "error", "wells",
                    f"well {well.name} composition must be nonnegative and sum to one")
        schedule = document.schedule
        if not schedule.timestep_seconds or len(schedule.timestep_seconds) != len(schedule.step_control):
            add("schedule.step_shape", "error", "schedule", "timesteps and control indices must be nonempty and aligned")
        if any(item <= 0 or not np.isfinite(item) for item in schedule.timestep_seconds):
            add("schedule.invalid_timestep", "error", "schedule", "timesteps must be finite and positive")
        if not schedule.controls or any(item < 0 or item >= len(schedule.controls) for item in schedule.step_control):
            add("schedule.invalid_control_index", "error", "schedule", "schedule references an undefined control period")
        known = set(names)
        period_names = [control.name for control in schedule.controls]
        if len(period_names) != len(set(period_names)):
            add("schedule.duplicate_control_name", "error", "schedule",
                "control-period names must be unique")
        for control in schedule.controls:
            unknown = sorted(set(control.well_overrides)-known)
            if unknown:
                add("schedule.unknown_well", "error", "schedule",
                    f"control {control.name} references unknown wells", wells=unknown)
            for well_name, override in control.well_overrides.items():
                unsupported = sorted(
                    set(override)-{"control_type", "target_si", "status", "limits"}
                )
                if unsupported:
                    add("schedule.unsupported_override", "error", "schedule",
                        f"control {control.name} has unsupported overrides for {well_name}",
                        fields=unsupported)
                if ("control_type" in override and
                        str(override["control_type"]).lower() not in valid_control):
                    add("schedule.invalid_well_control", "error", "schedule",
                        f"control {control.name} has an invalid control type for {well_name}")
                try:
                    finite_target = "target_si" not in override or np.isfinite(float(override["target_si"]))
                except (TypeError, ValueError):
                    finite_target = False
                if not finite_target:
                    add("schedule.invalid_well_target", "error", "schedule",
                        f"control {control.name} has a non-finite target for {well_name}")
        return tuple(issues)

    def create(self, project_id: str, grid_snapshot_id: str, name: str,
               document: ModelConstructionDocument, *, actor="user") -> ModelDraft:
        self.projects.get(project_id); snapshot = self.grids.get(grid_snapshot_id)
        if snapshot.project_id != project_id: raise ValueError("grid snapshot belongs to another project")
        encoded, digest = self._canonical(document); issues = self.validate(grid_snapshot_id, document)
        status = "invalid" if any(item["severity"] == "error" for item in issues) else "ready"
        now = utc_now(); draft_id = f"draft_{uuid4().hex}"
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO app_model_drafts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                               (draft_id, project_id, grid_snapshot_id, str(name).strip() or "Model draft",
                                1, status, digest, encoded, json.dumps(issues), now, now))
            connection.execute("INSERT INTO app_model_draft_events VALUES(?,?,?,?,?,?,?)",
                               (draft_id, 1, digest, encoded, actor, "draft created", now))
        return self.get(draft_id)

    def update(self, draft_id: str, document: ModelConstructionDocument, *, expected_version: int,
               actor="user", reason="model edited") -> ModelDraft:
        current = self.get(draft_id)
        if current.version != int(expected_version): raise RuntimeError("draft version conflict; reload before editing")
        encoded, digest = self._canonical(document); issues = self.validate(current.grid_snapshot_id, document)
        status = "invalid" if any(item["severity"] == "error" for item in issues) else "ready"
        version, now = current.version+1, utc_now()
        with self.database.transaction() as connection:
            changed = connection.execute(
                "UPDATE app_model_drafts SET version=?,status=?,document_hash=?,document_json=?,issues_json=?,updated_at=? WHERE draft_id=? AND version=?",
                (version, status, digest, encoded, json.dumps(issues), now, draft_id, current.version)).rowcount
            if changed != 1: raise RuntimeError("draft version conflict; reload before editing")
            connection.execute("INSERT INTO app_model_draft_events VALUES(?,?,?,?,?,?,?)",
                               (draft_id, version, digest, encoded, actor, str(reason), now))
        return self.get(draft_id)

    def get(self, draft_id: str) -> ModelDraft:
        row = self.database.fetch_one("SELECT * FROM app_model_drafts WHERE draft_id=?", (draft_id,))
        if row is None: raise KeyError(f"unknown model draft: {draft_id}")
        return ModelDraft(row["draft_id"], row["project_id"], row["grid_snapshot_id"], row["name"],
                          row["version"], row["status"], row["document_hash"],
                          ModelConstructionDocument.from_dict(json.loads(row["document_json"])),
                          tuple(json.loads(row["issues_json"])), row["created_at"], row["updated_at"])

    def list(self, project_id: str) -> tuple[ModelDraft, ...]:
        self.projects.get(project_id)
        return tuple(self.get(row[0]) for row in self.database.fetch_all(
            "SELECT draft_id FROM app_model_drafts WHERE project_id=? ORDER BY updated_at", (project_id,)))

    def history(self, draft_id: str) -> tuple[dict, ...]:
        self.get(draft_id)
        return tuple({
            "draft_id": row["draft_id"], "version": row["version"],
            "document_hash": row["document_hash"], "actor": row["actor"],
            "reason": row["reason"], "created_at": row["created_at"],
        } for row in self.database.fetch_all(
            "SELECT draft_id,version,document_hash,actor,reason,created_at "
            "FROM app_model_draft_events WHERE draft_id=? ORDER BY version",
            (draft_id,),
        ))

    def materialize(self, draft_id: str) -> ModelRevision:
        draft = self.get(draft_id)
        if draft.status != "ready": raise ValueError("invalid model draft cannot be materialized")
        existing = self.database.fetch_one(
            "SELECT revision_id FROM app_model_revisions WHERE draft_id=? AND content_hash=?",
            (draft_id, draft.document_hash))
        if existing: return self.revision(existing[0])
        number = int(self.database.fetch_one(
            "SELECT COALESCE(MAX(revision_number),0)+1 FROM app_model_revisions WHERE draft_id=?", (draft_id,))[0])
        revision_id = "revision_" + hashlib.sha256(f"{draft_id}:{draft.document_hash}".encode()).hexdigest()[:24]
        project = self.projects.get(draft.project_id)
        relative = Path("models") / draft_id / f"revision_{number:04d}" / "model.json"
        target = Path(project.root_path) / relative; target.parent.mkdir(parents=True, exist_ok=False)
        payload = {"schema_version": 1, "revision_id": revision_id, "draft_id": draft_id,
                   "revision_number": number, "content_hash": draft.document_hash,
                   "grid_snapshot_id": draft.grid_snapshot_id, "document": draft.document.to_dict()}
        handle, temporary_name = tempfile.mkstemp(prefix=".model-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w") as stream: json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
            temporary.replace(target); os.chmod(target, 0o444)
            summary = {"name": draft.name, "grid_snapshot_id": draft.grid_snapshot_id,
                       "cells": self.grids.get(draft.grid_snapshot_id).cell_count,
                       "phases": list(draft.document.fluid.phases), "wells": len(draft.document.wells),
                       "report_steps": len(draft.document.schedule.timestep_seconds)}
            created = utc_now()
            with self.database.transaction() as connection:
                connection.execute("INSERT INTO app_model_revisions VALUES(?,?,?,?,?,?,?,?)",
                                   (revision_id, draft_id, draft.project_id, number, draft.document_hash,
                                    relative.as_posix(), json.dumps(summary), created))
            return self.revision(revision_id)
        except Exception:
            target.unlink(missing_ok=True); target.parent.rmdir(); raise
        finally: temporary.unlink(missing_ok=True)

    def revision(self, revision_id: str) -> ModelRevision:
        row = self.database.fetch_one("SELECT * FROM app_model_revisions WHERE revision_id=?", (revision_id,))
        if row is None: raise KeyError(f"unknown model revision: {revision_id}")
        return ModelRevision(row["revision_id"], row["draft_id"], row["project_id"],
                             row["revision_number"], row["content_hash"], row["relative_path"],
                             json.loads(row["summary_json"]), row["created_at"])

    def revision_document(self, revision_id: str) -> tuple[ModelRevision, str, ModelConstructionDocument]:
        revision = self.revision(revision_id)
        project = self.projects.get(revision.project_id)
        path = Path(project.root_path) / revision.relative_path
        try:
            payload = json.loads(path.read_text())
            if (payload["revision_id"] != revision.revision_id or
                    payload["draft_id"] != revision.draft_id or
                    payload["content_hash"] != revision.content_hash):
                raise RuntimeError("model revision lineage does not match its catalog record")
            document = ModelConstructionDocument.from_dict(payload["document"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("model revision artifact is unreadable or malformed") from error
        _, digest = self._canonical(document)
        if digest != revision.content_hash:
            raise RuntimeError("model revision content checksum does not match its lineage")
        return revision, str(payload["grid_snapshot_id"]), document

    def revisions(self, draft_id: str) -> tuple[ModelRevision, ...]:
        self.get(draft_id)
        return tuple(self.revision(row[0]) for row in self.database.fetch_all(
            "SELECT revision_id FROM app_model_revisions WHERE draft_id=? "
            "ORDER BY revision_number", (draft_id,),
        ))

    def project_revisions(self, project_id: str) -> tuple[ModelRevision, ...]:
        """Every materialized revision across every draft in a project.

        Runnable models are selected by revision, not by draft, so the
        simulation launcher needs one flat, most-recent-first list rather
        than requiring a per-draft lookup for every draft in the project.
        """
        self.projects.get(project_id)
        return tuple(self.revision(row[0]) for row in self.database.fetch_all(
            "SELECT revision_id FROM app_model_revisions WHERE project_id=? "
            "ORDER BY created_at DESC", (project_id,),
        ))

    def compile(self, revision_id: str, grid) -> ConstructedModelInputs:
        from pymrst.params import make_rock, add_well
        from pymrst.modules.ad_props import init_simple_adi_fluid
        from pymrst.modules.ad_core.statefunctions.state_function import State
        from pymrst.modules.ad_core.utils.schedule import Control, Schedule
        revision, snapshot_id, doc = self.revision_document(revision_id)
        expected = self.grids.get(snapshot_id)
        if int(grid.cells.num) != expected.cell_count: raise ValueError("runtime grid does not match revision cell count")
        nc = expected.cell_count
        poro = self._source(snapshot_id, doc.rock.porosity, nc)
        components = [self._source(snapshot_id, doc.rock.permeability_x, nc)]
        if doc.rock.permeability_y is not None: components.append(self._source(snapshot_id, doc.rock.permeability_y, nc))
        if doc.rock.permeability_z is not None: components.append(self._source(snapshot_id, doc.rock.permeability_z, nc))
        perm = np.column_stack(components)
        ntg = None if doc.rock.net_to_gross is None else self._source(snapshot_id, doc.rock.net_to_gross, nc)
        rock = make_rock(grid, perm, poro, ntg)
        fluid = init_simple_adi_fluid(mu=doc.fluid.viscosity_pa_s, rho=doc.fluid.density_kg_m3,
                                     n=doc.fluid.relperm_exponent, c=doc.fluid.compressibility_pa_inverse,
                                     p_ref=doc.fluid.reference_pressure_pa, phases="".join(doc.fluid.phases))
        state_values = {"pressure": self._source(snapshot_id, doc.initial_state.pressure, nc)}
        saturation_columns = []
        for phase in doc.fluid.phases:
            values = self._source(
                snapshot_id, doc.initial_state.saturations[phase], nc,
            )
            state_values[f"s{phase}"] = values
            saturation_columns.append(values)
        state_values["s"] = np.column_stack(saturation_columns)
        if doc.initial_state.dissolved_gas_ratio is not None:
            state_values["rs"] = self._source(snapshot_id, doc.initial_state.dissolved_gas_ratio, nc)
        state = State(state_values); wells = None
        for definition in doc.wells:
            wells = add_well(wells, grid, rock, np.asarray(definition.cells), well_type=definition.control_type,
                             val=definition.target_si, radius=definition.radius_m, well_dir=definition.direction,
                             name=definition.name, compi=np.asarray(definition.composition), sign=definition.sign,
                             status=definition.status, lims=dict(definition.limits))
        base_wells = tuple(wells or ()); controls = []
        for period in doc.schedule.controls:
            owned = deepcopy(base_wells)
            for well in owned:
                override = period.well_overrides.get(well.name, {})
                if "control_type" in override: well.type = str(override["control_type"]).lower()
                if "target_si" in override: well.val = float(override["target_si"])
                if "status" in override: well.status = bool(override["status"])
                if "limits" in override: well.lims = dict(override["limits"])
            controls.append(Control(W=owned))
        schedule = Schedule(np.asarray(doc.schedule.timestep_seconds), np.asarray(doc.schedule.step_control), controls)
        return ConstructedModelInputs(grid, rock, fluid, state, base_wells, schedule, revision)


__all__ = ["ConstructedModelInputs", "ModelConstructionService"]
