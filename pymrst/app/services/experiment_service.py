"""Governed application facade over Research Studio experiment owners."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from pymrst.research_studio.experiments import (
    BranchDefinition, CaseReference, ExecutionProfile, ExperimentDefinition,
    OutputRequest, OwnerReference, ValidationRequirement, canonical_json,
    content_digest, materialize_variant_matrix, variant_matrix_from_dict,
)
from pymrst.research_studio.runtime.identifiers import CaseId

from ..database.models import AppDatabase
from ..schemas._common import json_value, nonempty, utc_now
from ..schemas.experiment import (
    BranchRecord, BranchRequest, ExecutionTarget, ExecutionTargetKind,
    ExperimentExecution, ExperimentRecord, ExperimentRequest,
    ExperimentSourceKind, QualificationRecord,
)
from .project_service import ProjectService, _file_hash
from .run_service import RunService


_MEMBER_RUNNERS = {"manufactured-qualification", "spe1-time-integration"}
_BRANCH_RUNNERS = {"manufactured-branch", "spe1-branch"}
_SAFE_SLURM = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_TIME = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")


class ExperimentService:
    def __init__(self, database: AppDatabase, projects: ProjectService,
                 runs: RunService, *, maximum_members: int = 100_000):
        self.database, self.projects, self.runs = database, projects, runs
        self.maximum_members = int(maximum_members)

    def _source(self, project_id: str, kind: ExperimentSourceKind, source_id: str):
        if kind is ExperimentSourceKind.MODEL_REVISION:
            row = self.database.fetch_one(
                "SELECT revision_id,content_hash FROM app_model_revisions "
                "WHERE revision_id=? AND project_id=?", (source_id, project_id),
            )
        else:
            row = self.database.fetch_one(
                "SELECT configuration_id,configuration_hash AS content_hash "
                "FROM app_run_configurations WHERE configuration_id=? AND project_id=?",
                (source_id, project_id),
            )
        if row is None:
            raise KeyError(f"unknown project {kind.value}: {source_id}")
        return str(row[0]), str(row[1])

    def _matrix(self, value):
        matrix = variant_matrix_from_dict(dict(value))
        kind = matrix.matrix_type
        if kind == "cartesian":
            count = 1
            for axis in (*matrix.parameter_axes, *matrix.component_axes):
                count *= len(axis.values if hasattr(axis, "values") else axis.owners)
        elif kind == "sampled":
            count = matrix.sample_count
        else:
            count = len(matrix.variants)
        if not 1 <= count <= self.maximum_members:
            raise ValueError(
                f"experiment member count {count} exceeds configured limit {self.maximum_members}"
            )
        return matrix, count

    def create(self, project_id: str, request: ExperimentRequest) -> ExperimentRecord:
        self.projects.get(project_id)
        if request.member_runner not in _MEMBER_RUNNERS:
            raise ValueError(f"unregistered experiment runner: {request.member_runner}")
        source_id, source_hash = self._source(
            project_id, request.source_kind, request.source_id,
        )
        matrix, expected_count = self._matrix(request.matrix)
        owner = OwnerReference(
            f"pymrst.app.{request.source_kind.value}",
            {"source_id": source_id}, fingerprint=source_hash,
        )
        definition = ExperimentDefinition(
            name=request.name, hypothesis=request.hypothesis,
            description=request.description,
            base_case=CaseReference(
                CaseId(f"case_{source_hash[:24]}"), owner, source_hash,
            ),
            model=owner, schedule=owner, variants=(), datasets=(),
            execution_profile=ExecutionProfile(
                "pymrst-studio", {"member_runner": request.member_runner},
            ),
            validation=(ValidationRequirement(
                "finite-member-results", {"required": True},
            ),),
            outputs=(OutputRequest(
                "requested-results", request.outputs, "report_step",
            ),),
            random_seed=request.random_seed, tags=request.tags,
        )
        materialized = materialize_variant_matrix(definition, matrix)
        if len(materialized.coordinates) != expected_count:
            raise RuntimeError("variant matrix expansion count is inconsistent")
        identity = content_digest({
            "definition": definition.to_dict(), "matrix": matrix.to_dict(),
            "source_kind": request.source_kind.value, "source_id": source_id,
        })
        experiment_id = f"experiment_{identity}"
        project = self.projects.get(project_id)
        relative = Path("experiments") / experiment_id / "experiment.json"
        path = Path(project.root_path) / relative
        payload = {
            "schema_version": 1, "experiment_id": experiment_id,
            "project_id": project_id, "source_kind": request.source_kind.value,
            "source_id": source_id, "content_hash": identity,
            "definition": definition.to_dict(), "matrix": matrix.to_dict(),
            "materialized": materialized.to_dict(),
        }
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=False)
            temporary = path.with_name(".experiment.json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n",
                encoding="utf-8",
            )
            temporary.replace(path); os.chmod(path, 0o444)
        elif json.loads(path.read_text()) != payload:
            raise RuntimeError("experiment identity collision")
        created = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_experiments VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (experiment_id, project_id, request.source_kind.value, source_id,
                 request.name, identity, canonical_json(definition.to_dict()),
                 canonical_json(matrix.to_dict()), canonical_json(materialized.to_dict()),
                 len(materialized.coordinates), created),
            )
        return self.get(experiment_id)

    @staticmethod
    def _record(row) -> ExperimentRecord:
        definition = json.loads(row["definition_json"])
        matrix = json.loads(row["matrix_json"])
        materialized = json.loads(row["materialized_json"])
        return ExperimentRecord(
            row["experiment_id"], row["project_id"], row["source_kind"],
            row["source_id"], row["name"], row["content_hash"],
            row["member_count"], definition, matrix,
            tuple(materialized["coordinates"]), row["created_at"],
        )

    def get(self, experiment_id: str) -> ExperimentRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_experiments WHERE experiment_id=?", (experiment_id,),
        )
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return self._record(row)

    def list(self, project_id: str) -> tuple[ExperimentRecord, ...]:
        self.projects.get(project_id)
        return tuple(self._record(row) for row in self.database.fetch_all(
            "SELECT * FROM app_experiments WHERE project_id=? "
            "ORDER BY created_at,experiment_id", (project_id,),
        ))

    def _artifact_path(self, experiment: ExperimentRecord) -> Path:
        project = self.projects.get(experiment.project_id)
        return Path(project.root_path) / "experiments" / experiment.experiment_id / "experiment.json"

    def _artifact(self, experiment: ExperimentRecord) -> dict:
        from pymrst.research_studio.experiments import MaterializedVariantMatrix
        path = self._artifact_path(experiment)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            digest = content_digest({
                "definition": payload["definition"], "matrix": payload["matrix"],
                "source_kind": payload["source_kind"], "source_id": payload["source_id"],
            })
            MaterializedVariantMatrix.from_dict(payload["materialized"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("experiment artifact is unreadable or malformed") from error
        if (payload["experiment_id"] != experiment.experiment_id or
                payload["content_hash"] != experiment.content_hash or
                digest != experiment.content_hash):
            raise RuntimeError("experiment artifact checksum or lineage mismatch")
        return payload

    def prepare_execution(self, experiment_id: str,
                          target: ExecutionTarget) -> ExperimentExecution:
        experiment = self.get(experiment_id)
        execution_id = f"execution_{uuid4().hex}"
        project = self.projects.get(experiment.project_id)
        relative_root = Path("experiments") / experiment_id / "executions" / execution_id
        root = Path(project.root_path) / relative_root
        runner = str(experiment.definition["execution_profile"]["configuration"]["member_runner"])
        remote_script = None if target.kind is ExecutionTargetKind.LOCAL else self._remote_script(
            target, "materialized.json", runner, output_name="ensemble-results",
        )
        root.mkdir(parents=True, exist_ok=False)
        materialized = self._artifact(experiment)["materialized"]
        materialized_path = root / "materialized.json"
        materialized_path.write_text(
            json.dumps(materialized, indent=2, sort_keys=True, allow_nan=False)+"\n"
        )
        os.chmod(materialized_path, 0o444)
        configuration_id = ""
        status = "queued" if target.kind is ExecutionTargetKind.LOCAL else "staged"
        summary = {"member_count": experiment.member_count}
        if target.kind is ExecutionTargetKind.LOCAL:
            configuration = self.runs.create_configuration(
                experiment.project_id, f"{experiment.name} ensemble",
                "pymrst-ensemble", {
                    "execution_id": execution_id,
                    "materialized_path": str(materialized_path.resolve()),
                    "materialized_sha256": _file_hash(materialized_path),
                    "member_runner": runner, "max_workers": target.max_workers,
                    "memory_per_worker_bytes": target.memory_per_worker_bytes,
                },
            )
            configuration_id = configuration.configuration_id
        else:
            script_path = root / ("submit.slurm" if target.kind is ExecutionTargetKind.SLURM else "run-remote.sh")
            script_path.write_text(remote_script, encoding="utf-8")
            script_path.chmod(0o555)
            summary["submission_artifact"] = script_path.name
            summary["submission_mode"] = "explicit-user-submission"
        manifest = {
            "schema_version": 1, "execution_id": execution_id,
            "experiment_id": experiment_id, "target": target.to_dict(),
            "configuration_id": configuration_id,
            "materialized_sha256": _file_hash(materialized_path),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        os.chmod(manifest_path, 0o444)
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_experiment_executions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (execution_id, experiment_id, experiment.project_id,
                 target.kind.value, configuration_id or None, None, status,
                 json.dumps(target.to_dict()), relative_root.as_posix(),
                 json.dumps(summary), now, now),
            )
        return self.execution(execution_id, synchronize=False)

    @staticmethod
    def _remote_script(target: ExecutionTarget, materialized: str, runner: str,
                       *, output_name: str) -> str:
        common = (
            f"python3 -m pymrst.app.workers.ensemble_cli --materialized {materialized} "
            f"--output {output_name} --runner {runner} --workers {target.max_workers} "
            f"--memory-per-worker {target.memory_per_worker_bytes}\n"
        )
        if target.kind is ExecutionTargetKind.SSH:
            host = str(target.configuration.get("host", "")).strip()
            remote_root = str(target.configuration.get("remote_root", "")).strip()
            if not host or not remote_root or any(char.isspace() for char in host):
                raise ValueError("SSH targets require a whitespace-free host and remote_root")
            return "#!/bin/sh\nset -eu\n" + common
        partition = str(target.configuration.get("partition", "compute"))
        account = str(target.configuration.get("account", ""))
        walltime = str(target.configuration.get("walltime", "01:00:00"))
        nodes = int(target.configuration.get("nodes", 1))
        if (not _SAFE_SLURM.fullmatch(partition) or
                (account and not _SAFE_SLURM.fullmatch(account)) or
                not _SAFE_TIME.fullmatch(walltime) or not 1 <= nodes <= 1024):
            raise ValueError("invalid Slurm partition/account/walltime/nodes")
        directives = ["#!/bin/sh", f"#SBATCH --partition={partition}",
                      f"#SBATCH --time={walltime}", f"#SBATCH --nodes={nodes}",
                      f"#SBATCH --cpus-per-task={target.max_workers}"]
        if account:
            directives.append(f"#SBATCH --account={account}")
        return "\n".join(directives)+"\nset -eu\n"+common

    def bind_run(self, execution_id: str, run_id: str) -> ExperimentExecution:
        execution = self.execution(execution_id, synchronize=False)
        run = self.runs.get(run_id)
        if run.project_id != execution.project_id or run.configuration_id != execution.configuration_id:
            raise ValueError("run does not belong to the experiment execution")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_experiment_executions SET run_id=?,status=?,updated_at=? "
                "WHERE execution_id=? AND run_id IS NULL",
                (run_id, run.status.value, utc_now(), execution_id),
            )
        return self.execution(execution_id)

    @staticmethod
    def _execution_record(row) -> ExperimentExecution:
        return ExperimentExecution(
            row["execution_id"], row["experiment_id"], row["project_id"],
            row["target_kind"], row["configuration_id"] or "", row["run_id"] or "",
            row["status"], json.loads(row["target_json"]), row["relative_path"],
            json.loads(row["summary_json"]), row["created_at"], row["updated_at"],
        )

    def execution(self, execution_id: str, *, synchronize=True) -> ExperimentExecution:
        row = self.database.fetch_one(
            "SELECT * FROM app_experiment_executions WHERE execution_id=?",
            (execution_id,),
        )
        if row is None:
            raise KeyError(f"unknown experiment execution: {execution_id}")
        record = self._execution_record(row)
        if synchronize and record.run_id:
            run = self.runs.get(record.run_id)
            summary = dict(record.summary)
            task = self.runs.tasks.get(run.task_id)
            if task.result is not None:
                summary.update(task.result)
            if run.status.value != record.status or summary != record.summary:
                with self.database.transaction() as connection:
                    connection.execute(
                        "UPDATE app_experiment_executions SET status=?,summary_json=?,updated_at=? "
                        "WHERE execution_id=?",
                        (run.status.value, json.dumps(json_value(summary)), utc_now(), execution_id),
                    )
                return self.execution(execution_id, synchronize=False)
        return record

    def executions(self, experiment_id: str) -> tuple[ExperimentExecution, ...]:
        self.get(experiment_id)
        ids = self.database.fetch_all(
            "SELECT execution_id FROM app_experiment_executions WHERE experiment_id=? "
            "ORDER BY created_at,execution_id", (experiment_id,),
        )
        return tuple(self.execution(row[0]) for row in ids)

    def record_qualification(self, execution_id: str, *, reference_kind: str,
                             reference_id: str, tolerances, metrics, evidence=None):
        execution = self.execution(execution_id)
        reference_kind = str(reference_kind).lower()
        if reference_kind not in {"mrst", "opm", "analytic", "laboratory"}:
            raise ValueError("unsupported qualification reference kind")
        reference_id = nonempty(reference_id, "qualification reference id")
        tolerances, metrics = json_value(tolerances), json_value(metrics)
        if not metrics or set(metrics)-set(tolerances):
            raise ValueError("every qualification metric requires a tolerance")
        passed = True
        for name, value in metrics.items():
            policy = tolerances[name]
            absolute = float(value["maximum_absolute_error"])
            relative = float(value["maximum_relative_error"])
            absolute_limit = float(policy["absolute"])
            relative_limit = float(policy["relative"])
            if min(absolute_limit, relative_limit, absolute, relative) < 0:
                raise ValueError("qualification errors and tolerances must be nonnegative")
            passed &= absolute <= absolute_limit or relative <= relative_limit
        payload = {
            "execution_id": execution_id, "reference_kind": reference_kind,
            "reference_id": reference_id, "tolerances": tolerances,
            "metrics": metrics,
        }
        qualification_id = f"qualification_{content_digest(payload)[:32]}"
        created = utc_now(); status = "passed" if passed else "failed"
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_experiment_qualifications VALUES(?,?,?,?,?,?,?,?,?,?)",
                (qualification_id, execution.experiment_id, execution_id,
                 reference_kind, reference_id, status, json.dumps(tolerances),
                 json.dumps(metrics), json.dumps(json_value(evidence or {})), created),
            )
        return self.qualification(qualification_id)

    def qualification(self, qualification_id: str) -> QualificationRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_experiment_qualifications WHERE qualification_id=?",
            (qualification_id,),
        )
        if row is None:
            raise KeyError(f"unknown experiment qualification: {qualification_id}")
        return QualificationRecord(
            row["qualification_id"], row["experiment_id"], row["execution_id"],
            row["reference_kind"], row["reference_id"], row["status"],
            json.loads(row["tolerances_json"]), json.loads(row["metrics_json"]),
            json.loads(row["evidence_json"]), row["created_at"],
        )

    def qualifications(self, experiment_id: str) -> tuple[QualificationRecord, ...]:
        self.get(experiment_id)
        return tuple(self.qualification(row[0]) for row in self.database.fetch_all(
            "SELECT qualification_id FROM app_experiment_qualifications "
            "WHERE experiment_id=? ORDER BY created_at", (experiment_id,),
        ))

    def prepare_branch(self, execution_id: str, request: BranchRequest) -> BranchRecord:
        execution = self.execution(execution_id)
        if execution.status != "completed":
            raise ValueError("branches require a completed parent execution")
        if request.runner_kind not in _BRANCH_RUNNERS:
            raise ValueError(f"unregistered branch runner: {request.runner_kind}")
        outcomes = dict(execution.summary.get("report", {})).get("outcomes", [])
        parent = next((item for item in outcomes
                       if item["coordinate"]["member_id"] == request.parent_member_id
                       and item["status"] == "completed"), None)
        if parent is None:
            raise KeyError(f"unknown completed parent member: {request.parent_member_id}")
        parent_store = Path(parent["store_path"]).resolve()
        run = self.runs.get(execution.run_id)
        project = self.projects.get(execution.project_id)
        run_root = (Path(project.root_path) / run.run_relative_path).resolve()
        if run_root not in parent_store.parents or not (parent_store / "member.json").is_file():
            raise RuntimeError("parent member store is outside its recorded execution")
        definition = BranchDefinition(
            request.name, request.parent_member_id, request.parent_time_index,
            request.parameters, request.components, request.controls,
            request.properties, request.required_fields, "branch",
        )
        configuration = self.runs.create_configuration(
            execution.project_id, f"{request.name} branch", "pymrst-branch", {
                "definition": definition.to_dict(),
                "parent_store": str(parent_store),
                "parent_record_sha256": _file_hash(parent_store / "member.json"),
                "runner_kind": request.runner_kind,
                "seed": int(self.get(execution.experiment_id).definition["random_seed"]),
            },
        )
        now = utc_now()
        relative = Path("experiments") / execution.experiment_id / "branches" / definition.branch_id
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_experiment_branches VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (definition.branch_id, execution.experiment_id, execution_id,
                 execution.project_id, request.parent_member_id,
                 configuration.configuration_id, None, "queued",
                 json.dumps(definition.to_dict()), relative.as_posix(),
                 json.dumps({}), now, now),
            )
        return self.branch(definition.branch_id, synchronize=False)

    def bind_branch_run(self, branch_id: str, run_id: str) -> BranchRecord:
        branch = self.branch(branch_id, synchronize=False)
        run = self.runs.get(run_id)
        if (run.project_id != branch.project_id or
                run.configuration_id != branch.configuration_id):
            raise ValueError("run does not belong to the branch")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_experiment_branches SET run_id=?,status=?,updated_at=? "
                "WHERE branch_id=? AND run_id IS NULL",
                (run_id, run.status.value, utc_now(), branch_id),
            )
        return self.branch(branch_id)

    @staticmethod
    def _branch_record(row) -> BranchRecord:
        return BranchRecord(
            row["branch_id"], row["experiment_id"], row["execution_id"],
            row["project_id"], row["parent_member_id"], row["configuration_id"],
            row["run_id"] or "", row["status"], json.loads(row["definition_json"]),
            row["relative_path"], json.loads(row["summary_json"]),
            row["created_at"], row["updated_at"],
        )

    def branch(self, branch_id: str, *, synchronize=True) -> BranchRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_experiment_branches WHERE branch_id=?", (branch_id,),
        )
        if row is None:
            raise KeyError(f"unknown experiment branch: {branch_id}")
        record = self._branch_record(row)
        if synchronize and record.run_id:
            run = self.runs.get(record.run_id); task = self.runs.tasks.get(run.task_id)
            summary = {} if task.result is None else dict(task.result)
            if run.status.value != record.status or summary != record.summary:
                with self.database.transaction() as connection:
                    connection.execute(
                        "UPDATE app_experiment_branches SET status=?,summary_json=?,updated_at=? "
                        "WHERE branch_id=?",
                        (run.status.value, json.dumps(json_value(summary)), utc_now(), branch_id),
                    )
                return self.branch(branch_id, synchronize=False)
        return record

    def branches(self, experiment_id: str) -> tuple[BranchRecord, ...]:
        self.get(experiment_id)
        return tuple(self.branch(row[0]) for row in self.database.fetch_all(
            "SELECT branch_id FROM app_experiment_branches WHERE experiment_id=? "
            "ORDER BY created_at", (experiment_id,),
        ))


__all__ = ["ExperimentService"]
