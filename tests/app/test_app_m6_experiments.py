"""Acceptance coverage for App M6 experiments, HPC staging, and qualification."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import (
    BranchRequest, ExecutionTarget, ExperimentRequest, ProjectCreate,
)


def _matrix(*, failing=False):
    variants = [
        {"name": "low", "parameters": {"test.multiplier": 0.5},
         "components": {}, "comparison_group": "sensitivity"},
        {"name": "high", "parameters": {"test.multiplier": 2.0},
         "components": {}, "comparison_group": "sensitivity"},
    ]
    if failing:
        variants.insert(1, {
            "name": "broken", "parameters": {"test.fail": True},
            "components": {}, "comparison_group": "sensitivity",
        })
    return {"schema_version": 1, "matrix_type": "explicit", "variants": variants}


def _case(tmp_path, *, matrix=None):
    app = create_application(tmp_path / "workspace")
    project = app.projects.create(ProjectCreate("Ensemble study"))
    configuration = app.runs.create_configuration(
        project.project_id, "Base model", "protocol-qualification", {"steps": 1},
    )
    request = ExperimentRequest(
        "Permeability sensitivity", "Permeability controls pressure response",
        "A deterministic, bounded qualification ensemble.",
        "run_configuration", configuration.configuration_id, matrix or _matrix(),
        outputs=("pressure", "producer_rate"), random_seed=731,
        tags=("qualification", "sensitivity"),
    )
    experiment = app.experiments.create(project.project_id, request)
    return app, project, experiment


def test_experiment_definition_is_immutable_deterministic_and_bounded(tmp_path):
    app, project, experiment = _case(tmp_path)
    with app:
        duplicate = app.experiments.create(
            project.project_id,
            ExperimentRequest(
                "Permeability sensitivity", "Permeability controls pressure response",
                "A deterministic, bounded qualification ensemble.",
                "run_configuration", experiment.source_id, _matrix(),
                outputs=("pressure", "producer_rate"), random_seed=731,
                tags=("qualification", "sensitivity"),
            ),
        )
        assert duplicate == experiment
        assert experiment.member_count == 2
        assert len({item["member_id"] for item in experiment.coordinates}) == 2
        path = (Path(project.root_path) / "experiments" /
                experiment.experiment_id / "experiment.json")
        assert path.is_file() and not bool(path.stat().st_mode & 0o200)
        with pytest.raises(ValueError, match="exceeds configured limit"):
            app.experiments.maximum_members = 1
            app.experiments.create(
                project.project_id,
                ExperimentRequest(
                    "Too large", "Bound check", "Must be rejected before expansion",
                    "run_configuration", experiment.source_id, _matrix(),
                ),
            )


def test_local_ensemble_runs_in_isolated_worker_and_publishes_uncertainty(tmp_path):
    app, _, experiment = _case(tmp_path, matrix=_matrix(failing=True))
    with app:
        execution = app.experiments.prepare_execution(
            experiment.experiment_id,
            ExecutionTarget("local", "Test process pool", 2, 64*1024*1024),
        )
        run = app.isolated.submit(execution.configuration_id)
        execution = app.experiments.bind_run(execution.execution_id, run.run_id)
        deadline = time.monotonic()+20
        while execution.status not in {"completed", "failed", "cancelled"}:
            assert time.monotonic() < deadline
            time.sleep(0.05)
            execution = app.experiments.execution(execution.execution_id)
        assert execution.status == "completed"
        assert execution.summary["report"]["summary"] == {
            "completed": 2, "failed": 1, "resumed": 0,
        }
        pressure = execution.summary["uncertainty"]["fields.pressure.final_mean"]
        assert pressure["count"] == 2
        assert pressure["minimum"] < pressure["p50"] < pressure["maximum"]
        events = app.runs.events(execution.run_id)
        assert any(item.event_type == "progress" for item in events)

        parent = next(item for item in execution.summary["report"]["outcomes"]
                      if item["status"] == "completed")
        branch = app.experiments.prepare_branch(
            execution.execution_id,
            BranchRequest(
                "Higher-rate continuation", parent["coordinate"]["member_id"],
                controls={"rate": 2.0}, properties={"mobility": 1.5},
                required_fields=("pressure",),
            ),
        )
        child = app.isolated.submit(branch.configuration_id)
        branch = app.experiments.bind_branch_run(branch.branch_id, child.run_id)
        deadline = time.monotonic()+20
        while branch.status not in {"completed", "failed", "cancelled"}:
            assert time.monotonic() < deadline
            time.sleep(0.05)
            branch = app.experiments.branch(branch.branch_id)
        assert branch.status == "completed"
        assert branch.summary["outcome"]["parent_member_id"] == parent["coordinate"]["member_id"]
        assert branch.summary["outcome"]["parent_time"] == 4.0
        assert app.experiments.branches(experiment.experiment_id) == (branch,)


@pytest.mark.parametrize("target", [
    ExecutionTarget("slurm", "Research cluster", 8, 256*1024*1024,
                    {"partition": "compute", "account": "reservoir", "walltime": "02:00:00", "nodes": 2}),
    ExecutionTarget("ssh", "Remote workstation", 4, 128*1024*1024,
                    {"host": "compute.example", "remote_root": "/scratch/pymrst"}),
])
def test_remote_targets_create_reviewable_credential_free_bundles(tmp_path, target):
    app, project, experiment = _case(tmp_path)
    with app:
        execution = app.experiments.prepare_execution(experiment.experiment_id, target)
        assert execution.status == "staged" and not execution.run_id
        root = Path(project.root_path) / execution.relative_path
        script = root / execution.summary["submission_artifact"]
        assert script.is_file()
        assert "pymrst.app.workers.ensemble_cli" in script.read_text()
        assert "password" not in json.dumps(execution.target).lower()


def test_mrst_and_opm_qualification_records_retain_evidence(tmp_path):
    app, _, experiment = _case(tmp_path)
    with app:
        execution = app.experiments.prepare_execution(
            experiment.experiment_id,
            ExecutionTarget("slurm", "Staged", 2, 64*1024*1024,
                            {"partition": "compute", "walltime": "00:30:00"}),
        )
        tolerances = {"pressure": {"absolute": 100.0, "relative": 1e-4}}
        passed = app.experiments.record_qualification(
            execution.execution_id, reference_kind="mrst", reference_id="mrst-2024a",
            tolerances=tolerances,
            metrics={"pressure": {"maximum_absolute_error": 30.0,
                                   "maximum_relative_error": 2e-6}},
            evidence={"fixture_sha256": "a"*64},
        )
        failed = app.experiments.record_qualification(
            execution.execution_id, reference_kind="opm", reference_id="flow-2026.04",
            tolerances=tolerances,
            metrics={"pressure": {"maximum_absolute_error": 200.0,
                                   "maximum_relative_error": 2e-4}},
        )
        assert passed.status == "passed" and failed.status == "failed"
        assert [item.reference_kind for item in app.experiments.qualifications(
            experiment.experiment_id)] == ["mrst", "opm"]


def test_experiment_artifact_tampering_blocks_execution(tmp_path):
    app, project, experiment = _case(tmp_path)
    with app:
        path = (Path(project.root_path) / "experiments" /
                experiment.experiment_id / "experiment.json")
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        payload["definition"]["random_seed"] += 1
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="checksum"):
            app.experiments.prepare_execution(
                experiment.experiment_id, ExecutionTarget(),
            )


def test_app_m6_schema_tables_exist(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        assert app.database.schema_version == APP_DATABASE_SCHEMA_VERSION
        names = {row[0] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"app_experiments", "app_experiment_executions",
                "app_experiment_qualifications", "app_experiment_branches"} <= names
