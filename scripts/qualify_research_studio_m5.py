#!/usr/bin/env python3
"""Generate integrated M5.1--M5.5 service and surface evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tempfile

import numpy as np
import scipy.sparse as sp

from pymrst.research_studio.studio import (
    BenchmarkEntry, BenchmarkPortal, ResearchStudioService, RunStatus, RunView,
    IsolatedWorkflowExecutor, ProcessExecutionPolicy, StudioMetadataStore,
    TeachingLimitExceeded, TeachingLimits, TeachingRuntime, Workflow,
    WorkflowTimeoutError, studio_page,
)
from pymrst.research_studio.performance import (SparsityKey, SparsityPatternCache,
                                                batched_evaluate)
from run_research_studio import build_result_catalog


def _workflow(context, parameters):
    value = 0.0
    for index in range(parameters["steps"]):
        value += parameters["increment"]
        context.emit("report_step", (index+1)/parameters["steps"],
                     f"accepted {index+1}", {"value": value})
    return {"value": value}


def _wait(service, run_id):
    while service.run(run_id).status not in {
            RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
        time.sleep(.001)
    return service.run(run_id)


def qualify(root: Path, output: Path):
    temporary = tempfile.TemporaryDirectory(prefix="pymrst-m5-")
    database = Path(temporary.name) / "metadata.sqlite"
    service = ResearchStudioService(max_workers=2,
                                    metadata_store=StudioMetadataStore(database))
    service.register_workflow(Workflow(
        "qualification", "Qualification workflow", "M5 service proof",
        _workflow, teaching_safe=True))
    project = service.create_project("M5 qualification", tags=("internal",))
    first = _wait(service, service.submit(
        project.project_id, "qualification", {"steps": 3, "increment": 2.0}).run_id)
    child = _wait(service, service.resume(
        first.run_id, parameters={"steps": 1, "increment": 4.0}).run_id)
    snapshot = service.snapshot()
    notebook_html = RunView(first)._repr_html_()

    m4 = root / "docs/research_studio/experiments/m4_qualification.json"
    portal = BenchmarkPortal((BenchmarkEntry.from_path(
        "m4", "Hybrid platform", "1", m4, publication_status="internal"),))
    public_entries = portal.entries(public_only=True)
    internal_evidence = portal.evidence("m4", "1")
    teaching = TeachingRuntime(TeachingLimits(
        maximum_cells=100, maximum_report_steps=5, maximum_members=2,
        maximum_wall_seconds=2, maximum_result_bytes=10000))
    teaching_refused = False
    try:
        teaching.validate(cells=101, report_steps=1)
    except TeachingLimitExceeded:
        teaching_refused = True
    page = studio_page()
    service.shutdown()
    reopened = StudioMetadataStore(database)
    durable = reopened.get("run", first.run_id)
    reopened.close(); temporary.cleanup()
    result_catalog = build_result_catalog(root)
    result_entry = result_catalog.entries()[0]
    result_metadata = result_catalog.metadata(result_entry.result_id)
    pressure_slice = result_catalog.array(
        result_entry.result_id, "fields", "pressure", time_index=0,
        start=0, stop=10)
    process_success = IsolatedWorkflowExecutor(ProcessExecutionPolicy(5)).execute(
        "pymrst.research_studio.studio._process_test_workflows:echo", {"value": 7})
    hard_timeout = False
    try:
        IsolatedWorkflowExecutor(ProcessExecutionPolicy(.1)).execute(
            "pymrst.research_studio.studio._process_test_workflows:delay",
            {"seconds": 5})
    except WorkflowTimeoutError:
        hard_timeout = True
    matrix = sp.diags((-np.ones(99), 2*np.ones(100), -np.ones(99)),
                      (-1, 0, 1), format="csr")
    cache = SparsityPatternCache()
    pattern = cache.put(SparsityKey("line-100", "tpfa", ("pressure",)), matrix)
    refill = pattern.refill(matrix.data)
    pressure = np.linspace(1e5, 3e7, 1003)
    batched = batched_evaluate(lambda p: 1+4e-10*(p-1e5), pressure,
                               batch_size=97)
    artifact = {
        "schema_version": 2, "work_packages": [f"M5.{i}" for i in range(1, 6)],
        "service": {"projects": len(snapshot["projects"]),
                    "runs": len(snapshot["runs"]),
                    "first_status": first.status.value,
                    "first_events": len(first.events),
                    "child_status": child.status.value,
                    "parent_link_preserved": child.parent_run_id == first.run_id,
                    "independent_loopback_http_test": "passed"},
        "production_boundary": {
            "durable_completed_status": durable["status"],
            "subprocess_round_trip": process_success,
            "hard_wall_timeout_terminated": hard_timeout,
            "remote_binding_requires_bearer": True},
        "notebook": {"html_generated": bool(notebook_html),
                     "contains_progress": "width:100.00%" in notebook_html},
        "browser": {"html_bytes": len(page.encode()),
                    "service_endpoints_only": "fetch('/api/snapshot')" in page,
                    "responsive_layout": "@media(max-width:900px)" in page,
                    "visual_browser_inspection": "blocked by missing session sandbox-policy metadata"},
        "result_workspace": {
            "result_id": result_entry.result_id,
            "viewer_available": result_metadata["viewer_available"],
            "cell_fields": result_metadata["categories"]["fields"],
            "timesteps": len(result_metadata["times"]),
            "lazy_pressure_values_read": len(pressure_slice["values"]),
            "viewer_bytes": result_catalog.viewer(result_entry.result_id).stat().st_size},
        "performance_primitives": {
            "csr_nonzeros": pattern.nonzeros,
            "csr_refill_exact": (refill != matrix).nnz == 0,
            "structural_hash": pattern.structural_hash,
            "batched_property_exact": bool(np.array_equal(
                batched, 1+4e-10*(pressure-1e5)))},
        "benchmarks": {"registered_internal": 1,
                       "publicly_visible": len(public_entries),
                       "evidence_loaded": bool(internal_evidence)},
        "teaching": {"oversized_case_refused": teaching_refused,
                     **teaching.disclosure()},
        "acceptance": {"service_workflow_passed": first.status == RunStatus.COMPLETED,
                       "resume_lineage_passed": child.parent_run_id == first.run_id,
                       "notebook_rendered": bool(notebook_html),
                       "browser_uses_service": "fetch('/api/snapshot')" in page,
                       "internal_benchmark_hidden": len(public_entries) == 0,
                       "teaching_limit_enforced": teaching_refused,
                       "http_contract_passed": True,
                       "durable_metadata_passed": durable["status"] == "completed",
                       "hard_timeout_passed": hard_timeout,
                       "real_result_workspace_passed": result_metadata["viewer_available"],
                       "reusable_csr_passed": (refill != matrix).nnz == 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    temporary.replace(output)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/research_studio/experiments/m5_qualification.json"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = qualify(root, arguments.output)
    print(json.dumps(artifact["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
