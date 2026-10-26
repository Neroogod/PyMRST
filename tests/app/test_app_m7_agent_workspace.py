"""Acceptance tests for durable, evidence-linked App M7 investigations."""

from __future__ import annotations

from pathlib import Path
import time
import json
from types import SimpleNamespace

import numpy as np

import pytest

from pymrst.app import create_application
from pymrst.app.schemas import InvestigationRequest, ProjectCreate
from pymrst.research_studio.agents.base import ToolCall
from pymrst.research_studio.agents.orchestrator import InvestigationPlan


def finding(identifier="finding_pressure"):
    return {
        "schema_version": 1, "finding_id": identifier,
        "agent": "numerical_diagnostics", "category": "convergence.pressure",
        "severity": "warning", "claim_kind": "observed",
        "statement": "Pressure convergence slowed at report step 12.",
        "evidence": [{
            "kind": "convergence", "uri": "run://run_reference/events/12",
            "label": "Report-step convergence history",
            "locator": {"report_step": 12, "newton_iterations": 9},
            "checksum": "a" * 64,
        }],
        "confidence": 0.96, "affected_objects": ["report_step:12"],
        "recommendation": {
            "recommendation_id": "recommendation_timestep",
            "statement": "Review timestep growth near report step 12.",
            "rationale": "Iteration growth is localized and evidence-linked.",
            "requires_confirmation": True, "status": "proposed",
        },
    }


def result_run(app, project_id, *, pressure_offset=0.0, saturation=0.25):
    configuration = app.runs.create_configuration(
        project_id, f"Comparison {pressure_offset} {saturation}",
        "protocol-qualification", {"pressure_offset": pressure_offset, "saturation": saturation},
    )
    run = app.runs.create_run(configuration.configuration_id)
    root = Path(app.projects.get(project_id).root_path) / run.run_relative_path
    state_root = root / "states"; state_root.mkdir()
    for step in (1, 2):
        relative = f"states/report_{step:04d}.npz"
        np.savez_compressed(
            root / relative,
            pressure=np.arange(4, dtype=float) + 2.0e7 + step + pressure_offset,
            sW=np.full(4, saturation),
        )
        app.runs.append_event(run.run_id, "state_saved", step, {
            "report_step": step, "relative_path": relative,
        })
        app.runs.append_event(run.run_id, "progress", step, {
            "progress": step / 2, "report_step": step, "report_steps": 2,
            "simulated_time_seconds": step * 86400.0, "newton_iterations": 3,
        })
    app.results.index_run(run.run_id)
    return run


def test_investigation_findings_review_and_report_are_durable(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Agent workspace"))
        investigation = app.investigations.create(project.project_id, InvestigationRequest(
            "Why did pressure convergence slow?",
            selection={"report_steps": [12]}, attributes={"source": "studio"},
        ))
        assert app.investigations.create(project.project_id, InvestigationRequest(
            "Why did pressure convergence slow?",
            selection={"report_steps": [12]}, attributes={"source": "studio"},
        )) == investigation
        item = app.investigations.add_finding(investigation.investigation_id, finding())
        assert item.review_status.value == "pending"
        with pytest.raises(ValueError, match="human-accepted"):
            app.investigations.build_report(
                investigation.investigation_id, title="Convergence investigation",
            )
        reviewed = app.investigations.review(item.finding_id, "accepted", "Verified")
        assert reviewed.review_note == "Verified"
        report = app.investigations.build_report(
            investigation.investigation_id, title="Convergence investigation",
            qualified=True,
        )
        path = Path(project.root_path) / report.relative_path
        assert path.is_file()
        assert "Report-step convergence history" in path.read_text()
        assert report.status == "qualified"
        assert app.investigations.get(investigation.investigation_id).status.value == "completed"

    with create_application(root) as reopened:
        assert reopened.investigations.report(report.report_id) == report
        assert reopened.investigations.findings(investigation.investigation_id)[0] == reviewed


def test_finding_requires_typed_evidence_and_closed_investigation_is_immutable(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Evidence policy"))
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("What failed?"),
        )
        invalid = finding("missing_evidence")
        invalid["evidence"] = []
        with pytest.raises(ValueError, match="supporting evidence"):
            app.investigations.add_finding(owner.investigation_id, invalid)
        item = app.investigations.add_finding(owner.investigation_id, finding("valid"))
        app.investigations.review(item.finding_id, "accepted")
        app.investigations.build_report(owner.investigation_id, title="Failure report")
        with pytest.raises(ValueError, match="closed"):
            app.investigations.add_finding(owner.investigation_id, finding("late"))


def test_app_m7_schema_contains_agent_history_tables(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        tables = {row["name"] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )}
        assert {"app_investigations", "app_investigation_findings",
                "app_engineering_reports", "app_variant_proposals",
                "app_investigation_plans",
                "app_investigation_executions", "app_conflict_decisions",
                "app_investigation_syntheses"} <= tables


def test_automatic_diagnostics_are_read_only_deterministic_and_evidence_linked(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Automatic diagnostics"))
        configuration = app.runs.create_configuration(
            project.project_id, "Probe", "protocol-qualification", {"steps": 3},
        )
        run = app.isolated.submit(configuration.configuration_id)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = app.runs.get(run.run_id)
            if run.status.value in {"completed", "failed", "cancelled"}:
                break
            time.sleep(.02)
        assert run.status.value == "completed"
        owner = app.investigations.create(project.project_id, InvestigationRequest(
            "Assess convergence", (run.run_id,),
        ))
        generated = app.investigations.analyze(owner.investigation_id)
        assert {
            "numerical.run_status", "numerical.nonlinear_iterations",
            "numerical.summary",
        } <= {item.finding["category"] for item in generated}
        assert {"reservoir_performance", "conservation"} <= {
            item.finding["agent"] for item in generated
        }
        assert generated[0].finding["evidence"][0]["uri"] == f"run://{run.run_id}/events"
        assert len(generated[0].finding["evidence"][0]["checksum"]) == 64
        # Analysis stores evidence but does not start a child run or alter status.
        assert app.runs.list(project.project_id) == (run,)
        assert app.runs.get(run.run_id).status.value == "completed"


def test_variant_proposal_requires_accepted_recommendation_and_never_executes(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Controlled proposal"))
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Should timestep controls change?"),
        )
        item = app.investigations.add_finding(owner.investigation_id, finding("proposal_source"))
        with pytest.raises(ValueError, match="human-accepted"):
            app.investigations.propose_variant(
                item.finding_id, name="Timestep study", proposal_kind="run_configuration",
                proposal={"max_timestep_days": 5},
            )
        app.investigations.review(item.finding_id, "accepted", "Evidence verified")
        proposal = app.investigations.propose_variant(
            item.finding_id, name="Timestep study", proposal_kind="run_configuration",
            proposal={"max_timestep_days": 5, "purpose": "diagnostic"},
        )
        assert proposal.status.value == "proposed"
        assert app.runs.list(project.project_id) == ()
        approved = app.investigations.decide_proposal(
            proposal.proposal_id, "approved", "Run separately after resource review",
        )
        assert approved.status.value == "approved"
        assert app.runs.list(project.project_id) == ()
        with pytest.raises(ValueError, match="immutable"):
            app.investigations.decide_proposal(
                proposal.proposal_id, "rejected", "Changed mind",
            )


def test_project_only_investigation_runs_deck_grid_rock_initialization_and_well_specialists(tmp_path):
    source = tmp_path / "CASE.DATA"
    source.write_text(
        "RUNSPEC\nMETRIC\nDIMENS\n2 1 1 /\nGRID\nPROPS\nSOLUTION\nSCHEDULE\n"
    )
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Input specialists"))
        asset = app.projects.import_input(project.project_id, source)
        app.models.inspect(project.project_id, asset.asset_id)
        grid = SimpleNamespace(
            cells=SimpleNamespace(
                num=2, centroids=np.array([[0.5, .5, .5], [1.5, .5, .5]]),
                volumes=np.array([1.0, 1.0]), index_map=np.arange(2),
            ),
            faces=SimpleNamespace(num=11), nodes=SimpleNamespace(num=12),
        )
        app.grids.register(project.project_id, grid, properties={
            "PORO": np.array([.2, .25]), "PERMX": np.array([100., 120.]),
            "PRESSURE": np.array([2.0e7, 2.1e7]),
            "SWAT": np.array([.2, .3]), "SOIL": np.array([.8, .7]),
        })
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Is this case ready to simulate?"),
        )
        generated = app.investigations.analyze(owner.investigation_id)
        agents = {item.finding["agent"] for item in generated}
        assert {"deck_review", "grid_quality", "rock_fluid", "initialization",
                "wells_schedule"} <= agents
        assert all(item.finding["evidence"][0]["checksum"] for item in generated)
        assert app.runs.list(project.project_id) == ()


def test_saved_state_performance_and_conservation_specialists_use_indexed_evidence(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Result specialists"))
        configuration = app.runs.create_configuration(
            project.project_id, "Evidence fixture", "protocol-qualification", {},
        )
        run = app.runs.create_run(configuration.configuration_id)
        app.runs.append_event(run.run_id, "convergence", 1.0, {
            "newton_iterations": 5, "timestep_cuts": 0,
            "residuals": {"MB_oil": 2.5e-7},
        })
        field_rows = []
        for step, mean in ((1, 2.5e7), (2, 2.4e7)):
            fields = [{
                "name": "pressure", "shape": [2], "dtype": "<f8",
                "minimum": mean - 1e5, "maximum": mean + 1e5,
                "mean": mean, "nonfinite_count": 0,
            }]
            field_rows.append((
                f"state_{step}", run.run_id, step, float(step),
                f"states/report_{step:04d}.npz", str(step) * 64,
                json.dumps(fields), "2026-01-01T00:00:00+00:00",
            ))
        with app.database.transaction() as connection:
            connection.executemany(
                "INSERT INTO app_result_states VALUES(?,?,?,?,?,?,?,?)", field_rows,
            )
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Assess depletion and balance", (run.run_id,)),
        )
        generated = app.investigations.analyze(owner.investigation_id)
        statements = [item.finding["statement"] for item in generated]
        assert any("Mean reservoir pressure changed" in value for value in statements)
        assert any("material-balance residual" in value for value in statements)


def test_cross_run_full_array_comparison_uses_explicit_tolerances(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Cross-run investigation"))
        reference = result_run(app, project.project_id, pressure_offset=0)
        candidate = result_run(app, project.project_id, pressure_offset=2)
        owner = app.investigations.create(project.project_id, InvestigationRequest(
            "Do the candidate and reference agree?", (reference.run_id, candidate.run_id),
            attributes={"comparison_tolerances": {
                "pressure": {"absolute": 1.0, "relative": 1e-4},
                "sW": {"absolute": 1e-12, "relative": 1e-12},
            }},
        ))
        generated = app.investigations.analyze(owner.investigation_id)
        disagreements = [item for item in generated
                         if item.finding["agent"] == "simulator_disagreement"]
        pressure = next(item for item in disagreements
                        if item.finding["category"] == "simulator_disagreement.pressure")
        assert pressure.finding["severity"] == "error"
        assert "maximum absolute difference 2" in pressure.finding["statement"]
        assert app.database.fetch_one(
            "SELECT COUNT(*) FROM app_result_comparisons",
        )[0] == 1


def test_full_array_invariant_check_detects_saturation_violation(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Full-array invariants"))
        run = result_run(app, project.project_id, saturation=1.2)
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Are states physical?", (run.run_id,)),
        )
        generated = app.investigations.analyze(owner.investigation_id)
        invariant = [item for item in generated if item.finding["category"] == "invariant"]
        assert invariant
        assert any("sW@" in item.finding["statement"] for item in invariant)
        assert all(item.finding["evidence"][0]["checksum"] for item in invariant)


def test_application_planning_is_idempotent_persistent_and_does_not_execute(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Planning history"))
        run = result_run(app, project.project_id)
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Plan diagnostics", (run.run_id,)),
        )
        first = app.investigations.plan(owner.investigation_id)
        second = app.investigations.plan(owner.investigation_id)
        assert first == second
        calls = first.plan["plan"]["calls"]
        assert {item["tool"] for item in calls} == {
            "analyze_numerical_diagnostics", "analyze_reservoir_performance",
            "analyze_conservation", "analyze_invariants",
        }
        assert len(app.runs.list(project.project_id)) == 1
        assert app.investigations.findings(owner.investigation_id) == ()
    with create_application(root) as reopened:
        assert reopened.investigations.investigation_plan(first.plan_id) == first


def test_analysis_executes_only_the_selected_persisted_plan_and_seals_ledger(tmp_path):
    class NumericalOnlyProvider:
        def complete(self, messages, tools, response_type):
            run_id = messages[-1]["content"]["run_ids"][0]
            return InvestigationPlan((ToolCall(
                "numerical_only", "analyze_numerical_diagnostics",
                {"run_id": run_id},
            ),))

    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Exact plan execution"))
        run = result_run(app, project.project_id)
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Only inspect convergence", (run.run_id,)),
        )
        plan = app.investigations.plan(
            owner.investigation_id, provider=NumericalOnlyProvider(),
            provider_name="test-numerical-only",
        )
        findings = app.investigations.analyze(
            owner.investigation_id, plan_id=plan.plan_id,
        )
        assert {item.finding["agent"] for item in findings} == {"numerical_diagnostics"}
        executions = app.investigations.executions(owner.investigation_id)
        assert len(executions) == 1
        execution = executions[0]
        assert execution.plan_id == plan.plan_id
        assert execution.plan_hash == plan.content_hash
        assert execution.status == "completed"
        assert [item["call"]["call_id"] for item in execution.execution["calls"]] == [
            "numerical_only"
        ]
        assert execution.execution["calls"][0]["result"]["ok"] is True
        assert app.investigations.execution(execution.execution_id) == execution

    with create_application(root) as reopened:
        assert reopened.investigations.executions(owner.investigation_id)[0] == execution


def test_cross_specialist_synthesis_requires_immutable_human_conflict_resolution(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Synthesis review"))
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Which pressure interpretation controls?"),
        )
        left = finding("finding_left")
        left["agent"] = "material_balance"
        left["statement"] = "Pressure behavior indicates acceptable balance."
        left["claim_kind"] = "uncertain"
        left["confidence"] = 0.72
        right = finding("finding_right")
        right["agent"] = "numerical_diagnostics"
        right["statement"] = "Pressure behavior indicates unacceptable balance."
        first = app.investigations.add_finding(owner.investigation_id, left)
        second = app.investigations.add_finding(owner.investigation_id, right)
        app.investigations.review(first.finding_id, "accepted", "Evidence inspected")
        app.investigations.review(second.finding_id, "accepted", "Evidence inspected")

        synthesis = app.investigations.synthesize(owner.investigation_id)
        assert synthesis.synthesis["summary"]["unresolved_conflict_count"] == 1
        assert synthesis.synthesis["summary"]["uncertainty_count"] == 2
        conflict = synthesis.synthesis["conflicts"][0]
        with pytest.raises(ValueError, match="human resolution"):
            app.investigations.build_report(
                owner.investigation_id, title="Blocked report", qualified=True,
            )

        resolved = app.investigations.resolve_conflict(
            owner.investigation_id, conflict["conflict_id"], action="resolve",
            reviewer="reservoir.engineer@example.com",
            rationale="The material-balance evidence controls this interpretation.",
            selected_finding_ids=(first.finding_id,),
        )
        assert resolved.synthesis["summary"]["unresolved_conflict_count"] == 0
        assert resolved.synthesis["conflicts"][0]["resolution"]["reviewer"] == (
            "reservoir.engineer@example.com"
        )
        with pytest.raises(ValueError, match="immutable"):
            app.investigations.resolve_conflict(
                owner.investigation_id, conflict["conflict_id"], action="dismiss",
                reviewer="second.reviewer@example.com", rationale="Try to overwrite.",
            )
        report = app.investigations.build_report(
            owner.investigation_id, title="Resolved report", qualified=True,
        )
        assert report.status == "qualified"

    with create_application(root) as reopened:
        history = reopened.investigations.syntheses(owner.investigation_id)
        assert history[-1] == resolved


def test_evidence_bound_narrative_requires_human_approval_and_is_durable(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Narrative review"))
        owner = app.investigations.create(
            project.project_id, InvestigationRequest("Do pressure results agree?"),
        )
        accepted = app.investigations.add_finding(
            owner.investigation_id, finding("finding_accepted"),
        )
        app.investigations.add_finding(
            owner.investigation_id, finding("finding_still_pending"),
        )
        app.investigations.review(accepted.finding_id, "accepted", "Evidence checked")
        draft = app.investigations.draft_narrative(owner.investigation_id)
        assert draft.status == "draft"
        referenced = {
            identifier for section in draft.narrative["sections"]
            for identifier in section["finding_ids"]
        }
        assert referenced == {accepted.finding_id}
        assert draft.narrative["sections"][0]["evidence"]

        # Draft prose is not report content before a named human decision.
        before = app.investigations.build_report(
            owner.investigation_id, title="Draft-only report",
        )
        assert before.report["executive_summary"] != draft.narrative["executive_summary"]

    # Reopen to prove that draft and review lifecycle are database-backed.
    with create_application(root) as reopened:
        stored = reopened.investigations.narrative(draft.narrative_id)
        assert stored.status == "draft"
        approved = reopened.investigations.review_narrative(
            draft.narrative_id, status="approved", reviewer="reservoir.engineer",
            note="Every cited finding and caveat was checked.",
        )
        assert approved.status == "approved"
        with pytest.raises(ValueError, match="immutable"):
            reopened.investigations.review_narrative(
                draft.narrative_id, status="rejected", reviewer="another.engineer",
                note="Attempt to replace the recorded decision.",
            )
        report = reopened.investigations.build_report(
            owner.investigation_id, title="Approved narrative report", qualified=True,
        )
        assert report.report["executive_summary"] == draft.narrative["executive_summary"]
        section = report.report["sections"][0]
        assert section["finding_ids"] == [accepted.finding_id]
        assert section["evidence"]
