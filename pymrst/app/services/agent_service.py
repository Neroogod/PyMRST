"""Durable application boundary for evidence-controlled investigations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from pymrst.research_studio.agents.context import InvestigationContext, Selection
from pymrst.research_studio.agents.permissions import AgentPermission, PermissionGrant
from pymrst.research_studio.agents.schemas.findings import EngineeringFinding
from pymrst.research_studio.agents.base import ToolCall, ToolDescriptor, ToolResult
from pymrst.research_studio.agents.orchestrator import InvestigationPlan
from pymrst.research_studio.agents.specialists.numerical_diagnostics import NumericalDiagnosticsSpecialist
from pymrst.research_studio.agents.tools.application_run_tools import (
    AnalyzeApplicationRunTool, RunEventCatalog, RunEventSnapshot,
)
from pymrst.research_studio.agents.tools.application_case_tools import (
    AnalyzeDeckInspectionTool, AnalyzeGridSnapshotTool,
    AnalyzeInitializationSnapshotTool, AnalyzeRockFluidSnapshotTool,
    AnalyzeWellsScheduleSnapshotTool, ApplicationCaseCatalog,
    ApplicationCaseSnapshot,
)
from pymrst.research_studio.agents.specialists.deck_review import DeckReviewSpecialist
from pymrst.research_studio.agents.specialists.grid_quality import GridQualitySpecialist
from pymrst.research_studio.agents.specialists.initialization import InitializationSpecialist
from pymrst.research_studio.agents.specialists.rock_fluid import RockFluidSpecialist
from pymrst.research_studio.agents.specialists.wells_schedule import WellsScheduleSpecialist
from pymrst.research_studio.agents.tools.application_result_tools import (
    AnalyzeApplicationConservationTool, AnalyzeApplicationPerformanceTool,
    ApplicationResultCatalog, ApplicationResultSnapshot,
)
from pymrst.research_studio.agents.specialists.conservation import ConservationSpecialist
from pymrst.research_studio.agents.specialists.reservoir_performance import ReservoirPerformanceSpecialist
from pymrst.research_studio.agents.tools.application_comparison_tools import (
    AnalyzeApplicationComparisonTool, ApplicationComparisonCatalog,
    ApplicationComparisonSnapshot,
)
from pymrst.research_studio.agents.specialists.simulator_disagreement import SimulatorDisagreementSpecialist
from pymrst.research_studio.agents.tools.validation_tools import InvariantTool, ValidationDataCatalog
from pymrst.research_studio.agents.specialists.conservation import InvariantSpecialist
from pymrst.research_studio.agents.planning import (
    BoundedInvestigationPlanner, DeterministicPlanningProvider,
)
from pymrst.research_studio.agents.review import (
    HumanReviewDecision, ReviewAction, ReviewTarget,
)
from pymrst.research_studio.agents.synthesis import synthesize_findings
from pymrst.research_studio.agents.narrative import (
    DeterministicNarrativeProvider, EvidenceLinkedNarrative,
    NarrativeSynthesizer, findings_hash,
)
from pymrst.research_studio.agents.tools.report_tools import (
    EngineeringReportBuilder, render_markdown,
)

from ..database.models import AppDatabase
from ..schemas._common import json_value, utc_now
from ..schemas.agent import (
    EngineeringReportRecord, FindingRecord, FindingReviewStatus,
    InvestigationRecord, InvestigationRequest, InvestigationStatus,
    VariantProposalRecord, VariantProposalStatus,
    InvestigationPlanRecord, InvestigationExecutionRecord,
    InvestigationSynthesisRecord,
    InvestigationNarrativeRecord,
)
from .project_service import ProjectService
from .result_service import ResultService


def _encoded(value) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _investigation(row) -> InvestigationRecord:
    return InvestigationRecord(
        row["investigation_id"], row["project_id"], row["question"],
        InvestigationStatus(row["status"]), json.loads(row["context_json"]),
        row["content_hash"], row["created_at"], row["updated_at"],
    )


def _finding(row) -> FindingRecord:
    return FindingRecord(
        row["finding_id"], row["investigation_id"], row["project_id"],
        json.loads(row["finding_json"]), row["content_hash"],
        FindingReviewStatus(row["review_status"]), row["review_note"],
        row["created_at"], row["updated_at"],
    )


def _report(row) -> EngineeringReportRecord:
    return EngineeringReportRecord(
        row["report_id"], row["investigation_id"], row["project_id"],
        row["content_hash"], json.loads(row["report_json"]),
        row["relative_path"], row["status"], row["created_at"],
    )


def _proposal(row) -> VariantProposalRecord:
    return VariantProposalRecord(
        row["proposal_id"], row["investigation_id"], row["finding_id"],
        row["project_id"], row["name"], row["proposal_kind"],
        row["content_hash"], json.loads(row["proposal_json"]),
        VariantProposalStatus(row["status"]), row["decision_note"],
        row["created_at"], row["updated_at"],
    )


def _plan(row) -> InvestigationPlanRecord:
    return InvestigationPlanRecord(
        row["plan_id"], row["investigation_id"], row["project_id"],
        row["provider"], row["content_hash"], json.loads(row["plan_json"]),
        row["created_at"],
    )


def _execution(row) -> InvestigationExecutionRecord:
    return InvestigationExecutionRecord(
        row["execution_id"], row["investigation_id"], row["plan_id"],
        row["project_id"], row["plan_hash"], row["content_hash"],
        row["status"], json.loads(row["execution_json"]), row["created_at"],
        row["completed_at"],
    )


def _synthesis(row) -> InvestigationSynthesisRecord:
    return InvestigationSynthesisRecord(
        row["synthesis_id"], row["investigation_id"], row["project_id"],
        row["finding_set_hash"], row["decision_set_hash"],
        row["content_hash"], json.loads(row["synthesis_json"]), row["created_at"],
    )


def _narrative(row) -> InvestigationNarrativeRecord:
    value = json.loads(row["narrative_json"])
    value["status"] = row["status"]
    return InvestigationNarrativeRecord(
        row["narrative_id"], row["investigation_id"], row["project_id"],
        row["provider"], row["provider_configuration_hash"],
        row["source_findings_hash"], row["content_hash"], value,
        row["status"], row["reviewer"], row["review_note"],
        row["created_at"], row["updated_at"],
    )


class AgentInvestigationService:
    """Persist agent outputs without granting hidden simulation authority."""

    def __init__(
        self, database: AppDatabase, projects: ProjectService,
        results: ResultService | None = None, providers=None,
    ) -> None:
        self.database, self.projects, self.results = database, projects, results
        self.providers = providers
        self.report_builder = EngineeringReportBuilder()
        self.narrative_synthesizer = NarrativeSynthesizer()

    def create(self, project_id: str, request: InvestigationRequest) -> InvestigationRecord:
        project = self.projects.get(project_id)
        if not isinstance(request, InvestigationRequest):
            raise TypeError("investigation creation requires InvestigationRequest")
        for run_id in request.run_ids:
            row = self.database.fetch_one(
                "SELECT project_id FROM app_runs WHERE run_id=?", (run_id,),
            )
            if row is None or row["project_id"] != project_id:
                raise ValueError(f"run does not belong to investigation project: {run_id}")
        context = {
            "question": request.question, "case_id": project_id,
            "case_content_hash": project.content_hash,
            "run_ids": list(request.run_ids), "selection": dict(request.selection),
            "attributes": dict(request.attributes),
            "permission": PermissionGrant(AgentPermission.ANALYZE).to_dict(),
        }
        content_hash = _digest(context)
        existing = self.database.fetch_one(
            "SELECT * FROM app_investigations WHERE project_id=? AND content_hash=?",
            (project_id, content_hash),
        )
        if existing is not None:
            return _investigation(existing)
        now = utc_now()
        record = InvestigationRecord(
            f"investigation_{uuid4().hex}", project_id, request.question,
            InvestigationStatus.OPEN, context, content_hash, now, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_investigations VALUES(?,?,?,?,?,?,?,?)",
                (record.investigation_id, project_id, record.question,
                 record.status.value, _encoded(context), content_hash, now, now),
            )
        return record

    def get(self, investigation_id: str) -> InvestigationRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigations WHERE investigation_id=?",
            (str(investigation_id),),
        )
        if row is None:
            raise KeyError(f"unknown investigation: {investigation_id}")
        return _investigation(row)

    def list(self, project_id: str) -> tuple[InvestigationRecord, ...]:
        self.projects.get(project_id)
        return tuple(_investigation(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigations WHERE project_id=? ORDER BY created_at",
            (project_id,),
        ))

    def add_finding(self, investigation_id: str, payload) -> FindingRecord:
        owner = self.get(investigation_id)
        if owner.status is not InvestigationStatus.OPEN:
            raise ValueError("findings cannot be added to a closed investigation")
        finding = EngineeringFinding.from_dict(payload)
        data = finding.to_dict()
        content_hash, now = _digest(data), utc_now()
        existing = self.database.fetch_one(
            "SELECT * FROM app_investigation_findings WHERE investigation_id=? AND content_hash=?",
            (investigation_id, content_hash),
        )
        if existing is not None:
            return _finding(existing)
        if self.database.fetch_one(
            "SELECT finding_id FROM app_investigation_findings WHERE finding_id=?",
            (finding.finding_id,),
        ) is not None:
            raise ValueError(f"finding id already exists: {finding.finding_id}")
        record = FindingRecord(
            finding.finding_id, investigation_id, owner.project_id, data,
            content_hash, FindingReviewStatus.PENDING, "", now, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_investigation_findings VALUES(?,?,?,?,?,?,?,?,?)",
                (record.finding_id, investigation_id, owner.project_id,
                 _encoded(data), content_hash, record.review_status.value, "", now, now),
            )
        return record

    def plan(
        self, investigation_id: str, *, provider=None,
        provider_name="deterministic-v1", maximum_calls=32,
        provider_metadata=None,
    ) -> InvestigationPlanRecord:
        owner = self.get(investigation_id)
        if owner.status is not InvestigationStatus.OPEN:
            raise ValueError("closed investigations cannot be replanned")
        context = InvestigationContext(
            owner.investigation_id, owner.question, owner.project_id,
            owner.context["case_content_hash"],
            PermissionGrant(AgentPermission.ANALYZE),
            tuple(owner.context.get("run_ids", ())),
            Selection(**owner.context.get("selection", {})),
            attributes=owner.context.get("attributes", {}),
        )
        descriptors = []
        has_case = self.database.fetch_one(
            "SELECT EXISTS(SELECT 1 FROM app_models WHERE project_id=?) OR "
            "EXISTS(SELECT 1 FROM app_grid_snapshots WHERE project_id=?)",
            (owner.project_id, owner.project_id),
        )[0]
        if has_case:
            descriptors.extend(ToolDescriptor(name, description, AgentPermission.ANALYZE)
                               for name, description in (
                ("analyze_deck_review", "Inspect deck structure and validation findings"),
                ("analyze_grid_quality", "Inspect grid topology and geometry"),
                ("analyze_rock_fluid", "Inspect rock and fluid properties"),
                ("analyze_initialization", "Inspect initial pressure and saturations"),
                ("analyze_wells_schedule", "Inspect wells and schedule controls"),
            ))
        if context.run_ids:
            descriptors.extend(ToolDescriptor(name, description, AgentPermission.ANALYZE)
                               for name, description in (
                ("analyze_numerical_diagnostics", "Inspect convergence and failures"),
                ("analyze_reservoir_performance", "Inspect saved reservoir performance"),
                ("analyze_conservation", "Inspect conservation evidence"),
            ))
        project = self.projects.get(owner.project_id)

        def has_complete_state_payload(run_id):
            run = self.database.fetch_one(
                "SELECT run_relative_path FROM app_runs WHERE run_id=? AND project_id=?",
                (run_id, owner.project_id),
            )
            rows = self.database.fetch_all(
                "SELECT relative_path FROM app_result_states WHERE run_id=? ORDER BY report_step",
                (run_id,),
            )
            if run is None or not rows:
                return False
            root = Path(project.root_path) / run["run_relative_path"]
            return all((root / row["relative_path"]).is_file() for row in rows)

        full_array_runs = {
            str(run_id) for run_id in context.run_ids
            if has_complete_state_payload(run_id)
        }
        if context.run_ids and full_array_runs == {str(item) for item in context.run_ids}:
            descriptors.append(ToolDescriptor(
                "analyze_invariants", "Check checksum-verified full saved state arrays",
                AgentPermission.ANALYZE,
            ))
        if len(context.run_ids) >= 2 and full_array_runs == {
            str(item) for item in context.run_ids
        }:
            descriptors.append(ToolDescriptor(
                "compare_reference_runs", "Compare bounded full run arrays",
                AgentPermission.ANALYZE,
            ))
        planner = BoundedInvestigationPlanner(
            provider or DeterministicPlanningProvider(),
            provider_name=provider_name, maximum_calls=maximum_calls,
            provider_metadata=(provider_metadata or {
                "kind": "deterministic", "implementation": "deterministic-v1",
            }),
        )
        planned = planner.plan(context, tuple(descriptors))
        value = planned.to_dict()
        plan_id = "plan_" + planned.content_hash[:24]
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_investigation_plans VALUES(?,?,?,?,?,?,?)",
                (plan_id, investigation_id, owner.project_id, planned.provider,
                 planned.content_hash, _encoded(value), now),
            )
        return self.investigation_plan(plan_id)

    def investigation_plan(self, plan_id: str) -> InvestigationPlanRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_plans WHERE plan_id=?", (plan_id,),
        )
        if row is None:
            raise KeyError(f"unknown investigation plan: {plan_id}")
        return _plan(row)

    def plans(self, investigation_id: str) -> tuple[InvestigationPlanRecord, ...]:
        self.get(investigation_id)
        return tuple(_plan(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigation_plans WHERE investigation_id=? ORDER BY created_at",
            (investigation_id,),
        ))

    def execution(self, execution_id: str) -> InvestigationExecutionRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_executions WHERE execution_id=?",
            (execution_id,),
        )
        if row is None:
            raise KeyError(f"unknown investigation execution: {execution_id}")
        return _execution(row)

    def executions(self, investigation_id: str) -> tuple[InvestigationExecutionRecord, ...]:
        self.get(investigation_id)
        return tuple(_execution(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigation_executions WHERE investigation_id=? "
            "ORDER BY created_at", (investigation_id,),
        ))

    def _conflict_decisions(self, investigation_id: str):
        return tuple(HumanReviewDecision.from_dict(json.loads(row["decision_json"]))
                     for row in self.database.fetch_all(
            "SELECT decision_json FROM app_conflict_decisions "
            "WHERE investigation_id=? ORDER BY created_at", (investigation_id,),
        ))

    def synthesize(self, investigation_id: str) -> InvestigationSynthesisRecord:
        owner = self.get(investigation_id)
        records = tuple(item for item in self.findings(investigation_id)
                        if item.review_status is not FindingReviewStatus.REJECTED)
        findings = tuple(EngineeringFinding.from_dict(item.finding) for item in records)
        decisions = self._conflict_decisions(investigation_id)
        synthesis = synthesize_findings(findings, decisions)
        value = synthesis.to_dict()
        finding_set_hash = _digest([
            {"finding_id": item.finding_id, "content_hash": item.content_hash}
            for item in records
        ])
        decision_set_hash = _digest([item.to_dict() for item in decisions])
        content_hash = _digest(value)
        synthesis_id = "synthesis_" + _digest({
            "investigation_id": investigation_id,
            "finding_set_hash": finding_set_hash,
            "decision_set_hash": decision_set_hash,
        })[:24]
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_investigation_syntheses "
                "VALUES(?,?,?,?,?,?,?,?)",
                (synthesis_id, investigation_id, owner.project_id,
                 finding_set_hash, decision_set_hash, content_hash,
                 _encoded(value), now),
            )
        return self.synthesis(synthesis_id)

    def synthesis(self, synthesis_id: str) -> InvestigationSynthesisRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_syntheses WHERE synthesis_id=?",
            (synthesis_id,),
        )
        if row is None:
            raise KeyError(f"unknown investigation synthesis: {synthesis_id}")
        return _synthesis(row)

    def syntheses(self, investigation_id: str) -> tuple[InvestigationSynthesisRecord, ...]:
        self.get(investigation_id)
        return tuple(_synthesis(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigation_syntheses WHERE investigation_id=? "
            "ORDER BY created_at", (investigation_id,),
        ))

    def _accepted_narrative_findings(self, investigation_id: str):
        accepted = tuple(
            EngineeringFinding.from_dict(item.finding)
            for item in self.findings(investigation_id)
            if item.review_status is FindingReviewStatus.ACCEPTED
        )
        if not accepted:
            raise ValueError("narrative synthesis requires human-accepted findings")
        synthesis = synthesize_findings(
            accepted, self._conflict_decisions(investigation_id),
        )
        if synthesis.unresolved_conflicts:
            raise ValueError(
                "narrative synthesis requires human resolution of all accepted-finding conflicts"
            )
        excluded = set()
        for conflict in synthesis.conflicts:
            decision = conflict.resolution
            if decision is not None and decision.action is ReviewAction.RESOLVE:
                excluded.update(set(conflict.finding_ids) - set(decision.selected_finding_ids))
        selected = tuple(item for item in accepted if item.finding_id not in excluded)
        if not selected:
            raise ValueError("conflict decisions excluded every accepted finding")
        return selected

    def draft_narrative(
        self, investigation_id: str, *, provider_id="deterministic-v1",
        provider=None, provider_configuration_hash: str | None = None,
    ) -> InvestigationNarrativeRecord:
        owner = self.get(investigation_id)
        provider_id = str(provider_id).strip()
        if not provider_id:
            raise ValueError("narrative provider id cannot be empty")
        findings = self._accepted_narrative_findings(investigation_id)
        if provider is None and provider_id == "deterministic-v1":
            provider = DeterministicNarrativeProvider()
            provider_configuration_hash = _digest({"provider": provider_id, "version": 1})
        elif provider is None:
            if self.providers is None:
                raise RuntimeError("reasoning-provider service is unavailable")
            record = self.providers.get(provider_id)
            provider = self.providers.build(provider_id)
            provider_configuration_hash = record.content_hash
        configuration_hash = str(provider_configuration_hash or "").strip()
        if not configuration_hash:
            raise ValueError("external narrative providers require a configuration hash")
        narrative = self.narrative_synthesizer.synthesize(
            investigation_id=investigation_id, question=owner.question,
            findings=findings, provider=provider, provider_name=provider_id,
            provider_configuration_hash=configuration_hash,
        )
        value, now = narrative.to_dict(), utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_investigation_narratives "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (narrative.narrative_id, investigation_id, owner.project_id,
                 provider_id, configuration_hash, narrative.source_findings_hash,
                 narrative.content_hash, _encoded(value), "draft", "", "", now, now),
            )
        return self.narrative(narrative.narrative_id)

    def narrative(self, narrative_id: str) -> InvestigationNarrativeRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_narratives WHERE narrative_id=?",
            (str(narrative_id),),
        )
        if row is None:
            raise KeyError(f"unknown investigation narrative: {narrative_id}")
        return _narrative(row)

    def narratives(self, investigation_id: str):
        self.get(investigation_id)
        return tuple(_narrative(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigation_narratives WHERE investigation_id=? "
            "ORDER BY created_at", (investigation_id,),
        ))

    def review_narrative(
        self, narrative_id: str, *, status: str, reviewer: str, note: str,
    ) -> InvestigationNarrativeRecord:
        status = str(status).strip().lower()
        if status not in {"approved", "rejected"}:
            raise ValueError("narrative review must approve or reject")
        reviewer, note = str(reviewer).strip(), str(note).strip()
        if not reviewer or not note:
            raise ValueError("narrative review requires a named reviewer and rationale")
        current = self.narrative(narrative_id)
        if current.status != "draft":
            if current.status == status:
                return current
            raise ValueError("a narrative review decision is immutable")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_investigation_narratives SET status=?,reviewer=?,"
                "review_note=?,updated_at=? WHERE narrative_id=? AND status='draft'",
                (status, reviewer, note, now, narrative_id),
            )
        return self.narrative(narrative_id)

    def resolve_conflict(
        self, investigation_id: str, conflict_id: str, *, action: str,
        reviewer: str, rationale: str, selected_finding_ids=(),
    ) -> InvestigationSynthesisRecord:
        current = self.synthesize(investigation_id)
        conflict = next((item for item in current.synthesis["conflicts"]
                         if item["conflict_id"] == conflict_id), None)
        if conflict is None:
            raise KeyError(f"unknown investigation conflict: {conflict_id}")
        if conflict["resolution"] is not None:
            raise ValueError("conflict already has an immutable human decision")
        action_value = ReviewAction(action)
        if action_value not in {ReviewAction.RESOLVE, ReviewAction.DISMISS}:
            raise ValueError("conflicts may only be resolved or dismissed")
        canonical = {
            "target": ReviewTarget.CONFLICT.value, "target_id": conflict_id,
            "action": action_value.value, "reviewer": str(reviewer).strip(),
            "rationale": str(rationale).strip(),
            "selected_finding_ids": list(selected_finding_ids), "metadata": {},
        }
        canonical["decision_id"] = "decision_" + _digest(canonical)[:24]
        decision = HumanReviewDecision.from_dict(canonical)
        if action_value is ReviewAction.RESOLVE and not decision.selected_finding_ids:
            raise ValueError("resolved conflicts require at least one selected finding")
        if not set(decision.selected_finding_ids).issubset(conflict["finding_ids"]):
            raise ValueError("conflict resolution selected an unrelated finding")
        digest, now = _digest(decision.to_dict()), utc_now()
        with self.database.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO app_conflict_decisions VALUES(?,?,?,?,?,?)",
                    (decision.decision_id, investigation_id, conflict_id,
                     digest, _encoded(decision.to_dict()), now),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("conflict already has an immutable human decision") from error
        return self.synthesize(investigation_id)

    def analyze(
        self, investigation_id: str, *, plan_id: str | None = None,
    ) -> tuple[FindingRecord, ...]:
        """Execute the latest persisted, bounded plan over immutable evidence.

        Planning and execution are deliberately separate.  This routine never
        invokes a reasoning provider and never starts, retries, or mutates a
        simulation. Every persisted call must execute exactly once, and its
        typed result is retained in the application execution ledger.
        """
        owner = self.get(investigation_id)
        if owner.status is not InvestigationStatus.OPEN:
            raise ValueError("closed investigations cannot be analyzed again")
        available_plans = self.plans(investigation_id)
        if plan_id is not None:
            plan_record = self.investigation_plan(plan_id)
            if plan_record.investigation_id != investigation_id:
                raise ValueError("execution plan belongs to another investigation")
        else:
            plan_record = available_plans[-1] if available_plans else self.plan(investigation_id)
        plan_value = dict(plan_record.plan)
        if plan_value.get("content_hash") != plan_record.content_hash:
            raise ValueError("persisted investigation plan failed its identity check")
        raw_calls = plan_value.get("plan", {}).get("calls", ())
        plan = InvestigationPlan(tuple(ToolCall.from_dict(item) for item in raw_calls))
        calls_by_tool: dict[str, list[ToolCall]] = {}
        for call in plan.calls:
            calls_by_tool.setdefault(call.tool, []).append(call)
        executed: dict[str, dict] = {}

        def planned(tool_name: str, **arguments):
            return tuple(call for call in calls_by_tool.get(tool_name, ())
                         if all(str(call.arguments.get(key)) == str(value)
                                for key, value in arguments.items()))

        def retain(call: ToolCall, result: ToolResult):
            if call.call_id in executed:
                raise RuntimeError(f"planned call executed more than once: {call.call_id}")
            if result.call_id != call.call_id:
                raise RuntimeError("tool result identity does not match the persisted plan")
            executed[call.call_id] = {
                "call": call.to_dict(), "result": result.to_dict(),
            }

        run_ids = tuple(owner.context.get("run_ids", ()))
        catalog = RunEventCatalog()
        snapshots = []
        for run_id in run_ids:
            run_row = self.database.fetch_one(
                "SELECT * FROM app_runs WHERE run_id=? AND project_id=?",
                (run_id, owner.project_id),
            )
            if run_row is None:
                raise ValueError(f"investigation run is unavailable: {run_id}")
            events = [dict(row) for row in self.database.fetch_all(
                "SELECT sequence,event_type,elapsed_seconds,payload_json,created_at "
                "FROM app_run_events WHERE run_id=? ORDER BY sequence", (run_id,),
            )]
            event_values = [{
                "sequence": row["sequence"], "event_type": row["event_type"],
                "elapsed_seconds": row["elapsed_seconds"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            } for row in events]
            checksum = _digest({"run": dict(run_row), "events": event_values})
            snapshots.append(catalog.register(RunEventSnapshot(
                str(run_id), str(run_row["status"]), tuple(event_values), checksum,
            )))
        tool, specialist = AnalyzeApplicationRunTool(catalog), NumericalDiagnosticsSpecialist()
        produced = []
        for snapshot in snapshots:
            for call in planned(tool.descriptor.name, run_id=snapshot.run_id):
                result = tool.invoke(call)
                retain(call, result)
                for item in specialist.analyze(result):
                    produced.append(self.add_finding(investigation_id, item.to_dict()))
        result_catalog = ApplicationResultCatalog()
        for snapshot in snapshots:
            state_rows = self.database.fetch_all(
                "SELECT report_step,simulated_time_seconds,content_hash,fields_json "
                "FROM app_result_states WHERE run_id=? ORDER BY report_step",
                (snapshot.run_id,),
            )
            states = tuple({
                "report_step": row["report_step"],
                "simulated_time_seconds": row["simulated_time_seconds"],
                "content_hash": row["content_hash"],
                "fields": json.loads(row["fields_json"]),
            } for row in state_rows)
            result_checksum = _digest({
                "run_checksum": snapshot.checksum, "states": states,
            })
            result_snapshot = result_catalog.register(ApplicationResultSnapshot(
                snapshot.run_id, result_checksum, states, snapshot.events,
            ))
            result_pipelines = (
                (AnalyzeApplicationPerformanceTool(result_catalog), ReservoirPerformanceSpecialist()),
                (AnalyzeApplicationConservationTool(result_catalog), ConservationSpecialist()),
            )
            for result_tool, result_specialist in result_pipelines:
                for call in planned(
                    result_tool.descriptor.name, run_id=result_snapshot.run_id,
                ):
                    result = result_tool.invoke(call)
                    retain(call, result)
                    for item in result_specialist.analyze(result):
                        produced.append(self.add_finding(investigation_id, item.to_dict()))
            invariant_calls = planned("analyze_invariants", run_id=snapshot.run_id)
            if invariant_calls and self.results is not None and states:
                available = {field["name"] for field in states[0]["fields"]}
                selected = tuple(sorted(
                    available & {"pressure", "sW", "sO", "sG", "rs", "rv"}
                ))
                if selected:
                    try:
                        arrays = self.results.analysis_fields(
                            snapshot.run_id, fields=selected, maximum_values=5_000_000,
                        )
                    except (KeyError, OSError, RuntimeError, ValueError):
                        arrays = {}
                    if arrays:
                        validation_catalog = ValidationDataCatalog(maximum_values=5_000_000)
                        dataset = validation_catalog.register(
                            f"full_arrays_{snapshot.run_id}", arrays,
                        )
                        checks = []
                        for name in arrays:
                            base = name.split("@", 1)[0]
                            checks.append({"kind": "finite", "field": name})
                            if base in {"sW", "sO", "sG"}:
                                checks.append({
                                    "kind": "bounded", "field": name,
                                    "lower": 0.0, "upper": 1.0, "tolerance": 1e-10,
                                })
                            elif base == "pressure":
                                checks.append({
                                    "kind": "bounded", "field": name,
                                    "lower": 0.0, "upper": 1e20, "tolerance": 0.0,
                                })
                        invariant_tool = InvariantTool(
                            validation_catalog, maximum_values=5_000_000,
                        )
                        for call in invariant_calls:
                            invariant_result = invariant_tool.invoke(ToolCall(
                                call.call_id, invariant_tool.descriptor.name,
                                {"dataset_id": dataset.dataset_id, "checks": checks},
                            ))
                            retain(call, invariant_result)
                            for item in InvariantSpecialist().analyze(invariant_result):
                                produced.append(self.add_finding(investigation_id, item.to_dict()))
        if self.results is not None and len(snapshots) >= 2:
            comparison_catalog = ApplicationComparisonCatalog()
            reference = snapshots[0].run_id
            tolerances = dict(owner.context.get("attributes", {}).get(
                "comparison_tolerances", {},
            ))
            for candidate in snapshots[1:]:
                comparison_calls = planned(
                    "compare_reference_runs", reference_run_id=reference,
                    candidate_run_id=candidate.run_id,
                )
                if not comparison_calls:
                    continue
                reference_count = self.database.fetch_one(
                    "SELECT COUNT(*) FROM app_result_states WHERE run_id=?", (reference,),
                )[0]
                candidate_count = self.database.fetch_one(
                    "SELECT COUNT(*) FROM app_result_states WHERE run_id=?", (candidate.run_id,),
                )[0]
                if not reference_count or not candidate_count:
                    continue
                comparison = self.results.compare(reference, candidate.run_id)
                comparison_snapshot = comparison_catalog.register(
                    ApplicationComparisonSnapshot(
                        comparison.comparison_id, reference, candidate.run_id,
                        comparison.content_hash, comparison.summary,
                    )
                )
                comparison_tool = AnalyzeApplicationComparisonTool(comparison_catalog)
                for call in comparison_calls:
                    comparison_result = comparison_tool.invoke(ToolCall(
                        call.call_id, comparison_tool.descriptor.name,
                        {"comparison_id": comparison_snapshot.comparison_id,
                         "tolerances": tolerances},
                    ))
                    retain(call, comparison_result)
                    for item in SimulatorDisagreementSpecialist().analyze(comparison_result):
                        produced.append(self.add_finding(investigation_id, item.to_dict()))
        model_row = self.database.fetch_one(
            "SELECT summary_json FROM app_models WHERE project_id=? "
            "ORDER BY created_at DESC,model_id DESC LIMIT 1", (owner.project_id,),
        )
        grid_row = self.database.fetch_one(
            "SELECT summary_json FROM app_grid_snapshots WHERE project_id=? "
            "ORDER BY created_at DESC,snapshot_id DESC LIMIT 1", (owner.project_id,),
        )
        if model_row is not None or grid_row is not None:
            model = json.loads(model_row[0]) if model_row is not None else {
                "model_id": "", "sections": [], "occurrences": [], "issues": [],
            }
            grid = None if grid_row is None else json.loads(grid_row[0])
            checksum = _digest({"model": model, "grid": grid})
            case_catalog = ApplicationCaseCatalog()
            case = case_catalog.register(ApplicationCaseSnapshot(
                owner.project_id, checksum, model, grid,
            ))
            pipelines = (
                (AnalyzeDeckInspectionTool(case_catalog), DeckReviewSpecialist()),
                (AnalyzeGridSnapshotTool(case_catalog), GridQualitySpecialist()),
                (AnalyzeRockFluidSnapshotTool(case_catalog), RockFluidSpecialist()),
                (AnalyzeInitializationSnapshotTool(case_catalog), InitializationSpecialist()),
                (AnalyzeWellsScheduleSnapshotTool(case_catalog), WellsScheduleSpecialist()),
            )
            for case_tool, case_specialist in pipelines:
                for call in planned(case_tool.analysis_kind, case_id=case.case_id):
                    result = case_tool.invoke(call)
                    retain(call, result)
                    for item in case_specialist.analyze(result):
                        produced.append(self.add_finding(investigation_id, item.to_dict()))
        missing = [call.call_id for call in plan.calls if call.call_id not in executed]
        if missing:
            raise RuntimeError(
                "persisted plan could not be satisfied by current immutable evidence: "
                + ", ".join(missing)
            )
        if not produced:
            raise ValueError(
                "automatic diagnostics require a bound run, inspected model, or grid snapshot"
            )
        completed = utc_now()
        execution_value = {
            "schema_version": 1, "plan_id": plan_record.plan_id,
            "plan_hash": plan_record.content_hash,
            "calls": [executed[call.call_id] for call in plan.calls],
            "finding_ids": [item.finding_id for item in produced],
            "status": "completed",
        }
        execution_hash = _digest(execution_value)
        execution_id = "execution_" + execution_hash[:24]
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_investigation_executions "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (execution_id, investigation_id, plan_record.plan_id,
                 owner.project_id, plan_record.content_hash, execution_hash,
                 "completed", _encoded(execution_value), completed, completed),
            )
        self.synthesize(investigation_id)
        return tuple(produced)

    def findings(self, investigation_id: str) -> tuple[FindingRecord, ...]:
        self.get(investigation_id)
        return tuple(_finding(row) for row in self.database.fetch_all(
            "SELECT * FROM app_investigation_findings WHERE investigation_id=? ORDER BY created_at",
            (investigation_id,),
        ))

    def review(self, finding_id: str, status, note: str = "") -> FindingRecord:
        status = FindingReviewStatus(status)
        if status is FindingReviewStatus.PENDING:
            raise ValueError("human review must accept or reject a finding")
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_findings WHERE finding_id=?",
            (finding_id,),
        )
        if row is None:
            raise KeyError(f"unknown finding: {finding_id}")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_investigation_findings SET review_status=?,review_note=?,updated_at=? WHERE finding_id=?",
                (status.value, str(note).strip(), now, finding_id),
            )
        return _finding(self.database.fetch_one(
            "SELECT * FROM app_investigation_findings WHERE finding_id=?", (finding_id,),
        ))

    def build_report(self, investigation_id: str, *, title: str, qualified=False) -> EngineeringReportRecord:
        owner = self.get(investigation_id)
        accepted = tuple(
            EngineeringFinding.from_dict(item.finding)
            for item in self.findings(investigation_id)
            if item.review_status is FindingReviewStatus.ACCEPTED
        )
        if not accepted:
            raise ValueError("a report requires at least one human-accepted finding")
        accepted_synthesis = synthesize_findings(
            accepted, self._conflict_decisions(investigation_id),
        )
        if qualified and accepted_synthesis.unresolved_conflicts:
            raise ValueError(
                "qualified reports require human resolution of all accepted-finding conflicts"
            )
        approved_narrative = None
        if not accepted_synthesis.unresolved_conflicts:
            narrative_findings = self._accepted_narrative_findings(investigation_id)
            accepted_hash = findings_hash(narrative_findings)
            for record in reversed(self.narratives(investigation_id)):
                if record.status == "approved" and record.source_findings_hash == accepted_hash:
                    approved_narrative = EvidenceLinkedNarrative.from_dict(record.narrative)
                    accepted = narrative_findings
                    accepted_synthesis = synthesize_findings(
                        accepted, self._conflict_decisions(investigation_id),
                    )
                    break
        context = InvestigationContext(
            owner.investigation_id, owner.question, owner.project_id,
            owner.context["case_content_hash"],
            PermissionGrant(AgentPermission.ANALYZE),
            tuple(owner.context.get("run_ids", ())),
            Selection(**owner.context.get("selection", {})),
            attributes=owner.context.get("attributes", {}),
        )
        report = self.report_builder.build(
            context, accepted, title=title, qualified=bool(qualified),
            limitations=(
                "Only human-accepted findings are included.",
                *(f"Unresolved conflict {item.conflict_id}: {item.reason}"
                  for item in accepted_synthesis.unresolved_conflicts),
                *(item.explanation for item in accepted_synthesis.uncertainties),
            ),
            narrative=approved_narrative,
        )
        project = self.projects.get(owner.project_id)
        relative = Path("reports") / f"{report.report_id}.md"
        target = Path(project.root_path) / relative
        if not target.exists():
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(render_markdown(report), encoding="utf-8")
            temporary.replace(target)
        now = utc_now()
        status = "qualified" if report.qualified else "draft"
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_engineering_reports VALUES(?,?,?,?,?,?,?,?)",
                (report.report_id, investigation_id, owner.project_id,
                 report.content_hash, report.to_json(), relative.as_posix(), status, now),
            )
            connection.execute(
                "UPDATE app_investigations SET status=?,updated_at=? WHERE investigation_id=?",
                (InvestigationStatus.COMPLETED.value, now, investigation_id),
            )
        return self.report(report.report_id)

    def report(self, report_id: str) -> EngineeringReportRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_engineering_reports WHERE report_id=?", (report_id,),
        )
        if row is None:
            raise KeyError(f"unknown engineering report: {report_id}")
        return _report(row)

    def reports(self, project_id: str) -> tuple[EngineeringReportRecord, ...]:
        self.projects.get(project_id)
        return tuple(_report(row) for row in self.database.fetch_all(
            "SELECT * FROM app_engineering_reports WHERE project_id=? ORDER BY created_at",
            (project_id,),
        ))

    def propose_variant(
        self, finding_id: str, *, name: str, proposal_kind: str, proposal,
    ) -> VariantProposalRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_investigation_findings WHERE finding_id=?",
            (finding_id,),
        )
        if row is None:
            raise KeyError(f"unknown finding: {finding_id}")
        finding = _finding(row)
        if finding.review_status is not FindingReviewStatus.ACCEPTED:
            raise ValueError("variant proposals require a human-accepted finding")
        finding_value = EngineeringFinding.from_dict(finding.finding)
        if finding_value.recommendation is None:
            raise ValueError("finding has no engineering recommendation to propose")
        name = str(name).strip()
        if not name:
            raise ValueError("variant proposal name cannot be empty")
        proposal_kind = str(proposal_kind).strip()
        allowed = {"input_patch", "run_configuration", "checkpoint_branch", "experiment"}
        if proposal_kind not in allowed:
            raise ValueError(f"unsupported variant proposal kind: {proposal_kind}")
        value = json_value(dict(proposal or {}))
        if not value:
            raise ValueError("variant proposal payload cannot be empty")

        def keys(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    yield str(key).lower()
                    yield from keys(child)
            elif isinstance(item, list):
                for child in item:
                    yield from keys(child)
        forbidden = set(keys(value)) & {
            "password", "token", "secret", "private_key", "api_key",
        }
        if forbidden:
            raise ValueError("variant proposals cannot persist credentials")
        canonical = {
            "finding_id": finding_id, "name": name,
            "proposal_kind": proposal_kind, "proposal": value,
            "recommendation": finding_value.recommendation.to_dict(),
        }
        digest, now = _digest(canonical), utc_now()
        existing = self.database.fetch_one(
            "SELECT * FROM app_variant_proposals WHERE investigation_id=? AND content_hash=?",
            (finding.investigation_id, digest),
        )
        if existing is not None:
            return _proposal(existing)
        record = VariantProposalRecord(
            f"proposal_{uuid4().hex}", finding.investigation_id, finding_id,
            finding.project_id, name, proposal_kind, digest, canonical,
            VariantProposalStatus.PROPOSED, "", now, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_variant_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.proposal_id, record.investigation_id, finding_id,
                 record.project_id, name, proposal_kind, digest, _encoded(canonical),
                 record.status.value, "", now, now),
            )
        return record

    def proposals(self, investigation_id: str) -> tuple[VariantProposalRecord, ...]:
        self.get(investigation_id)
        return tuple(_proposal(row) for row in self.database.fetch_all(
            "SELECT * FROM app_variant_proposals WHERE investigation_id=? ORDER BY created_at",
            (investigation_id,),
        ))

    def decide_proposal(self, proposal_id: str, status, note: str) -> VariantProposalRecord:
        status = VariantProposalStatus(status)
        if status is VariantProposalStatus.PROPOSED:
            raise ValueError("proposal decisions must approve or reject")
        row = self.database.fetch_one(
            "SELECT * FROM app_variant_proposals WHERE proposal_id=?", (proposal_id,),
        )
        if row is None:
            raise KeyError(f"unknown variant proposal: {proposal_id}")
        current = _proposal(row)
        if current.status is not VariantProposalStatus.PROPOSED:
            if current.status is status:
                return current
            raise ValueError("a decided variant proposal is immutable")
        note, now = str(note).strip(), utc_now()
        if not note:
            raise ValueError("proposal decisions require a review note")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_variant_proposals SET status=?,decision_note=?,updated_at=? WHERE proposal_id=?",
                (status.value, note, now, proposal_id),
            )
        return _proposal(self.database.fetch_one(
            "SELECT * FROM app_variant_proposals WHERE proposal_id=?", (proposal_id,),
        ))


__all__ = ["AgentInvestigationService"]
