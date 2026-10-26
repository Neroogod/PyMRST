import { FormEvent, useEffect, useState } from "react";
import {
  api, type EngineeringReportRecord, type FindingRecord,
  type InvestigationRecord, type ProjectRecord, type RunRecord,
  type VariantProposalRecord,
  type InvestigationPlanRecord,
  type InvestigationExecutionRecord,
  type InvestigationSynthesisRecord,
  type ReasoningProviderRecord,
} from "../api/client";
import { NarrativeReviewPanel } from "./NarrativeReviewPanel";

type Evidence = { kind: string; uri: string; label: string; locator: Record<string, unknown>; checksum?: string };

export function AssistantWorkspace({ project, reportsOnly = false }: { project: ProjectRecord; reportsOnly?: boolean }) {
  const [items, setItems] = useState<InvestigationRecord[]>([]);
  const [reports, setReports] = useState<EngineeringReportRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [findings, setFindings] = useState<FindingRecord[]>([]);
  const [proposals, setProposals] = useState<VariantProposalRecord[]>([]);
  const [plans, setPlans] = useState<InvestigationPlanRecord[]>([]);
  const [executions, setExecutions] = useState<InvestigationExecutionRecord[]>([]);
  const [syntheses, setSyntheses] = useState<InvestigationSynthesisRecord[]>([]);
  const [providers, setProviders] = useState<ReasoningProviderRecord[]>([]);
  const [planningProvider, setPlanningProvider] = useState("deterministic-v1");
  const [question, setQuestion] = useState("Why did this simulation converge differently from the reference?");
  const [runId, setRunId] = useState("");
  const [compareRunId, setCompareRunId] = useState("");
  const [reportTitle, setReportTitle] = useState("Reservoir engineering investigation");
  const [reviewer, setReviewer] = useState("");
  const [resolutionRationale, setResolutionRationale] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const selected = items.find((item) => item.investigation_id === selectedId) ?? items.at(-1);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.investigations(project.project_id, controller.signal),
      api.engineeringReports(project.project_id, controller.signal),
      api.runs(project.project_id, controller.signal),
      api.reasoningProviders(controller.signal),
    ]).then(([history, documents, runValues, providerValues]) => {
      setItems(history); setReports(documents); setRuns(runValues);
      setProviders(providerValues.filter((item) => item.enabled));
      setSelectedId(history.at(-1)?.investigation_id ?? "");
      setRunId(runValues.at(-1)?.run_id ?? "");
    }).catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, [project.project_id]);

  useEffect(() => {
    setFindings([]); setProposals([]); setPlans([]); setExecutions([]); setSyntheses([]);
    if (!selected) return;
    const controller = new AbortController();
    Promise.all([api.findings(selected.investigation_id, controller.signal), api.variantProposals(selected.investigation_id, controller.signal), api.investigationPlans(selected.investigation_id, controller.signal), api.investigationExecutions(selected.investigation_id, controller.signal), api.investigationSyntheses(selected.investigation_id, controller.signal)]).then(([findingValues, proposalValues, planValues, executionValues, synthesisValues]) => { setFindings(findingValues); setProposals(proposalValues); setPlans(planValues); setExecutions(executionValues); setSyntheses(synthesisValues); })
      .catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, [selected?.investigation_id]);

  async function create(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const item = await api.createInvestigation(project.project_id, {
        question, run_ids: [runId, compareRunId].filter(Boolean), selection: {},
        attributes: { workspace: "assistant", authority: "analyze_only", comparison_tolerances: {
          pressure: { absolute: 100, relative: 1e-4 },
          sW: { absolute: 1e-6, relative: 1e-5 },
          sO: { absolute: 1e-6, relative: 1e-5 },
          sG: { absolute: 1e-6, relative: 1e-5 },
        } },
      });
      setItems((current) => current.some((value) => value.investigation_id === item.investigation_id) ? current : [...current, item]);
      setSelectedId(item.investigation_id); setNotice("Investigation context sealed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not create investigation"); }
    finally { setBusy(false); }
  }

  async function analyze() {
    if (!selected) return; setBusy(true); setError(""); setNotice("");
    try {
      setFindings(await api.analyzeInvestigation(selected.investigation_id, plans.at(-1)?.plan_id));
      setExecutions(await api.investigationExecutions(selected.investigation_id));
      setSyntheses(await api.investigationSyntheses(selected.investigation_id));
      setNotice("The persisted read-only plan completed and its execution ledger was sealed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Analysis failed"); }
    finally { setBusy(false); }
  }

  async function resolveConflict(conflictId: string, selectedFindingIds?: string[]) {
    if (!selected || !reviewer.trim() || !resolutionRationale.trim()) {
      setError("Conflict decisions require a named reviewer and rationale."); return;
    }
    setBusy(true); setError("");
    try {
      const item = await api.resolveInvestigationConflict(
        selected.investigation_id, conflictId,
        { action: selectedFindingIds ? "resolve" : "dismiss", reviewer, rationale: resolutionRationale, selected_finding_ids: selectedFindingIds },
      );
      setSyntheses((current) => [...current, item]);
      setNotice("Immutable human conflict decision recorded and synthesis recomputed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Conflict resolution failed"); }
    finally { setBusy(false); }
  }

  async function createPlan() {
    if (!selected) return; setBusy(true); setError("");
    try {
      const item = await api.planInvestigation(selected.investigation_id, planningProvider);
      setPlans((current) => current.some((value) => value.plan_id === item.plan_id) ? current : [...current, item]);
      setNotice("Bounded provider-independent tool plan recorded. No tools executed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Planning failed"); }
    finally { setBusy(false); }
  }

  async function review(item: FindingRecord, status: "accepted" | "rejected") {
    setBusy(true); setError("");
    try {
      const updated = await api.reviewFinding(item.finding_id, status, `Reviewed in PyMRST Studio: ${status}`);
      setFindings((current) => current.map((value) => value.finding_id === updated.finding_id ? updated : value));
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Review failed"); }
    finally { setBusy(false); }
  }

  async function buildReport() {
    if (!selected) return; setBusy(true); setError("");
    try {
      const report = await api.buildEngineeringReport(selected.investigation_id, reportTitle, true);
      setReports((current) => current.some((item) => item.report_id === report.report_id) ? current : [...current, report]);
      setItems((current) => current.map((item) => item.investigation_id === selected.investigation_id ? { ...item, status: "completed" } : item));
      setNotice(`Qualified report written to ${report.relative_path}.`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Report generation failed"); }
    finally { setBusy(false); }
  }

  async function propose(item: FindingRecord) {
    const recommendation = item.finding.recommendation as { statement?: string; rationale?: string } | null;
    if (!recommendation) return; setBusy(true); setError("");
    try {
      const proposal = await api.createVariantProposal(item.finding_id, {
        name: recommendation.statement ?? "Diagnostic variant",
        proposal_kind: "run_configuration",
        proposal: {
          purpose: "bounded_diagnostic", source_finding_id: item.finding_id,
          rationale: recommendation.rationale ?? "Evidence-backed specialist recommendation",
          execution_authorized: false,
        },
      });
      setProposals((current) => current.some((value) => value.proposal_id === proposal.proposal_id) ? current : [...current, proposal]);
      setNotice("Immutable proposal created. No model change or run was started.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Proposal failed"); }
    finally { setBusy(false); }
  }

  async function decideProposal(item: VariantProposalRecord, status: "approved" | "rejected") {
    setBusy(true); setError("");
    try {
      const updated = await api.decideVariantProposal(item.proposal_id, status, `${status} in PyMRST Studio after engineering review`);
      setProposals((current) => current.map((value) => value.proposal_id === updated.proposal_id ? updated : value));
      setNotice(status === "approved" ? "Proposal approved. Execution remains a separate controlled action." : "Proposal rejected and retained for audit.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Proposal decision failed"); }
    finally { setBusy(false); }
  }

  if (reportsOnly) return <section className="agent-room report-library">
    <div className="agent-head"><span className="eyebrow">Engineering record</span><h2>Qualified reports</h2><p>Immutable documents assembled only from human-accepted, evidence-linked findings.</p></div>
    <div className="report-cards">{reports.map((item) => <article key={item.report_id}><b>{item.status}</b><h3>{String(item.report.title ?? "Engineering report")}</h3><p>{String(item.report.executive_summary ?? "")}</p><code>{item.content_hash.slice(0, 16)}</code><span>{item.relative_path}</span></article>)}{!reports.length && <div className="empty-research">No engineering reports have been approved yet.</div>}</div>
  </section>;

  const boundRuns = selected?.context.run_ids ?? [];
  const accepted = findings.filter((item) => item.review_status === "accepted").length;
  const synthesis = syntheses.at(-1);
  return <section className="agent-room">
    <aside className="investigation-history"><span className="eyebrow">Investigation history</span><h3>Engineering questions</h3>{items.map((item) => <button className={selected?.investigation_id === item.investigation_id ? "active" : ""} onClick={() => setSelectedId(item.investigation_id)} key={item.investigation_id}><strong>{item.question}</strong><span>{item.status} · {new Date(item.created_at).toLocaleString()}</span></button>)}</aside>
    <main className="assistant-main">
      <div className="agent-head"><span className="eyebrow">Evidence-controlled assistant</span><h2>Reservoir engineering investigation</h2><p>The assistant receives explicit case/run context. Analysis is deterministic and read-only; findings require your review before publication.</p></div>
      <form className="assistant-question" onSubmit={create}><textarea id="assistant-question" name="assistant-question" aria-label="Investigation question" required value={question} onChange={(event) => setQuestion(event.target.value)} /><select id="assistant-run" name="assistant-run" aria-label="Reference run" value={runId} onChange={(event) => setRunId(event.target.value)}><option value="">Project inputs only</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>Reference {run.run_id.slice(-8)} · {run.status}</option>)}</select><select id="assistant-compare-run" name="assistant-compare-run" aria-label="Candidate run" value={compareRunId} onChange={(event) => setCompareRunId(event.target.value)} disabled={!runId}><option value="">No candidate comparison</option>{runs.filter((run) => run.run_id !== runId).map((run) => <option key={run.run_id} value={run.run_id}>Candidate {run.run_id.slice(-8)} · {run.status}</option>)}</select><button className="button primary" disabled={busy}>Seal investigation</button></form>
      {selected && <div className="planning-provider-bar"><div><span className="eyebrow">Planning provider</span><p>The provider proposes tool calls only; PyMRST validates and executes them separately.</p></div><select id="assistant-planning-provider" name="assistant-planning-provider" aria-label="Planning provider" value={planningProvider} onChange={(event) => setPlanningProvider(event.target.value)}><option value="deterministic-v1">Deterministic offline planner</option>{providers.map((item) => <option key={item.provider_id} value={item.provider_id}>{item.provider_id} · {item.model}</option>)}</select></div>}
      {selected && <div className="investigation-detail">
        <div className="investigation-context"><span className="eyebrow">Current question</span><h3>{selected.question}</h3><code>{selected.content_hash}</code><dl><dt>Authority</dt><dd>Read and analyze only</dd><dt>Bound runs</dt><dd>{boundRuns.length || "Project inputs"}</dd><dt>Status</dt><dd>{selected.status}</dd></dl><button className="button ghost" disabled={busy || selected.status !== "open"} onClick={createPlan}>Plan investigation</button><button className="button primary" disabled={busy || selected.status !== "open"} onClick={analyze}>Execute read-only analysis</button></div>
        <div className="plan-ledger"><div className="ledger-head"><div><span className="eyebrow">Provider-independent plan</span><h3>Bounded tool graph</h3></div><span>{executions.at(-1)?.status ?? plans.at(-1)?.provider ?? "Not planned"}</span></div>{plans.at(-1)?.plan.plan?.calls?.map((call, index) => { const completed = executions.at(-1)?.execution.calls?.find((item) => item.call.call_id === call.call_id); return <article key={call.call_id}><b className={completed?.result.ok ? "completed" : ""}>{completed?.result.ok ? "✓" : index + 1}</b><strong>{call.tool.replaceAll("_", " ")}</strong><code>{JSON.stringify(call.arguments)}</code></article>; }) ?? <p>Create a plan to inspect the proposed deterministic tool graph before execution.</p>}</div>
        <div className="synthesis-ledger"><div className="ledger-head"><div><span className="eyebrow">Cross-specialist synthesis</span><h3>Conflict and uncertainty review</h3></div><span>{synthesis ? `${Number(synthesis.synthesis.summary.unresolved_conflict_count ?? 0)} unresolved` : "Not synthesized"}</span></div>{synthesis && <div className="synthesis-metrics"><span><b>{Number(synthesis.synthesis.summary.agent_count ?? 0)}</b> specialists</span><span><b>{Number(synthesis.synthesis.summary.evidence_reference_count ?? 0)}</b> evidence links</span><span><b>{synthesis.synthesis.uncertainties.length}</b> uncertainty notes</span><span><b>{synthesis.synthesis.corroborations.length}</b> corroborations</span></div>}{synthesis?.synthesis.conflicts.some((item) => !item.resolution) && <div className="resolution-identity"><input id="conflict-reviewer" name="conflict-reviewer" placeholder="Reviewer name or email" value={reviewer} onChange={(event) => setReviewer(event.target.value)} /><textarea id="conflict-rationale" name="conflict-rationale" placeholder="Engineering rationale for this decision" value={resolutionRationale} onChange={(event) => setResolutionRationale(event.target.value)} /></div>}{synthesis?.synthesis.conflicts.map((conflict) => <article className={conflict.resolution ? "resolved" : "unresolved"} key={conflict.conflict_id}><div><b>{conflict.resolution ? "Resolved" : "Needs review"}</b><strong>{conflict.category}</strong><p>{conflict.reason}</p><small>{conflict.shared_objects.join(" · ")}</small></div>{conflict.resolution ? <p>{conflict.resolution.reviewer}: {conflict.resolution.rationale}</p> : <div className="conflict-actions">{conflict.finding_ids.map((findingId) => <button disabled={busy} key={findingId} onClick={() => resolveConflict(conflict.conflict_id, [findingId])}>Use {findings.find((item) => item.finding_id === findingId)?.finding.agent ?? findingId}</button>)}<button disabled={busy} onClick={() => resolveConflict(conflict.conflict_id)}>Dismiss conflict</button></div>}</article>)}{synthesis?.synthesis.uncertainties.map((item) => <details key={item.finding_id}><summary>{item.uncertainty_kind} uncertainty · confidence {item.confidence.toFixed(3)}</summary><p>{item.explanation}</p></details>)}{!synthesis && <p>Execute the bounded plan to generate deterministic synthesis.</p>}</div>
        <div className="evidence-ledger"><div className="ledger-head"><div><span className="eyebrow">Claim-to-evidence ledger</span><h3>Findings and human review</h3></div><span>{accepted}/{findings.length} accepted</span></div>{findings.map((item) => { const evidence = (item.finding.evidence ?? []) as Evidence[]; const hasRecommendation = Boolean(item.finding.recommendation); return <article key={item.finding_id}><b className={item.review_status}>{item.review_status}</b><div><strong>{String(item.finding.statement ?? item.finding_id)}</strong><small>{String(item.finding.claim_kind)} · {String(item.finding.severity)} · confidence {Number(item.finding.confidence).toFixed(2)}</small>{evidence.map((source) => <details key={source.uri}><summary>{source.label}</summary><code>{source.uri}</code><span>SHA-256 {source.checksum?.slice(0, 16) ?? "not supplied"}…</span></details>)}</div><div className="finding-actions"><button disabled={busy || item.review_status === "accepted"} onClick={() => review(item, "accepted")}>Accept</button><button disabled={busy || item.review_status === "rejected"} onClick={() => review(item, "rejected")}>Reject</button>{hasRecommendation && <button disabled={busy || item.review_status !== "accepted"} onClick={() => propose(item)}>Propose variant</button>}</div></article>; })}{!findings.length && <div className="empty-research">The investigation is registered. Run deterministic specialists to generate traceable findings.</div>}</div>
        <div className="proposal-review"><div className="ledger-head"><div><span className="eyebrow">Controlled change boundary</span><h3>Variant proposals</h3></div><span>{proposals.length}</span></div>{proposals.map((item) => <article key={item.proposal_id}><b className={item.status}>{item.status}</b><div><strong>{item.name}</strong><span>{item.proposal_kind} · {item.content_hash.slice(0, 12)}</span></div>{item.status === "proposed" && <div className="finding-actions"><button disabled={busy} onClick={() => decideProposal(item, "approved")}>Approve intent</button><button disabled={busy} onClick={() => decideProposal(item, "rejected")}>Reject</button></div>}</article>)}{!proposals.length && <p>No change proposals. Accepted findings remain analytical evidence only.</p>}</div>
        <NarrativeReviewPanel investigationId={selected.investigation_id} providers={providers} acceptedCount={accepted} unresolvedConflicts={Number(synthesis?.synthesis.summary.unresolved_conflict_count ?? 0)} />
        <div className="report-author"><span className="eyebrow">Human-controlled publication</span><h3>Build engineering report</h3><input id="report-title" name="report-title" aria-label="Report title" value={reportTitle} onChange={(event) => setReportTitle(event.target.value)} /><button className="button ghost" disabled={busy || selected.status !== "open" || accepted === 0} onClick={buildReport}>Publish accepted findings</button><p>Rejected and pending claims are excluded. Publication closes this investigation.</p></div>
      </div>}
      {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
    </main>
  </section>;
}
