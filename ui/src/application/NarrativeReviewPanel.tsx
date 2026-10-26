import { useEffect, useState } from "react";
import {
  api, type InvestigationNarrativeRecord, type ReasoningProviderRecord,
} from "../api/client";

export function NarrativeReviewPanel({
  investigationId, providers, acceptedCount, unresolvedConflicts,
}: {
  investigationId: string;
  providers: ReasoningProviderRecord[];
  acceptedCount: number;
  unresolvedConflicts: number;
}) {
  const [items, setItems] = useState<InvestigationNarrativeRecord[]>([]);
  const [provider, setProvider] = useState("deterministic-v1");
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const latest = items.at(-1);

  useEffect(() => {
    const controller = new AbortController();
    setItems([]); setError(""); setMessage("");
    api.investigationNarratives(investigationId, controller.signal)
      .then(setItems)
      .catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, [investigationId]);

  async function draft() {
    setBusy(true); setError(""); setMessage("");
    try {
      const item = await api.draftInvestigationNarrative(investigationId, provider);
      setItems((current) => current.some((value) => value.narrative_id === item.narrative_id)
        ? current : [...current, item]);
      setMessage("Evidence-bound draft created. It is not publishable until human review.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Narrative synthesis failed");
    } finally { setBusy(false); }
  }

  async function review(status: "approved" | "rejected") {
    if (!latest || !reviewer.trim() || !note.trim()) {
      setError("Narrative review requires a named reviewer and rationale."); return;
    }
    setBusy(true); setError(""); setMessage("");
    try {
      const updated = await api.reviewInvestigationNarrative(
        latest.narrative_id, status, reviewer, note,
      );
      setItems((current) => current.map((item) =>
        item.narrative_id === updated.narrative_id ? updated : item));
      setMessage(status === "approved"
        ? "Narrative approved for inclusion in a matching report."
        : "Narrative rejected and retained in the audit history.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Narrative review failed");
    } finally { setBusy(false); }
  }

  return <div className="narrative-review">
    <div className="ledger-head"><div><span className="eyebrow">Evidence-bound answer</span>
      <h3>Provider-assisted narrative</h3></div><span>{latest?.status ?? "Not drafted"}</span></div>
    <p className="narrative-policy">The provider sees accepted finding summaries only. PyMRST binds citations and uncertainty notes; a human controls publication.</p>
    <div className="narrative-controls">
      <select id="narrative-provider" name="narrative-provider" aria-label="Narrative provider" value={provider} onChange={(event) => setProvider(event.target.value)}>
        <option value="deterministic-v1">Deterministic offline narrative</option>
        {providers.map((item) => <option key={item.provider_id} value={item.provider_id}>
          {item.provider_id} · {item.model}
        </option>)}
      </select>
      <button className="button ghost" disabled={busy || acceptedCount === 0 || unresolvedConflicts > 0} onClick={draft}>Draft answer</button>
    </div>
    {unresolvedConflicts > 0 && <p className="narrative-warning">Resolve accepted-finding conflicts before drafting narrative prose.</p>}
    {latest && <article className={`narrative-card ${latest.status}`}>
      <header><div><b>{latest.status}</b><h4>{latest.narrative.title}</h4></div><code>{latest.content_hash.slice(0, 16)}</code></header>
      <p>{latest.narrative.executive_summary}</p>
      {latest.narrative.sections.map((section) => <details key={section.section_id} open>
        <summary>{section.heading} · {section.finding_ids.length} finding(s)</summary>
        <p>{section.narrative}</p>
        {section.uncertainty_caveats.map((caveat) => <small key={caveat}>Uncertainty: {caveat}</small>)}
        <ul>{section.evidence.map((evidence) => <li key={`${evidence.uri}-${evidence.label}`}>
          <strong>{evidence.label}</strong><code>{evidence.uri}</code>
        </li>)}</ul>
      </details>)}
      {latest.status === "draft" && <div className="narrative-decision">
        <input id="narrative-reviewer" name="narrative-reviewer" aria-label="Reviewer name or email" placeholder="Reviewer name or email" value={reviewer} onChange={(event) => setReviewer(event.target.value)} />
        <textarea id="narrative-review-note" name="narrative-review-note" aria-label="Review rationale" placeholder="Review rationale" value={note} onChange={(event) => setNote(event.target.value)} />
        <button disabled={busy} onClick={() => review("approved")}>Approve narrative</button>
        <button disabled={busy} onClick={() => review("rejected")}>Reject</button>
      </div>}
      {latest.status !== "draft" && <footer>{latest.reviewer}: {latest.review_note}</footer>}
    </article>}
    {message && <div className="agent-notice">{message}</div>}{error && <div className="form-error">{error}</div>}
  </div>;
}
