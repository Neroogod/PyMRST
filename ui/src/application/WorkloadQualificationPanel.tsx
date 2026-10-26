import { useEffect, useMemo, useState } from "react";
import {
  api, type WorkloadBudget, type WorkloadDefinition,
  type WorkloadQualification, type WorkloadTier,
} from "../api/client";

const terminal = new Set(["passed", "failed", "cancelled", "interrupted"]);
const bytes = (value: number) => value >= 1024 ** 3
  ? `${(value / 1024 ** 3).toFixed(2)} GiB` : `${(value / 1024 ** 2).toFixed(1)} MiB`;
const seconds = (value: number) => `${value.toFixed(value < 10 ? 3 : 1)} s`;

export function WorkloadQualificationPanel() {
  const [catalog, setCatalog] = useState<WorkloadDefinition[]>([]);
  const [records, setRecords] = useState<WorkloadQualification[]>([]);
  const [workloadId, setWorkloadId] = useState("numerical-core");
  const [tier, setTier] = useState<WorkloadTier>("smoke");
  const [seed, setSeed] = useState(2026);
  const [budget, setBudget] = useState<WorkloadBudget | null>(null);
  const [busy, setBusy] = useState(""); const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const definition = catalog.find((item) => item.workload_id === workloadId);
  const active = records.find((item) => !terminal.has(item.status));
  const latest = records[0];
  const selectedBudget = budget ?? definition?.default_budgets[tier] ?? null;
  const summary = useMemo(() => ({
    passed: records.filter((item) => item.status === "passed").length,
    failed: records.filter((item) => item.status === "failed").length,
  }), [records]);

  async function reload(signal?: AbortSignal) {
    const [definitions, history] = await Promise.all([
      api.workloadCatalog(signal), api.workloadQualifications(30, signal),
    ]);
    setCatalog(definitions); setRecords(history);
  }

  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal).catch((caught) => {
      if (caught.name !== "AbortError") setError(caught.message);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!active) return;
    let current = true;
    const poll = window.setInterval(() => {
      api.workloadQualification(active.qualification_id).then((record) => {
        if (!current) return;
        setRecords((values) => [record, ...values.filter((item) => item.qualification_id !== record.qualification_id)]);
      }).catch((caught) => current && setError(caught.message));
    }, 900);
    return () => { current = false; window.clearInterval(poll); };
  }, [active?.qualification_id]);

  function selectWorkload(value: string) {
    setWorkloadId(value); setBudget(null);
    const next = catalog.find((item) => item.workload_id === value);
    if (next && !next.tiers.includes(tier)) setTier(next.tiers[0]);
  }

  function setBudgetValue(name: keyof WorkloadBudget, value: number) {
    if (!selectedBudget) return;
    setBudget({ ...selectedBudget, [name]: value });
  }

  async function submit() {
    if (!selectedBudget) return;
    setBusy("submit"); setError(""); setNotice("");
    try {
      const record = await api.submitWorkloadQualification({
        workload_id: workloadId, tier, seed, budget: selectedBudget,
        warmup_runs: 1, measured_runs: 3,
      });
      setRecords((values) => [record, ...values]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Qualification submission failed"); }
    finally { setBusy(""); }
  }

  async function cancel(record: WorkloadQualification) {
    setBusy(`cancel:${record.qualification_id}`); setError("");
    try {
      const value = await api.cancelWorkloadQualification(record.qualification_id);
      setRecords((items) => [value, ...items.filter((item) => item.qualification_id !== value.qualification_id)]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed"); }
    finally { setBusy(""); }
  }

  async function verify(record: WorkloadQualification) {
    setBusy(`verify:${record.qualification_id}`); setError(""); setNotice("");
    try {
      const result = await api.verifyWorkloadQualification(record.qualification_id);
      if (!result.valid) throw new Error(result.reason);
      setNotice(`Evidence verified: ${result.content_hash.slice(0, 20)}…`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Verification failed"); }
    finally { setBusy(""); }
  }

  return <section className="workload-settings">
    <div className="settings-intro"><span className="eyebrow">Performance acceptance evidence</span><h2>Installation workloads</h2><p>Run deterministic, process-isolated workloads before committing this machine to larger studies. Absolute runtime, CPU, memory and I/O limits are checked together with regression ratios against the most recent passing run on this workspace.</p></div>
    <div className="workload-overview"><div><strong>{catalog.length}</strong><small>qualified workload families</small></div><div><strong>{records.length}</strong><small>retained executions</small></div><div className="passed"><strong>{summary.passed}</strong><small>passed</small></div><div className={summary.failed ? "failed" : ""}><strong>{summary.failed}</strong><small>failed or over budget</small></div>{latest && <div className={`latest ${latest.status}`}><strong>{latest.status.replaceAll("_", " ")}</strong><small>{latest.workload_id} · {latest.tier}</small></div>}</div>
    <div className="workload-layout"><form onSubmit={(event) => { event.preventDefault(); submit(); }}><span className="eyebrow">New acceptance run</span><label htmlFor="workload-id">Workload<select id="workload-id" name="workload-id" value={workloadId} onChange={(event) => selectWorkload(event.target.value)}>{catalog.map((item) => <option key={item.workload_id} value={item.workload_id}>{item.title}{item.available ? "" : " — unavailable"}</option>)}</select></label><p>{definition?.description}</p>{definition && !definition.available && <p className="provider-check-error">{definition.unavailable_reason}</p>}{definition?.requirements.length ? <p><strong>Requires:</strong> {definition.requirements.join(" · ")}</p> : null}<label htmlFor="workload-tier">Scale tier<select id="workload-tier" name="workload-tier" value={tier} onChange={(event) => { setTier(event.target.value as WorkloadTier); setBudget(null); }}>{definition?.tiers.map((item) => <option value={item} key={item}>{item.replaceAll("_", " ")}</option>)}</select></label><label htmlFor="workload-seed">Deterministic seed<input id="workload-seed" name="workload-seed" type="number" min="0" max="4294967295" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>{selectedBudget && <div className="budget-grid"><label htmlFor="workload-budget-wall">Wall limit, s<input id="workload-budget-wall" name="workload-budget-wall" type="number" min="0.1" step="0.1" value={selectedBudget.maximum_wall_seconds} onChange={(event) => setBudgetValue("maximum_wall_seconds", Number(event.target.value))} /></label><label htmlFor="workload-budget-cpu">CPU limit, s<input id="workload-budget-cpu" name="workload-budget-cpu" type="number" min="0.1" step="0.1" value={selectedBudget.maximum_cpu_seconds} onChange={(event) => setBudgetValue("maximum_cpu_seconds", Number(event.target.value))} /></label><label htmlFor="workload-budget-memory">Memory limit, MiB<input id="workload-budget-memory" name="workload-budget-memory" type="number" min="1" value={Math.round(selectedBudget.maximum_peak_rss_bytes / 1024 ** 2)} onChange={(event) => setBudgetValue("maximum_peak_rss_bytes", Number(event.target.value) * 1024 ** 2)} /></label><label htmlFor="workload-budget-io">I/O limit, MiB<input id="workload-budget-io" name="workload-budget-io" type="number" min="1" value={Math.round(selectedBudget.maximum_disk_bytes / 1024 ** 2)} onChange={(event) => setBudgetValue("maximum_disk_bytes", Number(event.target.value) * 1024 ** 2)} /></label><label htmlFor="workload-budget-wall-regression">Wall regression<input id="workload-budget-wall-regression" name="workload-budget-wall-regression" type="number" min="1" step="0.01" value={selectedBudget.maximum_wall_regression_ratio} onChange={(event) => setBudgetValue("maximum_wall_regression_ratio", Number(event.target.value))} /></label><label htmlFor="workload-budget-memory-regression">Memory regression<input id="workload-budget-memory-regression" name="workload-budget-memory-regression" type="number" min="1" step="0.01" value={selectedBudget.maximum_memory_regression_ratio} onChange={(event) => setBudgetValue("maximum_memory_regression_ratio", Number(event.target.value))} /></label></div>}<button className="button primary" disabled={Boolean(active) || busy === "submit" || definition?.available === false}>{busy === "submit" ? "Submitting…" : active ? "Qualification already running" : definition?.available === false ? "Dataset unavailable" : "Run isolated qualification"}</button></form><div className="workload-history"><div><span className="eyebrow">Retained evidence</span><h3>Qualification history</h3></div>{records.map((record) => <article key={record.qualification_id}><div className="workload-record-head"><span className={record.status} /><div><strong>{record.workload_id} · {record.tier.replaceAll("_", " ")}</strong><small>{new Date(record.created_at).toLocaleString()} · seed {String(record.request.seed)}</small></div><b>{record.status}</b></div>{!terminal.has(record.status) && <><div className="workload-progress"><span style={{ width: `${record.progress * 100}%` }} /></div><p>{record.message}</p><button disabled={Boolean(busy)} onClick={() => cancel(record)}>{busy === `cancel:${record.qualification_id}` ? "Cancelling…" : "Cancel safely"}</button></>}{record.metrics.wall_seconds !== undefined && <><dl><dt>Wall time</dt><dd>{seconds(Number(record.metrics.wall_seconds))}</dd><dt>CPU time</dt><dd>{seconds(Number(record.metrics.cpu_seconds))}</dd><dt>Peak memory</dt><dd>{bytes(Number(record.metrics.peak_rss_bytes))}</dd>{record.metrics.report_steps !== undefined && <><dt>Report steps</dt><dd>{Number(record.metrics.report_steps).toLocaleString()}</dd><dt>Newton iterations</dt><dd>{Number(record.metrics.newton_iterations).toLocaleString()}</dd></>}<dt>Operations</dt><dd>{Number(record.metrics.operations).toLocaleString()}</dd><dt>Evidence</dt><dd>{record.content_hash ? `${record.content_hash.slice(0, 16)}…` : "—"}</dd><dt>Baseline</dt><dd>{record.comparison.baseline_qualification_id?.slice(0, 20) || "First accepted run"}</dd></dl>{record.comparison.correctness && <p>{Object.values(record.comparison.correctness).filter(Boolean).length}/{Object.keys(record.comparison.correctness).length} correctness invariants passed</p>}{record.content_hash && <button disabled={Boolean(busy)} onClick={() => verify(record)}>{busy === `verify:${record.qualification_id}` ? "Verifying…" : "Verify evidence"}</button>}</>}{record.error && <p className="provider-check-error">{String(record.error.message ?? "Qualification failed")}</p>}</article>)}{!records.length && <div className="empty-research">No workload qualification evidence has been recorded.</div>}</div></div>
    {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
  </section>;
}
