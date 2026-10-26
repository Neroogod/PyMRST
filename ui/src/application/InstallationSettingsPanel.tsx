import { useEffect, useMemo, useState } from "react";
import { api, type EnvironmentQualification } from "../api/client";

export function InstallationSettingsPanel() {
  const [record, setRecord] = useState<EnvironmentQualification | null>(null);
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const grouped = useMemo(() => Object.entries((record?.checks ?? []).reduce(
    (result, item) => ({ ...result, [item.category]: [...(result[item.category] ?? []), item] }),
    {} as Record<string, EnvironmentQualification["checks"]>,
  )), [record]);

  useEffect(() => {
    const controller = new AbortController();
    api.environmentQualification(controller.signal).then(setRecord).catch((caught) => {
      if (caught.name !== "AbortError") setError(caught.message);
    });
    return () => controller.abort();
  }, []);

  async function qualify() {
    setBusy(true); setError("");
    try { setRecord(await api.runEnvironmentQualification()); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Environment qualification failed"); }
    finally { setBusy(false); }
  }

  return <section className="installation-settings">
    <div className="settings-intro"><span className="eyebrow">Installed runtime</span><h2>Environment qualification</h2><p>Verify that this installation can safely persist projects and launch isolated workers before committing a field-scale run. Optional research capabilities are reported separately from the required numerical runtime.</p></div>
    <div className="qualification-head"><div className={`qualification-seal ${record?.status ?? "unknown"}`}><span /> <div><strong>{record ? record.status : "Not qualified"}</strong><small>{record ? `${new Date(record.created_at).toLocaleString()} · ${record.content_hash.slice(0, 14)}…` : "Run the first-launch checks"}</small></div></div><button className="button primary" disabled={busy} onClick={qualify}>{busy ? "Qualifying…" : record ? "Run qualification again" : "Qualify installation"}</button></div>
    {record && <><div className="platform-strip"><span>{record.platform.system} {record.platform.release}</span><span>{record.platform.machine}</span><span>Python {record.platform.python}</span><span>PyMRST {record.platform.pymrst}</span><b>{record.summary.passed} passed · {record.summary.warning} warnings · {record.summary.failed} failed</b></div><div className="qualification-groups">{grouped.map(([category, checks]) => <article key={category}><h3>{category.replaceAll("_", " ")}</h3>{checks.map((check) => <div className="qualification-check" key={check.check_id}><span className={check.status} /><div><strong>{check.title}{check.required && <em>required</em>}</strong><p>{check.message}</p>{check.remediation && <small>{check.remediation}</small>}</div></div>)}</article>)}</div></>}
    {!record && !error && <div className="empty-research">No qualification evidence exists for this installation yet.</div>}
    {error && <div className="form-error">{error}</div>}
  </section>;
}
