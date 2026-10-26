import { useEffect, useState } from "react";
import {
  api, type BackupJobRecord, type BackupPlan, type BackupPolicy,
  type BackupQualificationRecord, type BackupRecord,
} from "../api/client";

function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = value / 1024; let unit = units[0];
  for (let index = 1; index < units.length && amount >= 1024; index += 1) {
    amount /= 1024; unit = units[index];
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
}

const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);

export function BackupSettingsPanel() {
  const [plan, setPlan] = useState<BackupPlan | null>(null);
  const [items, setItems] = useState<BackupRecord[]>([]);
  const [jobs, setJobs] = useState<BackupJobRecord[]>([]);
  const [qualifications, setQualifications] = useState<BackupQualificationRecord[]>([]);
  const [policy, setPolicy] = useState<BackupPolicy | null>(null);
  const [selected, setSelected] = useState("");
  const [activeJobId, setActiveJobId] = useState("");
  const [restoreTarget, setRestoreTarget] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const active = items.find((item) => item.backup_id === selected) ?? items.at(-1);
  const activeJob = jobs.find((item) => item.job_id === activeJobId)
    ?? [...jobs].reverse().find((item) => !terminal.has(item.status));
  const activeQualification = qualifications.filter((item) =>
    item.backup_id === active?.backup_id).at(-1);

  async function reload(signal?: AbortSignal) {
    const [nextPlan, history, jobHistory, qualificationHistory, nextPolicy] = await Promise.all([
      api.backupPlan(signal), api.backups(signal), api.backupJobs(signal),
      api.backupQualifications(undefined, signal), api.backupPolicy(signal),
    ]);
    setPlan(nextPlan); setItems(history); setJobs(jobHistory);
    setQualifications(qualificationHistory); setPolicy(nextPolicy);
    setSelected((current) => current || history.at(-1)?.backup_id || "");
    setActiveJobId([...jobHistory].reverse().find(
      (item) => !terminal.has(item.status),
    )?.job_id || "");
  }

  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal).catch((caught) => {
      if (caught.name !== "AbortError") setError(caught.message);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!activeJobId) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await api.backupJob(activeJobId);
        setJobs((current) => current.some((item) => item.job_id === job.job_id)
          ? current.map((item) => item.job_id === job.job_id ? job : item)
          : [...current, job]);
        if (terminal.has(job.status)) {
          window.clearInterval(timer); setActiveJobId("");
          await reload();
          if (job.status === "completed") {
            setSelected(job.backup_id); setNotice("Background backup completed and was verified.");
          } else if (job.status === "failed") {
            setError(String(job.error?.message ?? "Background backup failed"));
          }
        }
      } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not read backup progress"); }
    }, 500);
    return () => window.clearInterval(timer);
  }, [activeJobId]);

  async function create() {
    setBusy("create"); setError(""); setNotice("");
    try {
      const job = await api.submitBackupJob();
      setJobs((current) => [...current, job]); setActiveJobId(job.job_id);
      setNotice("Backup queued outside the interface thread. You may continue working.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Backup submission failed"); }
    finally { setBusy(""); }
  }

  async function cancel() {
    if (!activeJob) return; setBusy("cancel"); setError("");
    try {
      await api.cancelBackupJob(activeJob.job_id);
      setNotice("Cancellation requested; the current streaming block will finish safely.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed"); }
    finally { setBusy(""); }
  }

  async function verify() {
    if (!active) return; setBusy("verify"); setError(""); setNotice("");
    try {
      const item = await api.verifyBackup(active.backup_id);
      setItems((current) => current.map((value) => value.backup_id === item.backup_id ? item : value));
      setNotice("Archive, manifest, member checksums, and resource bounds passed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Verification failed"); }
    finally { setBusy(""); }
  }

  async function qualify() {
    if (!active) return; setBusy("qualify"); setError(""); setNotice("");
    try {
      const item = await api.qualifyBackup(active.backup_id);
      setQualifications((current) => [...current, item]);
      setNotice("Disposable recovery drill passed and its temporary workspace was removed.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Recovery drill failed"); }
    finally { setBusy(""); }
  }

  async function savePolicy() {
    if (!policy) return; setBusy("policy"); setError(""); setNotice("");
    try {
      setPolicy(await api.saveBackupPolicy(policy));
      setNotice(policy.enabled ? "Automatic policy enabled." : "Automatic policy disabled.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Policy update failed"); }
    finally { setBusy(""); }
  }

  async function restore() {
    if (!active) return; setBusy("restore"); setError(""); setNotice("");
    try {
      const receipt = await api.restoreBackup(active.backup_id, restoreTarget, confirmation);
      setNotice(`Recovery staged at ${receipt.target_root}; database integrity is ${receipt.database_integrity}.`);
      setRestoreTarget(""); setConfirmation("");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Restore failed"); }
    finally { setBusy(""); }
  }

  return <section className="backup-settings">
    <div className="settings-intro"><span className="eyebrow">Production recovery</span><h2>Workspace backup and recovery</h2><p>Backups run outside the interface thread, stream checksummed project files, and use a transactionally consistent metadata snapshot. Recovery is always staged into a new workspace.</p></div>
    {activeJob && !terminal.has(activeJob.status) && <div className="backup-progress"><div><strong>{activeJob.message}</strong><span>{(activeJob.progress * 100).toFixed(0)}%</span></div><progress max="1" value={activeJob.progress} /><button disabled={busy === "cancel" || activeJob.cancellation_requested} onClick={cancel}>{activeJob.cancellation_requested ? "Cancelling…" : "Cancel safely"}</button></div>}
    <div className="backup-grid">
      <article className="backup-plan"><span className="eyebrow">Dry-run inventory</span><h3>Current backup scope</h3>{plan ? <dl><dt>Projects</dt><dd>{plan.project_count}</dd><dt>Files</dt><dd>{plan.file_count.toLocaleString()}</dd><dt>Estimated size</dt><dd>{bytes(plan.estimated_bytes)}</dd><dt>Database schema</dt><dd>v{plan.database_schema_version}</dd><dt>Included roots</dt><dd>{plan.includes.join(" · ")}</dd></dl> : <p>Calculating bounded scope…</p>}<button className="button primary" disabled={Boolean(busy) || !plan || Boolean(activeJob && !terminal.has(activeJob.status))} onClick={create}>{busy === "create" ? "Queueing…" : "Create background backup"}</button>
        {policy && <div className="backup-policy"><h4>Automatic policy</h4><label htmlFor="backup-policy-enabled"><input id="backup-policy-enabled" name="backup-policy-enabled" type="checkbox" checked={policy.enabled} onChange={(event) => setPolicy({ ...policy, enabled: event.target.checked })} /> Enable scheduler</label><label htmlFor="backup-policy-interval">Interval, hours<input id="backup-policy-interval" name="backup-policy-interval" type="number" min="1" max="8760" value={policy.interval_hours} onChange={(event) => setPolicy({ ...policy, interval_hours: Number(event.target.value) })} /></label><label htmlFor="backup-policy-retained">Retain verified archives<input id="backup-policy-retained" name="backup-policy-retained" type="number" min="1" max="1000" value={policy.retained_verified_backups} onChange={(event) => setPolicy({ ...policy, retained_verified_backups: Number(event.target.value) })} /></label><label htmlFor="backup-policy-auto-verify"><input id="backup-policy-auto-verify" name="backup-policy-auto-verify" type="checkbox" checked={policy.auto_verify} onChange={(event) => setPolicy({ ...policy, auto_verify: event.target.checked })} /> Verify every backup</label><label htmlFor="backup-policy-drill"><input id="backup-policy-drill" name="backup-policy-drill" type="checkbox" checked={policy.recovery_drill} onChange={(event) => setPolicy({ ...policy, recovery_drill: event.target.checked })} /> Run disposable recovery drill</label><button disabled={Boolean(busy)} onClick={savePolicy}>Save policy</button></div>}
      </article>
      <div className="backup-history"><span className="eyebrow">Recovery history</span><h3>Managed archives</h3>{items.map((item) => <button className={active?.backup_id === item.backup_id ? "active" : ""} key={item.backup_id} onClick={() => setSelected(item.backup_id)}><span className={`backup-state ${item.pruned_at ? "pruned" : item.status}`} /><strong>{new Date(item.created_at).toLocaleString()}</strong><small>{item.pruned_at ? "pruned" : item.status} · {bytes(item.size_bytes)} · {item.file_count.toLocaleString()} files</small></button>)}{!items.length && <p>No workspace backup has been created.</p>}
      </div>
      {active && <article className="backup-detail"><header><div><span className="eyebrow">Selected archive</span><h3>{active.backup_id}</h3></div><b className={active.pruned_at ? "pruned" : active.status}>{active.pruned_at ? "pruned" : active.status}</b></header><code>{active.content_hash}</code><dl><dt>Archive</dt><dd>{active.relative_path}</dd><dt>Projects</dt><dd>{active.project_count}</dd><dt>Files</dt><dd>{active.file_count.toLocaleString()}</dd><dt>Last verified</dt><dd>{active.verified_at ? new Date(active.verified_at).toLocaleString() : "Not verified"}</dd><dt>Recovery drill</dt><dd>{activeQualification?.status ?? "Not run"}</dd></dl><div className="backup-actions"><button className="button ghost" disabled={Boolean(busy) || Boolean(active.pruned_at)} onClick={verify}>{busy === "verify" ? "Verifying…" : "Verify archive"}</button><button className="button ghost" disabled={Boolean(busy) || active.status !== "verified" || Boolean(active.pruned_at)} onClick={qualify}>{busy === "qualify" ? "Drilling…" : "Run recovery drill"}</button></div>
        {!active.pruned_at && <div className="restore-boundary"><h4>Stage disaster recovery</h4><p>The destination must be a new, nonexistent workspace. Enter the complete archive hash to authorize extraction.</p><input id="backup-restore-target" name="backup-restore-target" placeholder="Absolute destination path" value={restoreTarget} onChange={(event) => setRestoreTarget(event.target.value)} /><input id="backup-restore-confirmation" name="backup-restore-confirmation" placeholder="Confirm complete SHA-256" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /><button disabled={Boolean(busy) || active.status !== "verified" || confirmation !== active.content_hash || !restoreTarget.trim()} onClick={restore}>{busy === "restore" ? "Restoring…" : "Restore into new workspace"}</button></div>}</article>}
    </div>
    {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
  </section>;
}
