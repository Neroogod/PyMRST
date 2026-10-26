import { useEffect, useState } from "react";
import { invokeDesktop, isDesktopApplication } from "./bootstrap";

interface UpdateStatus {
  enabled: boolean;
  current_version: string;
  trust_model: string;
}

interface UpdateMetadata {
  version: string;
  current_version: string;
  notes: string | null;
  publication_date: string | null;
}

export function UpdateSettingsPanel() {
  const desktop = isDesktopApplication();
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [candidate, setCandidate] = useState<UpdateMetadata | null>(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!desktop) return;
    invokeDesktop<UpdateStatus>("desktop_update_status").then(setStatus).catch((caught) => {
      setError(caught instanceof Error ? caught.message : String(caught));
    });
  }, [desktop]);

  async function check() {
    setBusy("check"); setError(""); setNotice(""); setCandidate(null);
    try {
      const update = await invokeDesktop<UpdateMetadata | null>("check_for_update");
      setCandidate(update);
      if (!update) setNotice("This installation is current.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally { setBusy(""); }
  }

  async function install() {
    if (!candidate) return;
    const approved = window.confirm(
      `Install PyMRST Studio ${candidate.version}? Save work and stop active simulations first.`,
    );
    if (!approved) return;
    setBusy("install"); setError(""); setNotice("");
    try {
      await invokeDesktop<void>("install_pending_update", { userConfirmed: true });
      setCandidate(null);
      setNotice("The verified update was installed. Restart PyMRST Studio to use it.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally { setBusy(""); }
  }

  return <section className="update-settings">
    <div className="settings-intro"><span className="eyebrow">Signed distribution channel</span><h2>Application updates</h2><p>Update metadata is fetched only from the release endpoint pinned into this build. Tauri verifies the downloaded artifact against the embedded Ed25519 public key before installation; checks never install silently.</p></div>
    <div className="update-card"><div className={`qualification-seal ${status?.enabled ? "ready" : "degraded"}`}><span /><div><strong>{desktop ? status?.enabled ? "Verified channel ready" : "Channel not configured" : "Desktop only"}</strong><small>{status ? `Version ${status.current_version} · ${status.trust_model}` : desktop ? "Reading native release policy…" : "Use the installed PyMRST Studio application to manage updates"}</small></div></div><button className="button primary" disabled={!status?.enabled || Boolean(busy)} onClick={check}>{busy === "check" ? "Checking…" : "Check for updates"}</button></div>
    {candidate && <article className="update-candidate"><div><span className="eyebrow">Verified candidate</span><h3>PyMRST Studio {candidate.version}</h3><small>Current {candidate.current_version}{candidate.publication_date ? ` · Published ${new Date(candidate.publication_date).toLocaleString()}` : ""}</small></div><p>{candidate.notes || "No release notes were supplied."}</p><button className="button primary" disabled={Boolean(busy)} onClick={install}>{busy === "install" ? "Downloading and verifying…" : "Review and install"}</button></article>}
    {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
  </section>;
}
